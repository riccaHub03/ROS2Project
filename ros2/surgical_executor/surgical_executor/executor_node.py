"""Nodo ``surgical_pick_and_place``: invia a ROS 2 la sequenza pianificata da ``pipeline.motion``.

ATTENZIONE: i topic di default sono quelli della piattaforma di simulazione ``lbr_pybullet_tutorial``
(percorso cartesiano + stato + comando pinza). Per muovere il KUKA reale serve un altro *backend*
(``lbr_fri_ros2_stack``: ros2_control / MoveIt e driver della pinza): vedi docs/ros2.md.
Il nodo rifiuta qualunque movimento se la presa non e' ``VALIDATED_...`` nel report della pipeline.

Richiede ``pipeline`` nel PYTHONPATH (nel container: ``/workspace``, impostato dall'entrypoint).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from pipeline.common import paths
from pipeline.common.report import StageError, StageReport
from pipeline.config_loader import load_environment_config
from pipeline.motion.pick_and_place import (MotionRefused, PickAndPlacePlan, Waypoint, grasp_pose_from_report,
                                            plan_pick_and_place)


def make_pose(frame_id: str, xyz, quat) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in xyz)
    (msg.pose.orientation.x, msg.pose.orientation.y,
     msg.pose.orientation.z, msg.pose.orientation.w) = (float(v) for v in quat)
    return msg


class PickAndPlaceExecutor(Node):
    def __init__(self):
        super().__init__("surgical_pick_and_place")
        for name, default in (
            ("grasp_report", ""), ("config_path", ""), ("destination_tray", "tray_1"), ("frame_id", "lbr_link_0"),
            ("path_topic", "/lbr_tutorial/cartesian_path"), ("status_topic", "/lbr_tutorial/status"),
            ("gripper_topic", "/lbr/gripper/command"), ("ee_pose_topic", "/lbr/ee_pose"),
            ("joint_states_topic", "/lbr/joint_states"), ("render_video", True),
        ):
            self.declare_parameter(name, default)
        get = lambda n: self.get_parameter(n).value

        self.frame_id = get("frame_id")
        self.config = load_environment_config(get("config_path") or None)
        timeouts = self.config["motion_control"]["timeouts"]
        self.step_timeout = float(timeouts["step_timeout_sec"])
        self.bringup_timeout = float(timeouts["bringup_wait_sec"])
        self.accept_timeout = float(timeouts["accept_timeout_sec"])
        self.gripper_settle = float(timeouts["gripper_settle_sec"])

        self.path_pub = self.create_publisher(PathMsg, get("path_topic"), 10)
        self.gripper_pub = self.create_publisher(String, get("gripper_topic"), 10)
        self.create_subscription(String, get("status_topic"), self._on_status,
                                 QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(JointState, get("joint_states_topic"), self._on_joint_states, 10)

        self.events: List[dict] = []
        self.ready = False
        self.recording = False
        self.tool_grasped = False
        self.trajectory: List[dict] = []
        self._t0: Optional[float] = None

    # -- callback ------------------------------------------------------------------------------------------
    def _on_status(self, msg: String):
        try:
            event = json.loads(msg.data)
        except ValueError:
            self.get_logger().warning(f"messaggio di stato non JSON ignorato: {msg.data[:80]}")
            return
        if event.get("event") == "ready":
            self.ready = True
        self.events.append(event)

    def _on_joint_states(self, msg: JointState):
        if not self.recording or len(msg.position) < 7:
            return
        if "lbr_A1" in msg.name:
            q = [float(msg.position[msg.name.index(f"lbr_A{i}")]) for i in range(1, 8)]
        else:
            q = [float(v) for v in msg.position[:7]]
        self.trajectory.append({"time": time.monotonic() - (self._t0 or time.monotonic()),
                                "q": q, "grasped": self.tool_grasped})

    # -- primitive ------------------------------------------------------------------------------------------
    def _spin_for(self, seconds: float):
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_platform(self) -> bool:
        self.get_logger().info(f"Attesa della piattaforma ({self.get_parameter('status_topic').value})...")
        end = time.monotonic() + self.bringup_timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.ready and self.path_pub.get_subscription_count() > 0:
                return True
        return False

    def send_path(self, waypoints: List[Waypoint], label: str) -> bool:
        msg = PathMsg()
        msg.header.frame_id = self.frame_id
        msg.poses = [make_pose(self.frame_id, xyz, quat) for xyz, quat in waypoints]
        self.events.clear()
        self.path_pub.publish(msg)

        request_id, deadline = None, time.monotonic() + self.accept_timeout
        while rclpy.ok() and request_id is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            for event in self.events:
                if event.get("event") == "accepted":
                    request_id = event.get("request_id")
                    break
                if event.get("event") == "rejected":
                    self.get_logger().error(f"{label}: rifiutato dal controllore: {event.get('reason')}")
                    return False
        if request_id is None:
            self.get_logger().error(f"{label}: timeout in attesa di accettazione della traiettoria")
            return False

        deadline = time.monotonic() + self.step_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            for event in self.events:
                if event.get("request_id") != request_id:
                    continue
                if event.get("event") == "succeeded":
                    x, y, z = waypoints[-1][0]
                    self.get_logger().info(f"{label}: completato, arrivo ({x:.2f}, {y:.2f}, {z:.2f}) m")
                    return True
                if event.get("event") in ("failed", "rejected"):
                    self.get_logger().error(f"{label}: errore di esecuzione: {event.get('reason')}")
                    return False
        self.get_logger().error(f"{label}: timeout di esecuzione ({self.step_timeout:.0f} s)")
        return False

    # -- missione -------------------------------------------------------------------------------------------
    def load_plan(self) -> PickAndPlacePlan:
        report_file = self.get_parameter("grasp_report").value or str(paths.report_path(paths.GRASP))
        report = StageReport.load(report_file, expect_stage="grasp")
        xyz, quat = grasp_pose_from_report(report.data)
        return plan_pick_and_place(xyz, quat, self.get_parameter("destination_tray").value, self.config)

    def run(self) -> bool:
        try:
            plan = self.load_plan()
        except (MotionRefused, StageError, KeyError) as exc:
            self.get_logger().error(f"Movimento RIFIUTATO: {exc}")
            return False
        if not self.wait_for_platform():
            self.get_logger().error("Piattaforma robotica non pronta: nessun movimento inviato")
            return False

        self.get_logger().info(f"Presa {plan.grasp_xyz} -> {plan.destination_tray} {plan.deposit_xyz}")
        self.trajectory, self._t0, self.recording, self.tool_grasped = [], time.monotonic(), True, False
        try:
            for sequence in plan.sequences:
                if not self.send_path(sequence.waypoints, sequence.name):
                    return False
                if sequence.gripper_after:
                    self.gripper_pub.publish(String(data=sequence.gripper_after))
                    self.tool_grasped = sequence.gripper_after == "grasp"
                    self._spin_for(self.gripper_settle)
        finally:
            self.recording = False
        self.get_logger().info("Missione completata")
        self._save(plan)
        return True

    def _save(self, plan: PickAndPlacePlan):
        report = StageReport(stage="robot_trajectory", data={
            "trajectory": self.trajectory, "tool_initial_xyz": plan.grasp_xyz,
            "tray_final_xyz": plan.deposit_xyz, "total_points": len(self.trajectory)})
        saved = report.save(paths.report_path(paths.TRAJECTORY))
        self.get_logger().info(f"Traiettoria registrata: {saved} ({len(self.trajectory)} punti)")
        if self.get_parameter("render_video").value:
            from pipeline.video_recorder import render_offline_video
            render_offline_video(trajectory_log=self.trajectory, p_tool_initial=plan.grasp_xyz,
                                 p_tray_final=plan.deposit_xyz, config=self.config, fps=30)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = PickAndPlaceExecutor()
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
