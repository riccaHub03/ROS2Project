"""One ROS task: instruction -> DDS RGB-D -> existing AI worker -> ROS actions."""
from __future__ import annotations

import json
import multiprocessing
import queue
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from pipeline.common.report import StageReport, write_json
from pipeline.ros2.contracts import JOINT_NAMES, duration_ns, ingest_rgbd, load_phase_samples
from pipeline.ros2.simulation import pose_message, snapshot_qos


class Coordinator(Node):
    def __init__(self, spec, process=None):
        super().__init__("surgical_coordinator", namespace=spec["namespace"])
        self.spec, self.out, self.process = spec, Path(spec["out_dir"]), process
        self.instruction = None
        self.images = {}
        self.feedback_count = 0
        self.feedback_stamp = -1
        self.feedback_at = 0.
        self.joint_log = []
        self.recording = False
        self.action_feedback = 0
        self.action_results = []
        self._active_handle = None
        self.status_pub = self.create_publisher(String, "status", snapshot_qos())
        self.task_pub = self.create_publisher(String, "task", snapshot_qos())
        self.result_pub = self.create_publisher(String, "result", snapshot_qos())
        self.grasp_pub = self.create_publisher(PoseStamped, "grasp_pose", snapshot_qos())
        self.candidates_pub = self.create_publisher(PoseArray, "grasp_candidates", snapshot_qos())
        self.create_subscription(String, "instruction", self.on_instruction, snapshot_qos())
        self.create_subscription(JointState, "joint_states", self.on_joints, 10)
        for name, msg in (("color", Image), ("depth", Image), ("info", CameraInfo)):
            self.create_subscription(msg, "camera/" + name,
                                     lambda value, key=name: self.images.__setitem__(key, value), snapshot_qos())
        self.service_clients = {name: self.create_client(Trigger, name) for name in ("initialize", "plan", "finish", "stop")}
        self.arm = ActionClient(self, FollowJointTrajectory, "arm/follow_joint_trajectory")
        self.gripper = ActionClient(self, GripperCommand, "gripper/gripper_cmd")

    def on_instruction(self, msg):
        if self.instruction is None and msg.data.strip():
            self.instruction = msg.data.strip()

    def on_joints(self, msg):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if (list(msg.name) != JOINT_NAMES or len(msg.position) != 7
                or not np.isfinite(msg.position).all() or stamp <= self.feedback_stamp):
            return
        self.feedback_stamp, self.feedback_at = stamp, time.monotonic()
        self.feedback_count += 1
        if self.recording:
            self.joint_log.append(dict(stamp_ns=stamp, position=list(msg.position), velocity=list(msg.velocity)))

    def status(self, stage):
        self.current_stage = stage
        self.get_logger().info(stage)
        self.status_pub.publish(String(data=json.dumps(dict(run_id=self.out.name, stage=stage))))

    def healthy(self, motion=False):
        if not rclpy.ok():
            raise KeyboardInterrupt
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"ROS simulation process exited ({self.process.returncode}); inspect ros2_simulation.log")
        if motion and time.monotonic() - self.feedback_at > .75:
            raise RuntimeError("Measured simulation feedback stopped advancing (>0.75 s)")

    def wait(self, future, timeout, *, motion=False):
        deadline = time.monotonic() + timeout
        while not future.done():
            self.healthy(motion)
            if time.monotonic() > deadline:
                raise TimeoutError(f"ROS operation exceeded {timeout:.1f} s")
            time.sleep(.01)
        self.healthy(motion)
        return future.result()

    def service(self, name, timeout=210):
        client = self.service_clients[name]
        deadline = time.monotonic() + 20
        while not client.wait_for_service(timeout_sec=.2):
            self.healthy()
            if time.monotonic() > deadline:
                raise TimeoutError(f"ROS service {name} unavailable")
        result = self.wait(client.call_async(Trigger.Request()), timeout)
        if not result.success:
            raise RuntimeError(f"ROS {name}: {result.message}")
        return json.loads(result.message)

    def perceive(self):
        from pipeline.sim.graphical import _compute
        context = multiprocessing.get_context("spawn")
        events = context.Queue()
        settings = self.spec["settings"]
        worker = context.Process(target=_compute, args=(settings["vision"], events, settings["environment"],
                                dict(backend="graspgen", graspgen_config=settings["graspgen_config"]),
                                False, settings))
        worker.start()
        deadline, dead_since = time.monotonic() + 600, None
        try:
            while time.monotonic() < deadline:
                self.healthy()
                try:
                    kind, data = events.get(timeout=.1)
                except queue.Empty:
                    if worker.exitcode is not None:
                        dead_since = dead_since or time.monotonic()
                        if time.monotonic() - dead_since > 2:
                            raise RuntimeError(f"Perception worker exited without a result ({worker.exitcode})")
                    continue
                if kind == "status":
                    self.status(data)
                elif kind == "error":
                    raise RuntimeError(data)
                elif kind == "result":
                    return data
            raise TimeoutError("Perception worker exceeded 600 s")
        finally:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
            events.close()

    def send_phase(self, phase, samples):
        self.status(phase["label"])
        deadline = time.monotonic() + 2
        while time.monotonic() - self.feedback_at > .75:
            self.healthy()
            if time.monotonic() > deadline:
                raise RuntimeError("Fresh measured joint feedback required before every phase")
            time.sleep(.01)
        if phase["kind"] == "arm":
            client, goal = self.arm, FollowJointTrajectory.Goal()
            goal.trajectory.header.frame_id = self.spec["namespace"].strip("/") + "/" + self.spec.get("base_frame_id", "lbr_link_0")
            goal.trajectory.joint_names = JOINT_NAMES
            for i, q in enumerate(samples):
                point = JointTrajectoryPoint()
                point.positions = list(map(float, q))
                ns = duration_ns(i)
                point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 10**9)
                goal.trajectory.points.append(point)
        else:
            client, goal = self.gripper, GripperCommand.Goal()
            goal.command.position, goal.command.max_effort = phase["end_width"], 0.
        if not client.wait_for_server(timeout_sec=10):
            raise RuntimeError("ROS action server unavailable")
        before = self.action_feedback
        handle = self.wait(client.send_goal_async(goal, feedback_callback=self.on_action_feedback), 10, motion=True)
        if not handle.accepted:
            raise RuntimeError(f"Validated phase rejected by ROS server: {phase['label']}")
        self._active_handle = handle
        try:
            response = self.wait(handle.get_result_async(), max(30., phase["duration_s"] * 3 + 10), motion=True)
            result = response.result
            ok = response.status == GoalStatus.STATUS_SUCCEEDED
            ok = ok and (result.error_code == 0 if phase["kind"] == "arm" else result.reached_goal)
            if not ok:
                raise RuntimeError(f"ROS action failed in {phase['label']}: {result}")
            if self.action_feedback == before:
                raise RuntimeError("Action completed without measured execution feedback")
            self.action_results.append(dict(index=phase["index"], kind=phase["kind"], label=phase["label"],
                                            status="SUCCEEDED", feedback_messages=self.action_feedback-before))
        except BaseException:
            handle.cancel_goal_async()
            raise
        finally:
            self._active_handle = None

    def on_action_feedback(self, message):
        self.action_feedback += 1

    def stop(self):
        if self._active_handle:
            self._active_handle.cancel_goal_async()
        client = self.service_clients["stop"]
        if rclpy.ok() and client.service_is_ready():
            future = client.call_async(Trigger.Request())
            deadline = time.monotonic() + 3
            while not future.done() and rclpy.ok() and time.monotonic() < deadline:
                time.sleep(.02)

    def execute_task(self):
        from pipeline import plan
        settings = self.spec["settings"]
        self.status("Interpreting instruction")
        task = plan.run(self.instruction, offline=settings["planner"] == "deterministic",
                        tool_hint=settings["scene_tool"] if settings["planner"] == "deterministic" else None,
                        require_llm=settings["strict"], expected_instrument=settings["scene_tool"],
                        ollama_url=settings["ollama_url"], model=settings["llm_model"],
                        timeout_s=settings.get("llm_timeout_s", 120), config_path=settings["environment"])
        self.task_pub.publish(String(data=json.dumps(dict(run_id=self.out.name, **task.data))))
        self.status("Initializing ROS simulation and capturing RGB-D")
        self.service("initialize")
        deadline = time.monotonic() + 20
        while len(self.images) != 3:
            self.healthy()
            if time.monotonic() > deadline:
                raise TimeoutError("Aligned RGB-D and CameraInfo not received through ROS 2")
            time.sleep(.02)
        capture = json.loads((self.out / "sim_capture.json").read_text())
        camera_transport = ingest_rgbd(self.images["color"], self.images["depth"], self.images["info"], capture, self.out)
        self.perceive()
        self.status("Validating GraspGen candidates and planning the complete path")
        execution = self.service("plan")
        if (execution["robot_description"]["model_sha256"]
                != capture["data"]["robot_description"]["model_sha256"]):
            raise ValueError("Robot model changed between acquisition and motion planning")
        samples = load_phase_samples(self.out, execution)
        grasp = StageReport.load(self.out / "grasp_simulation_selection.json").data
        stamp = self.get_clock().now().to_msg()
        selected = np.asarray(grasp["simulation_selected"]["T_base_tcp"])
        from pipeline.frame_transformer import quaternion_from_matrix
        self.grasp_pub.publish(pose_message(self.spec["namespace"].strip("/") + "/" + self.spec.get("base_frame_id", "lbr_link_0"),
                                           selected[:3, 3], quaternion_from_matrix(selected[:3, :3]), stamp))
        poses = PoseArray()
        poses.header.frame_id = self.spec["namespace"].strip("/") + "/" + self.spec.get("base_frame_id", "lbr_link_0")
        poses.header.stamp = stamp
        for candidate in grasp["candidates"]:
            transform = np.asarray(candidate["T_base_tcp"])
            poses.poses.append(pose_message(poses.header.frame_id, transform[:3, 3],
                                            quaternion_from_matrix(transform[:3, :3]), stamp).pose)
        self.candidates_pub.publish(poses)
        self.recording = True
        try:
            for phase, q in zip(execution["phases"], samples):
                self.send_phase(phase, q)
            motion = self.service("finish", timeout=20)
        finally:
            self.recording = False
            write_json(self.out / "ros2_measured_joint_states.json", self.joint_log)
        if motion["status"] != "COMPLETED":
            raise RuntimeError("All ROS actions finished, but instrument placement was not verified")
        evidence = dict(rgbd_received_via_ros2=camera_transport["rgbd"] == "ROS2 sensor_msgs",
                        arm_actions_succeeded=sum(r["kind"] == "arm" for r in self.action_results),
                        gripper_actions_succeeded=sum(r["kind"] == "gripper" for r in self.action_results),
                        measured_joint_messages=len(self.joint_log), action_feedback_messages=self.action_feedback,
                        placement_verified=motion["tool_settled_on_destination"] and motion["tool_center_inside_destination_xy"])
        report = StageReport("ros2_run", data=dict(namespace=self.spec["namespace"], evidence=evidence,
                                                   actions=self.action_results, execution_scope="pybullet_only",
                                                   physical_grasp_validated=False))
        report.save(self.out / "ros2_run.json")
        self.result_pub.publish(String(data=json.dumps(dict(status="COMPLETED", **report.data))))
        self.status("ROS task completed; placement verified")
        return evidence
