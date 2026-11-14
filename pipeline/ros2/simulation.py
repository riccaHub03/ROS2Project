"""Standard ROS actions driving the existing scene, on PyBullet's main thread.

This is a bounded simulation action server, not ros2_control or a robot driver.
It accepts only the next exact phase authorized by the in-memory planner.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import pybullet as p
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from pipeline.common.report import StageReport
from pipeline.common.reconstruction import CameraModel
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import quaternion_from_matrix
from pipeline.ros2.contracts import DT, JOINT_NAMES, validate_gripper, validate_trajectory
from pipeline.ros2.engine import SimulationEngine


def snapshot_qos():
    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      reliability=ReliabilityPolicy.RELIABLE)


def pose_message(frame, position, quaternion, stamp):
    msg = PoseStamped()
    msg.header.frame_id, msg.header.stamp = frame, stamp
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
    (msg.pose.orientation.x, msg.pose.orientation.y,
     msg.pose.orientation.z, msg.pose.orientation.w) = map(float, quaternion)
    return msg


class SimulationServer(Node):
    def __init__(self, spec):
        super().__init__("surgical_simulation", namespace=spec["namespace"])
        self.spec, self.out = spec, Path(spec["out_dir"])
        self.engine = None
        self.commands = queue.Queue()
        self.stop_event = threading.Event()
        self.closing = threading.Event()
        self.guard = threading.Lock()
        self.reserved = False
        self.state = None
        self.pending = None
        self.group = ReentrantCallbackGroup()
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "tcp_pose", 10)
        self.rgb_pub = self.create_publisher(Image, "camera/color", snapshot_qos())
        self.depth_pub = self.create_publisher(Image, "camera/depth", snapshot_qos())
        self.info_pub = self.create_publisher(CameraInfo, "camera/info", snapshot_qos())
        self.tf_static = StaticTransformBroadcaster(self)
        self.tf_dynamic = TransformBroadcaster(self)
        for name in ("initialize", "plan", "finish", "stop"):
            self.create_service(Trigger, name, self.service(name), callback_group=self.group)
        self.arm = ActionServer(self, FollowJointTrajectory, "arm/follow_joint_trajectory",
                                execute_callback=lambda h: self.execute(h, "arm"),
                                goal_callback=lambda g: self.accept(g, "arm"),
                                cancel_callback=self.cancel, callback_group=self.group)
        self.gripper = ActionServer(self, GripperCommand, "gripper/gripper_cmd",
                                    execute_callback=lambda h: self.execute(h, "gripper"),
                                    goal_callback=lambda g: self.accept(g, "gripper"),
                                    cancel_callback=self.cancel, callback_group=self.group)

    def frame(self, name):
        if name == "lbr_link_0":
            name = self.spec.get("base_frame_id", name)
        return self.spec["namespace"].strip("/") + "/" + name

    def request(self, kind):
        job = dict(kind=kind, done=threading.Event(), error=None, result=None)
        self.commands.put(job)
        return job

    def service(self, name):
        def callback(request, response):
            if name == "stop":
                self.stop_event.set()  # also observed inside the planner heartbeat
            job = self.request(name)
            deadline = time.monotonic() + 210
            while rclpy.ok() and not job["done"].wait(.05):
                if self.closing.is_set() or time.monotonic() > deadline:
                    self.stop_event.set()
                    response.success, response.message = False, "Simulation service stopped/timed out"
                    return response
            response.success = job["error"] is None and job["done"].is_set()
            response.message = job["error"] or json.dumps(job["result"])
            return response
        return callback

    def accept(self, goal, kind):
        with self.guard:
            try:
                if self.reserved or self.stop_event.is_set() or self.engine is None:
                    raise ValueError("Simulation busy, stopped or not initialized")
                phase = self.engine.next_phase()
                if kind == "arm":
                    trajectory = goal.trajectory
                    if (trajectory.header.stamp.sec or trajectory.header.stamp.nanosec
                            or trajectory.header.frame_id != self.frame("lbr_link_0")
                            or goal.multi_dof_trajectory.points or goal.multi_dof_trajectory.joint_names
                            or goal.path_tolerance or goal.goal_tolerance
                            or getattr(goal, "component_path_tolerance", [])
                            or getattr(goal, "component_goal_tolerance", [])
                            or goal.goal_time_tolerance.sec or goal.goal_time_tolerance.nanosec
                            or any(pt.velocities or pt.accelerations or pt.effort for pt in trajectory.points)):
                        raise ValueError("Only unscheduled position samples with server-fixed tolerances are supported")
                    validate_trajectory(trajectory.joint_names, [pt.positions for pt in trajectory.points],
                                        [pt.time_from_start.sec * 10**9 + pt.time_from_start.nanosec
                                         for pt in trajectory.points], phase)
                else:
                    validate_gripper(goal.command.position, goal.command.max_effort, phase)
                self.reserved = True
                return GoalResponse.ACCEPT
            except (ValueError, KeyError, TypeError) as exc:
                self.get_logger().error(f"Rejected {kind}: {exc}")
                return GoalResponse.REJECT

    def cancel(self, handle):
        self.stop_event.set()
        return CancelResponse.ACCEPT

    def execute(self, handle, kind):
        result = FollowJointTrajectory.Result() if kind == "arm" else GripperCommand.Result()
        job = self.request("execute")
        deadline = time.monotonic() + 200
        try:
            while rclpy.ok() and not job["done"].wait(.05):
                if handle.is_cancel_requested or time.monotonic() > deadline or self.closing.is_set():
                    self.stop_event.set()
                state = self.state
                if state:
                    if kind == "arm":
                        feedback = FollowJointTrajectory.Feedback()
                        feedback.header.stamp = state["stamp"]
                        feedback.joint_names = JOINT_NAMES
                        feedback.actual.positions = list(map(float, state["q"]))
                        feedback.actual.velocities = list(map(float, state["velocity"]))
                        feedback.desired.positions = list(map(float, state.get("desired", state["q"])))
                        feedback.error.positions = [a-b for a, b in zip(feedback.desired.positions,
                                                                 feedback.actual.positions)]
                    else:
                        feedback = GripperCommand.Feedback()
                        feedback.position, feedback.effort = state["width"], 0.
                        feedback.stalled, feedback.reached_goal = False, False
                    handle.publish_feedback(feedback)
                if self.closing.is_set() or time.monotonic() > deadline + 5:
                    break
            error = job["error"] or (None if job["done"].is_set() else "Action server stopped")
            if kind == "arm":
                result.error_code = (FollowJointTrajectory.Result.SUCCESSFUL if not error else
                                     FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED)
                result.error_string = error or "Validated phase executed with measured joint feedback"
            else:
                result.position = (self.state or {}).get("width", 0.)
                result.effort, result.stalled, result.reached_goal = 0., False, not error
            if handle.is_cancel_requested:
                handle.canceled()
            elif error:
                handle.abort()
            else:
                handle.succeed()
            return result
        finally:
            with self.guard:
                self.reserved = False

    def publish_camera(self):
        report = self.engine.scene.capture_report
        camera = report.data["camera"]
        rgb = np.asarray(PILImage.open(report.artifacts["rgb_png"]).convert("RGB"))
        depth = np.load(report.artifacts["depth_npy"]).astype("<f4")
        h, w = depth.shape
        stamp = self.get_clock().now().to_msg()
        for pub, encoding, step, data in ((self.rgb_pub, "rgb8", w*3, rgb.tobytes()),
                                           (self.depth_pub, "32FC1", w*4, depth.tobytes())):
            msg = Image()
            msg.header.frame_id, msg.header.stamp = self.frame("camera_optical"), stamp
            msg.height, msg.width, msg.encoding, msg.step, msg.data = h, w, encoding, step, data
            pub.publish(msg)
        fx, fy = camera["fx_fy"]
        cx, cy = camera["cx_cy"]
        info = CameraInfo()
        info.header.frame_id, info.header.stamp = self.frame("camera_optical"), stamp
        info.width, info.height, info.distortion_model = w, h, "plumb_bob"
        info.d = [0.] * 5
        info.k = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
        info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        info.p = [fx, 0., cx, 0., 0., fy, cy, 0., 0., 0., 1., 0.]
        self.info_pub.publish(info)
        optical = CameraModel.from_dict(camera).transformer()
        base = self.engine.scene.base_frame
        transforms = []
        for child, position, rotation in (("lbr_link_0", base.position, base.rotation),
                                           ("camera_optical", optical.cam_pos, optical.R_world_opt)):
            msg = TransformStamped()
            msg.header.frame_id, msg.header.stamp = self.frame("world"), stamp
            msg.child_frame_id = self.frame(child)
            msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, position)
            (msg.transform.rotation.x, msg.transform.rotation.y,
             msg.transform.rotation.z, msg.transform.rotation.w) = quaternion_from_matrix(rotation)
            transforms.append(msg)
        self.tf_static.sendTransform(transforms)

    def publish_state(self):
        state = dict(self.engine.last_state, stamp=self.get_clock().now().to_msg())
        self.state = state
        msg = JointState()
        msg.header.frame_id, msg.header.stamp = self.frame("lbr_link_0"), state["stamp"]
        msg.name, msg.position, msg.velocity = JOINT_NAMES, list(map(float, state["q"])), list(map(float, state["velocity"]))
        self.joint_pub.publish(msg)
        pose = pose_message(self.frame("world"), state["tcp"], state["quaternion"], state["stamp"])
        self.pose_pub.publish(pose)
        tf = TransformStamped()
        tf.header, tf.child_frame_id = pose.header, self.frame("tcp")
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = state["tcp"]
        tf.transform.rotation = pose.pose.orientation
        self.tf_dynamic.sendTransform(tf)

    def command(self, job):
        try:
            kind = job["kind"]
            if kind == "initialize":
                if self.engine or self.stop_event.is_set():
                    raise ValueError("Run already initialized or stopped")
                task = StageReport.load(self.out / "plan.json", expect_stage="plan").data
                self.engine = SimulationEngine(self.out, load_environment_config(self.spec["settings"]["environment"]),
                                               task, gui=self.spec["gui"], seed=self.spec["seed"],
                                               cancel_requested=self.stop_event.is_set)
                self.publish_camera()
                job["result"] = {"run_id": self.out.name, "camera": "ROS2 RGB-D snapshot"}
            elif kind == "plan":
                report = StageReport.load(self.out / "grasp_graspgen.json", expect_stage="grasp")
                job["result"] = self.engine.prepare(report.data)
                self.engine.scene.show_grasp(report.data)
                self.engine.scene.save_overview(self.out / "sim_robot_grasp.png")
            elif kind == "execute":
                if self.stop_event.is_set():
                    raise ValueError("Run stopped")
                self.engine.start()
                self.pending = job
                return
            elif kind == "finish":
                for _ in range(120):
                    self.engine.step()
                self.publish_state()
                job["result"] = self.engine.finish()
            elif kind == "stop":
                if self.engine:
                    self.engine.stop("ROS stop/cancellation requested")
                job["result"] = {"stopped": True}
            else:
                raise ValueError(f"Unknown command {kind}")
        except Exception as exc:
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if self.pending is not job:
                job["done"].set()

    def loop(self):
        deadline = time.monotonic()
        while rclpy.ok() and not self.closing.is_set():
            try:
                self.command(self.commands.get_nowait())
            except queue.Empty:
                pass
            if self.engine:
                if self.engine.scene.gui:
                    if not p.isConnected(self.engine.scene.cid):
                        raise RuntimeError("Simulation window disconnected")
                    keys = p.getKeyboardEvents(physicsClientId=self.engine.scene.cid)
                    if any(keys.get(k, 0) & p.KEY_WAS_TRIGGERED for k in (ord("q"), 27)):
                        self.stop_event.set()
                if self.stop_event.is_set() and not self.engine.fault:
                    self.engine.stop("ROS stop/cancellation or window quit")
                self.engine.step()
                if self.engine.ticks % 4 == 0:
                    self.publish_state()  # timestamps advance only with actual physics steps
                if self.pending and not self.engine.active:
                    self.pending["error"] = self.engine.fault
                    self.pending["result"] = {"phase": self.engine.phase_index}
                    self.pending["done"].set()
                    self.pending = None
            deadline += DT
            time.sleep(max(0, deadline - time.monotonic()))
            if time.monotonic() - deadline > .1:
                deadline = time.monotonic()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-file", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.run_file).read_text())
    os.environ["PIPELINE_OUTPUT_DIR"] = spec["out_dir"]
    rclpy.init(args=[])
    node = SimulationServer(spec)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        node.loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.closing.set()
        node.stop_event.set()
        if node.engine:
            node.engine.close()
        if node.pending:
            node.pending["error"] = "Simulation shut down"
            node.pending["done"].set()
        executor.shutdown(timeout_sec=3)
        thread.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
