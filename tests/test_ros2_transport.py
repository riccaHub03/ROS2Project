"""Real DDS/actions/physics smoke tests; fixture paths, no AI or physical grasp claim.

Run inside the Humble container. The fake planner supplies small certified joint
paths; the production ROS server and actual PyBullet servo execute them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

try:
    import rclpy
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


class FixtureMotion:
    def __init__(self, scene, report, **kwargs):
        from pipeline.ros2.contracts import DT
        from pipeline.sim.robot_scene import HOME
        self.scene = scene
        self.phases = []
        start = np.asarray(HOME, dtype=float)
        opened = .08
        for i, label in enumerate(["Approach", "Descend", "Close", "Lift", "Transfer", "Place", "Open", "Retreat", "Return"]):
            event = "attach" if i == 2 else "release" if i == 6 else None
            target = start.copy()
            if not event:
                target[0] = .015 if i % 2 == 0 else -.015
            count = 73
            alpha = np.linspace(0, 1, count)
            blend = 10*alpha**3 - 15*alpha**4 + 6*alpha**5
            samples = start + blend[:, None]*(target-start)
            width = .04 if event == "attach" else .08 if event == "release" else opened
            self.phases.append(dict(label=label, event=event, samples=samples, duration=(count-1)*DT,
                                    start_width=opened, end_width=width))
            opened, start = width, target
        out = Path(os.environ["PIPELINE_OUTPUT_DIR"])
        np.savez_compressed(out / "graspgen_sim_joint_path.npz", **{f"phase_{i}": ph["samples"] for i, ph in enumerate(self.phases)})
        self.index, self.elapsed, self.finished = 0, 0., False

    def tick(self, dt):
        ph = self.phases[self.index]
        self.elapsed += dt
        t = min(self.elapsed/ph["duration"], 1.)
        f = t*(len(ph["samples"])-1)
        left = min(int(f), len(ph["samples"])-2)
        self.scene.command_joints(ph["samples"][left]*(1-(f-left))+ph["samples"][left+1]*(f-left))
        self.scene.gripper_width = ph["start_width"]*(1-t)+ph["end_width"]*t
        if t >= 1:
            self.index += 1
            self.elapsed = 0.
            self.finished = self.index == len(self.phases)

    def report(self):
        return dict(status="INCOMPLETE", tool_center_inside_destination_xy=False,
                    tool_settled_on_destination=False, physical_grasp_validated=False)

    def close(self):
        pass


@unittest.skipUnless(HAS_ROS, "Requires the ROS 2 Humble runtime; run scripts/run_ros2_system.sh self-test")
class ROSTransport(unittest.TestCase):
    def setUp(self):
        import copy
        import trimesh
        from rclpy.executors import MultiThreadedExecutor
        from pipeline.common import paths
        from pipeline.common.report import StageReport
        from pipeline.config_loader import load_environment_config
        from pipeline.ros2.engine import SimulationEngine
        from pipeline.ros2.simulation import SimulationServer
        from pipeline.ros2.coordinator import Coordinator

        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name).resolve()
        self.old_output = os.environ.get(paths.OUTPUT_ENV)
        os.environ[paths.OUTPUT_ENV] = str(self.out)
        mesh = self.out / "fixture.stl"
        trimesh.creation.box([.12, .02, .006]).export(mesh)
        config = load_environment_config()
        config["surgical_instrument"]["stl_path"] = str(mesh)
        task = dict(action="pick_and_place", target_instrument="Scalpel", target_keyword="scalpel",
                    source_tray="tray_2", destination_tray="tray_1")
        StageReport("plan", data=task).save(self.out / "plan.json")
        spec = dict(out_dir=str(self.out), namespace="/test_" + self.out.name.replace("-", "_"),
                    settings={}, gui=False, seed=42)
        rclpy.init(args=[])
        self.server = SimulationServer(spec)
        self.server.engine = SimulationEngine(self.out, config, task, motion_factory=FixtureMotion)
        self.coordinator = Coordinator(spec)
        self.executor = MultiThreadedExecutor(num_threads=6)
        self.executor.add_node(self.server)
        self.executor.add_node(self.coordinator)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.loop_thread = threading.Thread(target=self.server.loop, daemon=True)
        self.loop_thread.start()
        self.server.publish_camera()
        deadline = time.monotonic() + 10
        while len(self.coordinator.images) != 3 or self.coordinator.feedback_count < 2:
            if time.monotonic() > deadline:
                self.fail("ROS RGB-D/state discovery timeout")
            time.sleep(.02)
        self.plan = self.server.engine.prepare({})

    def tearDown(self):
        self.server.closing.set()
        self.loop_thread.join(timeout=3)
        self.server.engine.close()
        self.executor.shutdown(timeout_sec=3)
        self.spin_thread.join(timeout=3)
        self.coordinator.destroy_node()
        self.server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.old_output is None:
            os.environ.pop("PIPELINE_OUTPUT_DIR", None)
        else:
            os.environ["PIPELINE_OUTPUT_DIR"] = self.old_output
        self.tmp.cleanup()

    def test_dds_rgbd_seven_arm_and_two_gripper_actions_with_measured_feedback(self):
        from pipeline.ros2.contracts import ingest_rgbd, load_phase_samples
        capture = json.loads((self.out / "sim_capture.json").read_text())
        images = self.coordinator.images
        ingest_rgbd(images["color"], images["depth"], images["info"], capture, self.out)
        samples = load_phase_samples(self.out, self.plan)
        self.coordinator.recording = True
        for phase, q in zip(self.plan["phases"], samples):
            self.coordinator.send_phase(phase, q)
        result = self.coordinator.service("finish")
        self.assertEqual(result["arm_actions_succeeded"], 7)
        self.assertEqual(result["gripper_actions_succeeded"], 2)
        self.assertGreater(len(self.coordinator.joint_log), 20)
        self.assertGreater(self.coordinator.action_feedback, 9)
        # Executing actions alone must never fabricate a successful placement.
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["tool_settled_on_destination"])

    def test_changed_trajectory_and_out_of_order_gripper_are_rejected(self):
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from pipeline.ros2.contracts import JOINT_NAMES
        from trajectory_msgs.msg import JointTrajectoryPoint
        arm = self.coordinator.arm
        self.assertTrue(arm.wait_for_server(timeout_sec=5))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.header.frame_id = self.server.frame("lbr_link_0")
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint(); point.positions = [0.]*7
        goal.trajectory.points = [point]
        handle = self.coordinator.wait(arm.send_goal_async(goal), 5)
        self.assertFalse(handle.accepted)
        goal = GripperCommand.Goal(); goal.command.position = .04
        self.assertTrue(self.coordinator.gripper.wait_for_server(timeout_sec=5))
        handle = self.coordinator.wait(self.coordinator.gripper.send_goal_async(goal), 5)
        self.assertFalse(handle.accepted)
        self.assertEqual(self.server.engine.completed, [])

    def test_stop_latches_fault_and_forbids_remaining_phases(self):
        self.coordinator.service("stop")
        self.assertIsNotNone(self.server.engine.fault)
        with self.assertRaises(ValueError):
            self.server.engine.next_phase()
        with self.assertRaises(RuntimeError):
            self.coordinator.service("finish")

    def arm_goal(self):
        from control_msgs.action import FollowJointTrajectory
        from pipeline.ros2.contracts import JOINT_NAMES, duration_ns
        from trajectory_msgs.msg import JointTrajectoryPoint
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.header.frame_id = self.server.frame("lbr_link_0")
        goal.trajectory.joint_names = JOINT_NAMES
        for i, q in enumerate(self.server.engine.motion.phases[0]["samples"]):
            pt = JointTrajectoryPoint(); pt.positions = list(map(float, q))
            pt.time_from_start.sec, pt.time_from_start.nanosec = divmod(duration_ns(i), 10**9)
            goal.trajectory.points.append(pt)
        return goal

    def test_active_goal_can_be_cancelled_and_concurrent_goal_is_rejected(self):
        from action_msgs.msg import GoalStatus
        client = self.coordinator.arm
        self.assertTrue(client.wait_for_server(timeout_sec=5))
        feedback = threading.Event()
        handle = self.coordinator.wait(client.send_goal_async(self.arm_goal(),
                                      feedback_callback=lambda message: feedback.set()), 5)
        self.assertTrue(handle.accepted)
        self.assertTrue(feedback.wait(timeout=2))
        duplicate = self.coordinator.wait(client.send_goal_async(self.arm_goal()), 5)
        self.assertFalse(duplicate.accepted)
        cancelled = self.coordinator.wait(handle.cancel_goal_async(), 5)
        self.assertTrue(cancelled.goals_canceling)
        response = self.coordinator.wait(handle.get_result_async(), 5)
        self.assertEqual(response.status, GoalStatus.STATUS_CANCELED)
        self.assertIsNotNone(self.server.engine.fault)

    def test_stale_measured_feedback_cancels_execution(self):
        phase = self.server.engine.motion.phases[0]
        start, end = phase["samples"][0], phase["samples"][-1]
        phase["samples"] = np.linspace(start, end, 481)
        phase["duration"] = 2.
        self.server.publish_state = lambda: None
        with self.assertRaisesRegex(RuntimeError, "feedback stopped advancing"):
            self.coordinator.send_phase({**self.plan["phases"][0], "duration_s": 2.}, phase["samples"])
        self.coordinator.stop()
        self.assertIsNotNone(self.server.engine.fault)


if __name__ == "__main__":
    unittest.main()
