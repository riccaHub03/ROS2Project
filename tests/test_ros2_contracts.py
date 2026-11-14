"""Reject stale/changed plans and mismatched RGB-D before ROS motion begins."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
from PIL import Image

from pipeline.ros2.contracts import (JOINT_NAMES, duration_ns, ingest_rgbd, load_phase_samples,
                                    phase_manifest, validate_gripper, validate_trajectory, verify_initial_joints)
from pipeline.system.orchestrator import finalize


class ROSContracts(unittest.TestCase):
    def setUp(self):
        self.phase = dict(event=None, samples=np.zeros((4, 7)), label="Approach", duration_s=3/240,
                          duration=3/240, start_width=.136, end_width=.136)

    def test_only_exact_approved_samples_and_timing_are_accepted(self):
        times = [duration_ns(i) for i in range(4)]
        validate_trajectory(JOINT_NAMES, np.zeros((4, 7)), times, self.phase)
        changed = np.zeros((4, 7)); changed[2, 3] = .01
        for names, points, stamp in ((JOINT_NAMES[::-1], np.zeros((4, 7)), times),
                                      (JOINT_NAMES, changed, times),
                                      (JOINT_NAMES, np.zeros((4, 7)), [0, 1, 2, 3]),
                                      (JOINT_NAMES, np.full((4, 7), np.nan), times)):
            with self.assertRaises(ValueError):
                validate_trajectory(names, points, stamp, self.phase)

    def test_phase_order_gripper_units_and_measured_start_are_enforced(self):
        with self.assertRaises(ValueError):
            validate_gripper(.04, 0., self.phase)
        closing = {**self.phase, "event": "attach", "end_width": .04}
        validate_gripper(.04, 0., closing)
        for width, effort in ((40., 0.), (.04, 20.), (float("nan"), 0.)):
            with self.assertRaises(ValueError):
                validate_gripper(width, effort, closing)
        verify_initial_joints(np.zeros(7), np.zeros(7))
        with self.assertRaises(ValueError):
            verify_initial_joints(np.ones(7), np.zeros(7))

    def test_npz_provenance_changes_and_cross_run_replays_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            file = out / "graspgen_sim_joint_path.npz"
            np.savez(file, phase_0=self.phase["samples"])
            manifest = phase_manifest([self.phase], out, out.name)
            self.assertEqual(load_phase_samples(out, manifest)[0].shape, (4, 7))
            with self.assertRaises(ValueError):
                load_phase_samples(out, {**manifest, "run_id": "another_run"})
            np.savez(file, phase_0=np.ones((4, 7)))
            with self.assertRaises(ValueError):
                load_phase_samples(out, manifest)

    def test_rgbd_requires_same_stamp_intrinsics_and_scene_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
            depth = np.ones((2, 3), dtype="<f4")
            Image.fromarray(pixels).save(out / "sim_capture_rgb.png")
            np.save(out / "sim_capture_depth.npy", depth)
            header = NS(stamp=NS(sec=1, nanosec=2), frame_id="run/camera")
            rgb = NS(header=header, width=3, height=2, encoding="rgb8", step=9, data=pixels.tobytes())
            dep = NS(header=header, width=3, height=2, encoding="32FC1", step=12,
                     data=depth.tobytes(), is_bigendian=0)
            info = NS(header=header, width=3, height=2, k=[2, 0, 1, 0, 2, 1, 0, 0, 1])
            capture = dict(data=dict(camera=dict(resolution_wh=[3, 2], fx_fy=[2, 2], cx_cy=[1, 1])))
            self.assertEqual(ingest_rgbd(rgb, dep, info, capture, out)["rgbd"], "ROS2 sensor_msgs")
            bad = copy.deepcopy(dep); bad.header.stamp.nanosec = 3
            with self.assertRaises(ValueError):
                ingest_rgbd(rgb, bad, info, capture, out)
            bad = copy.deepcopy(rgb); bad.data = bytes(18)
            with self.assertRaises(ValueError):
                ingest_rgbd(bad, dep, info, capture, out)

    def test_ros_manifest_requires_dds_actions_feedback_and_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            reports = {"plan.json": dict(planner_backend="ollama"),
                       "sim_segmentation.json": dict(segmentation_backend="sam3"),
                       "grasp_graspgen.json": dict(grasp_backend="graspgen", candidates=[{}]),
                       "sim_robot_motion.json": dict(execution_checks_passed=True, tool_settled_on_destination=True,
                                                       tool_center_inside_destination_xy=True),
                       "ros2_run.json": dict(evidence={})}
            def save():
                for name, data in reports.items():
                    (out/name).write_text(json.dumps(dict(status="COMPLETED", data=data)))
            manifest = dict(profile="full_system", transport="ros2")
            save()
            self.assertEqual(finalize(out, manifest, 0), 1)
            reports["ros2_run.json"]["evidence"] = dict(rgbd_received_via_ros2=True, arm_actions_succeeded=7,
                gripper_actions_succeeded=2, measured_joint_messages=20, action_feedback_messages=20)
            save()
            self.assertEqual(finalize(out, manifest, 0), 0)
            reports["sim_robot_motion.json"]["tool_settled_on_destination"] = False
            save()
            self.assertEqual(finalize(out, manifest, 0), 1)


if __name__ == "__main__":
    unittest.main()
