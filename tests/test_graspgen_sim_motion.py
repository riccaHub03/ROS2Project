"""Regression checks for the simulation adapter. Run manually; no model/server needed."""
import unittest
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

from pipeline.frame_transformer import RobotBaseFrame
from pipeline.graspgen.backend import require_motion_allowed
from pipeline.sim.graspgen_motion import GraspGenMotionPreview
from pipeline.sim.gripper_geometry import DEMO, ROBOTIQ_PROXY, for_config
from pipeline.sim.checked_joint_planner import CheckedJointPlanner


class SimulationIsolationTests(unittest.TestCase):
    def test_retiming_preserves_endpoints_and_sampled_dynamics(self):
        # Pure numerical regression: no physics connection or inference.
        planner = CheckedJointPlanner(None, {"motion_control": {"differential_ik": {
            "enabled": True, "joint_max_velocity_rad_s": [.35]*7}}}, lambda: None)
        start, end = np.zeros(7), np.array([.4, -.2, .1, .5, 0., -.3, .2])
        samples = planner.retime("regression", np.asarray([start, end]))
        np.testing.assert_allclose(samples[0], start, atol=1e-12)
        np.testing.assert_allclose(samples[-1], end, atol=1e-12)
        padded = np.vstack([np.repeat(samples[:1], 3, axis=0), samples,
                            np.repeat(samples[-1:], 3, axis=0)])
        limits = [planner.cfg.max_velocity * planner.cfg.velocity_safety_factor,
                  planner.cfg.max_acceleration, planner.cfg.max_jerk]
        for order, limit in enumerate(limits, 1):
            peak = np.max(np.abs(np.diff(padded, n=order, axis=0)), axis=0) / (1/240)**order
            self.assertTrue(np.all(peak <= limit + 1e-6))

    def test_transit_does_not_accept_collision_at_goal(self):
        class Cell:
            limits = [(-3., 3.)]*7
            def joint_margin(self, q):
                return 3.-float(np.max(np.abs(q)))
            def set_joints(self, q):
                self.q = q
        cell = Cell()
        planner = CheckedJointPlanner(cell, {}, lambda: None)
        def clearance():
            if cell.q[0] > .2:
                raise ValueError("obstacle")
        with self.assertRaisesRegex(ValueError, "obstacle"):
            planner.transit(np.zeros(7), np.ones(7)*.5, clearance)

    def test_prepared_official_model_has_tcp_and_mimic_joints(self):
        path = Path(__file__).resolve().parents[1] / "data/external/robotiq-prepared/description.json"
        if not path.is_file():
            self.skipTest("Prepare the official Robotiq assets first")
        metadata = json.loads(path.read_text())
        model = ET.parse(metadata["urdf"]).getroot()
        self.assertIsNotNone(model.find("link[@name='graspgen_tcp']"))
        self.assertEqual(len(model.findall("joint/mimic")), 5)
        widths = metadata["opening_widths_m"]
        self.assertTrue(all(a > b > 0 for a, b in zip(widths[:-1], widths[1:])))
        self.assertGreater(widths[0], .136)
        self.assertLess(widths[-1], .004)

    def test_simulation_authorization_never_unlocks_hardware(self):
        report = {"grasp_backend": "graspgen", "motion": {"allowed": False, "blocked_reasons": ["unverified"]},
                  "simulation_motion": {"allowed": True, "execution_scope": "pybullet_only"}}
        with self.assertRaisesRegex(ValueError, "motion blocked"):
            require_motion_allowed(report, gripper="robotiq_2f_140")

    def test_default_geometry_preserves_existing_scene(self):
        self.assertEqual(for_config({}), DEMO)
        self.assertEqual(DEMO.tcp_offset, (0., 0., .10))
        self.assertEqual(DEMO.opening, .080)

    def test_unknown_geometry_rejected(self):
        with self.assertRaises(ValueError):
            for_config({"simulation_gripper": "unverified_gripper"})

    def test_real_cloud_is_never_executed_by_sim_adapter(self):
        adapter = object.__new__(GraspGenMotionPreview)
        adapter.scene = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "simulated scene"):
            adapter._check_source({"grasp_backend": "graspgen", "source_pipeline": "real"})

    def test_base_mismatch_rejected_before_reading_artifacts(self):
        adapter = object.__new__(GraspGenMotionPreview)
        adapter.scene = SimpleNamespace(gripper_geometry=ROBOTIQ_PROXY, procedural_tool_fallback=False,
                                        base_frame=RobotBaseFrame())
        other = RobotBaseFrame.from_dict({"position_xyz_m": [1., 0., 0.]})
        with self.assertRaisesRegex(ValueError, "base frame differs"):
            adapter._check_source({"grasp_backend": "graspgen", "source_pipeline": "sim",
                                   "robot_base_pose_world": other.to_dict()})

    def test_arbitrary_tcp_transform_is_rejected(self):
        adapter = object.__new__(GraspGenMotionPreview)
        adapter.scene = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "TCP transform differs"):
            adapter._candidate({"T_base_gripper": np.eye(4), "T_base_tcp": np.eye(4)}, np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
