"""Piano di pick & place: logica pura, indipendente da ROS e PyBullet."""

import unittest
from pathlib import Path

from pipeline.config_loader import load_environment_config
from pipeline.motion.pick_and_place import (MotionRefused, READY, grasp_pose_from_report, plan_pick_and_place)

CONFIG = load_environment_config(Path(__file__).resolve().parents[1] / "config/surgical_environment.yaml")
REPORT = {"readiness": READY, "designated_grasp_geometry": {
    "grasp_centroid_base_frame_xyz_m": [0.55, -0.40, 0.09],
    "recommended_gripper_orientation": {"quaternion_xyzw": [0.0, 0.7071, 0.7071, 0.0]}}}


class PickAndPlacePlanTests(unittest.TestCase):
    def test_unvalidated_grasp_is_refused(self):
        for report in (None, {}, {"readiness": "NOT_READY", "readiness_blockers": ["frame EE nominale"]}):
            with self.assertRaises(MotionRefused):
                grasp_pose_from_report(report)
        with self.assertRaisesRegex(MotionRefused, "frame EE nominale"):
            grasp_pose_from_report({"readiness": "NOT_READY", "readiness_blockers": ["frame EE nominale"]})

    def test_unknown_tray_is_refused(self):
        with self.assertRaises(MotionRefused):
            plan_pick_and_place([0.5, 0, 0.1], None, "tray_99", CONFIG)

    def test_sequence_structure_and_heights(self):
        xyz, quat = grasp_pose_from_report(REPORT)
        plan = plan_pick_and_place(xyz, quat, "tray_1", CONFIG)
        self.assertEqual([s.gripper_after for s in plan.sequences], ["grasp", "release", None])
        heights = CONFIG["motion_control"]["heights_z_m"]
        grasp_sequence = plan.sequences[0].waypoints
        self.assertEqual(grasp_sequence[0][0][2], heights["staging_z"])
        self.assertEqual(grasp_sequence[1][0][2], heights["approach_z"])
        self.assertEqual(plan.sequences[0].waypoints[-1][0], plan.grasp_xyz)
        self.assertEqual(plan.sequences[1].waypoints[-1][0], plan.deposit_xyz)
        for sequence in plan.sequences:
            for position, orientation in sequence.waypoints:
                self.assertEqual((len(position), len(orientation)), (3, 4))

    def test_grasp_height_is_clamped_and_default_orientation_is_downward(self):
        plan = plan_pick_and_place([0.5, 0.1, 0.01], None, "tray_1", CONFIG)
        self.assertEqual(plan.grasp_xyz[2], 0.12)
        self.assertEqual(plan.sequences[0].waypoints[1][1], CONFIG["motion_control"]["downward_quaternion_xyzw"])
        tray = CONFIG["environment"]["trays"]["tray_1"]
        self.assertAlmostEqual(plan.deposit_xyz[2], tray["position_xyz_m"][2] + tray["dimensions_xyz_m"][2] / 2 + 0.03)


if __name__ == "__main__":
    unittest.main()
