"""Controlli su scena condivisa, rifiuto IK e trasferimento misurato in DIRECT."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import pybullet as p
import trimesh

from pipeline.config_loader import load_environment_config
from pipeline.sim import robot_scene
from pipeline.sim.robot_motion_preview import MotionPreview


class RobotPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.root_patch = patch.object(robot_scene, "ROOT", root)
        self.root_patch.start()
        mesh = root / "instrument.stl"
        trimesh.creation.box([0.14, 0.012, 0.004]).export(mesh)
        config = load_environment_config()
        config["surgical_instrument"]["stl_path"] = str(mesh)
        plan = {"target_instrument": "Scalpel", "source_tray": "tray_2", "destination_tray": "tray_1"}
        self.scene = robot_scene.RobotScene(config, plan, seed=42)

    def tearDown(self):
        self.scene.close()
        self.root_patch.stop()
        self.temp.cleanup()

    def test_camera_and_placement_use_same_pose(self):
        out = Path(self.temp.name) / "data/outputs"
        camera = json.loads((out / "sim_stage2_camera.json").read_text())
        placement = json.loads((out / "stage3_stl_placement.json").read_text())
        self.assertEqual(camera["source_stl_file"], placement["source_stl_file"])
        np.testing.assert_allclose(camera["settled_ground_truth_pose"]["position_xyz_m"],
                                   placement["placed_pose_base_frame"]["position_xyz_m"])
        mask = np.asarray(Image.open(camera["files"]["ground_truth_mask"]))
        self.assertGreater(np.count_nonzero(mask), 15)
        depth = np.load(camera["files"]["camera_depth_npy"])
        self.assertTrue(np.isfinite(depth[mask > 0]).all())

    def test_unreachable_ik_does_not_move_robot(self):
        before = [p.getJointState(self.scene.robot, j, physicsClientId=self.scene.cid)[0] for j in range(7)]
        with self.assertRaises(ValueError):
            self.scene.solve_ik([2.0, 0, 0.1], [0, 1, 0, 0], before)
        after = [p.getJointState(self.scene.robot, j, physicsClientId=self.scene.cid)[0] for j in range(7)]
        np.testing.assert_allclose(before, after, atol=1e-10)

    def test_motion_finishes_with_contact_on_destination(self):
        position = list(self.scene.tool_initial_pos)
        position[2] += 0.002
        grasp = {"designated_grasp_geometry": {
            "gripper_compatibility_check": {"parallel_jaw_suitable": True},
            "grasp_centroid_base_frame_xyz_m": position,
            "local_cross_section_m": {"width_along_v2": 0.012},
            "recommended_gripper_orientation": {"quaternion_xyzw": [0, 1, 0, 0]},
        }}
        motion = MotionPreview(self.scene, grasp)
        for _ in range(20000):
            motion.tick()
            self.scene.step()
            if motion.finished:
                break
        for _ in range(120):
            self.scene.step()
        report = motion.report()
        self.assertEqual(report["status"], "COMPLETED")
        self.assertTrue(report["tool_settled_on_destination"])
        self.assertGreater(report["destination_contact_count"], 0)
        self.assertGreater(report["final_tool_position_m"][1], 0.3)
        self.assertFalse(report["physical_grasp_validated"])
        self.assertTrue(all(item["tcp_error_mm"] < 10 for item in report["measured_phase_errors"]))


if __name__ == "__main__":
    unittest.main()
