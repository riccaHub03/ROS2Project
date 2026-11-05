"""Controlli su scena condivisa, rifiuto IK e trasferimento misurato in DIRECT."""

import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import pybullet as p
import trimesh
import yaml

from pipeline.config_loader import load_environment_config
from pipeline.sim import robot_scene
from pipeline.sim.robot_motion_preview import MotionPreview
from pipeline.frame_transformer import FrameTransformer, RobotBaseFrame, matrix_from_quaternion
from pipeline.sim.sim_step4_transforms import run_sim_transforms
from pipeline.step5_grasp_geometry import parse_ply, run_grasp_geometry


class RobotPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.root_patch = patch.object(robot_scene, "ROOT", root)
        self.root_patch.start()
        mesh = root / "instrument.stl"
        trimesh.creation.box([0.14, 0.012, 0.004]).export(mesh)
        config = load_environment_config()
        config["environment"]["kuka_base"].update(
            position_xyz_m=[0, 0, 0], orientation_rpy_rad=[0, 0, 0])
        config["surgical_instrument"]["stl_path"] = str(mesh)
        plan = {"target_instrument": "Scalpel", "source_tray": "tray_2", "destination_tray": "tray_1"}
        self.config, self.plan = config, plan
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
                                   placement["placed_pose_world_frame"]["position_xyz_m"])
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

    def test_translated_and_rotated_base_changes_robot_only(self):
        old_tcp, old_quat = self.scene.tcp_pose()
        old_tool = self.scene.tool_initial_pos
        old_camera = self.scene.camera_report["camera_metadata"]
        self.scene.close()
        config = copy.deepcopy(self.config)
        config["environment"]["kuka_base"].update(
            position_xyz_m=[0.06, 0.02, 0.01], orientation_rpy_rad=[0.04, -0.03, 0.1])
        self.scene = robot_scene.RobotScene(config, self.plan)
        frame = self.scene.base_frame
        expected_tcp, expected_quat = p.multiplyTransforms(
            frame.position, p.getQuaternionFromEuler(frame.rpy), old_tcp, old_quat)
        actual_tcp, actual_quat = self.scene.tcp_pose()
        np.testing.assert_allclose(actual_tcp, expected_tcp, atol=1e-6)
        np.testing.assert_allclose(matrix_from_quaternion(actual_quat),
                                   matrix_from_quaternion(expected_quat), atol=1e-6)
        np.testing.assert_allclose(self.scene.tool_initial_pos, old_tool)
        self.assertEqual(self.scene.camera_report["camera_metadata"], old_camera)
        placement = json.loads((Path(self.temp.name) / "data/outputs/stage3_stl_placement.json").read_text())
        np.testing.assert_allclose(frame.base_to_world(placement["placed_pose_base_frame"]["position_xyz_m"]),
                                   old_tool)

    def test_camera_points_and_grasp_are_invariant_in_world(self):
        root = Path(self.temp.name)
        camera = copy.deepcopy(self.scene.camera_report)
        mask_path = root / "mask-report.json"
        mask_path.write_text(json.dumps({"files": {"camera_mask_png": camera["files"]["ground_truth_mask"]}}))
        outputs = []
        for name, position, rpy in [("origin", [0, 0, 0], [0, 0, 0]),
                                     ("tilted", [0.06, -0.03, 0.02], [0.2, -0.15, 0.3])]:
            config = copy.deepcopy(self.config)
            config["environment"]["kuka_base"].update(position_xyz_m=position, orientation_rpy_rad=rpy)
            frame = RobotBaseFrame.from_config(config)
            case = root / name
            case.mkdir()
            cfg_path = case / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(config))
            camera["robot_base_pose_world"] = frame.to_dict()
            camera_path = case / "camera.json"
            camera_path.write_text(json.dumps(camera))
            transforms = run_sim_transforms(mask_report_path=str(mask_path), cam_report_path=str(camera_path),
                                             config_path=str(cfg_path), output_path=str(case / "transforms.json"))
            points, _ = parse_ply(transforms["pointcloud_file"])
            np.testing.assert_allclose(points.mean(axis=0), transforms["frames"]["robot_base_frame"], atol=0.0001)
            grasp = run_grasp_geometry(transforms_report_path=str(case / "transforms.json"),
                                        config_path=str(cfg_path), output_path=str(case / "grasp.json"))
            data = grasp["designated_grasp_geometry"]
            world_pos, world_quat = frame.pose_base_to_world(
                data["grasp_centroid_base_frame_xyz_m"],
                data["recommended_gripper_orientation"]["quaternion_xyzw"])
            outputs.append((frame.base_to_world(points), world_pos, matrix_from_quaternion(world_quat),
                            grasp["input_pointcloud"]["filtered_points"]))
        np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=2e-6)
        np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=0.0002)
        np.testing.assert_allclose(outputs[0][2], outputs[1][2], atol=0.0002)
        self.assertEqual(outputs[0][3], outputs[1][3])

    def test_pick_and_place_with_moved_and_yawed_base(self):
        self.scene.close()
        config = copy.deepcopy(self.config)
        config["environment"]["kuka_base"].update(
            position_xyz_m=[0.04, 0.01, 0], orientation_rpy_rad=[0, 0, 0.12])
        self.scene = robot_scene.RobotScene(config, self.plan)
        world_pos = np.asarray(self.scene.tool_initial_pos) + [0, 0, 0.002]
        base_pos, base_quat = self.scene.base_frame.pose_world_to_base(world_pos, [0, 1, 0, 0])
        grasp = {"designated_grasp_geometry": {
            "gripper_compatibility_check": {"parallel_jaw_suitable": True},
            "grasp_centroid_base_frame_xyz_m": base_pos.tolist(),
            "local_cross_section_m": {"width_along_v2": 0.012},
            "recommended_gripper_orientation": {"quaternion_xyzw": base_quat},
        }}
        motion = MotionPreview(self.scene, grasp)
        np.testing.assert_allclose(motion.waypoints[1]["tcp_position_m"], world_pos)
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
        self.assertEqual(report["waypoints_frame_id"], "world")

    def test_far_base_rejects_unreachable_grasp_before_motion(self):
        self.scene.close()
        config = copy.deepcopy(self.config)
        config["environment"]["kuka_base"]["position_xyz_m"] = [2, 0, 0]
        self.scene = robot_scene.RobotScene(config, self.plan)
        pos, quat = self.scene.base_frame.pose_world_to_base(self.scene.tool_initial_pos, [0, 1, 0, 0])
        grasp = {"designated_grasp_geometry": {
            "gripper_compatibility_check": {"parallel_jaw_suitable": True},
            "grasp_centroid_base_frame_xyz_m": pos.tolist(),
            "local_cross_section_m": {"width_along_v2": 0.012},
            "recommended_gripper_orientation": {"quaternion_xyzw": quat},
        }}
        before = self.scene.tcp_pose()[0]
        with self.assertRaises(ValueError):
            MotionPreview(self.scene, grasp)
        np.testing.assert_allclose(self.scene.tcp_pose()[0], before)
        self.assertIsNone(self.scene.attachment)

    def test_grasp_from_a_different_base_pose_is_rejected(self):
        other = RobotBaseFrame([0.1, 0, 0])
        grasp = {"robot_base_pose_world": other.to_dict()}
        with self.assertRaisesRegex(ValueError, "ricalcola"):
            MotionPreview(self.scene, grasp)


class RobotBaseFrameTests(unittest.TestCase):
    def test_camera_ray_and_pose_with_translated_rotated_base(self):
        base = RobotBaseFrame([0.1, -0.2, 0.03], [0.1, -0.08, 0.2])
        transformer = FrameTransformer([0.55, 0, 0.85], [0.55, 0, 0.05], robot_base_frame=base)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[240, 320] = 255
        depth = np.full(mask.shape, 0.8)
        camera_pts, base_pts = transformer.deproject_mask_to_pointcloud(mask, depth)
        np.testing.assert_allclose(camera_pts[0], [0, 0, 0.8])
        np.testing.assert_allclose(base.base_to_world(base_pts[0]), [0.55, 0, 0.05], atol=1e-12)
        np.testing.assert_allclose(transformer.transform_camera_to_world(camera_pts[0]), [0.55, 0, 0.05])
        np.testing.assert_allclose(transformer.transform_world_to_camera([0.55, 0, 0.05]), camera_pts[0])
        world_pos, world_quat = base.pose_base_to_world([0.3, 0.1, 0.2], [0, 1, 0, 0])
        expected_pos, expected_quat = p.multiplyTransforms(
            base.position, p.getQuaternionFromEuler(base.rpy), [0.3, 0.1, 0.2], [0, 1, 0, 0])
        np.testing.assert_allclose(world_pos, expected_pos, atol=1e-7)
        np.testing.assert_allclose(matrix_from_quaternion(world_quat), matrix_from_quaternion(expected_quat), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
