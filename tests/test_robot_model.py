"""Official iiwa 7 limits, analytic FK and shared scene/validator/TCP contracts."""
import copy
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
import trimesh

from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.robot_cell import RobotCell
from pipeline.robot_model import RobotModel, robot_settings
from pipeline.sim.gripper_geometry import DEMO, ROBOTIQ_PROXY
from pipeline.sim.robot_scene import RobotScene
from pipeline.sim.constrained_differential_ik import build_for_scene, DifferentialIKConfig


def analytic_flange(model, q):
    """Independent URDF chain product, including the fixed 35 mm flange joint."""
    xml = ET.parse(model.urdf).getroot()
    children = {j.find("child").get("link"): j for j in xml.findall("joint")}
    values = dict(zip(model.joint_names, q))
    chain, link = [], model.flange_link
    while link != model.settings["base_link"]:
        joint = children[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    transform = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
        step = np.eye(4)
        step[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        step[:3, 3] = xyz
        transform = transform @ step
        if joint.get("type") != "fixed":
            axis = np.asarray([float(x) for x in joint.find("axis").get("xyz").split()])
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * values[joint.get("name")]).as_matrix()
            transform = transform @ motion
    return transform


class RobotModelTests(unittest.TestCase):
    def test_default_is_official_iiwa7_with_source_limits_and_flange(self):
        model = RobotModel.from_config()
        self.assertEqual(model.model_id, "iiwa7_r800")
        self.assertEqual(model.flange_link, "lbr_link_ee")
        self.assertEqual(model.joint_names, tuple(f"lbr_A{i}" for i in range(1, 8)))
        np.testing.assert_allclose(model.upper, np.deg2rad([170, 120, 170, 120, 170, 120, 175]))
        np.testing.assert_allclose(model.lower, -np.asarray(model.upper))
        np.testing.assert_allclose(model.velocity, np.deg2rad([98, 98, 100, 130, 140, 180, 180]))
        self.assertEqual(model.effort, (200.,) * 7)
        # Official zero-joint height: 1.231 m at link 7, 1.266 m at flange.
        np.testing.assert_allclose(analytic_flange(model, np.zeros(7))[:3, 3], [0, 0, 1.266], atol=1e-12)

    def test_no_silent_iiwa14_fallback_and_legacy_override_is_explicit(self):
        legacy = RobotModel.from_config({"robot": {"model": "pybullet_iiwa14"}})
        self.assertNotEqual(legacy.model_sha256, RobotModel.from_config().model_sha256)
        with self.assertRaises(FileNotFoundError):
            RobotModel.from_config({"robot": {"urdf": "/missing/iiwa7.urdf"}})
        with self.assertRaises(ValueError):
            RobotCell(load_environment_config(), RobotBaseFrame(), robot_urdf=str(legacy.urdf))

    def test_changed_geometry_or_source_pin_is_rejected(self):
        settings = robot_settings()
        with tempfile.TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "prepared"
            shutil.copytree(Path(settings["urdf"]).parent, prepared)
            config = {"robot": dict(model="iiwa7_r800", urdf=str(prepared/"iiwa7.urdf"),
                                     metadata=str(prepared/"description.json"))}
            RobotModel.from_config(config)
            mesh = next((prepared/"meshes").glob("*.stl"))
            mesh.write_bytes(mesh.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError, "asset changed"):
                RobotModel.from_config(config)
        with self.assertRaisesRegex(ValueError, "source"):
            RobotModel.from_config({"robot": {"source": {"repository": "wrong", "revision": "wrong"}}})

    def test_controller_velocity_is_clamped_to_the_same_urdf(self):
        config = load_environment_config()
        config["motion_control"]["differential_ik"].update(enabled=True, joint_max_velocity_rad_s=[100.]*7)
        np.testing.assert_allclose(DifferentialIKConfig.from_config(config).max_velocity,
                                   RobotModel.from_config(config).velocity)
        config["motion_control"]["differential_ik"]["joint_max_velocity_rad_s"] = .1
        np.testing.assert_allclose(DifferentialIKConfig.from_config(config).max_velocity, [.1]*7)


class SharedKinematicsTests(unittest.TestCase):
    def test_closing_an_old_fk_copy_cannot_disconnect_a_reused_physics_client(self):
        from pipeline.sim.constrained_differential_ik import PyBulletKinematics
        model = RobotModel.from_config()
        clone = PyBulletKinematics(str(model.urdf), [0, 0, 0], [0, 0, 0, 1], list(range(7)), 7, [0, 0, .1])
        previous_id = clone.cid
        clone.close()
        current = p.connect(p.DIRECT)
        try:
            self.assertEqual(current, previous_id)
            clone.close()
            self.assertTrue(p.isConnected(current))
        finally:
            p.disconnect(current)

    def test_scene_validator_and_private_fk_match_the_analytic_chain_with_both_tcps(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "fixture.stl"
            trimesh.creation.box([.12, .02, .006]).export(mesh)
            rng = np.random.default_rng(7)
            for geometry in (DEMO, ROBOTIQ_PROXY):
                with self.subTest(gripper=geometry.name):
                    config = load_environment_config()
                    config["simulation_gripper"] = geometry.name
                    config["surgical_instrument"]["stl_path"] = str(mesh)
                    config["environment"]["kuka_base"].update(position_xyz_m=[.04, .01, .02],
                                                               orientation_rpy_rad=[.03, -.02, .2])
                    config["motion_control"]["differential_ik"]["enabled"] = True
                    task = dict(target_instrument="Scalpel", source_tray="tray_2", destination_tray="tray_1")
                    scene = RobotScene(config, task, capture=False)
                    cell = RobotCell(config, scene.base_frame)
                    _, clone = build_for_scene(scene)
                    try:
                        self.assertEqual(scene.robot_model.model_sha256, cell.robot_model.model_sha256)
                        self.assertEqual(scene.ee, 7)  # includes the fixed physical flange offset
                        poses = [np.zeros(7), scene.home]
                        poses += list(rng.uniform(np.asarray(cell.robot_model.lower)*.6,
                                                  np.asarray(cell.robot_model.upper)*.6, size=(8, 7)))
                        for q in poses:
                            for index, value in zip(scene.joints, q):
                                p.resetJointState(scene.robot, index, value, physicsClientId=scene.cid)
                            cell.set_joints(q)
                            flange = analytic_flange(cell.robot_model, q)
                            tcp_base = flange[:3, 3] + flange[:3, :3] @ geometry.tcp_offset
                            tcp_world = scene.base_frame.base_to_world(tcp_base)
                            expected_rotation = scene.base_frame.rotation @ flange[:3, :3]
                            np.testing.assert_allclose(scene.tcp_pose()[0], tcp_world, atol=3e-7)
                            np.testing.assert_allclose(cell.tcp_pose()[0], tcp_base, atol=3e-7)
                            cloned_position, cloned_rotation = clone.fk(q)
                            np.testing.assert_allclose(cloned_position, tcp_world, atol=3e-7)
                            np.testing.assert_allclose(cloned_rotation, expected_rotation, atol=6e-7)
                            solution = cell.solve_ik(tcp_base, cell.tcp_pose()[1], q)
                            self.assertTrue(solution.ok)
                            self.assertLess(solution.position_error_m, .0003)
                    finally:
                        clone.close()
                        cell.close()
                        scene.close()


if __name__ == "__main__":
    unittest.main()
