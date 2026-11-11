"""Sorgente dal vivo della pipeline reale: conversioni, TF, cattura e deproiezione, senza ROS ne' hardware."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

from pipeline.common import paths
from pipeline.common.reconstruction import CameraModel
from pipeline.common.report import StageError, StageReport
from pipeline.common.scene import Intrinsics
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import FrameTransformer, matrix_from_quaternion, quaternion_from_matrix
from pipeline.real import capture, deproject, preflight, ros_camera

ROOT = Path(__file__).resolve().parents[1]
H, W = 48, 64


def color_msg(encoding="rgb8", pad=0):
    pixels = np.zeros((H, W, 3), np.uint8)
    pixels[..., 0], pixels[..., 1], pixels[..., 2] = 10, 20, 30          # R, G, B
    channels = {"rgb8": pixels, "bgr8": pixels[..., ::-1]}[encoding]
    rows = np.concatenate([channels.reshape(H, W * 3), np.zeros((H, pad), np.uint8)], axis=1)
    return SimpleNamespace(encoding=encoding, height=H, width=W, step=W * 3 + pad, data=rows.tobytes())


def depth_msg(values_m, encoding="16UC1", big=False):
    if encoding == "16UC1":
        data = (np.asarray(values_m) * 1000).astype(">u2" if big else "<u2")
    else:
        data = np.asarray(values_m, np.float32).astype(">f4" if big else "<f4")
    return SimpleNamespace(encoding=encoding, height=H, width=W, step=W * data.itemsize,
                           is_bigendian=int(big), data=data.tobytes())


def camera_info(k0=600.0, distortion=0.0):
    return SimpleNamespace(k=[k0, 0, W / 2, 0, k0, H / 2, 0, 0, 1], d=[distortion] * 5, width=W, height=H,
                           header=SimpleNamespace(frame_id="cam_optical"))


def nadir_quaternion():
    return quaternion_from_matrix(FrameTransformer(camera_position_xyz=(0, 0, 1), camera_target_xyz=(0, 0, 0)).R_world_opt)


class ConversionTests(unittest.TestCase):
    def test_color_orders_and_row_padding(self):
        for encoding in ("rgb8", "bgr8"):
            rgb = ros_camera.image_to_rgb(color_msg(encoding, pad=5))
            self.assertEqual(rgb.shape, (H, W, 3))
            self.assertEqual(tuple(rgb[0, 0]), (10, 20, 30))
        with self.assertRaisesRegex(StageError, "unsupported colour encoding"):
            ros_camera.image_to_rgb(SimpleNamespace(encoding="mono8", height=H, width=W, step=W, data=b""))

    def test_depth_units_endianness_and_invalid(self):
        values = np.full((H, W), 1.25)
        values[0, 0] = 0
        for encoding in ("16UC1", "32FC1"):
            for big in (False, True):
                depth = ros_camera.depth_to_meters(depth_msg(values, encoding, big))
                self.assertAlmostEqual(float(depth[5, 5]), 1.25, places=3)
                self.assertEqual(float(depth[0, 0]), 0.0)
        nan = np.full((H, W), np.nan)
        self.assertEqual(float(ros_camera.depth_to_meters(depth_msg(nan, "32FC1")).max()), 0.0)
        with self.assertRaisesRegex(StageError, "unsupported depth encoding"):
            ros_camera.depth_to_meters(SimpleNamespace(encoding="8UC1", height=H, width=W, step=W, data=b""))

    def test_camera_info_rejects_uncalibrated_and_flags_distortion(self):
        intrinsics, distortion = ros_camera.camera_info_to_intrinsics(camera_info())
        self.assertEqual((intrinsics.fx, intrinsics.cx, intrinsics.cy), (600.0, W / 2, H / 2))
        self.assertFalse(distortion)
        self.assertTrue(ros_camera.camera_info_to_intrinsics(camera_info(distortion=0.1))[1])
        with self.assertRaisesRegex(StageError, "not calibrated"):
            ros_camera.camera_info_to_intrinsics(camera_info(k0=0.0))

    def test_median_depth_ignores_outliers_and_marks_sparse_pixels_invalid(self):
        good = np.full((H, W), 1.0, np.float32)
        spike = good.copy()
        spike[3, 3] = 9.0
        hole = good.copy()
        hole[4, 4] = 0
        frames = [good, good, spike, hole, good]
        result = ros_camera.median_depth(frames)
        self.assertEqual(float(result[3, 3]), 1.0)
        self.assertEqual(float(result[4, 4]), 1.0)             # un solo fotogramma senza dato: resta valido
        sparse = [np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), good]
        self.assertEqual(float(ros_camera.median_depth(sparse)[5, 5]), 0.0)

    def test_option_precedence(self):
        config = {"camera_overhead": {"ros2": {"color_topic": "/yaml/color", "timeout_s": 3.0}}}
        options = ros_camera.resolve_options(config, {"color_topic": None, "timeout_s": 7.0, "ee_frame": "flange"})
        self.assertEqual((options["color_topic"], options["timeout_s"], options["ee_frame"]),
                         ("/yaml/color", 7.0, "flange"))
        self.assertEqual(options["depth_topic"], ros_camera.DEFAULT_OPTIONS["depth_topic"])


class FrameTests(unittest.TestCase):
    def test_explicit_orientation_equals_lookat_for_the_nadir_camera(self):
        kwargs = dict(camera_position_xyz=(0.6, -0.42, 1.15), camera_target_xyz=(0.6, -0.42, 0.07),
                      camera_intrinsics={"fx_px": 600, "fy_px": 600, "cx_px": 32, "cy_px": 24})
        lookat = FrameTransformer(**kwargs)
        explicit = FrameTransformer(**kwargs, camera_orientation_xyzw=nadir_quaternion())
        np.testing.assert_allclose(lookat.R_world_opt, explicit.R_world_opt, atol=1e-9)
        np.testing.assert_allclose(lookat.deproject_pixel_to_camera_frame(40, 10, 1.0),
                                   explicit.deproject_pixel_to_camera_frame(40, 10, 1.0))

    def test_camera_model_orientation_roundtrip(self):
        model = CameraModel([0.1, 0.2, 1.0], [0.1, 0.2, 0.0], Intrinsics(W, H, 600, 600, 32, 24), nadir_quaternion())
        again = CameraModel.from_dict(model.to_dict())
        np.testing.assert_allclose(again.orientation_xyzw, model.orientation_xyzw, atol=1e-5)
        self.assertIsNone(CameraModel.from_dict({**model.to_dict(), "orientation_optical_xyzw": None}).orientation_xyzw)
        legacy = model.to_dict()
        legacy.pop("orientation_optical_xyzw")
        self.assertIsNone(CameraModel.from_dict(legacy).orientation_xyzw)


class LiveCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name) / "out"
        self.out.mkdir()
        config = load_environment_config(ROOT / "config/surgical_environment.yaml")
        config["camera_overhead"]["extrinsics"]["verified"] = True
        self.config_file = Path(self.temp.name) / "config.yaml"
        self.config_file.write_text(yaml.safe_dump(config))
        depth = np.full((H, W), 1.06, np.float32)
        depth[20:30, 20:44] = 1.048
        self.frame = ros_camera.LiveFrame(
            rgb=np.full((H, W, 3), 120, np.uint8), depth_m=depth, intrinsics=Intrinsics(W, H, 600, 600, W / 2, H / 2),
            distortion_nonzero=False, camera_frame="cam_optical", world_frame="world", color_stamp=10.0,
            depth_stamp=10.01, depth_valid_fraction=1.0, depth_frames_used=5,
            camera_pose_world=([0.6, -0.42, 1.15], nadir_quaternion()),
            ee_pose_world=([0.45, 0.0, 0.45], [0.0, 1.0, 0.0, 0.0]),
            options=dict(ros_camera.DEFAULT_OPTIONS, ee_frame="lbr_link_ee"))

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, frame=None):
        return capture.run(source="ros2", live_frame=frame or self.frame, config_path=str(self.config_file),
                           out_dir=str(self.out))

    def test_live_capture_is_motion_safe_with_verified_extrinsics(self):
        report = self.capture()
        data = report.data
        self.assertTrue(data["motion_safe"])
        self.assertEqual((data["source_type"], data["depth_source"]), ("live_ros2_rgbd", "measured_live_ros2"))
        self.assertEqual(data["ee_reference_pose"]["source"], "tf")
        self.assertTrue(data["acquisition"]["camera_pose_from_tf"])
        self.assertIn("orientation_optical_xyzw", data["camera"])
        np.testing.assert_allclose(np.load(report.artifact("depth_npy")), self.frame.depth_m)

    def test_missing_tf_or_unverified_extrinsics_block_motion(self):
        no_tf = ros_camera.LiveFrame(**{**self.frame.__dict__, "camera_pose_world": None})
        report = self.capture(no_tf)
        self.assertFalse(report.data["motion_safe"])
        self.assertTrue(any("NOMINAL camera pose" in w for w in report.warnings))
        config = yaml.safe_load(self.config_file.read_text())
        config["camera_overhead"]["extrinsics"]["verified"] = False
        self.config_file.write_text(yaml.safe_dump(config))
        self.assertFalse(self.capture().data["motion_safe"])

    def test_distortion_and_sparse_depth_are_warned(self):
        noisy = ros_camera.LiveFrame(**{**self.frame.__dict__, "distortion_nonzero": True, "depth_valid_fraction": 0.2})
        warnings = " ".join(self.capture(noisy).warnings)
        self.assertIn("distortion", warnings)
        self.assertIn("20%", warnings)

    def test_deprojection_uses_the_live_ee_pose(self):
        self.capture()
        mask = np.zeros((H, W), np.uint8)
        mask[20:30, 20:44] = 255
        mask_png = self.out / "real_segmentation_mask.png"
        Image.fromarray(mask).save(mask_png)
        StageReport("real_segmentation", artifacts={"mask_png": str(mask_png)}, data={}).save(
            self.out / paths.REAL_SEGMENTATION)
        report = deproject.run(config_path=str(self.config_file), out_dir=str(self.out))
        self.assertEqual(report.data["ee_reference_source"], "tf")
        self.assertTrue(report.data["motion_safe"])
        self.assertGreater(report.data["num_points"], 100)
        centroid_world = np.asarray(report.data["frames"]["world_frame"])
        # camera a z=1.15 m sopra il punto (0.6, -0.42), strumento a 1.048 m di profondita' -> z = 0.102 m
        np.testing.assert_allclose(centroid_world, [0.6, -0.42, 0.102], atol=0.02)

    def test_source_arguments_default_to_file(self):
        import argparse
        parser = argparse.ArgumentParser()
        capture.add_source_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(capture.source_kwargs(args)["source"], "file")
        args = parser.parse_args(["--source", "ros2", "--ee-frame", "flange", "--timeout", "4"])
        self.assertEqual(capture.source_kwargs(args)["ros"]["ee_frame"], "flange")
        self.assertEqual(capture.source_kwargs(args)["ros"]["timeout_s"], 4.0)


class PreflightTests(unittest.TestCase):
    def test_file_source_reports_config_blockers_without_ros(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(preflight, "check_software", lambda *a, **k: None):
                report = preflight.run(source="file", out_dir=tmp)
            names = {c["name"]: c["status"] for c in report.data["checks"]}
            self.assertEqual(names["config: estrinseche NON verificate"], "WARN")
            self.assertFalse(report.data["ready_for_motion"])
            self.assertTrue((Path(tmp) / "real_preflight.json").exists())

    def test_world_equal_to_base_requires_identity_base_pose(self):
        config = load_environment_config(ROOT / "config/surgical_environment.yaml")
        config["environment"]["kuka_base"]["position_xyz_m"] = [0.1, 0.0, 0.0]
        checks = preflight.Checks()
        preflight.check_config(checks, config, dict(ros_camera.DEFAULT_OPTIONS, world_frame="lbr_link_0"), ee_from_tf=True)
        self.assertTrue(checks.has("FAIL"))
        checks = preflight.Checks()
        preflight.check_config(checks, config, dict(ros_camera.DEFAULT_OPTIONS, world_frame="world"), ee_from_tf=True)
        self.assertFalse(checks.has("FAIL"))


if __name__ == "__main__":
    unittest.main()
