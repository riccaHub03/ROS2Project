"""Safety gates added for the real robot: tray parsing, confidence, calibration flags, reachability."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from PIL import Image

from pipeline import grasp, plan
from pipeline.common import paths
from pipeline.common.reconstruction import reject_depth_outliers
from pipeline.common.report import StageError, StageReport
from pipeline.common.vision import Instances
from pipeline.common.workspace import Workspace
from pipeline.real import capture as real_capture
from pipeline.real import deproject as real_deproject
from pipeline.real import segment as real_segment
from pipeline.sim.placement import sample_tool_pose


class FakeSegmenter:
    def __init__(self, score, box=(500, 530, 800, 1000), shape=(1080, 1920)):
        self.score, self.box, self.shape = score, box, shape

    def segment(self, image, prompt):
        y0, y1, x0, x1 = self.box
        mask = np.zeros((1,) + self.shape, dtype=bool)
        mask[0, y0:y1, x0:x1] = True
        return Instances(mask, np.array([self.score]))


class PlanTrayTests(unittest.TestCase):
    def test_direction_is_respected(self):
        self.assertEqual(plan.parse_trays("Passami il bisturi dal vassoio 1 al vassoio 2"), ("tray_1", "tray_2"))
        self.assertEqual(plan.parse_trays("Passami il bisturi dal vassoio 2 al vassoio 1"), ("tray_2", "tray_1"))
        self.assertEqual(plan.parse_trays("pass the scalpel from tray 1 to tray 2"), ("tray_1", "tray_2"))
        self.assertEqual(plan.parse_trays("put it into tray 1"), ("tray_2", "tray_1"))
        self.assertEqual(plan.parse_trays("Passami le forbici"), ("tray_2", "tray_1"))

    def test_same_source_and_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(plan, "deterministic_plan", return_value={
                    "action": "pick_and_place", "target_instrument": "scalpel", "target_keyword": "scalpel",
                    "source_tray": "tray_1", "destination_tray": "tray_1"}):
                with self.assertRaises(StageError):
                    plan.run("irrelevant", offline=True, out_dir=tmp)


class RealChainSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict("os.environ", {paths.ROOT_ENV: str(cls.tmp)})
        cls.env.start()
        cls.out = paths.output_dir()
        arr = np.full((1080, 1920, 3), 200, np.uint8)
        arr[500:530, 800:1000] = 30
        cls.image = cls.tmp / "photo.png"
        Image.fromarray(arr).save(cls.image)
        plan.run("Passami il bisturi dal vassoio 2 al vassoio 1", offline=True, out_dir=str(cls.out))

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _capture(self, **kw):
        return real_capture.run(image=str(self.image), out_dir=str(self.out), **kw)

    def test_low_confidence_fails_the_segmentation_stage(self):
        self._capture()
        with self.assertRaises(StageError):
            real_segment.run(out_dir=str(self.out), segmenter=FakeSegmenter(0.05))
        self.assertEqual(StageReport.load(self.out / paths.REAL_SEGMENTATION, require_completed=False).status, "FAILED")

    def test_huge_mask_is_rejected(self):
        self._capture()
        with self.assertRaises(StageError):
            real_segment.run(out_dir=str(self.out), segmenter=FakeSegmenter(0.9, box=(0, 1080, 0, 1920)))

    def test_nominal_capture_is_not_cleared_for_motion(self):
        cap = self._capture()
        self.assertFalse(cap.data["motion_safe"])
        self.assertEqual(cap.data["source_tray"], "tray_2")
        # nominal plane sits on the tray SURFACE (0.09 m), not on its centre (0.07 m)
        self.assertAlmostEqual(cap.data["camera"]["position_xyz_m"][2] - cap.data["nominal_distance_m"], 0.09, places=3)
        real_segment.run(out_dir=str(self.out), segmenter=FakeSegmenter(0.9))
        real_deproject.run(out_dir=str(self.out))
        rep = grasp.run(deprojection_path=str(self.out / paths.REAL_DEPROJECTION), out_dir=str(self.out))
        self.assertEqual(rep.data["readiness"], grasp.NOT_READY)
        self.assertTrue(any("nominal depth" in b for b in rep.data["readiness_blockers"]))

    def _verified_config(self, ee_pose=True):
        """Environment config where the lab has verified the camera extrinsics (and set the EE pose)."""
        cfg = yaml.safe_load(Path(paths.__file__).resolve().parents[2].joinpath(
            "config/surgical_environment.yaml").read_text())
        cfg["camera_overhead"]["extrinsics"]["verified"] = True
        if ee_pose:
            cfg["motion_control"]["ee_reference_pose"] = {"position_xyz_m": [0.45, 0.0, 0.45],
                                                          "quaternion_xyzw": [0.0, 1.0, 0.0, 0.0]}
        path = self.tmp / "verified_env.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return str(path)

    def _measured_inputs(self):
        cap = self._capture()
        depth_file = self.tmp / "depth.npy"
        np.save(depth_file, np.load(cap.artifact("depth_npy")))
        return str(depth_file), (1800.0, 1800.0, 960.0, 540.0)

    def test_depth_and_intrinsics_alone_do_not_clear_nominal_extrinsics(self):
        depth_file, intrinsics = self._measured_inputs()
        cap = self._capture(depth=depth_file, intrinsics=intrinsics)
        self.assertEqual(cap.data["depth_source"], "measured")
        self.assertTrue(cap.data["intrinsics_calibrated"])
        self.assertFalse(cap.data["extrinsics_verified"])
        self.assertFalse(cap.data["motion_safe"])

    def test_fully_verified_inputs_clear_the_grasp(self):
        depth_file, intrinsics = self._measured_inputs()
        cfg = self._verified_config()
        cap = self._capture(depth=depth_file, intrinsics=intrinsics, config_path=cfg)
        self.assertTrue(cap.data["motion_safe"])
        # instrument at the image centre = right under the camera, above the middle of tray_2 (reachable)
        real_segment.run(out_dir=str(self.out), segmenter=FakeSegmenter(0.9, box=(525, 555, 860, 1060)))
        dep = real_deproject.run(out_dir=str(self.out), config_path=cfg)
        self.assertIs(dep.data["geometry_verified_for_motion"], True)
        self.assertIs(dep.data["motion_safe"], True)
        rep = grasp.run(deprojection_path=str(self.out / paths.REAL_DEPROJECTION), out_dir=str(self.out),
                        config_path=cfg)
        self.assertEqual(rep.data["readiness_blockers"], [])
        self.assertEqual(rep.data["readiness"], grasp.READY)

    def test_unverified_inputs_never_set_the_verified_flag(self):
        self._capture()
        real_segment.run(out_dir=str(self.out), segmenter=FakeSegmenter(0.9))
        dep = real_deproject.run(out_dir=str(self.out))
        self.assertIs(dep.data["geometry_verified_for_motion"], False)

    def test_misaligned_depth_is_rejected(self):
        np.save(self.tmp / "bad.npy", np.zeros((10, 10), np.float32))
        with self.assertRaises(StageError):
            self._capture(depth=str(self.tmp / "bad.npy"))


class GeometryGateTests(unittest.TestCase):
    def test_workspace_annulus(self):
        ws = Workspace()
        self.assertTrue(ws.contains([0.60, -0.42, 0.09]))
        self.assertFalse(ws.contains([0.80, -0.40, 0.09]))       # too far
        self.assertFalse(ws.contains([0.20, 0.0, 0.09]))         # too close to the base
        self.assertTrue(ws.contains([0.20, 0.0, 0.30]))          # high above the table the minimum does not apply
        self.assertTrue(ws.blockers([0.9, 0.0, 0.09]))

    def test_placement_follows_a_moved_robot_base(self):
        base = (0.30, -0.30)                       # base shifted in world: the annulus moves with it
        for seed in range(20):
            x, y, _ = sample_tool_pose([0.60, -0.42], [0.62, 0.68], (0.14, 0.02), seed, base_xy=base)
            self.assertTrue(Workspace().contains([x - base[0], y - base[1], 0.09]))

    def test_random_placement_stays_reachable_and_inside_tray(self):
        for seed in range(20):
            x, y, yaw = sample_tool_pose([0.60, -0.42], [0.62, 0.68], (0.14, 0.02), seed)
            self.assertTrue(Workspace().contains([x, y, 0.09]))
            self.assertLess(abs(x - 0.60), 0.31)
            self.assertLess(abs(y + 0.42), 0.34)

    def test_outlier_pixels_are_removed(self):
        z = np.concatenate([np.full(200, 1.00), [1.5, 0.6]])
        pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
        pix = np.stack([np.arange(len(z)), np.arange(len(z))], axis=1)
        cam, world, kept = reject_depth_outliers(pts, pts.copy(), pix)
        self.assertEqual(len(cam), 200)


if __name__ == "__main__":
    unittest.main()
