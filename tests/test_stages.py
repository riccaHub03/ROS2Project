"""End-to-end run of the standalone stages on a synthetic STL (SAM 3 replaced by a depth-based fake)."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
import yaml

from PIL import Image

from pipeline import grasp, plan
from pipeline.common import paths
from pipeline.common.report import StageError, StageReport
from pipeline.common.vision import Instances
from pipeline.real import capture as real_capture
from pipeline.real import deproject as real_deproject
from pipeline.real import segment as real_segment
from pipeline.sim import capture, deproject, reference_pose, segment


class DepthBasedSegmenter:
    """Stands in for SAM 3: the tool is whatever sticks out above the tray plane in the depth map."""

    def __init__(self, depth: np.ndarray, tray_plane_depth: float):
        self.depth, self.plane = depth, tray_plane_depth

    def segment(self, image, prompt):
        mask = self.depth < self.plane - 0.003
        return Instances(mask[None].astype(np.float32), np.array([0.9]))


class StageChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict("os.environ", {paths.ROOT_ENV: str(cls.tmp)})
        cls.env.start()
        mesh = cls.tmp / "tool.stl"
        trimesh.creation.box([0.14, 0.012, 0.004]).export(mesh)
        cfg = yaml.safe_load(Path(paths.__file__).resolve().parents[2].joinpath("config/surgical_environment.yaml").read_text())
        cfg["surgical_instrument"]["stl_path"] = str(mesh)
        cls.config = str(cls.tmp / "env.yaml")
        Path(cls.config).write_text(yaml.safe_dump(cfg))
        cls.out = paths.output_dir()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_full_chain_through_reports_only(self):
        kw = dict(out_dir=str(self.out))
        plan.run("Passami il bisturi dal vassoio 2", offline=True, config_path=self.config, **kw)
        cap = capture.run(mode="single", config_path=self.config, **kw)
        self.assertEqual(cap.status, "COMPLETED")

        depth = np.load(cap.artifact("depth_npy"))
        camera = cap.data["camera"]["position_xyz_m"]
        tray_top = 0.07 + 0.04 / 2          # tray_2 centre z + half height (config defaults)
        seg = segment.run(segmenter=DepthBasedSegmenter(depth, camera[2] - tray_top), **kw)
        self.assertGreater(seg.data["active_pixel_count"], 15)

        reference_pose.run(seed=42, config_path=self.config, **kw)
        dep = deproject.run(reference_pose_path=str(self.out / paths.SIM_REFERENCE_POSE), config_path=self.config, **kw)
        self.assertEqual(dep.data["centroid_source"], "optical")
        self.assertLess(dep.data["ground_truth_validation"]["error_mm"], 20.0)

        g = grasp.run(**kw)
        geometry = g.data["designated_grasp_geometry"]
        self.assertTrue(geometry["gripper_compatibility_check"]["parallel_jaw_suitable"])
        self.assertEqual(len(geometry["recommended_gripper_orientation"]["quaternion_xyzw"]), 4)
        # every stage records the upstream reports it consumed
        self.assertIn("segmentation", dep.inputs)
        self.assertIn("deprojection", g.inputs)

    def test_stage_cli_reports_missing_upstream(self):
        empty = self.tmp / "empty_out"
        rc = segment.main(["--out-dir", str(empty)])
        self.assertEqual(rc, 1)
        failed = StageReport.load(empty / paths.SIM_SEGMENTATION, require_completed=False)
        self.assertEqual(failed.status, "FAILED")
        self.assertIn("plan", failed.error["message"])

    def test_segment_rejects_unknown_backend(self):
        with self.assertRaises(StageError):
            segment.run(backend="nope", out_dir=str(self.out))


class RectangleSegmenter:
    """Returns the target (elongated rectangle, 0.8) plus a weaker distractor, regardless of the prompt."""

    def segment(self, image, prompt):
        h, w = image.size[1], image.size[0]
        target, other = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        target[200:220, 200:420] = 1
        other[300:320, 100:200] = 1
        return Instances(np.stack([other, target]), np.array([0.3, 0.8]))


class RealChainTests(unittest.TestCase):
    def test_real_chain_from_photo_to_grasp(self):
        with tempfile.TemporaryDirectory() as tmp:
            kw = dict(out_dir=tmp)
            photo = Path(tmp) / "tray.jpg"
            Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(photo)
            plan.run("Passami le forbici dal vassoio 2", offline=True, **kw)
            cap = real_capture.run(image=str(photo), **kw)
            self.assertEqual(cap.data["camera"]["resolution_wh"], [640, 480])
            seg = real_segment.run(segmenter=RectangleSegmenter(), **kw)
            self.assertEqual(seg.data["active_pixel_count"], 20 * 220)
            self.assertEqual(seg.data["num_instances"], 2)
            dep = real_deproject.run(**kw)
            self.assertEqual(dep.data["num_points"], 20 * 220)
            g = grasp.run(deprojection_path=str(Path(tmp) / paths.REAL_DEPROJECTION), **kw)
            self.assertEqual(g.data["source_pipeline"], "real")
            self.assertTrue(g.data["designated_grasp_geometry"]["gripper_compatibility_check"]["parallel_jaw_suitable"])

    def test_real_segmentation_fails_when_nothing_detected(self):
        class Nothing:
            def segment(self, image, prompt):
                return Instances(np.zeros((0, 480, 640), np.float32), np.zeros(0))
        with tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp) / "tray.png"
            Image.new("RGB", (640, 480)).save(photo)
            plan.run("forbici", offline=True, out_dir=tmp)
            real_capture.run(image=str(photo), out_dir=tmp)
            with self.assertRaisesRegex(StageError, "no instrument detected"):
                real_segment.run(segmenter=Nothing(), out_dir=tmp)
            self.assertEqual(StageReport.load(Path(tmp) / paths.REAL_SEGMENTATION, require_completed=False).status, "FAILED")


if __name__ == "__main__":
    unittest.main()
