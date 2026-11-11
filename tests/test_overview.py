"""Mask + depth of EVERY instrument by default: helpers, stage prefixes, mode choice and the offscreen overview."""

import argparse
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
import yaml
from PIL import Image

from pipeline import plan
from pipeline.common import paths
from pipeline.common.report import StageReport
from pipeline.common.vision import Instances, all_instruments_from_mask, segment_all_instruments
from pipeline.sim import capture, overview, pipeline, segment


def _instance(h, w, box):
    mask = np.zeros((h, w), dtype=np.float32)
    y0, y1, x0, x1 = box
    mask[y0:y1, x0:x1] = 1.0
    return mask


class PromptAwareSegmenter:
    """Stands in for SAM 3: one instance for the target prompt, three for the generic one."""

    def __init__(self, shape):
        self.h, self.w = shape
        self.prompts = []

    def segment(self, image, prompt):
        self.prompts.append(prompt)
        h, w = self.h, self.w
        if prompt == "surgical instrument":
            boxes = [(10, 20, 10, 50), (30, 40, 10, 50), (50, 60, 10, 50), (70, 80, 10, 50)]
            return Instances(np.stack([_instance(h, w, b) for b in boxes]), np.array([0.9, 0.6, 0.4, 0.1]))
        return Instances(_instance(h, w, (30, 40, 10, 50))[None], np.array([0.8]))


class AllInstrumentsHelperTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.full((100, 120, 3), 200, np.uint8))

    def test_keeps_instances_above_the_score_and_labels_them_best_first(self):
        seg = PromptAwareSegmenter((100, 120))
        every = segment_all_instruments(self.image, seg)
        self.assertEqual(every.data["num_instances"], 3)            # 0.9, 0.6, 0.4 pass 0.30; 0.1 does not
        self.assertEqual(sorted(np.unique(every.labels)), [0, 1, 2, 3])
        self.assertEqual(int(every.labels[15, 20]), 1)              # best instance is label 1
        self.assertEqual(int((every.mask > 0).sum()), 3 * 10 * 40)  # union of the three boxes
        self.assertEqual(every.data["prompt"], "surgical instrument")
        self.assertFalse(every.warnings)

    def test_overlapping_instances_are_assigned_to_the_best_score(self):
        class Overlap:
            def segment(self, image, prompt):
                a, b = _instance(100, 120, (10, 50, 10, 50)), _instance(100, 120, (30, 70, 10, 50))
                return Instances(np.stack([b, a]), np.array([0.5, 0.9]))
        every = segment_all_instruments(self.image, Overlap())
        self.assertEqual(int(every.labels[40, 20]), 1)              # overlap row belongs to the 0.9 instance
        self.assertEqual(int(every.labels[60, 20]), 2)
        self.assertEqual(int((every.labels == 1).sum()), 40 * 40)

    def test_target_is_highlighted_and_empty_result_only_warns(self):
        seg = PromptAwareSegmenter((100, 120))
        target = _instance(100, 120, (30, 40, 10, 50)) > 0
        every = segment_all_instruments(self.image, seg, target=target)
        self.assertTrue((every.overlay[35, 20] != every.overlay[15, 20]).any())
        class Empty:
            def segment(self, image, prompt):
                return Instances(np.zeros((0, 100, 120), np.float32), np.zeros(0))
        nothing = segment_all_instruments(self.image, Empty())
        self.assertEqual(nothing.data["num_instances"], 0)
        self.assertTrue(nothing.warnings)

    def test_ground_truth_variant_is_one_body(self):
        rgb = np.full((100, 120, 3), 200, np.uint8)
        truth = _instance(100, 120, (10, 60, 10, 90)) > 0
        every = all_instruments_from_mask(rgb, truth)
        self.assertEqual(every.data["num_instances"], 1)
        self.assertEqual(every.data["segmentation_backend"], "simulator_object_id")
        self.assertEqual(int((every.mask > 0).sum()), int(truth.sum()))


class ModeChoiceTests(unittest.TestCase):
    def args(self, **kw):
        base = dict(mode=None, gui=False, headless=False, vision="sam3")
        base.update(kw)
        return argparse.Namespace(**base)

    def test_default_is_the_full_collection(self):
        self.assertEqual(pipeline.resolve_mode(self.args()), "all")
        self.assertEqual(pipeline.resolve_mode(self.args(gui=True)), "all")
        self.assertEqual(pipeline.resolve_mode(self.args(headless=True, vision="ground-truth")), "all")

    def test_ground_truth_without_robot_needs_one_body_per_instrument(self):
        self.assertEqual(pipeline.resolve_mode(self.args(vision="ground-truth")), "single")

    def test_explicit_mode_always_wins(self):
        self.assertEqual(pipeline.resolve_mode(self.args(mode="single", gui=True)), "single")
        self.assertEqual(pipeline.resolve_mode(self.args(mode="all", vision="ground-truth")), "all")

    def test_robot_run_no_longer_requires_single(self):
        parser = pipeline.build_parser()
        args = parser.parse_args(["--gui", "--mode", "all", "--tool", "Scalpel"])
        self.assertEqual(pipeline.resolve_mode(args), "all")


class OverviewStageTests(unittest.TestCase):
    """Synthetic collection (3 flat boxes in centimetres, like All_Instruments.obj) in a throw-away project root."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict("os.environ", {paths.ROOT_ENV: str(cls.tmp)})
        cls.env.start()
        (cls.tmp / "data" / "inputs").mkdir(parents=True)
        # authored in cm, Y is the height before the lay-flat rotation: boxes sit side by side along X / Z
        parts = []
        for dx, dz in ((-12.0, -8.0), (0.0, 0.0), (12.0, 8.0)):
            box = trimesh.creation.box([10.0, 0.4, 1.2])
            box.apply_translation([dx, 0.2, dz])
            parts.append(box)
        trimesh.util.concatenate(parts).export(cls.tmp / "data" / "inputs" / "All_Instruments_combined.obj")
        stl = cls.tmp / "tool.stl"
        trimesh.creation.box([0.14, 0.012, 0.004]).export(stl)
        cfg = yaml.safe_load(Path(paths.__file__).resolve().parents[2].joinpath(
            "config/surgical_environment.yaml").read_text())
        cfg["surgical_instrument"]["stl_path"] = str(stl)
        cls.config = str(cls.tmp / "env.yaml")
        Path(cls.config).write_text(yaml.safe_dump(cfg))
        cls.out = paths.output_dir()
        plan.run("Passami il bisturi dal vassoio 2 al vassoio 1", offline=True, config_path=cls.config,
                 out_dir=str(cls.out))

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_capture_prefix_never_overwrites_the_default_artifacts(self):
        sentinel = self.out / "sim_capture_rgb.png"
        sentinel.write_bytes(b"keep me")
        cap = capture.run(mode="all", config_path=self.config, out_dir=str(self.out),
                          report_path="other_capture.json", prefix="other")
        self.assertEqual(sentinel.read_bytes(), b"keep me")
        self.assertTrue(Path(cap.artifacts["rgb_png"]).name == "other_rgb.png")
        for key in ("rgb_png", "depth_npy", "depth_visual_png", "truth_mask_png"):
            self.assertTrue(Path(cap.artifacts[key]).exists(), key)
        sentinel.unlink()

    def test_segment_masks_every_instrument_by_default(self):
        cap = capture.run(mode="all", config_path=self.config, out_dir=str(self.out),
                          report_path="seg_capture.json", prefix="seg")
        fake = PromptAwareSegmenter(tuple(np.load(cap.artifacts["depth_npy"]).shape))
        rep = segment.run(backend="sam3", capture_path=str(self.out / "seg_capture.json"), out_dir=str(self.out),
                          report_path="seg_report.json", prefix="seg_out", segmenter=fake)
        self.assertEqual(fake.prompts, ["scalpel", "surgical instrument"])
        self.assertEqual(rep.data["all_instruments"]["num_instances"], 3)
        for key in ("mask_png", "overlay_png", "all_mask_png", "all_overlay_png", "all_instances_png"):
            self.assertTrue(Path(rep.artifacts[key]).exists(), key)
        labels = np.asarray(Image.open(rep.artifacts["all_instances_png"]))
        self.assertEqual(labels.dtype, np.uint16)
        self.assertEqual(int(labels.max()), 3)

    def test_segment_can_skip_the_extra_inference(self):
        cap = capture.run(mode="all", config_path=self.config, out_dir=str(self.out),
                          report_path="skip_capture.json", prefix="skip")
        fake = PromptAwareSegmenter(tuple(np.load(cap.artifacts["depth_npy"]).shape))
        rep = segment.run(backend="sam3", capture_path=str(self.out / "skip_capture.json"), out_dir=str(self.out),
                          report_path="skip_report.json", prefix="skip_out", segmenter=fake, all_instruments=False)
        self.assertEqual(fake.prompts, ["scalpel"])
        self.assertNotIn("all_instruments", rep.data)
        self.assertNotIn("all_mask_png", rep.artifacts)

    def test_ground_truth_in_all_mode_warns_that_it_cannot_isolate_the_target(self):
        capture.run(mode="all", config_path=self.config, out_dir=str(self.out),
                    report_path="gt_capture.json", prefix="gt")
        rep = segment.run(backend="ground-truth", capture_path=str(self.out / "gt_capture.json"),
                          out_dir=str(self.out), report_path="gt_report.json", prefix="gt_out")
        self.assertTrue(any("WHOLE collection" in w for w in rep.warnings))
        self.assertEqual(rep.data["all_instruments"]["num_instances"], 1)

    def test_overview_stage_keeps_the_grasp_artifacts_untouched(self):
        sentinel = self.out / "sim_segmentation_mask.png"
        sentinel.write_bytes(b"target mask for the grasp")
        rep = overview.run(backend="ground-truth", config_path=self.config, out_dir=str(self.out))
        self.assertEqual(rep.status, "COMPLETED")
        self.assertEqual(sentinel.read_bytes(), b"target mask for the grasp")
        for key in ("rgb_png", "depth_npy", "depth_visual_png", "all_mask_png", "all_overlay_png",
                    "all_instances_png", "target_mask_png", "target_overlay_png"):
            self.assertTrue(Path(rep.artifacts[key]).exists(), key)
        saved = StageReport.load(self.out / paths.SIM_OVERVIEW, expect_stage="sim_overview")
        self.assertEqual(saved.data["mode"], "all")
        self.assertGreater(saved.data["all_instruments"]["active_pixel_count"], 15)
        self.assertIn("STL scene", saved.data["note"])
        depth = np.load(rep.artifacts["depth_npy"])
        mask = np.asarray(Image.open(rep.artifacts["all_mask_png"])) > 128
        self.assertTrue(np.isfinite(depth[mask]).all())
        sentinel.unlink()

    def test_overview_with_sam3_tolerates_a_missing_target(self):
        class NoTarget(PromptAwareSegmenter):
            def segment(self, image, prompt):
                if prompt != "surgical instrument":
                    return Instances(np.zeros((1, self.h, self.w), np.float32), np.array([0.01]))
                return super().segment(image, prompt)
        shape = (480, 640)
        rep = overview.run(backend="sam3", config_path=self.config, out_dir=str(self.out),
                           segmenter=NoTarget(shape))
        self.assertEqual(rep.status, "COMPLETED")                    # informational: a weak target never fails it
        self.assertEqual(rep.data["all_instruments"]["num_instances"], 3)
        self.assertTrue(any("confidence" in w for w in rep.warnings))


class GraphicalOverviewTests(unittest.TestCase):
    """Headless robot run (worker process included): the overview is produced after the grasp and awaited."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = patch.dict("os.environ", {paths.ROOT_ENV: str(self.tmp)})
        self.env.start()
        (self.tmp / "data" / "inputs").mkdir(parents=True)
        parts = []
        for dx, dz in ((-12.0, -8.0), (0.0, 0.0), (12.0, 8.0)):
            box = trimesh.creation.box([10.0, 0.4, 1.2])
            box.apply_translation([dx, 0.2, dz])
            parts.append(box)
        trimesh.util.concatenate(parts).export(self.tmp / "data" / "inputs" / "All_Instruments_combined.obj")
        stl = self.tmp / "tool.stl"
        trimesh.creation.box([0.14, 0.012, 0.004]).export(stl)
        cfg = yaml.safe_load(Path(paths.__file__).resolve().parents[2].joinpath(
            "config/surgical_environment.yaml").read_text())
        cfg["surgical_instrument"]["stl_path"] = str(stl)
        self.config = str(self.tmp / "env.yaml")
        Path(self.config).write_text(yaml.safe_dump(cfg))
        self.out = paths.output_dir()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, **overrides):
        from pipeline.sim.graphical import run_graphical_pipeline
        args = argparse.Namespace(out_dir=None, vision="ground-truth", headless=True, gui=False, config=self.config,
                                  tool="Scalpel", seed=42, no_motion=True, exit_after_run=False,
                                  grasp_backend="pca", graspgen_config=None, graspgen_host=None,
                                  graspgen_port=None, graspgen_timeout_ms=None, mode="all")
        for key, value in overrides.items():
            setattr(args, key, value)
        return run_graphical_pipeline(args, "Passami lo strumento Scalpel dal vassoio 2 al vassoio 1")

    def test_default_run_also_produces_the_full_collection_views(self):
        self.assertEqual(self._run(), 0)
        manifest = StageReport.load(self.out / "graphical_run.json", require_completed=False)
        info = manifest.data["overview"]
        self.assertEqual(info["status"], "COMPLETED")
        self.assertTrue(info["enabled"])
        self.assertGreaterEqual(info["num_instruments"], 1)
        for key in ("rgb_png", "depth_visual_png", "all_mask_png", "all_overlay_png"):
            self.assertTrue(Path(info["artifacts"][key]).exists(), key)
        # the grasp was built from the STL scene, not from the collection
        grasp = StageReport.load(self.out / paths.GRASP)
        capture_stl = StageReport.load(self.out / paths.SIM_CAPTURE)
        self.assertEqual(capture_stl.data["mode"], "single")
        self.assertEqual(grasp.status, "COMPLETED")

    def test_single_mode_keeps_the_old_behaviour(self):
        self.assertEqual(self._run(mode="single"), 0)
        manifest = StageReport.load(self.out / "graphical_run.json", require_completed=False)
        self.assertEqual(manifest.data["overview"], {"enabled": False, "status": "DISABLED"})
        self.assertFalse((self.out / paths.SIM_OVERVIEW).exists())

    def test_overview_failure_never_invalidates_the_grasp(self):
        (self.tmp / "data" / "inputs" / "All_Instruments_combined.obj").unlink()
        self.assertEqual(self._run(), 0)
        manifest = StageReport.load(self.out / "graphical_run.json", require_completed=False)
        self.assertEqual(manifest.data["overview"]["status"], "FAILED")
        self.assertIn("All_Instruments", manifest.data["overview"]["error"])
        self.assertEqual(StageReport.load(self.out / paths.GRASP).status, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
