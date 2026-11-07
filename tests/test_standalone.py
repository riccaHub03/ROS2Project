"""Each stage must run alone, from a fresh output folder, given only the previous stage's report(s).

A full chain is produced once in folder A; then every stage is launched through its real CLI
(``main(argv)``) in a brand-new folder B, receiving nothing but explicit paths to the upstream
JSON reports. All files the stage writes (report + artifacts) must land in B.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
import yaml
from PIL import Image

from pipeline import clean, grasp, plan
from pipeline.common import paths
from pipeline.common.report import StageReport
from pipeline.real import capture as real_capture
from pipeline.real import deproject as real_deproject
from pipeline.real import segment as real_segment
from pipeline.sim import capture, deproject, reference_pose, segment
from tests.test_stages import RectangleSegmenter


class StandaloneStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict("os.environ", {paths.ROOT_ENV: str(cls.tmp)})
        cls.env.start()
        mesh = cls.tmp / "tool.stl"
        trimesh.creation.box([0.14, 0.012, 0.004]).export(mesh)
        cfg = yaml.safe_load((Path(paths.__file__).resolve().parents[2] / "config/surgical_environment.yaml").read_text())
        cfg["surgical_instrument"]["stl_path"] = str(mesh)
        cls.config = cls.tmp / "env.yaml"
        cls.config.write_text(yaml.safe_dump(cfg))

        # Folder A: the reference chain (sim, ground-truth mask => no SAM 3 needed).
        cls.a = cls.tmp / "A"
        kw = dict(out_dir=str(cls.a))
        plan.run("Passami il bisturi dal vassoio 2", offline=True, config_path=str(cls.config), **kw)
        capture.run(mode="single", config_path=str(cls.config), **kw)
        segment.run(backend="ground-truth", **kw)
        reference_pose.run(config_path=str(cls.config), **kw)
        deproject.run(reference_pose_path=str(cls.a / paths.SIM_REFERENCE_POSE), config_path=str(cls.config), **kw)
        grasp.run(**kw)
        # Real chain in the same folder A.
        photo = cls.tmp / "tray.png"
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)).save(photo)
        cls.photo = photo
        real_capture.run(image=str(photo), **kw)
        real_segment.run(segmenter=RectangleSegmenter(), **kw)
        real_deproject.run(**kw)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def a_(self, name):
        return str(self.a / name)

    def run_in_fresh_folder(self, module, name, argv, report_name):
        b = Path(tempfile.mkdtemp(dir=self.tmp))
        rc = module.main([*argv, "--out-dir", str(b)])
        self.assertEqual(rc, 0, f"{name} failed standalone")
        report = StageReport.load(b / report_name)
        written = {p.name for p in b.iterdir()}
        self.assertIn(report_name, written)
        for artifact in report.artifacts.values():
            self.assertEqual(Path(artifact).parent, b, f"{name}: artifact outside the output folder: {artifact}")
        return report, b

    def test_plan_alone(self):
        report, _ = self.run_in_fresh_folder(plan, "plan", ["--offline", "--instruction", "pinza dal vassoio 2"], paths.PLAN)
        self.assertEqual(report.data["target_keyword"], "forceps")

    def test_sim_capture_needs_only_plan(self):
        report, _ = self.run_in_fresh_folder(capture, "sim.capture", ["--plan", self.a_(paths.PLAN), "--mode", "single",
                                             "--config", str(self.config)], paths.SIM_CAPTURE)
        self.assertEqual(report.inputs["plan"], self.a_(paths.PLAN))

    def test_sim_segment_needs_plan_and_capture(self):
        self.run_in_fresh_folder(segment, "sim.segment", ["--backend", "ground-truth", "--plan", self.a_(paths.PLAN),
                                 "--capture", self.a_(paths.SIM_CAPTURE)], paths.SIM_SEGMENTATION)

    def test_sim_reference_pose_needs_only_plan(self):
        self.run_in_fresh_folder(reference_pose, "sim.reference_pose", ["--plan", self.a_(paths.PLAN),
                                 "--config", str(self.config)], paths.SIM_REFERENCE_POSE)

    def test_sim_deproject_needs_segmentation_and_capture(self):
        report, _ = self.run_in_fresh_folder(deproject, "sim.deproject", [
            "--segmentation", self.a_(paths.SIM_SEGMENTATION), "--capture", self.a_(paths.SIM_CAPTURE),
            "--reference-pose", self.a_(paths.SIM_REFERENCE_POSE), "--config", str(self.config)], paths.SIM_DEPROJECTION)
        self.assertLess(report.data["ground_truth_validation"]["error_mm"], 20.0)

    def test_grasp_needs_only_deprojection(self):
        for source in (paths.SIM_DEPROJECTION, paths.REAL_DEPROJECTION):
            report, _ = self.run_in_fresh_folder(grasp, "grasp", ["--deprojection", self.a_(source)], paths.GRASP)
            self.assertEqual(report.data["source_pipeline"], source.split("_")[0])
            self.assertTrue(any("plan report not found" in w for w in report.warnings))

    def test_real_capture_needs_only_image(self):
        self.run_in_fresh_folder(real_capture, "real.capture", ["--image", str(self.photo)], paths.REAL_CAPTURE)

    def test_real_deproject_needs_segmentation_and_capture(self):
        self.run_in_fresh_folder(real_deproject, "real.deproject", [
            "--segmentation", self.a_(paths.REAL_SEGMENTATION), "--capture", self.a_(paths.REAL_CAPTURE)],
            paths.REAL_DEPROJECTION)

    def test_wrong_or_failed_upstream_is_rejected(self):
        b = Path(tempfile.mkdtemp(dir=self.tmp))
        # a deprojection report given where a plan is expected
        rc = segment.main(["--plan", self.a_(paths.SIM_DEPROJECTION), "--capture", self.a_(paths.SIM_CAPTURE),
                           "--backend", "ground-truth", "--out-dir", str(b)])
        self.assertEqual(rc, 1)
        failed = StageReport.load(b / paths.SIM_SEGMENTATION, require_completed=False)
        self.assertEqual(failed.status, "FAILED")
        self.assertIn("expected 'plan'", failed.error["message"])

    def test_report_name_is_created_inside_out_dir(self):
        b = Path(tempfile.mkdtemp(dir=self.tmp))
        self.assertEqual(plan.main(["--offline", "--out-dir", str(b), "--report", "my_plan.json"]), 0)
        self.assertTrue((b / "my_plan.json").exists())

    def test_clean_dry_run_then_delete(self):
        b = Path(tempfile.mkdtemp(dir=self.tmp))
        (b / ".gitkeep").write_text("")
        plan.main(["--offline", "--out-dir", str(b)])
        clean.main(["--out-dir", str(b)])
        self.assertTrue((b / paths.PLAN).exists())          # dry run keeps everything
        clean.main(["--out-dir", str(b), "--yes"])
        self.assertEqual([p.name for p in b.iterdir()], [".gitkeep"])


if __name__ == "__main__":
    unittest.main()
