"""Unit tests for the shared pipeline layer (reports, paths, PLY, planner parsing, runner)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline import plan
from pipeline.common import paths
from pipeline.common.ply import read_ply, write_ply
from pipeline.common.report import SCHEMA_VERSION, StageError, StageReport, stage_run
from pipeline.common.scene import ConfigError, camera_pose, get_tray
from pipeline.config_loader import load_environment_config
from pipeline.orchestrate import Step, execute


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_roundtrip_with_numpy_values(self):
        with stage_run("demo", self.tmp / "demo.json", inputs={"a": self.tmp / "x.json", "skip": None}) as rep:
            rep.data = {"v": np.arange(3), "f": np.float32(1.5)}
            rep.artifacts = {"png": str(self.tmp / "a.png")}
        loaded = StageReport.load(self.tmp / "demo.json", expect_stage="demo")
        self.assertEqual(loaded.data, {"v": [0, 1, 2], "f": 1.5})
        self.assertEqual(loaded.schema_version, SCHEMA_VERSION)
        self.assertNotIn("skip", loaded.inputs)
        self.assertIsNotNone(loaded.duration_s)

    def test_failure_leaves_failed_report_and_wraps_error(self):
        path = self.tmp / "bad.json"
        with self.assertRaises(StageError) as ctx:
            with stage_run("bad", path):
                raise ValueError("boom")
        self.assertEqual(ctx.exception.stage, "bad")
        raw = json.loads(path.read_text())
        self.assertEqual(raw["status"], "FAILED")
        self.assertEqual(raw["error"]["type"], "ValueError")
        self.assertIn("boom", raw["error"]["traceback"])
        with self.assertRaises(StageError):      # downstream refuses a failed upstream
            StageReport.load(path)

    def test_load_diagnostics(self):
        with self.assertRaisesRegex(StageError, "not found"):
            StageReport.load(self.tmp / "missing.json", expect_stage="plan")
        with stage_run("other", self.tmp / "o.json"):
            pass
        with self.assertRaisesRegex(StageError, "expected 'plan'"):
            StageReport.load(self.tmp / "o.json", expect_stage="plan")
        legacy = self.tmp / "legacy.json"
        legacy.write_text(json.dumps({"stage": "plan", "status": "COMPLETED"}))
        with self.assertRaisesRegex(StageError, "schema_version"):
            StageReport.load(legacy)

    def test_require_and_artifact_errors(self):
        rep = StageReport("demo", data={"a": {"b": 1}}, artifacts={"gone": str(self.tmp / "nope.png")})
        self.assertEqual(rep.require("a", "b"), 1)
        with self.assertRaisesRegex(StageError, "data.a.c"):
            rep.require("a", "c")
        with self.assertRaisesRegex(StageError, "missing on disk"):
            rep.artifact("gone")


class PathsAndSceneTests(unittest.TestCase):
    def test_env_overrides(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {paths.ROOT_ENV: tmp}):
            self.assertEqual(paths.output_dir(), Path(tmp).resolve() / "data" / "outputs")
            self.assertEqual(paths.resolve(None, paths.PLAN), Path(tmp).resolve() / "data/outputs/plan.json")
            self.assertEqual(paths.resolve("x.json", paths.PLAN), Path("x.json"))

    def test_tray_and_camera(self):
        config = load_environment_config()
        tray = get_tray(config, "tray_2")
        self.assertAlmostEqual(tray.top_z, tray.position[2] + tray.dimensions[2] / 2)
        self.assertTrue(tray.contains_xy(tray.position))
        self.assertFalse(tray.contains_xy([tray.position[0] + 5, tray.position[1]]))
        pose = camera_pose(config, tray)
        self.assertEqual(pose.target, tray.position)
        with self.assertRaises(ConfigError):
            get_tray(config, "tray_9")

    def test_explicit_missing_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_environment_config("/nonexistent/config.yaml")


class PlyTests(unittest.TestCase):
    def test_roundtrip_with_and_without_colors(self):
        pts = np.random.default_rng(0).uniform(-1, 1, (20, 3))
        cols = np.random.default_rng(1).integers(0, 255, (20, 3))
        with tempfile.TemporaryDirectory() as tmp:
            write_ply(Path(tmp) / "c.ply", pts, cols)
            p2, c2 = read_ply(Path(tmp) / "c.ply")
            np.testing.assert_allclose(p2, pts, atol=1e-6)
            np.testing.assert_array_equal(c2, cols)
            write_ply(Path(tmp) / "n.ply", pts)
            p3, c3 = read_ply(Path(tmp) / "n.ply")
            self.assertIsNone(c3)
            self.assertEqual(len(p3), 20)
            with self.assertRaises(ValueError):
                write_ply(Path(tmp) / "bad.ply", pts, cols[:5])


class PlannerTests(unittest.TestCase):
    def test_keyword_and_normalization(self):
        self.assertEqual(plan.extract_keyword("hemostatic Forceps"), "forceps")
        self.assertEqual(plan.extract_keyword("Bisturi"), "scalpel")
        self.assertEqual(plan.extract_keyword("the weird tool"), "weird")
        config = load_environment_config()
        self.assertEqual(plan.normalize_tray("vassoio_1", "tray_2", config), "tray_1")
        self.assertEqual(plan.normalize_tray("left tray", "tray_1", config), "tray_2")
        self.assertEqual(plan.normalize_tray(None, "tray_1"), "tray_1")
        self.assertEqual(plan.normalize_action("passa"), "pick_and_place")
        self.assertEqual(plan.normalize_action("inspect it"), "inspect")

    def test_llm_output_cleanup(self):
        raw = '<think>hm</think>\n```json\n{"action": "pick"}\n```'
        self.assertEqual(plan.parse_llm_json(raw), {"action": "pick"})

    def test_offline_stage_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = plan.run("Passami il bisturi dal vassoio 2", offline=True, out_dir=tmp)
            self.assertEqual(rep.data["target_keyword"], "scalpel")
            self.assertEqual(rep.data["source_tray"], "tray_2")
            self.assertEqual(rep.data["planner_backend"], "deterministic_parser")
            self.assertTrue((Path(tmp) / "plan.json").exists())

    def test_cli_returns_nonzero_on_stage_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(plan.main(["--offline", "--out-dir", tmp]), 0)
            self.assertEqual(plan.main(["--offline", "--out-dir", tmp, "--config", "/no/such.yaml"]), 1)
            raw = json.loads((Path(tmp) / "plan.json").read_text())
            self.assertEqual(raw["status"], "FAILED")


class OrchestrateTests(unittest.TestCase):
    def test_resume_and_stop(self):
        calls = []
        steps = [Step(n, lambda n=n: calls.append(n) or StageReport(n)) for n in ("a", "b", "c")]
        execute(steps, start_at="b", stop_after="b")
        self.assertEqual(calls, ["b"])
        with self.assertRaises(StageError):
            execute(steps, start_at="zzz")
        with self.assertRaises(StageError):
            execute(steps, start_at="c", stop_after="a")


if __name__ == "__main__":
    unittest.main()
