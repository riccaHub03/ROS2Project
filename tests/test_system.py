"""Contract and orchestration regressions; no model downloads, inference or robot motion."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.common.report import StageError, StageReport
from pipeline.config_loader import load_environment_config
from pipeline.system.contracts import validate_task, validate_segmentation, validate_reconstruction
from pipeline.system.registry import load_profile
from pipeline.system.orchestrator import finalize, run
from pipeline import plan


class TaskContracts(unittest.TestCase):
    def setUp(self):
        self.task = dict(action="pick_and_place", target_instrument="Bone Forceps", target_keyword="forceps",
                         source_tray="tray_2", destination_tray="tray_1")
        self.config = load_environment_config()

    def test_rejects_semantically_invalid_llm_tasks(self):
        for change in (dict(action="inspect"), dict(source_tray="tray_99"),
                       dict(source_tray="tray_1"), dict(target_instrument="Scissors"),
                       dict(target_keyword="scissors"), dict(target_keyword=None)):
            with self.subTest(change=change), self.assertRaises(StageError):
                validate_task({**self.task, **change}, self.config, "Bone Forceps")
        validate_task(self.task, self.config, "Bone Forceps")

    def test_full_profile_cannot_fall_back_to_parser(self):
        with tempfile.TemporaryDirectory() as out, patch.object(plan, "query_ollama", return_value=(None, None)):
            with self.assertRaisesRegex(StageError, "fallback disabled"):
                plan.run("Move the forceps", require_llm=True, out_dir=out)
            self.assertEqual(json.loads((Path(out) / "plan.json").read_text())["status"], "FAILED")

    def test_llm_response_is_validated_and_recorded_without_tool_override(self):
        with tempfile.TemporaryDirectory() as out, patch.object(plan, "query_ollama", return_value=(self.task, "http://local")):
            result = plan.run("Move the Bone Forceps", require_llm=True, expected_instrument="Bone Forceps", out_dir=out)
            self.assertEqual(result.data["planner_backend"], "ollama")
            self.assertEqual(result.data["target_instrument"], "Bone Forceps")
            self.assertEqual(result.data["planner_model"], "qwen2.5:3b")
        with tempfile.TemporaryDirectory() as out, self.assertRaises(StageError):
            plan.run(require_llm=True, tool_hint="Bone Forceps", out_dir=out)

    def test_legacy_offline_planning_remains_available(self):
        with tempfile.TemporaryDirectory() as out:
            result = plan.run("Move the forceps from tray 2 to tray 1", offline=True, out_dir=out)
            self.assertEqual(result.data["planner_backend"], "deterministic_parser")

    def test_truth_masks_empty_masks_and_nonfinite_scores_are_rejected(self):
        valid = dict(segmentation_backend="sam3", confidence_score=0.9, active_pixel_count=100)
        validate_segmentation(valid)
        for change in (dict(segmentation_backend="simulator_object_id"), dict(active_pixel_count=0),
                       dict(confidence_score=float("nan")), dict(confidence_score=0.1)):
            with self.subTest(change=change), self.assertRaises(StageError):
                validate_segmentation({**valid, **change})

    def test_outside_tray_and_reference_fallback_cannot_reach_graspgen(self):
        settings = load_profile("full_system")
        valid = dict(centroid_source="optical", pointcloud_units="m", frames=dict(world_frame=[0.6, -0.42, 0.1]))
        with tempfile.TemporaryDirectory() as out, patch.dict("os.environ", {"PIPELINE_OUTPUT_DIR": out}):
            StageReport("plan", data=self.task).save(Path(out) / "plan.json")
            validate_reconstruction(valid, settings["environment"])
            for change in (dict(centroid_source="reference_pose_fallback"),
                           dict(frames=dict(world_frame=[2., 2., .1]))):
                with self.subTest(change=change), self.assertRaises(StageError):
                    validate_reconstruction({**valid, **change}, settings["environment"])


class Orchestration(unittest.TestCase):
    @staticmethod
    def args(out):
        return argparse.Namespace(out_dir=str(out), instruction="Move the Bone Forceps from tray 2 to tray 1",
                                  headless=True, exit_after_run=True, seed=42)

    def test_failed_preflight_never_starts_simulation_and_preserves_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            with patch("pipeline.system.preflight.check", return_value=[dict(name="ollama", ok=False, detail="offline")]), \
                 patch("pipeline.sim.graphical.run_graphical_pipeline") as graphical:
                self.assertEqual(run(load_profile("full_system"), self.args(out)), 1)
                graphical.assert_not_called()
                self.assertEqual(json.loads((out / "run_manifest.json").read_text())["status"], "FAILED")
                with self.assertRaises(FileExistsError):
                    run(load_profile("full_system"), self.args(out))

    def test_profiles_delegate_to_same_graspgen_path_with_distinct_backends(self):
        for profile in ("full_system", "graspgen_only"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "new"
                with patch("pipeline.system.preflight.check", return_value=[]), \
                     patch("pipeline.sim.graphical.run_graphical_pipeline", return_value=130) as graphical:
                    self.assertEqual(run(load_profile(profile), self.args(out)), 130)
                    options, instruction = graphical.call_args.args
                    self.assertEqual(options.grasp_backend, "graspgen")
                    self.assertTrue(options.graspgen_sim_motion)
                    self.assertEqual(options.mode, "single")
                    self.assertEqual(options.vision, "sam3" if profile == "full_system" else "ground-truth")
                    self.assertEqual(options.tool, None if profile == "full_system" else "Bone Forceps")

    def test_final_success_requires_real_backend_evidence_and_deposit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            reports = {
                "plan.json": dict(planner_backend="deterministic_parser"),
                "sim_segmentation.json": dict(segmentation_backend="sam3"),
                "grasp_graspgen.json": dict(grasp_backend="graspgen", candidates=[{}]),
                "graphical_run.json": {},
                "sim_robot_motion.json": dict(execution_checks_passed=True, tool_settled_on_destination=True,
                                               tool_center_inside_destination_xy=True),
            }
            def save():
                for name, data in reports.items():
                    (out / name).write_text(json.dumps(dict(status="COMPLETED", data=data)))
            manifest = dict(profile="full_system")
            save()
            self.assertEqual(finalize(out, manifest, 0), 1)  # silent parser fallback cannot count as success
            reports["plan.json"]["planner_backend"] = "ollama"
            save()
            self.assertEqual(finalize(out, manifest, 0), 0)
            reports["sim_robot_motion.json"]["tool_settled_on_destination"] = False
            save()
            self.assertEqual(finalize(out, manifest, 0), 1)


if __name__ == "__main__":
    unittest.main()
