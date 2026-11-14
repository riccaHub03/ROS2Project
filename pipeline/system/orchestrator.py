"""Compose existing stages; own run identity, preflight and evidence of actual backends."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from uuid import uuid4

from pipeline.common import paths
from pipeline.common.report import write_json


def _report(out, name):
    file = out / name
    return json.loads(file.read_text()) if file.is_file() else {}


def finalize(out, manifest, code, error=None):
    plan = _report(out, "plan.json")
    seg = _report(out, "sim_segmentation.json")
    grasp = _report(out, "grasp_graspgen.json")
    graphical = _report(out, "graphical_run.json")
    motion = _report(out, "sim_robot_motion.json")
    pd, sd, gd, md = (report.get("data", {}) for report in (plan, seg, grasp, motion))
    evidence = dict(task_llm_used=pd.get("planner_backend") == "ollama",
                    segmentation_sam3_used=sd.get("segmentation_backend") == "sam3",
                    graspgen_used=gd.get("grasp_backend") == "graspgen" and bool(gd.get("candidates")),
                    motion_checks_passed=md.get("execution_checks_passed") is True,
                    placement_verified=md.get("tool_settled_on_destination") is True and
                                       md.get("tool_center_inside_destination_xy") is True,
                    physical_grasp_validated=False)
    complete = code == 0 and graphical.get("status") == "COMPLETED" and motion.get("status") == "COMPLETED"
    complete = complete and all(evidence[k] for k in ("graspgen_used", "motion_checks_passed", "placement_verified"))
    if manifest["profile"] == "full_system":
        complete = complete and evidence["task_llm_used"] and evidence["segmentation_sam3_used"]
    cancelled = graphical.get("status") == "CANCELLED" or code == 130
    status = "COMPLETED" if complete else "CANCELLED" if cancelled else "FAILED"
    if status == "FAILED" and not error and not graphical.get("error"):
        error = "Required backend, motion or placement evidence is missing; inspect stage reports"
    manifest.update(status=status, finished_at=datetime.now().astimezone().isoformat(), evidence=evidence,
                    error=error or graphical.get("error"),
                    actual_backends=dict(planner=pd.get("planner_backend"), planner_model=pd.get("planner_model"),
                                         segmentation=sd.get("segmentation_backend"), sam3_model=sd.get("model_id"),
                                         grasp=gd.get("grasp_backend")),
                    stages={name: _report(out, name).get("status", "NOT_STARTED") for name in
                            ("plan.json", "sim_capture.json", "sim_segmentation.json", "sim_deprojection.json",
                             "grasp_graspgen.json", "graspgen_sim_motion_validation.json", "sim_robot_motion.json")},
                    artifacts={p.name: str(p) for p in sorted(out.iterdir()) if p.is_file() and not p.name.startswith(".")})
    write_json(out / "run_manifest.json", manifest)
    summary = [f"# {manifest['profile']} — {out.name}", "", f"Status: **{status}**", "",
               "Scope: PyBullet single-object simulation; ideal fixed-constraint grasp.", "",
               "| Evidence | Value |", "|---|---|"]
    summary += [f"| {key} | {value} |" for key, value in evidence.items()]
    if manifest["error"]:
        summary += ["", f"Error: {manifest['error']}"]
    summary += ["", "## Artifacts", ""]
    summary += [f"- [{name}]({path})" for name, path in manifest["artifacts"].items() if name not in ("summary.md", "run_manifest.json")]
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    return 0 if complete else 130 if cancelled else 1


def run(settings, args):
    from pipeline.system.preflight import check
    out = Path(args.out_dir).resolve() if args.out_dir else paths.project_root() / "data/outputs" / (
        settings["profile"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    # Exclusive creation also prevents overwriting previous runs or racing another launcher.
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, run_id=out.name, request_id=uuid4().hex,
                    profile=settings["profile"], status="PREFLIGHT", settings=settings,
                    started_at=datetime.now().astimezone().isoformat(), instruction=args.instruction)
    write_json(out / "run_manifest.json", manifest)
    previous_output = os.environ.get(paths.OUTPUT_ENV)
    code, error = 1, None
    try:
        checks = check(settings)
        manifest["preflight"] = checks
        write_json(out / "run_manifest.json", manifest)
        failed = [f"{c['name']}: {c['detail']}" for c in checks if not c["ok"]]
        if failed:
            raise RuntimeError("Preflight failed: " + "; ".join(failed))
        from pipeline.sim.pipeline import build_parser
        from pipeline.sim.graphical import run_graphical_pipeline
        argv = ["--gui" if not args.headless else "--headless", "--mode", "single",
                "--config", settings["environment"], "--vision", settings["vision"],
                "--grasp-backend", "graspgen", "--graspgen-sim-motion",
                "--graspgen-config", settings["graspgen_config"], "--out-dir", str(out),
                "--seed", str(args.seed)]
        options = build_parser().parse_args(argv)
        options.exit_after_run = args.exit_after_run
        options.system_options = settings
        options.tool = settings["scene_tool"] if settings["planner"] == "deterministic" else None
        manifest["status"] = "RUNNING"
        write_json(out / "run_manifest.json", manifest)
        code = run_graphical_pipeline(options, args.instruction)
    except KeyboardInterrupt:
        code, error = 130, "Interrupted by user"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if previous_output is None:
            os.environ.pop(paths.OUTPUT_ENV, None)
        else:
            os.environ[paths.OUTPUT_ENV] = previous_output
    result = finalize(out, manifest, code, error)
    print(f"{manifest['status']}: {out}")
    if manifest.get("error"):
        print(manifest["error"])
    return result
