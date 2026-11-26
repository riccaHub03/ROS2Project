"""Real-image pipeline: plan -> capture -> segment -> deproject -> grasp.

Run:  python run_pipeline_real.py --image data/inputs/surgical_scene.jpg          (from files)
      python run_pipeline_real.py --source ros2 --ee-frame lbr_link_ee         (live camera + TF)
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from pipeline import grasp, plan
from pipeline.common import paths
from pipeline.common.cli import setup_logging
from pipeline.common.report import StageError, StageReport
from pipeline.common.vision import REAL_MIN_CONFIDENCE, SAM3_MODEL_ID, Sam3Segmenter
from pipeline.orchestrate import Step, execute
from pipeline.real import capture, deproject, segment

STAGES = ["plan", "capture", "segment", "deproject", "grasp", "simulate"]
log = logging.getLogger("pipeline.real")


def check_plan(report: StageReport, *, require_llm: bool, expected_keyword: Optional[str]) -> StageReport:
    if require_llm and report.data.get("planner_backend") != "ollama":
        raise StageError("plan", "Ollama is required; stored plan is not from Ollama")
    if expected_keyword and str(report.data.get("target_keyword", "")).casefold() != expected_keyword.strip().casefold():
        raise StageError("plan", f"target keyword {report.data.get('target_keyword')!r} does not match "
                                 f"expected {expected_keyword!r}; check the LLM plan")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruction", default=plan.DEFAULT_INSTRUCTION, help="Clinical command for the LLM planner")
    parser.add_argument("--require-llm", action="store_true", help="Fail if Ollama is unavailable; do not use fallback")
    parser.add_argument("--expected-keyword", default=None, help="Reject a plan whose target keyword differs")
    parser.add_argument("--image", default=None, help="Real photo of the tray with all instruments (--source file)")
    parser.add_argument("--depth", default=None, help="Metric depth aligned with --image: .npy (m) or 16-bit PNG (mm). "
                        "Without it a nominal constant plane is used and the grasp is NOT cleared for robot motion")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"), default=None,
                        help="Calibrated camera intrinsics in pixels of --image")
    capture.add_source_arguments(parser)
    parser.add_argument("--min-confidence", type=float, default=REAL_MIN_CONFIDENCE,
                        help="SAM 3 acceptance threshold (default %(default)s)")
    parser.add_argument("--sam3-model", default=SAM3_MODEL_ID, help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--sam3-device", choices=("cpu", "cuda"), default="cpu", help="SAM 3 inference device")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--simulate-deposit", action="store_true",
                        help="after GraspGen, plan and execute an idealized deposit in PyBullet only")
    parser.add_argument("--sim-omit-return", action="store_true",
                        help="finish after the checked retreat instead of returning to the optional park pose")
    parser.add_argument("--sim-video", action="store_true", help="render the PyBullet preview to MP4 (requires ffmpeg)")
    parser.add_argument("--sim-grasp-attempts", type=int, default=3,
                        help="maximum independent GraspGen batches if candidates are rejected (default 3)")
    parser.add_argument("--from-stage", choices=STAGES, default=None)
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    grasp.add_backend_arguments(parser)
    return parser


def build_steps(args: argparse.Namespace) -> List[Step]:
    common = dict(out_dir=args.out_dir)
    cfg = dict(config_path=args.config, **common)
    out = paths.output_dir(args.out_dir)

    def run_plan():
        return check_plan(plan.run(args.instruction, require_llm=args.require_llm, **cfg),
                          require_llm=args.require_llm, expected_keyword=args.expected_keyword)

    def run_capture():
        if args.require_llm or args.expected_keyword:
            check_plan(StageReport.load(out / paths.PLAN, expect_stage="plan"),
                       require_llm=args.require_llm, expected_keyword=args.expected_keyword)
        return capture.run(image=args.image, depth=args.depth, intrinsics=args.intrinsics,
                           **capture.source_kwargs(args), **cfg)

    steps = [
        Step("plan", run_plan),
        Step("capture", run_capture),
        Step("segment", lambda: segment.run(min_confidence=args.min_confidence,
                                            segmenter=Sam3Segmenter(args.sam3_model, args.sam3_device), **common)),
        Step("deproject", lambda: deproject.run(**cfg)),
        Step("grasp", lambda: grasp.run(deprojection_path=str(out / paths.REAL_DEPROJECTION),
                                       **grasp.backend_kwargs(args), **cfg)),
    ]
    if args.simulate_deposit:
        from pipeline.real import simulation_preview
        steps.append(Step("simulate", lambda: simulation_preview.run(
            out_dir=str(out), clip_dir=args.clip_dir, source=args.source, config_path=args.config,
            omit_return=args.sim_omit_return, video=args.sim_video, max_attempts=args.sim_grasp_attempts,
            grasp_kwargs=grasp.backend_kwargs(args))))
    return steps


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        if (args.from_stage == "simulate" or args.stop_after == "simulate") and not args.simulate_deposit:
            raise StageError("pipeline", "simulation stage requires --simulate-deposit")
        if args.simulate_deposit:
            # This full chain must not silently substitute the deterministic parser for the LLM.
            args.require_llm = True
            if args.source not in ("clip", "ros2"):
                raise StageError("pipeline", "--simulate-deposit requires --source clip or --source ros2")
            if args.source == "clip" and args.config:
                raise StageError("pipeline", "clip preview currently requires the default cell configuration")
        if args.simulate_deposit and args.grasp_backend != "graspgen":
            raise StageError("pipeline", "--simulate-deposit requires --grasp-backend graspgen")
        if args.simulate_deposit and not 1 <= args.sim_grasp_attempts <= 10:
            raise StageError("pipeline", "--sim-grasp-attempts must be between 1 and 10")
        if args.simulate_deposit and args.graspgen_config is None:
            args.graspgen_config = str(paths.project_root() / "config/graspgen_simulation.yaml")
        if (args.source == "clip" or args.simulate_deposit) and args.from_stage in (None, "plan"):
            output = paths.output_dir(args.out_dir, create=False)
            if output.exists() and any(output.iterdir()):
                raise StageError("pipeline", f"output directory is not empty: {output}; choose a new --out-dir")
        if args.from_stage not in (None, "plan", "capture") and (args.require_llm or args.expected_keyword):
            saved = StageReport.load(paths.output_dir(args.out_dir) / paths.PLAN, expect_stage="plan")
            check_plan(saved, require_llm=args.require_llm, expected_keyword=args.expected_keyword)
        execute(build_steps(args), start_at=args.from_stage, stop_after=args.stop_after)
    except StageError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
