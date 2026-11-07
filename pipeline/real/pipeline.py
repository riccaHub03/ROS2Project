"""Real-image pipeline: plan -> capture -> segment -> deproject -> grasp.

Run:  python run_pipeline_real.py --image data/inputs/surgical_scene.jpg
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from pipeline import grasp, plan
from pipeline.common import paths
from pipeline.common.cli import setup_logging
from pipeline.common.report import StageError
from pipeline.orchestrate import Step, execute
from pipeline.real import capture, deproject, segment

STAGES = ["plan", "capture", "segment", "deproject", "grasp"]
log = logging.getLogger("pipeline.real")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruction", default=plan.DEFAULT_INSTRUCTION, help="Clinical command for the LLM planner")
    parser.add_argument("--image", default=None, help="Real photo of the tray with all instruments")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--from-stage", choices=STAGES, default=None)
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def build_steps(args: argparse.Namespace) -> List[Step]:
    common = dict(out_dir=args.out_dir)
    cfg = dict(config_path=args.config, **common)
    out = paths.output_dir(args.out_dir)
    return [
        Step("plan", lambda: plan.run(args.instruction, **cfg)),
        Step("capture", lambda: capture.run(image=args.image, **cfg)),
        Step("segment", lambda: segment.run(**common)),
        Step("deproject", lambda: deproject.run(**cfg)),
        Step("grasp", lambda: grasp.run(deprojection_path=str(out / paths.REAL_DEPROJECTION), **common)),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        execute(build_steps(args), start_at=args.from_stage, stop_after=args.stop_after)
    except StageError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
