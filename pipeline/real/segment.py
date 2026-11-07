"""Stage ``real_segmentation``: pick the target instrument among all tools on the real tray (SAM 3).

Input : ``plan.json`` + ``real_capture.json``
Output: ``real_segmentation.json`` (same schema as sim_segmentation)
        artifacts: mask_png, overlay_png (selected = green, other candidates = blue)

Run alone:  python -m pipeline.real.segment
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PIL import Image

from pipeline.common import paths
from pipeline.common.cli import run_cli, stage_parser
from pipeline.common.report import StageReport, stage_run
from pipeline.common.vision import Segmenter, get_segmenter, save_mask_and_overlay, segment_image

STAGE = "real_segmentation"
log = logging.getLogger(STAGE)
OTHERS_MIN_SCORE = 0.15


def run(*, plan_path: Optional[str] = None, capture_path: Optional[str] = None, out_dir: Optional[str] = None,
        report_path: Optional[str] = None, segmenter: Optional[Segmenter] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    capture_file = paths.resolve(capture_path, paths.REAL_CAPTURE, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.REAL_SEGMENTATION, out),
                   inputs={"plan": plan_file, "capture": capture_file}, logger=log) as report:
        plan = StageReport.load(plan_file, expect_stage="plan")
        capture = StageReport.load(capture_file, expect_stage="real_capture")
        image = Image.open(capture.artifact("rgb_png")).convert("RGB")

        # Full instrument name first, then the simplified keyword if nothing matched.
        prompts = [plan.require("target_instrument"), plan.require("target_keyword")]
        result = segment_image(image, prompts, segmenter or get_segmenter(), require_match=True,
                               others_min_score=OTHERS_MIN_SCORE)
        for message in result.warnings:
            report.warn(log, message)
        mask_png, overlay_png = out / "real_segmentation_mask.png", out / "real_segmentation_overlay.png"
        save_mask_and_overlay(result.mask, result.overlay, mask_png, overlay_png)
        report.artifacts = {"mask_png": str(mask_png), "overlay_png": str(overlay_png)}
        report.data = result.data
        log.info("isolated '%s' with confidence %.1f%%", result.data["prompt_used"],
                 result.data["confidence_score"] * 100)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0], config=False)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--capture", default=None, help="real_capture.json")
    return run_cli(parser, lambda a: run(plan_path=a.plan, capture_path=a.capture, out_dir=a.out_dir,
                                         report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
