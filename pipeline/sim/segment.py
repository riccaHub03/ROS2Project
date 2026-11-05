"""Stage ``sim_segmentation``: isolate the target instrument in the simulated camera view.

Input : ``plan.json`` + ``sim_capture.json``
Output: ``sim_segmentation.json`` -> data: segmentation_backend, prompt_used, confidence_score,
        active_pixel_count, bbox_2d_pixels, centroid_2d_pixels, num_instances
        artifacts: mask_png, overlay_png

Backends: ``sam3`` (default) or ``ground-truth`` (diagnostic: mask from PyBullet object IDs,
requires the ``truth_mask_png`` artifact written by the interactive robot scene).

Run alone:  python -m pipeline.sim.segment --backend sam3
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from PIL import Image

from pipeline.common import paths
from pipeline.common.cli import run_cli, stage_parser
from pipeline.common.report import StageError, StageReport, stage_run
from pipeline.common.vision import (Segmentation, Segmenter, get_segmenter, make_overlay, mask_metrics,
                                    save_mask_and_overlay, segment_image)

STAGE = "sim_segmentation"
log = logging.getLogger(STAGE)
MIN_MASK_PIXELS = 15


def run(*, backend: str = "sam3", plan_path: Optional[str] = None, capture_path: Optional[str] = None,
        out_dir: Optional[str] = None, report_path: Optional[str] = None,
        segmenter: Optional[Segmenter] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    capture_file = paths.resolve(capture_path, paths.SIM_CAPTURE, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.SIM_SEGMENTATION, out),
                   inputs={"plan": plan_file, "capture": capture_file}, logger=log) as report:
        plan = StageReport.load(plan_file, expect_stage="plan")
        capture = StageReport.load(capture_file, expect_stage="sim_capture")
        rgb = np.array(Image.open(capture.artifact("rgb_png")).convert("RGB"))

        if backend == "sam3":
            prompt = plan.require("target_keyword")
            log.info("segmenting '%s' (instrument: %s)", prompt, plan.require("target_instrument"))
            result = segment_image(Image.fromarray(rgb), [prompt], segmenter or get_segmenter())
        elif backend == "ground-truth":
            result = _from_ground_truth(capture, rgb)
        else:
            raise StageError(STAGE, f"unknown backend '{backend}'")

        for message in result.warnings:
            report.warn(log, message)
        mask_png, overlay_png = out / "sim_segmentation_mask.png", out / "sim_segmentation_overlay.png"
        save_mask_and_overlay(result.mask, result.overlay, mask_png, overlay_png)
        report.artifacts = {"mask_png": str(mask_png), "overlay_png": str(overlay_png)}
        report.data = result.data
        if result.data["active_pixel_count"] < MIN_MASK_PIXELS and backend != "sam3":
            raise StageError(STAGE, "ground-truth mask is empty or too small")
        log.info("%d mask pixels, confidence %s", result.data["active_pixel_count"], result.data["confidence_score"])
    return report


def _from_ground_truth(capture: StageReport, rgb: np.ndarray) -> Segmentation:
    mask = np.asarray(Image.open(capture.artifact("truth_mask_png")))
    selected = mask > 128
    data = {"segmentation_backend": "simulator_object_id", "prompt_used": None, "num_instances": 1,
            "confidence_score": None, **mask_metrics(selected)}
    return Segmentation(mask.astype(np.uint8), make_overlay(rgb, selected), data,
                        ["diagnostic backend: SAM 3 bypassed"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0], config=False)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--capture", default=None, help="sim_capture.json")
    parser.add_argument("--backend", choices=["sam3", "ground-truth"], default="sam3")
    return run_cli(parser, lambda a: run(backend=a.backend, plan_path=a.plan, capture_path=a.capture,
                                         out_dir=a.out_dir, report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
