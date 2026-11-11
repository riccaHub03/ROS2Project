"""Stage ``sim_segmentation``: isolate the target instrument, and optionally EVERY instrument, in the simulated view.

Input : ``plan.json`` + ``sim_capture.json``
Output: ``sim_segmentation.json`` -> data: segmentation_backend, prompt_used, confidence_score,
        active_pixel_count, bbox_2d_pixels, centroid_2d_pixels, num_instances and, with
        ``all_instruments``, ``all_instruments{prompt, num_instances, coverage_fraction, instances[...]}``
        artifacts: mask_png, overlay_png (target) and, with ``all_instruments``, all_mask_png,
        all_overlay_png (one colour per instrument, target in green), all_instances_png (16-bit labels)

Backends: ``sam3`` (default) or ``ground-truth`` (diagnostic: mask from PyBullet object IDs,
requires the ``truth_mask_png`` artifact written by the capture stage / the interactive robot scene).
With ``ground-truth`` the whole All_Instruments collection is ONE rigid body, so in ``all`` mode the
simulator mask covers every instrument but cannot isolate the target.

``prefix`` renames the artifacts (default ``sim_segmentation``) so that segmenting a second view, e.g. the
offscreen overview, never overwrites the masks the grasp is built on.

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
from pipeline.common.vision import (LOW_CONFIDENCE, AllInstruments, Segmentation, Segmenter,
                                    all_instruments_from_mask, get_segmenter, make_overlay, mask_metrics,
                                    save_mask_and_overlay, segment_all_instruments, segment_image)

STAGE = "sim_segmentation"
log = logging.getLogger(STAGE)
MIN_MASK_PIXELS = 15


def run(*, backend: str = "sam3", plan_path: Optional[str] = None, capture_path: Optional[str] = None,
        out_dir: Optional[str] = None, report_path: Optional[str] = None,
        segmenter: Optional[Segmenter] = None, prefix: str = "sim_segmentation",
        all_instruments: bool = True, require_target: bool = True) -> StageReport:
    """``all_instruments``: also mask every instrument; ``require_target``: fail if the target is not found
    with enough confidence (set False for informational views where the target may be absent)."""
    out = paths.output_dir(out_dir)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    capture_file = paths.resolve(capture_path, paths.SIM_CAPTURE, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.SIM_SEGMENTATION, out),
                   inputs={"plan": plan_file, "capture": capture_file}, logger=log) as report:
        plan = StageReport.load(plan_file, expect_stage="plan")
        capture = StageReport.load(capture_file, expect_stage="sim_capture")
        rgb = np.array(Image.open(capture.artifact("rgb_png")).convert("RGB"))

        sam = None
        if backend == "sam3":
            sam = segmenter or get_segmenter()
            prompt = plan.require("target_keyword")
            log.info("segmenting '%s' (instrument: %s)", prompt, plan.require("target_instrument"))
            result = segment_image(Image.fromarray(rgb), [prompt], sam,
                                   min_confidence=LOW_CONFIDENCE if require_target else None)
        elif backend == "ground-truth":
            result = _from_ground_truth(capture, rgb)
        else:
            raise StageError(STAGE, f"unknown backend '{backend}'")

        for message in result.warnings:
            report.warn(log, message)
        mask_png, overlay_png = out / f"{prefix}_mask.png", out / f"{prefix}_overlay.png"
        save_mask_and_overlay(result.mask, result.overlay, mask_png, overlay_png)
        report.artifacts = {"mask_png": str(mask_png), "overlay_png": str(overlay_png)}
        report.data = result.data
        if result.data["active_pixel_count"] < MIN_MASK_PIXELS and backend != "sam3":
            raise StageError(STAGE, "ground-truth mask is empty or too small")
        log.info("%d mask pixels, confidence %s", result.data["active_pixel_count"], result.data["confidence_score"])

        if all_instruments:
            every = _every_instrument(backend, capture, rgb, sam, target=result.mask > 128)
            for message in every.warnings:
                report.warn(log, message)
            all_mask, all_overlay = out / f"{prefix}_all_mask.png", out / f"{prefix}_all_overlay.png"
            all_labels = out / f"{prefix}_all_instances.png"
            save_mask_and_overlay(every.mask, every.overlay, all_mask, all_overlay)
            Image.fromarray(every.labels).save(all_labels)
            report.artifacts.update(all_mask_png=str(all_mask), all_overlay_png=str(all_overlay),
                                    all_instances_png=str(all_labels))
            report.data["all_instruments"] = every.data
            log.info("all instruments: %d instance(s), %.1f%% of the view", every.data["num_instances"],
                     every.data["coverage_fraction"] * 100)
    return report


def _from_ground_truth(capture: StageReport, rgb: np.ndarray) -> Segmentation:
    mask = np.asarray(Image.open(capture.artifact("truth_mask_png")))
    selected = mask > 128
    data = {"segmentation_backend": "simulator_object_id", "prompt_used": None, "num_instances": 1,
            "confidence_score": None, **mask_metrics(selected)}
    warnings = ["diagnostic backend: SAM 3 bypassed"]
    if capture.data.get("mode") == "all":
        warnings.append("ground-truth mask covers the WHOLE collection (one rigid body): it cannot isolate the target; "
                        "use --mode single or SAM 3 for the target")
    return Segmentation(mask.astype(np.uint8), make_overlay(rgb, selected), data, warnings)


def _every_instrument(backend: str, capture: StageReport, rgb: np.ndarray, segmenter: Optional[Segmenter],
                      *, target: np.ndarray) -> AllInstruments:
    if backend == "sam3":
        return segment_all_instruments(Image.fromarray(rgb), segmenter, target=target)
    truth = np.asarray(Image.open(capture.artifact("truth_mask_png")))
    return all_instruments_from_mask(rgb, truth, target=target)


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0], config=False)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--capture", default=None, help="sim_capture.json")
    parser.add_argument("--backend", choices=["sam3", "ground-truth"], default="sam3")
    parser.add_argument("--no-all-instruments", action="store_true",
                        help="Skip the mask of every instrument (saves one SAM 3 inference)")
    return run_cli(parser, lambda a: run(backend=a.backend, plan_path=a.plan, capture_path=a.capture,
                                         out_dir=a.out_dir, report_path=a.report,
                                         all_instruments=not a.no_all_instruments), argv)


if __name__ == "__main__":
    raise SystemExit(main())
