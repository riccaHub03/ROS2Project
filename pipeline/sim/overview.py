"""Stage ``sim_overview``: RGB-D picture and mask of EVERY instrument on the whole collection.

The robot scene (PyBullet window) always shows the single STL of the target. This stage renders the
full All_Instruments collection offscreen, saves its RGB / depth pictures and masks every instrument
(SAM 3 or the diagnostic simulator mask), highlighting the target. It is informational: the grasp is
still computed on the STL scene, so a failure here never invalidates a grasp.

Input : ``plan.json`` + environment config
Output: ``sim_overview.json`` -> data: scene, backend, source_tray, camera, target{...}, all_instruments{...}
        artifacts: rgb_png, depth_npy, depth_visual_png, all_mask_png, all_overlay_png, all_instances_png,
        target_mask_png, target_overlay_png
        (+ the two intermediate reports ``sim_overview_capture.json`` / ``sim_overview_segmentation.json``)

Run alone:  python -m pipeline.sim.overview --backend ground-truth
"""

from __future__ import annotations

import logging
from typing import List, Optional

from pipeline.common import paths
from pipeline.common.cli import run_cli, stage_parser
from pipeline.common.report import StageReport, stage_run
from pipeline.common.vision import Segmenter
from pipeline.sim import capture, segment

STAGE = "sim_overview"
log = logging.getLogger(STAGE)
CAPTURE_PREFIX = "sim_overview_capture"
SEGMENTATION_PREFIX = "sim_overview_segmentation"
CAPTURE_REPORT = f"{CAPTURE_PREFIX}.json"
SEGMENTATION_REPORT = f"{SEGMENTATION_PREFIX}.json"


def run(*, backend: str = "sam3", plan_path: Optional[str] = None, config_path: Optional[str] = None,
        out_dir: Optional[str] = None, report_path: Optional[str] = None,
        segmenter: Optional[Segmenter] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.SIM_OVERVIEW, out),
                   inputs={"plan": plan_file, "config": config_path}, logger=log) as report:
        cap = capture.run(mode="all", plan_path=plan_path, config_path=config_path, out_dir=out_dir,
                          report_path=CAPTURE_REPORT, prefix=CAPTURE_PREFIX)
        seg = segment.run(backend=backend, plan_path=plan_path, capture_path=str(out / CAPTURE_REPORT),
                          out_dir=out_dir, report_path=SEGMENTATION_REPORT, prefix=SEGMENTATION_PREFIX,
                          segmenter=segmenter, all_instruments=True, require_target=False)
        report.inputs.update(capture=str(out / CAPTURE_REPORT), segmentation=str(out / SEGMENTATION_REPORT))
        report.artifacts = {
            "rgb_png": cap.artifacts["rgb_png"], "depth_npy": cap.artifacts["depth_npy"],
            "depth_visual_png": cap.artifacts["depth_visual_png"],
            "all_mask_png": seg.artifacts["all_mask_png"], "all_overlay_png": seg.artifacts["all_overlay_png"],
            "all_instances_png": seg.artifacts["all_instances_png"],
            "target_mask_png": seg.artifacts["mask_png"], "target_overlay_png": seg.artifacts["overlay_png"]}
        target_keys = ("segmentation_backend", "prompt_used", "confidence_score", "active_pixel_count",
                       "bbox_2d_pixels", "centroid_2d_pixels")
        report.data = {
            "scene": "All_Instruments collection (offscreen PyBullet)", "mode": "all", "backend": backend,
            "source_tray": cap.data["source_tray"], "camera": cap.data["camera"],
            "target": {key: seg.data.get(key) for key in target_keys},
            "all_instruments": seg.data["all_instruments"],
            "note": "Informational view of the whole collection; the grasp is computed on the STL scene."}
        for message in cap.warnings + seg.warnings:
            report.warn(log, message)
        log.info("%d instrument(s) masked on the full collection", seg.data["all_instruments"]["num_instances"])
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--plan", default=None)
    parser.add_argument("--backend", choices=["sam3", "ground-truth"], default="sam3")
    return run_cli(parser, lambda a: run(backend=a.backend, plan_path=a.plan, config_path=a.config,
                                         out_dir=a.out_dir, report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
