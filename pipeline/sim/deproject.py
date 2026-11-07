"""Stage ``sim_deprojection``: segmentation mask + depth -> 3D instrument cloud and frame centroids.

Input : ``sim_segmentation.json`` + ``sim_capture.json`` + config
        (+ optional ``--reference-pose`` = sim_reference_pose.json, used when the optical
        centroid falls outside the source tray)
Output: ``sim_deprojection.json`` -> data: num_points, frames{camera_optical_frame, robot_base_frame,
        end_effector_frame}, centroid_source ('optical' | 'reference_pose_fallback'),
        ground_truth_validation{ground_truth_xyz_m, reconstructed_xyz_m, error_mm}
        artifacts: pointcloud_ply, frames_png

Run alone:  python -m pipeline.sim.deproject --reference-pose data/outputs/sim_reference_pose.json
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from PIL import Image

from pipeline.common import paths
from pipeline.common.cli import run_cli, stage_parser
from pipeline.common.ply import write_ply
from pipeline.common.reconstruction import CameraModel, reconstruct
from pipeline.common.report import StageReport, stage_run
from pipeline.common.scene import get_tray, round_list
from pipeline.config_loader import load_environment_config
from pipeline.sim.frames_plot import render_frames_figure

STAGE = "sim_deprojection"
log = logging.getLogger(STAGE)
TRAY_MARGIN_M = 0.05


def run(*, segmentation_path: Optional[str] = None, capture_path: Optional[str] = None,
        reference_pose_path: Optional[str] = None, config_path: Optional[str] = None,
        out_dir: Optional[str] = None, report_path: Optional[str] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    seg_file = paths.resolve(segmentation_path, paths.SIM_SEGMENTATION, out)
    capture_file = paths.resolve(capture_path, paths.SIM_CAPTURE, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.SIM_DEPROJECTION, out),
                   inputs={"segmentation": seg_file, "capture": capture_file,
                           "reference_pose": reference_pose_path, "config": config_path}, logger=log) as report:
        segmentation = StageReport.load(seg_file, expect_stage="sim_segmentation")
        capture = StageReport.load(capture_file, expect_stage="sim_capture")
        config = load_environment_config(config_path)

        mask = np.array(Image.open(segmentation.artifact("mask_png")).convert("L"))
        depth = np.load(capture.artifact("depth_npy"))
        rgb = np.array(Image.open(capture.artifact("rgb_png")).convert("RGB"))
        camera = CameraModel.from_dict(capture.require("camera"))

        rec = reconstruct(mask, depth, rgb, camera, config)
        tray = get_tray(config, capture.require("source_tray"))
        source = "optical"
        if tray.contains_xy(rec.centroid_world, TRAY_MARGIN_M):
            log.info("optical world centroid %s lies inside %s", round_list(rec.centroid_world), tray.key)
        elif reference_pose_path:
            pose = StageReport.load(reference_pose_path, expect_stage="sim_reference_pose").require("pose", "position_xyz_m")
            report.warn(log, f"optical centroid {round_list(rec.centroid_base)} is OUTSIDE {tray.key}; "
                             f"falling back to the certified reference pose {pose}")
            rec, source = rec.with_centroid_base(rec.transformer.base_frame.world_to_base(pose)), "reference_pose_fallback"
        else:
            report.warn(log, f"optical centroid {round_list(rec.centroid_base)} is OUTSIDE {tray.key} "
                             "and no --reference-pose was given: keeping the optical centroid")

        truth = capture.data.get("ground_truth_pose", {}).get("position_xyz_m")
        error_mm = float(np.linalg.norm(rec.centroid_world - np.array(truth))) * 1000.0 if truth else 0.0

        cloud, figure = out / "sim_deprojection_cloud.ply", out / "sim_deprojection_frames.png"
        write_ply(cloud, rec.points_base, rec.colors)
        render_frames_figure(figure, rgb, rec.transformer, rec.centroid_base, camera.position, config)

        report.artifacts = {"pointcloud_ply": str(cloud), "frames_png": str(figure)}
        report.data = {
            "num_points": int(len(rec.points_base)), "centroid_source": source,
            "coordinate_frame_id": rec.transformer.base_frame.frame_id,
            "robot_base_pose_world": rec.transformer.base_frame.to_dict(),
            "frames": rec.frames(config),
            "ground_truth_validation": {"ground_truth_xyz_m": truth,
                                        "reconstructed_xyz_m": round_list(rec.centroid_world),
                                        "frame_id": "world",
                                        "error_mm": round(error_mm, 2)},
        }
        log.info("%d points | base centroid %s | error vs ground truth %.2f mm",
                 len(rec.points_base), round_list(rec.centroid_base), error_mm)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--segmentation", default=None, help="sim_segmentation.json")
    parser.add_argument("--capture", default=None, help="sim_capture.json")
    parser.add_argument("--reference-pose", default=None,
                        help="sim_reference_pose.json, used as fallback when the centroid is outside the tray")
    return run_cli(parser, lambda a: run(segmentation_path=a.segmentation, capture_path=a.capture,
                                         reference_pose_path=a.reference_pose, config_path=a.config,
                                         out_dir=a.out_dir, report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
