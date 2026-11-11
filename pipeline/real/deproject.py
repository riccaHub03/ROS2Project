"""Stage ``real_deprojection``: real mask + calibrated depth -> 3D instrument cloud and frames.

Input : ``real_segmentation.json`` + ``real_capture.json`` + config
Output: ``real_deprojection.json`` -> data: num_points, frames{camera_optical_frame, robot_base_frame,
        end_effector_frame}
        artifacts: pointcloud_ply

Run alone:  python -m pipeline.real.deproject
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
from pipeline.common.scene import ee_reference_source, round_list
from pipeline.config_loader import load_environment_config

STAGE = "real_deprojection"
log = logging.getLogger(STAGE)


def run(*, segmentation_path: Optional[str] = None, capture_path: Optional[str] = None,
        config_path: Optional[str] = None, out_dir: Optional[str] = None,
        report_path: Optional[str] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    seg_file = paths.resolve(segmentation_path, paths.REAL_SEGMENTATION, out)
    capture_file = paths.resolve(capture_path, paths.REAL_CAPTURE, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.REAL_DEPROJECTION, out),
                   inputs={"segmentation": seg_file, "capture": capture_file, "config": config_path},
                   logger=log) as report:
        segmentation = StageReport.load(seg_file, expect_stage="real_segmentation")
        capture = StageReport.load(capture_file, expect_stage="real_capture")
        config = load_environment_config(config_path)

        mask = np.array(Image.open(segmentation.artifact("mask_png")).convert("L"))
        depth = np.load(capture.artifact("depth_npy"))
        rgb = np.array(Image.open(capture.artifact("rgb_png")).convert("RGB"))
        rec = reconstruct(mask, depth, rgb, CameraModel.from_dict(capture.require("camera")), config)

        cloud = out / "real_deprojection_cloud.ply"
        write_ply(cloud, rec.points_base, rec.colors)
        report.artifacts = {"pointcloud_ply": str(cloud)}
        # Physical-motion clearance: only measured depth + calibrated intrinsics + verified extrinsics
        verified = bool(capture.data.get("motion_safe", False))
        report.data = {"pointcloud_units": "m", "geometry_verified_for_motion": verified, "motion_safe": verified,
                       "depth_source": capture.data.get("depth_source"),
                       "geometry_verification_note": (
                           "Measured RGB-D depth, calibrated intrinsics and verified extrinsics" if verified else
                           "No measured RGB-D depth, calibrated intrinsics or verified extrinsic calibration"),
                       "num_points": int(len(rec.points_base)), "frames": rec.frames(config),
                       "ee_reference_source": ee_reference_source(config),
                       "coordinate_frame_id": rec.transformer.base_frame.frame_id,
                       "robot_base_pose_world": rec.transformer.base_frame.to_dict()}
        log.info("%d points | base centroid %s", len(rec.points_base), round_list(rec.centroid_base))
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--segmentation", default=None, help="real_segmentation.json")
    parser.add_argument("--capture", default=None, help="real_capture.json")
    return run_cli(parser, lambda a: run(segmentation_path=a.segmentation, capture_path=a.capture,
                                         config_path=a.config, out_dir=a.out_dir, report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
