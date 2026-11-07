"""Stage ``real_capture``: acquire the multi-instrument tray image (no STL needed).

Takes a photo from disk (a ROS 2 camera feed can be dumped to a file and passed with
``--image``). Metric depth is a calibrated constant plane derived from the config.

Input : image file (or auto-discovered in data/inputs) + environment config
Output: ``real_capture.json`` -> data: source_file, camera (same schema as sim_capture)
        artifacts: rgb_png, depth_npy

Run alone:  python -m pipeline.real.capture --image data/inputs/surgical_scene.jpg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from pipeline.common import paths
from pipeline.common.cli import run_cli, stage_parser
from pipeline.common.reconstruction import CameraModel
from pipeline.common.report import StageError, StageReport, stage_run
from pipeline.common.scene import Intrinsics, camera_intrinsics, get_tray
from pipeline.config_loader import load_environment_config

STAGE = "real_capture"
log = logging.getLogger(STAGE)
DEFAULT_IMAGES = ("surgical_scene.jpg", "blog-ferri-chirurgici.jpg")
FALLBACK_CAMERA_POSITION = [0.55, 0.0, 0.85]


def find_image(explicit: Optional[str]) -> Path:
    if explicit and Path(explicit).exists():
        return Path(explicit)
    for name in DEFAULT_IMAGES:
        candidate = paths.input_dir() / name
        if candidate.exists():
            return candidate
    raise StageError(STAGE, f"no tray image found (--image {explicit!r}; looked for {DEFAULT_IMAGES} "
                            f"in {paths.input_dir()})")


def run(*, image: Optional[str] = None, config_path: Optional[str] = None, out_dir: Optional[str] = None,
        report_path: Optional[str] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    with stage_run(STAGE, paths.output_report(report_path, paths.REAL_CAPTURE, out),
                   inputs={"image": image, "config": config_path}, logger=log) as report:
        source = find_image(image)
        report.inputs["image"] = str(source)
        config = load_environment_config(config_path)
        img = Image.open(source).convert("RGB")
        w, h = img.size

        cam_cfg = config.get("camera_overhead", {})
        extr = cam_cfg.get("extrinsics", {})
        position = extr.get("position_world_frame_xyz_m",
                            extr.get("position_robot_frame_xyz_m", FALLBACK_CAMERA_POSITION))
        target = extr.get("target_position_xyz_m", [position[0], position[1], 0.07])
        tray_z = get_tray(config, "tray_2").position[2]
        nominal_distance = float(position[2] - tray_z)

        base = camera_intrinsics(config)
        # The photo defines the real resolution and principal point; focal lengths come from the config.
        camera = CameraModel(list(position), list(target), Intrinsics(w, h, base.fx, base.fy, w / 2.0, h / 2.0))

        rgb_png, depth_npy = out / "real_capture_rgb.png", out / "real_capture_depth.npy"
        img.save(rgb_png)
        np.save(depth_npy, np.full((h, w), nominal_distance, dtype=np.float32))

        report.artifacts = {"rgb_png": str(rgb_png), "depth_npy": str(depth_npy)}
        report.data = {"source_file": str(source), "source_type": "real_surgical_tray_image",
                       "camera": camera.to_dict(), "nominal_distance_m": round(nominal_distance, 4)}
        log.info("captured %s (%dx%d), nominal depth %.3f m", source.name, w, h, nominal_distance)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--image", default=None, help="Tray photo (default: data/inputs/surgical_scene.jpg)")
    return run_cli(parser, lambda a: run(image=a.image, config_path=a.config, out_dir=a.out_dir,
                                         report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
