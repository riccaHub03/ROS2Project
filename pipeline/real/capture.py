"""Stage ``real_capture``: acquire the multi-instrument tray image (no STL needed).

Takes a photo from disk (a ROS 2 camera feed can be dumped to a file and passed with
``--image``). With ``--depth`` (aligned metric depth) and ``--intrinsics`` (camera calibration)
the capture is marked ``motion_safe``; without them it uses a nominal constant depth plane on the
tray surface and downstream stages refuse to clear the grasp for robot motion.

Input : image file (or auto-discovered in data/inputs) + environment config
Output: ``real_capture.json`` -> data: source_file, camera (same schema as sim_capture)
        artifacts: rgb_png, depth_npy

Run alone:  python -m pipeline.real.capture --image data/inputs/surgical_scene.jpg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

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
DEFAULT_TRAY = "tray_2"


def find_image(explicit: Optional[str]) -> Path:
    if explicit and Path(explicit).exists():
        return Path(explicit)
    for name in DEFAULT_IMAGES:
        candidate = paths.input_dir() / name
        if candidate.exists():
            return candidate
    raise StageError(STAGE, f"no tray image found (--image {explicit!r}; looked for {DEFAULT_IMAGES} "
                            f"in {paths.input_dir()})")


def load_depth(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Metric depth in metres from ``.npy`` (m) or a 16-bit PNG (mm), aligned with the RGB image."""
    if not Path(path).exists():
        raise StageError(STAGE, f"depth file not found: {path}")
    if str(path).lower().endswith(".npy"):
        depth = np.load(path).astype(np.float32)
    else:
        raw = np.array(Image.open(path))
        if raw.dtype != np.uint16:
            raise StageError(STAGE, f"{path}: expected a 16-bit PNG in millimetres, got {raw.dtype}")
        depth = raw.astype(np.float32) / 1000.0
    if depth.shape != tuple(shape):
        raise StageError(STAGE, f"depth shape {depth.shape} does not match the image {tuple(shape)}: "
                                "align/register the depth to the colour frame first")
    return depth


def run(*, image: Optional[str] = None, depth: Optional[str] = None,
        intrinsics: Optional[Sequence[float]] = None, plan_path: Optional[str] = None,
        config_path: Optional[str] = None, out_dir: Optional[str] = None,
        report_path: Optional[str] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.REAL_CAPTURE, out),
                   inputs={"image": image, "depth": depth, "plan": plan_file if plan_file.exists() else None,
                           "config": config_path}, logger=log) as report:
        source = find_image(image)
        report.inputs["image"] = str(source)
        config = load_environment_config(config_path)
        img = Image.open(source).convert("RGB")
        w, h = img.size

        if plan_file.exists():
            tray_key = StageReport.load(plan_file, expect_stage="plan").require("source_tray")
        else:
            tray_key = DEFAULT_TRAY
            report.warn(log, f"plan report not found ({plan_file}): assuming source tray '{tray_key}'")
        tray = get_tray(config, tray_key)

        cam_cfg = config.get("camera_overhead", {})
        extr = cam_cfg.get("extrinsics", {})
        position = extr.get("position_robot_frame_xyz_m", FALLBACK_CAMERA_POSITION)
        target = extr.get("target_position_xyz_m", [position[0], position[1], tray.top_z])

        # --- intrinsics: calibrated values if given, otherwise scaled from the config (nominal) ---
        base = camera_intrinsics(config)
        calibrated_intrinsics = intrinsics is not None
        if calibrated_intrinsics:
            fx, fy, cx, cy = (float(v) for v in intrinsics)
        else:
            scale = w / base.width                       # same field of view as the configured sensor
            fx, fy, cx, cy = base.fx * scale, base.fy * scale, w / 2.0, h / 2.0
            report.warn(log, f"intrinsics are NOT calibrated: scaled from the config ({base.width}x{base.height} "
                             f"-> {w}x{h}); pass --intrinsics FX FY CX CY from the camera calibration")
        camera = CameraModel(list(position), list(target), Intrinsics(w, h, fx, fy, cx, cy))

        # --- depth: measured if given, otherwise a nominal plane on the tray SURFACE ---------------
        measured_depth = depth is not None
        nominal_distance = float(position[2] - tray.top_z)
        if measured_depth:
            depth_map = load_depth(Path(depth), (h, w))
            depth_source = "measured"
        else:
            depth_map = np.full((h, w), nominal_distance, dtype=np.float32)
            depth_source = "nominal_constant_plane"
            report.warn(log, "no measured depth: using a constant plane on the tray surface; the instrument "
                             "height is unknown, so this result is NOT cleared for robot motion")

        rgb_png, depth_npy = out / "real_capture_rgb.png", out / "real_capture_depth.npy"
        img.save(rgb_png)
        np.save(depth_npy, depth_map)

        motion_safe = measured_depth and calibrated_intrinsics
        report.artifacts = {"rgb_png": str(rgb_png), "depth_npy": str(depth_npy)}
        report.data = {"source_file": str(source), "source_type": "real_surgical_tray_image",
                       "source_tray": tray.key, "camera": camera.to_dict(), "depth_source": depth_source,
                       "intrinsics_calibrated": calibrated_intrinsics,
                       "nominal_distance_m": round(nominal_distance, 4), "motion_safe": motion_safe}
        log.info("captured %s (%dx%d), depth %s, motion_safe=%s", source.name, w, h, depth_source, motion_safe)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--image", default=None, help="Tray photo (default: data/inputs/surgical_scene.jpg)")
    parser.add_argument("--depth", default=None, help="Metric depth aligned with the image: .npy (m) or 16-bit PNG (mm)")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"), default=None,
                        help="Calibrated intrinsics in pixels of --image")
    parser.add_argument("--plan", default=None, help="plan.json (source tray); default <out-dir>/plan.json")
    return run_cli(parser, lambda a: run(image=a.image, depth=a.depth, intrinsics=a.intrinsics, plan_path=a.plan,
                                         config_path=a.config, out_dir=a.out_dir, report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
