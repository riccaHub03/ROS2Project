"""Stage ``real_capture``: acquire the multi-instrument tray image (no STL needed).

Two sources:

* ``--source file`` (default): a photo from disk (``--image``, ``--depth``, ``--intrinsics``).
* ``--source ros2``: live from the physical RGB-D camera (colour, depth aligned to colour and CameraInfo topics)
  and the TF tree (camera pose from the hand-eye calibration, optional end-effector pose). See
  ``pipeline.real.ros_camera`` and ``camera_overhead.ros2`` in the YAML.

The capture is marked ``motion_safe`` only with measured depth, calibrated intrinsics and
``camera_overhead.extrinsics.verified: true`` (your attestation that the camera pose is a real hand-eye
calibration); otherwise downstream stages refuse to clear the grasp for robot motion.

Input : image file / live ROS 2 camera + environment config
Output: ``real_capture.json`` -> data: source_file, camera (same schema as sim_capture), acquisition (ros2),
        ee_reference_pose (ros2 with an EE frame)
        artifacts: rgb_png, depth_npy

Run alone:  python -m pipeline.real.capture --image data/inputs/surgical_scene.jpg
            python -m pipeline.real.capture --source ros2 --ee-frame lbr_link_ee
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
from pipeline.frame_transformer import matrix_from_quaternion
from pipeline.real import ros_camera

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


def add_source_arguments(parser) -> None:
    """Opzioni della sorgente, condivise con run_pipeline_real.py. I default vengono dal YAML."""
    parser.add_argument("--source", choices=("file", "ros2"), default="file",
                        help="file: --image/--depth; ros2: live from the RGB-D camera and TF")
    group = parser.add_argument_group("ROS 2 live source (--source ros2); defaults: camera_overhead.ros2 in the YAML")
    group.add_argument("--color-topic", default=None, help="sensor_msgs/Image colour (rgb8/bgr8)")
    group.add_argument("--depth-topic", default=None, help="sensor_msgs/Image depth ALIGNED to colour (16UC1 mm / 32FC1 m)")
    group.add_argument("--camera-info-topic", default=None, help="sensor_msgs/CameraInfo of the colour image")
    group.add_argument("--world-frame", default=None, help="TF frame in which the pipeline expresses poses (default: world)")
    group.add_argument("--camera-frame", default=None, help="optical TF frame of the camera (default: CameraInfo frame_id)")
    group.add_argument("--ee-frame", default=None, help="TF frame of the end-effector/flange: read at capture time")
    group.add_argument("--timeout", type=float, default=None, dest="timeout_s", help="seconds to wait for messages")
    group.add_argument("--depth-frames", type=int, default=None, help="depth frames for the median (default 5)")


ROS_OPTION_KEYS = ("color_topic", "depth_topic", "camera_info_topic", "world_frame", "camera_frame", "ee_frame",
                   "timeout_s", "depth_frames")


def source_kwargs(args) -> dict:
    return {"source": args.source, "ros": {k: getattr(args, k) for k in ROS_OPTION_KEYS}}


def run(*, image: Optional[str] = None, depth: Optional[str] = None,
        intrinsics: Optional[Sequence[float]] = None, plan_path: Optional[str] = None,
        config_path: Optional[str] = None, out_dir: Optional[str] = None,
        report_path: Optional[str] = None, source: str = "file", ros: Optional[dict] = None,
        live_frame: Optional[ros_camera.LiveFrame] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.REAL_CAPTURE, out),
                   inputs={"image": image, "depth": depth, "plan": plan_file if plan_file.exists() else None,
                           "config": config_path}, logger=log) as report:
        config = load_environment_config(config_path)
        live = None
        if source == "ros2":
            live = live_frame or ros_camera.acquire(ros_camera.resolve_options(config, ros))
            img = Image.fromarray(live.rgb)
            report.inputs["image"] = f"ros2:{live.options['color_topic']}"
            source_name = report.inputs["image"]
        else:
            source_path = find_image(image)
            report.inputs["image"] = str(source_path)
            img = Image.open(source_path).convert("RGB")
            source_name = str(source_path)
        w, h = img.size

        if plan_file.exists():
            tray_key = StageReport.load(plan_file, expect_stage="plan").require("source_tray")
        else:
            tray_key = DEFAULT_TRAY
            report.warn(log, f"plan report not found ({plan_file}): assuming source tray '{tray_key}'")
        tray = get_tray(config, tray_key)

        cam_cfg = config.get("camera_overhead", {})
        extr = cam_cfg.get("extrinsics", {})
        position = extr.get("position_world_frame_xyz_m",
                            extr.get("position_robot_frame_xyz_m", FALLBACK_CAMERA_POSITION))
        target = extr.get("target_position_xyz_m", [position[0], position[1], tray.top_z])

        # --- intrinsics: CameraInfo (live) or calibrated values if given, otherwise scaled from the config (nominal) ---
        base = camera_intrinsics(config)
        calibrated_intrinsics = intrinsics is not None or live is not None
        if live is not None:
            fx, fy, cx, cy = live.intrinsics.fx, live.intrinsics.fy, live.intrinsics.cx, live.intrinsics.cy
        elif intrinsics is not None:
            fx, fy, cx, cy = (float(v) for v in intrinsics)
        else:
            scale = w / base.width                       # same field of view as the configured sensor
            fx, fy, cx, cy = base.fx * scale, base.fy * scale, w / 2.0, h / 2.0
            report.warn(log, f"intrinsics are NOT calibrated: scaled from the config ({base.width}x{base.height} "
                             f"-> {w}x{h}); pass --intrinsics FX FY CX CY from the camera calibration")
        orientation = None
        if live is not None:
            if live.distortion_nonzero:
                report.warn(log, "CameraInfo reports lens distortion but the pipeline uses a pinhole model: use the "
                                 "RECTIFIED image (image_rect_color) or a driver that already undistorts")
            if live.camera_pose_world is not None:
                position, orientation = live.camera_pose_world
                z_axis = matrix_from_quaternion(orientation)[:, 2]          # asse ottico (avanti) nel world
                target = (np.asarray(position) + z_axis).tolist()
            else:
                report.warn(log, f"no TF {live.world_frame} -> {live.camera_frame}: using the NOMINAL camera pose "
                                 "from the YAML; publish the hand-eye calibration as a TF")
        camera = CameraModel(list(position), list(target), Intrinsics(w, h, fx, fy, cx, cy), orientation)
        extrinsics_verified = bool(extr.get("verified", False))
        if not extrinsics_verified:
            report.warn(log, "camera extrinsics are NOT verified (camera_overhead.extrinsics.verified is false): "
                             "hand-eye calibration required before robot motion")
        elif live is not None and live.camera_pose_world is None:
            extrinsics_verified = False
            report.warn(log, "extrinsics.verified is true but the camera pose is not read from the TF: not verified")

        # --- depth: measured if given, otherwise a nominal plane on the tray SURFACE ---------------
        measured_depth = depth is not None or live is not None
        nominal_distance = float(position[2] - tray.top_z)
        if live is not None:
            depth_map = live.depth_m
            depth_source = "measured_live_ros2"
            if live.depth_valid_fraction < 0.5:
                report.warn(log, f"only {live.depth_valid_fraction:.0%} of the depth pixels are valid: check the sensor")
        elif measured_depth:
            depth_map = load_depth(Path(depth), (h, w))
            depth_source = "measured"
        else:
            depth_map = np.full((h, w), nominal_distance, dtype=np.float32)
            depth_source = "nominal_constant_plane_from_photo"
            report.warn(log, "no measured depth: using a constant plane on the tray surface; the instrument "
                             "height is unknown, so this result is NOT cleared for robot motion")

        rgb_png, depth_npy = out / "real_capture_rgb.png", out / "real_capture_depth.npy"
        img.save(rgb_png)
        np.save(depth_npy, depth_map)

        motion_safe = measured_depth and calibrated_intrinsics and extrinsics_verified
        report.artifacts = {"rgb_png": str(rgb_png), "depth_npy": str(depth_npy)}
        report.data = {"source_file": source_name,
                       "source_type": "live_ros2_rgbd" if live is not None else "real_surgical_tray_image",
                       "source_tray": tray.key, "camera": camera.to_dict(), "depth_source": depth_source,
                       "intrinsics_calibrated": calibrated_intrinsics, "extrinsics_verified": extrinsics_verified,
                       "nominal_distance_m": round(nominal_distance, 4), "motion_safe": motion_safe}
        if live is not None:
            report.data["acquisition"] = live.acquisition_info()
            if live.ee_pose_world is not None:
                report.data["ee_reference_pose"] = {
                    "position_xyz_m": [round(v, 5) for v in live.ee_pose_world[0]],
                    "quaternion_xyzw": [round(v, 6) for v in live.ee_pose_world[1]],
                    "frame_id": live.world_frame, "child_frame": live.options["ee_frame"], "source": "tf"}
        log.info("captured %s (%dx%d), depth %s, motion_safe=%s", source_name, w, h, depth_source, motion_safe)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--image", default=None, help="Tray photo (default: data/inputs/surgical_scene.jpg)")
    parser.add_argument("--depth", default=None, help="Metric depth aligned with the image: .npy (m) or 16-bit PNG (mm)")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"), default=None,
                        help="Calibrated intrinsics in pixels of --image")
    parser.add_argument("--plan", default=None, help="plan.json (source tray); default <out-dir>/plan.json")
    add_source_arguments(parser)
    return run_cli(parser, lambda a: run(image=a.image, depth=a.depth, intrinsics=a.intrinsics, plan_path=a.plan,
                                         config_path=a.config, out_dir=a.out_dir, report_path=a.report,
                                         **source_kwargs(a)), argv)


if __name__ == "__main__":
    raise SystemExit(main())
