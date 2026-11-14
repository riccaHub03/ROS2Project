"""Pure contracts shared by the ROS coordinator, simulation server and tests.

Only the next phase of the in-memory, scene-validated plan may be executed.
This is a simulation adapter, never an authorization to move physical hardware.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from pipeline.robot_model import COMMAND_JOINT_NAMES

JOINT_NAMES = list(COMMAND_JOINT_NAMES)
DT = 1 / 240


def phase_kind(phase):
    return "gripper" if phase["event"] else "arm"


def duration_ns(index):
    return round(index * DT * 1_000_000_000)


def validate_trajectory(names, points, times_ns, phase):
    if phase_kind(phase) != "arm":
        raise ValueError("The next validated phase is a gripper command")
    if list(names) != JOINT_NAMES:
        raise ValueError("Joint names/order must be lbr_A1 through lbr_A7")
    actual = np.asarray(points, dtype=float)
    expected = np.asarray(phase["samples"], dtype=float)
    if (actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 7
            or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, atol=1e-10, rtol=0)):
        raise ValueError("Trajectory differs from the current scene-validated phase")
    if list(times_ns) != [duration_ns(i) for i in range(len(expected))]:
        raise ValueError("Trajectory timing differs from the validated 240 Hz samples")


def validate_gripper(position, effort, phase):
    if phase_kind(phase) != "gripper":
        raise ValueError("The next validated phase is an arm trajectory")
    if (not math.isfinite(position) or not math.isfinite(effort) or effort != 0
            or abs(position - phase["end_width"]) > 1e-9):
        raise ValueError("Requires the validated jaw opening and max_effort=0 (ideal simulation)")


def verify_initial_joints(actual, expected, tolerance=.03):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != (7,) or not np.isfinite(actual).all()
            or np.max(np.abs(actual - expected)) > tolerance):
        raise ValueError("Measured start joints differ from the validated phase; replan")


def phase_manifest(phases, out, run_id):
    file = Path(out) / "graspgen_sim_joint_path.npz"
    return dict(run_id=run_id, execution_scope="pybullet_only", joint_names=JOINT_NAMES,
                sample_period_s=DT, trajectory_file=file.name,
                trajectory_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                phases=[dict(index=i, label=p["label"], kind=phase_kind(p), event=p["event"],
                             samples_key=f"phase_{i}", duration_s=p["duration"],
                             start_width=p["start_width"], end_width=p["end_width"])
                        for i, p in enumerate(phases)])


def load_phase_samples(out, plan):
    out = Path(out).resolve()
    file = (out / plan["trajectory_file"]).resolve()
    if (file.parent != out or file.name != "graspgen_sim_joint_path.npz"
            or hashlib.sha256(file.read_bytes()).hexdigest() != plan["trajectory_sha256"]
            or plan["run_id"] != out.name or plan["execution_scope"] != "pybullet_only"):
        raise ValueError("Trajectory provenance/run mismatch")
    with np.load(file, allow_pickle=False) as arrays:
        return [arrays[p["samples_key"]].copy() for p in plan["phases"]]


def ingest_rgbd(rgb, depth, info, capture, out):
    """Use the pixels actually received over DDS, with an exact acquisition match."""
    from PIL import Image
    from pipeline.common.report import write_json

    def stamp(msg):
        return (msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id)
    camera = capture["data"]["camera"]
    w, h = camera["resolution_wh"]
    if (stamp(rgb) != stamp(depth) or stamp(rgb) != stamp(info)
            or not stamp(rgb)[2] or stamp(rgb)[:2] == (0, 0)
            or (rgb.width, rgb.height) != (w, h) or (depth.width, depth.height) != (w, h)
            or (info.width, info.height) != (w, h)
            or rgb.encoding != "rgb8" or depth.encoding != "32FC1"
            or rgb.step != w * 3 or depth.step != w * 4 or depth.is_bigendian
            or len(rgb.data) != h * w * 3 or len(depth.data) != h * w * 4):
        raise ValueError("RGB, depth and CameraInfo must describe the same aligned acquisition")
    fx, fy = camera["fx_fy"]
    cx, cy = camera["cx_cy"]
    if not np.allclose(info.k, [fx, 0, cx, 0, fy, cy, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("ROS CameraInfo differs from the captured camera")
    out = Path(out)
    colors = np.frombuffer(bytes(rgb.data), dtype=np.uint8).reshape(h, w, 3)
    depths = np.frombuffer(bytes(depth.data), dtype="<f4").reshape(h, w)
    # Verify identity with the scene's original snapshot before replacing artifacts
    # with DDS-delivered pixels. Truth geometry is retained for diagnostics only.
    if (not np.array_equal(colors, np.asarray(Image.open(out / "sim_capture_rgb.png")))
            or not np.array_equal(depths, np.load(out / "sim_capture_depth.npy"), equal_nan=True)):
        raise ValueError("Received RGB-D does not match this run's scene snapshot")
    Image.fromarray(colors).save(out / "sim_capture_rgb.png")
    np.save(out / "sim_capture_depth.npy", depths)
    capture["data"]["transport"] = dict(rgbd="ROS2 sensor_msgs", stamp_sec=stamp(rgb)[0],
                                         stamp_nanosec=stamp(rgb)[1], optical_frame=stamp(rgb)[2])
    write_json(out / "sim_capture.json", capture)
    return capture["data"]["transport"]
