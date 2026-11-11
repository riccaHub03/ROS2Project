"""Typed accessors over the environment config (trays, table, camera, motion)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from pipeline.common.report import StageError

Config = Dict[str, Any]


class ConfigError(StageError):
    def __init__(self, message: str):
        super().__init__("config", message)


@dataclass(frozen=True)
class Tray:
    key: str
    position: List[float]
    dimensions: List[float]
    name: str = ""

    @property
    def top_z(self) -> float:
        """Z of the tray's upper face."""
        return self.position[2] + self.dimensions[2] / 2.0

    def contains_xy(self, point: Sequence[float], margin: float = 0.0) -> bool:
        return all(abs(point[i] - self.position[i]) <= self.dimensions[i] / 2.0 + margin for i in (0, 1))


@dataclass(frozen=True)
class CameraPose:
    position: List[float]
    target: List[float]


def get_tray(config: Config, key: str) -> Tray:
    trays = config.get("environment", {}).get("trays", {})
    if key not in trays:
        raise ConfigError(f"tray '{key}' is not defined in the config (available: {sorted(trays)})")
    info = trays[key]
    return Tray(key=key, position=list(info["position_xyz_m"]), dimensions=list(info["dimensions_xyz_m"]),
                name=info.get("name", key))


def all_trays(config: Config) -> List[Tray]:
    return [get_tray(config, key) for key in config.get("environment", {}).get("trays", {})]


def camera_pose(config: Config, source_tray: Tray) -> CameraPose:
    """Overhead camera pose: tracks the source tray unless a fixed pose is configured."""
    cam = config.get("camera_overhead", {})
    extr = cam.get("extrinsics", {})
    fixed_pos, fixed_target = extr.get("position_robot_frame_xyz_m"), extr.get("target_position_xyz_m")
    tray_pos = list(source_tray.position)
    if cam.get("auto_track_source_tray", True) or not fixed_pos or fixed_pos == "auto":
        height = cam.get("height_above_tray_m", 1.08)
        return CameraPose([tray_pos[0], tray_pos[1], tray_pos[2] + height], tray_pos)
    target = fixed_target if fixed_target and fixed_target != "auto" else tray_pos
    return CameraPose(list(fixed_pos), list(target))


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def camera_intrinsics(config: Config) -> Intrinsics:
    cam = config.get("camera_overhead", {})
    intr = cam.get("intrinsics", {})
    w, h = int(cam.get("width_px", 640)), int(cam.get("height_px", 480))
    return Intrinsics(w, h, float(intr.get("fx_px", 615.0)), float(intr.get("fy_px", 615.0)),
                      float(intr.get("cx_px", w / 2.0)), float(intr.get("cy_px", h / 2.0)))


def ee_reference(config: Config) -> tuple[List[float], List[float]]:
    """End-effector pose (position, quaternion xyzw) used to express points in the EE frame.

    Taken from ``motion_control.ee_reference_pose`` when configured (on the real robot: the
    TF lbr_link_0 -> flange/TCP read at capture time). Otherwise a nominal staging pose is
    used, which is only meaningful in simulation (see :func:`ee_reference_source`).
    """
    motion = config.get("motion_control", {})
    configured = motion.get("ee_reference_pose") or {}
    staging_z = motion.get("heights_z_m", {}).get("staging_z", 0.45)
    position = configured.get("position_xyz_m") or [0.45, 0.0, staging_z]
    quaternion = (configured.get("quaternion_xyzw") or motion.get("downward_quaternion_xyzw")
                  or [0.0, 1.0, 0.0, 0.0])
    return list(position), list(quaternion)


def ee_reference_source(config: Config) -> str:
    """'configured' if the EE pose comes from the config, 'nominal' if it is the built-in default."""
    configured = config.get("motion_control", {}).get("ee_reference_pose") or {}
    return "configured" if configured.get("position_xyz_m") else "nominal"


def round_list(values: Sequence[float], digits: int = 4) -> List[float]:
    return [round(float(v), digits) for v in np.asarray(values).ravel()]
