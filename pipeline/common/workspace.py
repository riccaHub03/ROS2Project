"""Reachable workspace of the manipulator for top-down grasps (single source of truth).

Measured with an IK scan of the PyBullet KUKA iiwa model, gripper pointing down and a 0.10 m
TCP offset: the TCP can reach a horizontal annulus around the robot base. Values can be
overridden in ``surgical_environment.yaml`` (``motion_control.workspace``) and MUST be
re-measured with the real robot URDF, mounting and gripper before any motion on hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_R_MAX_M = 0.78       # largest horizontal TCP distance from the base axis
DEFAULT_R_MIN_M = 0.41       # smallest distance, enforced only below ``low_z_m``
DEFAULT_LOW_Z_M = 0.15       # grasp heights (close to the table) use the minimum radius
GRASP_MARGIN_M = 0.04        # extra safety kept from the limits when choosing grasp points


@dataclass(frozen=True)
class Workspace:
    r_min: float = DEFAULT_R_MIN_M
    r_max: float = DEFAULT_R_MAX_M
    low_z: float = DEFAULT_LOW_Z_M

    def radius(self, point: Sequence[float]) -> float:
        return math.hypot(point[0], point[1])

    def contains(self, point: Sequence[float], margin: float = 0.0) -> bool:
        r = self.radius(point)
        r_min = self.r_min + margin if point[2] < self.low_z else 0.0
        return r_min <= r <= self.r_max - margin

    def blockers(self, point: Sequence[float], margin: float = 0.0) -> List[str]:
        """Human readable reasons why ``point`` is not reachable (empty if it is)."""
        if self.contains(point, margin):
            return []
        r = self.radius(point)
        return [f"grasp point at {r:.3f} m from the robot base is outside the reachable annulus "
                f"[{self.r_min + margin:.2f}, {self.r_max - margin:.2f}] m"]


def from_config(config: Optional[Dict[str, Any]]) -> Workspace:
    ws = (config or {}).get("motion_control", {}).get("workspace", {}) or {}
    return Workspace(float(ws.get("reach_r_min_m", DEFAULT_R_MIN_M)),
                     float(ws.get("reach_r_max_m", DEFAULT_R_MAX_M)),
                     float(ws.get("low_z_m", DEFAULT_LOW_Z_M)))
