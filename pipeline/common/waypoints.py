"""Pick-and-place TCP targets shared by the motion preview and the grasp feasibility check.

Both consumers must walk EXACTLY the same path, otherwise a grasp validated offline could differ from the
one that is executed. All positions are in the WORLD frame (heights come from ``motion_control.heights_z_m``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

APPROACH_MIN_ABOVE_GRASP_M = 0.12     # approach point is at least this high above the grasp point
LIFT_MIN_ABOVE_GRASP_M = 0.18         # carrying height is at least this high above the grasp point
PLACE_CLEARANCE_M = 0.003             # tool bottom above the destination tray at release
RETURN_BASE_XYZ = (0.48, 0.0, 0.40)   # final retreat point, expressed in the robot base frame

# Labels are shared by the GUI and new motion reports; historical reports retain their labels.
APPROACH, DESCEND, GRASP, LIFT, TRANSFER, PLACE, RELEASE, RETREAT, RETURN = (
    "Approach", "Descend", "Close gripper - ideal constraint", "Lift", "Transfer",
    "Place", "Open gripper", "Retreat", "Return")


@dataclass(frozen=True)
class Target:
    label: str
    position: np.ndarray                 # world frame, metres
    event: Optional[str] = None          # None | "attach" | "release"


def pick_and_place_targets(point, *, destination_tray: Dict[str, Any], tool_bottom_z: float,
                           motion: Dict[str, Any], base_frame) -> List[Target]:
    """Ordered targets for one pick-and-place, from the grasp point (world) to the final retreat.

    ``tool_bottom_z`` is the height of the tool's underside while it rests on the source tray: the TCP height
    above it is preserved at the destination so that the tool is released ``PLACE_CLEARANCE_M`` above the tray.
    """
    point = np.asarray(point, dtype=float)
    heights = (motion or {}).get("heights_z_m", {})
    lift = max(heights.get("lift_clearance_z", 0.35), point[2] + LIFT_MIN_ABOVE_GRASP_M)
    approach = max(heights.get("approach_z", 0.32), point[2] + APPROACH_MIN_ABOVE_GRASP_M)
    place = np.asarray(destination_tray["position_xyz_m"], dtype=float).copy()
    place[2] += destination_tray["dimensions_xyz_m"][2] / 2 + PLACE_CLEARANCE_M + (point[2] - tool_bottom_z)
    return [
        Target(APPROACH, np.array([point[0], point[1], approach])),
        Target(DESCEND, point.copy()),
        Target(GRASP, point.copy(), "attach"),
        Target(LIFT, np.array([point[0], point[1], lift])),
        Target(TRANSFER, np.array([place[0], place[1], lift])),
        Target(PLACE, place.copy()),
        Target(RELEASE, place.copy(), "release"),
        Target(RETREAT, np.array([place[0], place[1], lift])),
        Target(RETURN, np.asarray(base_frame.base_to_world(list(RETURN_BASE_XYZ)), dtype=float)),
    ]
