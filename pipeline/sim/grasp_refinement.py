"""Bounded, explicitly reported geometric refinement of a GraspGen proposal.

The original model score is not a score of the refined pose. This checks CAD
clearance and opposing pad proximity only, never physical force closure.
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from pipeline.frame_transformer import quaternion_from_matrix

TCP = np.array([[0., 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, .195], [0, 0, 0, 1]])


def refine_grasp(cell, original, support_trays, heartbeat=lambda: None):
    """Search <=40 mm upwards and <=15 mm along closing axis; rotation fixed.

    Closing width is found from actual articulated pad/target distances. A pose
    must admit the entire opening-to-closing sweep, both opposing pads within
    1.5 mm of the target and no pad penetration, and all environment margins.
    """
    gripper = cell.official_gripper
    pads = sorted(gripper.pad_links, key=lambda i: gripper.links[i])
    if len(pads) != 2:
        raise ValueError("Refinement requires two named opposing Robotiq pads")
    up = cell.base_frame.rotation.T @ np.array([0., 0., 1.])
    last_error = "no geometrically feasible closing sweep"
    history = []

    def measure(transform, width, *, check_environment=True):
        heartbeat()
        tcp = transform @ TCP
        gripper.place(transform[:3, 3].tolist(), quaternion_from_matrix(tcp[:3, :3]), float(width))
        if check_environment:
            for name, body in cell.obstacles.items():
                _, issues = gripper.obstacle_clearance(body, .010, .0005, name in support_trays)
                if issues:
                    raise ValueError(f"{issues[0]} <-> {name} at opening {width:.5f} m")
        distances = np.full(2, .2)
        for contact in p.getClosestPoints(gripper.body, cell._tool_body, .2, physicsClientId=cell.cid):
            if contact[3] in pads:
                i = pads.index(contact[3])
                distances[i] = min(distances[i], contact[8])
            elif contact[8] < .0005:
                raise ValueError(f"Non-pad link {gripper.links[contact[3]]} collides with target")
        return distances

    for height in (0., .005, .010, .015, .020, .025, .030, .035, .040):
        transform = np.asarray(original, dtype=float).copy()
        transform[:3, 3] += height*up
        lateral = 0.
        try:
            for centering in range(3):
                upper = .136
                gaps = measure(transform, upper)
                if gaps.min() < 0:
                    raise ValueError("Target penetrates pads at maximum opening")
                found = False
                for lower in np.linspace(.136, .004, 45)[1:]:
                    lower_gaps = measure(transform, lower)
                    if lower_gaps.min() <= .0002:
                        # First contact, approached from the collision-free side.
                        for _ in range(9):
                            mid = (upper+lower)/2
                            middle_gaps = measure(transform, mid)
                            if middle_gaps.min() < .0002:
                                lower = mid
                            else:
                                upper = mid
                        gaps = measure(transform, upper)
                        found = True
                        break
                    upper, gaps = lower, lower_gaps
                if not found:
                    raise ValueError("Pads do not reach target within opening range")
                if gaps.max() <= .0015 and gaps.min() >= 0:
                    # A denser final sweep is also checked before robot planning.
                    for width in np.linspace(.136, upper, 70):
                        if measure(transform, width).min() < 0:
                            raise ValueError("Pad penetration during closing sweep")
                    return transform, float(upper), {
                        "method": "bounded_upward_and_closing_axis_CAD_refinement",
                        "original_T_base_gripper": np.asarray(original).tolist(),
                        "translation_base_m": (transform[:3, 3]-np.asarray(original)[:3, 3]).tolist(),
                        "upward_m": height, "closing_axis_shift_m": lateral,
                        "opposing_pad_gaps_m": gaps.tolist(), "closing_width_m": float(upper),
                        "model_rescored": False, "physical_grasp_validated": False,
                        "attempts": history}
                # Left pad lies on +X in the pinned model. Centre the jaw pair
                # between the closest opposing surfaces; never move >15 mm.
                shift = float(-(gaps[0]-gaps[1])/2)
                if abs(lateral+shift) > .015:
                    raise ValueError("Bilateral contact requires >15 mm closing-axis correction")
                transform[:3, 3] += transform[:3, 0]*shift
                lateral += shift
            raise ValueError("Opposing pads cannot both approach target within 1.5 mm")
        except ValueError as exc:
            last_error = str(exc)
            history.append({"upward_m": height, "reason": last_error})
    raise ValueError("No collision-free bilateral closing geometry: " + last_error)
