"""Sequenza di pick & place a waypoint cartesiani, calcolata dal report di presa e dalla configurazione.

Logica pura (niente ROS, niente PyBullet): il nodo ROS 2 ``surgical_executor`` si limita a inviare le
sequenze e a comandare la pinza. Tutte le coordinate sono nel riferimento della base del robot (``lbr_link_0``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

READY = "VALIDATED_GEOMETRY_READY_FOR_TRAJECTORY_PLAN"

Waypoint = Tuple[List[float], List[float]]      # (xyz [m], quaternione xyzw)


class MotionRefused(ValueError):
    """La presa non e' validata per il movimento: nessun comando va inviato al robot."""


@dataclass
class MacroSequence:
    name: str                                    # etichetta per il log
    waypoints: List[Waypoint]
    gripper_after: Optional[str] = None          # "grasp" | "release" | None


@dataclass
class PickAndPlacePlan:
    destination_tray: str
    grasp_xyz: List[float]
    deposit_xyz: List[float]
    sequences: List[MacroSequence] = field(default_factory=list)


def resolve_tray(destination: str, trays: Dict[str, Any]) -> str:
    if destination in trays:
        return destination
    raise MotionRefused(f"vassoio di destinazione '{destination}' non definito nella configurazione "
                        f"(disponibili: {', '.join(trays) or 'nessuno'})")


def grasp_pose_from_report(grasp_report: Optional[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """Centroide e quaternione di presa (frame base) da ``grasp.json``; rifiuta se non validata."""
    if not grasp_report or grasp_report.get("readiness") != READY:
        reasons = (grasp_report or {}).get("readiness_blockers") or ["nessun report di presa fornito"]
        raise MotionRefused("presa non validata: " + "; ".join(reasons))
    geometry = grasp_report["designated_grasp_geometry"]
    return (list(geometry["grasp_centroid_base_frame_xyz_m"]),
            list(geometry["recommended_gripper_orientation"]["quaternion_xyzw"]))


def plan_pick_and_place(grasp_xyz: Sequence[float], grasp_quat: Optional[Sequence[float]],
                        destination_tray: str, config: Dict[str, Any]) -> PickAndPlacePlan:
    trays = config["environment"]["trays"]
    tray_key = resolve_tray(destination_tray, trays)
    tray = trays[tray_key]
    motion = config["motion_control"]
    heights = motion["heights_z_m"]
    down = list(motion["downward_quaternion_xyzw"])
    tool = list(grasp_quat) if grasp_quat is not None else down

    p_tray = tray["position_xyz_m"]
    deposit_z = p_tray[2] + tray["dimensions_xyz_m"][2] / 2.0 + 0.03
    grasp_z = max(float(grasp_xyz[2]), 0.12)
    grasp = [float(grasp_xyz[0]), float(grasp_xyz[1]), grasp_z]

    stage = [0.55, grasp[1] * 0.5, float(heights["staging_z"])]
    approach = [grasp[0], grasp[1], float(heights["approach_z"])]
    lift = [grasp[0], grasp[1], float(heights["lift_clearance_z"])]
    tray_above = [p_tray[0], p_tray[1], float(heights["lift_clearance_z"])]
    deposit = [p_tray[0], p_tray[1], deposit_z]
    home = [0.55, 0.0, 0.50]

    return PickAndPlacePlan(
        destination_tray=tray_key, grasp_xyz=grasp, deposit_xyz=deposit,
        sequences=[
            MacroSequence("Discesa e presa", [(stage, down), (approach, tool), (grasp, tool)], "grasp"),
            MacroSequence(f"Trasferimento verso {tray_key}",
                          [(lift, tool), (tray_above, down), (deposit, down)], "release"),
            MacroSequence("Rientro in standby", [(tray_above, down), (home, down)], None),
        ])
