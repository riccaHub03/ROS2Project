"""Grasp candidates and their feasibility validation: contact region, orientation, IK/joint limits and collisions.

A grasp proposal is only useful if the arm can execute it. Every candidate (position, orientation, jaw opening,
approach direction in the robot base frame) is therefore walked through the SAME pick-and-place path the motion
preview executes (``pipeline.common.waypoints``) in a kinematic model of the cell (``pipeline.robot_cell``):

  (i)   contact region and jaw width          opening <= stroke, usable pinch depth on the instrument
  (ii)  orientation and TCP convention        unit quaternion, approach direction pointing down
  (iii) IK and joint limits                   at the pregrasp and along approach, lift, transfer and place
  (iv)  collisions                            arm, gripper and carried instrument against table, trays and itself

A valid IK solution alone does not make a grasp feasible: the first IK branch that solves a waypoint may drive
the arm through the tray, so several starting configurations are tried before a candidate is rejected.
The best FEASIBLE candidate (quality x robustness) is selected. This is a kinematic pre-filter in a model of the
cell, not a proof for the real robot: see ``LIMITATIONS``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.common import waypoints as wp
from pipeline.common.scene import round_list

log = logging.getLogger("feasibility")

DEFAULT_PARAMS: Dict[str, Any] = {
    "enabled": True,                    # run the check (otherwise geometry-only candidates are returned)
    "require_feasibility": True,        # no verified candidate -> the grasp is NOT_READY_FOR_ROBOT_MOTION
    "positions_along_zone": 5,          # candidates along the grasp zone of the instrument
    "both_jaw_orientations": True,      # also with the jaws reversed (180 deg about the approach axis)
    "pinch_target_m": 0.006,            # how far below the instrument top the fingertips should reach
    "min_pinch_m": 0.002,               # shallower than this the grip is marginal (warning, lowers the quality)
    "tip_clearance_m": 0.001,           # fingertips never go closer than this to the supporting tray
    "collision_margin_m": 0.010,        # minimum separation from table, trays (and between arm links)
    "contact_margin_m": 0.0005,         # fingertips/instrument just above the tray at grasp and release
    "path_step_m": 0.03,                # sampling of the vertical segments
    "transfer_step_m": 0.06,            # sampling of the horizontal transfer
    "max_joint_step_rad": 0.6,          # larger jumps between samples mean an arm configuration flip
    "min_joint_margin_rad": 0.03,       # keep this far from every joint limit
    "ik_position_tolerance_m": 0.003,
    "ik_angle_tolerance_rad": 0.035,
    "max_approach_tilt_deg": 30.0,
    "max_instrument_tilt_deg": 45.0,    # a top-down pinch needs an instrument that lies (roughly) flat
    "robot_urdf": None,                 # default: PyBullet kuka_iiwa (LBR iiwa 14 geometry); set the iiwa 7 URDF
}
LIMITATIONS = [
    "Kinematic pre-filter in a PyBullet model of the cell: it does not replace MoveIt / the robot controller.",
    "Default arm model is PyBullet's kuka_iiwa (LBR iiwa 14 geometry), not the iiwa 7 of the laboratory.",
    "Gripper and carried instrument are boxes; the demonstrative gripper is NOT the laboratory gripper.",
    "Only table and trays from the configuration are obstacles (solid blocks); other instruments are not modelled.",
    "Physical grasp stability (friction, forces, slip) is not evaluated.",
]
OPEN_WIDTH_M = 0.080
CLEARANCE_M = 0.020                     # extra jaw opening around the instrument (same as grasp.CLEARANCE_M)
TOOL_REST_HEIGHT_M = 0.001              # instrument underside above the tray (RobotScene.tool_bottom convention)
QUALITY_WEIGHTS = {"centrality": 0.35, "width_margin": 0.20, "support": 0.20, "pinch": 0.25}


def load_params(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_PARAMS)
    merged.update((config or {}).get("grasp_validation") or {})
    return merged


@dataclass
class Candidate:
    """One grasp proposal, robot base frame (g_i = position, orientation, opening, approach)."""
    id: int
    rel_position: float                 # position along the instrument main axis (0..1)
    jaws_reversed: bool
    position: np.ndarray                # TCP at the grasp
    quaternion: np.ndarray              # xyzw
    approach: np.ndarray                # unit vector, direction of motion of the gripper
    closing: np.ndarray                 # unit vector, jaw closing direction
    opening: float                      # required jaw opening (instrument width + clearance)
    local_width: float
    local_thickness: float
    pinch_depth: float                  # how deep the fingertips reach below the instrument top
    support_points: int
    quality: float
    quality_terms: Dict[str, float] = field(default_factory=dict)
    tilt_deg: float = 0.0               # how far the instrument is from lying flat (PCA normal vs vertical)


@dataclass
class ValidationContext:
    config: Dict[str, Any]
    base_frame: Any
    source_key: str
    destination_key: str
    tool_box: Dict[str, Any]            # half_extents, position (base), quaternion (base)
    support_top_world_z: float
    max_stroke: float
    params: Dict[str, Any]

    @property
    def tool_bottom_z(self) -> float:
        return self.support_top_world_z + TOOL_REST_HEIGHT_M

    @property
    def destination_tray(self) -> Dict[str, Any]:
        return self.config["environment"]["trays"][self.destination_key]


@dataclass
class FeasibilityResult:
    candidates: List[Dict[str, Any]]    # ranked: feasible first, then by score
    selected_id: Optional[int]
    summary: Dict[str, Any]


# --------------------------------------------------------------------------- scoring
def quality_score(rel_position: float, opening: float, max_stroke: float, support_points: int, support_ref: float,
                  pinch_depth: float, pinch_target: float) -> Tuple[float, Dict[str, float]]:
    """Heuristic geometric quality in [0, 1] (NOT a learned grasp-quality network)."""
    terms = {
        "centrality": float(np.clip(1.0 - abs(rel_position - 0.5) / 0.15, 0.0, 1.0)),
        "width_margin": float(np.clip(1.0 - opening / max_stroke, 0.0, 1.0)),
        "support": float(np.clip(support_points / max(support_ref, 1.0), 0.0, 1.0)),
        "pinch": float(np.clip(pinch_depth / max(pinch_target, 1e-6), 0.0, 1.0)),
    }
    return float(sum(QUALITY_WEIGHTS[k] * v for k, v in terms.items())), {k: round(v, 3) for k, v in terms.items()}


# --------------------------------------------------------------------------- the checks
def _quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def check_contact(cand: Candidate, ctx: ValidationContext) -> Dict[str, Any]:
    """(i) jaw opening against the stroke and usable pinch depth on the instrument."""
    issues: List[str] = []
    notes: List[str] = []
    if cand.local_width < 0.002:
        issues.append(f"degenerate cross-section ({cand.local_width * 1000:.1f} mm wide)")
    if cand.opening > ctx.max_stroke:
        issues.append(f"required opening {cand.opening * 1000:.0f} mm exceeds the gripper stroke "
                      f"{ctx.max_stroke * 1000:.0f} mm")
    if cand.tilt_deg > ctx.params["max_instrument_tilt_deg"]:
        issues.append(f"the instrument is tilted {cand.tilt_deg:.0f} deg from lying flat "
                      f"(> {ctx.params['max_instrument_tilt_deg']:.0f}): a top-down pinch is not suitable")
    if cand.pinch_depth < ctx.params["min_pinch_m"]:
        notes.append(f"shallow pinch ({cand.pinch_depth * 1000:.1f} mm below the top): thin instrument, "
                     "the grip relies on the very edge of the jaws")
    return {"passed": not issues, "required_opening_m": round(cand.opening, 4),
            "max_stroke_m": ctx.max_stroke, "pinch_depth_m": round(cand.pinch_depth, 4),
            "instrument_tilt_deg": round(cand.tilt_deg, 1),
            "support_points": cand.support_points, "issues": issues, "notes": notes}


def check_orientation(cand: Candidate, ctx: ValidationContext) -> Dict[str, Any]:
    """(ii) unit quaternion, proper rotation and approach direction pointing down (TCP z = approach)."""
    rotation = _quat_to_matrix(cand.quaternion)
    norm = float(np.linalg.norm(cand.quaternion))
    proper = abs(float(np.linalg.det(rotation)) - 1.0) < 1e-6 and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    z_axis_matches = bool(np.allclose(rotation[:, 2], cand.approach, atol=1e-3))
    approach_world = np.asarray(ctx.base_frame.rotation) @ cand.approach
    tilt = math.degrees(math.acos(float(np.clip(-approach_world[2], -1.0, 1.0))))
    issues = []
    if abs(norm - 1.0) > 1e-3 or not proper:
        issues.append("orientation is not a proper unit rotation")
    if not z_axis_matches:
        issues.append("TCP z axis differs from the approach direction (frame convention mismatch)")
    if tilt > ctx.params["max_approach_tilt_deg"]:
        issues.append(f"approach tilted {tilt:.0f} deg from vertical (> {ctx.params['max_approach_tilt_deg']:.0f})")
    return {"passed": not issues, "quaternion_norm": round(norm, 6), "approach_tilt_deg": round(tilt, 2),
            "tcp_z_matches_approach": z_axis_matches, "issues": issues}


def ik_seeds(target_base: Sequence[float], home: Sequence[float]) -> List[List[float]]:
    """Starting configurations: the first IK branch is often wrong (wrist or elbow on the other side)."""
    mirrored = list(home)
    mirrored[0] = math.atan2(target_base[1], target_base[0])
    seeds = [list(home), mirrored]
    for joint, value in ((2, 0.8), (2, -0.8), (6, 1.5), (6, -1.5)):
        variant = list(mirrored)
        variant[joint] = value
        seeds.append(variant)
    return seeds


def _segment(a: np.ndarray, b: np.ndarray, step: float) -> List[np.ndarray]:
    """Points from ``a`` (excluded) to ``b`` (included), at most ``step`` apart."""
    count = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
    return [a + (b - a) * (i / count) for i in range(1, count + 1)]


@dataclass
class _Sample:
    phase: str
    position: np.ndarray                # base frame
    width: float
    carried: bool
    relaxed: FrozenSet[Tuple[str, str]] = frozenset()
    attach_here: bool = False           # the instrument is picked up at this sample


def plan_samples(cand: Candidate, ctx: ValidationContext) -> List[_Sample]:
    """The sampled pick-and-place path of one candidate (base frame), mirroring the motion preview."""
    base, params = ctx.base_frame, ctx.params
    point_world = np.asarray(base.base_to_world(cand.position), dtype=float)
    targets = {t.label: t for t in wp.pick_and_place_targets(
        point_world, destination_tray=ctx.destination_tray, tool_bottom_z=ctx.tool_bottom_z,
        motion=ctx.config.get("motion_control", {}), base_frame=base)}

    def at(label: str) -> np.ndarray:
        return np.asarray(base.world_to_base(targets[label].position), dtype=float)

    approach, grasp, lift, transfer, place = (at(k) for k in (wp.APPROACH, wp.DESCEND, wp.LIFT, wp.TRANSFER, wp.PLACE))
    hold = max(0.004, cand.local_width)
    step, long_step = params["path_step_m"], params["transfer_step_m"]
    source_pair = frozenset({("fingers", ctx.source_key)})
    place_pair = frozenset({("fingers", ctx.destination_key), ("tool", ctx.destination_key)})

    samples = [_Sample(wp.APPROACH, approach, OPEN_WIDTH_M, False)]
    descent = _segment(approach, grasp, step)
    samples += [_Sample(wp.DESCEND, pos, OPEN_WIDTH_M, False) for pos in descent[:-1]]
    samples.append(_Sample(wp.DESCEND, descent[-1], hold, False, source_pair, attach_here=True))
    samples += [_Sample(wp.LIFT, pos, hold, True) for pos in _segment(grasp, lift, step)]
    samples += [_Sample(wp.TRANSFER, pos, hold, True) for pos in _segment(lift, transfer, long_step)]
    lowering = _segment(transfer, place, step)
    samples += [_Sample(wp.PLACE, pos, hold, True) for pos in lowering[:-1]]
    samples.append(_Sample(wp.PLACE, lowering[-1], hold, True, place_pair))
    return samples


@dataclass
class _Attempt:
    ok: bool
    progress: int
    failure: Optional[Dict[str, Any]] = None
    phases: List[Dict[str, Any]] = field(default_factory=list)
    totals: Dict[str, float] = field(default_factory=dict)


def _walk(cell, cand: Candidate, ctx: ValidationContext, samples: List[_Sample], seed: Sequence[float]) -> _Attempt:
    params = ctx.params
    margin, contact = params["collision_margin_m"], params["contact_margin_m"]
    q_prev = list(seed)
    phases: Dict[str, Dict[str, Any]] = {}
    worst = {"ik_error_mm": 0.0, "joint_margin_rad": math.inf, "joint_step_rad": 0.0, "clearance_mm": math.inf}
    cell.detach_tool()
    for index, sample in enumerate(samples):
        def fail(kind: str, detail: str) -> _Attempt:
            return _Attempt(False, index, {"phase": sample.phase, "sample": index, "kind": kind, "detail": detail})

        solution = cell.solve_ik(sample.position, cand.quaternion, q_prev)
        if not solution.converged or not np.isfinite(solution.position_error_m):
            return fail("ik", f"IK does not reach '{sample.phase}' ({solution.position_error_m * 1000:.1f} mm, "
                              f"{math.degrees(solution.angle_error_rad):.1f} deg off)")
        if solution.joint_margin_rad < params["min_joint_margin_rad"]:
            joint, joint_margin = cell.limiting_joint(solution.q)
            return fail("limits", f"joint {joint} at its limit during '{sample.phase}' "
                                  f"(margin {max(joint_margin, 0.0):.3f} rad, needs {params['min_joint_margin_rad']:.2f})")
        step = float(np.max(np.abs(np.asarray(solution.q) - np.asarray(q_prev)))) if index else 0.0
        if step > params["max_joint_step_rad"]:
            return fail("jump", f"arm configuration flips by {step:.2f} rad at '{sample.phase}'")
        if sample.attach_here:      # the instrument rests where it was seen; the box follows the TCP from now on
            tcp, tcp_orientation = cell.tcp_pose()
            cell.attach_tool(ctx.tool_box["half_extents"], ctx.tool_box["position"], ctx.tool_box["quaternion"],
                             tcp, tcp_orientation)
        result = cell.clearance(width=sample.width, carried=sample.carried and not sample.attach_here,
                                relaxed=sample.relaxed, margin=margin, contact_margin=contact)
        if not result.clear:
            return fail("collision", f"{result.violations[0]} at '{sample.phase}'")
        q_prev = solution.q
        stat = phases.setdefault(sample.phase, {"phase": sample.phase, "samples": 0, "max_ik_error_mm": 0.0,
                                                "min_joint_margin_rad": math.inf, "min_clearance_mm": math.inf})
        stat["samples"] += 1
        stat["max_ik_error_mm"] = max(stat["max_ik_error_mm"], solution.position_error_m * 1000)
        stat["min_joint_margin_rad"] = min(stat["min_joint_margin_rad"], solution.joint_margin_rad)
        stat["min_clearance_mm"] = min(stat["min_clearance_mm"], result.min_distance_m * 1000)
        worst["ik_error_mm"] = max(worst["ik_error_mm"], solution.position_error_m * 1000)
        worst["joint_margin_rad"] = min(worst["joint_margin_rad"], solution.joint_margin_rad)
        worst["joint_step_rad"] = max(worst["joint_step_rad"], step)
        worst["clearance_mm"] = min(worst["clearance_mm"], result.min_distance_m * 1000)
    cell.detach_tool()
    rounded = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} for s in phases.values()]
    return _Attempt(True, len(samples), None, rounded, {k: round(v, 3) for k, v in worst.items()})


def check_path(cell, cand: Candidate, ctx: ValidationContext, home: Sequence[float]) -> Dict[str, Any]:
    """(iii) + (iv): IK, joint limits and collisions along the whole path, over several starting configurations."""
    samples = plan_samples(cand, ctx)
    best: Optional[_Attempt] = None
    for seed_index, seed in enumerate(ik_seeds(samples[0].position, home)):
        attempt = _walk(cell, cand, ctx, samples, seed)
        if attempt.ok:
            totals = attempt.totals
            return {"passed": True, "seed_index": seed_index, "samples": len(samples), "phases": attempt.phases,
                    "max_ik_error_mm": totals["ik_error_mm"], "min_joint_margin_rad": totals["joint_margin_rad"],
                    "max_joint_step_rad": totals["joint_step_rad"], "min_clearance_mm": totals["clearance_mm"],
                    "failure": None}
        if best is None or attempt.progress > best.progress:
            best = attempt
    return {"passed": False, "seed_index": None, "samples": len(samples), "phases": [],
            "attempts": len(ik_seeds(samples[0].position, home)), "failure": best.failure if best else None}


def _robustness(path: Dict[str, Any]) -> float:
    margin = float(np.clip(path["min_joint_margin_rad"] / 0.35, 0.0, 1.0))
    clearance = float(np.clip(path["min_clearance_mm"] / 30.0, 0.0, 1.0))
    return 0.5 * margin + 0.5 * clearance


# --------------------------------------------------------------------------- evaluation
def describe(cand: Candidate) -> Dict[str, Any]:
    return {"id": cand.id, "rel_position_along_axis": round(cand.rel_position, 3), "jaws_reversed": cand.jaws_reversed,
            "pose_base_frame": {"position_xyz_m": round_list(cand.position), "quaternion_xyzw": round_list(cand.quaternion, 5),
                                "approach_vector": round_list(cand.approach), "jaw_closing_vector": round_list(cand.closing)},
            "opening_m": round(cand.opening, 4),
            "local_cross_section_m": {"width_along_v2": round(cand.local_width, 4),
                                      "thickness_along_v3": round(cand.local_thickness, 4)},
            "quality": round(cand.quality, 3), "quality_terms": cand.quality_terms}


def evaluate(candidates: List[Candidate], ctx: ValidationContext) -> FeasibilityResult:
    """Validate every candidate in the robot cell and rank them (feasible first, then by score)."""
    from pipeline.robot_cell import RobotCell
    from pipeline.sim.robot_scene import HOME

    params = ctx.params
    started = time.perf_counter()
    entries: List[Dict[str, Any]] = []
    with RobotCell(ctx.config, ctx.base_frame, robot_urdf=params["robot_urdf"],
                   position_tolerance_m=params["ik_position_tolerance_m"],
                   angle_tolerance_rad=params["ik_angle_tolerance_rad"]) as cell:
        for cand in candidates:
            entry = describe(cand)
            contact, orientation = check_contact(cand, ctx), check_orientation(cand, ctx)
            entry["contact"], entry["orientation"] = contact, orientation
            if contact["passed"] and orientation["passed"]:
                path = check_path(cell, cand, ctx, HOME)
            else:
                path = {"passed": False, "skipped": True, "failure": None, "samples": 0}
            entry["kinematics"] = {k: v for k, v in path.items() if k not in ("min_clearance_mm",)}
            failure = path.get("failure")
            entry["collision"] = {"passed": True if path["passed"] else (False if failure and failure["kind"] == "collision" else None),
                                  "min_clearance_mm": path.get("min_clearance_mm"),
                                  "margin_mm": params["collision_margin_m"] * 1000}
            entry["feasible"] = bool(contact["passed"] and orientation["passed"] and path["passed"])
            reasons = contact["issues"] + orientation["issues"]
            if path.get("failure"):
                reasons.append(path["failure"]["detail"])
            elif not path["passed"] and not path.get("skipped"):
                reasons.append("no IK configuration solves the whole path")
            entry["failure_reasons"] = reasons
            entry["score"] = round(cand.quality * (0.5 + 0.5 * _robustness(path)), 4) if entry["feasible"] else 0.0
            entries.append(entry)
    entries.sort(key=lambda e: (not e["feasible"], -e["score"], -e["quality"], e["id"]))
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
        entry["selected"] = False
    selected = next((e for e in entries if e["feasible"]), None)
    if selected:
        selected["selected"] = True
    summary = {"engine": "pybullet_robot_cell", "ik_checked": True, "collision_checked": True,
               "workspace_checked": True, "contact_region_checked": True, "contact_checked": False,
               "validated_phases": [wp.APPROACH, wp.DESCEND, wp.LIFT, wp.TRANSFER, wp.PLACE],
               "candidates": len(entries), "feasible": sum(e["feasible"] for e in entries),
               "selected_id": selected["id"] if selected else None,
               "duration_s": round(time.perf_counter() - started, 3),
               "parameters": {k: v for k, v in params.items() if k != "robot_urdf"},
               "robot_model": params["robot_urdf"] or "pybullet kuka_iiwa (LBR iiwa 14 geometry)",
               "limitations": LIMITATIONS}
    return FeasibilityResult(entries, selected["id"] if selected else None, summary)


def geometry_only(candidates: List[Candidate], ctx: ValidationContext, reason: Optional[str]) -> FeasibilityResult:
    """Candidates with the geometric checks only (contact, orientation): no IK, no collisions.

    Used when the robot-cell check is disabled or cannot run. Nothing is marked feasible (``feasible: None``).
    """
    entries: List[Dict[str, Any]] = []
    for cand in candidates:
        entry = describe(cand)
        entry["contact"], entry["orientation"] = check_contact(cand, ctx), check_orientation(cand, ctx)
        entry["kinematics"] = {"passed": None, "skipped": True, "reason": reason}
        entry["collision"] = {"passed": None, "min_clearance_mm": None, "margin_mm": ctx.params["collision_margin_m"] * 1000}
        entry["feasible"] = None
        entry["failure_reasons"] = entry["contact"]["issues"] + entry["orientation"]["issues"]
        entry["score"] = round(cand.quality, 4)
        entries.append(entry)
    entries.sort(key=lambda e: (bool(e["failure_reasons"]), -e["quality"], e["id"]))
    for rank, entry in enumerate(entries, start=1):
        entry["rank"], entry["selected"] = rank, False
    summary = {"engine": None, "ik_checked": False, "collision_checked": False, "workspace_checked": True,
               "contact_region_checked": True, "contact_checked": False, "validated_phases": [],
               "candidates": len(entries), "feasible": None, "selected_id": None, "reason": reason,
               "limitations": LIMITATIONS}
    return FeasibilityResult(entries, None, summary)
