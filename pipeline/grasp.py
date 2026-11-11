"""Stage ``grasp``: selectable PCA geometry (default) or official remote GraspGen inference.

The tool is split along its main axis into Zone A (finger rings, 0-35%), Zone B (shank/hinge,
35-65%, the grasp zone) and Zone C (blades, 65-100%, forbidden). The PCA backend proposes SEVERAL parallel-jaw
candidates along Zone B (position, orientation, jaw opening, approach direction; jaws in both directions) and
validates each one in a kinematic model of the cell (``pipeline.feasibility``): contact region, orientation,
IK and joint limits, collisions along approach, lift, transfer and place. The best feasible candidate becomes
the designated grasp; with none, the grasp is NOT_READY_FOR_ROBOT_MOTION with the reasons.

Input : ``*_deprojection.json`` (sim or real) + ``plan.json`` (source/destination tray, tool name) + environment
        config (trays, table, base pose, ``grasp_validation``)
Output: ``grasp.json`` -> data: source_pipeline, target_instrument, input_pointcloud,
        instrument_bounding_dimensions_m, principal_axes_base_frame, functional_zones_segmentation,
        grasp_candidates (pose, opening, quality, score and every check per candidate, with the reasons of a
        rejection), selected_candidate_id, designated_grasp_geometry (the selected candidate), feasibility,
        readiness, readiness_blockers
        artifacts: zones_ply, overview_png, candidates_png

Run alone:  python -m pipeline.grasp --deprojection data/outputs/real_deprojection.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from pipeline import feasibility
from pipeline.common import paths
from pipeline.common.cli import run_cli, stage_parser
from pipeline.common.ply import read_ply, write_ply
from pipeline.common.report import StageError, StageReport, stage_run
from pipeline.common.scene import round_list
from pipeline.common.workspace import GRASP_MARGIN_M, Workspace, from_config
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import RobotBaseFrame, quaternion_from_matrix

STAGE = "grasp"
log = logging.getLogger(STAGE)

MIN_CLOUD_POINTS = 15
BASE_PLANE_FRACTION = 0.10         # lowest 10% of the height span is tray/table contamination
MIN_HEIGHT_SPAN_M = 0.015
ZONE_BOUNDS = (0.35, 0.65)
MAX_GRIPPER_STROKE_M = 0.080
CLEARANCE_M = 0.020
ZONE_COLORS = {"A": (40, 120, 220), "B": (40, 220, 80), "C": (220, 50, 50)}


def isolate_tool(points: np.ndarray, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Drop the lowest slice of the cloud (support plane) when the cloud is tall enough."""
    heights = points @ np.asarray(world_up)
    span = heights.max() - heights.min()
    if span > MIN_HEIGHT_SPAN_M:
        tool = points[heights >= heights.min() + BASE_PLANE_FRACTION * span]
        if len(tool) >= 10:
            return tool
    return points


def principal_axes(points: np.ndarray, base_frame: RobotBaseFrame | None = None):
    """Centroid and right-handed (v1 longitudinal, v2 lateral, v3 up-normal) axes."""
    centroid = points.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(points - centroid, rowvar=False))
    evecs = evecs[:, np.argsort(evals)[::-1]]
    v1, v3 = evecs[:, 0], evecs[:, 2]
    base = base_frame or RobotBaseFrame()
    world_up = base.rotation.T @ np.array([0.0, 0.0, 1.0])
    world_diagonal = base.rotation.T @ np.array([1.0, 1.0, 0.0])
    if np.dot(v3, world_up) < 0:
        v3 = -v3
    if np.dot(v1, world_diagonal) < 0:
        v1 = -v1
    v2 = np.cross(v3, v1)
    return centroid, v1, v2 / np.linalg.norm(v2), v3


def gripper_frame(v2: np.ndarray, v3: np.ndarray):
    """z_ee approaches down along -v3, y_ee (jaw closing) along v2, x_ee completes the triad."""
    z_ee, y_ee = -v3 / np.linalg.norm(v3), v2 / np.linalg.norm(v2)
    x_ee = np.cross(y_ee, z_ee)
    x_ee /= np.linalg.norm(x_ee)
    return x_ee, y_ee, z_ee


def analyze(tool_points: np.ndarray, base_frame: RobotBaseFrame | None = None) -> Dict[str, Any]:
    centroid, v1, v2, v3 = principal_axes(tool_points, base_frame)
    centered = tool_points - centroid
    s = centered @ v1
    length = float(s.max() - s.min())
    width, thickness = float(np.ptp(centered @ v2)), float(np.ptp(centered @ v3))
    rel_s = (s - s.min()) / length
    low, high = ZONE_BOUNDS
    masks = {"A": rel_s < low, "B": (rel_s >= low) & (rel_s <= high), "C": rel_s > high}
    zone_b = tool_points[masks["B"]]
    if len(zone_b) >= 3:
        grasp_centroid, grasp_width, grasp_thickness = zone_b.mean(axis=0), float(np.ptp(zone_b @ v2)), float(np.ptp(zone_b @ v3))
    else:  # sparse cloud: fall back to the whole tool
        grasp_centroid, grasp_width, grasp_thickness = centroid, width * 0.4, thickness
    x_ee, y_ee, z_ee = gripper_frame(v2, v3)
    quat = quaternion_from_matrix(np.column_stack([x_ee, y_ee, z_ee]))
    required = grasp_width + CLEARANCE_M
    return {"centroid": centroid, "axes": (v1, v2, v3), "rel_s": rel_s, "masks": masks, "length": length,
            "width": width, "thickness": thickness, "grasp_centroid": grasp_centroid,
            "grasp_width": grasp_width, "grasp_thickness": grasp_thickness, "ee_axes": (x_ee, y_ee, z_ee),
            "quat": quat, "required_stroke": required,
            "graspable": bool(required <= MAX_GRIPPER_STROKE_M and grasp_width > 0.002)}


def plot_overview(path: Path, tool_points: np.ndarray, a: Dict[str, Any], tool_name: str) -> None:
    m, (v1, v2, _) = a["masks"], a["axes"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    styles = {"A": ("royalblue", "Zone A: rings (handle)", 25, 0.8), "B": ("limegreen", "Zone B: shank/hinge (GRASP)", 35, 0.9),
              "C": ("crimson", "Zone C: blades (forbidden)", 25, 0.8)}
    for zone, (color, label, size, alpha) in styles.items():
        pts = tool_points[m[zone]]
        ax1.scatter(pts[:, 0], pts[:, 1], c=color, s=size, label=label, alpha=alpha)
        ax2.hist(a["rel_s"][m[zone]] * 100, bins=15, color=color, alpha=0.7, label=label)
    c = a["centroid"]
    ax1.arrow(c[0], c[1], v1[0] * 0.05, v1[1] * 0.05, color="black", width=0.0015, head_width=0.005, label="Longitudinal axis v1")
    ax1.arrow(c[0], c[1], v2[0] * 0.05, v2[1] * 0.05, color="darkorange", width=0.0015, head_width=0.005, label="Lateral axis v2 (jaw closing)")
    ax1.scatter([a["grasp_centroid"][0]], [a["grasp_centroid"][1]], c="gold", s=180, marker="*", edgecolors="black",
                linewidths=1.5, label="Grasp centroid (Zone B)", zorder=5)
    ax1.set(xlabel="X (robot base frame) [m]", ylabel="Y (robot base frame) [m]",
            title=f"Grasp zone segmentation ({tool_name.replace('_', ' ').title()})")
    ax1.axis("equal")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)
    for edge, label in zip(ZONE_BOUNDS, ("A/B", "B/C")):
        ax2.axvline(edge * 100, color="gray", linestyle="--", label=f"Zone {label} threshold ({edge * 100:.0f}%)")
    ax2.set(xlabel="Relative longitudinal position [%]", ylabel="Surface point density",
            title="Material distribution along the longitudinal axis")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _zone_report(a: Dict[str, Any]) -> Dict[str, Any]:
    n = {z: int(a["masks"][z].sum()) for z in "ABC"}
    return {
        "zone_A_proximal_rings": {
            "description": "Finger loops of the handle", "longitudinal_range_rel": [0.00, ZONE_BOUNDS[0]],
            "points_count": n["A"], "grasp_classification": "EXCLUDED_HIGH_COMPLIANCE_HOLLOW_STRUCTURE",
            "risk": "Torsional flex, rotation around the fingers and gripper slip"},
        "zone_B_central_shank": {
            "description": "Shank and pivot hinge", "longitudinal_range_rel": list(ZONE_BOUNDS),
            "points_count": n["B"], "grasp_classification": "PRIMARY_OPTIMAL_GRASP_ZONE",
            "advantage": "Maximum structural rigidity, minimal moment of inertia, planar surface for parallel jaws"},
        "zone_C_distal_blades": {
            "description": "Cutting blades and tips", "longitudinal_range_rel": [ZONE_BOUNDS[1], 1.00],
            "points_count": n["C"], "grasp_classification": "FORBIDDEN_CUTTING_EDGE_EXCLUSION_ZONE",
            "risk": "Damage to the cutting edge, collision and clinical hazard"},
    }


READY = "VALIDATED_GEOMETRY_READY_FOR_TRAJECTORY_PLAN"
NOT_READY = "NOT_READY_FOR_ROBOT_MOTION"


def readiness_blockers(a: Dict[str, Any], workspace: Workspace, deprojection: StageReport,
                       validation: Optional[Dict[str, Any]] = None) -> List[str]:
    """Everything that forbids sending this grasp to the robot (empty list = geometry cleared).

    ``a``: the designated candidate (graspable, required_stroke, grasp_centroid). ``validation``: outcome of the
    feasibility check ({ran, required, total, feasible, reason}); when it ran, IK and collisions supersede the
    coarse reachable-annulus test.
    """
    blockers: List[str] = []
    if not a["graspable"]:
        blockers.append(f"required jaw opening {a['required_stroke'] * 1000:.0f} mm exceeds the gripper stroke "
                        f"{MAX_GRIPPER_STROKE_M * 1000:.0f} mm (or the grasp section is degenerate)")
    if validation and validation.get("ran"):
        if validation["feasible"] == 0:
            blockers.append(f"none of the {validation['total']} grasp candidates is feasible: {validation['reason']}")
    else:
        blockers += workspace.blockers(a["grasp_centroid"], GRASP_MARGIN_M)
        if validation and validation.get("required"):
            blockers.append("grasp feasibility (IK, joint limits, collisions) was not verified: "
                            f"{validation.get('reason') or 'check disabled'}")
    if deprojection.data.get("motion_safe") is False:
        blockers.append("input geometry comes from a nominal depth plane and/or uncalibrated camera "
                        f"(depth_source={deprojection.data.get('depth_source')}): measured RGB-D and calibration required")
    if deprojection.data.get("ee_reference_source") == "nominal" and deprojection.stage.startswith("real"):
        blockers.append("end-effector frame is the nominal config pose: set motion_control.ee_reference_pose "
                        "from the robot TF before using end_effector_frame")
    return blockers


# --------------------------------------------------------------------------- candidates
def generate_candidates(tool_points: np.ndarray, a: Dict[str, Any], base_frame: RobotBaseFrame,
                        support_top_world_z: float, params: Dict[str, Any]) -> List[feasibility.Candidate]:
    """Parallel-jaw candidates along Zone B: several positions, jaws in both directions (robot base frame).

    The gripper approaches VERTICALLY (the instrument lies on a horizontal tray) and the jaws close along the
    horizontal projection of the instrument's lateral axis (v2). The TCP sits ``pinch_target_m`` below the visible
    top of the instrument but never closer than ``tip_clearance_m`` to the supporting tray, so thin instruments
    are pinched as deep as the tray allows. How far the instrument is from lying flat is kept in ``tilt_deg``.
    """
    v1, v2, v3 = a["axes"]
    up = np.asarray(base_frame.rotation, dtype=float).T @ np.array([0.0, 0.0, 1.0])   # world up in the base frame
    z_ee = -up
    lateral = v2 - (v2 @ up) * up
    if np.linalg.norm(lateral) < 0.3:                      # lateral axis nearly vertical: close across the other axis
        lateral = v1 - (v1 @ up) * up
    y_ee = lateral / np.linalg.norm(lateral)
    x_ee = np.cross(y_ee, z_ee)
    tilt = float(np.degrees(np.arccos(np.clip(v3 @ up, -1.0, 1.0))))
    rel_s = a["rel_s"]
    low, high = ZONE_BOUNDS
    count = max(1, int(params["positions_along_zone"]))
    positions = (np.linspace(low + 0.03, high - 0.03, count) if count > 1 else np.array([(low + high) / 2.0]))
    world = np.asarray(base_frame.base_to_world(tool_points), dtype=float)
    heights = world[:, 2] - support_top_world_z
    support_ref = max(8.0, int(a["masks"]["B"].sum()) / 3.0)
    flips = (False, True) if params["both_jaw_orientations"] else (False,)
    candidates: List[feasibility.Candidate] = []
    for s in positions:
        near = np.abs(rel_s - s) <= 0.05
        if near.sum() < 8:                                   # sparse cloud: take the nearest points instead
            near = np.zeros_like(near)
            near[np.argsort(np.abs(rel_s - s))[:min(12, len(rel_s))]] = True
        slab = tool_points[near]
        across = slab @ y_ee
        width = float(np.ptp(across))
        centre = slab.mean(axis=0)
        centre = centre + y_ee * ((across.max() + across.min()) / 2.0 - centre @ y_ee)  # jaws straddle the extent
        slab_heights = heights[near]
        top = float(np.percentile(slab_heights, 95)) if len(slab_heights) > 3 else float(slab_heights.max())
        tcp_height = max(params["tip_clearance_m"], top - params["pinch_target_m"])
        pinch = max(0.0, top - tcp_height)
        centre_world = np.asarray(base_frame.base_to_world(centre), dtype=float)
        tcp = np.asarray(base_frame.world_to_base(
            [centre_world[0], centre_world[1], support_top_world_z + tcp_height]), dtype=float)
        opening = width + CLEARANCE_M
        quality, terms = feasibility.quality_score(float(s), opening, MAX_GRIPPER_STROKE_M, int(near.sum()),
                                                   support_ref, pinch, params["pinch_target_m"])
        for reversed_jaws in flips:
            x, y = (-x_ee, -y_ee) if reversed_jaws else (x_ee, y_ee)
            candidates.append(feasibility.Candidate(
                id=len(candidates) + 1, rel_position=float(s), jaws_reversed=reversed_jaws, position=tcp,
                quaternion=np.asarray(quaternion_from_matrix(np.column_stack([x, y, z_ee]))),
                approach=np.asarray(z_ee, dtype=float), closing=np.asarray(y, dtype=float), opening=opening,
                local_width=width, local_thickness=float(np.ptp(slab @ v3)), pinch_depth=pinch,
                support_points=int(near.sum()), quality=quality, quality_terms=terms, tilt_deg=tilt))
    return candidates


def tool_box(tool_points: np.ndarray, a: Dict[str, Any], base_frame: RobotBaseFrame,
             support_top_world_z: float) -> Dict[str, Any]:
    """Bounding box of the instrument resting on the tray (robot base frame): what the gripper carries.

    The box lies FLAT on the tray, rotated only about the vertical axis to follow the instrument: the small tilt
    of the PCA normal that mask-edge noise produces must not make a corner dip into the tray when it is placed.
    """
    v1 = a["axes"][0]
    up = np.asarray(base_frame.rotation, dtype=float).T @ np.array([0.0, 0.0, 1.0])
    x_axis = v1 - (v1 @ up) * up
    x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 1e-6 else np.cross(up, a["axes"][1])
    y_axis = np.cross(up, x_axis)
    centered = tool_points - a["centroid"]
    world = np.asarray(base_frame.base_to_world(tool_points), dtype=float)
    top = float(np.percentile(world[:, 2] - support_top_world_z, 98))
    thickness = max(top - feasibility.TOOL_REST_HEIGHT_M, 0.002)
    centre_xy = np.asarray(base_frame.base_to_world(a["centroid"]), dtype=float)[:2]
    centre_world = [centre_xy[0], centre_xy[1], support_top_world_z + feasibility.TOOL_REST_HEIGHT_M + thickness / 2]
    return {"half_extents": [float(np.ptp(centered @ x_axis)) / 2, float(np.ptp(centered @ y_axis)) / 2, thickness / 2],
            "position": [float(v) for v in base_frame.world_to_base(centre_world)],
            "quaternion": quaternion_from_matrix(np.column_stack([x_axis, y_axis, up]))}


def resolve_trays(config: Dict[str, Any], plan_data: Optional[Dict[str, Any]], base_frame: RobotBaseFrame,
                  tool_points: np.ndarray) -> Tuple[Optional[str], Optional[str], List[str]]:
    """(source, destination, inferred): tray keys from the plan; missing ones are inferred from the geometry."""
    trays = config.get("environment", {}).get("trays", {})
    if not trays:
        return None, None, []
    source = (plan_data or {}).get("source_tray")
    destination = (plan_data or {}).get("destination_tray")
    inferred: List[str] = []
    if source not in trays:
        centre = np.asarray(base_frame.base_to_world(tool_points.mean(axis=0)), dtype=float)[:2]
        source = min(trays, key=lambda k: float(np.linalg.norm(np.asarray(trays[k]["position_xyz_m"][:2]) - centre)))
        inferred.append(f"source tray '{source}' (closest to the instrument)")
    if destination not in trays or destination == source:
        others = [k for k in trays if k != source]
        destination = others[0] if others else source
        inferred.append(f"destination tray '{destination}'")
    return source, destination, inferred


def plot_candidates(path: Path, tool_points: np.ndarray, entries: List[Dict[str, Any]],
                    candidates: Dict[int, feasibility.Candidate], tool_name: str) -> None:
    """Top view of every candidate (jaws, opening) coloured by feasibility, plus a ranked table with the reasons.

    Jaws reversed (180 deg about the approach axis) have the same footprint as the standard ones: they are drawn
    as a dashed, slightly larger outline under it and their label sits on the opposite side.
    """
    fig, (ax, table) = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax.scatter(tool_points[:, 0], tool_points[:, 1], c="#9aa0a6", s=6, alpha=0.7, zorder=1)

    def colour(entry: Dict[str, Any]) -> str:
        return "#d93025" if entry.get("feasible") is False else ("#188038" if entry.get("feasible") else "#f9ab00")

    # rejected first, feasible next, the selected one on top
    order = sorted(entries, key=lambda e: (e["selected"], bool(e.get("feasible")), -e.get("rank", 0)))
    for entry in order:
        cand = candidates[entry["id"]]
        color = colour(entry)
        closing = cand.closing[:2] * (-1.0 if cand.jaws_reversed else 1.0)      # the un-reversed direction
        finger_axis = np.array([-closing[1], closing[0]])
        grow = 0.0025 if cand.jaws_reversed else 0.0
        for sign in (-1, 1):                                                    # the two fingertips seen from above
            tip = cand.position[:2] + sign * closing * (cand.local_width / 2 + 0.004)
            corners = [tip + finger_axis * (0.013 + grow) * u + closing * (0.004 + grow) * v
                       for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            if cand.jaws_reversed:
                ax.add_patch(patches.Polygon(corners, closed=True, facecolor=color, alpha=0.10, edgecolor=color,
                                             linewidth=1.1, linestyle="--", zorder=2))
            else:
                ax.add_patch(patches.Polygon(corners, closed=True, facecolor=color,
                                             alpha=0.9 if entry["selected"] else 0.5,
                                             edgecolor="#fbbc04" if entry["selected"] else color,
                                             linewidth=2.6 if entry["selected"] else 1.0, zorder=3))
        side = 1.0 if cand.jaws_reversed else -1.0
        ax.annotate(str(entry["id"]), cand.position[:2] + closing * side * (cand.local_width / 2 + 0.016),
                    ha="center", va="center", fontsize=8, fontweight="bold", color=color, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
    ax.set(xlabel="X (robot base frame) [m]", ylabel="Y (robot base frame) [m]",
           title=f"Grasp candidates ({tool_name.replace('_', ' ').title()}): green feasible, red rejected, "
                 "gold outline = selected, dashed = jaws reversed")
    ax.title.set_fontsize(9)
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    table.axis("off")
    lines = [f"{'#':>2} {'pos':>5} {'jaws':>4} {'qual':>5} {'score':>5}  result"]
    for entry in sorted(entries, key=lambda e: e.get("rank", 0)):
        if entry.get("feasible"):
            state = "SELECTED" if entry["selected"] else "feasible"
        else:
            state = "not verified" if entry.get("feasible") is None else "rejected"
        reason = "" if entry.get("feasible") else (entry["failure_reasons"][0] if entry["failure_reasons"] else "")
        lines.append(f"{entry['id']:>2} {entry['rel_position_along_axis']:>5.2f} {'rev' if entry['jaws_reversed'] else 'std':>4} "
                     f"{entry['quality']:>5.2f} {entry['score']:>5.2f}  {state}" + (f": {reason[:70]}" if reason else ""))
    table.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _designated(entry: Dict[str, Any], cand: feasibility.Candidate, a: Dict[str, Any]) -> Dict[str, Any]:
    """``designated_grasp_geometry`` of one candidate (the format every consumer already reads)."""
    x_axis = np.cross(cand.closing, cand.approach)
    return {
        "selected_zone": "Zone B (Central Shank / Pivot Hinge)",
        "candidate_id": cand.id, "candidate_score": entry.get("score"), "candidate_quality": round(cand.quality, 3),
        "grasp_centroid_base_frame_xyz_m": round_list(cand.position),
        "local_cross_section_m": {"width_along_v2": round(cand.local_width, 4),
                                  "thickness_along_v3": round(cand.local_thickness, 4)},
        "pinch_depth_m": round(cand.pinch_depth, 4),
        "recommended_gripper_orientation": {
            "quaternion_xyzw": round_list(cand.quaternion, 5), "approach_vector_base": round_list(cand.approach),
            "jaw_closing_vector_base": round_list(cand.closing), "finger_alignment_vector_base": round_list(x_axis)},
        "gripper_compatibility_check": {
            "parallel_jaw_suitable": bool(cand.opening <= MAX_GRIPPER_STROKE_M and cand.local_width > 0.002),
            "required_opening_m": round(cand.opening, 4), "max_allowed_stroke_m": MAX_GRIPPER_STROKE_M,
            "safety_clearance_m": CLEARANCE_M},
    }


def _validate_candidates(candidates, ctx, report) -> Optional[feasibility.FeasibilityResult]:
    """Run the feasibility check; a check that cannot run is reported, never silently skipped."""
    try:
        return feasibility.evaluate(candidates, ctx)
    except Exception as exc:                    # PyBullet/URDF problems: the grasp is then not cleared (see blockers)
        log.debug("feasibility check failed", exc_info=True)
        report.warn(log, f"feasibility check could not run: {type(exc).__name__}: {exc}")
        return None


def _run_pca(*, deprojection_path: Optional[str] = None, plan_path: Optional[str] = None,
             config_path: Optional[str] = None, out_dir: Optional[str] = None,
             report_path: Optional[str] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    deprojection_file = paths.resolve(deprojection_path, paths.SIM_DEPROJECTION, out)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.GRASP, out),
                   inputs={"deprojection": deprojection_file, "plan": plan_file, "config": config_path},
                   logger=log) as report:
        deprojection = StageReport.load(deprojection_file)
        config = load_environment_config(config_path)
        workspace = from_config(config)
        params = feasibility.load_params(config)
        if not deprojection.stage.endswith("_deprojection"):
            raise StageError(STAGE, f"{deprojection_file} is a '{deprojection.stage}' report, not a deprojection")
        source = deprojection.stage.split("_")[0]
        plan_data = None
        if plan_file.exists():
            plan_data = StageReport.load(plan_file, expect_stage="plan").data
            tool_name = plan_data["target_instrument"]
        else:
            tool_name = "surgical_instrument"
            report.warn(log, f"plan report not found ({plan_file}); using generic tool name")

        ply = deprojection.artifact("pointcloud_ply")
        points, _ = read_ply(ply)
        if len(points) < MIN_CLOUD_POINTS:
            raise StageError(STAGE, f"too few 3D points ({len(points)}) for geometric analysis")
        base_frame = RobotBaseFrame.from_dict(deprojection.data.get("robot_base_pose_world", {}))
        world_up = base_frame.rotation.T @ np.array([0.0, 0.0, 1.0])
        tool_points = isolate_tool(points, world_up)
        log.info("analyzing %d points (raw: %d)", len(tool_points), len(points))

        a = analyze(tool_points, base_frame)

        # --- candidates and their validation in the robot cell ---------------------------------------
        source_key, destination_key, inferred = resolve_trays(config, plan_data, base_frame, tool_points)
        for what in inferred:
            report.warn(log, f"plan does not define it: using {what}")
        validation: Dict[str, Any] = {"ran": False, "required": bool(params["require_feasibility"]), "reason": None}
        result: Optional[feasibility.FeasibilityResult] = None
        if source_key is None:
            candidates = []
        else:
            support_top = float(config["environment"]["trays"][source_key]["position_xyz_m"][2]
                                + config["environment"]["trays"][source_key]["dimensions_xyz_m"][2] / 2)
            candidates = generate_candidates(tool_points, a, base_frame, support_top, params)
        if not candidates:
            raise StageError(STAGE, "no grasp candidate could be generated (trays missing from the configuration?)")
        by_id = {c.id: c for c in candidates}
        ctx = feasibility.ValidationContext(
            config=config, base_frame=base_frame, source_key=source_key, destination_key=destination_key,
            tool_box=tool_box(tool_points, a, base_frame, support_top), support_top_world_z=support_top,
            max_stroke=MAX_GRIPPER_STROKE_M, params=params)
        if params["enabled"]:
            result = _validate_candidates(candidates, ctx, report)
            validation["reason"] = "the check could not run (see warnings)" if result is None else None
        else:
            validation["reason"] = "disabled (grasp_validation.enabled: false)"
        if result is None:
            result = feasibility.geometry_only(candidates, ctx, validation["reason"])
        else:
            validation.update(ran=True, total=len(result.candidates), feasible=result.summary["feasible"])

        entries = result.candidates
        if result.selected_id is not None:
            chosen_entry = next(e for e in entries if e["id"] == result.selected_id)
        else:                                       # nothing verified: show the best geometry, but do not clear it
            chosen_entry = max(entries, key=lambda e: (e["contact"]["passed"], e["quality"]))
            if validation["ran"]:
                best = max(entries, key=lambda e: e["quality"])
                validation["reason"] = (f"best candidate #{best['id']}: "
                                        f"{best['failure_reasons'][0] if best['failure_reasons'] else 'unknown'}")
        chosen = by_id[chosen_entry["id"]]
        designated = _designated(chosen_entry, chosen, a)

        blockers = readiness_blockers(
            {"graspable": designated["gripper_compatibility_check"]["parallel_jaw_suitable"],
             "required_stroke": chosen.opening, "grasp_centroid": chosen.position}, workspace, deprojection, validation)
        for message in blockers:
            report.warn(log, f"NOT READY: {message}")

        colors = np.zeros((len(tool_points), 3), dtype=np.uint8)
        for zone, rgb in ZONE_COLORS.items():
            colors[a["masks"][zone]] = rgb
        zones_ply, overview = out / "grasp_zones.ply", out / "grasp_overview.png"
        candidates_png = out / "grasp_candidates.png"
        write_ply(zones_ply, tool_points, colors)
        plot_overview(overview, tool_points, a, tool_name)
        plot_candidates(candidates_png, tool_points, entries, by_id, tool_name)

        v1, v2, v3 = a["axes"]
        report.artifacts = {"zones_ply": str(zones_ply), "overview_png": str(overview),
                            "candidates_png": str(candidates_png)}
        report.data = {
            "grasp_backend": "pca",
            "source_pipeline": source, "target_instrument": tool_name,
            "coordinate_frame_id": base_frame.frame_id,
            "robot_base_pose_world": base_frame.to_dict(),
            "input_pointcloud": {"source_ply": str(ply), "total_points": int(len(points)),
                                 "filtered_points": int(len(tool_points))},
            "instrument_bounding_dimensions_m": {"length_along_v1": round(a["length"], 4),
                                                 "width_along_v2": round(a["width"], 4),
                                                 "thickness_along_v3": round(a["thickness"], 4)},
            "principal_axes_base_frame": {"v1_longitudinal": round_list(v1), "v2_lateral": round_list(v2),
                                          "v3_normal": round_list(v3)},
            "functional_zones_segmentation": _zone_report(a),
            "source_tray": source_key, "destination_tray": destination_key,
            "grasp_candidates": entries,
            "selected_candidate_id": result.selected_id,
            "designated_grasp_geometry": designated,
            "readiness": NOT_READY if blockers else READY,
            "readiness_blockers": blockers,
            "feasibility": result.summary,
        }
        log.info("%s | L=%.1f W=%.1f T=%.1f mm | %d candidates, %s feasible | grasp #%d %s | opening %.1f/%.1f mm",
                 tool_name, a["length"] * 1000, a["width"] * 1000, a["thickness"] * 1000, len(entries),
                 result.summary["feasible"] if validation["ran"] else "?", chosen.id,
                 round_list(chosen.position), chosen.opening * 1000, MAX_GRIPPER_STROKE_M * 1000)
        log.info("readiness: %s", report.data["readiness"])
    return report


def add_backend_arguments(parser):
    """Shared flags for standalone grasp, sim, real and graphical launchers."""
    parser.add_argument("--grasp-backend", choices=["pca", "graspgen"], default="pca")
    parser.add_argument("--graspgen-config", default=None, help="Client/TCP YAML (default: config/graspgen.yaml)")
    parser.add_argument("--graspgen-host", default=None)
    parser.add_argument("--graspgen-port", type=int, default=None)
    parser.add_argument("--graspgen-timeout-ms", type=int, default=None)


def backend_kwargs(args):
    return dict(backend=args.grasp_backend, graspgen_config=args.graspgen_config,
                graspgen_host=args.graspgen_host, graspgen_port=args.graspgen_port,
                graspgen_timeout_ms=args.graspgen_timeout_ms)


def run(*, deprojection_path=None, plan_path=None, out_dir=None, report_path=None,
        backend="pca", graspgen_config=None, graspgen_host=None, graspgen_port=None,
        graspgen_timeout_ms=None, config_path=None) -> StageReport:
    """Keep grasp.json as the pipeline interface and retain backend-specific reports.

    ``config_path`` is the ENVIRONMENT config (workspace limits for the PCA readiness gates);
    ``graspgen_config`` is the separate GraspGen client YAML.

    A failed GraspGen run overwrites the current interface with FAILED; an older
    successful PCA result can never be consumed as an implicit fallback.
    """
    if backend not in {"pca", "graspgen"}:
        raise StageError(STAGE, f"unknown grasp backend {backend!r}")
    out = paths.output_dir(out_dir)
    target = paths.output_report(report_path, paths.GRASP, out)
    archive = out / f"grasp_{backend}.json"
    kwargs = dict(deprojection_path=deprojection_path, plan_path=plan_path,
                  out_dir=out_dir, report_path=str(target.resolve()))
    try:
        if backend == "pca":
            return _run_pca(**kwargs, config_path=config_path)
        from pipeline.graspgen.backend import run as run_graspgen
        return run_graspgen(**kwargs, config_path=graspgen_config, host=graspgen_host,
                            port=graspgen_port, timeout_ms=graspgen_timeout_ms)
    finally:
        if target.exists() and target.resolve() != archive.resolve():
            StageReport.load(target, require_completed=False).save(archive)


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--deprojection", default=None,
                        help="sim_deprojection.json or real_deprojection.json (default: sim)")
    parser.add_argument("--plan", default=None)
    add_backend_arguments(parser)
    return run_cli(parser, lambda a: run(deprojection_path=a.deprojection, plan_path=a.plan,
                                         config_path=a.config, out_dir=a.out_dir, report_path=a.report,
                                         **backend_kwargs(a)), argv)


if __name__ == "__main__":
    raise SystemExit(main())
