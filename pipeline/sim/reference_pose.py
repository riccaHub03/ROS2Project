"""Stage ``sim_reference_pose``: certified placement of the instrument STL inside the source tray.

PCA of the STL gives centroid and a flat orientation; a seeded yaw/jitter places it within the
tray margins. The resulting pose is the fallback used by ``sim.deproject`` when the optical
centroid lands outside the tray.

Input : ``plan.json`` + environment config (+ optional ``--tool`` override)
Output: ``sim_reference_pose.json`` -> data: target_instrument, source_stl_file, source_tray,
        tray_bounding_box_m, stl_dimensions_m, pose{position_xyz_m, orientation_xyzw, yaw_deg},
        wall_clearances_m
        artifacts: placement_png

Run alone:  python -m pipeline.sim.reference_pose --tool Scalpel --seed 42
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from pipeline.common import paths
from pipeline.common.cli import run_cli, stage_parser
from pipeline.common.report import StageError, StageReport, stage_run
from pipeline.common.scene import Tray, get_tray, round_list
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import quaternion_from_matrix
from pipeline.mesh_loader import find_instrument_mesh
from pipeline.sim.placement import rotated_half_extents

STAGE = "sim_reference_pose"
log = logging.getLogger(STAGE)
MIN_WALL_CLEARANCE_M = 0.03
SAFETY_MARGIN_M = 0.05


def compute_mesh_pca(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid, eigenvalues and right-handed principal axes (columns, longest first)."""
    centroid = vertices.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(vertices - centroid, rowvar=False))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] = -evecs[:, 2]
    return centroid, evals, evecs


def load_vertices_m(stl_path: str) -> np.ndarray:
    mesh = trimesh.load(stl_path)
    if hasattr(mesh, "geometry"):
        mesh = list(mesh.geometry.values())[0]
    verts = np.array(mesh.vertices)
    return verts * (0.001 if np.max(np.ptp(verts, axis=0)) > 2.0 else 1.0)  # mm -> m


def place_in_tray(verts_m: np.ndarray, tray: Tray, seed: Optional[int]) -> dict:
    rng = random.Random(seed)
    centroid, _, axes = compute_mesh_pca(verts_m)
    v1, v2, v3 = axes[:, 0], axes[:, 1], axes[:, 2]
    length, width, thickness = (float(np.ptp(verts_m @ v)) for v in (v1, v2, v3))

    yaw_deg = rng.uniform(-25.0, 25.0) if seed is not None else 15.0
    yaw = math.radians(yaw_deg)
    r_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    rotation = r_yaw @ np.column_stack([v1, v2, v3]).T

    # The rotation maps the main axis v1 onto +x, so the length runs along x and the width along y.
    ext_x, ext_y = rotated_half_extents((length, width), yaw)
    margin_x = tray.dimensions[0] / 2.0 - ext_x - SAFETY_MARGIN_M
    margin_y = tray.dimensions[1] / 2.0 - ext_y - SAFETY_MARGIN_M
    jx = rng.uniform(-max(margin_x * 0.4, 0.01), max(margin_x * 0.4, 0.01))
    jy = rng.uniform(-max(margin_y * 0.4, 0.01), max(margin_y * 0.4, 0.01))
    x, y = tray.position[0] + jx, tray.position[1] + jy
    z = tray.top_z + thickness / 2.0 + 0.003

    half_x, half_y = tray.dimensions[0] / 2.0, tray.dimensions[1] / 2.0
    clearances = {
        "dist_wall_neg_x": round((x - ext_x) - (tray.position[0] - half_x), 3),
        "dist_wall_pos_x": round((tray.position[0] + half_x) - (x + ext_x), 3),
        "dist_wall_neg_y": round((y - ext_y) - (tray.position[1] - half_y), 3),
        "dist_wall_pos_y": round((tray.position[1] + half_y) - (y + ext_y), 3),
    }
    clearances["is_within_tray_margins"] = bool(min(clearances.values()) > MIN_WALL_CLEARANCE_M)
    return {"centroid_local": centroid, "rotation": rotation, "position": [round(x, 4), round(y, 4), round(z, 4)],
            "quaternion": quaternion_from_matrix(rotation), "yaw_deg": yaw_deg,
            "dimensions": (length, width, thickness), "clearances": clearances}


def plot_placement(path: Path, verts_m: np.ndarray, placed: dict, tray: Tray, tool_name: str) -> None:
    pos, yaw = placed["position"], math.radians(placed["yaw_deg"])
    cl = placed["clearances"]
    fig, ax = plt.subplots(figsize=(10, 10))
    ty, tx = tray.dimensions[1], tray.dimensions[0]
    ax.add_patch(patches.Rectangle((tray.position[1] - ty / 2, tray.position[0] - tx / 2), ty, tx, linewidth=3,
                                   edgecolor="navy", facecolor="#d0d4dc",
                                   label=f"Tray {tray.key} ({ty * 1000:.0f}x{tx * 1000:.0f} mm)"))
    ax.add_patch(patches.Rectangle((tray.position[1] - ty / 2 + SAFETY_MARGIN_M, tray.position[0] - tx / 2 + SAFETY_MARGIN_M),
                                   ty - 2 * SAFETY_MARGIN_M, tx - 2 * SAFETY_MARGIN_M, linewidth=1.5, edgecolor="green",
                                   linestyle="--", facecolor="none", label="Safety area (> 5 cm clearance)"))
    cloud = (verts_m - placed["centroid_local"]) @ placed["rotation"].T + np.array(pos)
    ax.scatter(cloud[:, 1], cloud[:, 0], c="#333333", s=2, alpha=0.6, label=f"STL mesh: {tool_name}")
    ax.scatter([pos[1]], [pos[0]], c="red", s=120, marker="X", zorder=5, label="STL centroid (PCA)")
    ax.arrow(pos[1], pos[0], math.sin(yaw) * 0.08, math.cos(yaw) * 0.08, color="red", width=0.003,
             head_width=0.015, label="Main axis v1 (yaw)")
    ax.text(tray.position[1], tray.position[0] + tx / 2 + 0.02,
            f"Wall distances: +X={cl['dist_wall_pos_x'] * 1000:.0f}mm, -X={cl['dist_wall_neg_x'] * 1000:.0f}mm, "
            f"+Y={cl['dist_wall_pos_y'] * 1000:.0f}mm, -Y={cl['dist_wall_neg_y'] * 1000:.0f}mm",
            ha="center", fontsize=9, fontweight="bold", color="navy")
    ax.set(xlabel="Y (world) [m]", ylabel="X (world) [m]",
           title=f"STL placement in tray: {tool_name.title()} (flat on Z, yaw {placed['yaw_deg']:.1f} deg)")
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(*, tool: Optional[str] = None, seed: Optional[int] = 42, plan_path: Optional[str] = None,
        config_path: Optional[str] = None, out_dir: Optional[str] = None,
        report_path: Optional[str] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.SIM_REFERENCE_POSE, out),
                   inputs={"plan": plan_file, "config": config_path}, logger=log) as report:
        plan = StageReport.load(plan_file, expect_stage="plan")
        config = load_environment_config(config_path)
        tool_name = tool or plan.require("target_instrument")
        keyword = tool or plan.require("target_keyword")
        tray = get_tray(config, plan.require("source_tray"))

        stl_path = find_instrument_mesh(target_keyword=keyword, config=config)
        if not stl_path or not Path(stl_path).exists():
            raise StageError(STAGE, f"no STL mesh found for '{keyword}'")
        log.info("target STL: %s", Path(stl_path).name)
        verts_m = load_vertices_m(stl_path)
        placed = place_in_tray(verts_m, tray, seed)

        png = out / "sim_reference_pose.png"
        plot_placement(png, verts_m, placed, tray, tool_name)
        length, width, thickness = placed["dimensions"]
        report.artifacts = {"placement_png": str(png)}
        report.data = {
            "target_instrument": tool_name, "source_stl_file": stl_path, "source_tray": tray.key,
            "tray_bounding_box_m": {"center_xyz": tray.position, "dimensions_xyz": tray.dimensions},
            "stl_dimensions_m": {"length_v1": round(length, 4), "width_v2": round(width, 4),
                                 "thickness_v3": round(thickness, 4)},
            "pose": {"position_xyz_m": placed["position"], "orientation_xyzw": round_list(placed["quaternion"], 5),
                     "yaw_deg": round(placed["yaw_deg"], 2)},
            "wall_clearances_m": placed["clearances"],
        }
        log.info("placed at %s, yaw %.1f deg, min wall clearance %.0f mm", placed["position"], placed["yaw_deg"],
                 min(v for k, v in placed["clearances"].items() if k.startswith("dist")) * 1000)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--plan", default=None)
    parser.add_argument("--tool", default=None, help="Override the planned instrument (e.g. 'Scalpel')")
    parser.add_argument("--seed", type=int, default=42)
    return run_cli(parser, lambda a: run(tool=a.tool, seed=a.seed, plan_path=a.plan, config_path=a.config,
                                         out_dir=a.out_dir, report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
