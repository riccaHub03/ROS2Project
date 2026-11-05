"""Stage ``sim_capture``: build the 3D scene in PyBullet and render the overhead RGB-D view.

Input : ``plan.json`` + environment config
Output: ``sim_capture.json`` -> data: mode, simulated_instrument, mesh_file, source_tray,
        ground_truth_pose, camera (see CameraModel.to_dict)
        artifacts: rgb_png, depth_npy, depth_visual_png, truth_mask_png (ground-truth tool mask)

Run alone:  python -m pipeline.sim.capture --mode single
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data
import trimesh
from PIL import Image

from pipeline.common import paths
from pipeline.common.cli import run_cli, stage_parser
from pipeline.common.reconstruction import CameraModel
from pipeline.common.report import StageError, StageReport, stage_run
from pipeline.common.scene import Config, camera_intrinsics, camera_pose, get_tray, round_list
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import quaternion_from_matrix
from pipeline.mesh_loader import find_instrument_mesh, inspect_stl_bounds

STAGE = "sim_capture"
log = logging.getLogger(STAGE)

NEAR_M, FAR_M = 0.1, 2.0
ALL_SCENE_SCALE = [0.01] * 3            # All_Instruments.obj is authored in centimetres
ALL_SCENE_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # lay flat, Z up
TINY_RENDERER_MAX_FACES = 50_000        # TinyRenderer silently drops very dense meshes


# --------------------------------------------------------------------------- assets
def _combine(mesh_path: Path, out_path: Path) -> Path:
    scene = trimesh.load(mesh_path)
    combined = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    combined.export(out_path)
    return out_path


def all_instruments_obj() -> Path:
    """Combined All_Instruments mesh, generated (and cached) from the OBJ/FBX collection on demand."""
    inputs = paths.input_dir()
    combined = inputs / "All_Instruments_combined.obj"
    if combined.exists():
        return combined
    source = inputs / "surgical-instruments-collection" / "source"
    raw_obj, raw_fbx = source / "All_Instruments.obj", source / "All_Instruments.fbx"
    if not raw_obj.exists() and raw_fbx.exists():
        subprocess.run(["assimp", "export", str(raw_fbx), str(raw_obj)], check=True)
    if raw_obj.exists():
        return _combine(raw_obj, combined)
    raise StageError(STAGE, f"All_Instruments collection not found under {source}")


def render_friendly_mesh(mesh_path: str) -> str:
    """Convex-hull proxy for meshes TinyRenderer cannot draw (> 50k faces); cached on disk."""
    try:
        mesh = trimesh.load(mesh_path)
        if hasattr(mesh, "geometry"):
            mesh = list(mesh.geometry.values())[0]
        if len(mesh.faces) <= TINY_RENDERER_MAX_FACES:
            return mesh_path
        cache = paths.cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        proxy = cache / f"{Path(mesh_path).stem}_opt.obj"
        if not proxy.exists():
            log.info("simplifying %d faces -> convex hull for TinyRenderer", len(mesh.faces))
            mesh.convex_hull.export(proxy)
        return str(proxy)
    except Exception as exc:
        log.warning("using the original mesh (simplification failed: %s)", exc)
        return mesh_path


# --------------------------------------------------------------------------- scene
def _box(cid: int, position: List[float], dimensions: List[float], color: List[float]) -> int:
    half = [d / 2.0 for d in dimensions]
    return p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid),
                             p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid),
                             position, physicsClientId=cid)


def build_environment(cid: int, config: Config) -> None:
    env = config.get("environment", {})
    table = env.get("surgical_table", {})
    _box(cid, table.get("center_xyz_m", [0.60, 0.0, -0.05]), table.get("dimensions_xyz_m", [0.85, 1.60, 0.20]),
         table.get("color_rgba", [0.15, 0.35, 0.60, 1.0]))
    for tray in env.get("trays", {}).values():
        _box(cid, tray["position_xyz_m"], tray["dimensions_xyz_m"], tray.get("color_rgba", [0.8, 0.82, 0.85, 1.0]))


def spawn_tool(cid: int, mode: str, tray, plan_data: Dict[str, Any], config: Config
               ) -> Tuple[int, str, List[float], List[float]]:
    """Place the instrument (or the whole collection) in the source tray; returns (body, mesh, pos, quat)."""
    if mode == "all":
        mesh_file = str(all_instruments_obj())
        scale, orientation = ALL_SCENE_SCALE, quaternion_from_matrix(ALL_SCENE_ROTATION)
        height_offset, color = 0.005, [0.25, 0.26, 0.30, 1.0]
        log.info("loading full collection %s into %s", Path(mesh_file).name, tray.key)
    else:
        stl_path = find_instrument_mesh(target_keyword=plan_data["target_instrument"], config=config)
        if not stl_path or not Path(stl_path).exists():
            raise StageError(STAGE, f"no STL mesh found for '{plan_data['target_instrument']}'")
        bounds = inspect_stl_bounds(stl_path)
        scale, orientation = bounds["recommended_scale"], bounds["flat_orientation"]
        log.info("loading %s | dims %s cm | unit %s", Path(stl_path).name,
                 [round(d * 100, 1) for d in bounds["scaled_dimensions_m"]], bounds["detected_unit"])
        mesh_file, height_offset, color = render_friendly_mesh(stl_path), 0.008, [0.18, 0.18, 0.20, 1.0]
    position = [tray.position[0], tray.position[1], tray.top_z + height_offset]
    visual = p.createVisualShape(p.GEOM_MESH, fileName=mesh_file, meshScale=scale, rgbaColor=color, physicsClientId=cid)
    body = p.createMultiBody(0, -1, visual, position, orientation, physicsClientId=cid)
    return body, mesh_file, position, list(orientation)


def render_rgbd(cid: int, camera: CameraModel, tool_body: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """RGB, metric depth, ground-truth tool mask (from PyBullet object IDs) and vertical FOV."""
    i = camera.intrinsics
    fov_y = math.degrees(2.0 * math.atan(i.height / (2.0 * i.fy)))
    view = p.computeViewMatrix(camera.position, camera.target, [-1.0, 0.0, 0.0])
    proj = p.computeProjectionMatrixFOV(fov_y, i.width / i.height, NEAR_M, FAR_M)
    img = p.getCameraImage(i.width, i.height, view, proj, renderer=p.ER_TINY_RENDERER,
                           flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=cid)
    rgb = np.array(img[2], dtype=np.uint8).reshape((i.height, i.width, 4))[:, :, :3]
    zbuf = np.array(img[3], dtype=np.float32).reshape((i.height, i.width))
    depth = (FAR_M * NEAR_M) / (FAR_M - (FAR_M - NEAR_M) * zbuf)
    ids = np.asarray(img[4]).reshape((i.height, i.width))
    truth = (ids >= 0) & ((ids & ((1 << 24) - 1)) == tool_body)
    return rgb, depth, truth, fov_y


# --------------------------------------------------------------------------- stage
def run(*, mode: str = "all", plan_path: Optional[str] = None, config_path: Optional[str] = None,
        out_dir: Optional[str] = None, report_path: Optional[str] = None) -> StageReport:
    out = paths.output_dir(out_dir)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    with stage_run(STAGE, paths.output_report(report_path, paths.SIM_CAPTURE, out),
                   inputs={"plan": plan_file, "config": config_path}, logger=log) as report:
        plan = StageReport.load(plan_file, expect_stage="plan")
        config = load_environment_config(config_path)
        tray = get_tray(config, plan.require("source_tray"))
        pose = camera_pose(config, tray)
        camera = CameraModel(pose.position, pose.target, camera_intrinsics(config))

        cid = p.connect(p.DIRECT)
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81, physicsClientId=cid)
            build_environment(cid, config)
            tool_body, mesh_file, position, orientation = spawn_tool(cid, mode, tray, plan.data, config)
            rgb, depth, truth, fov_y = render_rgbd(cid, camera, tool_body)
        finally:
            p.disconnect(cid)

        rgb_png, depth_npy = out / "sim_capture_rgb.png", out / "sim_capture_depth.npy"
        depth_vis, truth_png = out / "sim_capture_depth_vis.png", out / "sim_capture_truth_mask.png"
        Image.fromarray(rgb).save(rgb_png)
        Image.fromarray(truth.astype(np.uint8) * 255).save(truth_png)
        np.save(depth_npy, depth)
        # Depth visualization: clip the background/floor (depth beyond table) so contrast
        # is fully allocated to the instruments and tray rather than squashed by the floor.
        table_limit = float(camera.position[2] - tray.position[2]) + 0.10
        valid = depth < table_limit
        if np.any(valid):
            d_min = float(np.min(depth[valid]))
            d_max = float(np.max(depth[valid]))
            d_range = d_max - d_min
            if d_range > 1e-4:
                # Closer to camera (higher relief) = brighter (white)
                d_norm = np.clip(1.0 - (depth - d_min) / d_range, 0.0, 1.0)
            else:
                d_norm = np.zeros_like(depth)
            d_norm[~valid] = 0.0
        else:
            d_min, d_max = float(np.min(depth)), float(np.max(depth))
            d_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)
        Image.fromarray((d_norm * 255.0).astype(np.uint8)).save(depth_vis)

        report.artifacts = {"rgb_png": str(rgb_png), "depth_npy": str(depth_npy), "depth_visual_png": str(depth_vis),
                            "truth_mask_png": str(truth_png)}
        report.data = {
            "mode": mode, "simulated_instrument": plan.require("target_instrument"), "mesh_file": mesh_file,
            "source_tray": tray.key,
            "ground_truth_pose": {"position_xyz_m": round_list(position), "orientation_xyzw": round_list(orientation, 5)},
            "camera": {**camera.to_dict(), "fov_y_deg": round(fov_y, 2)},
        }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = stage_parser(__doc__.splitlines()[0])
    parser.add_argument("--plan", default=None, help="plan.json (default: <out-dir>/plan.json)")
    parser.add_argument("--mode", choices=["all", "single"], default="all",
                        help="'all': whole All_Instruments scene; 'single': only the planned STL")
    return run_cli(parser, lambda a: run(mode=a.mode, plan_path=a.plan, config_path=a.config,
                                         out_dir=a.out_dir, report_path=a.report), argv)


if __name__ == "__main__":
    raise SystemExit(main())
