#!/usr/bin/env python3
"""
Pipeline Simulazione - Step 2: Spawning Scena 3D & Rendering Telecamera RGB-D.
Carica la scena 3D (la collezione completa All_Instruments.obj oppure il singolo STL)
sul tavolo/vassoio operatorio e acquisisce il feed ottico simulato della telecamera RGB-D.
Elimina completamente le fotografie statiche 2D, realizzando una simulazione 3D pura.
Input: data/outputs/stage1_task_plan.json + config/surgical_environment.yaml
Output: data/outputs/sim_stage2_camera.json (+ sim_camera_rgb.png, sim_camera_depth.npy)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import json
import math
import random
import argparse
import datetime
import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data
import trimesh

from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import RobotBaseFrame, quaternion_from_matrix


def get_or_create_all_instruments_obj(base_dir):
    """Verifica e ritorna il modello 3D combinato della collezione All_Instruments."""
    combined_path = os.path.join(base_dir, "data", "inputs", "All_Instruments_combined.obj")
    if os.path.exists(combined_path):
        return combined_path

    raw_obj = os.path.join(base_dir, "data", "inputs", "surgical-instruments-collection", "source", "All_Instruments.obj")
    if os.path.exists(raw_obj):
        scene = trimesh.load(raw_obj)
        combined = scene.to_geometry() if hasattr(scene, 'to_geometry') else scene.dump(concatenate=True)
        combined.export(combined_path)
        return combined_path

    raw_fbx = os.path.join(base_dir, "data", "inputs", "surgical-instruments-collection", "source", "All_Instruments.fbx")
    if os.path.exists(raw_fbx):
        # Conversione assimp
        os.system(f"assimp export '{raw_fbx}' '{raw_obj}'")
        scene = trimesh.load(raw_obj)
        combined = scene.to_geometry() if hasattr(scene, 'to_geometry') else scene.dump(concatenate=True)
        combined.export(combined_path)
        return combined_path

    raise FileNotFoundError("All_Instruments non trovato!")


def run_sim_camera_render(plan_path=None, config_path=None, output_path=None, mode="all", seed=42):
    print("\n" + "="*70)
    print(f"[PIPELINE SIM - STAGE 2] RENDER TELECAMERA RGB-D SU SCENA 3D (MODE: {mode.upper()})")
    print("   (Zero fotografie 2D usate / Rendering puro da geometria 3D)")
    print("="*70)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 1. Lettura dello Stage 1 (Piano LLM)
    if plan_path is None:
        plan_path = os.path.join(base_dir, "data", "outputs", "stage1_task_plan.json")
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"Report Stage 1 non trovato in {plan_path}!")

    with open(plan_path, "r", encoding="utf-8") as f:
        stage1_data = json.load(f)

    target_tool = stage1_data.get("target_instrument", "vascular scissors")
    source_tray_key = stage1_data.get("source_tray", "tray_2")

    # 2. Caricamento Parametri Ambiente da YAML
    config = load_environment_config(config_path)
    env_cfg = config.get("environment", {})
    trays_cfg = env_cfg.get("trays", {})
    table_cfg = env_cfg.get("surgical_table", {})
    cam_cfg = config.get("camera_overhead", {})

    tray_info = trays_cfg.get(source_tray_key, {
        "position_xyz_m": [0.60, -0.42, 0.07],
        "dimensions_xyz_m": [0.62, 0.68, 0.04]
    })
    tray_pos = tray_info["position_xyz_m"]
    tray_dims = tray_info["dimensions_xyz_m"]

    # 3. Inizializzazione PyBullet e Creazione Scena da YAML
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=cid)

    # Tavolo operatorio da YAML
    t_center = table_cfg.get("center_xyz_m", [0.60, 0.0, -0.05])
    t_dims = table_cfg.get("dimensions_xyz_m", [0.85, 1.60, 0.20])
    c_tab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t_dims[0]/2, t_dims[1]/2, t_dims[2]/2], physicsClientId=cid)
    v_tab = p.createVisualShape(p.GEOM_BOX, halfExtents=[t_dims[0]/2, t_dims[1]/2, t_dims[2]/2],
                                rgbaColor=table_cfg.get("color_rgba", [0.15, 0.35, 0.60, 1.0]), physicsClientId=cid)
    p.createMultiBody(0, c_tab, v_tab, t_center, physicsClientId=cid)

    # Vassoi chirurgici da YAML
    for t_key, t_data in trays_cfg.items():
        pos = t_data.get("position_xyz_m", [0.60, 0.42, 0.07])
        dims = t_data.get("dimensions_xyz_m", [0.62, 0.68, 0.04])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2], physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2],
                                  rgbaColor=t_data.get("color_rgba", [0.8, 0.82, 0.85, 1.0]), physicsClientId=cid)
        p.createMultiBody(0, col, vis, pos, physicsClientId=cid)

    # 4. Caricamento della Geometria 3D
    if mode == "all":
        # Scena 3D completa All_Instruments posizionata all'interno del Vassoio 2 (Origine)
        mesh_3d_file = get_or_create_all_instruments_obj(base_dir)
        print(f"Caricamento Scena 3D Completa: {os.path.basename(mesh_3d_file)} all'interno di {source_tray_key}")
        scale_vec = [0.01, 0.01, 0.01] # cm -> metri

        # Rotazione per adagiare gli strumenti sul piano orizzontale (Z up)
        R_rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0]
        ], dtype=np.float64)
        q_rot = quaternion_from_matrix(R_rot)
        
        # Posiziona esattamente sul fondo interno del vassoio di origine
        tray_top_z = tray_pos[2] + tray_dims[2] / 2.0
        spawn_pos = [tray_pos[0], tray_pos[1], tray_top_z + 0.005]

        # Finitura satinata acciaio chirurgico per alto contrasto ottico col vassoio inox
        v_tool = p.createVisualShape(p.GEOM_MESH, fileName=mesh_3d_file, meshScale=scale_vec,
                                     rgbaColor=[0.25, 0.26, 0.30, 1.0], physicsClientId=cid)
        tool_id = p.createMultiBody(0, -1, v_tool, spawn_pos, q_rot, physicsClientId=cid)
        settled_pos = spawn_pos
        settled_orn = q_rot
    else:
        # Singolo STL dello strumento target (Coerenza ottica e geometrica al 100%)
        from pipeline.mesh_loader import find_instrument_mesh, inspect_stl_bounds
        stl_path = find_instrument_mesh(target_keyword=target_tool, config=config)
        if not stl_path or not os.path.exists(stl_path):
            raise FileNotFoundError(f"Mesh STL non trovata per lo strumento: '{target_tool}'!")
        
        bounds = inspect_stl_bounds(stl_path)
        scale_vec = bounds["recommended_scale"]
        q_rot = bounds["flat_orientation"]
        print(f"Caricamento Strumento STL nel Vassoio: {os.path.basename(stl_path)}")
        print(f"   -> Dimensioni: {[round(d*100, 1) for d in bounds['scaled_dimensions_m']]} cm | Unità: {bounds['detected_unit']}")

        # Ottimizzazione rendering per PyBullet TinyRenderer se la mesh supera 50.000 triangoli
        # TinyRenderer scarta silenziosamente le mesh con >1M di poligoni (scansioni 3D-COSI ad altissima densità)
        mesh_3d_file = stl_path
        try:
            m_test = trimesh.load(stl_path)
            if hasattr(m_test, "geometry"):
                m_test = list(m_test.geometry.values())[0]
            if len(m_test.faces) > 50000:
                cache_dir = os.path.join(base_dir, "data", "inputs", ".cache")
                os.makedirs(cache_dir, exist_ok=True)
                opt_obj = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(stl_path))[0]}_opt.obj")
                if not os.path.exists(opt_obj):
                    print(f"Ottimizzazione poligonale per TinyRenderer: {len(m_test.faces)} facce -> Convex Hull...")
                    hull = m_test.convex_hull
                    hull.export(opt_obj)
                mesh_3d_file = opt_obj
                print(f"Mesh ottimizzata pronta per il rendering: {os.path.basename(mesh_3d_file)}")
        except Exception as e:
            print(f"Uso mesh originale ({e})")

        tray_top_z = tray_pos[2] + tray_dims[2] / 2.0
        spawn_pos = [tray_pos[0], tray_pos[1], tray_top_z + 0.008]
        v_tool = p.createVisualShape(p.GEOM_MESH, fileName=mesh_3d_file, meshScale=scale_vec,
                                     rgbaColor=[0.18, 0.18, 0.20, 1.0], physicsClientId=cid)
        tool_id = p.createMultiBody(0, -1, v_tool, spawn_pos, q_rot, physicsClientId=cid)
        settled_pos = spawn_pos
        settled_orn = q_rot

    # 5. Rendering Vista Telecamera RGB-D da Parametri YAML
    cam_extr = cam_cfg.get("extrinsics", {})
    cam_pos_cfg = cam_extr.get("position_world_frame_xyz_m", cam_extr.get("position_robot_frame_xyz_m"))
    cam_target_cfg = cam_extr.get("target_position_xyz_m")
    auto_track = cam_cfg.get("auto_track_source_tray", True)
    height_z = cam_cfg.get("height_above_tray_m", 1.08)

    # Se auto_track è attivo o extrinsics sono "auto", la telecamera segue dinamicamente il vassoio di origine
    if auto_track or cam_pos_cfg == "auto" or not cam_pos_cfg:
        cam_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + height_z]
        cam_target = [tray_pos[0], tray_pos[1], tray_pos[2]]
    else:
        cam_pos = cam_pos_cfg
        cam_target = cam_target_cfg if (cam_target_cfg and cam_target_cfg != "auto") else [tray_pos[0], tray_pos[1], tray_pos[2]]

    intr = cam_cfg.get("intrinsics", {})
    w, h = int(cam_cfg.get("width_px", 640)), int(cam_cfg.get("height_px", 480))
    fx, fy = float(intr.get("fx_px", 615.0)), float(intr.get("fy_px", 615.0))
    fov_y = 2.0 * math.atan(h / (2.0 * fy)) * (180.0 / math.pi)
    near_val, far_val = 0.1, 2.0

    v_mat = p.computeViewMatrix(cam_pos, cam_target, [-1.0, 0.0, 0.0])
    p_mat = p.computeProjectionMatrixFOV(fov_y, w / h, near_val, far_val)

    img_arr = p.getCameraImage(w, h, v_mat, p_mat, renderer=p.ER_TINY_RENDERER)
    p.disconnect(cid)

    rgb = np.array(img_arr[2], dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
    d_buf = np.array(img_arr[3], dtype=np.float32).reshape((h, w))
    depth_m = (far_val * near_val) / (far_val - (far_val - near_val) * d_buf)

    # Salvataggio Immagine e Mappa di Profondità
    out_dir = os.path.join(base_dir, "data", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_rgb_path = os.path.join(out_dir, "sim_camera_rgb.png")
    out_depth_path = os.path.join(out_dir, "sim_camera_depth.npy")
    out_vis_path = os.path.join(out_dir, "sim_camera_depth_visual.png")

    Image.fromarray(rgb).save(out_rgb_path)
    np.save(out_depth_path, depth_m)

    d_vis = (np.clip((depth_m - 0.5) / 0.5, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(d_vis).save(out_vis_path)

    if output_path is None:
        output_path = os.path.join(out_dir, "sim_stage2_camera.json")

    report = {
        "stage": "sim_stage2_camera",
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": mode,
        "simulated_instrument": target_tool,
        "3d_source_file": mesh_3d_file,
        "source_tray": source_tray_key,
        "robot_base_pose_world": RobotBaseFrame.from_config(config).to_dict(),
        "settled_ground_truth_pose": {
            "frame_id": "world",
            "position_xyz_m": [round(x, 4) for x in settled_pos],
            "orientation_xyzw": [round(q, 5) for q in settled_orn]
        },
        "camera_metadata": {
            "frame_id": "world",
            "position_xyz_m": cam_pos,
            "target_position_xyz_m": cam_target,
            "resolution_wh": [w, h],
            "fov_y_deg": round(fov_y, 2),
            "fx_fy": [fx, fy]
        },
        "files": {
            "camera_rgb_image": out_rgb_path,
            "camera_depth_npy": out_depth_path,
            "camera_depth_visual": out_vis_path
        },
        "status": "COMPLETED"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Vista telecamera RGB-D simulata salvata:")
    print(f"   - Immagine RGB telecamera: {out_rgb_path}")
    print(f"   - Mappa di profondità    : {out_depth_path}")
    print(f"   - Report JSON            : {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim Step 2: 3D Camera RGB-D Render")
    parser.add_argument("--plan", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["all", "single"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_sim_camera_render(plan_path=args.plan, config_path=args.config, mode=args.mode, seed=args.seed)
