#!/usr/bin/env python3
"""
Pipeline Simulazione - Step 3b: Posizionamento STL nel Vassoio con Report Visivo PNG.
Esegue:
  1. Identificazione del file STL corrispondente allo strumento target (da Stage 1 e SAM 3).
  2. Analisi PCA della mesh STL per calcolare il baricentro e l'assetto piatto sul vassoio.
  3. Posizionamento all'interno dei margini fisici del vassoio sorgente da YAML.
  4. Generazione del report visivo PNG (sim_stl_tray_placement.png) con ingombri e distanze dai bordi.
  5. Esportazione del report JSON strutturato (data/outputs/stage3_stl_placement.json).

Input: data/outputs/stage1_task_plan.json + config/surgical_environment.yaml
Output: data/outputs/stage3_stl_placement.json (+ sim_stl_tray_placement.png)
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
import trimesh
from PIL import Image
import pybullet as p
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import quaternion_from_matrix
from pipeline.mesh_loader import find_instrument_mesh


def compute_mesh_pca(vertices):
    """Calcola baricentro ed autovettori principali PCA della mesh STL."""
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    v1 = evecs[:, 0]  # Asse longitudinale (lunghezza)
    v2 = evecs[:, 1]  # Asse laterale (larghezza)
    v3 = evecs[:, 2]  # Asse normale (spessore minimo)

    if np.linalg.det(evecs) < 0:
        v3 = -v3
        evecs[:, 2] = v3

    return centroid, evals, evecs


def run_sim_stl_placement(plan_path=None, config_path=None, output_path=None, tool_override=None, seed=42):
    print("\n" + "="*70)
    print("📐 [PIPELINE SIM - STAGE 3B] POSIZIONAMENTO STL NEL VASSOIO & REPORT VISIVO PNG")
    print("="*70)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(base_dir, "data", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Carica Piano LLM
    if plan_path is None:
        plan_path = os.path.join(out_dir, "stage1_task_plan.json")
    
    stage1_data = {}
    if os.path.exists(plan_path):
        with open(plan_path, "r", encoding="utf-8") as f:
            stage1_data = json.load(f)

    target_tool = tool_override or stage1_data.get("target_instrument", "vascular scissors")
    target_keyword = tool_override or stage1_data.get("target_keyword", "scissors")
    source_tray_key = stage1_data.get("source_tray", "tray_2")

    # 2. Carica Parametri Ambiente da YAML
    config = load_environment_config(config_path)
    env_cfg = config.get("environment", {})
    trays_cfg = env_cfg.get("trays", {})
    tray_info = trays_cfg.get(source_tray_key, {
        "position_xyz_m": [0.60, -0.42, 0.07],
        "dimensions_xyz_m": [0.62, 0.68, 0.04]
    })
    tray_pos = tray_info["position_xyz_m"]
    tray_dims = tray_info["dimensions_xyz_m"]

    # 3. Trova e analizza Mesh STL
    stl_path = find_instrument_mesh(target_keyword=target_keyword, config=config)
    if not stl_path or not os.path.exists(stl_path):
        raise FileNotFoundError(f"Nessuna mesh STL trovata per '{target_keyword}'!")

    print(f"📦 Mesh STL target identificata: {os.path.basename(stl_path)}")
    mesh = trimesh.load(stl_path)
    if hasattr(mesh, "geometry"):
        mesh = list(mesh.geometry.values())[0]

    # Auto-scaling (se in mm converte in m)
    verts = np.array(mesh.vertices)
    span = np.ptp(verts, axis=0)
    scale_factor = 0.001 if np.max(span) > 2.0 else 1.0
    verts_m = verts * scale_factor

    centroid_local, evals, evecs = compute_mesh_pca(verts_m)
    v1, v2, v3 = evecs[:, 0], evecs[:, 1], evecs[:, 2]

    length_m = float(np.ptp(verts_m @ v1))
    width_m = float(np.ptp(verts_m @ v2))
    thickness_m = float(np.ptp(verts_m @ v3))

    # Matrice di rotazione per adagiare l'asse normale v3 piatto sul piano orizzontale (+Z)
    R_align = np.column_stack([v1, v2, v3]).T
    # Aggiungi una rotazione piana controllata sul vassoio (es. yaw casuale o inclinato a 20°)
    yaw_deg = random.uniform(-25.0, 25.0) if seed else 15.0
    yaw_rad = math.radians(yaw_deg)
    R_yaw = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
        [math.sin(yaw_rad),  math.cos(yaw_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    R_final = R_yaw @ R_align
    q_final = quaternion_from_matrix(R_final)

    # 4. Calcola posizione baricentrica all'interno dei margini del vassoio
    margin_x = (tray_dims[0] - width_m) / 2.0 - 0.05
    margin_y = (tray_dims[1] - length_m) / 2.0 - 0.05
    jitter_x = random.uniform(-max(margin_x*0.4, 0.01), max(margin_x*0.4, 0.01))
    jitter_y = random.uniform(-max(margin_y*0.4, 0.01), max(margin_y*0.4, 0.01))

    tray_floor_z = tray_pos[2] + tray_dims[2] / 2.0
    tool_pos_x = tray_pos[0] + jitter_x
    tool_pos_y = tray_pos[1] + jitter_y
    tool_pos_z = tray_floor_z + thickness_m / 2.0 + 0.003
    tool_world_pos = [round(tool_pos_x, 4), round(tool_pos_y, 4), round(tool_pos_z, 4)]

    # Calcolo distanze minime dalle pareti interne del vassoio
    dist_wall_neg_x = round((tool_pos_x - length_m/2.0) - (tray_pos[0] - tray_dims[0]/2.0), 3)
    dist_wall_pos_x = round((tray_pos[0] + tray_dims[0]/2.0) - (tool_pos_x + length_m/2.0), 3)
    dist_wall_neg_y = round((tool_pos_y - width_m/2.0) - (tray_pos[1] - tray_dims[1]/2.0), 3)
    dist_wall_pos_y = round((tray_pos[1] + tray_dims[1]/2.0) - (tool_pos_y + width_m/2.0), 3)

    # 5. Generazione Render Ottico 3D da PyBullet e Grafico Diagnostico PNG
    out_vis_png = os.path.join(out_dir, "sim_stl_tray_placement.png")

    fig, ax = plt.subplots(figsize=(10, 10))
    # Disegna il vassoio chirurgico (Acciaio Inox)
    tray_patch = patches.Rectangle(
        (tray_pos[1] - tray_dims[1]/2.0, tray_pos[0] - tray_dims[0]/2.0),
        tray_dims[1], tray_dims[0],
        linewidth=3, edgecolor="navy", facecolor="#d0d4dc", label=f"Vassoio: {source_tray_key} ({tray_dims[1]*1000:.0f}x{tray_dims[0]*1000:.0f} mm)"
    )
    ax.add_patch(tray_patch)

    # Bordo interno di sicurezza (5 cm clearance)
    inner_patch = patches.Rectangle(
        (tray_pos[1] - tray_dims[1]/2.0 + 0.05, tray_pos[0] - tray_dims[0]/2.0 + 0.05),
        tray_dims[1] - 0.10, tray_dims[0] - 0.10,
        linewidth=1.5, edgecolor="green", linestyle="--", facecolor="none", label="Area di Sicurezza (Clearance > 5 cm)"
    )
    ax.add_patch(inner_patch)

    # Proietta i vertici trasformati della mesh STL nel piano del vassoio (X-Y)
    verts_trans = (verts_m - centroid_local) @ R_final.T + np.array(tool_world_pos)
    ax.scatter(verts_trans[:, 1], verts_trans[:, 0], c="#333333", s=2, alpha=0.6, label=f"Mesh STL: {target_tool}")

    # Segna il baricentro e l'asse principale
    ax.scatter([tool_world_pos[1]], [tool_world_pos[0]], c="red", s=120, marker="X", zorder=5, label="Baricentro STL (PCA)")
    arrow_len = 0.08
    ax.arrow(tool_world_pos[1], tool_world_pos[0],
             math.sin(yaw_rad)*arrow_len, math.cos(yaw_rad)*arrow_len,
             color="red", width=0.003, head_width=0.015, label="Asse Principale v1 (Yaw)")

    # Quote metriche e annotazioni
    ax.text(tray_pos[1], tray_pos[0] + tray_dims[0]/2.0 + 0.02,
            f"Distanze pareti: DX={dist_wall_pos_x*1000:.0f}mm, SX={dist_wall_neg_x*1000:.0f}mm, UP={dist_wall_pos_y*1000:.0f}mm, DOWN={dist_wall_neg_y*1000:.0f}mm",
            ha="center", fontsize=9, fontweight="bold", color="navy")

    ax.set_xlabel("Y (Robot Base Frame) [m]")
    ax.set_ylabel("X (Robot Base Frame) [m]")
    ax.set_title(f"Report Posizionamento STL nel Vassoio: {target_tool.title()}\n(Piatto su piano Z, Yaw: {yaw_deg:.1f}°)")
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_vis_png, dpi=150)
    plt.close(fig)

    # 6. Report JSON
    if output_path is None:
        output_path = os.path.join(out_dir, "stage3_stl_placement.json")

    report = {
        "stage": "stage3_stl_placement",
        "timestamp": datetime.datetime.now().isoformat(),
        "target_instrument": target_tool,
        "source_stl_file": stl_path,
        "source_tray": source_tray_key,
        "tray_bounding_box_m": {
            "center_xyz": tray_pos,
            "dimensions_xyz": tray_dims
        },
        "stl_dimensions_m": {
            "length_v1": round(length_m, 4),
            "width_v2": round(width_m, 4),
            "thickness_v3": round(thickness_m, 4)
        },
        "placed_pose_base_frame": {
            "position_xyz_m": tool_world_pos,
            "orientation_quaternion_xyzw": [round(float(q), 5) for q in q_final],
            "yaw_deg": round(yaw_deg, 2)
        },
        "wall_clearances_m": {
            "dist_wall_neg_x": dist_wall_neg_x,
            "dist_wall_pos_x": dist_wall_pos_x,
            "dist_wall_neg_y": dist_wall_neg_y,
            "dist_wall_pos_y": dist_wall_pos_y,
            "is_within_tray_margins": bool(min(dist_wall_neg_x, dist_wall_pos_x, dist_wall_neg_y, dist_wall_pos_y) > 0.03)
        },
        "artifacts_generated": {
            "placement_visual_png": out_vis_png
        },
        "status": "COMPLETED"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Strumento STL posizionato con successo dentro {source_tray_key}:")
    print(f"   - Baricentro (Base Frame): {tool_world_pos} m")
    print(f"   - Orientamento Yaw       : {yaw_deg:.1f}° (Piatto sul fondo vassoio)")
    print(f"   - Margine minimo pareti  : {min(dist_wall_neg_x, dist_wall_pos_x, dist_wall_neg_y, dist_wall_pos_y)*1000:.0f} mm")
    print(f"   - Report JSON salvato in : {output_path}")
    print(f"   - Immagine PNG salvata in: {out_vis_png}")
    print("="*70 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim Step 3b: STL Tray Placement & Visual Report")
    parser.add_argument("--plan", type=str, default=None)
    parser.add_argument("--tool", type=str, default=None, help="Override strumento target (es. 'Scalpel', 'Bone Forceps')")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_sim_stl_placement(plan_path=args.plan, output_path=args.output, tool_override=args.tool, seed=args.seed)
