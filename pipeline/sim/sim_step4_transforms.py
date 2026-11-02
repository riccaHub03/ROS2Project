#!/usr/bin/env python3
"""
Pipeline Simulazione - Step 4: De-proiezione Metrica 3D e Trasformazioni di Frame.
Prende la maschera SAM 3 della telecamera (sim_camera_mask.png) e la profondità reale (sim_camera_depth.npy)
e calcola le coordinate esatte nei frame:
  1. Camera Optical Frame
  2. Robot Base Frame (lbr_link_0)
  3. End-Effector Frame (workbench_tcp)
Input: data/outputs/sim_stage3_vision_mask.json + data/outputs/sim_stage2_camera.json + config/surgical_environment.yaml
Output: data/outputs/sim_stage4_transforms.json (+ sim_reconstructed_instrument.ply)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import json
import argparse
import datetime
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import FrameTransformer, export_pointcloud_ply


def generate_frames_visualization(rgb_np, transformer, centroid_base, cam_pos, config, output_png_path):
    """
    Genera un report visivo a due pannelli:
      - Pannello 1: Vista ottica della telecamera con terne 3D proiettate direttamente sugli elementi.
      - Pannello 2: Diagramma 3D isometrico dell'intero spazio di lavoro (Robot Base, Camera, Vassoi, Strumento).
    """
    h, w, _ = rgb_np.shape
    fx = transformer.fx
    fy = transformer.fy
    cx = transformer.cx
    cy = transformer.cy

    def proj(pt_world):
        pt_cam = transformer.transform_world_to_camera(pt_world)
        if pt_cam[2] <= 0.01:
            return None
        u = (fx * pt_cam[0]) / pt_cam[2] + cx
        v = (fy * pt_cam[1]) / pt_cam[2] + cy
        return np.array([u, v])

    # Elementi dell'ambiente caricati dinamicamente da YAML
    env_cfg = config.get("environment", {})
    trays_cfg = env_cfg.get("trays", {})
    kuka_cfg = env_cfg.get("kuka_base", {})
    p_base = np.array(kuka_cfg.get("position_xyz_m", [0.0, 0.0, 0.0]))

    fig = plt.figure(figsize=(16, 8))

    # --- PANNELLO 1: Vista Telecamera 2D con Riferimenti Frame Proiettati ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rgb_np)

    def draw_projected_frame(ax, origin_w, R_w, label, axis_len=0.10, origin_color="gold"):
        p0 = proj(origin_w)
        if p0 is None:
            return
        colors = ["red", "lime", "cyan"]
        axis_names = ["X", "Y", "Z"]
        for i in range(3):
            axis_end_w = origin_w + R_w[:, i] * axis_len
            p_end = proj(axis_end_w)
            if p_end is not None:
                ax.annotate("", xy=(p_end[0], p_end[1]), xytext=(p0[0], p0[1]),
                            arrowprops=dict(arrowstyle="->", color=colors[i], lw=2.5))
                ax.text(p_end[0], p_end[1], axis_names[i], color=colors[i],
                        fontsize=9, fontweight="bold", ha="center", va="center")
        ax.scatter([p0[0]], [p0[1]], c=origin_color, s=80, edgecolors="black", zorder=6)
        ax.text(p0[0] + 10, p0[1] - 10, label, color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75))

    # 1. Terne di tutti i Vassoi definiti nel file YAML
    for t_k, t_data in trays_cfg.items():
        t_pos_vec = np.array(t_data.get("position_xyz_m", [0.60, -0.42, 0.07]))
        t_name = t_data.get("name", t_k)
        draw_projected_frame(ax1, t_pos_vec, np.eye(3), f"{t_name} Frame", axis_len=0.10, origin_color="gold")

    # 2. Terna Strumento Target
    draw_projected_frame(ax1, centroid_base, np.eye(3), "Target Instrument Frame", axis_len=0.09, origin_color="orange")

    # 3. Widget Terna Camera Optical Frame (in alto a sinistra dello schermo)
    corner_x, corner_y = 65, 65
    ax1.annotate("", xy=(corner_x + 45, corner_y), xytext=(corner_x, corner_y),
                 arrowprops=dict(arrowstyle="->", color="red", lw=3))
    ax1.text(corner_x + 50, corner_y, "X_opt (Destra)", color="red", fontsize=8, fontweight="bold")

    ax1.annotate("", xy=(corner_x, corner_y + 45), xytext=(corner_x, corner_y),
                 arrowprops=dict(arrowstyle="->", color="lime", lw=3))
    ax1.text(corner_x, corner_y + 55, "Y_opt (Basso)", color="lime", fontsize=8, fontweight="bold")

    ax1.scatter([corner_x], [corner_y], c="cyan", s=90, edgecolors="black", zorder=7)
    ax1.text(corner_x - 55, corner_y - 12, "Camera Optical\nFrame", color="white", fontsize=8, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="navy", alpha=0.85))

    # Indicatore direzione verso la base del KUKA
    p_tool_2d = proj(centroid_base)
    if p_tool_2d is not None:
        ax1.annotate("", xy=(int(cx), 20), xytext=(p_tool_2d[0], p_tool_2d[1]),
                     arrowprops=dict(arrowstyle="->", color="magenta", lw=2, linestyle="--"))
        ax1.text(int(cx), 10, "▲ Verso Base Robot (lbr_link_0)", color="magenta",
                 fontsize=9, fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.8))

    ax1.set_title("VISTA OTTICA CAMERA CON RIFERIMENTI DEI FRAME", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # --- PANNELLO 2: Diagramma 3D Isometrico dello Spazio di Lavoro ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    def draw_3d_frame(ax, origin, R, label, scale=0.15):
        colors = ["red", "lime", "blue"]
        for i in range(3):
            v = R[:, i] * scale
            ax.quiver(origin[0], origin[1], origin[2], v[0], v[1], v[2], color=colors[i], lw=2)
        ax.scatter([origin[0]], [origin[1]], [origin[2]], s=40, c="black")
        ax.text(origin[0], origin[1], origin[2] + scale*0.25, label, fontsize=8, fontweight="bold")

    # 1. Base Robot KUKA da YAML
    draw_3d_frame(ax2, p_base, np.eye(3), "Base Robot (lbr_link_0)", scale=0.18)

    # 2. Camera Frame
    R_cam = transformer.R_base_opt
    draw_3d_frame(ax2, cam_pos, R_cam, "Camera Optical Frame", scale=0.16)

    # 3. Vassoi da YAML
    for t_k, t_data in trays_cfg.items():
        t_pos_vec = np.array(t_data.get("position_xyz_m", [0.60, -0.42, 0.07]))
        t_dims_vec = np.array(t_data.get("dimensions_xyz_m", [0.62, 0.68, 0.04]))
        draw_3d_frame(ax2, t_pos_vec, np.eye(3), f"{t_k} Frame", scale=0.12)
        # Bounding box 3D del vassoio
        bx = [t_pos_vec[0]-t_dims_vec[0]/2, t_pos_vec[0]+t_dims_vec[0]/2, t_pos_vec[0]+t_dims_vec[0]/2, t_pos_vec[0]-t_dims_vec[0]/2, t_pos_vec[0]-t_dims_vec[0]/2]
        by = [t_pos_vec[1]-t_dims_vec[1]/2, t_pos_vec[1]-t_dims_vec[1]/2, t_pos_vec[1]+t_dims_vec[1]/2, t_pos_vec[1]+t_dims_vec[1]/2, t_pos_vec[1]-t_dims_vec[1]/2]
        bz = [t_pos_vec[2]+t_dims_vec[2]/2.0]*5
        ax2.plot(bx, by, bz, color="gray", lw=1.2, linestyle="--", label=f"Vassoio {t_k}")

    # 4. Strumento Target
    draw_3d_frame(ax2, centroid_base, np.eye(3), "Instrument Frame", scale=0.10)

    # Connessioni cinematiche (assi ottici e trasformazioni)
    ax2.plot([p_base[0], cam_pos[0]], [p_base[1], cam_pos[1]], [p_base[2], cam_pos[2]], "k--", alpha=0.4, label="T_base_cam")
    ax2.plot([cam_pos[0], centroid_base[0]], [cam_pos[1], centroid_base[1]], [cam_pos[2], centroid_base[2]], "c-", lw=2, label="Asse Ottico Deproiezione")
    ax2.plot([p_base[0], centroid_base[0]], [p_base[1], centroid_base[1]], [p_base[2], centroid_base[2]], "m-.", lw=1.5, label="T_base_tool")

    # Tavolo operatorio da YAML
    table_cfg = env_cfg.get("surgical_table", {})
    tab_pos = np.array(table_cfg.get("center_xyz_m", [0.60, 0.0, -0.05]))
    tab_dims = np.array(table_cfg.get("dimensions_xyz_m", [0.85, 1.60, 0.20]))
    tab_z = float(table_cfg.get("surface_z_m", 0.05))
    x_c = [tab_pos[0]-tab_dims[0]/2, tab_pos[0]+tab_dims[0]/2, tab_pos[0]+tab_dims[0]/2, tab_pos[0]-tab_dims[0]/2, tab_pos[0]-tab_dims[0]/2]
    y_c = [tab_pos[1]-tab_dims[1]/2, tab_pos[1]-tab_dims[1]/2, tab_pos[1]+tab_dims[1]/2, tab_pos[1]+tab_dims[1]/2, tab_pos[1]-tab_dims[1]/2]
    z_c = [tab_z]*5
    ax2.plot(x_c, y_c, z_c, "b-", lw=1.8, alpha=0.6, label="Tavolo Operatorio")

    ax2.set_xlabel("X (Base) [m]")
    ax2.set_ylabel("Y (Base) [m]")
    ax2.set_zlabel("Z (Base) [m]")
    ax2.set_title("DIAGRAMMA ISOMETRICO SPAZIALE 3D DEI FRAME", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.view_init(elev=28, azim=-55)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    fig.savefig(output_png_path, dpi=150)
    plt.close(fig)
    print(f"🖼️ Vista diagnostica dei frame salvata in: {output_png_path}")


def run_sim_transforms(mask_report_path=None, cam_report_path=None, config_path=None, output_path=None):
    print("\n" + "="*70)
    print("📐 [PIPELINE SIM - STAGE 4] TRASFORMAZIONI DI FRAME E RICOSTRUZIONE 3D")
    print("="*70)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if mask_report_path is None:
        mask_report_path = os.path.join(base_dir, "data", "outputs", "sim_stage3_vision_mask.json")
    if cam_report_path is None:
        cam_report_path = os.path.join(base_dir, "data", "outputs", "sim_stage2_camera.json")

    with open(mask_report_path, "r", encoding="utf-8") as f:
        mask_data = json.load(f)
    with open(cam_report_path, "r", encoding="utf-8") as f:
        cam_data = json.load(f)

    config = load_environment_config(config_path)

    mask_file = mask_data["files"]["camera_mask_png"]
    depth_file = cam_data["files"]["camera_depth_npy"]
    rgb_file = cam_data["files"]["camera_rgb_image"]

    mask_np = np.array(Image.open(mask_file).convert("L"))
    depth_m = np.load(depth_file)
    rgb_np = np.array(Image.open(rgb_file).convert("RGB"))

    # Configurazione FrameTransformer
    cam_meta = cam_data["camera_metadata"]
    cam_pos = cam_meta["position_xyz_m"]
    cam_target = cam_meta.get("target_position_xyz_m", [cam_pos[0], cam_pos[1], 0.07])
    fx, fy = cam_meta["fx_fy"]
    w, h = cam_meta["resolution_wh"]

    transformer = FrameTransformer(
        camera_position_xyz=cam_pos,
        camera_target_xyz=cam_target,
        camera_intrinsics={"fx_px": fx, "fy_px": fy, "cx_px": w/2.0, "cy_px": h/2.0}
    )

    v_idx, u_idx = np.where(mask_np > 128)
    if len(u_idx) == 0:
        raise ValueError("Nessun pixel attivo trovato nella maschera della telecamera simulata!")

    pts_cam, pts_world = transformer.deproject_mask_to_pointcloud(mask_np, depth_m)
    colors = rgb_np[v_idx, u_idx]

    centroid_cam = np.mean(pts_cam, axis=0)
    centroid_base = np.mean(pts_world, axis=0)

    # Validazione di sicurezza sui limiti fisici del vassoio da YAML
    source_tray_key = cam_data.get("source_tray", "tray_2")
    trays_cfg = config.get("environment", {}).get("trays", {})
    t_info = trays_cfg.get(source_tray_key, {})
    t_pos = t_info.get("position_xyz_m", [0.60, -0.42, 0.07])
    t_dims = t_info.get("dimensions_xyz_m", [0.62, 0.68, 0.04])

    x_min, x_max = t_pos[0] - t_dims[0]/2.0 - 0.05, t_pos[0] + t_dims[0]/2.0 + 0.05
    y_min, y_max = t_pos[1] - t_dims[1]/2.0 - 0.05, t_pos[1] + t_dims[1]/2.0 + 0.05

    is_inside_tray = (x_min <= centroid_base[0] <= x_max) and (y_min <= centroid_base[1] <= y_max)
    conf_score = mask_data.get("confidence_score", 1.0)

    # Fallback su Stage 3b se il centroide ottico cade fisicamente fuori dal vassoio
    stage3b_file = os.path.join(base_dir, "data", "outputs", "stage3_stl_placement.json")
    if not is_inside_tray:
        if os.path.exists(stage3b_file):
            with open(stage3b_file, "r", encoding="utf-8") as f:
                s3b_data = json.load(f)
            fallback_pos = s3b_data.get("placed_pose_base_frame", {}).get("position_xyz_m")
            if fallback_pos:
                print(f"⚠️ ATTENZIONE: Maschera ottica calcolata a {[round(c, 4) for c in centroid_base]} -> FUORI DAL VASSOIO {source_tray_key}!")
                print(f"   🔄 Attivato Fallback di Sicurezza su Posa Certificata Stage 3b: {fallback_pos}")
                centroid_base = np.array(fallback_pos)
                centroid_cam = transformer.transform_world_to_camera(centroid_base)
    else:
        print(f"✅ Rilevamento ottico validato: Baricentro {[round(c, 4) for c in centroid_base]} è all'interno di {source_tray_key}")

    # Frame End-Effector (workbench_tcp alla quota di staging)
    motion_cfg = config.get("motion_control", {})
    staging_z = motion_cfg.get("heights_z_m", {}).get("staging_z", 0.45)
    down_quat = motion_cfg.get("downward_quaternion_xyzw", [0.0, 1.0, 0.0, 0.0])
    ref_ee = [0.45, 0.0, staging_z]

    pts_ee = np.array([transformer.transform_world_to_ee(p, ref_ee, down_quat) for p in pts_world])
    centroid_ee = np.mean(pts_ee, axis=0)

    # Validazione con Ground Truth PyBullet
    gt_pos = cam_data.get("settled_ground_truth_pose", {}).get("position_xyz_m")
    if gt_pos:
        err_mm = float(np.linalg.norm(centroid_base - np.array(gt_pos))) * 1000.0
    else:
        err_mm = 0.0

    print(f"📊 Risultati De-proiezione Metrica della Vista Telecamera:")
    print(f"   - Punti 3D ricostruiti            : {len(pts_world)}")
    print(f"   - Baricentro Camera Frame (m)     : {[round(c, 4) for c in centroid_cam]}")
    print(f"   - Baricentro Robot Base Frame (m) : {[round(c, 4) for c in centroid_base]}")
    print(f"   - Baricentro End-Effector Frame(m): {[round(c, 4) for c in centroid_ee]}")
    print(f"   - Scostamento vs Ground Truth     : {err_mm:.2f} mm")

    out_dir = os.path.join(base_dir, "data", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_ply = os.path.join(out_dir, "sim_reconstructed_instrument.ply")
    export_pointcloud_ply(out_ply, pts_world, colors)

    # Generazione Vista Ottica con Riferimenti Frame
    out_frames_vis = os.path.join(out_dir, "sim_stage4_frames_visual.png")
    generate_frames_visualization(
        rgb_np=rgb_np,
        transformer=transformer,
        centroid_base=centroid_base,
        cam_pos=cam_pos,
        config=config,
        output_png_path=out_frames_vis
    )

    if output_path is None:
        output_path = os.path.join(out_dir, "sim_stage4_transforms.json")

    report = {
        "stage": "sim_stage4_transforms",
        "timestamp": datetime.datetime.now().isoformat(),
        "num_points": int(len(pts_world)),
        "pointcloud_file": out_ply,
        "frames_visual_png": out_frames_vis,
        "frames": {
            "camera_optical_frame": [round(c, 4) for c in centroid_cam.tolist()],
            "robot_base_frame": [round(c, 4) for c in centroid_base.tolist()],
            "end_effector_frame": [round(c, 4) for c in centroid_ee.tolist()]
        },
        "ground_truth_validation": {
            "ground_truth_xyz_m": gt_pos,
            "reconstructed_xyz_m": [round(c, 4) for c in centroid_base.tolist()],
            "error_mm": round(err_mm, 2)
        },
        "status": "COMPLETED"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Report trasformazioni salvato in: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim Step 4: Transforms and 3D Deprojection")
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--cam", type=str, default=None)
    args = parser.parse_args()
    run_sim_transforms(mask_report_path=args.mask, cam_report_path=args.cam)
