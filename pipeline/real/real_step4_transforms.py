#!/usr/bin/env python3
"""
Pipeline Reale - Step 4: De-proiezione Metrica 3D e Frame Cinematici Reali.
Converte i pixel della maschera reale in coordinate 3D metriche nei frame:
  1. Camera Optical Frame
  2. Robot Base Frame (lbr_link_0)
  3. End-Effector Frame (workbench_tcp)
Input: data/outputs/real_stage3_vision_mask.json + data/outputs/real_stage2_scene.json + config/surgical_environment.yaml
Output: data/outputs/real_stage4_transforms.json (+ real_reconstructed_instrument.ply)
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

from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import FrameTransformer, export_pointcloud_ply


def run_real_transforms(mask_report_path=None, scene_report_path=None, config_path=None, output_path=None):
    print("\n" + "="*70)
    print("[PIPELINE REALE - STAGE 4] TRASFORMAZIONI DI FRAME E RICOSTRUZIONE 3D")
    print("="*70)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if mask_report_path is None:
        mask_report_path = os.path.join(base_dir, "data", "outputs", "real_stage3_vision_mask.json")
    if scene_report_path is None:
        scene_report_path = os.path.join(base_dir, "data", "outputs", "real_stage2_scene.json")

    with open(mask_report_path, "r", encoding="utf-8") as f:
        mask_data = json.load(f)
    with open(scene_report_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    config = load_environment_config(config_path)

    mask_file = mask_data["files"]["binary_mask"]
    depth_file = scene_data["files"]["scene_depth_npy"]
    rgb_file = scene_data["files"]["scene_rgb_image"]

    mask_np = np.array(Image.open(mask_file).convert("L"))
    depth_m = np.load(depth_file)
    rgb_np = np.array(Image.open(rgb_file).convert("RGB"))

    cam_pos = scene_data["camera_extrinsics"]["position_xyz_m"]
    fx, fy = scene_data["camera_intrinsics"]["fx_fy"]
    cx, cy = scene_data["camera_intrinsics"]["cx_cy"]

    cam_cfg = config.get("camera_overhead", {})
    cam_target = cam_cfg.get("extrinsics", {}).get("target_position_xyz_m", [cam_pos[0], cam_pos[1], 0.07])

    transformer = FrameTransformer(
        camera_position_xyz=cam_pos,
        camera_target_xyz=cam_target,
        camera_intrinsics={"fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy}
    )

    v_idx, u_idx = np.where(mask_np > 128)
    if len(u_idx) == 0:
        raise ValueError("Nessun pixel attivo trovato nella maschera dello strumento reale!")

    pts_cam, pts_world = transformer.deproject_mask_to_pointcloud(mask_np, depth_m)
    colors = rgb_np[v_idx, u_idx]

    centroid_cam = np.mean(pts_cam, axis=0)
    centroid_base = np.mean(pts_world, axis=0)

    motion_cfg = config.get("motion_control", {})
    staging_z = motion_cfg.get("heights_z_m", {}).get("staging_z", 0.45)
    down_quat = motion_cfg.get("downward_quaternion_xyzw", [0.0, 1.0, 0.0, 0.0])
    ref_ee = [0.45, 0.0, staging_z]

    pts_ee = np.array([transformer.transform_world_to_ee(p, ref_ee, down_quat) for p in pts_world])
    centroid_ee = np.mean(pts_ee, axis=0)

    print(f"Coordinate Spaziali Ricostruite dello Strumento Reale:")
    print(f"   - Punti 3D isolati               : {len(pts_world)}")
    print(f"   - Baricentro Camera Frame (m)    : {[round(c, 4) for c in centroid_cam]}")
    print(f"   - Baricentro Robot Base Frame (m): {[round(c, 4) for c in centroid_base]}")
    print(f"   - Baricentro End-Effector (m)    : {[round(c, 4) for c in centroid_ee]}")

    out_dir = os.path.join(base_dir, "data", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_ply = os.path.join(out_dir, "real_reconstructed_instrument.ply")
    export_pointcloud_ply(out_ply, pts_world, colors)

    if output_path is None:
        output_path = os.path.join(out_dir, "real_stage4_transforms.json")

    report = {
        "stage": "real_stage4_transforms",
        "timestamp": datetime.datetime.now().isoformat(),
        "num_points": int(len(pts_world)),
        "pointcloud_file": out_ply,
        "frames": {
            "camera_optical_frame": [round(c, 4) for c in centroid_cam.tolist()],
            "robot_base_frame": [round(c, 4) for c in centroid_base.tolist()],
            "end_effector_frame": [round(c, 4) for c in centroid_ee.tolist()]
        },
        "status": "COMPLETED"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report trasformazioni reali salvato in: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Step 4: Real Transforms and Deprojection")
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    args = parser.parse_args()
    run_real_transforms(mask_report_path=args.mask, scene_report_path=args.scene)
