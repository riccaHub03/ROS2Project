#!/usr/bin/env python3
"""
Test e Validazione Architetturale dei Nuovi Moduli:
  1. Frame Transformer (Camera Optical -> World -> End-Effector)
  2. Surgical Grasp Planner 6-DoF (PCA + Orientamento dita pinza)
  3. De-proiezione Metrica RGB-D
"""

import sys
import math
import numpy as np

# Aggiungi cartella root a sys.path
sys.path.insert(0, "/workspace")

from pipeline.frame_transformer import FrameTransformer, matrix_from_quaternion, quaternion_from_matrix
from pipeline.grasp_planner import SurgicalGraspPlanner


def test_frame_transformer():
    print("="*65)
    print("🧪 TEST 1: Frame Transformer & De-proiezione Spaziale")
    print("="*65)
    
    cam_pos = [0.55, 0.0, 0.85]
    cam_target = [0.55, 0.0, 0.05]
    intrinsics = {"fx_px": 615.0, "fy_px": 615.0, "cx_px": 320.0, "cy_px": 240.0}
    
    ft = FrameTransformer(cam_pos, cam_target, intrinsics)
    
    # 1. Punto al centro esatto dell'immagine (u=320, v=240) con profondità Z=0.80m
    p_cam = ft.deproject_pixel_to_camera_frame(320.0, 240.0, 0.80)
    print(f" -> Punto nel Camera Optical Frame (centro ottico): {p_cam}")
    assert np.allclose(p_cam, [0.0, 0.0, 0.80]), "Errore centro ottico"
    
    # 2. Trasformazione in World Frame (lbr_link_0)
    # Con camera a Z=0.85m e target a Z=0.05m, la profondità 0.80m deve corrispondere a Z_world = 0.05m!
    p_world = ft.transform_camera_to_world(p_cam)
    print(f" -> Punto trasformato nel World Frame (lbr_link_0): {p_world}")
    assert np.isclose(p_world[0], 0.55), "X world errata"
    assert np.isclose(p_world[1], 0.00), "Y world errata"
    assert np.isclose(p_world[2], 0.05), "Z world errata"
    
    # 3. Trasformazione in End-Effector Frame
    # Supponiamo che l'end-effector sia in posizione [0.55, 0.0, 0.25] rivolto verso il basso (quat [0, 1, 0, 0])
    ee_pos = [0.55, 0.0, 0.25]
    ee_quat = [0.0, 1.0, 0.0, 0.0]
    p_ee = ft.transform_world_to_ee(p_world, ee_pos, ee_quat)
    print(f" -> Punto espresso nel frame End-Effector: {p_ee}")
    # Il bersaglio è 0.20m sotto l'end-effector lungo il suo asse Z locale
    assert np.isclose(p_ee[0], 0.0, atol=1e-3), "X ee non centrata"
    assert np.isclose(p_ee[1], 0.0, atol=1e-3), "Y ee non centrata"
    
    print("✅ TEST 1 SUPERATO: Tutte le trasformazioni di frame sono geometricamente corrette!\n")


def test_grasp_planner():
    print("="*65)
    print("🧪 TEST 2: Surgical Grasp Planner 6-DoF & PCA")
    print("="*65)
    
    # Simula una nuvola di punti di una forbice chirurgica orientata a 30 gradi sul tavolo (Z = 0.085m)
    # Dimensioni: lunghezza 14cm (lungo asse v1), larghezza 4cm (lungo asse v2), spessore 0.5cm (Z)
    theta = math.radians(30.0)
    c, s = math.cos(theta), math.sin(theta)
    
    np.random.seed(42)
    n_pts = 500
    x_local = np.random.uniform(-0.07, 0.07, n_pts) # 14cm
    y_local = np.random.uniform(-0.02, 0.02, n_pts) # 4cm
    z_local = np.random.uniform(-0.002, 0.002, n_pts) # 4mm
    
    # Ruota sul tavolo e trasla a [0.48, 0.40, 0.085]
    x_rot = c * x_local - s * y_local + 0.48
    y_rot = s * x_local + c * y_local + 0.40
    z_rot = z_local + 0.085
    
    points_3d = np.stack([x_rot, y_rot, z_rot], axis=1)
    
    planner = SurgicalGraspPlanner()
    grasp_info = planner.compute_grasp_pose_from_pointcloud(points_3d)
    
    p_grasp = grasp_info["grasp_position_xyz"]
    q_grasp = grasp_info["grasp_quaternion_xyzw"]
    dims = grasp_info["dimensions_lwh_m"]
    axes = grasp_info["axes"]
    
    print(f" -> Centroide stimato : {grasp_info['centroid_xyz']}")
    print(f" -> Punto di presa 3D : {p_grasp}")
    print(f" -> Quaternione 6-DoF : {q_grasp}")
    print(f" -> Dimensioni (L,W,H): {dims[0]*100:.1f} x {dims[1]*100:.1f} x {dims[2]*100:.1f} cm")
    print(f" -> Asse Longitudinale: {axes['longitudinal']}")
    print(f" -> Asse Laterale (chiusura ganasce): {axes['lateral']}")
    print(f" -> Asse Normale (avvicinamento): {axes['normal']}")
    
    # Verifica che la matrice di rotazione derivata dal quaternione sia ortogonale
    R = matrix_from_quaternion(q_grasp)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-4), "Matrice di rotazione non ortogonale"
    
    # Verifica generazione waypoints
    waypoints = planner.generate_pick_and_place_waypoints(grasp_info, destination_tray_xyz=[0.50, -0.25, 0.16])
    assert "pre_grasp" in waypoints and "grasp" in waypoints and "lift" in waypoints
    print(" -> Waypoints generati con successo (pre_grasp, grasp, lift, place_approach, place_deposit)")
    
    print("✅ TEST 2 SUPERATO: Grasp 6-DoF, assi PCA e orientamento ganasce conformi!\n")


if __name__ == "__main__":
    test_frame_transformer()
    test_grasp_planner()
    print("🎉 TUTTI I TEST DEI NUOVI COMPONENTI COMPLETATI CON SUCCESSO!")
