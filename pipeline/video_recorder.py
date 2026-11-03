"""
Modulo Dedicato: Registrazione Video Post-Cinematica ad Alta Definizione (Offline Video Renderer).
Genera il video MP4 del task chirurgico riproducendo la traiettoria reale registrata,
completamente disaccoppiato dal loop di controllo in real-time.
Zero impatto su watchdog, zero rallentamenti di fisica e massima resa visiva.
"""

import os
import math
import json
import cv2
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path


def render_offline_video(trajectory_log=None, p_tool_initial=None, p_tray_final=None,
                         config=None, output_path=None, fps=30):
    """
    Riproduce offline la traiettoria registrata e genera un video MP4 ad alta qualità (720p/HD).
    Non interferisce con ROS 2, non usa controllori e non ha vincoli di tempo reale.
    """
    if trajectory_log is None or len(trajectory_log) == 0:
        log_file = "/workspace/data/outputs/robot_trajectory_log.json"
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                data = json.load(f)
                trajectory_log = data.get("trajectory", [])
                p_tool_initial = data.get("tool_initial_xyz", p_tool_initial)
                p_tray_final = data.get("tray_final_xyz", p_tray_final)

    if not trajectory_log:
        print("Nessuna traiettoria registrata trovata per il rendering offline.")
        return None

    if output_path is None:
        out_dir = "/workspace/data/outputs"
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "medical_pick_and_place.mp4")

    print("\n" + "=" * 65)
    print("AVVIO RENDERING VIDEO OFFLINE POST-CINEMATICA (ALTA DEFINIZIONE)")
    print("=" * 65)
    print(f"   -> Traiettoria da riprodurre: {len(trajectory_log)} campioni temporali")
    print(f"   -> Risoluzione: 1280x720 (HD widescreen) a {fps} FPS")

    # Inizializza PyBullet in modalità DIRECT headless (completamente isolato)
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    # Carica il robot KUKA LBR iiwa
    robot_urdf = Path(pybullet_data.getDataPath()) / "kuka_iiwa" / "model.urdf"
    robot = p.loadURDF(str(robot_urdf), useFixedBase=True, physicsClientId=client)
    joint_indices = [i for i in range(p.getNumJoints(robot, physicsClientId=client))
                     if p.getJointInfo(robot, i, physicsClientId=client)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]

    # Elementi sala operatoria caricati dinamicamente da config
    if config is None:
        from pipeline.config_loader import load_environment_config
        config = load_environment_config()
    
    env_cfg = config.get("environment", {})
    table_cfg = env_cfg.get("surgical_table", {})
    trays_cfg = env_cfg.get("trays", {})

    # 1. Tavolo operatorio da YAML
    t_center = table_cfg.get("center_xyz_m", [0.60, 0.0, -0.05])
    t_dims = table_cfg.get("dimensions_xyz_m", [0.85, 1.60, 0.20])
    t_color = table_cfg.get("color_rgba", [0.15, 0.35, 0.60, 1.0])
    t_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[t_dims[0]/2, t_dims[1]/2, t_dims[2]/2],
                                rgbaColor=t_color, physicsClientId=client)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=t_vis,
                      basePosition=t_center, physicsClientId=client)

    # 2. Vassoi chirurgici da YAML
    for t_k, t_data in trays_cfg.items():
        tr_pos = t_data.get("position_xyz_m", [0.60, 0.42, 0.07])
        tr_dims = t_data.get("dimensions_xyz_m", [0.62, 0.68, 0.04])
        tr_color = t_data.get("color_rgba", [0.80, 0.82, 0.85, 1.0])
        tr_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[tr_dims[0]/2, tr_dims[1]/2, tr_dims[2]/2],
                                     rgbaColor=tr_color, physicsClientId=client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=tr_vis,
                          basePosition=tr_pos, physicsClientId=client)

    # 4. Strumento chirurgico (auto-discovery e auto-scaling per dataset 3D-COSI e STL)
    tool_pos = list(p_tool_initial) if p_tool_initial else [0.483, 0.406, 0.085]
    from pipeline.mesh_loader import find_instrument_mesh, load_instrument_in_pybullet
    mesh_file = find_instrument_mesh(config=config)

    if mesh_file:
        tool_body, mesh_info = load_instrument_in_pybullet(client, mesh_file, base_pos=tool_pos)
        print(f"   -> Rendering offline strumento STL ({mesh_info['detected_unit']}): {mesh_file}")
    else:
        tool_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.065, 0.022, 0.005], physicsClientId=client)
        tool_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.065, 0.022, 0.005],
                                       rgbaColor=[0.92, 0.92, 0.95, 1.0], physicsClientId=client)
        tool_body = p.createMultiBody(baseMass=0.08, baseCollisionShapeIndex=tool_col, baseVisualShapeIndex=tool_vis,
                                      basePosition=tool_pos, physicsClientId=client)
        p.changeDynamics(tool_body, -1, lateralFriction=0.8, spinningFriction=0.05, physicsClientId=client)

    # Configurazione telecamera HD
    width, height = 1280, 720
    camera_target = [0.45, 0.0, 0.25]
    camera_distance = 1.35
    camera_yaw = 48.0
    camera_pitch = -28.0

    view = p.computeViewMatrixFromYawPitchRoll(camera_target, camera_distance,
                                              camera_yaw, camera_pitch, 0, 2)
    projection = p.computeProjectionMatrixFOV(54, width/height, 0.05, 15.0)

    # Resample uniforme della traiettoria a 30 FPS fissi
    times = [pt["time"] for pt in trajectory_log]
    total_duration = max(times[-1], 1.0)
    num_frames = int(math.ceil(total_duration * fps))

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Rendering in corso ({num_frames} frames a {fps} FPS)...")
    last_idx = 0
    flange_index = 6 # End effector link 7

    for frame_idx in range(num_frames):
        target_t = frame_idx / fps
        # Trova il punto temporale più vicino
        while last_idx < len(times) - 1 and times[last_idx + 1] <= target_t:
            last_idx += 1

        sample = trajectory_log[last_idx]
        q = sample["q"]
        is_grasped = sample.get("grasped", False)

        # Applica configurazione giunti al modello
        for idx, val in zip(joint_indices[:7], q[:7]):
            p.resetJointState(robot, idx, val, physicsClientId=client)

        # Se lo strumento è afferrato, muovilo solidale al TCP dell'effettore
        if is_grasped:
            ee_state = p.getLinkState(robot, flange_index, computeForwardKinematics=True, physicsClientId=client)
            ee_pos = ee_state[4]
            ee_orn = ee_state[5]
            # Offset verso il basso della pinza
            p.resetBasePositionAndOrientation(tool_body, [ee_pos[0], ee_pos[1], ee_pos[2] - 0.03], ee_orn, physicsClientId=client)
        elif last_idx > len(times) // 2 and p_tray_final:
            # Deposita lo strumento stabilmente nel vassoio
            p.resetBasePositionAndOrientation(tool_body, [p_tray_final[0], p_tray_final[1], 0.092], [0, 0, 0, 1], physicsClientId=client)

        # Render frame con TinyRenderer (super-veloce in offline)
        frame_data = p.getCameraImage(width, height, view, projection,
                                      lightDirection=[1.0, 1.0, 2.0],
                                      lightColor=[1.0, 1.0, 1.0],
                                      lightDistance=3.0,
                                      shadow=0,
                                      renderer=p.ER_TINY_RENDERER,
                                      flags=p.ER_NO_SEGMENTATION_MASK,
                                      physicsClientId=client)
        rgb = np.asarray(frame_data[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    p.disconnect(physicsClientId=client)

    print(f"VIDEO HD COMPLETATO CON SUCCESSO:")
    print(f"   -> File salvato: {output_path}")
    print(f"   -> Risoluzione: {width}x{height} (720p HD Widescreen), {num_frames} frames, {total_duration:.1f}s")
    print(f"   -> Qualità nitida, senza scatti e senza ritardi di rete!")
    print("=" * 65)
    return output_path


if __name__ == "__main__":
    render_offline_video()
