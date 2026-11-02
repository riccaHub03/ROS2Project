"""
Modulo 4: Controllo Cinematico e Traiettorie del Manipolatore KUKA LBR iiwa (ROS 2 Control).
Esegue la sequenza di Pick & Place cartesiana a micro-passi interpolati,
cattura i fotogrammi dalla telecamera simulata PyBullet e registra il video MP4 ad alta qualità.
"""

import os
import sys
import time
import json
import math
import shutil
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from sensor_msgs.msg import JointState

from pipeline.video_recorder import render_offline_video

def create_stamped_pose(xyz, quat=[0.0, 1.0, 0.0, 0.0]):
    msg = PoseStamped()
    msg.header.frame_id = 'lbr_link_0'
    msg.header.stamp.sec = 0
    msg.header.stamp.nanosec = 0
    msg.pose.position.x = float(xyz[0])
    msg.pose.position.y = float(xyz[1])
    msg.pose.position.z = float(xyz[2])
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg

class MedicalRobotExecutor(Node):
    def __init__(self, config=None):
        super().__init__('medical_robot_executor')
        self.config = config or {}
        
        # Topic ROS 2 Tutorial Platform
        self.path_pub = self.create_publisher(Path, '/lbr_tutorial/cartesian_path', 10)
        self.gripper_pub = self.create_publisher(String, '/lbr/gripper/command', 10)
        
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_sub = self.create_subscription(String, '/lbr_tutorial/status', self.on_status, qos)
        self.ee_sub = self.create_subscription(PoseStamped, '/lbr/ee_pose', self.on_ee_pose, 10)
        
        # Sottoscrizione leggera agli stati giunti reali per il log traiettoria offline (zero overhead di rete)
        self.joint_sub = self.create_subscription(JointState, '/lbr/joint_states', self.on_joint_states, 10)

        self.events = []
        self.ready = False
        self.current_ee = None
        self.recorded_trajectory = []
        self.recording_active = False
        self.tool_grasped = False
        self.start_traj_time = None

    def on_joint_states(self, msg):
        if not self.recording_active:
            return
        if len(msg.position) >= 7:
            try:
                if 'lbr_A1' in msg.name:
                    q = [float(msg.position[msg.name.index(f'lbr_A{i}')]) for i in range(1, 8)]
                else:
                    q = [float(v) for v in msg.position[:7]]
                t = time.monotonic() - self.start_traj_time if self.start_traj_time else 0.0
                self.recorded_trajectory.append({
                    'time': t,
                    'q': q,
                    'grasped': self.tool_grasped
                })
            except Exception:
                pass

    def on_ee_pose(self, msg):
        self.current_ee = msg

    def on_status(self, msg):
        try:
            event = json.loads(msg.data)
            if event.get('event') == 'ready':
                self.ready = True
            self.events.append(event)
        except Exception:
            pass

    def wait_for_simulator_ready(self, timeout=25.0):
        print("⏳ Connessione alla piattaforma robotica ROS 2 in corso (/lbr_tutorial/status)...")
        start = time.monotonic()
        while rclpy.ok() and time.monotonic() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.ready and self.path_pub.get_subscription_count() > 0:
                print("✅ Simulatore KUKA LBR iiwa collegato e pronto!")
                return True
        return False

    def send_cartesian_trajectory(self, waypoints, step_name, timeout=40.0):
        """Invia una traiettoria cartesiana continua e fluida composta da più waypoints sequenziali."""
        path_msg = Path()
        path_msg.header.frame_id = 'lbr_link_0'
        for xyz, quat in waypoints:
            path_msg.poses.append(create_stamped_pose(xyz, quat))

        self.events.clear()
        self.path_pub.publish(path_msg)

        start = time.monotonic()
        req_id = None
        while rclpy.ok() and time.monotonic() - start < 10.0:
            rclpy.spin_once(self, timeout_sec=0.05)
            for ev in self.events:
                if ev.get('event') == 'accepted':
                    req_id = ev.get('request_id')
                    break
                elif ev.get('event') == 'rejected':
                    print(f"   ❌ Rifiutato dal controllore: {ev.get('reason')}")
                    return False
            if req_id is not None:
                break

        if req_id is None:
            print("   ❌ Timeout in attesa di accettazione della traiettoria")
            return False

        start = time.monotonic()
        while rclpy.ok() and time.monotonic() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            for ev in self.events:
                if ev.get('request_id') == req_id:
                    if ev.get('event') == 'succeeded':
                        target_xyz = waypoints[-1][0]
                        print(f"   -> {step_name}: Completato con moto fluido! [Arrivo: ({target_xyz[0]:.2f}, {target_xyz[1]:.2f}, {target_xyz[2]:.2f})m]")
                        return True
                    elif ev.get('event') in ('failed', 'rejected'):
                        print(f"   ❌ Errore esecuzione: {ev.get('reason')}")
                        return False

        print(f"   ❌ Timeout esecuzione per {step_name}")
        return False

    def execute_pick_and_place(self, p_centroid, destination_tray="tray_1", grasp_quat=None):
        print("\n" + "="*65)
        print("🤖 [FASE 4/4] ESECUZIONE FLUIDA PICK & PLACE (CONTINUOUS BLENDING)")
        print("="*65)

        # Coordinate dei vassoi dalla configurazione centralizzata
        env_cfg = self.config.get("environment", {})
        trays_cfg = env_cfg.get("trays", {})
        if destination_tray in trays_cfg:
            tray_key = destination_tray
        else:
            tray_key = "tray_2" if "2" in destination_tray.lower() else "tray_1"
        
        tray_data = trays_cfg.get(tray_key, {})
        p_tray = tray_data.get("position_xyz_m", [0.60, 0.42, 0.07])
        dims_tray = tray_data.get("dimensions_xyz_m", [0.62, 0.68, 0.04])
        deposit_z = p_tray[2] + dims_tray[2] / 2.0 + 0.03

        motion_cfg = self.config.get("motion_control", {})
        heights = motion_cfg.get("heights_z_m", {})
        z_approach = float(heights.get("approach_z", 0.32))
        z_lift = float(heights.get("lift_clearance_z", 0.35))
        z_staging = float(heights.get("staging_z", 0.45))
        down_quat = motion_cfg.get("downward_quaternion_xyzw", [0.0, 1.0, 0.0, 0.0])
        tool_orient = grasp_quat if grasp_quat is not None else down_quat

        grasp_z = max(float(p_centroid[2]), 0.12)
        p_grasp = [float(p_centroid[0]), float(p_centroid[1]), grasp_z]

        # Waypoints spaziali
        p_stage = [0.55, float(p_grasp[1]) * 0.5, z_staging]
        p_approach = [p_grasp[0], p_grasp[1], z_approach]
        p_lift = [p_grasp[0], p_grasp[1], z_lift]
        p_tray_above = [p_tray[0], p_tray[1], z_lift]
        p_tray_deposit = [p_tray[0], p_tray[1], deposit_z]
        p_home = [0.55, 0.0, 0.50]

        print(f"Punto Presa Strumento : ({p_grasp[0]:.3f}, {p_grasp[1]:.3f}, {p_grasp[2]:.3f})m")
        print(f"Orientamento Presa 6D : {tool_orient}")
        print(f"Vassoio di Deposito   : ({p_tray_deposit[0]:.3f}, {p_tray_deposit[1]:.3f}, {p_tray_deposit[2]:.3f})m [{tray_key}]")
        print("-" * 65)

        # Inizio registrazione leggera della traiettoria per il render offline
        self.recorded_trajectory = []
        self.start_traj_time = time.monotonic()
        self.recording_active = True
        self.tool_grasped = False

        # MACRO-SEQUENZA 1: Discesa Fluida Continua verso lo Strumento
        print("\n▶ Avvio Macro-Fase 1/3: Discesa Continua e Presa 6-DoF sullo Strumento...")
        traj_grasp = [
            (p_stage, down_quat),
            (p_approach, tool_orient),
            (p_grasp, tool_orient)
        ]
        if not self.send_cartesian_trajectory(traj_grasp, "Discesa e Presa"):
            return False

        print("   🤏 Presa fisica dello strumento attivata (comando pinza)...")
        self.gripper_pub.publish(String(data="grasp"))
        self.tool_grasped = True
        t_pause = time.monotonic()
        while rclpy.ok() and time.monotonic() - t_pause < 0.8:
            rclpy.spin_once(self, timeout_sec=0.05)

        # MACRO-SEQUENZA 2: Sollevamento, Trasferimento in Quota e Rilascio nel Vassoio
        print(f"\n▶ Avvio Macro-Fase 2/3: Trasferimento in Quota e Rilascio nel Vassoio ({tray_key})...")
        traj_place = [
            (p_lift, tool_orient),
            (p_tray_above, down_quat),
            (p_tray_deposit, down_quat)
        ]
        if not self.send_cartesian_trajectory(traj_place, f"Trasferimento verso {tray_key}"):
            return False

        print("   👐 Rilascio fisico: apertura pinza e deposito per gravità...")
        self.gripper_pub.publish(String(data="release"))
        self.tool_grasped = False
        t_pause = time.monotonic()
        while rclpy.ok() and time.monotonic() - t_pause < 0.8:
            rclpy.spin_once(self, timeout_sec=0.05)

        # MACRO-SEQUENZA 3: Risalita e Rientro Morbido in Standby
        print("\n▶ Avvio Macro-Fase 3/3: Risalita e Rientro in Standby...")
        traj_home = [
            (p_tray_above, down_quat),
            (p_home, down_quat)
        ]
        if not self.send_cartesian_trajectory(traj_home, "Rientro in Standby"):
            return False

        self.recording_active = False

        print("\n" + "=" * 65)
        print("🎉 MISSIONE CHIRURGICA COMPLETATA CON SUCCESSO (FLUIDITÀ CONTINUA)!")
        print("=" * 65)

        self.save_trajectory_and_render_video(p_grasp, p_tray_deposit)
        return True

    def save_trajectory_and_render_video(self, p_grasp, p_tray_deposit):
        # 1. Salva log della traiettoria su file JSON
        out_dir = "/workspace/data/outputs"
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "robot_trajectory_log.json")

        log_data = {
            "trajectory": self.recorded_trajectory,
            "tool_initial_xyz": p_grasp,
            "tray_final_xyz": p_tray_deposit,
            "total_points": len(self.recorded_trajectory),
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"📝 Traiettoria giunti registrata con successo: {log_path} ({len(self.recorded_trajectory)} punti)")

        # 2. Rendering offline video HD
        render_offline_video(
            trajectory_log=self.recorded_trajectory,
            p_tool_initial=p_grasp,
            p_tray_final=p_tray_deposit,
            config=self.config,
            fps=30
        )
