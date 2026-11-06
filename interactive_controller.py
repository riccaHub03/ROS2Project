#!/usr/bin/env python3
"""
Interactive Robot Controller (PyBullet KUKA LBR iiwa)
=====================================================
Permette di visualizzare e controllare interattivamente il manipolatore
chirurgico KUKA LBR iiwa nell'ambiente operatorio completo, utilizzando:
  - COMANDI W-S-A-D (Prioritari): Movimento cartesiano XY rapido e diretto
  - QUOTA Z (Q / E): Alza e abbassa la pinza
  - ROTAZIONE YAW (U / O oppure J / L): Ruota l'orientamento della pinza
  - APERTURA PINZA (Spazio): Apre e chiude le dita
  - PRESA / RILASCIO (G oppure Click Destro): Afferra o deposita lo strumento
  - MODALITÀ PUNTATORE MOUSE (Rotella Mouse oppure Tasto M): Attiva/disattiva
    l'inseguimento del cursore sul tavolo/vassoio in tempo reale
  - MONITOR CHIRURGICO SEPARATO (Tasto F): Feed live ad alta definizione della
    telecamera endoscopica montata sull'organo terminale (TCP / End-Point)
  - CAMBIO VISTA (V oppure C): Alterna tra la vista telecamera di progetto e la vista 3D

Include la vista fedele della telecamera RGB-D overhead implementata
nella configurazione (config/surgical_environment.yaml).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

# Tentativo di importazione OpenCV per la schermata separata del monitor chirurgico
try:
    import cv2
except ImportError:
    cv2 = None

# Inclusione dei percorsi di progetto per importare i moduli di pipeline
PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent

for candidate in (str(PROJECT_ROOT), str(PARENT_ROOT), "/workspace"):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from pipeline.config_loader import load_environment_config
except ImportError:
    load_environment_config = None

try:
    from pipeline.mesh_loader import find_instrument_mesh
except ImportError:
    find_instrument_mesh = None

try:
    from pipeline.sim.placement import sample_tool_pose
except Exception:
    sample_tool_pose = None

try:
    from pipeline.sim.robot_scene import prepare_mesh
except Exception:  # trimesh o dipendenze mancanti: si usa il fallback procedurale
    prepare_mesh = None

# Costanti cinematiche conformi al progetto
TCP_OFFSET = (0.0, 0.0, 0.10)  # 100 mm lungo asse Z della flangia (link 6)
HOME_JOINTS = (0.0, 0.35, 0.0, -1.3, 0.0, 1.0, 0.0)
DEFAULT_EE_ORIENTATION = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])  # Pinza rivolta verso il basso (Nadir)

# Area raggiungibile dal KUKA con pinza verso il basso, misurata con scansione IK:
# corona circolare attorno alla base (xy). Fuori da qui l'IK non converge.
REACH_R_MAX = 0.78   # m, raggio massimo orizzontale del TCP
REACH_R_MIN = 0.41   # m, raggio minimo (solo a quota di presa, vedi REACH_LOW_Z)
REACH_LOW_Z = 0.15   # m, sotto questa quota vale anche il raggio minimo
IK_OK_ERR = 0.006    # m, errore TCP accettato come soluzione IK valida
IK_MAX_ERR = 0.02    # m, oltre questo errore il target viene rifiutato


class InteractiveRobotController:
    """Controllore interattivo multimodale per il robot chirurgico KUKA LBR iiwa."""

    VIEW_CAMERA = 0   # Vista Telecamera Overhead implementata nel progetto
    VIEW_ORBIT = 1    # Vista 3D libera orbitale per ispezione
    VIEW_WRIST = 2    # Vista Eye-in-Hand solidale al TCP della pinza
    VIEW_SURGEON = 3  # Vista dalla postazione del chirurgo operatore

    VIEW_NAMES = {
        VIEW_CAMERA: "TELECAMERA OVERHEAD (Configurazione di Progetto)",
        VIEW_ORBIT: "3D ORBITALE (Libera)",
        VIEW_WRIST: "TELECAMERA POLSO (Eye-in-Hand)",
        VIEW_SURGEON: "POSTAZIONE CHIRURGO",
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        tool_name: str = "Scissors",
        source_tray: str = "tray_2",
        gui: bool = True,
        initial_view: int = VIEW_CAMERA,
        jog_speed: float = 0.35,
        turbo_multiplier: float = 2.2,
        start_with_mouse_pointer: bool = False,
        show_endoscope_window: bool = True,
        seed: Optional[int] = None,
    ):
        self.gui = gui
        self.seed = seed
        self.tool_name = tool_name
        self.active_tray = source_tray
        self.current_view = initial_view
        self.jog_speed = jog_speed
        self.turbo_multiplier = turbo_multiplier
        self.mouse_pointer_mode = start_with_mouse_pointer  # False di default: W-S-A-D è prioritario
        self.running = True

        # Configurazione telecamera endoscopica separata sul TCP ad alte prestazioni
        self.show_endoscope_window = (cv2 is not None) and self.gui and show_endoscope_window
        self.endoscope_window_name = "Monitor Chirurgico - Telecamera Endoscopica TCP"
        self.endoscope_render_w = 256
        self.endoscope_render_h = 192
        self.endoscope_display_w = 576
        self.endoscope_display_h = 432
        self.step_count = 0
        self.endoscope_window_created = False
        self.last_cam_time = 0.0
        self.cam_interval = 1.0 / 10.0  # 10 FPS: il rendering software non deve bloccare il loop GUI

        # Renderer software: l'OpenGL hardware nello stesso thread della GUI causa scatti
        self.camera_renderer = p.ER_TINY_RENDERER

        # Precalcolo maschera di vignettatura per evitare ricalcoli costosi ad ogni frame
        self.vignette_mask = self._precompute_vignette(self.endoscope_display_w, self.endoscope_display_h)
        self.axis_line_ids = [-1, -1, -1]

        # Caricamento configurazione dell'ambiente
        self.config = self._load_config(config_path)

        # Connessione a PyBullet
        conn_mode = p.GUI if self.gui else p.DIRECT
        self.cid = p.connect(conn_mode)
        if self.cid < 0:
            raise RuntimeError("Impossibile connettersi al motore fisico PyBullet.")

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.cid)
        p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=self.cid)

        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.cid)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.cid)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.cid)
            # Le scorciatoie native di PyBullet usano gli stessi tasti W/A/S/D/G/J/K...:
            # W attiva il wireframe, A i riquadri rossi AABB, G nasconde i pannelli.
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=self.cid)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=self.cid)
            # Le anteprime RGB/Depth/Maschera integrate rendono la scena ad ogni frame: spente
            for flag in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                         p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
                p.configureDebugVisualizer(flag, 0, physicsClientId=self.cid)

        # Client DIRECT separato solo cinematico: serve a verificare l'IK senza muovere il robot reale
        self.ik_cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.ik_cid)
        self.ik_robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.ik_cid)

        # Stato interno del robot e degli attuatori
        self.robot_id = -1
        self.gripper_id = -1
        self.endoscope_cam_id = -1
        self.tool_id = -1
        self.tool_mesh_file = None
        self.attachment_constraint = None
        self.gripper_width = 0.080  # Apertura massima 80 mm
        self.target_gripper_width = 0.080

        # Parametri cinematici del TCP e target IK
        self.target_tcp_pos = np.array([0.60, -0.42, 0.28], dtype=float)
        self.target_tcp_yaw = 0.0
        self.target_tcp_orn = np.array(DEFAULT_EE_ORIENTATION, dtype=float)
        self.travel_z = 0.28  # Quota di navigazione orizzontale sicura

        # Stato del mouse
        self.mouse_world_target = np.array([0.60, -0.42, 0.08], dtype=float)
        self.mouse_cursor_id = -1
        self.mouse_in_window = False

        # Macchina a stati per azione Pick & Place animata su click sinistro
        self.action_state: Optional[str] = None
        self.action_timer: float = 0.0
        self.action_target_pos = np.array([0.60, -0.42, 0.09], dtype=float)

        # Tracciamento e debug a schermo
        self.hud_text_id = -1
        self.hud_keys_id = -1
        self.trail_points: List[np.ndarray] = []
        self.trail_lines: List[int] = []
        self.max_trail_points = 120

        self.last_valid_target = self.target_tcp_pos.copy()
        self.ik_failed_frames = 0
        self._hud_cache: Tuple[str, str] = ("", "")
        self.slider_last: Dict[str, float] = {}

        # Slider GUI
        self.sliders: Dict[str, int] = {}
        self.buttons: Dict[str, int] = {}
        self.last_button_clicks: Dict[str, int] = {}

        # Costruzione della scena
        self._build_environment()
        self._build_robot()
        self._build_tool()
        self._init_mouse_cursor()
        self._init_gui_controls()

        # Inizializza visuale telecamera
        self.set_camera_view(self.current_view)
        self.sync_gripper()

        # Inizializza il target TCP sulla posa effettiva
        actual_pos, actual_orn = self.get_tcp_pose()
        self.target_tcp_pos = np.array(actual_pos, dtype=float)
        self.target_tcp_orn = np.array(actual_orn, dtype=float)
        self.last_valid_target = self.target_tcp_pos.copy()

        if self.show_endoscope_window:
            print(f"[Monitor] Telecamera Endoscopica TCP attiva in finestra separata: '{self.endoscope_window_name}'")

    # -------------------------------------------------------------------------
    # Inizializzazione della Scena
    # -------------------------------------------------------------------------
    def _load_config(self, config_path: Optional[str]) -> dict:
        if load_environment_config is not None:
            try:
                return load_environment_config(config_path)
            except Exception as exc:
                print(f"[Config] Avviso: fallback configurazione ({exc})")

        # Configurazione minima di sicurezza coerente con surgical_environment.yaml
        return {
            "environment": {
                "kuka_base": {"position_xyz_m": [0.0, 0.0, 0.0]},
                "surgical_table": {
                    "center_xyz_m": [0.60, 0.0, -0.05],
                    "dimensions_xyz_m": [0.85, 1.60, 0.20],
                    "color_rgba": [0.15, 0.35, 0.60, 1.0],
                },
                "trays": {
                    "tray_1": {
                        "name": "Vassoio Raccolta Destro (+Y)",
                        "position_xyz_m": [0.60, 0.42, 0.07],
                        "dimensions_xyz_m": [0.62, 0.68, 0.04],
                        "color_rgba": [0.80, 0.82, 0.85, 1.0],
                    },
                    "tray_2": {
                        "name": "Vassoio Strumenti Sinistro (-Y)",
                        "position_xyz_m": [0.60, -0.42, 0.07],
                        "dimensions_xyz_m": [0.62, 0.68, 0.04],
                        "color_rgba": [0.80, 0.82, 0.85, 1.0],
                    },
                },
            },
            "camera_overhead": {
                "auto_track_source_tray": True,
                "height_above_tray_m": 1.08,
                "extrinsics": {
                    "position_robot_frame_xyz_m": [0.60, -0.42, 1.15],
                    "target_position_xyz_m": [0.60, -0.42, 0.07],
                },
                "intrinsics": {"fx_px": 615.0, "fy_px": 615.0, "cx_px": 320.0, "cy_px": 240.0},
                "width_px": 640,
                "height_px": 480,
            },
        }

    def _create_box(self, position, dimensions, color):
        half = (np.asarray(dimensions, dtype=float) / 2.0).tolist()
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.cid)
        return p.createMultiBody(0, col, vis, position, physicsClientId=self.cid)

    def _build_environment(self):
        env = self.config.get("environment", {})

        # Pavimento di sicurezza
        p.loadURDF("plane.urdf", [0, 0, -0.16], physicsClientId=self.cid)

        # Tavolo chirurgico sterile principale
        table = env.get("surgical_table", {})
        self.table_id = self._create_box(
            table.get("center_xyz_m", [0.60, 0.0, -0.05]),
            table.get("dimensions_xyz_m", [0.85, 1.60, 0.20]),
            table.get("color_rgba", [0.15, 0.35, 0.60, 1.0]),
        )

        # Vassoi chirurgici
        self.trays = env.get("trays", {})
        self.tray_bodies = {}
        for name, tray in self.trays.items():
            pos = tray.get("position_xyz_m", [0.60, 0.0, 0.07])
            dims = tray.get("dimensions_xyz_m", [0.62, 0.68, 0.04])
            color = tray.get("color_rgba", [0.80, 0.82, 0.85, 1.0])
            self.tray_bodies[name] = self._create_box(pos, dims, color)

            if self.gui:
                label_pos = [pos[0] - 0.22, pos[1], pos[2] + 0.06]
                p.addUserDebugText(
                    f"{tray.get('name', name)}",
                    label_pos,
                    textColorRGB=[0.1, 0.2, 0.4],
                    textSize=1.1,
                    physicsClientId=self.cid,
                )

    def _build_robot(self):
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.cid)
        self.joints = list(range(7))
        self.ee_link = 6
        self.limits = [p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)[8:10] for j in self.joints]

        # Inizializza i giunti alla posa HOME
        for j, q in zip(self.joints, HOME_JOINTS):
            p.resetJointState(self.robot_id, j, q, physicsClientId=self.cid)
        self._command_joints(HOME_JOINTS)

        # Pinza parallela a 2 dita visuale montata sul link 6
        palm = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.032, 0.025, 0.0125],
            visualFramePosition=[0, 0, 0.025],
            rgbaColor=[0.18, 0.20, 0.24, 1.0],
            physicsClientId=self.cid,
        )
        finger = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.013, 0.004, 0.025],
            rgbaColor=[0.24, 0.28, 0.35, 1.0],
            physicsClientId=self.cid,
        )
        self.gripper_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=palm,
            baseCollisionShapeIndex=-1,
            linkMasses=[0, 0],
            linkCollisionShapeIndices=[-1, -1],
            linkVisualShapeIndices=[finger, finger],
            linkPositions=[[0, 0.005, 0.075], [0, -0.005, 0.075]],
            linkOrientations=[[0, 0, 0, 1]] * 2,
            linkInertialFramePositions=[[0, 0, 0]] * 2,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_PRISMATIC] * 2,
            linkJointAxis=[[0, 1, 0], [0, -1, 0]],
            physicsClientId=self.cid,
        )

        # Miniatura fisica 3D della telecamera endoscopica montata sul polso del robot
        cam_tube = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.006,
            length=0.032,
            visualFramePosition=[0.0, -0.042, 0.040],
            visualFrameOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            rgbaColor=[0.08, 0.10, 0.14, 1.0],
            physicsClientId=self.cid,
        )
        self.endoscope_cam_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=cam_tube,
            basePosition=[0, 0, 0],
            physicsClientId=self.cid,
        )

    def _build_tool(self):
        """Carica lo strumento chirurgico nel vassoio attivo (con fallback robusto)."""
        tray = self.trays.get(self.active_tray, next(iter(self.trays.values())))
        tray_pos = tray.get("position_xyz_m", [0.60, -0.42, 0.07])
        tray_dims = tray.get("dimensions_xyz_m", [0.62, 0.68, 0.04])
        top_z = tray_pos[2] + tray_dims[2] / 2.0

        mesh_path = None
        if find_instrument_mesh is not None:
            try:
                mesh_path = find_instrument_mesh(target_keyword=self.tool_name, config=self.config)
            except Exception:
                mesh_path = None

        col = vis = None
        if mesh_path and os.path.exists(mesh_path) and prepare_mesh is not None:
            try:
                # Stessa preparazione della pipeline: scala in metri, centrata, adagiata, semplificata
                prepared, info = prepare_mesh(mesh_path)
                dims = [float(d) for d in info["dimensions_m"]]
                dims = [max(d, 0.004) for d in dims]
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d / 2 for d in dims], physicsClientId=self.cid)
                vis = p.createVisualShape(p.GEOM_MESH, fileName=prepared,
                                          rgbaColor=[0.35, 0.38, 0.42, 1.0], physicsClientId=self.cid)
                self.tool_mesh_file = prepared
            except Exception as exc:
                print(f"[Strumento] Mesh non preparabile ({exc}): uso geometria procedurale")
                col = vis = None

        if col is None:
            # Fallback procedurale: geometria realistica di strumento chirurgico
            dims = [0.14, 0.025, 0.006]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2], physicsClientId=self.cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2],
                                      rgbaColor=[0.25, 0.28, 0.32, 1.0], physicsClientId=self.cid)

        x, y, yaw = tray_pos[0], tray_pos[1], 0.2
        if sample_tool_pose is not None:
            x, y, yaw = sample_tool_pose(tray_pos, tray_dims, dims[:2], self.seed)
        spawn_pos = [x, y, top_z + dims[2] / 2.0 + 0.002]
        spawn_orn = p.getQuaternionFromEuler([0, 0, yaw])
        # Fermo finche' non viene afferrato (diventa dinamico alla presa) e senza urti col braccio:
        # prima un link che sfiorava uno strumento grande lo scagliava fuori dal vassoio.
        self.tool_id = p.createMultiBody(0.0, col, vis, spawn_pos, spawn_orn, physicsClientId=self.cid)
        for j in range(-1, 7):
            p.setCollisionFilterPair(self.robot_id, self.tool_id, j, -1, 0, physicsClientId=self.cid)
        self.tool_initial_pos = spawn_pos
        self.tool_initial_orn = spawn_orn

    def _init_mouse_cursor(self):
        """Crea un indicatore grafico 3D che segue il puntatore del mouse quando la modalità puntatore è attiva."""
        if not self.gui:
            return
        vis_cursor = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.016,
            rgbaColor=[0.0, 0.85, 1.0, 0.70],  # Ciano luminoso semitrasparente
            physicsClientId=self.cid,
        )
        init_pos = [0.60, -0.42, 0.08] if self.mouse_pointer_mode else [0.0, 0.0, -10.0]
        self.mouse_cursor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_cursor,
            basePosition=init_pos,
            physicsClientId=self.cid,
        )

    def _init_gui_controls(self):
        """Inizializza gli slider e i pulsanti interattivi nella GUI PyBullet."""
        if not self.gui:
            return

        self.sliders["tcp_z"] = p.addUserDebugParameter("Quota Z Altezza [m]", 0.065, 0.55, float(self.travel_z), physicsClientId=self.cid)
        self.sliders["tcp_yaw"] = p.addUserDebugParameter("Target Yaw [deg]", -180.0, 180.0, 0.0, physicsClientId=self.cid)
        self.sliders["gripper"] = p.addUserDebugParameter("Apertura Pinza [mm]", 0.0, 80.0, 80.0, physicsClientId=self.cid)

        self.buttons["toggle_mouse_mode"] = p.addUserDebugParameter("-> Toggle Modo Puntatore (Rotella / M)", 1, 0, 0, physicsClientId=self.cid)
        self.buttons["toggle_endoscope"] = p.addUserDebugParameter("-> Toggle Monitor Chirurgico (F)", 1, 0, 0, physicsClientId=self.cid)
        self.buttons["reset_home"] = p.addUserDebugParameter("-> Reset Posa HOME (H)", 1, 0, 0, physicsClientId=self.cid)
        self.buttons["toggle_view"] = p.addUserDebugParameter("-> Cambia Vista Telecamera (V)", 1, 0, 0, physicsClientId=self.cid)
        self.buttons["toggle_grasp"] = p.addUserDebugParameter("-> Presa / Rilascio Strumento (G / Click DX)", 1, 0, 0, physicsClientId=self.cid)
        self.buttons["switch_tray"] = p.addUserDebugParameter("-> Centra Vassoio 1 / 2 (T)", 1, 0, 0, physicsClientId=self.cid)

        for btn in self.buttons:
            self.last_button_clicks[btn] = 0
        self.slider_last = {"tcp_z": float(self.travel_z), "tcp_yaw": 0.0, "gripper": 80.0}

    # -------------------------------------------------------------------------
    # Gestione delle Viste della Telecamera (PyBullet)
    # -------------------------------------------------------------------------
    def set_camera_view(self, view_mode: int):
        """Imposta la telecamera della finestra principale interattiva."""
        self.current_view = view_mode % 4
        if not self.gui:
            return

        tray = self.trays.get(self.active_tray, next(iter(self.trays.values())))
        tray_pos = tray.get("position_xyz_m", [0.60, -0.42, 0.07])
        cam_cfg = self.config.get("camera_overhead", {})
        height = cam_cfg.get("height_above_tray_m", 1.08)

        if self.current_view == self.VIEW_CAMERA:
            # Vista Overhead fedele implementata nel progetto (Nadir)
            cam_target = [tray_pos[0], tray_pos[1], tray_pos[2]]
            p.resetDebugVisualizerCamera(
                cameraDistance=height,
                cameraYaw=-90.0,
                cameraPitch=-89.9,
                cameraTargetPosition=cam_target,
                physicsClientId=self.cid,
            )
        elif self.current_view == self.VIEW_ORBIT:
            # Vista 3D orbitale libera per ispezione
            p.resetDebugVisualizerCamera(
                cameraDistance=1.80,
                cameraYaw=48.0,
                cameraPitch=-32.0,
                cameraTargetPosition=[0.45, 0.0, 0.25],
                physicsClientId=self.cid,
            )
        elif self.current_view == self.VIEW_WRIST:
            # Vista Eye-in-Hand: la telecamera segue la pinza
            tcp_pos, _ = self.get_tcp_pose()
            p.resetDebugVisualizerCamera(
                cameraDistance=0.40,
                cameraYaw=-90.0,
                cameraPitch=-85.0,
                cameraTargetPosition=tcp_pos,
                physicsClientId=self.cid,
            )
        elif self.current_view == self.VIEW_SURGEON:
            # Vista operatore / chirurgo (dal lato opposto del tavolo)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.35,
                cameraYaw=180.0,
                cameraPitch=-28.0,
                cameraTargetPosition=[0.60, 0.0, 0.10],
                physicsClientId=self.cid,
            )

        self._update_hud()

    def cycle_camera_view(self):
        """Passa ciclicamente alla vista successiva nella finestra PyBullet."""
        self.set_camera_view(self.current_view + 1)

    # -------------------------------------------------------------------------
    # Telecamera Endoscopica TCP (Schermata Separata Monitor Chirurgico)
    # -------------------------------------------------------------------------
    def _precompute_vignette(self, w: int, h: int) -> Optional[np.ndarray]:
        """Precalcola la vignettatura endoscopica una sola volta in memoria per eliminare i colli di bottiglia CPU."""
        if cv2 is None:
            return None
        try:
            cx, cy = w // 2, h // 2
            vignette = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(vignette, (cx, cy), int(max(w, h) * 0.58), 255, -1)
            vignette_blur = cv2.GaussianBlur(vignette, (45, 45), 0)
            return cv2.cvtColor(vignette_blur, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None

    def get_endoscope_camera_matrices(self) -> Tuple[List[float], List[float], float, float]:
        """Calcola le matrici view e projection per la telecamera endoscopica solidale all'end-point."""
        state = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True, physicsClientId=self.cid)
        ee_pos = state[4]
        ee_orn = state[5]

        # Posizione e target locali nel sistema di coordinate del link 6 (end-point)
        # La telecamera è montata accanto alle dita della pinza e mira verso il punto di presa/tavolo
        cam_local_pos = [0.0, -0.042, 0.040]
        cam_local_target = [0.0, 0.0, 0.220]
        cam_local_up = [-1.0, 0.0, 0.0]

        cam_world_pos, _ = p.multiplyTransforms(ee_pos, ee_orn, cam_local_pos, [0, 0, 0, 1])
        cam_world_target, _ = p.multiplyTransforms(ee_pos, ee_orn, cam_local_target, [0, 0, 0, 1])

        rot_mat = np.asarray(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        cam_world_up = rot_mat @ np.array(cam_local_up, dtype=float)

        view_mat = p.computeViewMatrix(cam_world_pos, cam_world_target, cam_world_up.tolist())
        fov = 68.0  # Ottica grandangolare endoscopica tipica laparoscopia
        aspect = float(self.endoscope_render_w) / float(self.endoscope_render_h)
        near, far = 0.015, 2.0
        proj_mat = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        return view_mat, proj_mat, near, far

    def _update_endoscope_window(self):
        """Genera e visualizza il feed video in tempo reale della telecamera endoscopica in una finestra separata ad alti FPS."""
        if cv2 is None or not self.show_endoscope_window:
            return

        now = time.monotonic()
        if (now - self.last_cam_time) < self.cam_interval:
            return
        self.last_cam_time = now

        try:
            view_mat, proj_mat, near, far = self.get_endoscope_camera_matrices()
            rw, rh = self.endoscope_render_w, self.endoscope_render_h
            dw, dh = self.endoscope_display_w, self.endoscope_display_h

            # Tenta prima con il renderer hardware OpenGL; se fallisce o non supportato nel contesto, fallback a tiny
            try:
                rendered = p.getCameraImage(
                    rw, rh, view_mat, proj_mat,
                    renderer=self.camera_renderer,
                    physicsClientId=self.cid,
                )
            except Exception:
                self.camera_renderer = p.ER_TINY_RENDERER
                rendered = p.getCameraImage(
                    rw, rh, view_mat, proj_mat,
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=self.cid,
                )

            rgb = np.asarray(rendered[2], dtype=np.uint8).reshape((rh, rw, 4))[:, :, :3]
            d_buf = np.asarray(rendered[3], dtype=np.float32).reshape((rh, rw))
            depth_m = (far * near) / (far - (far - near) * d_buf)

            # Conversione BGR e upscaling nitido per la visualizzazione a schermo
            frame_raw = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame_raw, (dw, dh), interpolation=cv2.INTER_LINEAR)

            # Applicazione istantanea della maschera di vignettatura precalcolata (zero lag CPU)
            if self.vignette_mask is not None:
                frame = cv2.multiply(frame, self.vignette_mask, scale=1.0 / 255.0)

            # Telemetria laser: misura la distanza esatta al centro del mirino
            center_dist_cm = float(depth_m[rh // 2, rw // 2]) * 100.0
            cx, cy = int(dw // 2), int(dh // 2)

            # -------------------------------------------------------------
            # Overlay Grafico HUD Stile Monitor Chirurgico (ad alta definizione)
            # -------------------------------------------------------------
            in_range = center_dist_cm < 16.0
            reticle_color = (0, 255, 60) if in_range else (0, 220, 255)  # Verde se vicino, Ciano se lontano

            # 1. Mirino chirurgico centrale con crocetta
            cv2.line(frame, (cx - 28, cy), (cx - 8, cy), reticle_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx + 8, cy), (cx + 28, cy), reticle_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 28), (cx, cy - 8), reticle_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy + 8), (cx, cy + 28), reticle_color, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, reticle_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 38, reticle_color, 1, cv2.LINE_AA)

            # Parentesi di puntamento (Bracketing)
            b_len, b_box = 14, 52
            # Angolo alto-sinistra
            cv2.line(frame, (cx - b_box, cy - b_box), (cx - b_box + b_len, cy - b_box), reticle_color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx - b_box, cy - b_box), (cx - b_box, cy - b_box + b_len), reticle_color, 2, cv2.LINE_AA)
            # Angolo alto-destra
            cv2.line(frame, (cx + b_box, cy - b_box), (cx + b_box - b_len, cy - b_box), reticle_color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx + b_box, cy - b_box), (cx + b_box, cy - b_box + b_len), reticle_color, 2, cv2.LINE_AA)
            # Angolo basso-sinistra
            cv2.line(frame, (cx - b_box, cy + b_box), (cx - b_box + b_len, cy + b_box), reticle_color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx - b_box, cy + b_box), (cx - b_box, cy + b_box - b_len), reticle_color, 2, cv2.LINE_AA)
            # Angolo basso-destra
            cv2.line(frame, (cx + b_box, cy + b_box), (cx + b_box - b_len, cy + b_box), reticle_color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx + b_box, cy + b_box), (cx + b_box, cy + b_box - b_len), reticle_color, 2, cv2.LINE_AA)

            # Telemetria laser accanto al mirino
            range_text = f"DIST: {center_dist_cm:.1f} cm"
            if in_range:
                range_text += " [IN RANGE]"
            cv2.putText(frame, range_text, (cx + 46, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, reticle_color, 1, cv2.LINE_AA)

            # 2. Barra superiore con branding chirurgico e live indicator
            cv2.rectangle(frame, (0, 0), (dw, 28), (15, 18, 22), -1)
            cv2.line(frame, (0, 28), (dw, 28), (40, 50, 60), 1)

            # Pallino rosso lampeggiante
            rec_color = (0, 0, 255) if (int(time.time() * 2) % 2 == 0) else (0, 0, 140)
            cv2.circle(frame, (16, 14), 5, rec_color, -1, cv2.LINE_AA)
            cv2.putText(frame, "LIVE", (28, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, "ENDOSCOPIO CHIRURGICO TCP", (82, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "KUKA iiwa STERILE", (dw - 160, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 180, 200), 1, cv2.LINE_AA)

            # 3. Barra inferiore con stato di presa
            cv2.rectangle(frame, (0, dh - 30), (dw, dh), (15, 18, 22), -1)
            cv2.line(frame, (0, dh - 30), (dw, dh - 30), (40, 50, 60), 1)

            grip_str = f"PINZA: {self.gripper_width*1000:.0f} mm ({'APERTA' if self.gripper_width > 0.04 else 'CHIUSA'})"
            tool_status = "PRESO" if self.attachment_constraint else "LIBERO"
            status_color = (0, 255, 60) if self.attachment_constraint else (180, 180, 180)
            status_text = f"STRUMENTO ({self.tool_name}): {tool_status}"

            cv2.putText(frame, grip_str, (14, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 210, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, status_text, (dw - 230, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)

            # Creazione finestra ridimensionabile al primo avvio
            if not self.endoscope_window_created:
                cv2.namedWindow(self.endoscope_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.endoscope_window_name, dw, dh)
                self.endoscope_window_created = True

            cv2.imshow(self.endoscope_window_name, frame)
            cv2.waitKey(1)

        except Exception as exc:
            print(f"[Monitor] Avviso visualizzazione endoscopio: {exc}")
            self.show_endoscope_window = False

    # -------------------------------------------------------------------------
    # Cinematica e Movimento Robot
    # -------------------------------------------------------------------------
    def get_tcp_pose(self) -> Tuple[List[float], List[float]]:
        state = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True, physicsClientId=self.cid)
        return p.multiplyTransforms(state[4], state[5], TCP_OFFSET, [0, 0, 0, 1])

    def _command_joints(self, joint_angles):
        p.setJointMotorControlArray(
            self.robot_id,
            self.joints,
            p.POSITION_CONTROL,
            targetPositions=list(joint_angles),
            forces=[600.0] * 7,
            positionGains=[0.65] * 7,
            velocityGains=[1.0] * 7,
            physicsClientId=self.cid,
        )

    def sync_gripper(self):
        """Allinea visivamente la pinza e il corpo della telecamera al link 6 del robot."""
        state = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True, physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.gripper_id, state[4], state[5], physicsClientId=self.cid)
        if hasattr(self, "endoscope_cam_id") and self.endoscope_cam_id >= 0:
            p.resetBasePositionAndOrientation(self.endoscope_cam_id, state[4], state[5], physicsClientId=self.cid)

        q = max(0.0, (self.gripper_width - 0.002) / 2.0)
        p.resetJointState(self.gripper_id, 0, q, physicsClientId=self.cid)
        p.resetJointState(self.gripper_id, 1, q, physicsClientId=self.cid)

    def clamp_to_workspace(self, pos) -> np.ndarray:
        """Proietta il target nella zona realmente raggiungibile dal KUKA (corona circolare attorno alla base)."""
        pos = np.array(pos, dtype=float)
        pos[2] = np.clip(pos[2], 0.065, 0.58)
        r = math.hypot(pos[0], pos[1])
        r_min = REACH_R_MIN if pos[2] < REACH_LOW_Z else 0.0
        if r < 1e-6:
            pos[0], pos[1], r = max(r_min, 0.3), 0.0, max(r_min, 0.3)
        target_r = float(np.clip(r, r_min, REACH_R_MAX))
        if abs(target_r - r) > 1e-9:
            pos[0] *= target_r / r
            pos[1] *= target_r / r
        return pos

    def _ik_error_and_q(self, flange_pos, target_orn, seed):
        """IK sul client cinematico separato: ritorna (errore TCP in m, giunti)."""
        cid = self.ik_cid
        for j, q in enumerate(seed):
            p.resetJointState(self.ik_robot, j, q, physicsClientId=cid)
        q = list(seed)
        for _ in range(3):  # piu' passate dal punto trovato per convergere
            q = list(p.calculateInverseKinematics(
                self.ik_robot, self.ee_link, flange_pos, target_orn,
                lowerLimits=[lim[0] for lim in self.limits],
                upperLimits=[lim[1] for lim in self.limits],
                jointRanges=[lim[1] - lim[0] for lim in self.limits],
                restPoses=list(HOME_JOINTS),
                jointDamping=[0.01] * 7,
                maxNumIterations=60,
                residualThreshold=1e-5,
                physicsClientId=cid,
            )[:7])
            q = [max(lo, min(hi, v)) for v, (lo, hi) in zip(q, self.limits)]
            for j, v in enumerate(q):
                p.resetJointState(self.ik_robot, j, v, physicsClientId=cid)
        state = p.getLinkState(self.ik_robot, self.ee_link, computeForwardKinematics=True, physicsClientId=cid)
        err = float(np.linalg.norm(np.asarray(state[4]) - np.asarray(flange_pos)))
        return err, q

    def solve_and_apply_ik(self, target_pos, target_orn) -> bool:
        """Risolve l'IK con piu' punti di partenza e comanda i giunti solo se la soluzione e' affidabile."""
        flange_pos, _ = p.multiplyTransforms(target_pos, target_orn, [0, 0, -TCP_OFFSET[2]], [0, 0, 0, 1])
        current = [p.getJointState(self.robot_id, j, physicsClientId=self.cid)[0] for j in self.joints]
        base_angle = math.atan2(target_pos[1], target_pos[0])
        mirrored = [max(self.limits[0][0], min(self.limits[0][1], base_angle))] + list(HOME_JOINTS[1:])

        best_err, best_q = None, None
        for seed in (current, mirrored, list(HOME_JOINTS)):
            err, q = self._ik_error_and_q(flange_pos, target_orn, seed)
            if best_err is None or err < best_err:
                best_err, best_q = err, q
            if best_err < IK_OK_ERR:
                break

        if best_err > IK_MAX_ERR:
            return False
        self._command_joints(best_q)
        return True

    # -------------------------------------------------------------------------
    # Presa e Rilascio Strumento (Grasp Constraint)
    # -------------------------------------------------------------------------
    def toggle_grasp(self, force_release=False, force_attach=False):
        """Attacca o stacca lo strumento chirurgico dalla pinza."""
        if self.attachment_constraint is not None and not force_attach:
            # Rilascio
            p.removeConstraint(self.attachment_constraint, physicsClientId=self.cid)
            self.attachment_constraint = None
            self.target_gripper_width = 0.080
            print("[Grasp] Strumento rilasciato.")
            return False
        elif self.attachment_constraint is None and not force_release:
            # Presa
            tcp_pos, _ = self.get_tcp_pose()
            tool_pos, tool_orn = p.getBasePositionAndOrientation(self.tool_id, physicsClientId=self.cid)
            dist = np.linalg.norm(np.asarray(tcp_pos) - np.asarray(tool_pos))

            if dist > 0.22:
                print(f"[Grasp] Distanza eccessiva per afferrare: {dist*1000:.1f} mm. Avvicinati allo strumento.")
                return False

            state = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True, physicsClientId=self.cid)
            inv = p.invertTransform(state[0], state[1])
            relative = p.multiplyTransforms(*inv, tool_pos, tool_orn)

            p.changeDynamics(self.tool_id, -1, mass=0.080, physicsClientId=self.cid)
            self.attachment_constraint = p.createConstraint(
                self.robot_id,
                self.ee_link,
                self.tool_id,
                -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                relative[0],
                [0, 0, 0],
                parentFrameOrientation=relative[1],
                childFrameOrientation=[0, 0, 0, 1],
                physicsClientId=self.cid,
            )
            p.changeConstraint(self.attachment_constraint, maxForce=600, physicsClientId=self.cid)
            for j in range(-1, 7):
                p.setCollisionFilterPair(self.robot_id, self.tool_id, j, -1, 0, physicsClientId=self.cid)
            self.target_gripper_width = 0.015
            print("[Grasp] Strumento agganciato con successo alla pinza.")
            return True
        return False

    def reset_home(self):
        """Ripristina la posa HOME del robot."""
        for j, q in zip(self.joints, HOME_JOINTS):
            p.resetJointState(self.robot_id, j, q, physicsClientId=self.cid)
        self._command_joints(HOME_JOINTS)
        actual_pos, actual_orn = self.get_tcp_pose()
        self.target_tcp_pos = np.array(actual_pos, dtype=float)
        self.target_tcp_yaw = 0.0
        self.target_tcp_orn = np.array(actual_orn, dtype=float)
        self.target_gripper_width = 0.080
        self.gripper_width = 0.080
        self.travel_z = 0.28
        self.action_state = None
        self.trail_points.clear()
        print("[Robot] Ripristinata posa HOME.")

    def switch_tray_focus(self):
        """Alterna il focus tra Vassoio 1 e Vassoio 2."""
        self.active_tray = "tray_1" if self.active_tray == "tray_2" else "tray_2"
        tray = self.trays[self.active_tray]
        tray_pos = tray.get("position_xyz_m", [0.60, 0.0, 0.07])
        self.target_tcp_pos[0] = tray_pos[0]
        self.target_tcp_pos[1] = tray_pos[1]
        self.target_tcp_pos[2] = max(self.target_tcp_pos[2], 0.25)
        print(f"[Ambiente] Focus impostato su: {tray.get('name', self.active_tray)}")
        if self.current_view == self.VIEW_CAMERA:
            self.set_camera_view(self.VIEW_CAMERA)

    # -------------------------------------------------------------------------
    # Raycasting e Controllo tramite Puntatore del Mouse
    # -------------------------------------------------------------------------
    def _raycast_mouse_to_table(self, mouse_x: float, mouse_y: float, plane_z: float = 0.08) -> Optional[np.ndarray]:
        """Converte le coordinate 2D del puntatore del mouse (pixel) in un punto 3D sul piano del vassoio."""
        if not self.gui:
            return None

        try:
            cam = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        except Exception:
            return None

        if cam is None or len(cam) < 4:
            return None

        width, height = cam[0], cam[1]
        if width <= 0 or height <= 0:
            return None

        # Se il mouse è sopra il pannello dei parametri a destra, non effettuare raycast 3D
        if mouse_x > (width - 240):
            return None

        view_mat = np.array(cam[2], dtype=float).reshape((4, 4), order='F')
        proj_mat = np.array(cam[3], dtype=float).reshape((4, 4), order='F')

        pv = proj_mat @ view_mat
        try:
            inv_pv = np.linalg.inv(pv)
        except np.linalg.LinAlgError:
            return None

        # Coordinate normalizzate del dispositivo (NDC da -1 a +1)
        ndc_x = (2.0 * mouse_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * mouse_y / height)

        near_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_clip = np.array([ndc_x, ndc_y, 1.0, 1.0])

        near_world = inv_pv @ near_clip
        if abs(near_world[3]) < 1e-8:
            return None
        near_world /= near_world[3]

        far_world = inv_pv @ far_clip
        if abs(far_world[3]) < 1e-8:
            return None
        far_world /= far_world[3]

        ray_dir = far_world[:3] - near_world[:3]
        norm = np.linalg.norm(ray_dir)
        if norm < 1e-6 or abs(ray_dir[2]) < 1e-5:
            return None
        ray_dir /= norm

        # Intersezione raggio con il piano orizzontale Z = plane_z (superficie vassoio)
        t = (plane_z - near_world[2]) / ray_dir[2]
        if t <= 0:
            return None

        hit_x = near_world[0] + t * ray_dir[0]
        hit_y = near_world[1] + t * ray_dir[1]

        # Limita al volume realmente raggiungibile dal KUKA
        return self.clamp_to_workspace([hit_x, hit_y, max(plane_z, 0.065)])

    def _poll_mouse(self):
        """Elabora gli eventi del mouse: il raycast viene fatto una sola volta per frame sull'ultimo movimento."""
        if not self.gui:
            return

        try:
            events = p.getMouseEvents(physicsClientId=self.cid)
        except Exception:
            return

        last_move = None
        for ev in events:
            if len(ev) < 5:
                continue
            ev_type, mx, my, btn_idx, btn_state = ev[:5]
            self.mouse_in_window = True

            if ev_type == 1:  # MOUSE_MOVE_EVENT
                last_move = (mx, my)
                continue

            if not (btn_state & p.KEY_WAS_TRIGGERED):
                continue

            # Click Centrale (Rotella): Attiva / Disattiva la modalità puntatore mouse
            if btn_idx == 1:
                self.mouse_pointer_mode = not self.mouse_pointer_mode
                if self.mouse_pointer_mode:
                    actual_pos, _ = self.get_tcp_pose()
                    self.mouse_world_target[:2] = actual_pos[:2]
                    print("[Rotella Mouse] MODALITÀ PUNTATORE ATTIVATA (Il robot segue il cursore)")
                else:
                    self._hide_mouse_cursor()
                    print("[Rotella Mouse] MODALITÀ PUNTATORE DISATTIVATA (Controllo W-S-A-D prioritario)")

            # Click Sinistro: in modalità puntatore, Pick & Place sul punto cliccato
            elif btn_idx == 0 and self.mouse_pointer_mode and self.action_state is None:
                world_pt = self._raycast_mouse_to_table(mx, my, plane_z=0.08)
                if world_pt is not None:
                    self.action_state = "DESCEND"
                    self.action_timer = 0.0
                    self.action_target_pos = np.array([world_pt[0], world_pt[1], 0.095], dtype=float)
                    action_type = "Deposito / Rilascio" if self.attachment_constraint else "Presa"
                    print(f"[Click Sinistro] Operazione {action_type} sul punto [{world_pt[0]:.3f}, {world_pt[1]:.3f}]")

            # Click Destro: Presa / Rilascio immediato
            elif btn_idx == 2:
                self.toggle_grasp()

        if last_move is not None and self.mouse_pointer_mode:
            world_pt = self._raycast_mouse_to_table(last_move[0], last_move[1], plane_z=0.08)
            if world_pt is not None:
                self.mouse_world_target = world_pt
                if self.mouse_cursor_id >= 0:
                    p.resetBasePositionAndOrientation(
                        self.mouse_cursor_id,
                        [world_pt[0], world_pt[1], world_pt[2] + 0.005],
                        [0, 0, 0, 1],
                        physicsClientId=self.cid,
                    )

    def _hide_mouse_cursor(self):
        if self.mouse_cursor_id >= 0:
            p.resetBasePositionAndOrientation(self.mouse_cursor_id, [0.0, 0.0, -10.0], [0, 0, 0, 1],
                                              physicsClientId=self.cid)

    def _update_pick_place_state_machine(self, dt: float):
        """Gestisce l'animazione fluida e scattante di presa e rilascio avviata con il click del mouse."""
        if self.action_state is None:
            return

        self.action_timer += dt

        if self.action_state == "DESCEND":
            self.target_tcp_pos[0] += (self.action_target_pos[0] - self.target_tcp_pos[0]) * min(1.0, 16.0 * dt)
            self.target_tcp_pos[1] += (self.action_target_pos[1] - self.target_tcp_pos[1]) * min(1.0, 16.0 * dt)
            self.target_tcp_pos[2] += (self.action_target_pos[2] - self.target_tcp_pos[2]) * min(1.0, 14.0 * dt)

            actual_pos, _ = self.get_tcp_pose()
            if abs(actual_pos[2] - self.action_target_pos[2]) < 0.015 or self.action_timer > 0.8:
                self.action_state = "ACTION"
                self.action_timer = 0.0

        elif self.action_state == "ACTION":
            if self.attachment_constraint is not None:
                self.toggle_grasp(force_release=True)
            else:
                self.toggle_grasp(force_attach=True)

            if self.action_timer > 0.20:
                self.action_state = "ASCEND"
                self.action_timer = 0.0

        elif self.action_state == "ASCEND":
            self.target_tcp_pos[2] += (self.travel_z - self.target_tcp_pos[2]) * min(1.0, 14.0 * dt)
            actual_pos, _ = self.get_tcp_pose()
            if abs(actual_pos[2] - self.travel_z) < 0.02 or self.action_timer > 0.8:
                self.action_state = None
                self.action_timer = 0.0
                print("[Mouse] Sequenza completata.")

    # -------------------------------------------------------------------------
    # Input Processing: Tastiera W-S-A-D e Slider GUI
    # -------------------------------------------------------------------------
    def _poll_keyboard(self, dt: float) -> Tuple[np.ndarray, float, float]:
        """Legge gli eventi tastiera da PyBullet con supporto per W-S-A-D e scatto TURBO con SHIFT."""
        delta_pos = np.zeros(3, dtype=float)
        delta_yaw = 0.0
        delta_gripper = 0.0

        try:
            keys = p.getKeyboardEvents(physicsClientId=self.cid)
        except Exception:
            return delta_pos, delta_yaw, delta_gripper

        def key_active(*key_chars_or_codes):
            for k in key_chars_or_codes:
                if isinstance(k, str):
                    for ch in (k.lower(), k.upper()):
                        st = keys.get(ord(ch), 0)
                        if st & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                            return True
                else:
                    st = keys.get(k, 0)
                    if st & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                        return True
            return False

        def key_triggered(*key_chars_or_codes):
            for k in key_chars_or_codes:
                if isinstance(k, str):
                    for ch in (k.lower(), k.upper()):
                        st = keys.get(ord(ch), 0)
                        if st & p.KEY_WAS_TRIGGERED:
                            return True
                else:
                    st = keys.get(k, 0)
                    if st & p.KEY_WAS_TRIGGERED:
                        return True
            return False

        # Rileva se SHIFT è tenuto premuto per lo scatto TURBO veloce
        shift_active = False
        for k in (p.B3G_SHIFT,):
            st = keys.get(k, 0)
            if st & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                shift_active = True
                break

        current_speed = self.jog_speed * (self.turbo_multiplier if shift_active else 1.0)
        step_dist = current_speed * dt

        # W: Avanti (+X verso i vassoi)
        if key_active('w', p.B3G_UP_ARROW):
            delta_pos[0] += step_dist
            self.mouse_pointer_mode = False

        # S: Indietro (-X verso la base)
        if key_active('s', p.B3G_DOWN_ARROW):
            delta_pos[0] -= step_dist
            self.mouse_pointer_mode = False

        # A: Sinistra (+Y verso tray_1 destro)
        if key_active('a', p.B3G_LEFT_ARROW):
            delta_pos[1] += step_dist
            self.mouse_pointer_mode = False

        # D: Destra (-Y verso tray_2 sinistro)
        if key_active('d', p.B3G_RIGHT_ARROW):
            delta_pos[1] -= step_dist
            self.mouse_pointer_mode = False

        # Q: Quota Z Su (+Z)
        if key_active('q', p.B3G_PAGE_UP):
            self.travel_z = min(0.58, self.travel_z + step_dist)
            delta_pos[2] += step_dist

        # E: Quota Z Giù (-Z)
        if key_active('e', p.B3G_PAGE_DOWN):
            self.travel_z = max(0.065, self.travel_z - step_dist)
            delta_pos[2] -= step_dist

        # U / J: Rotazione Yaw antioraria
        yaw_rate = 2.6 * (self.turbo_multiplier if shift_active else 1.0)
        if key_active('u', 'j'):
            delta_yaw -= yaw_rate * dt

        # O / L: Rotazione Yaw oraria
        if key_active('o', 'l'):
            delta_yaw += yaw_rate * dt

        # Spazio: Toggle Apertura/Chiusura Pinza
        if key_triggered(' ', 32):
            self.target_gripper_width = 0.080 if self.target_gripper_width < 0.04 else 0.005
            print(f"[Pinza] {'Aperta (80 mm)' if self.target_gripper_width > 0.04 else 'Chiusa'}")

        # G: Presa o Rilascio dello strumento
        if key_triggered('g'):
            self.toggle_grasp()

        # H: Reset Home
        if key_triggered('h'):
            self.reset_home()

        # V / C: Cambio Vista Telecamera PyBullet
        if key_triggered('v', 'c'):
            self.cycle_camera_view()

        # F: Toggle Finestra Monitor Chirurgico Endoscopico
        if key_triggered('f'):
            if cv2 is not None:
                self.show_endoscope_window = not self.show_endoscope_window
                if not self.show_endoscope_window:
                    try:
                        cv2.destroyWindow(self.endoscope_window_name)
                    except Exception:
                        pass
                print(f"[Tasto F] Monitor Chirurgico Endoscopico: {'ATTIVO' if self.show_endoscope_window else 'DISATTIVATO'}")

        # T: Cambio Vassoio di lavoro
        if key_triggered('t'):
            self.switch_tray_focus()

        # M: Toggle Modalità Puntatore Mouse
        if key_triggered('m'):
            self.mouse_pointer_mode = not self.mouse_pointer_mode
            if not self.mouse_pointer_mode:
                self._hide_mouse_cursor()
            print(f"[Tasto M] Modalità Puntatore: {'ATTIVA' if self.mouse_pointer_mode else 'DISATTIVATA'}")

        # + / -: Regola velocità jog base
        if key_triggered('+', '='):
            self.jog_speed = min(1.20, self.jog_speed + 0.05)
            print(f"[Velocita Base] {self.jog_speed * 1000:.0f} mm/s (Shift Turbo: {self.jog_speed * self.turbo_multiplier * 1000:.0f} mm/s)")
        if key_triggered('-', '_'):
            self.jog_speed = max(0.08, self.jog_speed - 0.05)
            print(f"[Velocita Base] {self.jog_speed * 1000:.0f} mm/s (Shift Turbo: {self.jog_speed * self.turbo_multiplier * 1000:.0f} mm/s)")

        # Esc: Chiusura
        if key_triggered(27):
            self.running = False

        return delta_pos, delta_yaw, delta_gripper

    def _poll_gui_controls(self):
        """Controlla se l'utente ha cliccato pulsanti o spostato gli slider a schermo."""
        if not self.gui:
            return

        # Lettura pulsanti a schermo
        for btn_name, param_id in self.buttons.items():
            try:
                clicks = p.readUserDebugParameter(param_id, physicsClientId=self.cid)
                if clicks > self.last_button_clicks[btn_name]:
                    self.last_button_clicks[btn_name] = clicks
                    if btn_name == "toggle_mouse_mode":
                        self.mouse_pointer_mode = not self.mouse_pointer_mode
                        if not self.mouse_pointer_mode:
                            self._hide_mouse_cursor()
                        print(f"[GUI] Modalità Puntatore: {'ATTIVA' if self.mouse_pointer_mode else 'DISATTIVATA'}")
                    elif btn_name == "toggle_endoscope":
                        if cv2 is not None:
                            self.show_endoscope_window = not self.show_endoscope_window
                            if not self.show_endoscope_window:
                                try:
                                    cv2.destroyWindow(self.endoscope_window_name)
                                except Exception:
                                    pass
                            print(f"[GUI] Monitor Chirurgico: {'ATTIVO' if self.show_endoscope_window else 'DISATTIVATO'}")
                    elif btn_name == "reset_home":
                        self.reset_home()
                    elif btn_name == "toggle_view":
                        self.cycle_camera_view()
                    elif btn_name == "toggle_grasp":
                        self.toggle_grasp()
                    elif btn_name == "switch_tray":
                        self.switch_tray_focus()
            except Exception:
                pass

        # Slider: agiscono solo quando l'utente li sposta, cosi' non combattono tastiera e presa
        def slider_moved(name, threshold):
            if name not in self.sliders:
                return None
            try:
                value = float(p.readUserDebugParameter(self.sliders[name], physicsClientId=self.cid))
            except Exception:
                return None
            if abs(value - self.slider_last.get(name, value)) < threshold:
                return None
            self.slider_last[name] = value
            return value

        z_val = slider_moved("tcp_z", 0.002)
        if z_val is not None and self.action_state is None:
            self.travel_z = z_val
            self.target_tcp_pos[2] = z_val

        yaw_deg = slider_moved("tcp_yaw", 0.5)
        if yaw_deg is not None:
            self.target_tcp_yaw = math.radians(yaw_deg)

        grip_mm = slider_moved("gripper", 0.5)
        if grip_mm is not None:
            self.target_gripper_width = grip_mm / 1000.0

    # -------------------------------------------------------------------------
    # Visualizzazioni Grafiche e Annotazioni 3D
    # -------------------------------------------------------------------------
    def _update_hud(self):
        """Aggiorna il testo informativo in sovrimpressione a schermo nella finestra principale."""
        if not self.gui:
            return

        tcp_pos, _ = self.get_tcp_pose()
        view_str = self.VIEW_NAMES.get(self.current_view, "SCONOSCIUTA")
        grasp_str = "BLOCCATO IN PINZA" if self.attachment_constraint is not None else "LIBERO"
        grip_state = f"{self.gripper_width * 1000:.0f} mm ({'APERTA' if self.gripper_width > 0.04 else 'CHIUSA'})"
        mode_str = ">>> PUNTATORE MOUSE ATTIVO <<< (Premi Rotella / M per W-S-A-D)" if self.mouse_pointer_mode else "TASTIERA W-S-A-D (Premi Rotella / M per Puntatore)"
        monitor_str = "ATTIVO (Finestra Separata)" if self.show_endoscope_window else "DISATTIVATO"

        header_msg = (
            f"VISTA 3D: [{view_str}]  |  MODALITÀ: [{mode_str}]\n"
            f"MONITOR ENDOSCOPICO (TCP): [{monitor_str}] (Tasto F per alternare)\n"
            f"TCP KUKA: X={tcp_pos[0]:.3f}m | Y={tcp_pos[1]:.3f}m | Z={tcp_pos[2]:.3f}m | Yaw={math.degrees(self.target_tcp_yaw):.1f} deg\n"
            f"PINZA: {grip_state} | STRUMENTO ({self.tool_name}): {grasp_str} | FOCUS: {self.active_tray.upper()} | VELOCITÀ: {self.jog_speed*1000:.0f} mm/s (SHIFT: Turbo {self.jog_speed*self.turbo_multiplier*1000:.0f} mm/s)"
        )

        keys_msg = (
            "[COMANDI PRINCIPALI TASTIERA W-S-A-D]:\n"
            "  * W / S: Spostamento Avanti (+X) / Indietro (-X)  |  SHIFT + [Tasto]: TURBO VELOCE\n"
            "  * A / D: Spostamento Sinistra (+Y) / Destra (-Y)  |  + / - : Regola Velocità Base\n"
            "  * Q / E: Quota Z Su / Giu  |  U / O (oppure J / L): Ruota Yaw\n"
            "  * SPAZIO: Apri / Chiudi pinza  |  G (o Click Destro): Presa / Rilascio\n"
            "  * F: Mostra / Nascondi finestra Monitor Chirurgico Endoscopico\n"
            "  * ROTELLA MOUSE (o Tasto M): Attiva/Disattiva Modalità Puntatore\n"
            "  * V o C: Alterna Vista Telecamera <-> 3D Orbitale  |  H: Reset HOME | Esc: Esci"
        )

        if (header_msg, keys_msg) == self._hud_cache:
            return
        self._hud_cache = (header_msg, keys_msg)

        self.hud_text_id = p.addUserDebugText(
            header_msg,
            [0.10, -0.75, 0.98],
            textColorRGB=[0.02, 0.10, 0.35],
            textSize=1.1,
            replaceItemUniqueId=self.hud_text_id,
            physicsClientId=self.cid,
        )

        self.hud_keys_id = p.addUserDebugText(
            keys_msg,
            [0.10, -0.75, 0.68],
            textColorRGB=[0.15, 0.20, 0.28],
            textSize=0.95,
            replaceItemUniqueId=self.hud_keys_id,
            physicsClientId=self.cid,
        )

    def _update_tcp_frame(self):
        """Disegna la terna cartesiana 3D solidale al TCP della pinza con riutilizzo degli ID (zero lag)."""
        if not self.gui:
            return

        tcp_pos, tcp_orn = self.get_tcp_pose()
        rot_mat = np.asarray(p.getMatrixFromQuaternion(tcp_orn)).reshape(3, 3)
        origin = np.asarray(tcp_pos)
        axis_len = 0.055

        colors = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.2, 1.0]]  # X: Rosso, Y: Verde, Z: Blu
        for i, color in enumerate(colors):
            end_pt = origin + rot_mat[:, i] * axis_len
            self.axis_line_ids[i] = p.addUserDebugLine(
                origin.tolist(),
                end_pt.tolist(),
                lineColorRGB=color,
                lineWidth=2.5,
                replaceItemUniqueId=self.axis_line_ids[i],
                physicsClientId=self.cid,
            )

    def _update_motion_trail(self):
        """Disegna la scia verde del percorso percorso dal TCP."""
        if not self.gui:
            return

        tcp_pos, _ = self.get_tcp_pose()
        pt = np.asarray(tcp_pos)

        if not self.trail_points or np.linalg.norm(pt - self.trail_points[-1]) > 0.02:
            if self.trail_points:
                line_id = p.addUserDebugLine(
                    self.trail_points[-1].tolist(),
                    pt.tolist(),
                    lineColorRGB=[0.1, 0.75, 0.25],
                    lineWidth=2.0,
                    lifeTime=8.0,
                    physicsClientId=self.cid,
                )
                self.trail_lines.append(line_id)
                if len(self.trail_lines) > self.max_trail_points:
                    self.trail_lines.pop(0)
            self.trail_points.append(pt)

            if len(self.trail_points) > self.max_trail_points:
                self.trail_points.pop(0)

    # -------------------------------------------------------------------------
    # Ciclo di Controllo Principale
    # -------------------------------------------------------------------------
    def step(self, dt: float = 1.0 / 60.0):
        """Esegue un ciclo di polling comandi, cinematica e simulazione."""
        if not p.isConnected(self.cid):
            self.running = False
            return

        # 1. Lettura Eventi Mouse (Raycasting e Click)
        self._poll_mouse()

        # 2. Aggiornamento Macchina a Stati Pick & Place
        self._update_pick_place_state_machine(dt)

        # 3. Lettura Tastiera W-S-A-D e Slider GUI
        d_pos_kb, d_yaw_kb, d_grip_kb = self._poll_keyboard(dt)
        self._poll_gui_controls()

        # 4. Aggiornamento coordinate target
        if self.action_state is None:
            if self.mouse_pointer_mode:
                # Modalità puntatore: il robot segue il cursore sul piano XY
                smoothing = min(1.0, 16.0 * dt)
                self.target_tcp_pos[0] += (self.mouse_world_target[0] - self.target_tcp_pos[0]) * smoothing
                self.target_tcp_pos[1] += (self.mouse_world_target[1] - self.target_tcp_pos[1]) * smoothing
                self.target_tcp_pos[2] += (self.travel_z - self.target_tcp_pos[2]) * smoothing
            else:
                # Modalità W-S-A-D: i tasti muovono direttamente il robot in tempo reale
                self.target_tcp_pos += d_pos_kb

        self.target_tcp_yaw += d_yaw_kb

        # Limiti dello spazio di lavoro realmente raggiungibile (corona circolare attorno alla base)
        self.target_tcp_pos = self.clamp_to_workspace(self.target_tcp_pos)

        base_quat = p.getQuaternionFromEuler([math.pi, 0.0, self.target_tcp_yaw])
        self.target_tcp_orn = np.array(base_quat, dtype=float)

        # 5. Controllo Pinza
        self.target_gripper_width = np.clip(self.target_gripper_width + d_grip_kb, 0.0, 0.080)
        self.gripper_width += (self.target_gripper_width - self.gripper_width) * min(1.0, 18.0 * dt)

        # 6. Risoluzione Cinematica Inversa (IK): se il target non e' raggiungibile si torna all'ultimo valido
        if self.solve_and_apply_ik(self.target_tcp_pos.tolist(), self.target_tcp_orn.tolist()):
            self.last_valid_target = self.target_tcp_pos.copy()
            self.ik_failed_frames = 0
        else:
            self.target_tcp_pos = self.last_valid_target.copy()
            self.ik_failed_frames += 1
            if self.ik_failed_frames == 1:
                print("[IK] Posa non raggiungibile con questo orientamento: target riportato all'ultimo punto valido.")

        # 7. Passo di simulazione PyBullet
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.cid)
        self.sync_gripper()

        # 8. Se la vista PyBullet principale è impostata su Eye-in-Hand (polso), sincronizza la visuale
        if self.current_view == self.VIEW_WRIST:
            tcp_pos, _ = self.get_tcp_pose()
            p.resetDebugVisualizerCamera(
                cameraDistance=0.40,
                cameraYaw=-90.0,
                cameraPitch=-85.0,
                cameraTargetPosition=tcp_pos,
                physicsClientId=self.cid,
            )

        # 9. Aggiornamento finestra separata Monitor Chirurgico (Endoscopio TCP)
        self.step_count += 1
        if self.show_endoscope_window:
            self._update_endoscope_window()

        # 10. Aggiornamento grafiche e HUD PyBullet (a frequenza ridotta per non appesantire la GUI)
        if self.step_count % 10 == 0:
            self._update_hud()
        if self.step_count % 3 == 0:
            self._update_tcp_frame()
            self._update_motion_trail()

    def run(self, max_duration: Optional[float] = None):
        """Avvia il loop interattivo in tempo reale con sincronizzazione fluida a 60 FPS."""
        print("\n" + "=" * 76)
        print("🤖 CONTROLLER INTERATTIVO KUKA LBR iiwa - W-S-A-D & ENDOSCOPIO TCP")
        print("=" * 76)
        print("Finestre attive:")
        print("  1. Finestra 3D PyBullet: Vista d'insieme telecamera overhead di progetto")
        if self.show_endoscope_window:
            print("  2. Monitor Chirurgico Separato: Vista Endoscopica TCP in tempo reale!")
        print("Comandi principali:")
        print("  - [W / S]: Spostamento Avanti (+X) / Indietro (-X)")
        print("  - [A / D]: Spostamento Sinistra (+Y) / Destra (-Y)")
        print("  - [Q / E]: Quota Z Su / Giu")
        print("  - [SHIFT + Tasto]: MODALITÀ TURBO VELOCE (scatto rapido)")
        print("  - [+ / -]: Regola la velocità di base")
        print("  - [U / O] (o J/L): Ruota Yaw della pinza")
        print("  - [Spazio]: Apri / Chiudi pinza")
        print("  - [G] o Click Destro: Presa / Rilascio dello strumento")
        print("  - [F]: Mostra / Nascondi finestra Monitor Chirurgico Endoscopico")
        print("  - [Rotella Mouse o M]: Attiva / Disattiva modalità puntatore")
        print("  - [H]: Ripristina la posa iniziale HOME")
        print("=" * 76 + "\n")

        t_start = time.monotonic()
        t_last = time.monotonic()
        target_fps = 60.0
        target_frame_time = 1.0 / target_fps

        try:
            while self.running and p.isConnected(self.cid):
                t_frame_start = time.monotonic()

                if max_duration and (t_frame_start - t_start) >= max_duration:
                    break

                # Delta time reale (in secondi) sincronizzato con il clock di sistema
                dt = min(0.040, max(0.005, t_frame_start - t_last))
                t_last = t_frame_start

                self.step(dt=dt)

                elapsed = time.monotonic() - t_frame_start
                sleep_time = max(0.0, target_frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Uscita] Interruzione utente ricevuta.")
        finally:
            self.close()

    def close(self):
        """Rilascia le risorse e chiude la simulazione."""
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
        ik_cid = getattr(self, "ik_cid", None)
        if ik_cid is not None and p.isConnected(ik_cid):
            p.disconnect(ik_cid)
        print("[Simulazione] Finestre PyBullet e Monitor Chirurgico chiuse.")


def main():
    parser = argparse.ArgumentParser(
        description="Controllore Interattivo KUKA LBR iiwa con Monitor Chirurgico Endoscopico su End-Point"
    )
    parser.add_argument("--tool", type=str, default="Scissors",
                        help="Strumento chirurgico da caricare nel vassoio (es. 'Scissors', 'Scalpel', 'Bone Forceps')")
    parser.add_argument("--tray", type=str, choices=["tray_1", "tray_2"], default="tray_2",
                        help="Vassoio di partenza per il robot e lo strumento (tray_1 destro o tray_2 sinistro)")
    parser.add_argument("--view", type=str, choices=["camera", "orbit", "wrist", "surgeon"], default="camera",
                        help="Vista telecamera iniziale nella finestra principale ('camera'=overhead, 'orbit'=3D, 'wrist'=polso)")
    parser.add_argument("--speed", type=float, default=0.35,
                        help="Velocita cartesiana di jog in m/s (default: 0.35 m/s)")
    parser.add_argument("--turbo", type=float, default=2.2,
                        help="Moltiplicatore velocita tenendo premuto SHIFT (default: 2.2)")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Disabilita la finestra separata del monitor endoscopico TCP")
    parser.add_argument("--mouse-pointer", action="store_true",
                        help="Avvia subito con la modalita puntatore mouse attiva anziche W-S-A-D")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed della posizione casuale dello strumento nel vassoio (default: diversa a ogni avvio)")
    parser.add_argument("--headless", action="store_true",
                        help="Esegui in modalita DIRECT senza finestra grafica (per test automatici)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Durata massima in secondi prima dell'uscita automatica (per test)")
    parser.add_argument("--config", type=str, default=None,
                        help="Percorso personalizzato del file di configurazione surgical_environment.yaml")
    args = parser.parse_args()

    view_map = {
        "camera": InteractiveRobotController.VIEW_CAMERA,
        "orbit": InteractiveRobotController.VIEW_ORBIT,
        "wrist": InteractiveRobotController.VIEW_WRIST,
        "surgeon": InteractiveRobotController.VIEW_SURGEON,
    }

    controller = InteractiveRobotController(
        config_path=args.config,
        tool_name=args.tool,
        source_tray=args.tray,
        gui=not args.headless,
        initial_view=view_map[args.view],
        jog_speed=args.speed,
        turbo_multiplier=args.turbo,
        start_with_mouse_pointer=args.mouse_pointer,
        show_endoscope_window=not args.no_monitor,
        seed=args.seed,
    )

    controller.run(max_duration=args.duration)


if __name__ == "__main__":
    main()
