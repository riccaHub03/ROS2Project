"""
Modulo caricamento e validazione della configurazione centralizzata.
File: config/surgical_environment.yaml
"""

import os
import yaml

DEFAULT_CONFIG_PATH = "/workspace/config/surgical_environment.yaml"
FALLBACK_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "surgical_environment.yaml")

def load_environment_config(config_path=None):
    if config_path is None:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            config_path = DEFAULT_CONFIG_PATH
        elif os.path.exists(FALLBACK_CONFIG_PATH):
            config_path = FALLBACK_CONFIG_PATH
        else:
            config_path = "config/surgical_environment.yaml"

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg

    # Default di sicurezza coerenti con surgical_environment.yaml se il file non è trovato
    return {
        "environment": {
            "kuka_base": {
                "position_xyz_m": [0.0, 0.0, 0.0]
            },
            "surgical_table": {
                "center_xyz_m": [0.60, 0.0, -0.05],
                "dimensions_xyz_m": [0.85, 1.60, 0.20],
                "surface_z_m": 0.05,
                "color_rgba": [0.15, 0.35, 0.60, 1.0]
            },
            "trays": {
                "tray_1": {
                    "name": "Vassoio Raccolta e Decontaminazione Destro",
                    "alias": ["tray_1", "vassoio_1", "tray 1", "right tray"],
                    "position_xyz_m": [0.60, 0.42, 0.07],
                    "dimensions_xyz_m": [0.62, 0.68, 0.04],
                    "color_rgba": [0.80, 0.82, 0.85, 1.0]
                },
                "tray_2": {
                    "name": "Vassoio Strumenti Sterili Sinistro",
                    "alias": ["tray_2", "vassoio_2", "tray 2", "left tray"],
                    "position_xyz_m": [0.60, -0.42, 0.07],
                    "dimensions_xyz_m": [0.62, 0.68, 0.04],
                    "color_rgba": [0.80, 0.82, 0.85, 1.0]
                }
            }
        },
        "camera_overhead": {
            "auto_track_source_tray": True,
            "height_above_tray_m": 1.08,
            "extrinsics": {
                "position_robot_frame_xyz_m": [0.60, -0.42, 1.15],
                "target_position_xyz_m": [0.60, -0.42, 0.07]
            },
            "intrinsics": {
                "fx_px": 615.0,
                "fy_px": 615.0,
                "cx_px": 320.0,
                "cy_px": 240.0
            },
            "width_px": 640,
            "height_px": 480
        },
        "motion_control": {
            "velocities": {
                "cartesian_linear_speed_m_s": 0.16,
                "cartesian_angular_speed_rad_s": 0.80,
                "joint_speed_rad_s": 1.00
            },
            "heights_z_m": {
                "staging_z": 0.45,
                "approach_z": 0.32,
                "lift_clearance_z": 0.35
            },
            "downward_quaternion_xyzw": [0.0, 1.0, 0.0, 0.0],
            "timeouts": {"step_timeout_sec": 30.0, "bringup_wait_sec": 25.0}
        },
        "video_rendering": {
            "framerate_hz": 30.0,
            "output_filename": "medical_pick_and_place.mp4"
        }
    }
