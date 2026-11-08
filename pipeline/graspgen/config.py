"""Validated client configuration; imports neither torch nor the model package."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PINNED_REVISION = "2dd8852e1be60f5f9d277fafcc621835cdf59110"
DEFAULT_CONFIG = ROOT / "config/graspgen.yaml"


def rigid_transform(value, label="transform") -> np.ndarray:
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-5, rtol=0)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4, rtol=0)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-4, rtol=0)):
        raise ValueError(f"{label} must be a finite rigid 4x4 transform (SO(3), metres)")
    return t


def load_config(path=None, *, host=None, port=None, timeout_ms=None):
    source = Path(path).resolve() if path else DEFAULT_CONFIG
    cfg = copy.deepcopy(yaml.safe_load(source.read_text()))
    if cfg["planner"] != "diffusion":
        raise ValueError("This backend requires planner=diffusion; GraspMoE is a separate experiment")
    if cfg["expected_revision"] != PINNED_REVISION:
        raise ValueError(f"expected_revision must match the supported pin {PINNED_REVISION}")
    if cfg["model_gripper"] != "robotiq_2f_140":
        raise ValueError("This integration currently defines TCP/visualization for robotiq_2f_140 only")
    if cfg["checkpoint_config"] != "graspgen_robotiq_2f_140.yml":
        raise ValueError("checkpoint_config must correspond to model_gripper=robotiq_2f_140")
    for key, value in (("host", host), ("port", port), ("timeout_ms", timeout_ms)):
        if value is not None:
            cfg["server"][key] = value
    for key in ("port", "timeout_ms"):
        v = cfg["server"][key]
        if type(v) is not int or v <= 0 or (key == "port" and v > 65535):
            raise ValueError(f"server.{key} is invalid")
    if not isinstance(cfg["server"]["host"], str) or not cfg["server"]["host"]:
        raise ValueError("server.host must be nonempty")
    if cfg["input"]["units"] != "m":
        raise ValueError("GraspGen input must be in metres (input.units=m), no automatic rescaling")
    for key in ("num_grasps", "min_grasps", "max_tries"):
        if type(cfg["inference"][key]) is not int or cfg["inference"][key] <= 0:
            raise ValueError(f"inference.{key} must be a positive integer")
    topk = cfg["inference"]["topk_num_grasps"]
    threshold = cfg["inference"]["grasp_threshold"]
    if type(topk) is not int or (topk != -1 and topk <= 0):
        raise ValueError("inference.topk_num_grasps must be -1 or positive")
    if not np.isfinite(threshold) or (threshold != -1 and not 0 <= threshold <= 1):
        raise ValueError("inference.grasp_threshold must be -1 or in [0,1]")
    inp = cfg["input"]
    if not (3 <= inp["min_points"] <= inp["max_points"]
            and 0 < inp["min_extent_m"] < inp["max_extent_m"]):
        raise ValueError("invalid input point count / metric extent limits")
    if cfg["inference"]["remove_outliers"] and inp["min_points"] < 21:
        raise ValueError("Official outlier removal uses K=20; input.min_points must be >=21")
    rigid_transform(cfg["robot"]["T_gripper_tcp"], "robot.T_gripper_tcp")
    for section, keys in (("server", ["require_provenance"]),
                          ("inference", ["remove_outliers"]),
                          ("robot", ["tcp_transform_verified", "compatibility_verified", "motion_enabled"])):
        for key in keys:
            if type(cfg[section][key]) is not bool:
                raise ValueError(f"{section}.{key} must be a YAML boolean")
    code = Path(cfg["submodule"])
    cfg["submodule"] = str(code if code.is_absolute() else ROOT / code)
    cfg["config_source"] = str(source)
    return cfg
