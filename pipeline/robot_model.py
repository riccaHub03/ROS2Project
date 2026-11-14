"""One robot model for physics, kinematics and collision checking.

Frames and joints are resolved by name; limits come from the actual URDF.
The model choice is shared through config/robot.yaml and frozen in each run.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config/robot.yaml"


def robot_settings(config=None):
    registry = yaml.safe_load(REGISTRY.read_text())
    override = (config or {}).get("robot", {}) or {}
    if not isinstance(override, dict):
        raise ValueError("robot must be a mapping with a model name")
    name = override.get("model", registry["default_model"])
    if name not in registry["models"]:
        raise ValueError(f"Unknown robot model {name!r}; choose an entry in config/robot.yaml")
    profile = copy.deepcopy(registry["models"][name])
    unknown = set(override) - set(profile) - {"model"}
    if unknown:
        raise ValueError("Unknown robot settings: " + ", ".join(sorted(unknown)))
    profile.update(copy.deepcopy(override))
    profile["model"] = name
    for key in ("urdf", "metadata"):
        value = profile.get(key)
        if value and not str(value).startswith("pybullet://"):
            profile[key] = str((ROOT / value).resolve())
    return profile


DEFAULT_HOME = tuple(robot_settings()["home_positions_rad"])
COMMAND_JOINT_NAMES = tuple(robot_settings()["command_joint_names"])


def file_sha256(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


@dataclass(frozen=True)
class RobotModel:
    settings: dict
    urdf: Path
    joint_names: tuple
    lower: tuple
    upper: tuple
    velocity: tuple
    effort: tuple
    home: tuple
    parent_links: dict
    model_sha256: str

    @property
    def model_id(self):
        return self.settings["model"]

    @property
    def flange_link(self):
        return self.settings["flange_link"]

    @classmethod
    def from_config(cls, config=None):
        settings = robot_settings(config)
        name = settings["model"]
        value = settings["urdf"]
        if value.startswith("pybullet://"):
            import pybullet_data
            urdf = Path(pybullet_data.getDataPath()) / value.removeprefix("pybullet://")
        else:
            urdf = Path(value)
        if not urdf.is_file():
            raise FileNotFoundError(f"Robot {name}: {urdf} missing. Run bash scripts/run_ros2_system.sh prepare")
        assets = {"urdf": file_sha256(urdf)}
        if settings["metadata"]:
            metadata_file = Path(settings["metadata"])
            metadata = json.loads(metadata_file.read_text())
            if (metadata["model"] != name or metadata["source"] != settings["source"]
                    or (metadata_file.parent / metadata["urdf"]).resolve() != urdf.resolve()):
                raise ValueError("Prepared robot model/source does not match config/robot.yaml")
            for relative, digest in metadata["assets_sha256"].items():
                path = (metadata_file.parent / relative).resolve()
                if metadata_file.parent.resolve() not in path.parents or file_sha256(path) != digest:
                    raise ValueError(f"Prepared robot asset changed: {relative}; prepare it again")
            assets.update(metadata["assets_sha256"])
        xml = ET.parse(urdf).getroot()
        joints = {joint.get("name"): joint for joint in xml.findall("joint")}
        links = {link.get("name") for link in xml.findall("link")}
        names = tuple(settings["native_joint_names"])
        movable = {j.get("name") for j in joints.values() if j.get("type") != "fixed"}
        if len(names) != 7 or len(set(names)) != 7 or set(names) != movable:
            raise ValueError("Robot must contain exactly the seven configured arm joints")
        if settings["base_link"] not in links or settings["flange_link"] not in links:
            raise ValueError("Configured base/flange link is missing from the robot URDF")
        values = {key: tuple(float(joints[n].find("limit").get(key)) for n in names)
                  for key in ("lower", "upper", "velocity", "effort")}
        if (not np.isfinite(list(values.values())).all()
                or any(a >= b for a, b in zip(values["lower"], values["upper"]))
                or min(values["velocity"]) <= 0 or min(values["effort"]) <= 0):
            raise ValueError("URDF joint limits must be finite and ordered with positive velocity/effort")
        home = tuple(settings["home_positions_rad"])
        if (len(home) != 7 or not np.isfinite(home).all()
                or any(not lo <= q <= hi for q, lo, hi in zip(home, values["lower"], values["upper"]))):
            raise ValueError("Home position lies outside the model's joint limits")
        parents = {j.find("child").get("link"): j.find("parent").get("link") for j in joints.values()}
        identity = dict(assets=assets, model=name, base_link=settings["base_link"],
                        flange_link=settings["flange_link"], joint_names=list(names),
                        command_joint_names=settings["command_joint_names"], home_positions_rad=home)
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return cls(settings, urdf, names, values["lower"], values["upper"], values["velocity"], values["effort"],
                   home, parents, fingerprint)

    def load(self, cid, position=(0, 0, 0), quaternion=(0, 0, 0, 1)):
        import pybullet as p
        robot = p.loadURDF(str(self.urdf), basePosition=list(position), baseOrientation=list(quaternion),
                           useFixedBase=True, physicsClientId=cid)
        infos = [p.getJointInfo(robot, i, physicsClientId=cid) for i in range(p.getNumJoints(robot, physicsClientId=cid))]
        indices = {info[1].decode(): i for i, info in enumerate(infos)}
        links = {info[12].decode(): i for i, info in enumerate(infos)}
        joints = [indices[name] for name in self.joint_names]
        flange = links[self.flange_link]
        if any(infos[i][2] != p.JOINT_REVOLUTE for i in joints):
            raise ValueError("All configured arm joints must be revolute")
        if not np.allclose([infos[i][8:10] for i in joints], np.column_stack([self.lower, self.upper])):
            raise ValueError("Loaded physics joint limits differ from the validated URDF")
        return robot, joints, flange

    def mount_adjacent_indices(self, cid, robot):
        """Exclude the flange and last two physical wrist links from mount tests."""
        import pybullet as p
        infos = [p.getJointInfo(robot, i, physicsClientId=cid) for i in range(p.getNumJoints(robot, physicsClientId=cid))]
        links = {info[12].decode(): i for i, info in enumerate(infos)}
        current, found, physical = self.flange_link, set(), 0
        while current in self.parent_links:
            index = links[current]
            found.add(index)
            if infos[index][2] != p.JOINT_FIXED:
                physical += 1
                if physical == 2:
                    break
            current = self.parent_links[current]
        return frozenset(found)

    def report(self, gripper_geometry=None):
        result = dict(model=self.model_id, urdf=str(self.urdf), model_sha256=self.model_sha256,
                      urdf_sha256=file_sha256(self.urdf), source=self.settings["source"],
                      base_link=self.settings["base_link"], flange_link=self.flange_link,
                      native_joint_names=list(self.joint_names), command_joint_names=self.settings["command_joint_names"],
                      lower_limits_rad=list(self.lower), upper_limits_rad=list(self.upper),
                      velocity_limits_rad_s=list(self.velocity), effort_limits_nm=list(self.effort),
                      home_positions_rad=list(self.home))
        if gripper_geometry:
            result.update(gripper=gripper_geometry.name,
                          tcp_offset_flange_xyz_m=list(gripper_geometry.tcp_offset),
                          tcp_orientation_flange_xyzw=[0., 0., 0., 1.],
                          tcp_calibration="nominal simulation mounting, not physically measured")
        return result
