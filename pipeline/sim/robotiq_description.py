"""Pinned official Robotiq articulated geometry shared by view and validator.

Finger joints follow the official mimic relationships kinematically. This is
not a force/torque gripper controller and cannot validate grasp stability.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[2]
REVISION = "5b7f9189cb26197a4f6850937a1ee4ce667a3577"


def prepared_description():
    source = ROOT / "data/external/robotiq-prepared/description.json"
    if not source.is_file():
        raise ValueError("Prepare the official gripper: python3 scripts/prepare_robotiq_description.py")
    metadata = json.loads(source.read_text())
    urdf = Path(metadata["urdf"])
    if not urdf.is_absolute():
        urdf = source.parent / urdf
    if (metadata["revision"] != REVISION or
            hashlib.sha256(urdf.read_bytes()).hexdigest() != metadata["urdf_sha256"]):
        raise ValueError("Prepared Robotiq description changed; re-prepare the pinned source")
    return metadata, urdf


class RobotiqDescription:
    def __init__(self, cid):
        self.cid = cid
        self.metadata, urdf = prepared_description()
        self.body = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=cid)
        self.links = {-1: "graspgen_base"}
        self.controls = []
        by_name = {j["name"]: j for j in self.metadata["joints"]}
        for i in range(p.getNumJoints(self.body, physicsClientId=cid)):
            info = p.getJointInfo(self.body, i, physicsClientId=cid)
            self.links[i] = info[12].decode()
            if info[2] == p.JOINT_REVOLUTE:
                self.controls.append((i, by_name[info[1].decode()]["multiplier"]))
        self.pad_links = {i for i, name in self.links.items() if name.endswith("inner_finger_pad")}

    def place(self, flange_position, flange_quaternion, width):
        # root = GraspGen base; flange axes = TCP axes. Inverse of the
        # official model-to-project TCP yaw, with no translational mounting offset.
        root, rotation = p.multiplyTransforms(flange_position, flange_quaternion,
                                             [0, 0, 0], [0, 0, .7071067811865476, .7071067811865476])
        p.resetBasePositionAndOrientation(self.body, root, rotation, physicsClientId=self.cid)
        widths = np.asarray(self.metadata["opening_widths_m"])
        if not widths[-1] <= width <= min(.13603458, widths[0]):
            raise ValueError(f"Opening {width:g} m outside the official Robotiq model")
        angle = float(np.interp(width, widths[::-1], self.metadata["opening_angles_rad"][::-1]))
        for joint, factor in self.controls:
            p.resetJointState(self.body, joint, angle * factor, physicsClientId=self.cid)

    def obstacle_clearance(self, obstacle, margin, contact_margin, allow_pad_contact=False):
        closest = .03
        violations = []
        for point in p.getClosestPoints(self.body, obstacle, .03, physicsClientId=self.cid):
            link, distance = point[3], point[8]
            required = contact_margin if allow_pad_contact and link in self.pad_links else margin
            closest = min(closest, distance)
            if distance < required:
                violations.append(f"official gripper {self.links[link]}: {distance*1000:.1f} mm (< {required*1000:.1f})")
        return closest, violations
