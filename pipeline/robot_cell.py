"""Kinematic and collision model of the robot cell, used to validate grasp candidates before any motion.

A PyBullet DIRECT client holds the KUKA model, the table and the trays, a box model of the demonstrative
gripper and of the carried instrument. Everything is expressed in the robot BASE frame (the arm sits at the
origin); scene objects defined in the world frame are moved with the configured base pose. The client is only
used for inverse kinematics and distance queries: nothing is simulated.

Limitations (reported with every validation): the default model is PyBullet's ``kuka_iiwa`` (LBR iiwa 14
geometry; pass ``robot_urdf`` for the iiwa 7), the gripper and the instrument are boxes, trays are solid blocks
and objects that are not in the configuration (other instruments, people) are not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from pipeline.sim.robot_scene import (FINGER_CENTER_Z, FINGER_HALF_EXTENTS, FINGER_REST_Y, HOME, PALM_CENTER_Z,
                                      PALM_HALF_EXTENTS, TCP_OFFSET)

DEFAULT_URDF = "kuka_iiwa/model.urdf"
PROBE_M = 0.03                 # distances beyond this are reported as "far enough"
NEAR_FAR_PLANE = -10.0         # parking spot for bodies that are not in use


@dataclass
class IkSolution:
    q: List[float]
    position_error_m: float
    angle_error_rad: float
    joint_margin_rad: float     # smallest distance of any joint to its limit (negative = outside)
    converged: bool             # TCP reached within the tolerances
    within_limits: bool

    @property
    def ok(self) -> bool:
        return self.converged and self.within_limits


@dataclass
class Clearance:
    min_distance_m: float                       # closest approach to table/trays over the non-relaxed pairs
    violations: List[str] = field(default_factory=list)

    @property
    def clear(self) -> bool:
        return not self.violations


class RobotCell:
    def __init__(self, config: Dict[str, Any], base_frame, *, robot_urdf: Optional[str] = None, ee_link: int = 6,
                 position_tolerance_m: float = 0.003, angle_tolerance_rad: float = 0.035):
        self.config, self.base_frame, self.ee = config, base_frame, ee_link
        self.position_tolerance, self.angle_tolerance = position_tolerance_m, angle_tolerance_rad
        self.cid = p.connect(p.DIRECT)
        if self.cid < 0:
            raise RuntimeError("PyBullet is not available for the grasp feasibility check")
        self.obstacles: Dict[str, int] = {}
        self._parts: Dict[str, int] = {}
        self._tool_body = -1
        self._tcp_to_tool: Optional[Tuple[List[float], List[float]]] = None
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
            self.robot = p.loadURDF(robot_urdf or DEFAULT_URDF, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                                    physicsClientId=self.cid)
            infos = [p.getJointInfo(self.robot, i, physicsClientId=self.cid) for i in range(
                p.getNumJoints(self.robot, physicsClientId=self.cid))]
            self.joints = [i for i, info in enumerate(infos) if info[2] == p.JOINT_REVOLUTE][:7]
            if len(self.joints) != 7:
                raise ValueError(f"the robot model must have 7 revolute joints, found {len(self.joints)}")
            self.limits = [(infos[i][8], infos[i][9]) for i in self.joints]
            self._build_environment()
            self._build_gripper()
            self.set_joints(HOME)
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ construction
    def _box_body(self, half_extents: Sequence[float], position: Sequence[float], orientation: Sequence[float]) -> int:
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=self.cid)
        return p.createMultiBody(0, shape, -1, list(position), list(orientation), physicsClientId=self.cid)

    def _build_environment(self) -> None:
        env = self.config.get("environment", {})
        solids = {}
        if env.get("surgical_table"):
            solids["table"] = env["surgical_table"]
        for key, tray in env.get("trays", {}).items():
            solids[key] = tray
        for name, item in solids.items():
            center = item.get("center_xyz_m") or item["position_xyz_m"]
            position, orientation = self.base_frame.pose_world_to_base(center, [0, 0, 0, 1])
            half = [d / 2.0 for d in item["dimensions_xyz_m"]]
            self.obstacles[name] = self._box_body(half, position, orientation)

    def _build_gripper(self) -> None:
        park = [0, 0, NEAR_FAR_PLANE]
        self._parts["palm"] = self._box_body(PALM_HALF_EXTENTS, park, [0, 0, 0, 1])
        self._parts["finger_a"] = self._box_body(FINGER_HALF_EXTENTS, park, [0, 0, 0, 1])
        self._parts["finger_b"] = self._box_body(FINGER_HALF_EXTENTS, park, [0, 0, 0, 1])

    def close(self) -> None:
        if getattr(self, "cid", -1) >= 0 and p.isConnected(self.cid):
            p.disconnect(self.cid)

    def __enter__(self) -> "RobotCell":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ kinematics
    def set_joints(self, q: Sequence[float]) -> None:
        for joint, value in zip(self.joints, q):
            p.resetJointState(self.robot, joint, float(value), physicsClientId=self.cid)

    def joint_positions(self) -> List[float]:
        return [p.getJointState(self.robot, j, physicsClientId=self.cid)[0] for j in self.joints]

    def flange_pose(self) -> Tuple[List[float], List[float]]:
        state = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True, physicsClientId=self.cid)
        return list(state[4]), list(state[5])

    def tcp_pose(self) -> Tuple[List[float], List[float]]:
        position, orientation = self.flange_pose()
        tcp, _ = p.multiplyTransforms(position, orientation, TCP_OFFSET, [0, 0, 0, 1])
        return list(tcp), orientation

    def joint_margin(self, q: Sequence[float]) -> float:
        return min(min(v - lo, hi - v) for v, (lo, hi) in zip(q, self.limits))

    def limiting_joint(self, q: Sequence[float]) -> Tuple[str, float]:
        """KUKA name (A1..A7) and margin of the joint closest to its limit."""
        margins = [min(v - lo, hi - v) for v, (lo, hi) in zip(q, self.limits)]
        index = int(np.argmin(margins))
        return f"A{index + 1}", float(margins[index])

    def solve_ik(self, tcp_position: Sequence[float], orientation: Sequence[float],
                 seed: Sequence[float]) -> IkSolution:
        """Joint solution that puts the TCP at the pose, started from ``seed`` (the arm is left on the result)."""
        flange, _ = p.multiplyTransforms(list(tcp_position), list(orientation),
                                         [0, 0, -TCP_OFFSET[2]], [0, 0, 0, 1])
        lower, upper = [lim[0] for lim in self.limits], [lim[1] for lim in self.limits]
        self.set_joints(seed)
        q = list(seed)
        error = angle = float("inf")
        for _ in range(6):
            # restPoses = the current iterate: anchoring them to the seed makes the solver stall 3-5 mm short
            q = list(p.calculateInverseKinematics(
                self.robot, self.ee, flange, list(orientation), lowerLimits=lower, upperLimits=upper,
                jointRanges=[hi - lo for lo, hi in self.limits], restPoses=list(q), jointDamping=[0.01] * 7,
                maxNumIterations=150, residualThreshold=1e-7, physicsClientId=self.cid)[:7])
            self.set_joints(q)
            position, quaternion = self.tcp_pose()
            error = float(np.linalg.norm(np.asarray(position) - np.asarray(tcp_position)))
            angle = 2.0 * math.acos(float(np.clip(abs(np.dot(quaternion, orientation)), 0.0, 1.0)))
            if error < 0.0003 and angle < 0.002:          # converged well below the acceptance tolerance
                break
        margin = self.joint_margin(q)
        finite = bool(np.isfinite(q).all())
        return IkSolution(q, error, angle, margin if finite else float("-inf"),
                          finite and error < self.position_tolerance and angle < self.angle_tolerance,
                          finite and margin >= -1e-6)

    # ------------------------------------------------------------------ gripper and carried instrument
    def _place_gripper(self, width: float) -> None:
        flange, orientation = self.flange_pose()
        y = width / 2.0 + FINGER_REST_Y - 0.001                      # = REST + (width - 0.002) / 2
        offsets = {"palm": [0, 0, PALM_CENTER_Z], "finger_a": [0, y, FINGER_CENTER_Z],
                   "finger_b": [0, -y, FINGER_CENTER_Z]}
        for part, offset in offsets.items():
            position, quaternion = p.multiplyTransforms(flange, orientation, offset, [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(self._parts[part], position, quaternion, physicsClientId=self.cid)

    def attach_tool(self, half_extents: Sequence[float], position: Sequence[float], orientation: Sequence[float],
                    tcp_position: Sequence[float], tcp_orientation: Sequence[float]) -> None:
        """Rigidly attach a box (the instrument, given at rest in the base frame) to the TCP frame."""
        self.detach_tool()
        inverse = p.invertTransform(list(tcp_position), list(tcp_orientation))
        self._tcp_to_tool = p.multiplyTransforms(inverse[0], inverse[1], list(position), list(orientation))
        self._tool_body = self._box_body(half_extents, [0, 0, NEAR_FAR_PLANE], [0, 0, 0, 1])

    def detach_tool(self) -> None:
        if self._tool_body >= 0:
            p.removeBody(self._tool_body, physicsClientId=self.cid)
        self._tool_body, self._tcp_to_tool = -1, None

    def _place_tool(self) -> None:
        tcp, orientation = self.tcp_pose()
        position, quaternion = p.multiplyTransforms(tcp, orientation, *self._tcp_to_tool)
        p.resetBasePositionAndOrientation(self._tool_body, position, quaternion, physicsClientId=self.cid)

    # ------------------------------------------------------------------ collisions
    def _closest(self, a: int, b: int, distance: float,
                 link_filter: Optional[Callable[[int, int], bool]] = None) -> Optional[Tuple[float, int, int]]:
        best = None
        for point in p.getClosestPoints(a, b, distance, physicsClientId=self.cid):
            if link_filter is not None and not link_filter(point[3], point[4]):
                continue
            if best is None or point[8] < best[0]:
                best = (point[8], point[3], point[4])
        return best

    def clearance(self, *, width: float, carried: bool = False,
                  relaxed: FrozenSet[Tuple[str, str]] = frozenset(), margin: float = 0.01,
                  contact_margin: float = 0.0005, self_margin: float = 0.002) -> Clearance:
        """Distances at the current joint state.

        ``margin``: minimum separation from table and trays for the arm, gripper and carried instrument.
        ``relaxed``: (part, obstacle) pairs allowed down to ``contact_margin`` (the fingertips just above the tray
        at grasp and release); parts are ``fingers``, ``palm`` and ``tool``. Self-collisions use ``self_margin``.
        """
        self._place_gripper(width)
        if carried and self._tool_body >= 0:
            self._place_tool()
        closest = PROBE_M
        violations: List[str] = []

        def judge(label: str, part: str, obstacle: str, found: Optional[Tuple[float, int, int]]) -> None:
            nonlocal closest
            relaxed_pair = (part, obstacle) in relaxed
            required = contact_margin if relaxed_pair else margin
            distance = PROBE_M if found is None else found[0]
            if not relaxed_pair:
                closest = min(closest, distance)
            if distance < required:
                violations.append(f"{label} <-> {obstacle}: {distance * 1000:.1f} mm (< {required * 1000:.1f} mm)")

        for name, body in self.obstacles.items():
            found = self._closest(self.robot, body, PROBE_M)
            judge(f"arm link {found[1]}" if found else "arm", "arm", name, found)
            for part, key in (("palm", "palm"), ("fingers", "finger_a"), ("fingers", "finger_b")):
                judge(f"gripper {part}", part, name, self._closest(self._parts[key], body, PROBE_M))
            if carried and self._tool_body >= 0:
                judge("carried instrument", "tool", name, self._closest(self._tool_body, body, PROBE_M))

        # arm against itself (non-adjacent links) and gripper against the upper arm
        for found in [self._closest(self.robot, self.robot, self_margin,
                                    lambda a, b: abs(a - b) >= 2)]:
            if found is not None and found[0] < self_margin:
                violations.append(f"arm self-collision link {found[1]} <-> link {found[2]}: {found[0] * 1000:.1f} mm")
        for key in ("palm", "finger_a", "finger_b"):
            found = self._closest(self._parts[key], self.robot, self_margin, lambda a, b: b <= self.ee - 2)
            if found is not None and found[0] < self_margin:
                violations.append(f"gripper <-> arm link {found[2]}: {found[0] * 1000:.1f} mm")
        return Clearance(closest, violations)
