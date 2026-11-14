"""Kinematic and collision model of the robot cell, used to validate grasp candidates before any motion.

A PyBullet DIRECT client holds the KUKA model, the table and the trays, a box model of the demonstrative
gripper and of the carried instrument. Everything is expressed in the robot BASE frame (the arm sits at the
origin); scene objects defined in the world frame are moved with the configured base pose. The client is only
used for inverse kinematics and distance queries: nothing is simulated.

The default arm is the pinned LBR iiwa 7 R800 description shared with the scene.
The demonstrative gripper and instrument are boxes, trays are solid blocks
and objects that are not in the configuration (other instruments, people) are not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from pipeline.sim.gripper_geometry import for_config
from pipeline.robot_model import RobotModel

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
    def __init__(self, config: Dict[str, Any], base_frame, *, robot_urdf: Optional[str] = None, ee_link: Optional[int] = None,
                 position_tolerance_m: float = 0.003, angle_tolerance_rad: float = 0.035):
        self.config, self.base_frame = config, base_frame
        self.robot_model = RobotModel.from_config(config)
        if robot_urdf is not None:
            from pathlib import Path
            if Path(robot_urdf).resolve() != self.robot_model.urdf.resolve():
                raise ValueError("Validator URDF differs from shared robot settings; configure robot in config/robot.yaml")
        self.home = self.robot_model.home
        self.gripper_geometry = for_config(config)
        self.position_tolerance, self.angle_tolerance = position_tolerance_m, angle_tolerance_rad
        self.cid = p.connect(p.DIRECT)
        if self.cid < 0:
            raise RuntimeError("PyBullet is not available for the grasp feasibility check")
        self.obstacles: Dict[str, int] = {}
        self._parts: Dict[str, int] = {}
        self._tool_body = -1
        self._tool_shape = None
        self._tcp_to_tool: Optional[Tuple[List[float], List[float]]] = None
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
            self.robot, self.joints, self.ee = self.robot_model.load(self.cid)
            if ee_link is not None and ee_link != self.ee:
                raise ValueError("Validator flange index differs from the configured flange link")
            self.mount_adjacent = self.robot_model.mount_adjacent_indices(self.cid, self.robot)
            self.limits = list(zip(self.robot_model.lower, self.robot_model.upper))
            self._build_environment()
            self._build_gripper()
            self.set_joints(self.home)
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
        from pipeline.sim.gripper_geometry import ROBOTIQ_PROXY
        self.official_gripper = None
        if self.gripper_geometry == ROBOTIQ_PROXY:
            from pipeline.sim.robotiq_description import RobotiqDescription
            self.official_gripper = RobotiqDescription(self.cid)
            return
        park = [0, 0, NEAR_FAR_PLANE]
        self._parts["palm"] = self._box_body(self.gripper_geometry.palm_half, park, [0, 0, 0, 1])
        self._parts["finger_a"] = self._box_body(self.gripper_geometry.finger_half, park, [0, 0, 0, 1])
        self._parts["finger_b"] = self._box_body(self.gripper_geometry.finger_half, park, [0, 0, 0, 1])

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
        tcp, _ = p.multiplyTransforms(position, orientation, self.gripper_geometry.tcp_offset, [0, 0, 0, 1])
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
                                         [0, 0, -self.gripper_geometry.tcp_offset[2]], [0, 0, 0, 1])
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
        if self.official_gripper is not None:
            self.official_gripper.place(flange, orientation, width)
            return
        y = width / 2.0 + self.gripper_geometry.finger_rest_y - 0.001                      # = REST + (width - 0.002) / 2
        offsets = {"palm": [0, 0, self.gripper_geometry.palm_z], "finger_a": [0, y, self.gripper_geometry.finger_z],
                   "finger_b": [0, -y, self.gripper_geometry.finger_z]}
        for part, offset in offsets.items():
            position, quaternion = p.multiplyTransforms(flange, orientation, offset, [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(self._parts[part], position, quaternion, physicsClientId=self.cid)

    def attach_tool(self, half_extents: Sequence[float], position: Sequence[float], orientation: Sequence[float],
                    tcp_position: Sequence[float], tcp_orientation: Sequence[float]) -> None:
        """Rigidly attach a box (the instrument, given at rest in the base frame) to the TCP frame."""
        self.detach_tool()
        inverse = p.invertTransform(list(tcp_position), list(tcp_orientation))
        self._tcp_to_tool = p.multiplyTransforms(inverse[0], inverse[1], list(position), list(orientation))
        self._tool_body = self._create_tool_body(half_extents, [0, 0, NEAR_FAR_PLANE], [0, 0, 0, 1])

    def configure_tool_mesh(self, filename):
        self._tool_shape = p.createCollisionShape(p.GEOM_MESH, fileName=str(filename), physicsClientId=self.cid)

    def _create_tool_body(self, half_extents, position, orientation):
        if self._tool_shape is None:
            return self._box_body(half_extents, position, orientation)
        return p.createMultiBody(0, self._tool_shape, -1, list(position), list(orientation), physicsClientId=self.cid)

    def detach_tool(self) -> None:
        if self._tool_body >= 0:
            p.removeBody(self._tool_body, physicsClientId=self.cid)
        self._tool_body, self._tcp_to_tool = -1, None

    def set_stationary_tool(self, half_extents, position, orientation):
        """Target/released instrument obstacle; retained while the arm approaches/retreats."""
        if self._tool_body >= 0 and self._tcp_to_tool is None:
            p.resetBasePositionAndOrientation(self._tool_body, position, orientation, physicsClientId=self.cid)
            return
        self.detach_tool()
        self._tool_body = self._create_tool_body(half_extents, position, orientation)

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
                  contact_margin: float = 0.0005, self_margin: float = 0.002,
                  check_target: bool = False, allow_finger_contact: bool = True,
                  tool_support_contacts: FrozenSet[str] = frozenset()) -> Clearance:
        """Distances at the current joint state.

        ``margin``: minimum separation from table and trays for the arm, gripper and carried instrument.
        ``relaxed``: (part, obstacle) pairs allowed down to ``contact_margin`` (the fingertips just above the tray
        at grasp and release); parts are ``fingers``, ``palm`` and ``tool``. Self-collisions use ``self_margin``.
        ``tool_support_contacts`` permits the instrument to touch ONLY the top
        of the named support during lift/place, with 10 micrometres numerical
        tolerance. It never applies to the arm, fingers, side walls or transit.
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
            comparison_epsilon = 0.
            distance = PROBE_M if found is None else found[0]
            if part == "tool" and obstacle in tool_support_contacts and found is not None:
                upward = self.base_frame.rotation.T @ np.array([0., 0., 1.])
                near = p.getClosestPoints(self._tool_body, self.obstacles[obstacle], contact_margin,
                                          physicsClientId=self.cid)
                if near and all(float(np.dot(point[7], upward)) >= .95 for point in near):
                    required = -1e-5
                    # Bullet transforms use single-precision link poses. Keep
                    # the 10 um contact budget, allowing 0.1 um for comparison
                    # roundoff ONLY on the declared top-support contact.
                    comparison_epsilon = 1e-7
            if not relaxed_pair:
                closest = min(closest, distance)
            if distance < required - comparison_epsilon:
                violations.append(f"{label} <-> {obstacle}: {distance * 1000:.6f} mm (< {required * 1000:.6f} mm)")

        for name, body in self.obstacles.items():
            found = self._closest(self.robot, body, PROBE_M)
            judge(f"arm link {found[1]}" if found else "arm", "arm", name, found)
            if self.official_gripper is not None:
                distance, issues = self.official_gripper.obstacle_clearance(
                    body, margin, contact_margin, ("fingers", name) in relaxed)
                closest = min(closest, distance)
                violations.extend(f"{issue} <-> {name}" for issue in issues)
            else:
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
            if key not in self._parts:
                continue
            found = self._closest(self._parts[key], self.robot, self_margin, lambda a, b: b not in self.mount_adjacent)
            if found is not None and found[0] < self_margin:
                violations.append(f"gripper <-> arm link {found[2]}: {found[0] * 1000:.1f} mm")
        if self.official_gripper is not None:
            found = self._closest(self.official_gripper.body, self.robot, self_margin, lambda a, b: b not in self.mount_adjacent)
            if found is not None and found[0] < self_margin:
                violations.append(f"official gripper <-> arm link {found[2]}: {found[0]*1000:.1f} mm")
        if check_target and self._tool_body >= 0:
            found = self._closest(self._tool_body, self.robot, self_margin, lambda a, b: b <= self.ee)
            if found is not None and found[0] < self_margin:
                violations.append(f"instrument <-> arm link {found[2]}: {found[0] * 1000:.1f} mm")
            for key in (["palm"] if allow_finger_contact else ["palm", "finger_a", "finger_b"]):
                if key not in self._parts:
                    continue
                found = self._closest(self._parts[key], self._tool_body, contact_margin)
                if found is not None and found[0] < contact_margin:
                    violations.append(f"instrument <-> {key}: {found[0] * 1000:.1f} mm")
            if self.official_gripper is not None:
                distance, issues = self.official_gripper.obstacle_clearance(
                    self._tool_body, contact_margin, 0.0 if allow_finger_contact else contact_margin,
                    allow_finger_contact)
                closest = min(closest, distance)
                violations.extend(f"{issue} <-> instrument" for issue in issues)
        return Clearance(closest, violations)
