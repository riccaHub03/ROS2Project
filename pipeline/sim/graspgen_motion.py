"""Simulation-only GraspGen -> checked nominal-gripper pick and place.

This never changes motion.allowed or physical readiness. It uses official
articulated link meshes and ideal attachment, not a ROS/hardware executor.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
import logging
from collections import Counter

import numpy as np
import pybullet as p

from pipeline.common import paths
from pipeline.common.report import StageReport
from pipeline.common.waypoints import (pick_and_place_targets, Target, APPROACH, DESCEND,
                                      TRANSFER, RETURN, RETREAT, LIFT, PLACE, RELEASE)
from pipeline.frame_transformer import RobotBaseFrame, quaternion_from_matrix
from pipeline.graspgen.config import rigid_transform
from pipeline.robot_cell import RobotCell
from pipeline.sim import constrained_differential_ik as diff_ik
from pipeline.sim.gripper_geometry import ROBOTIQ_PROXY
from pipeline.sim.robot_motion_preview import MotionPreview
from pipeline.sim.checked_joint_planner import CheckedJointPlanner
from pipeline.sim.grasp_refinement import refine_grasp, TCP

log = logging.getLogger("graspgen.motion")


class PlanningCancelled(RuntimeError):
    pass


class GraspGenMotionPreview(MotionPreview):
    """Choose highest scoring candidate with a complete checked trajectory.

    A bounded CAD refinement is recorded separately from the original model pose
    and score. No centroid/PCA replacement or claim of physical force closure.
    """

    def __init__(self, scene, report):
        self.scene = scene
        self._display_report = report
        self._runtime_error = None
        self.planner = self.cell = None
        self.started = time.monotonic()
        self._last_heartbeat = 0.
        self._candidate_deadline = float("inf")
        self._phase_label = "initialization"
        self.validation = {"allowed": False, "execution_scope": "pybullet_only",
                           "gripper_model": ROBOTIQ_PROXY.name,
                           "run_id": paths.output_dir().name,
                           "physical_grasp_validated": False, "attempts": []}
        try:
            self._check_source(report)
            pos, orn = p.getBasePositionAndOrientation(scene.tool, physicsClientId=scene.cid)
            self.tool_position_base, self.tool_orientation_base = scene.base_frame.pose_world_to_base(pos, orn)
            self.tool_half = np.asarray(scene.mesh_info["dimensions_m"]) / 2
            self.cell = RobotCell(scene.config, scene.base_frame,
                                  position_tolerance_m=.001, angle_tolerance_rad=.005)
            self.cell.configure_tool_mesh(scene.mesh_file)
            self.start_q = np.asarray([p.getJointState(scene.robot, j, physicsClientId=scene.cid)[0]
                                       for j in scene.joints])
            failures = self.validation["attempts"]
            ranked = sorted(report["candidates"], key=lambda c: -float(c["score"]))
            self.validation["candidate_count"] = len(ranked)
            self.validation["status"] = "CHECKING"
            report.pop("simulation_selected", None)
            for index, candidate in enumerate(ranked):
                self._candidate_deadline = time.monotonic() + 30.
                candidate_started = time.monotonic()
                report["simulation_checking"] = candidate
                scene.show_grasp(report)
                self._progress = f"GraspGen {index + 1}/{len(ranked)}; score={candidate['score']:.3f}"
                self._phase_label = "geometry"
                try:
                    self._candidate(candidate, None)
                except PlanningCancelled:
                    raise
                except (ValueError, RuntimeError) as exc:
                    failures.append({"server_index": candidate["server_index"], "score_rank": index + 1,
                                     "score": candidate["score"], "reason": str(exc),
                                     "phase": self._phase_label,
                                     "elapsed_s": round(time.monotonic()-candidate_started, 3)})
                    log.info("Candidate %s rejected at %s in %.2fs: %s", candidate['server_index'],
                             self._phase_label, time.monotonic()-candidate_started, exc)
                    self._save_validation(report)
                    continue
                report.pop("simulation_checking", None)
                self.validation.update(allowed=True, status="PLANNED_NOT_EXECUTED",
                                       selected_server_index=candidate["server_index"],
                                       selected_score=candidate["score"],
                                       estimated_closing_width_m=self.close_width,
                                       geometric_refinement=self.executed_candidate["geometric_refinement"],
                                       score_semantics="Original model score; refined pose was not rescored",
                                       robotiq_description_revision=self.cell.official_gripper.metadata["revision"],
                                       robotiq_urdf_sha256=self.cell.official_gripper.metadata["urdf_sha256"],
                                       checks={"ik": True, "joint_limits": True,
                                               "sampled_path_collisions": True, "physical_contact": False},
                                       collision_sample_period_s=4 * diff_ik.CONTROL_DT,
                                       support_contact_policy={"lift": scene.source,
                                                               "place_release": scene.destination,
                                                               "minimum_upward_normal_dot": .95,
                                                               "numerical_slop_m": .00001,
                                                               "comparison_epsilon_m": 1e-7,
                                                               "scene_solver_contact_slop_m": 1e-6},
                                       limitations=["Official Robotiq link meshes; convex instrument mesh approximation",
                                                    "Finger joints driven kinematically by URDF mimic relationships",
                                                    "Ideal fixed constraint, no friction/contact grasp validation",
                                                    "Only configured table/trays and robot self collisions",
                                                    "Surgical functional zones not checked"])
                # Preserve the discriminator-only selection as well as the executable one.
                self.original_selected = copy.deepcopy(report["selected"])
                report["simulation_selected"] = copy.deepcopy(self.executed_candidate)
                report["simulation_motion"] = copy.deepcopy(self.validation)
                trajectory_file = paths.output_dir() / "graspgen_sim_joint_path.npz"
                np.savez_compressed(trajectory_file, **{f"phase_{i}": phase["samples"]
                                                        for i, phase in enumerate(self.phases)})
                self.validation["trajectory_npz"] = str(trajectory_file)
                self.validation["planning_elapsed_s"] = round(time.monotonic()-self.started, 3)
                report["simulation_motion"] = copy.deepcopy(self.validation)
                self.index = 0
                self.elapsed = self.total = 0.
                self.results = []
                self.finished = False
                self._announced = -1
                self._ticks = 0
                self.scene.show_path(self.waypoints)
                self._save_validation(report)
                report["simulation_motion"] = copy.deepcopy(self.validation)
                return
            raise ValueError("No executable GraspGen candidate: inspect graspgen_sim_motion_validation.json")
        except BaseException as exc:
            report.pop("simulation_checking", None)
            self.validation["error"] = str(exc)
            self.validation["status"] = "CANCELLED" if isinstance(exc, PlanningCancelled) else "FAILED"
            self._save_validation(report)
            self.close()
            raise

    def _save_validation(self, report):
        if not hasattr(self, "index") or "planning_elapsed_s" not in self.validation:
            self.validation["planning_elapsed_s"] = round(time.monotonic() - self.started, 3)
        self.validation["rejections_by_phase"] = dict(Counter(
            item.get("phase", "unknown") for item in self.validation["attempts"]))
        ok = self.validation["allowed"] and self._runtime_error is None
        status = "COMPLETED" if ok else "FAILED" if self.validation.get("error") or self._runtime_error else "RUNNING"
        StageReport("graspgen_sim_motion_validation", status=status,
                    data=self.validation).save(
            paths.output_dir() / "graspgen_sim_motion_validation.json")

    def _heartbeat(self):
        now = time.monotonic()
        if now-self.started > 180:
            raise PlanningCancelled("Total planning budget 180 s exceeded; untested candidates are not rejected grasps")
        if now > self._candidate_deadline:
            raise RuntimeError(f"Candidate budget 30 s exceeded at {self._phase_label}")
        if now-self._last_heartbeat > .15:
            if self.scene.gui:
                keys = p.getKeyboardEvents(physicsClientId=self.scene.cid)
                if any(keys.get(key, 0) & p.KEY_WAS_TRIGGERED for key in (ord('q'), 27)):
                    raise PlanningCancelled("Planning cancelled by user; robot held stationary")
            self.scene.status(f"{self._progress}: {self._phase_label} ({now-self.started:.0f}s); Q cancels")
            self._last_heartbeat = now

    def _check_source(self, report):
        scene = self.scene
        if report.get("grasp_backend") != "graspgen" or report.get("source_pipeline") != "sim":
            raise ValueError("Simulation motion requires a GraspGen report from this simulated scene")
        if scene.gripper_geometry != ROBOTIQ_PROXY or scene.procedural_tool_fallback:
            raise ValueError("Simulation requires the nominal Robotiq proxy and a dataset mesh")
        base = RobotBaseFrame.from_dict(report["robot_base_pose_world"])
        if not (np.allclose(base.position, scene.base_frame.position)
                and np.allclose(base.rotation, scene.base_frame.rotation)):
            raise ValueError("Report base frame differs from the live scene")
        dep = StageReport.load(paths.output_dir() / paths.SIM_DEPROJECTION, expect_stage="sim_deprojection")
        capture = StageReport.load(paths.output_dir() / paths.SIM_CAPTURE, expect_stage="sim_capture")
        if not dep.data.get("motion_safe") or dep.data.get("centroid_source") != "optical":
            raise ValueError("Requires optical simulator depth without centroid fallback")
        cloud = paths.output_dir() / "sim_deprojection_cloud.ply"
        if hashlib.sha256(cloud.read_bytes()).hexdigest() != report["input_pointcloud"]["sha256"]:
            raise ValueError("GraspGen input does not match the current simulated cloud")
        self.validation.update(capture_report=str(paths.output_dir() / paths.SIM_CAPTURE),
                               input_cloud_sha256=report["input_pointcloud"]["sha256"],
                               source_stl=scene.mesh_info["source_mesh"],
                               base_frame=scene.base_frame.to_dict())
        if capture.data["source_stl_file"] != scene.mesh_info["source_mesh"]:
            raise ValueError("Capture mesh differs from the live scene")
        exact = "scene_object_pose_exact" in capture.data
        pose = capture.data.get("scene_object_pose_exact", capture.data["ground_truth_pose"])
        # Older capture reports round positions to 4 decimal places (50 um).
        # New reports retain the unrounded pose for this consistency check.
        position, orientation = p.getBasePositionAndOrientation(scene.tool, physicsClientId=scene.cid)
        expected_quat = np.asarray(pose["orientation_xyzw"], dtype=float)
        expected_quat /= np.linalg.norm(expected_quat)
        if not (np.allclose(pose["position_xyz_m"], position, atol=1e-6 if exact else 5.1e-5, rtol=0)
                and abs(np.dot(expected_quat, orientation)) > .9999):
            raise ValueError("Capture object pose differs from the live scene")
        provenance = report.get("server_metadata", {}).get("ros2project", {})
        if (provenance.get("planner") != "diffusion"
                or provenance.get("gripper") != "robotiq_2f_140"
                or provenance.get("revision") != report["configuration"]["expected_revision"]):
            raise ValueError("Missing compatible official server provenance")

    def _candidate(self, candidate, vertices):
        scene = self.scene
        # TCP y is GraspGen closing x; flange axes use this same TCP orientation.
        canonical = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, .195], [0, 0, 0, 1.]])
        base_tcp = rigid_transform(candidate["T_base_gripper"]) @ canonical
        if not np.allclose(base_tcp, rigid_transform(candidate["T_base_tcp"]), atol=1e-5):
            raise ValueError("TCP transform differs from the nominal Robotiq simulation adapter")
        self._heartbeat()
        self.phases, self.waypoints = [], []
        self.cell.set_stationary_tool(self.tool_half, self.tool_position_base, self.tool_orientation_base)
        self.cell.set_joints(self.start_q)
        self.planner = CheckedJointPlanner(self.cell, scene.config, self._heartbeat)
        self._phase_label = "bilateral pad geometry and closing sweep"
        refined, self.close_width, refinement = refine_grasp(
            self.cell, rigid_transform(candidate["T_base_gripper"]),
            (scene.source, scene.destination), self._heartbeat)
        self.executed_candidate = copy.deepcopy(candidate)
        # The source proposal and source score remain in report['candidates'].
        # Only simulation_selected contains the derived execution pose.
        self.executed_candidate["geometric_refinement"] = refinement
        base_tcp = refined @ TCP
        world_base = np.eye(4)
        world_base[:3, :3] = scene.base_frame.rotation
        world_base[:3, 3] = scene.base_frame.position
        for key, transform in (("T_base_gripper", refined), ("T_base_tcp", base_tcp),
                               ("T_world_gripper", world_base@refined),
                               ("T_world_tcp", world_base@base_tcp)):
            self.executed_candidate[key] = transform.tolist()
        centered = np.asarray(candidate["T_centered_gripper"]).copy()
        centered[:3, 3] += refined[:3, 3]-np.asarray(candidate["T_base_gripper"])[:3, 3]
        self.executed_candidate["T_centered_gripper"] = centered.tolist()
        self.executed_candidate["tcp_quaternion_xyzw_base"] = quaternion_from_matrix(base_tcp[:3, :3])
        self.executed_candidate["score_semantics"] = "Original proposal score; refined pose not model-rescored"
        self._display_report["simulation_checking"] = self.executed_candidate
        scene.show_grasp(self._display_report)
        point, quat = scene.base_frame.pose_base_to_world(base_tcp[:3, 3], quaternion_from_matrix(base_tcp[:3, :3]))
        point, quat = np.asarray(point), np.asarray(quat)
        R = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        flange_quat = quaternion_from_matrix(base_tcp[:3, :3])
        opened = lambda: self._clearance(ROBOTIQ_PROXY.opening, False)
        self._phase_label = "grasp IK (multiple seeds)"
        grasp_q = self.planner.solve(base_tcp[:3, 3], flange_quat, self.start_q, opened, multistart=True)
        pregrasp = point - R[:, 2] * .12
        pre_base, pre_quat = scene.base_frame.pose_world_to_base(pregrasp, quat)
        self._phase_label = "grasp-to-pregrasp Cartesian corridor"
        # Solve backwards from the grasp to preserve its feasible IK branch.
        descent = self.planner.cartesian(grasp_q, pre_base, pre_quat, opened)[::-1].copy()
        targets = pick_and_place_targets(point, destination_tray=scene.trays[scene.destination],
                                         tool_bottom_z=scene.tool_bottom,
                                         motion=scene.config["motion_control"], base_frame=scene.base_frame)
        targets[0] = Target(APPROACH, pregrasp)
        # Withdraw along the grasp axis first after opening, before a free transit.
        release_point = next(t.position for t in targets if t.event == "release")
        targets = [Target(RETREAT, release_point - R[:, 2] * .12) if t.label == RETREAT else t
                   for t in targets]
        seed, carried = self.start_q.copy(), False
        for target in targets:
            self._phase_label = target.label
            self._heartbeat()
            start_width = self.close_width if carried else ROBOTIQ_PROXY.opening
            end_width = (self.close_width if target.event == "attach" else ROBOTIQ_PROXY.opening
                         if target.event == "release" else start_width)
            clearance = lambda: self._clearance(start_width, carried)
            target_base, target_quat = scene.base_frame.pose_world_to_base(target.position, quat)
            if target.event:
                samples = np.tile(seed, (145, 1))
            else:
                if target.label == APPROACH:
                    path = self.planner.transit(seed, descent[0], clearance)
                elif target.label == DESCEND:
                    path = descent
                elif target.label in (TRANSFER, RETURN):
                    goal = self.planner.solve(target_base, target_quat, seed, clearance, multistart=True)
                    path = self.planner.transit(seed, goal, clearance)
                else:
                    path = self.planner.cartesian(seed, target_base, target_quat, clearance)
                self.cell.set_joints(seed)
                old_position, old_quat = self.cell.tcp_pose()
                from scipy.spatial.transform import Rotation
                angle = (Rotation.from_quat(old_quat).inv()*Rotation.from_quat(target_quat)).magnitude()
                velocities = scene.config["motion_control"]["velocities"]
                minimum_duration = max(
                    np.linalg.norm(np.asarray(old_position)-target_base) / velocities["cartesian_linear_speed_m_s"],
                    angle / velocities["cartesian_angular_speed_rad_s"])
                samples = self.planner.retime(target.label, path, minimum_duration)
            phase = {"label": target.label, "position": np.asarray(target.position), "orientation": quat.copy(),
                     "event": target.event, "samples": samples,
                     "duration": (len(samples)-1)*diff_ik.CONTROL_DT,
                     "start_width": start_width, "end_width": end_width, "carried": carried}
            self._validate_phase(phase)
            self.phases.append(phase)
            self.waypoints.append({"phase": target.label, "tcp_position_m": target.position.tolist(),
                                   "quaternion_xyzw": quat.tolist(), "duration_s": phase["duration"]})
            seed = samples[-1]
            if target.event == "attach":
                tcp, orn = self.cell.tcp_pose()
                self.cell.attach_tool(self.tool_half, self.tool_position_base, self.tool_orientation_base, tcp, orn)
                carried = True
            elif target.event == "release":
                position, orientation = p.getBasePositionAndOrientation(self.cell._tool_body,
                                                                       physicsClientId=self.cell.cid)
                self.cell.set_stationary_tool(self.tool_half, position, orientation)
                carried = False
        self.cell.set_stationary_tool(self.tool_half, self.tool_position_base, self.tool_orientation_base)

    def _clearance(self, width, carried, phase_label=None):
        # Relax fingertips at the supporting surfaces, never allow penetration.
        relaxed = frozenset((part, tray) for part in ("fingers", "tool")
                            for tray in (self.scene.source, self.scene.destination))
        label = phase_label if phase_label is not None else self._phase_label
        supports = (frozenset([self.scene.source]) if carried and label == LIFT else
                    frozenset([self.scene.destination]) if carried and label in (PLACE, RELEASE) else frozenset())
        result = self.cell.clearance(width=width, carried=carried, relaxed=relaxed,
                                     margin=.010, contact_margin=.0005, check_target=True,
                                     allow_finger_contact=width <= self.close_width + .001,
                                     tool_support_contacts=supports)
        if not result.clear:
            raise ValueError("Simulation path collision: " + result.violations[0])

    def _validate_phase(self, phase):
        samples = phase["samples"]
        if np.max(np.abs(np.diff(samples, axis=0))) > .25:
            raise ValueError("IK branch discontinuity exceeds tutor platform 0.25 rad limit")
        for i in sorted(set(range(0, len(samples), 4)) | {len(samples) - 1}):
            self._heartbeat()
            q = samples[i]
            if not np.isfinite(q).all() or self.cell.joint_margin(q) < .03:
                raise ValueError(f"Joint limit at {phase['label']}")
            self.cell.set_joints(q)
            alpha = i / (len(samples) - 1)
            self._clearance(phase["start_width"] * (1-alpha) + phase["end_width"] * alpha, phase["carried"])
        actual, orn = self.cell.tcp_pose()
        expected = self.scene.base_frame.world_to_base(phase["position"])
        expected_quat = self.scene.base_frame.pose_world_to_base(phase["position"], phase["orientation"])[1]
        if (np.linalg.norm(np.asarray(actual) - expected) > .001
                or abs(np.dot(orn, expected_quat)) < math.cos(.005 / 2)):
            raise ValueError(f"TCP tracking outside tolerance at {phase['label']}")

    def tick(self, dt=diff_ik.CONTROL_DT):
        try:
            self._tick_checked(dt)
        except (ValueError, RuntimeError) as exc:
            self._runtime_error = str(exc)
            self.validation["execution_error"] = self._runtime_error
            self.validation["execution_error_phase"] = self.phases[self.index]["label"]
            self.validation["execution_elapsed_s"] = self.total
            self.validation["allowed"] = False
            self.validation["status"] = "EXECUTION_FAILED"
            self._save_validation(None)
            raise

    def _tick_checked(self, dt):
        if self.finished:
            return
        phase = self.phases[self.index]
        # Check measured joints as well as the planned path. No stale authorization
        # can let the robot continue after tracking into an obstacle.
        q = [p.getJointState(self.scene.robot, j, physicsClientId=self.scene.cid)[0] for j in self.scene.joints]
        self._ticks += 1
        if self._ticks % 4 == 0:
            self.cell.set_joints(q)
            if self.cell.joint_margin(q) < .02:
                raise ValueError("Measured robot joint limit")
            if self.scene.attachment is not None and self.cell._tcp_to_tool is None:
                pos, orn = p.getBasePositionAndOrientation(self.scene.tool, physicsClientId=self.scene.cid)
                pos, orn = self.scene.base_frame.pose_world_to_base(pos, orn)
                tcp, tcp_orn = self.cell.tcp_pose()
                self.cell.attach_tool(self.tool_half, pos, orn, tcp, tcp_orn)
            elif self.scene.attachment is None:
                pos, orn = p.getBasePositionAndOrientation(self.scene.tool, physicsClientId=self.scene.cid)
                pos, orn = self.scene.base_frame.pose_world_to_base(pos, orn)
                self.cell.set_stationary_tool(self.tool_half, pos, orn)
            self._clearance(self.scene.gripper_width, self.scene.attachment is not None, phase["label"])
        old_index = self.index
        super().tick(dt)
        alpha = 1. if self.finished or self.index != old_index else min(self.elapsed / phase["duration"], 1.)
        self.scene.gripper_width = phase["start_width"] * (1-alpha) + phase["end_width"] * alpha

    def report(self):
        result = super().report()
        result.update(gripper_model=ROBOTIQ_PROXY.name, simulation_motion=self.validation,
                      finger_control="URDF mimic joint positions (kinematic, not force controlled)",
                      execution_checks_passed=self._runtime_error is None,
                      tcp_offset_flange_xyz_m=list(ROBOTIQ_PROXY.tcp_offset),
                      collision_free_path_validated=False,
                      sampled_collision_checks_passed=True, physical_grasp_validated=False)
        return result

    def close(self):
        if self.cell is not None:
            self.cell.close()
            self.cell = None
