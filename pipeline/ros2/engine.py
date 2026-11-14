"""Main-thread PyBullet engine, independent of ROS bindings for numerical tests."""
from __future__ import annotations

import time

import numpy as np
import pybullet as p

from pipeline.common.report import StageReport, write_json
from pipeline.ros2.contracts import DT, phase_manifest, verify_initial_joints


class SimulationEngine:
    def __init__(self, out, config, task, *, gui=False, seed=42, cancel_requested=lambda: False,
                 scene_factory=None, motion_factory=None):
        from pipeline.sim.robot_scene import RobotScene
        from pipeline.sim.graspgen_motion import GraspGenMotionPreview
        self.out = out
        self.scene = (scene_factory or RobotScene)(config, task, gui=gui, seed=seed)
        self.motion_factory = motion_factory or GraspGenMotionPreview
        self.cancel_requested = cancel_requested
        self.motion = None
        self.active = False
        self.fault = None
        self.completed = []
        self.phase_index = None
        self.ticks = 0
        self.arm_actions = self.gripper_actions = 0
        self.last_state = None

    def joints(self):
        states = p.getJointStates(self.scene.robot, self.scene.joints, physicsClientId=self.scene.cid)
        return [s[0] for s in states], [s[1] for s in states]

    def prepare(self, report):
        if self.motion is not None or self.fault:
            raise ValueError("This run has already been planned or stopped")
        self.motion = self.motion_factory(self.scene, report, cancel_requested=self.cancel_requested)
        StageReport("grasp", data=report).save(self.out / "grasp_simulation_selection.json")
        result = phase_manifest(self.motion.phases, self.out, self.out.name)
        result["robot_description"] = self.scene.robot_model.report(self.scene.gripper_geometry)
        write_json(self.out / "ros2_execution_plan.json", result)
        return result

    def next_phase(self):
        if self.fault or not self.motion or self.motion.finished or self.active:
            raise ValueError(self.fault or "No idle validated phase available")
        return self.motion.phases[self.motion.index]

    def start(self):
        phase = self.next_phase()
        verify_initial_joints(self.joints()[0], phase["samples"][0])
        self.phase_index = self.motion.index
        self.active = True

    def stop(self, reason):
        if self.fault is None:
            self.fault = str(reason)
        self.active = False
        self.scene.command_joints(self.joints()[0])

    def step(self):
        if self.active:
            try:
                self.motion.tick(DT)
            except (ValueError, RuntimeError) as exc:
                self.stop(exc)
        self.scene.step()
        self.ticks += 1
        q, v = self.joints()
        tcp, quat = self.scene.tcp_pose()
        self.last_state = dict(q=q, velocity=v, tcp=list(tcp), quaternion=list(quat),
                               width=float(self.scene.gripper_width), at=time.monotonic(), tick=self.ticks)
        if self.active:
            phase = self.motion.phases[self.phase_index]
            coordinate = min(self.motion.elapsed / phase["duration"], 1) * (len(phase["samples"]) - 1)
            # The existing motion implementation checks measured geometry every four
            # ticks. Also enforce a joint-space path tolerance on every physics step.
            if self.motion.index != self.phase_index:
                desired = phase["samples"][-1]
            else:
                left = min(int(coordinate), len(phase["samples"]) - 2)
                alpha = coordinate - left
                desired = phase["samples"][left] * (1-alpha) + phase["samples"][left+1] * alpha
            self.last_state["desired"] = desired.tolist()
            if np.max(np.abs(np.asarray(q) - desired)) > .05:
                self.stop("Measured joint path tolerance exceeded (0.05 rad)")
            elif self.motion.index != self.phase_index:
                if np.max(np.abs(np.asarray(q) - phase["samples"][-1])) > .01:
                    self.stop("Measured joint goal tolerance exceeded (0.01 rad)")
                else:
                    self.active = False
                    self.completed.append(self.phase_index)
                    if phase["event"]:
                        self.gripper_actions += 1
                    else:
                        self.arm_actions += 1

    def finish(self):
        if (self.fault or not self.motion or not self.motion.finished or self.active
                or self.completed != list(range(len(self.motion.phases)))):
            raise ValueError(self.fault or "All validated phases must finish before checking placement")
        # The caller steps physics for at least 0.5 simulated seconds before this.
        result = self.motion.report()
        result.update(backend="ros2_follow_joint_trajectory_pybullet_servo",
                      transport="ros2", arm_actions_succeeded=self.arm_actions,
                      gripper_actions_succeeded=self.gripper_actions,
                      joint_path_tolerance_rad=.05, joint_goal_tolerance_rad=.01,
                      execution_checks_passed=not self.fault,
                      sampled_collision_checks_passed=not self.fault,
                      physical_grasp_validated=False)
        StageReport("sim_robot_motion", status=result["status"], data=result).save(self.out / "sim_robot_motion.json")
        self.scene.save_overview(self.out / "sim_robot_result.png")
        return result

    def close(self):
        if self.motion is not None:
            self.motion.close()
        self.scene.close()
