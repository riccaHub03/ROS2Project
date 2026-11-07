"""Anteprima IK con servo PyBullet e presa idealizzata tramite vincolo fisso."""

import math

import numpy as np
import pybullet as p

from pipeline.frame_transformer import RobotBaseFrame


class MotionPreview:
    def __init__(self, scene, grasp_report):
        self.scene = scene
        if "robot_base_pose_world" in grasp_report:
            source_frame = RobotBaseFrame.from_dict(grasp_report["robot_base_pose_world"])
            if (not np.allclose(source_frame.position, scene.base_frame.position)
                    or not np.allclose(source_frame.rotation, scene.base_frame.rotation)):
                raise ValueError("La presa usa una posa della base diversa: ricalcola deproject e grasp")
        data = grasp_report["designated_grasp_geometry"]
        if not data["gripper_compatibility_check"]["parallel_jaw_suitable"]:
            raise ValueError("La zona di presa supera l'apertura della pinza: movimento annullato")
        point, quat = scene.base_frame.pose_base_to_world(
            data["grasp_centroid_base_frame_xyz_m"],
            data["recommended_gripper_orientation"]["quaternion_xyzw"],
        )
        quat = np.asarray(quat)
        tray = scene.trays[scene.destination]
        place = np.asarray(tray["position_xyz_m"], dtype=float).copy()
        place[2] += tray["dimensions_xyz_m"][2] / 2 + 0.003 + (point[2] - scene.tool_bottom)
        self.close_width = max(0.004, data["local_cross_section_m"]["width_along_v2"])
        if self.close_width > 0.080:
            raise ValueError("Apertura richiesta fuori dal campo della pinza")
        motion = scene.config.get("motion_control", {})
        heights = motion.get("heights_z_m", {})
        lift = max(heights.get("lift_clearance_z", 0.35), point[2] + 0.18)
        approach = max(heights.get("approach_z", 0.32), point[2] + 0.12)
        self.phases = []
        current, current_quat = scene.tcp_pose()
        seed = [p.getJointState(scene.robot, j, physicsClientId=scene.cid)[0] for j in scene.joints]
        targets = [
            ("Avvicinamento", [point[0], point[1], approach], None),
            ("Discesa", point, None),
            ("Chiusura pinza - vincolo ideale", point, "attach"),
            ("Sollevamento", [point[0], point[1], lift], None),
            ("Trasferimento", [place[0], place[1], lift], None),
            ("Deposito", place, None),
            ("Apertura pinza", place, "release"),
            ("Risalita", [place[0], place[1], lift], None),
            ("Rientro", scene.base_frame.base_to_world([0.48, 0, 0.40]), None),
        ]
        self.waypoints = []
        for label, position, event in targets:
            position = np.asarray(position, dtype=float)
            if event:
                duration = 0.6
                samples = [seed, seed]
            else:
                distance = np.linalg.norm(position - current)
                angle = 2 * math.acos(float(np.clip(abs(np.dot(current_quat, quat)), 0, 1)))
                duration = max(0.6, distance / 0.12, angle / 0.8)
                count = max(3, int(math.ceil(duration * 12)))
                samples = [seed]
                for alpha in np.linspace(0, 1, count)[1:]:
                    xyz = np.asarray(current) * (1 - alpha) + position * alpha
                    orn = p.getQuaternionSlerp(current_quat, quat, float(alpha))
                    seed = scene.solve_ik(xyz, orn, seed)
                    samples.append(seed)
            self.phases.append({"label": label, "duration": duration, "samples": np.asarray(samples),
                                "position": position, "orientation": quat.copy(), "event": event})
            self.waypoints.append({"phase": label, "tcp_position_m": position.tolist(),
                                   "quaternion_xyzw": quat.tolist(), "duration_s": duration})
            current, current_quat = position, quat
        self.index = 0
        self.elapsed = 0.0
        self.total = 0.0
        self.results = []
        self.finished = False
        self._announced = -1
        scene.show_path(self.waypoints)

    def tick(self, dt=1 / 240):
        if self.finished:
            return
        phase = self.phases[self.index]
        if self._announced != self.index:
            self.scene.status(phase["label"])
            self._announced = self.index
        self.elapsed += dt
        self.total += dt
        alpha = min(self.elapsed / phase["duration"], 1.0)
        samples = phase["samples"]
        coordinate = alpha * (len(samples) - 1)
        left = min(int(coordinate), len(samples) - 2)
        fraction = coordinate - left
        q = samples[left] * (1 - fraction) + samples[left + 1] * fraction
        self.scene.command_joints(q)
        if phase["event"] == "attach":
            self.scene.gripper_width = 0.080 * (1 - alpha) + self.close_width * alpha
        elif phase["event"] == "release":
            self.scene.gripper_width = self.close_width * (1 - alpha) + 0.080 * alpha
        if alpha < 1:
            return
        actual, orn = self.scene.tcp_pose()
        error = float(np.linalg.norm(np.asarray(actual) - phase["position"]))
        angle = 2 * math.acos(float(np.clip(abs(np.dot(orn, phase["orientation"])), 0, 1)))
        if error > 0.010 or angle > 0.08:
            if self.elapsed < phase["duration"] + 2:
                return
            raise RuntimeError(f"Tracking fallito in {phase['label']}: {error * 1000:.1f} mm, {math.degrees(angle):.1f} gradi")
        if phase["event"] == "attach":
            self.scene.attach_tool()
        elif phase["event"] == "release":
            self.scene.release_tool()
        self.results.append({"phase": phase["label"], "tcp_error_mm": error * 1000,
                             "orientation_error_deg": math.degrees(angle)})
        self.index += 1
        self.elapsed = 0
        self.finished = self.index == len(self.phases)

    def report(self):
        position, _ = p.getBasePositionAndOrientation(self.scene.tool, physicsClientId=self.scene.cid)
        tray = self.scene.trays[self.scene.destination]
        center, dims = np.asarray(tray["position_xyz_m"]), np.asarray(tray["dimensions_xyz_m"])
        inside = bool(np.all(np.abs(np.asarray(position[:2]) - center[:2]) < dims[:2] / 2))
        contacts = p.getContactPoints(bodyA=self.scene.tool, bodyB=self.scene.tray_bodies[self.scene.destination],
                                      physicsClientId=self.scene.cid)
        velocity, _ = p.getBaseVelocity(self.scene.tool, physicsClientId=self.scene.cid)
        settled = inside and bool(contacts) and np.linalg.norm(velocity) < 0.02
        return {
            "status": "COMPLETED" if self.finished and settled else "INCOMPLETE",
            "backend": "native_pybullet_position_servo", "ik_and_joint_limits_checked": True,
            "collision_free_path_validated": False, "physical_grasp_validated": False,
            "gripper_model": "visual_parallel_jaws", "attachment_model": "ideal_fixed_constraint",
            "tcp_offset_flange_xyz_m": [0, 0, 0.10], "waypoints": self.waypoints,
            "waypoints_frame_id": "world", "robot_base_pose_world": self.scene.base_frame.to_dict(),
            "measured_phase_errors": self.results, "simulation_duration_s": self.total,
            "final_tool_position_m": position, "tool_center_inside_destination_xy": inside,
            "final_tool_position_frame_id": "world",
            "final_tool_position_base_frame_m": self.scene.base_frame.world_to_base(position).tolist(),
            "destination_contact_count": len(contacts), "tool_settled_on_destination": bool(settled),
        }
