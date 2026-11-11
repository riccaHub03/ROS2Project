"""Scena condivisa da camera, analisi geometrica e anteprima KUKA nativa."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data
import trimesh

from pipeline.common import paths
from pipeline.common.reconstruction import CameraModel
from pipeline.common.report import StageReport, write_json
from pipeline.common.scene import camera_intrinsics, camera_pose, get_tray, round_list
from pipeline.mesh_loader import find_instrument_mesh
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.sim.placement import sample_tool_pose

log = logging.getLogger("robot_scene")
TCP_OFFSET = (0.0, 0.0, 0.10)
HOME = (0.0, 0.35, 0.0, -1.3, 0.0, 1.0, 0.0)


def prepare_mesh(source):
    """Centra la mesh in metri e la adagia con la normale PCA lungo Z."""
    source = Path(source)
    key = hashlib.sha256(
        f"flat-v1:{source.resolve()}:{source.stat().st_mtime_ns}:{source.stat().st_size}".encode()
    ).hexdigest()[:16]
    cache = paths.cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    obj = cache / f"robot_scene_{key}.obj"
    meta = obj.with_suffix(".json")
    if obj.exists() and meta.exists():
        return str(obj), json.loads(meta.read_text())
    mesh = trimesh.load(source, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4:
        raise ValueError(f"Mesh non valida: {source}")
    span = np.ptp(mesh.vertices, axis=0)
    scale = 0.001 if max(span) > 5 else (0.01 if max(span) > 0.6 else 1.0)
    vertices = np.asarray(mesh.vertices) * scale
    center = vertices.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((vertices - center).T))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    vertices = (vertices - center) @ axes
    vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    mesh.vertices = vertices
    dimensions = np.ptp(vertices, axis=0)
    original_faces = len(mesh.faces)
    if original_faces > 60000:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=60000, aggression=3)
        except ImportError as exc:
            raise RuntimeError(
                "Manca fast-simplification. Installa: python -m pip install fast-simplification"
            ) from exc
    mesh.export(obj)
    data = {
        "source_mesh": str(source.resolve()), "scale_to_meters": scale,
        "dimensions_m": dimensions.tolist(), "original_faces": original_faces,
        "render_faces": len(mesh.faces), "simplification": "quadric_decimation",
    }
    write_json(meta, data)
    return str(obj), data


class RobotScene:
    def __init__(self, config, plan, *, gui=False, seed=42):
        self.config, self.plan, self.gui = config, plan, gui
        self.base_frame = RobotBaseFrame.from_config(config)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if self.cid < 0:
            raise RuntimeError("Impossibile aprire PyBullet")
        self.attachment = None
        self.gripper_width = 0.080
        self.markers = []
        self.path_lines = []
        self.last_tcp = None
        self.step_count = 0
        self.status_id = -1
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
            p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
            p.setTimeStep(1 / 240, physicsClientId=self.cid)
            p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=self.cid)
            if gui:
                # PyBullet's built-in key shortcuts (W wireframe, A AABB boxes, G hide panels...) clash with ours
                p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=self.cid)
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.cid)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.cid)
                p.resetDebugVisualizerCamera(1.9, 48, -32, [0.45, 0, 0.30], physicsClientId=self.cid)
            self._build_environment()
            self._build_robot()
            self._build_tool(seed)
            self.sync_gripper()
            self.capture()
            if gui:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.cid)
        except BaseException:
            self.close()
            raise

    def box(self, position, dimensions, color):
        half = (np.asarray(dimensions) / 2).tolist()
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.cid)
        return p.createMultiBody(0, col, vis, position, physicsClientId=self.cid)

    def _build_environment(self):
        env = self.config["environment"]
        p.loadURDF("plane.urdf", [0, 0, -0.16], physicsClientId=self.cid)
        table = env["surgical_table"]
        self.box(table["center_xyz_m"], table["dimensions_xyz_m"], table["color_rgba"])
        self.trays = env["trays"]
        self.tray_bodies = {}
        for name, tray in self.trays.items():
            self.tray_bodies[name] = self.box(tray["position_xyz_m"], tray["dimensions_xyz_m"], tray["color_rgba"])
        self.source = self.plan["source_tray"]
        self.destination = self.plan["destination_tray"]
        if self.source not in self.trays or self.destination not in self.trays:
            raise ValueError("La visualizzazione richiede vassoi definiti nella configurazione")
        if self.source == self.destination:
            raise ValueError("Origine e destinazione devono essere vassoi diversi")
        if self.gui:
            for name, tray in self.trays.items():
                role = "origine" if name == self.source else "destinazione"
                position = np.asarray(tray["position_xyz_m"]) + [-0.2, 0, 0.08]
                p.addUserDebugText(f"{name} - {role}", position, [0.05, 0.12, 0.25],
                                   textSize=1.1, physicsClientId=self.cid)

    def _build_robot(self):
        self.robot = p.loadURDF(
            "kuka_iiwa/model.urdf", basePosition=self.base_frame.position.tolist(),
            baseOrientation=self.base_frame.quaternion, useFixedBase=True, physicsClientId=self.cid,
        )
        self.joints = list(range(7))
        self.ee = 6
        self.limits = [p.getJointInfo(self.robot, j, physicsClientId=self.cid)[8:10] for j in self.joints]
        self.reset_robot()
        palm = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.032, 0.025, 0.0125],
                                   visualFramePosition=[0, 0, 0.025], rgbaColor=[0.15, 0.18, 0.22, 1],
                                   physicsClientId=self.cid)
        finger = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.013, 0.004, 0.025],
                                     rgbaColor=[0.2, 0.25, 0.3, 1], physicsClientId=self.cid)
        self.gripper = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=palm, baseCollisionShapeIndex=-1,
            linkMasses=[0, 0], linkCollisionShapeIndices=[-1, -1],
            linkVisualShapeIndices=[finger, finger], linkPositions=[[0, 0.005, 0.075], [0, -0.005, 0.075]],
            linkOrientations=[[0, 0, 0, 1]] * 2, linkInertialFramePositions=[[0, 0, 0]] * 2,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2, linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_PRISMATIC] * 2, linkJointAxis=[[0, 1, 0], [0, -1, 0]],
            physicsClientId=self.cid,
        )

    def _build_tool(self, seed):
        source = find_instrument_mesh(target_keyword=self.plan["target_instrument"], config=self.config)
        if not source:
            raise FileNotFoundError("Nessuna mesh trovata: aggiungi gli STL in data/inputs")
        self.mesh_file, self.mesh_info = prepare_mesh(source)
        dims = np.asarray(self.mesh_info["dimensions_m"])
        tray = self.trays[self.source]
        top = tray["position_xyz_m"][2] + tray["dimensions_xyz_m"][2] / 2
        x, y, yaw = sample_tool_pose(tray["position_xyz_m"], tray["dimensions_xyz_m"], dims[:2], seed,
                                    base_xy=self.base_frame.position[:2])
        self.tool_initial_pos = [x, y, top + dims[2] / 2 + 0.001]
        self.tool_initial_orn = p.getQuaternionFromEuler([0, 0, yaw])
        vis = p.createVisualShape(p.GEOM_MESH, fileName=self.mesh_file,
                                  rgbaColor=[0.28, 0.31, 0.35, 1], physicsClientId=self.cid)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=(dims / 2).tolist(), physicsClientId=self.cid)
        self.tool = p.createMultiBody(0, col, vis, self.tool_initial_pos, self.tool_initial_orn,
                                     physicsClientId=self.cid)
        self.tool_bottom = top + 0.001

    def reset_robot(self):
        for j, q in zip(self.joints, HOME):
            p.resetJointState(self.robot, j, q, physicsClientId=self.cid)
        self.command_joints(HOME)

    def command_joints(self, q):
        p.setJointMotorControlArray(self.robot, self.joints, p.POSITION_CONTROL,
                                   targetPositions=list(q), forces=[500] * 7,
                                   positionGains=[0.3] * 7, velocityGains=[1] * 7,
                                   physicsClientId=self.cid)

    def tcp_pose(self):
        state = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True, physicsClientId=self.cid)
        return p.multiplyTransforms(state[4], state[5], TCP_OFFSET, [0, 0, 0, 1])

    def sync_gripper(self):
        state = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True, physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.gripper, state[4], state[5], physicsClientId=self.cid)
        q = max(0, (self.gripper_width - 0.002) / 2)
        for j in (0, 1):
            p.resetJointState(self.gripper, j, q, physicsClientId=self.cid)

    def solve_ik(self, position, orientation, seed):
        flange, _ = p.multiplyTransforms(position, orientation, [0, 0, -TCP_OFFSET[2]], [0, 0, 0, 1])
        current = [p.getJointState(self.robot, j, physicsClientId=self.cid)[0] for j in self.joints]
        try:
            for j, q in zip(self.joints, seed):
                p.resetJointState(self.robot, j, q, physicsClientId=self.cid)
            for _ in range(4):
                q = p.calculateInverseKinematics(
                    self.robot, self.ee, flange, orientation, jointDamping=[0.01] * 7,
                    maxNumIterations=150, residualThreshold=1e-6, physicsClientId=self.cid,
                )[:7]
                for j, value in zip(self.joints, q):
                    p.resetJointState(self.robot, j, value, physicsClientId=self.cid)
                actual, quat = self.tcp_pose()
                error = np.linalg.norm(np.asarray(actual) - position)
                angle = 2 * math.acos(float(np.clip(abs(np.dot(quat, orientation)), 0, 1)))
                if error < 0.003 and angle < 0.035:
                    break
            if not np.isfinite(q).all() or any(value < lo or value > hi for value, (lo, hi) in zip(q, self.limits)):
                raise ValueError("Soluzione IK fuori dai limiti articolari")
            if error >= 0.003 or angle >= 0.035:
                raise ValueError(f"Posa non raggiungibile: errore IK {error * 1000:.1f} mm, {math.degrees(angle):.1f} gradi")
            return list(q)
        finally:
            for j, value in zip(self.joints, current):
                p.resetJointState(self.robot, j, value, physicsClientId=self.cid)

    def attach_tool(self):
        if self.attachment is not None:
            return
        state = p.getLinkState(self.robot, self.ee, computeForwardKinematics=True, physicsClientId=self.cid)
        tool_pos, tool_orn = p.getBasePositionAndOrientation(self.tool, physicsClientId=self.cid)
        inv = p.invertTransform(state[0], state[1])
        relative = p.multiplyTransforms(*inv, tool_pos, tool_orn)
        p.changeDynamics(self.tool, -1, mass=0.080, physicsClientId=self.cid)
        self.attachment = p.createConstraint(
            self.robot, self.ee, self.tool, -1, p.JOINT_FIXED, [0, 0, 0], relative[0], [0, 0, 0],
            parentFrameOrientation=relative[1], childFrameOrientation=[0, 0, 0, 1], physicsClientId=self.cid,
        )
        p.changeConstraint(self.attachment, maxForce=500, physicsClientId=self.cid)
        # Il trasferimento e' dimostrativo: il vincolo non misura contatti o forza di presa.
        for j in range(-1, 7):
            p.setCollisionFilterPair(self.robot, self.tool, j, -1, 0, physicsClientId=self.cid)

    def release_tool(self):
        if self.attachment is not None:
            p.removeConstraint(self.attachment, physicsClientId=self.cid)
            self.attachment = None

    def restore(self):
        self.release_tool()
        p.changeDynamics(self.tool, -1, mass=0, physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.tool, self.tool_initial_pos, self.tool_initial_orn, physicsClientId=self.cid)
        p.resetBaseVelocity(self.tool, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        self.gripper_width = 0.080
        self.last_tcp = None
        self.reset_robot()
        self.sync_gripper()

    def capture(self):
        """Render the overhead RGB-D view and write the sim_capture / sim_reference_pose reports."""
        out = paths.output_dir()
        pose = camera_pose(self.config, get_tray(self.config, self.source))
        camera = CameraModel(pose.position, pose.target, camera_intrinsics(self.config))
        i = camera.intrinsics
        w, h, fx, fy, cx, cy = i.width, i.height, i.fx, i.fy, i.cx, i.cy
        near, far = 0.1, 3.0
        view = p.computeViewMatrix(camera.position, camera.target, [-1, 0, 0])
        proj = p.computeProjectionMatrix(-cx * near / fx, (w - cx) * near / fx,
                                         -(h - cy) * near / fy, cy * near / fy, near, far)
        rendered = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=self.cid)
        rgb = np.asarray(rendered[2], dtype=np.uint8).reshape(h, w, 4)[..., :3]
        zbuf = np.asarray(rendered[3], dtype=np.float32).reshape(h, w)
        depth = far * near / (far - (far - near) * zbuf)
        ids = np.asarray(rendered[4]).reshape(h, w)
        mask = (ids >= 0) & ((ids & ((1 << 24) - 1)) == self.tool)
        self.ground_truth_mask = mask
        rgb_png, depth_npy = out / "sim_capture_rgb.png", out / "sim_capture_depth.npy"
        depth_vis = out / "sim_capture_depth_vis.png"
        truth_png, overview_png = out / "sim_capture_truth_mask.png", out / "sim_robot_scene.png"
        Image.fromarray(rgb).save(rgb_png)
        np.save(depth_npy, depth)
        Image.fromarray((mask * 255).astype(np.uint8)).save(truth_png)
        self.save_overview(overview_png)

        tray_info = self.trays[self.source]
        table_limit = float(camera.position[2] - tray_info["position_xyz_m"][2]) + 0.10
        valid = depth < table_limit
        if np.any(valid):
            d_min = float(np.min(depth[valid]))
            d_max = float(np.max(depth[valid]))
            d_range = d_max - d_min
            if d_range > 1e-4:
                d_norm = np.clip(1.0 - (depth - d_min) / d_range, 0.0, 1.0)
            else:
                d_norm = np.zeros_like(depth)
            d_norm[~valid] = 0.0
        else:
            d_min, d_max = float(np.min(depth)), float(np.max(depth))
            d_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)
        Image.fromarray((d_norm * 255.0).astype(np.uint8)).save(depth_vis)

        instrument = self.plan["target_instrument"]
        initial_pose = {"position_xyz_m": round_list(self.tool_initial_pos),
                        "orientation_xyzw": round_list(self.tool_initial_orn, 5)}
        base_pos, base_quat = self.base_frame.pose_world_to_base(self.tool_initial_pos, self.tool_initial_orn)
        pose_base = {"position_xyz_m": base_pos.tolist(), "orientation_xyzw": base_quat}
        self.capture_report = StageReport(
            stage="sim_capture", inputs={"plan": "in-memory"},
            artifacts={"rgb_png": str(rgb_png), "depth_npy": str(depth_npy), "depth_visual_png": str(depth_vis),
                       "truth_mask_png": str(truth_png)},
            data={"mode": "single", "simulated_instrument": instrument, "mesh_file": self.mesh_file,
                  "source_stl_file": self.mesh_info["source_mesh"], "source_tray": self.source,
                  "ground_truth_pose": initial_pose, "camera": camera.to_dict(),
                  "robot_base_pose_world": self.base_frame.to_dict()})
        self.capture_report.save(out / paths.SIM_CAPTURE)
        StageReport(
            stage="sim_reference_pose", inputs={"plan": "in-memory"}, artifacts={"placement_png": str(overview_png)},
            data={"target_instrument": instrument, "source_stl_file": self.mesh_info["source_mesh"],
                  "source_tray": self.source,
                  "stl_dimensions_m": dict(zip(("length_v1", "width_v2", "thickness_v3"), self.mesh_info["dimensions_m"])),
                  "pose": initial_pose, "pose_base_frame": pose_base,
                  "robot_base_pose_world": self.base_frame.to_dict(),
                  "pose_source": "same_pybullet_scene_as_rgbd"},
        ).save(out / paths.SIM_REFERENCE_POSE)

    def save_overview(self, path):
        view = p.computeViewMatrixFromYawPitchRoll([0.45, 0, 0.30], 1.9, 48, -32, 0, 2)
        proj = p.computeProjectionMatrixFOV(48, 1.5, 0.05, 5)
        rendered = p.getCameraImage(1050, 700, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=self.cid)
        Image.fromarray(np.asarray(rendered[2], dtype=np.uint8).reshape(700, 1050, 4)[..., :3]).save(path)

    def status(self, text, color=(0.05, 0.1, 0.3)):
        print(text, flush=True)
        if self.gui:
            self.status_id = p.addUserDebugText(text, [0.05, -0.75, 0.9], textColorRGB=color,
                                               textSize=1.2, replaceItemUniqueId=self.status_id,
                                               physicsClientId=self.cid)

    def show_grasp(self, report):
        if report.get("grasp_backend") == "graspgen":
            from pipeline.graspgen.preview import gripper_lines
            for candidate in report["candidates"][:20]:
                line = self.base_frame.base_to_world(gripper_lines(candidate["T_base_gripper"]))
                for a, b in zip(line[:-1], line[1:]):
                    p.addUserDebugLine(a, b, [0.1, 0.7, 0.3], 2, physicsClientId=self.cid)
            self.status("GraspGen: nominal Robotiq poses; physical grasp UNVALIDATED")
        data = report["designated_grasp_geometry"]
        pos, quat = self.base_frame.pose_base_to_world(
            data["grasp_centroid_base_frame_xyz_m"],
            data["recommended_gripper_orientation"]["quaternion_xyzw"],
        )
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.008, rgbaColor=[0.1, 1, 0.2, 0.9], physicsClientId=self.cid)
        self.markers.append(p.createMultiBody(0, -1, vis, pos, physicsClientId=self.cid))
        rotation = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        for axis, color in zip(rotation.T, ([1, 0, 0], [0, 0.75, 0], [0, 0, 1])):
            p.addUserDebugLine(pos, np.asarray(pos) + axis * 0.07, color, 3, physicsClientId=self.cid)

    def show_path(self, waypoints):
        if not self.gui:
            return
        for line in self.path_lines:
            p.removeUserDebugItem(line, physicsClientId=self.cid)
        self.path_lines.clear()
        previous, _ = self.tcp_pose()
        for waypoint in waypoints:
            target = waypoint["tcp_position_m"]
            self.path_lines.append(p.addUserDebugLine(previous, target, [0.9, 0.6, 0.05], 2,
                                                       physicsClientId=self.cid))
            previous = target

    def step(self):
        p.stepSimulation(physicsClientId=self.cid)
        self.sync_gripper()
        self.step_count += 1
        if self.gui and self.step_count % 8 == 0:
            current, _ = self.tcp_pose()
            if self.last_tcp is not None and np.linalg.norm(np.asarray(current) - self.last_tcp) > 0.001:
                p.addUserDebugLine(self.last_tcp, current, [0.05, 0.65, 0.25], 2,
                                   lifeTime=10, physicsClientId=self.cid)
            self.last_tcp = current

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
