#!/usr/bin/env python3
"""
Modulo: Frame Transformer e Gestione Trasformazioni di Coordinate ROS 2.
Gestisce il passaggio matematico e TF2 tra:
  1. Camera Optical Frame (Z avanti lungo la linea di vista, X destra, Y basso)
  2. Camera Link (montaggio fisico della telecamera)
  3. World e base robot (lbr_link_0), distinti quando la base e' spostata
  4. Flangia / TCP (lbr_link_ee / workbench_tcp), distinti tramite il montaggio della pinza
"""

import math
import numpy as np

try:
    from geometry_msgs.msg import TransformStamped, PoseStamped, Point, Quaternion
    import tf2_ros
    HAS_ROS2_TF = True
except ImportError:
    HAS_ROS2_TF = False


def quaternion_from_matrix(matrix):
    """
    Estrae il quaternione unitario [x, y, z, w] da una matrice di rotazione 3x3.
    Implementazione numericamente stabile (Shepperd / Bar-Itzhack).
    """
    M = np.array(matrix, dtype=np.float64)[:3, :3]
    t = np.trace(M)
    if t > 0.0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (M[2, 1] - M[1, 2]) * s
        y = (M[0, 2] - M[2, 0]) * s
        z = (M[1, 0] - M[0, 1]) * s
    elif (M[0, 0] > M[1, 1]) and (M[0, 0] > M[2, 2]):
        s = 2.0 * math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2])
        w = (M[2, 1] - M[1, 2]) / s
        x = 0.25 * s
        y = (M[0, 1] + M[1, 0]) / s
        z = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = 2.0 * math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2])
        w = (M[0, 2] - M[2, 0]) / s
        x = (M[0, 1] + M[1, 0]) / s
        y = 0.25 * s
        z = (M[1, 2] + M[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1])
        w = (M[1, 0] - M[0, 1]) / s
        x = (M[0, 2] + M[2, 0]) / s
        y = (M[1, 2] + M[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.tolist()


def matrix_from_quaternion(q):
    """
    Calcola la matrice di rotazione 3x3 a partire dal quaternione [x, y, z, w].
    """
    x, y, z, w = q
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float64)


class RobotBaseFrame:
    """Posa della base nel mondo e conversioni di punti e pose fra i due frame."""

    def __init__(self, position_xyz=(0, 0, 0), orientation_rpy=(0, 0, 0), frame_id="lbr_link_0"):
        self.position = np.asarray(position_xyz, dtype=float)
        self.rpy = np.asarray(orientation_rpy, dtype=float)
        if (self.position.shape != (3,) or self.rpy.shape != (3,)
                or not np.isfinite(self.position).all() or not np.isfinite(self.rpy).all()):
            raise ValueError("La posa della base richiede tre coordinate e tre angoli RPY finiti")
        self.frame_id = frame_id
        roll, pitch, yaw = self.rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        self.rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        self.quaternion = quaternion_from_matrix(self.rotation)

    @classmethod
    def from_config(cls, config):
        return cls.from_dict(config.get("environment", {}).get("kuka_base", {}))

    @classmethod
    def from_dict(cls, pose):
        return cls(pose.get("position_xyz_m", [0, 0, 0]),
                   pose.get("orientation_rpy_rad", [0, 0, 0]),
                   pose.get("frame_id", "lbr_link_0"))

    def to_dict(self):
        return {"parent_frame_id": "world", "frame_id": self.frame_id,
                "position_xyz_m": self.position.tolist(),
                "orientation_rpy_rad": self.rpy.tolist(),
                "orientation_xyzw": self.quaternion}

    def world_to_base(self, points):
        return (np.asarray(points, dtype=float) - self.position) @ self.rotation

    def base_to_world(self, points):
        return np.asarray(points, dtype=float) @ self.rotation.T + self.position

    def pose_world_to_base(self, position, quaternion):
        rotation = self.rotation.T @ matrix_from_quaternion(quaternion)
        return self.world_to_base(position), quaternion_from_matrix(rotation)

    def pose_base_to_world(self, position, quaternion):
        rotation = self.rotation @ matrix_from_quaternion(quaternion)
        return self.base_to_world(position), quaternion_from_matrix(rotation)


class FrameTransformer:
    """
    Gestore trasformazioni cinematiche per sensori RGB-D e cinematica robotica.
    Supporta de-proiezione dei pixel (u, v) + Depth metrica in 3D (Camera Frame),
    trasformazione nella base robot, nel mondo e nell'End-Effector Frame.
    """

    def __init__(self, camera_position_xyz=(0.60, -0.42, 1.15),
                 camera_target_xyz=(0.60, -0.42, 0.07),
                 camera_intrinsics=None,
                 camera_up_vector=(-1.0, 0.0, 0.0), robot_base_frame=None, camera_orientation_xyzw=None):
        """``camera_orientation_xyzw``: rotazione del frame ottico (x destra, y basso, z avanti) nel world,
        da calibrazione mano-occhio o TF. Se assente la posa e' ricavata da position/target/up (look-at)."""
        self.cam_pos = np.array(camera_position_xyz, dtype=np.float64)
        self.cam_target = np.array(camera_target_xyz, dtype=np.float64)
        self.base_frame = robot_base_frame or RobotBaseFrame()

        # Intrinsics camera
        intr = camera_intrinsics or {}
        self.fx = float(intr.get("fx_px", 615.0))
        self.fy = float(intr.get("fy_px", 615.0))
        self.cx = float(intr.get("cx_px", 320.0))
        self.cy = float(intr.get("cy_px", 240.0))

        # Posa ottica nel mondo; la posa della base viene applicata separatamente.
        # Supporta qualsiasi posa della telecamera definita in surgical_environment.yaml
        view_dir = self.cam_target - self.cam_pos
        norm_view = np.linalg.norm(view_dir)
        z_opt = view_dir / norm_view if norm_view > 1e-6 else np.array([0.0, 0.0, -1.0])

        up_vec = np.array(camera_up_vector, dtype=np.float64)
        # Rimuovi la componente parallela a z_opt per garantire ortogonalità
        up_ortho = up_vec - np.dot(up_vec, z_opt) * z_opt
        norm_up = np.linalg.norm(up_ortho)
        if norm_up < 1e-6:
            up_ortho = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], z_opt) * z_opt
            norm_up = np.linalg.norm(up_ortho)
        up_unit = up_ortho / norm_up

        # In convenzione OpenCV/ROS Optical Frame:
        # Y_opt punta in BASSO nell'immagine (opposto all'asse up della telecamera)
        y_opt = -up_unit
        # X_opt punta a DESTRA nell'immagine (terna destrorsa: Y_opt x Z_opt)
        x_opt = np.cross(y_opt, z_opt)
        x_opt /= np.linalg.norm(x_opt)
        # Ri-ortogonalizza y_opt per precisione numerica assoluta (Z x X)
        y_opt = np.cross(z_opt, x_opt)

        self.R_world_opt = np.column_stack([x_opt, y_opt, z_opt])
        if camera_orientation_xyzw is not None:
            self.R_world_opt = matrix_from_quaternion(camera_orientation_xyzw)
        self.R_base_opt = self.base_frame.rotation.T @ self.R_world_opt

        # Matrice di trasformazione omogenea 4x4 (Optical -> Robot Base)
        self.T_base_opt = np.eye(4, dtype=np.float64)
        self.T_base_opt[:3, :3] = self.R_base_opt
        self.T_base_opt[:3, 3] = self.base_frame.world_to_base(self.cam_pos)

        self.T_world_opt = np.eye(4, dtype=np.float64)
        self.T_world_opt[:3, :3] = self.R_world_opt
        self.T_world_opt[:3, 3] = self.cam_pos
        self.T_opt_world = np.linalg.inv(self.T_world_opt)

        # Inversa (Robot Base -> Optical)
        self.T_opt_base = np.linalg.inv(self.T_base_opt)

    def deproject_pixel_to_camera_frame(self, u, v, depth_m):
        """
        De-proietta un pixel (u, v) con profondità metrica depth_m nel frame ottico della telecamera.
        Ritorna coordinate [X_c, Y_c, Z_c] in metri.
        """
        depth_m = float(depth_m)
        x_c = (float(u) - self.cx) * depth_m / self.fx
        y_c = (float(v) - self.cy) * depth_m / self.fy
        z_c = depth_m
        return np.array([x_c, y_c, z_c], dtype=np.float64)

    def deproject_mask_to_pointcloud(self, mask_np, depth_img_np, return_pixels=False):
        """
        De-proietta tutti i pixel attivi della maschera usando l'immagine di profondità RGB-D.
        Ritorna:
          - points_camera: (N, 3) punti 3D nel frame ottico
          - points_base: (N, 3) punti 3D nel frame base del robot (lbr_link_0)
          - (solo se return_pixels) pixels: (N, 2) coordinate (v, u) dei pixel validi,
            allineate riga per riga ai punti (per associare colori corretti).
        """
        v_indices, u_indices = np.where(mask_np > 128)
        if len(u_indices) == 0:
            empty = np.empty((0, 3))
            return (empty, empty, np.empty((0, 2), dtype=int)) if return_pixels else (empty, empty)

        depths = depth_img_np[v_indices, u_indices].astype(np.float64)
        valid = (depths > 0.1) & (depths < 3.0) & np.isfinite(depths)

        u_valid = u_indices[valid]
        v_valid = v_indices[valid]
        d_valid = depths[valid]

        x_c = (u_valid - self.cx) * d_valid / self.fx
        y_c = (v_valid - self.cy) * d_valid / self.fy
        z_c = d_valid
        pts_camera = np.stack([x_c, y_c, z_c], axis=1)

        # Trasformazione nel frame base robot
        pts_hom = np.hstack([pts_camera, np.ones((len(pts_camera), 1))])
        pts_base_hom = (self.T_base_opt @ pts_hom.T).T
        pts_base = pts_base_hom[:, :3]

        if return_pixels:
            return pts_camera, pts_base, np.stack([v_valid, u_valid], axis=1)
        return pts_camera, pts_base

    def transform_camera_to_world(self, point_camera):
        """Trasforma un punto dal frame ottico al riferimento fisso world."""
        pt_hom = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
        return (self.T_world_opt @ pt_hom)[:3]

    def transform_world_to_camera(self, point_world):
        """Trasforma un punto dal riferimento fisso world al frame ottico."""
        pt_hom = np.array([point_world[0], point_world[1], point_world[2], 1.0])
        return (self.T_opt_world @ pt_hom)[:3]

    def transform_world_to_ee(self, point_world, ee_position_xyz, ee_orientation_xyzw):
        """
        Trasforma un punto world nel frame EE, la cui posa e' anch'essa espressa in world.
        Ritorna [X_ee, Y_ee, Z_ee].
        """
        R_base_ee = matrix_from_quaternion(ee_orientation_xyzw)
        t_base_ee = np.array(ee_position_xyz, dtype=np.float64)

        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = R_base_ee
        T_base_ee[:3, 3] = t_base_ee

        T_ee_base = np.linalg.inv(T_base_ee)
        pt_hom = np.array([point_world[0], point_world[1], point_world[2], 1.0])
        return (T_ee_base @ pt_hom)[:3]

    def transform_ee_to_world(self, point_ee, ee_position_xyz, ee_orientation_xyzw):
        """Trasforma un punto relativo all'end-effector nel riferimento world."""
        R_base_ee = matrix_from_quaternion(ee_orientation_xyzw)
        t_base_ee = np.array(ee_position_xyz, dtype=np.float64)

        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = R_base_ee
        T_base_ee[:3, 3] = t_base_ee

        pt_hom = np.array([point_ee[0], point_ee[1], point_ee[2], 1.0])
        return (T_base_ee @ pt_hom)[:3]

    def broadcast_static_tf(self, node):
        """
        Pubblica i frame TF2 statici su ROS 2:
          - lbr_link_0 -> camera_link
          - camera_link -> camera_optical_frame
        """
        if not HAS_ROS2_TF:
            node.get_logger().warning("tf2_ros non disponibile, broadcast statico saltato.")
            return

        broadcaster = tf2_ros.StaticTransformBroadcaster(node)
        stamp = node.get_clock().now().to_msg()

        # 1. lbr_link_0 -> camera_link (corpo della telecamera)
        t1 = TransformStamped()
        t1.header.stamp = stamp
        t1.header.frame_id = self.base_frame.frame_id
        t1.child_frame_id = "camera_link"
        t1.transform.translation.x = float(self.T_base_opt[0, 3])
        t1.transform.translation.y = float(self.T_base_opt[1, 3])
        t1.transform.translation.z = float(self.T_base_opt[2, 3])
        # Orientamento nominale camera_link
        q1 = quaternion_from_matrix(self.R_base_opt)
        t1.transform.rotation.x = q1[0]
        t1.transform.rotation.y = q1[1]
        t1.transform.rotation.z = q1[2]
        t1.transform.rotation.w = q1[3]

        # 2. camera_link -> camera_optical_frame
        t2 = TransformStamped()
        t2.header.stamp = stamp
        t2.header.frame_id = "camera_link"
        t2.child_frame_id = "camera_optical_frame"
        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0
        t2.transform.rotation.x = 0.0
        t2.transform.rotation.y = 0.0
        t2.transform.rotation.z = 0.0
        t2.transform.rotation.w = 1.0

        broadcaster.sendTransform([t1, t2])
        node.get_logger().info("TF statici pubblicati: lbr_link_0 -> camera_link -> camera_optical_frame")
