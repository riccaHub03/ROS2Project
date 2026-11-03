#!/usr/bin/env python3
"""
Modulo: Frame Transformer e Gestione Trasformazioni di Coordinate ROS 2.
Gestisce il passaggio matematico e TF2 tra:
  1. Camera Optical Frame (Z avanti lungo la linea di vista, X destra, Y basso)
  2. Camera Link (montaggio fisico della telecamera)
  3. Robot Base / Global Frame (lbr_link_0 / world)
  4. End-Effector / TCP Frame (lbr_iiwa_link_7 / workbench_tcp)
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


class FrameTransformer:
    """
    Gestore trasformazioni cinematiche per sensori RGB-D e cinematica robotica.
    Supporta de-proiezione dei pixel (u, v) + Depth metrica in 3D (Camera Frame),
    trasformazione in World Frame (lbr_link_0) e in End-Effector Frame (workbench_tcp).
    """

    def __init__(self, camera_position_xyz=[0.60, -0.42, 1.15],
                 camera_target_xyz=[0.60, -0.42, 0.07],
                 camera_intrinsics=None,
                 camera_up_vector=[-1.0, 0.0, 0.0]):
        self.cam_pos = np.array(camera_position_xyz, dtype=np.float64)
        self.cam_target = np.array(camera_target_xyz, dtype=np.float64)

        # Intrinsics camera
        intr = camera_intrinsics or {}
        self.fx = float(intr.get("fx_px", 615.0))
        self.fy = float(intr.get("fy_px", 615.0))
        self.cx = float(intr.get("cx_px", 320.0))
        self.cy = float(intr.get("cy_px", 240.0))

        # Calcolo analitico e dinamico della Matrice di Rotazione: Camera Optical Frame -> Robot Base Frame
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

        self.R_base_opt = np.column_stack([x_opt, y_opt, z_opt])

        # Matrice di trasformazione omogenea 4x4 (Optical -> Robot Base)
        self.T_base_opt = np.eye(4, dtype=np.float64)
        self.T_base_opt[:3, :3] = self.R_base_opt
        self.T_base_opt[:3, 3] = self.cam_pos

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

    def deproject_mask_to_pointcloud(self, mask_np, depth_img_np):
        """
        De-proietta tutti i pixel attivi della maschera usando l'immagine di profondità RGB-D.
        Ritorna:
          - points_camera: (N, 3) punti 3D nel frame ottico
          - points_world: (N, 3) punti 3D nel frame base del robot (lbr_link_0)
        """
        v_indices, u_indices = np.where(mask_np > 128)
        if len(u_indices) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

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
        pts_world_hom = (self.T_base_opt @ pts_hom.T).T
        pts_world = pts_world_hom[:, :3]

        return pts_camera, pts_world

    def transform_camera_to_world(self, point_camera):
        """Trasforma un punto [X_c, Y_c, Z_c] dal frame ottico al frame base (lbr_link_0)."""
        pt_hom = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
        return (self.T_base_opt @ pt_hom)[:3]

    def transform_world_to_camera(self, point_world):
        """Trasforma un punto [X_w, Y_w, Z_w] dal frame base al frame ottico."""
        pt_hom = np.array([point_world[0], point_world[1], point_world[2], 1.0])
        return (self.T_opt_base @ pt_hom)[:3]

    def transform_world_to_ee(self, point_world, ee_position_xyz, ee_orientation_xyzw):
        """
        Trasforma un punto dal frame base al frame dell'end-effector (workbench_tcp / flange).
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
        """Trasforma un punto relativo all'end-effector nel frame globale del robot."""
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
        t1.header.frame_id = "lbr_link_0"
        t1.child_frame_id = "camera_link"
        t1.transform.translation.x = float(self.cam_pos[0])
        t1.transform.translation.y = float(self.cam_pos[1])
        t1.transform.translation.z = float(self.cam_pos[2])
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


def export_pointcloud_ply(filepath, points_xyz, colors_rgb=None):
    """Esporta una nuvola di punti in formato standard ASCII Stanford PLY."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    n_pts = len(points_xyz)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors_rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n_pts):
            pt = points_xyz[i]
            if colors_rgb is not None:
                c = colors_rgb[i]
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

