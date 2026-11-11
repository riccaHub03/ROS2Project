#!/usr/bin/env python3
"""Camera RGB-D FINTA per provare la pipeline reale senza hardware (e per i test).

Pubblica colore, profondita' allineata e CameraInfo (come un driver reale) e le TF world->camera (e world->ee).
La scena e' un vassoio piano con un "strumento" rettangolare un po' piu' alto. Lancio, in un terminale:

    python3 scripts/fake_ros2_camera.py
poi, in un altro:
    python3 -m pipeline.real.preflight --source ros2 --ee-frame fake_ee
    python3 -m pipeline.real.capture --source ros2 --ee-frame fake_ee

Questo NON valida la calibrazione ne' la segmentazione: prova solo l'acquisizione e la catena di frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.frame_transformer import FrameTransformer, quaternion_from_matrix  # noqa: E402

# Camera sopra il vassoio che guarda in basso, con l'orientamento del simulatore (look-at verso il basso, up = -X).
NADIR_OPTICAL_QUAT = tuple(float(v) for v in quaternion_from_matrix(
    FrameTransformer(camera_position_xyz=(0, 0, 1), camera_target_xyz=(0, 0, 0)).R_world_opt))


def build_scene(width: int, height: int, camera_height: float, tray_top: float):
    plane = camera_height - tray_top
    depth = np.full((height, width), plane, dtype=np.float32)
    rgb = np.full((height, width, 3), (180, 182, 186), dtype=np.uint8)
    y0, y1, x0, x1 = height // 2 - 20, height // 2 + 20, width // 2 - 110, width // 2 + 110
    depth[y0:y1, x0:x1] = plane - 0.012                       # lo strumento sporge di 12 mm
    rgb[y0:y1, x0:x1] = (70, 76, 84)
    return rgb, depth


class FakeCamera(Node):
    def __init__(self, a):
        super().__init__("fake_rgbd_camera")
        self.a = a
        self.rgb, self.depth = build_scene(a.width, a.height, a.camera_z, a.tray_top)
        self.color_pub = self.create_publisher(Image, a.color_topic, qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, a.depth_topic, qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, a.camera_info_topic, qos_profile_sensor_data)
        self.create_timer(1.0 / a.rate, self.tick)
        if not a.no_tf:
            self.tf = StaticTransformBroadcaster(self)
            self.tf.sendTransform([self.transform(a.world_frame, a.camera_frame, (a.camera_x, a.camera_y, a.camera_z),
                                                  NADIR_OPTICAL_QUAT),
                                   self.transform(a.world_frame, a.ee_frame, (0.45, 0.0, 0.45), (0.0, 1.0, 0.0, 0.0))])

    def transform(self, parent, child, xyz, quat):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w) = quat
        return t

    def tick(self):
        a = self.a
        stamp = self.get_clock().now().to_msg()
        color = Image(height=a.height, width=a.width, encoding="bgr8" if a.bgr else "rgb8",
                      step=a.width * 3, data=(self.rgb[..., ::-1] if a.bgr else self.rgb).tobytes())
        if a.depth_encoding == "16UC1":
            depth = Image(height=a.height, width=a.width, encoding="16UC1", step=a.width * 2,
                          data=(self.depth * 1000).astype("<u2").tobytes())
        else:
            depth = Image(height=a.height, width=a.width, encoding="32FC1", step=a.width * 4,
                          data=self.depth.astype("<f4").tobytes())
        info = CameraInfo(height=a.height, width=a.width, k=[a.fx, 0.0, a.width / 2, 0.0, a.fx, a.height / 2, 0.0, 0.0, 1.0],
                          d=[0.0] * 5)
        for msg in (color, depth, info):
            msg.header.stamp, msg.header.frame_id = stamp, a.camera_frame
        self.color_pub.publish(color)
        self.depth_pub.publish(depth)
        self.info_pub.publish(info)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fx", type=float, default=615.0)
    p.add_argument("--rate", type=float, default=10.0)
    p.add_argument("--camera-x", type=float, default=0.60)
    p.add_argument("--camera-y", type=float, default=-0.42)
    p.add_argument("--camera-z", type=float, default=1.15)
    p.add_argument("--tray-top", type=float, default=0.09, help="quota della superficie del vassoio nel world")
    p.add_argument("--depth-encoding", choices=("16UC1", "32FC1"), default="16UC1")
    p.add_argument("--bgr", action="store_true", help="colore bgr8 invece di rgb8")
    p.add_argument("--no-tf", action="store_true", help="non pubblicare la TF (prova del caso di errore)")
    p.add_argument("--world-frame", default="world")
    p.add_argument("--camera-frame", default="fake_camera_color_optical_frame")
    p.add_argument("--ee-frame", default="fake_ee")
    p.add_argument("--color-topic", default="/camera/color/image_raw")
    p.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    args = p.parse_args()
    rclpy.init()
    node = FakeCamera(args)
    print(f"Camera finta attiva ({args.width}x{args.height}, profondita' {args.depth_encoding}). Ctrl+C per fermare.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
