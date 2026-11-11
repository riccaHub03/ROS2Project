"""Acquisizione dal vivo da una camera RGB-D ROS 2 e dalla TF, senza ``cv_bridge``.

Serve allo stadio ``real_capture`` (``--source ros2``) e al preflight. Legge:

  * l'immagine a colori e la profondita' **allineata al colore** (``16UC1`` in mm o ``32FC1`` in m);
  * ``CameraInfo`` (intrinseche calibrate);
  * dalla TF, la posa del frame ottico della camera nel ``world`` (calibrazione mano-occhio pubblicata come
    trasformazione) e, se richiesta, la posa dell'end-effector nel ``world``.

La profondita' e' la mediana di ``depth_frames`` fotogrammi (riduce il rumore del sensore). Nessun comando
viene inviato al robot: sono solo sottoscrizioni.

Le funzioni di conversione lavorano su qualunque oggetto con gli stessi attributi dei messaggi ROS: si provano
senza ROS. ``rclpy`` viene importato solo quando serve.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.common.report import StageError
from pipeline.common.scene import Intrinsics

STAGE = "real_capture"
log = logging.getLogger("real_capture.ros2")

DEFAULT_OPTIONS: Dict[str, Any] = {
    "color_topic": "/camera/color/image_raw",
    "depth_topic": "/camera/aligned_depth_to_color/image_raw",
    "camera_info_topic": "/camera/color/camera_info",
    "world_frame": "world",          # riferimento in cui la pipeline esprime camera, base e vassoi
    "camera_frame": None,            # None: il frame_id di CameraInfo (deve essere il frame OTTICO)
    "ee_frame": None,                # es. lbr_link_ee: se impostato, la posa EE viene letta dalla TF
    "timeout_s": 10.0,               # attesa dei messaggi
    "depth_frames": 5,               # fotogrammi di profondita' per la mediana
    "max_sync_s": 0.2,               # scarto massimo colore/profondita'
    "tf_timeout_s": 3.0,
}

_COLOR_ENCODINGS = {"rgb8": (3, (0, 1, 2)), "bgr8": (3, (2, 1, 0)), "rgba8": (4, (0, 1, 2)), "bgra8": (4, (2, 1, 0))}


# ---------------------------------------------------------------------------------------------------
# Conversioni (pure)
# ---------------------------------------------------------------------------------------------------
def image_to_rgb(msg) -> np.ndarray:
    """``sensor_msgs/Image`` a colori -> array HxWx3 uint8 in ordine RGB."""
    encoding = str(msg.encoding).lower()
    if encoding not in _COLOR_ENCODINGS:
        raise StageError(STAGE, f"unsupported colour encoding '{msg.encoding}' (supported: "
                                f"{', '.join(_COLOR_ENCODINGS)}); publish rgb8/bgr8 from the camera driver")
    channels, order = _COLOR_ENCODINGS[encoding]
    h, w = int(msg.height), int(msg.width)
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, int(msg.step))
    return np.ascontiguousarray(rows[:, : w * channels].reshape(h, w, channels)[..., list(order)])


def depth_to_meters(msg) -> np.ndarray:
    """``sensor_msgs/Image`` di profondita' -> HxW float32 in metri; 0 dove non valida."""
    encoding = str(msg.encoding).lower()
    if encoding == "16uc1":
        dtype, scale = (">u2" if msg.is_bigendian else "<u2"), 1.0 / 1000.0      # millimetri
    elif encoding == "32fc1":
        dtype, scale = (">f4" if msg.is_bigendian else "<f4"), 1.0               # metri
    else:
        raise StageError(STAGE, f"unsupported depth encoding '{msg.encoding}' (supported: 16UC1 in mm, 32FC1 in m)")
    h, w = int(msg.height), int(msg.width)
    itemsize = np.dtype(dtype).itemsize
    rows = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(h, int(msg.step) // itemsize)
    depth = rows[:, :w].astype(np.float32) * scale
    depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
    return depth


def camera_info_to_intrinsics(msg) -> Tuple[Intrinsics, bool]:
    """``CameraInfo`` -> (intrinseche, distorsione non nulla). Rifiuta una camera non calibrata (K nulla)."""
    k = list(msg.k)
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    if fx <= 0 or fy <= 0:
        raise StageError(STAGE, "CameraInfo has a zero focal length: the camera is not calibrated. "
                                "Calibrate it (camera_calibration) or use the factory intrinsics of the driver")
    distortion = any(abs(float(c)) > 1e-9 for c in getattr(msg, "d", []))
    return Intrinsics(int(msg.width), int(msg.height), fx, fy, cx, cy), distortion


def transform_to_pose(transform_stamped) -> Tuple[List[float], List[float]]:
    """``TransformStamped`` -> (posizione xyz [m], quaternione xyzw) del frame figlio nel frame padre."""
    t, r = transform_stamped.transform.translation, transform_stamped.transform.rotation
    return [float(t.x), float(t.y), float(t.z)], [float(r.x), float(r.y), float(r.z), float(r.w)]


def stamp_seconds(msg) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def median_depth(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Mediana per pixel sui soli valori validi; un pixel e' valido se lo e' in almeno meta' dei fotogrammi."""
    stack = np.stack(frames).astype(np.float32)
    valid = stack > 0
    stack[~valid] = np.nan
    with np.errstate(all="ignore"):
        result = np.nanmedian(stack, axis=0)
    result[valid.sum(axis=0) * 2 < len(frames)] = np.nan
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


def resolve_options(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Default < ``camera_overhead.ros2`` del YAML < opzioni da riga di comando (le non-None)."""
    options = dict(DEFAULT_OPTIONS)
    options.update((config.get("camera_overhead", {}) or {}).get("ros2", {}) or {})
    options.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return options


# ---------------------------------------------------------------------------------------------------
# Acquisizione
# ---------------------------------------------------------------------------------------------------
@dataclass
class LiveFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: Intrinsics
    distortion_nonzero: bool
    camera_frame: str
    world_frame: str
    color_stamp: float
    depth_stamp: float
    depth_valid_fraction: float
    depth_frames_used: int
    camera_pose_world: Optional[Tuple[List[float], List[float]]] = None   # frame ottico nel world (TF)
    ee_pose_world: Optional[Tuple[List[float], List[float]]] = None       # end-effector nel world (TF)
    options: Dict[str, Any] = field(default_factory=dict)

    def acquisition_info(self) -> Dict[str, Any]:
        o = self.options
        return {"color_topic": o["color_topic"], "depth_topic": o["depth_topic"],
                "camera_info_topic": o["camera_info_topic"], "camera_frame": self.camera_frame,
                "world_frame": self.world_frame, "ee_frame": o.get("ee_frame"),
                "color_stamp_s": round(self.color_stamp, 4), "depth_stamp_s": round(self.depth_stamp, 4),
                "sync_s": round(abs(self.color_stamp - self.depth_stamp), 4),
                "depth_frames_median": self.depth_frames_used,
                "depth_valid_fraction": round(self.depth_valid_fraction, 4),
                "distortion_nonzero": self.distortion_nonzero,
                "camera_pose_from_tf": self.camera_pose_world is not None,
                "ee_pose_from_tf": self.ee_pose_world is not None}


def _import_ros():
    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener
    except ImportError as exc:
        raise StageError(STAGE, f"ROS 2 is not available ({exc}): run inside the container, which sources "
                                "/opt/ros/humble/setup.bash, or use --source file") from exc
    return rclpy, Duration, qos_profile_sensor_data, Time, CameraInfo, Image, Buffer, TransformListener


def acquire(options: Dict[str, Any]) -> LiveFrame:
    """Attende colore, profondita', CameraInfo e TF; ritorna un fotogramma coerente o fallisce con un motivo."""
    rclpy, Duration, sensor_qos, Time, CameraInfo, Image, Buffer, TransformListener = _import_ros()
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = rclpy.create_node("pipeline_real_capture")
    try:
        colors: List[Any] = []
        depths: List[Any] = []
        infos: List[Any] = []
        want = max(1, int(options["depth_frames"]))
        def on_color(msg):
            colors.append(msg)
            del colors[:-6]                       # bastano gli ultimi, serve quello piu' vicino alla profondita'

        def on_info(msg):
            if not infos:
                infos.append(msg)

        node.create_subscription(Image, options["color_topic"], on_color, sensor_qos)
        def on_depth(msg):
            depths.append(msg)
            del depths[:-want]                    # ultimi `want` fotogrammi

        node.create_subscription(Image, options["depth_topic"], on_depth, sensor_qos)
        node.create_subscription(CameraInfo, options["camera_info_topic"], on_info, sensor_qos)
        buffer = Buffer()
        TransformListener(buffer, node)

        def synchronised() -> bool:
            if not (colors and len(depths) >= want and infos):
                return False
            last = stamp_seconds(depths[-1])
            return min(abs(stamp_seconds(c) - last) for c in colors) <= float(options["max_sync_s"])

        deadline = time.monotonic() + float(options["timeout_s"])
        while time.monotonic() < deadline and not synchronised():
            rclpy.spin_once(node, timeout_sec=0.05)
        missing = [name for name, got in (("colour image", colors), ("depth image", len(depths) >= want and depths),
                                          ("CameraInfo", infos)) if not got]
        if missing:
            topics = sorted(name for name, _ in node.get_topic_names_and_types())
            raise StageError(STAGE, f"no {', '.join(missing)} within {options['timeout_s']:.0f} s. Expected topics: "
                                    f"{options['color_topic']}, {options['depth_topic']}, {options['camera_info_topic']}. "
                                    f"Camera topics visible: {[t for t in topics if 'camera' in t or 'image' in t][:12]}. "
                                    "Check the driver and the topic names (camera_overhead.ros2 in the YAML)")

        depth_msgs = list(depths)
        color_msg = min(colors, key=lambda m: abs(stamp_seconds(m) - stamp_seconds(depth_msgs[-1])))
        sync = abs(stamp_seconds(color_msg) - stamp_seconds(depth_msgs[-1]))
        if sync > float(options["max_sync_s"]):
            raise StageError(STAGE, f"colour and depth are {sync:.2f} s apart (max {options['max_sync_s']} s): "
                                    "enable the driver's synchronised/aligned stream")
        rgb = image_to_rgb(color_msg)
        depth = median_depth([depth_to_meters(m) for m in depth_msgs])
        if depth.shape != rgb.shape[:2]:
            raise StageError(STAGE, f"depth {depth.shape[::-1]} and colour {rgb.shape[1::-1]} differ: use the depth "
                                    "ALIGNED to the colour frame (e.g. aligned_depth_to_color)")
        intrinsics, distortion = camera_info_to_intrinsics(infos[0])
        if (intrinsics.width, intrinsics.height) != (rgb.shape[1], rgb.shape[0]):
            raise StageError(STAGE, f"CameraInfo is {intrinsics.width}x{intrinsics.height} but the image is "
                                    f"{rgb.shape[1]}x{rgb.shape[0]}: wrong camera_info topic or resolution")
        camera_frame = options.get("camera_frame") or infos[0].header.frame_id
        if not camera_frame:
            raise StageError(STAGE, "CameraInfo has no frame_id: set camera_overhead.ros2.camera_frame")

        def lookup(child: str) -> Optional[Tuple[List[float], List[float]]]:
            end = time.monotonic() + float(options["tf_timeout_s"])
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
                if buffer.can_transform(options["world_frame"], child, Time()):
                    return transform_to_pose(buffer.lookup_transform(options["world_frame"], child, Time(),
                                                                     Duration(seconds=1.0)))
            return None

        camera_pose = lookup(camera_frame)
        ee_pose = lookup(options["ee_frame"]) if options.get("ee_frame") else None
        return LiveFrame(rgb, depth, intrinsics, distortion, camera_frame, options["world_frame"],
                         stamp_seconds(color_msg), stamp_seconds(depth_msgs[-1]),
                         float((depth > 0).mean()), len(depth_msgs), camera_pose, ee_pose, dict(options))
    finally:
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()
