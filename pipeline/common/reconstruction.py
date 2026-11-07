"""Mask + depth -> 3D instrument point cloud and centroids in camera/base/end-effector frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from pipeline.common.report import StageError
from pipeline.common.scene import Config, Intrinsics, ee_reference, round_list
from pipeline.frame_transformer import FrameTransformer, RobotBaseFrame


@dataclass(frozen=True)
class CameraModel:
    """Camera pose + pinhole intrinsics, serialized in capture reports as ``data.camera``."""
    position: List[float]
    target: List[float]
    intrinsics: Intrinsics

    def to_dict(self) -> Dict[str, Any]:
        i = self.intrinsics
        return {"position_xyz_m": round_list(self.position), "target_position_xyz_m": round_list(self.target),
                "frame_id": "world", "resolution_wh": [i.width, i.height],
                "fx_fy": [i.fx, i.fy], "cx_cy": [i.cx, i.cy]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CameraModel":
        w, h = d["resolution_wh"]
        fx, fy = d["fx_fy"]
        cx, cy = d.get("cx_cy", [w / 2.0, h / 2.0])
        return cls(list(d["position_xyz_m"]), list(d["target_position_xyz_m"]), Intrinsics(w, h, fx, fy, cx, cy))

    def transformer(self, base_frame: RobotBaseFrame | None = None) -> FrameTransformer:
        i = self.intrinsics
        return FrameTransformer(camera_position_xyz=self.position, camera_target_xyz=self.target,
                                camera_intrinsics={"fx_px": i.fx, "fy_px": i.fy, "cx_px": i.cx, "cy_px": i.cy},
                                robot_base_frame=base_frame)


@dataclass
class Reconstruction:
    transformer: FrameTransformer
    points_base: np.ndarray
    colors: np.ndarray
    centroid_camera: np.ndarray
    centroid_base: np.ndarray

    @property
    def centroid_world(self) -> np.ndarray:
        return self.transformer.base_frame.base_to_world(self.centroid_base)

    def with_centroid_base(self, centroid_base: np.ndarray) -> "Reconstruction":
        """Same cloud, but with the centroid overridden (e.g. by a certified reference pose)."""
        world = self.transformer.base_frame.base_to_world(centroid_base)
        return Reconstruction(self.transformer, self.points_base, self.colors,
                              self.transformer.transform_world_to_camera(world),
                              np.asarray(centroid_base))

    def frames(self, config: Config) -> Dict[str, List[float]]:
        ee_pos, ee_quat = ee_reference(config)
        centroid_ee = self.transformer.transform_world_to_ee(self.centroid_world, ee_pos, ee_quat)
        return {"camera_optical_frame": round_list(self.centroid_camera),
                "robot_base_frame": round_list(self.centroid_base),
                "world_frame": round_list(self.centroid_world),
                "end_effector_frame": round_list(centroid_ee)}


def reconstruct(mask: np.ndarray, depth: np.ndarray, rgb: np.ndarray, camera: CameraModel,
                config: Config) -> Reconstruction:
    if not (mask > 128).any():
        raise StageError("reconstruction", "no active pixel in the instrument mask")
    transformer = camera.transformer(RobotBaseFrame.from_config(config))
    pts_cam, pts_base, pixels = transformer.deproject_mask_to_pointcloud(mask, depth, return_pixels=True)
    if len(pts_base) == 0:
        raise StageError("reconstruction", "mask pixels have no valid depth (outside 0.1-3.0 m)")
    return Reconstruction(transformer, pts_base, rgb[pixels[:, 0], pixels[:, 1]],
                          pts_cam.mean(axis=0), pts_base.mean(axis=0))
