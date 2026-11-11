"""Wrap authors' JSON/NPY/ASCII PLY in a portable deprojection report for our backend."""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.common.ply import read_ply, write_ply
from pipeline.common.report import StageReport
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.graspgen.config import rigid_transform


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--units", choices=["m"], required=True, help="Explicitly assert metric input")
    parser.add_argument("--frame", choices=["base", "world", "camera"], required=True)
    parser.add_argument("--T-base-from-cloud", type=Path, help="4x4 JSON matrix, required with --frame camera")
    parser.add_argument("--base-pose", type=Path, help="JSON RobotBaseFrame.to_dict(), default identity")
    args = parser.parse_args(argv)
    if args.input.suffix == ".json":
        points = np.asarray(json.loads(args.input.read_text())["pc"], dtype=float)
    elif args.input.suffix == ".npy":
        points = np.load(args.input, allow_pickle=False)
    elif args.input.suffix == ".ply":
        points, _ = read_ply(args.input)
    else:
        parser.error("Supported: authors' JSON with pc, NPY XYZ, ASCII PLY")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not len(points):
        parser.error("Input must be nonempty finite (N,3) metric points")
    base = RobotBaseFrame.from_dict(json.loads(args.base_pose.read_text()) if args.base_pose else {})
    data = {"num_points": len(points), "pointcloud_units": "m",
            "coordinate_frame_id": {"base": base.frame_id, "world": "world", "camera": "camera_optical"}[args.frame],
            "robot_base_pose_world": base.to_dict(), "geometry_verified_for_motion": False,
            "source_cloud": str(args.input.resolve()), "source_cloud_frame_asserted_by_user": True}
    if args.frame == "camera":
        if not args.T_base_from_cloud:
            parser.error("--frame camera requires --T-base-from-cloud (no guessed extrinsics)")
        data["T_base_from_pointcloud"] = rigid_transform(json.loads(args.T_base_from_cloud.read_text())).tolist()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cloud = write_ply(out / "imported_cloud.ply", points)
    report = StageReport("imported_deprojection", data=data, artifacts={"pointcloud_ply": str(cloud)})
    print(report.save(out / "imported_deprojection.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
