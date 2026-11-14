"""Fresh ROS/GraspGen-planner/physics run, replaying recorded neural proposals.

Manual integration check, not a full-model benchmark. Refuses a different scene.
No production launcher fallback is installed by this module.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from pipeline.common.ply import read_ply
from pipeline.common.report import StageReport
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.graspgen.backend import rank_candidates
from pipeline.ros2.cli import parser, run
from pipeline.ros2.coordinator import Coordinator
from pipeline.sim import deproject
from pipeline.system.registry import load_profile
from pipeline.config_loader import load_environment_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--robot-model", choices=["recorded", "iiwa7_r800"], default="recorded",
                   help="Keep the recording's arm, or explicitly replan its proposals for iiwa 7")
    args = p.parse_args()
    source = Path(args.source_run).resolve()
    old_capture = json.loads((source / "sim_capture.json").read_text())["data"]
    recorded_model = old_capture.get("robot_description", {}).get("model", "pybullet_iiwa14")
    target_model = recorded_model if args.robot_model == "recorded" else args.robot_model
    changed_robot = target_model != recorded_model

    def recorded_perception(node):
        capture = json.loads((node.out / "sim_capture.json").read_text())["data"]
        for key in ("position_xyz_m", "orientation_xyzw"):
            if not np.allclose(capture["scene_object_pose_exact"][key], old_capture["scene_object_pose_exact"][key],
                               atol=1e-7, rtol=0):
                raise ValueError("Recorded proposals cannot be replayed on a different object pose")
        current_rgb = np.asarray(Image.open(node.out / "sim_capture_rgb.png"), dtype=float)
        recorded_rgb = np.asarray(Image.open(source / "sim_capture_rgb.png"), dtype=float)
        current_depth, recorded_depth = np.load(node.out / "sim_capture_depth.npy"), np.load(source / "sim_capture_depth.npy")
        # A different arm changes background pixels. Reusing proposals for the
        # same object is explicit and allowed only if its observed pixels/depth
        # remain unchanged. A fresh path is always planned for the new robot.
        region = np.asarray(Image.open(source / "sim_segmentation_mask.png")) > 0 if changed_robot else np.ones(current_depth.shape, bool)
        if (not np.allclose(current_depth[region], recorded_depth[region], atol=1e-5, rtol=0)
                or np.abs(current_rgb[region]-recorded_rgb[region]).mean() > 1):
            raise ValueError("Scene capture differs from the recording")
        mask = Image.open(source / "sim_segmentation_mask.png")
        mask.save(node.out / "sim_segmentation_mask.png")
        segmentation = json.loads((source / "sim_segmentation.json").read_text())["data"]
        segmentation.update(segmentation_backend="recorded_sam3_mask", model_id="recorded; not loaded")
        StageReport("sim_segmentation", data=segmentation,
                    artifacts={"mask_png": str(node.out / "sim_segmentation_mask.png")}).save(node.out / "sim_segmentation.json")
        dep = deproject.run(config_path=node.spec["settings"]["environment"])
        cloud = node.out / "sim_deprojection_cloud.ply"
        points, _ = read_ply(cloud)
        data = copy.deepcopy(json.loads((source / "grasp_graspgen.json").read_text())["data"])
        old_center = np.asarray(data["input_pointcloud"]["center_base_xyz_m"])
        center = points.mean(0)
        if np.linalg.norm(center-old_center) > .0001:
            raise ValueError("Reconstructed cloud differs from the recorded model input")
        by_index = {c["server_index"]: c for c in data["candidates"]}
        candidates = rank_candidates([by_index[i]["T_centered_gripper"] for i in range(len(by_index))],
            [by_index[i]["score"] for i in range(len(by_index))], center,
            RobotBaseFrame.from_dict(dep.data["robot_base_pose_world"]),
            np.asarray(data["configuration"]["robot"]["T_gripper_tcp"]))
        data.update(candidates=candidates, selected=candidates[0], inference_mode="recorded_proposals",
                    recorded_source_run=source.name, segmentation_backend="recorded_sam3_mask",
                    robot_description=dep.data["robot_description"])
        data["input_pointcloud"].update(source_ply=str(cloud), sha256=hashlib.sha256(cloud.read_bytes()).hexdigest(),
                                        center_base_xyz_m=center.tolist())
        report = StageReport("grasp", data=data)
        report.save(node.out / "grasp_graspgen.json")
        report.save(node.out / "grasp.json")
        return data

    options = parser().parse_args(["--profile", "graspgen_only", "--headless", "--out-dir", args.out_dir])
    settings = load_profile("graspgen_only")
    settings["validation_mode"] = "recorded_proposals_replanned_for_iiwa7" if changed_robot else "recorded_ai_outputs"
    with tempfile.TemporaryDirectory(prefix="ros2-recorded-model-") as tmp:
        config = load_environment_config(settings["environment"])
        config["robot"] = {"model": target_model}
        environment = Path(tmp) / "environment.yaml"
        environment.write_text(yaml.safe_dump(config))
        settings["environment"] = str(environment)
        with patch.object(Coordinator, "perceive", recorded_perception):
            return run(settings, options, [dict(name="recorded_output_test", ok=True,
                                               detail="Real ROS, planner and physics; recorded neural proposals, no inference")])


if __name__ == "__main__":
    raise SystemExit(main())
