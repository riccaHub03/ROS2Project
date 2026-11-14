"""Point cloud -> official diffusion/discriminator server -> ranked metric poses.

T_A_B maps column coordinates in B into A. The server returns the grasp BASE,
not the finger-tip TCP. Its internal mean subtraction is already undone by
GraspGenSampler.sample; only OUR outer centering is undone here.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from pipeline.common import paths
from pipeline.common.ply import read_ply, write_ply
from pipeline.common.report import StageError, StageReport, stage_run, write_json
from pipeline.frame_transformer import RobotBaseFrame, quaternion_from_matrix
from pipeline.graspgen import client
from pipeline.graspgen.config import load_config, rigid_transform

log = logging.getLogger("grasp.graspgen")


def base_transform(base):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = base.rotation, base.position
    return t


def points_to_base(points, data, base):
    frame = data.get("coordinate_frame_id")
    if frame == base.frame_id:
        t = np.eye(4)
    elif frame == "world":
        t = np.linalg.inv(base_transform(base))
    elif frame and "T_base_from_pointcloud" in data:
        t = rigid_transform(data["T_base_from_pointcloud"], "T_base_from_pointcloud")
    else:
        raise ValueError(f"Unknown pointcloud frame {frame!r}; need base/world or explicit T_base_from_pointcloud")
    return points @ t[:3, :3].T + t[:3, 3], t


def validate_metadata(metadata, cfg):
    if not isinstance(metadata, dict):
        raise ValueError("GraspGen metadata must be a dictionary")
    if metadata.get("gripper_name") != cfg["model_gripper"]:
        raise ValueError(f"Checkpoint gripper mismatch: server={metadata.get('gripper_name')!r}, "
                         f"configured={cfg['model_gripper']!r}")
    if metadata.get("model_name") != "diffusion-discriminator":
        raise ValueError("Server must use the official diffusion-discriminator model")
    if Path(metadata.get("gripper_config", "")).name != cfg["checkpoint_config"]:
        raise ValueError("Server checkpoint configuration does not match checkpoint_config")
    provenance = metadata.get("ros2project", {})
    if provenance and (provenance.get("revision") != cfg["expected_revision"]
                       or provenance.get("planner") != "diffusion"):
        raise ValueError("Server revision/planner differs from the configured official pin")
    if cfg["server"]["require_provenance"]:
        if not provenance or not all(provenance.get(k) for k in
                                     ("config_sha256", "generator_sha256", "discriminator_sha256")):
            raise ValueError("Server provenance missing; start python -m pipeline.graspgen.server on the GPU")


def rank_candidates(poses, scores, center, base, t_gripper_tcp):
    poses, scores = np.asarray(poses, dtype=float), np.asarray(scores, dtype=float)
    if poses.size == 0 and scores.size == 0:
        raise ValueError("GraspGen returned no grasp candidates; inspect cloud / thresholds / GPU log")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or scores.shape != (len(poses),):
        raise ValueError(f"Malformed GraspGen response: poses={poses.shape}, scores={scores.shape}")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Discriminator scores must be finite and in [0,1]")
    world_base = base_transform(base)
    candidates = []
    # Sort globally: upstream sorts per retry and then concatenates, not globally.
    for i in np.argsort(-scores, kind="stable"):
        centered = rigid_transform(poses[i], f"candidate {i}").copy()
        base_gripper = centered.copy()
        base_gripper[:3, 3] += center
        base_tcp = base_gripper @ t_gripper_tcp
        candidates.append({
            "server_index": int(i), "score": float(scores[i]),
            "T_centered_gripper": centered.tolist(),
            "T_base_gripper": base_gripper.tolist(),
            "T_world_gripper": (world_base @ base_gripper).tolist(),
            "T_base_tcp": base_tcp.tolist(), "T_world_tcp": (world_base @ base_tcp).tolist(),
            "tcp_quaternion_xyzw_base": quaternion_from_matrix(base_tcp[:3, :3]),
        })
    return candidates


def require_motion_allowed(data, *, gripper=None):
    """Every consumer must gate motion independently of visualization success."""
    if data.get("grasp_backend") != "graspgen":
        return
    motion = data.get("motion", {})
    if not motion.get("allowed", False):
        raise ValueError("GraspGen motion blocked: " + "; ".join(motion.get("blocked_reasons", ["unverified grasp"])))
    if gripper is not None and data["configuration"]["robot"]["gripper"] != gripper:
        raise ValueError(f"GraspGen motion blocked: executor gripper {gripper} differs from configured hardware")


def run(*, deprojection_path=None, plan_path=None, out_dir=None, report_path=None,
        config_path=None, host=None, port=None, timeout_ms=None):
    out = paths.output_dir(out_dir).resolve()
    dep_file = paths.resolve(deprojection_path, paths.SIM_DEPROJECTION, out)
    plan_file = paths.resolve(plan_path, paths.PLAN, out)
    with stage_run("grasp", paths.output_report(report_path, paths.GRASP, out),
                   inputs={"deprojection": dep_file, "plan": plan_file, "graspgen_config": config_path},
                   logger=log) as report:
        report.data = {"grasp_backend": "graspgen", "readiness": "FAILED_OR_PENDING",
                       "physical_grasp_validated": False,
                       "motion": {"allowed": False, "blocked_reasons": ["inference incomplete"]}}
        cfg = load_config(config_path, host=host, port=port, timeout_ms=timeout_ms)
        report.data["configuration"] = cfg
        from pipeline.graspgen.prepare import check_source
        report.data["local_client_revision"] = check_source(cfg["submodule"])
        dep = StageReport.load(dep_file)
        if not dep.stage.endswith("_deprojection"):
            raise StageError("grasp", f"{dep_file} is not a deprojection report")
        ply = Path(dep.artifacts["pointcloud_ply"])
        if not ply.is_absolute():
            ply = dep_file.resolve().parent / ply
        if not ply.is_file():
            raise ValueError(f"Cloud missing: {ply}; use the bundle tool to transfer/rebase saved reports")
        points, _ = read_ply(ply)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Point cloud must contain finite XYZ coordinates")
        if len(points) < cfg["input"]["min_points"]:
            raise ValueError(f"Too few cloud points: {len(points)}")
        units = dep.data.get("pointcloud_units", cfg["input"]["units"])
        if units != "m":
            raise ValueError(f"Point cloud units must be m, got {units!r}")
        if "pointcloud_units" not in dep.data:
            report.warn(log, "Legacy cloud: metres asserted by input.units=m; units cannot be inferred from PLY")
        base = RobotBaseFrame.from_dict(dep.require("robot_base_pose_world"))
        points, cloud_to_base = points_to_base(points, dep.data, base)
        extent = float(np.max(np.ptp(points, axis=0)))
        if not cfg["input"]["min_extent_m"] <= extent <= cfg["input"]["max_extent_m"]:
            raise ValueError(f"Cloud extent {extent:g} m outside configured limits; verify units/segmentation")
        count = len(points)
        if count > cfg["input"]["max_points"]:
            rng = np.random.default_rng(cfg["input"]["seed"])
            points = points[np.sort(rng.choice(count, cfg["input"]["max_points"], replace=False))]
        center = points.mean(axis=0)
        centered = np.ascontiguousarray(points - center, dtype=np.float32)
        cloud_file = out / "grasp_graspgen_input_centered.ply"
        write_ply(cloud_file, centered)
        report.artifacts["centered_cloud_ply"] = str(cloud_file)
        exact_cloud = out / "grasp_graspgen_input_centered.npy"
        np.save(exact_cloud, centered, allow_pickle=False)
        report.artifacts["centered_cloud_npy"] = str(exact_cloud)
        report.data.update(
            source_pipeline=dep.stage.split("_")[0], coordinate_frame_id=base.frame_id,
            robot_base_pose_world=base.to_dict(),
            robot_description=dep.data.get("robot_description"),
            target_instrument=(StageReport.load(plan_file, expect_stage="plan").require("target_instrument")
                               if plan_file.exists() else "surgical_instrument"),
            input_pointcloud={"source_ply": str(ply.resolve()), "sha256": hashlib.sha256(ply.read_bytes()).hexdigest(),
                              "total_points": count, "sent_points": len(points), "units": units,
                              "max_extent_m": extent, "center_base_xyz_m": center.tolist(),
                              "source_frame": dep.data.get("coordinate_frame_id"),
                              "T_base_from_pointcloud": cloud_to_base.tolist()},
            checks={"finite_cloud": True, "metric_extent": True, "units_asserted": True,
                    "frame_transform": True, "cloud_centered": True, "pca_filter_used": False,
                    "ik_checked": False, "collision_checked": False, "contact_checked": False,
                    "surgical_functional_zones_checked": False})

        def check_server(metadata):
            report.data["server_metadata"] = metadata
            validate_metadata(metadata, cfg)
            report.data["checks"]["checkpoint_gripper_matches"] = True

        poses, scores, metadata = client.infer(centered, cfg, metadata_callback=check_server)
        t_tcp = rigid_transform(cfg["robot"]["T_gripper_tcp"])
        candidates = rank_candidates(poses, scores, center, base, t_tcp)
        report.data["checks"].update(rigid_candidates=True, global_score_sort=True)
        best = candidates[0]
        tcp = np.asarray(best["T_base_tcp"])
        robot = cfg["robot"]
        checks = {
            "motion_enabled": robot["motion_enabled"],
            "same_gripper": robot["gripper"] == cfg["model_gripper"],
            "tcp_transform_verified": robot["tcp_transform_verified"],
            "compatibility_verified": robot["compatibility_verified"],
            "verification_evidence_present": bool(robot["verification_evidence"].strip()),
            "metric_depth_and_extrinsics_verified": dep.data.get("geometry_verified_for_motion") is True,
            "server_provenance_present": bool(metadata.get("ros2project")),
        }
        blocked = [k for k, passed in checks.items() if not passed]
        report.data.update(
            candidates=candidates, selected_candidate_index=0, selected=best,
            score_semantics="official discriminator confidence, not physical success validation",
            gripper_convention={"pose_origin": "gripper base", "approach_axis": "+Z", "closing_axis": "+X",
                                "tcp_closing_axis": "+Y", "T_gripper_tcp": t_tcp.tolist(),
                                "nominal_robotiq_depth_m": 0.195},
            motion={"allowed": not blocked, "checks": checks, "blocked_reasons": blocked},
            readiness="GRASP_CANDIDATES_UNVALIDATED",
            designated_grasp_geometry={
                "selected_zone": "model candidate; surgical zones not validated",
                "grasp_centroid_base_frame_xyz_m": tcp[:3, 3].tolist(),
                "pose_semantics": "TCP; original gripper-base pose retained in selected.T_base_gripper",
                "recommended_gripper_orientation": {
                    "quaternion_xyzw": best["tcp_quaternion_xyzw_base"],
                    "approach_vector_base": tcp[:3, 2].tolist(),
                    "jaw_closing_vector_base": tcp[:3, 1].tolist(),
                    "finger_alignment_vector_base": tcp[:3, 0].tolist()},
                "gripper_compatibility_check": {
                    "parallel_jaw_suitable": checks["same_gripper"] and robot["compatibility_verified"],
                    "model_gripper": cfg["model_gripper"], "robot_gripper": robot["gripper"],
                    "opening_prediction_available": False}})
        candidates_file = out / "grasp_graspgen_candidates.json"
        write_json(candidates_file, {"units": "m", "frame_id": base.frame_id, "candidates": candidates})
        report.artifacts["candidates_json"] = str(candidates_file)
        from pipeline.graspgen.preview import plot_candidates
        overview = out / "grasp_graspgen_overview.png"
        plot_candidates(overview, points, candidates, cfg["visualization"]["max_candidates"])
        report.artifacts["overview_png"] = str(overview)
        report.warn(log, "GraspGen candidates are unvalidated: no scene collisions, IK, surgical zones or physical contact checked")
        if blocked:
            report.warn(log, "Motion blocked: " + ", ".join(blocked))
    return report
