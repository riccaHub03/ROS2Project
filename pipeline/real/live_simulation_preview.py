"""Preview a calibrated ROS RGB-D grasp in a synthetic PyBullet cell, never on hardware.

The observed surface is approximated by a convex proxy. No image-derived object
or grasp is moved to a different position. A calibrated live camera/TF, matching
base frame and a compatible configured source tray are required.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pybullet as p
import trimesh
from PIL import Image, ImageDraw

from pipeline.common import paths
from pipeline.common.ply import read_ply, write_ply
from pipeline.common.report import StageReport, write_json
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.graspgen.backend import validate_metadata
from pipeline.sim.gripper_geometry import ROBOTIQ_PROXY
from scripts.test_clip_sim import ClipMotion, ClipScene


def prepare_live(run: Path, config: dict, out: Path):
    capture = StageReport.load(run / 'real_capture.json', expect_stage='real_capture')
    dep = StageReport.load(run / 'real_deprojection.json', expect_stage='real_deprojection')
    grasp = StageReport.load(run / 'grasp_graspgen.json', expect_stage='grasp')
    plan = StageReport.load(run / 'plan.json', expect_stage='plan')
    acquisition = capture.data.get('acquisition', {})
    if capture.data.get('source_type') != 'live_ros2_rgbd' or capture.data.get('motion_safe') is not True:
        raise ValueError('live preview requires measured ROS RGB-D and verified camera extrinsics')
    if not config.get('camera_overhead', {}).get('extrinsics', {}).get('verified', False):
        raise ValueError('live preview requires camera extrinsics verified in the current configuration')
    if not acquisition.get('camera_pose_from_tf') or acquisition.get('distortion_nonzero') is not False:
        raise ValueError('live preview requires camera TF and rectified, distortion-free input')
    if float(acquisition.get('depth_valid_fraction', 0)) < .5:
        raise ValueError('live depth has fewer than 50% valid pixels')
    if dep.data.get('motion_safe') is not True or dep.data.get('pointcloud_units') != 'm':
        raise ValueError('live point cloud must come from verified metric geometry')
    original = grasp.data
    if original.get('source_pipeline') != 'real' or original.get('grasp_backend') != 'graspgen':
        raise ValueError('live preview requires a GraspGen report from real perception')
    if original.get('motion', {}).get('allowed') is not False:
        raise ValueError('this preview requires the physical-motion gate to remain blocked')
    validate_metadata(original['server_metadata'], original['configuration'])
    cloud = dep.artifact('pointcloud_ply')
    digest = hashlib.sha256(cloud.read_bytes()).hexdigest()
    if digest != original['input_pointcloud']['sha256']:
        raise ValueError('GraspGen cloud hash does not match live deprojection')
    base = RobotBaseFrame.from_dict(dep.require('robot_base_pose_world'))
    configured_base = RobotBaseFrame.from_config(config)
    if (base.frame_id != configured_base.frame_id or not np.allclose(base.position, configured_base.position)
            or not np.allclose(base.rotation, configured_base.rotation)):
        raise ValueError('live capture and simulation cell use different robot base frames')
    if not (np.allclose(base.position, 0) and np.allclose(base.rotation, np.eye(3))):
        raise ValueError('live preview currently requires the world frame to coincide with the robot base')
    if dep.data.get('coordinate_frame_id') != base.frame_id:
        raise ValueError('live cloud must be expressed in the configured robot base frame')
    points, colors = read_ply(cloud)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 21 or not np.isfinite(points).all():
        raise ValueError('live point cloud is invalid or too small')
    source, destination = plan.require('source_tray'), plan.require('destination_tray')
    trays = config['environment']['trays']
    if source not in trays or destination not in trays or source == destination:
        raise ValueError('live preview needs different configured source and destination trays')
    tray = trays[source]
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    tray_center, tray_dims = np.asarray(tray['position_xyz_m']), np.asarray(tray['dimensions_xyz_m'])
    tray_top = tray_center[2] + tray_dims[2] / 2
    if np.any(lo[:2] < tray_center[:2] - tray_dims[:2] / 2) or np.any(hi[:2] > tray_center[:2] + tray_dims[:2] / 2):
        raise ValueError('live cloud does not fit the configured source tray; no automatic relocation is performed')
    if not -.005 <= lo[2] - tray_top <= .020:
        raise ValueError('live cloud height does not match source tray; verify calibration and environment')
    mesh = trimesh.points.PointCloud(points - center).convex_hull
    if not mesh.is_volume or np.min(hi - lo) <= .0005:
        raise ValueError('live cloud cannot form a non-degenerate convex proxy')
    mesh_file = out / 'live_cloud_convex_proxy.obj'
    mesh.export(mesh_file)
    write_ply(out / 'live_cloud_in_simulation.ply', points, colors)
    manifest = {'mode': 'calibrated_live_proxy_pybullet_only', 'source_cloud_sha256': digest,
                'source_run': str(run), 'T_sim_base_from_original_base': np.eye(4).tolist(),
                'proxy_mesh': str(mesh_file), 'proxy_extents_m': (hi - lo).tolist(),
                'proxy_position_world_m': center.tolist(), 'hardware_motion_allowed': False,
                'physical_grasp_validated': False,
                'assumptions': ['single-view convex hull fills holes and does not recover hidden surfaces',
                                'live points retained in the configured robot base without relocation',
                                'only configured table and trays are collision obstacles',
                                'ideal fixed attachment, no friction/contact validation']}
    write_json(out / 'live_simulation_manifest.json', manifest)
    return original, copy.deepcopy(original), plan.data, mesh_file, center, hi - lo, manifest, cloud


class LiveMotion(ClipMotion):
    def _check_source(self, report):
        super()._check_source(report)
        self.validation['source_kind'] = 'calibrated_live_ros2_pybullet_preview'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--omit-return', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error('Output must be a new directory')
    out.mkdir(parents=True)
    os.environ[paths.OUTPUT_ENV] = str(out)
    config = load_environment_config(args.config)
    config['simulation_gripper'] = ROBOTIQ_PROXY.name
    config.setdefault('motion_control', {})['return_after_place'] = not args.omit_return
    scene = planner = video = video_log = None
    steps = 0
    result = {'status': 'FAILED', 'hardware_motion_allowed': False, 'physical_grasp_validated': False,
              'trajectory_executed': False}
    try:
        original, report, plan, mesh, center, dims, manifest, cloud = prepare_live(run, config, out)
        manifest['optional_return_planned'] = not args.omit_return
        write_json(out / 'live_simulation_manifest.json', manifest)
        scene = ClipScene(config, plan, mesh, center, dims, cloud)
        scene.save_overview(out / 'scene_initial.png')
        planner = LiveMotion(scene, report, original, np.eye(4), manifest)
        write_json(out / 'transformed_grasp_report.json', report)
        if args.video:
            ffmpeg = shutil.which('ffmpeg')
            if not ffmpeg:
                raise RuntimeError('ffmpeg is required for video')
            video_log = (out / 'ffmpeg.log').open('wb')
            video = subprocess.Popen([ffmpeg, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                      '-s', '640x480', '-r', '10', '-i', '-', '-an', '-c:v', 'libx264',
                                      '-pix_fmt', 'yuv420p', str(out / 'live_simulation.mp4')],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=video_log)

        def record(label):
            if video is None:
                return
            view = p.computeViewMatrixFromYawPitchRoll([.45, 0, .30], 1.9, 48, -32, 0, 2)
            projection = p.computeProjectionMatrixFOV(48, 640 / 480, .05, 5)
            rendered = p.getCameraImage(640, 480, view, projection, renderer=p.ER_TINY_RENDERER,
                                        physicsClientId=scene.cid)
            rgb = np.asarray(rendered[2], dtype=np.uint8).reshape(480, 640, 4)[..., :3]
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 639, 37), fill=(10, 10, 10))
            draw.text((8, 4), 'SIMULATION - approximate object / ideal grasp', fill='white')
            draw.text((8, 20), label, fill='white')
            video.stdin.write(image.tobytes())

        while not planner.finished and steps < 240 * 180:
            planner.tick()
            scene.step()
            steps += 1
            if video is not None and steps % 24 == 0:
                record(planner.phases[min(planner.index, len(planner.phases) - 1)]['label'])
        for i in range(480):
            scene.step()
            if video is not None and i % 24 == 0:
                record('Final position')
        if video is not None:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError('ffmpeg failed; see ffmpeg.log')
            video = None
            result['video_mp4'] = str(out / 'live_simulation.mp4')
        motion = planner.report()
        scene.save_overview(out / 'scene_final.png')
        result.update(status=motion['status'] if planner.finished else 'EXECUTION_TIMEOUT',
                      trajectory_executed=steps > 0, simulation_steps=steps, motion=motion,
                      selected_server_index=planner.validation['selected_server_index'])
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        result.update(status='FAILED', error=str(exc), trajectory_executed=steps > 0, simulation_steps=steps)
        if scene is not None:
            scene.save_overview(out / 'scene_failure.png')
        logging.exception('Live simulation preview failed')
    finally:
        if video is not None:
            try:
                video.stdin.close()
            except OSError:
                pass
            video.wait()
        if video_log is not None:
            video_log.close()
        if planner is not None:
            planner.close()
        if scene is not None:
            scene.close()
        write_json(out / 'live_simulation_result.json', result)
    print('Result:', result['status'], out / 'live_simulation_result.json', flush=True)
    return 0 if result['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    raise SystemExit(main())
