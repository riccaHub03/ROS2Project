#!/usr/bin/env python3
"""Replan saved clip GraspGen candidates in a *synthetic* PyBullet cell only.

The observed one-view cloud becomes a convex proxy; the clip's nominal plane
pose and an explicit placement map the cloud AND grasps to the simulated tray.
This does not certify physical geometry, contact, calibration or robot motion.
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
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.common import paths
from pipeline.common.ply import read_ply, write_ply
from pipeline.common.reconstruction import CameraModel
from pipeline.common.report import StageReport, write_json
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import RobotBaseFrame, quaternion_from_matrix
from pipeline.graspgen.backend import validate_metadata
from pipeline.graspgen.config import rigid_transform
from pipeline.sim.graspgen_motion import GraspGenMotionPreview
from pipeline.sim.gripper_geometry import ROBOTIQ_PROXY
from pipeline.sim.robot_scene import RobotScene


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(run, clip, config, out):
    grasp = StageReport.load(run / 'grasp_graspgen.json', expect_stage='grasp').data
    capture = StageReport.load(run / 'real_capture.json', expect_stage='real_capture').data
    dep = StageReport.load(run / 'real_deprojection.json', expect_stage='real_deprojection').data
    clip_input = json.loads((run / 'clip_input.json').read_text())
    if grasp.get('source_pipeline') != 'real' or grasp.get('grasp_backend') != 'graspgen':
        raise ValueError('Expected GraspGen report from a real RGB-D capture')
    if grasp.get('motion', {}).get('allowed') is not False:
        raise ValueError('Expected physical motion to remain blocked')
    validate_metadata(grasp['server_metadata'], grasp['configuration'])
    cloud_file = run / 'real_deprojection_cloud.ply'
    if sha(cloud_file) != grasp['input_pointcloud']['sha256']:
        raise ValueError('Source cloud hash does not match the GraspGen report')
    base = RobotBaseFrame.from_dict(dep['robot_base_pose_world'])
    if not (np.allclose(base.position, 0) and np.allclose(base.rotation, np.eye(3))):
        raise ValueError('This demonstration requires coincident simulation base and world')
    idx = clip_input['index']
    frames = json.loads((clip / 'frames.json').read_text())['frames']
    frame = next(f for f in frames if f['index'] == idx)
    nominal = rigid_transform(frame['T_world_from_color_nominal'], 'clip nominal pose')
    old_base_camera = CameraModel.from_dict(capture['camera']).transformer(base).T_base_opt
    mapping = nominal @ np.linalg.inv(old_base_camera)
    points, colors = read_ply(cloud_file)
    transformed = points @ mapping[:3, :3].T + mapping[:3, 3]
    tray = config['environment']['trays']['tray_2']
    center_xy = (transformed.min(axis=0) + transformed.max(axis=0))[:2] / 2
    top = tray['position_xyz_m'][2] + tray['dimensions_xyz_m'][2] / 2
    shift = np.array([tray['position_xyz_m'][0] - center_xy[0],
                      tray['position_xyz_m'][1] - center_xy[1], top + .001 - transformed[:, 2].min()])
    mapping[:3, 3] += shift
    moved = points @ mapping[:3, :3].T + mapping[:3, 3]
    lo, hi = moved.min(axis=0), moved.max(axis=0)
    center = (lo + hi) / 2
    mesh = trimesh.points.PointCloud(moved - center).convex_hull
    if not mesh.is_volume or np.min(hi - lo) <= .0005:
        raise ValueError('Cloud proxy must be a non-degenerate closed convex hull')
    mesh_path = out / 'clip_cloud_convex_proxy.obj'
    mesh.export(mesh_path)
    write_ply(out / 'clip_cloud_in_simulation.ply', moved, colors)
    result = copy.deepcopy(grasp)
    old_center = np.asarray(grasp['input_pointcloud']['center_base_xyz_m'])
    new_center = mapping[:3, :3] @ old_center + mapping[:3, 3]
    for candidate in result['candidates']:
        bg = mapping @ rigid_transform(candidate['T_base_gripper'])
        bt = mapping @ rigid_transform(candidate['T_base_tcp'])
        centered = bg.copy()
        centered[:3, 3] -= new_center
        candidate.update(T_centered_gripper=centered.tolist(), T_base_gripper=bg.tolist(),
                         T_world_gripper=bg.tolist(), T_base_tcp=bt.tolist(), T_world_tcp=bt.tolist(),
                         tcp_quaternion_xyzw_base=quaternion_from_matrix(bt[:3, :3]))
    result['selected'] = result['candidates'][0]
    result['robot_base_pose_world'] = base.to_dict()
    manifest = {'mode': 'clip_proxy_pybullet_only', 'source_run': str(run), 'source_cloud_sha256': sha(cloud_file),
                'source_grasp_sha256': sha(run / 'grasp_graspgen.json'), 'frame_index': idx,
                'T_sim_base_from_original_base': mapping.tolist(),
                'T_clip_nominal_from_color': nominal.tolist(), 'placement_translation_m': shift.tolist(),
                'proxy_mesh': str(mesh_path), 'proxy_extents_m': (hi - lo).tolist(),
                'proxy_position_world_m': center.tolist(), 'scale_assumed_m_per_unit': clip_input['depth_scale_m_per_unit_assumed'],
                'assumptions': ['single-view convex hull fills holes and does not recover hidden surfaces',
                                'clip plane pose is nominal and simulation placement is hypothetical',
                                'depth scale unverified, RGB distortion uncorrected',
                                'only configured table and trays are collision obstacles',
                                'ideal fixed attachment, no force/contact validation'],
                'hardware_motion_allowed': False, 'physical_grasp_validated': False}
    write_json(out / 'clip_simulation_manifest.json', manifest)
    return grasp, result, mesh_path, center, hi - lo, mapping, manifest


class ClipScene(RobotScene):
    def __init__(self, config, plan, mesh_path, center, dimensions, source):
        self._clip_mesh = mesh_path
        self._clip_center = center
        self._clip_dimensions = dimensions
        self._clip_source = source
        super().__init__(config, plan, gui=False, capture=False, procedural_tool_fallback=False)

    def _build_tool(self, seed):
        self.mesh_file = str(self._clip_mesh)
        self.mesh_info = {'dimensions_m': self._clip_dimensions.tolist(), 'source_mesh': str(self._clip_source)}
        self.tool_initial_pos = self._clip_center.tolist()
        self.tool_initial_orn = [0, 0, 0, 1]
        self.tool_bottom = float(self.tool_initial_pos[2] - self._clip_dimensions[2] / 2)
        col = p.createCollisionShape(p.GEOM_MESH, fileName=self.mesh_file, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=self.mesh_file, rgbaColor=[.25, .6, .65, 1], physicsClientId=self.cid)
        self.tool = p.createMultiBody(0, col, vis, self.tool_initial_pos, self.tool_initial_orn, physicsClientId=self.cid)
        for link in range(-1, p.getNumJoints(self.gripper, physicsClientId=self.cid)):
            p.setCollisionFilterPair(self.gripper, self.tool, link, -1, 0, physicsClientId=self.cid)

    def show_grasp(self, report):
        pass  # Headless: candidate checks are recorded, not drawn by PyBullet debug lines.

    def status(self, text, color=(0, 0, 0)):
        pass


class ClipMotion(GraspGenMotionPreview):
    """Simulation-only adapter; leave the original simulator and hardware gates unchanged."""
    def __init__(self, scene, transformed, original, mapping, manifest):
        self._original, self._mapping, self._manifest = original, mapping, manifest
        super().__init__(scene, transformed)

    def _check_source(self, report):
        original = self._original
        if self.scene.gripper_geometry != ROBOTIQ_PROXY or report.get('source_pipeline') != 'real':
            raise ValueError('Clip preview requires real source and the Robotiq simulation model')
        if report.get('motion', {}).get('allowed') is not False:
            raise ValueError('Physical motion gate must stay blocked')
        validate_metadata(original['server_metadata'], original['configuration'])
        if len(report['candidates']) != len(original['candidates']):
            raise ValueError('Candidate count changed')
        for a, b in zip(report['candidates'], original['candidates']):
            if a['server_index'] != b['server_index'] or a['score'] != b['score']:
                raise ValueError('Candidate identity changed')
            for key in ('T_base_gripper', 'T_base_tcp'):
                if not np.allclose(a[key], self._mapping @ rigid_transform(b[key]), atol=1e-6):
                    raise ValueError('Candidate and cloud transforms do not match')
        self.validation.update(source_kind='real_clip_mapped_to_hypothetical_simulation',
                               source_cloud_sha256=self._manifest['source_cloud_sha256'],
                               assumptions=self._manifest['assumptions'], hardware_motion_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--clip', type=Path, default=ROOT / 'execution/clip')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--omit-return', action='store_true',
                        help='End after the checked retreat; do not plan the optional park pose')
    parser.add_argument('--video', action='store_true', help='Render a headless MP4 of the simulation (requires ffmpeg)')
    args = parser.parse_args()
    run, clip, out = args.run_dir.resolve(), args.clip.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error('Output must be a new directory')
    out.mkdir(parents=True)
    os.environ[paths.OUTPUT_ENV] = str(out)
    config = load_environment_config(None)
    config['simulation_gripper'] = ROBOTIQ_PROXY.name
    config.setdefault('motion_control', {})['return_after_place'] = not args.omit_return
    original, report, mesh, center, dims, mapping, manifest = prepare(run, clip, config, out)
    manifest['optional_return_planned'] = not args.omit_return
    write_json(out / 'clip_simulation_manifest.json', manifest)
    plan = StageReport.load(run / 'plan.json', expect_stage='plan').data
    if (plan['source_tray'], plan['destination_tray']) != ('tray_2', 'tray_1'):
        raise ValueError('This demonstration places the proxy on tray_2 and transfers to tray_1')
    scene = planner = video = video_log = None
    steps = 0
    result = {'status': 'FAILED', 'hardware_motion_allowed': False, 'physical_grasp_validated': False,
              'trajectory_executed': False, 'source_manifest': str(out / 'clip_simulation_manifest.json')}
    try:
        scene = ClipScene(config, plan, mesh, center, dims, run / 'real_deprojection_cloud.ply')
        scene.save_overview(out / 'scene_initial.png')
        planner = ClipMotion(scene, report, original, mapping, manifest)
        write_json(out / 'transformed_grasp_report.json', report)
        if args.video:
            ffmpeg = shutil.which('ffmpeg')
            if not ffmpeg:
                raise RuntimeError('Video requested but ffmpeg was not found')
            video_log = (out / 'ffmpeg.log').open('wb')
            video = subprocess.Popen([ffmpeg, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                      '-s', '640x480', '-r', '10', '-i', '-', '-an', '-c:v', 'libx264',
                                      '-pix_fmt', 'yuv420p', str(out / 'clip_simulation.mp4')],
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
        for _ in range(480):
            scene.step()
            if video is not None and _ % 24 == 0:
                record('Final position')
        if video is not None:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError('ffmpeg failed; see ffmpeg.log')
            video = None
            result['video_mp4'] = str(out / 'clip_simulation.mp4')
        motion_report = planner.report()
        scene.save_overview(out / 'scene_final.png')
        result.update(status=motion_report['status'], trajectory_executed=True, simulation_steps=steps,
                      motion=motion_report, selected_server_index=planner.validation['selected_server_index'])
        if not planner.finished:
            result['status'] = 'EXECUTION_TIMEOUT'
    except (OSError, RuntimeError, ValueError) as exc:
        result.update(status='FAILED', error=str(exc), trajectory_executed=steps > 0,
                      simulation_steps=steps)
        if scene is not None:
            scene.save_overview(out / 'scene_failure.png')
        logging.exception('Clip simulation failed')
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
        write_json(out / 'clip_simulation_result.json', result)
    print('Result:', result['status'], out / 'clip_simulation_result.json', flush=True)
    return 0 if result['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    raise SystemExit(main())
