"""Real-input simulation preview contracts; no GPU, robot or ROS needed."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from pipeline.common.ply import read_ply, write_ply
from pipeline.common.report import StageError, StageReport, write_json
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.graspgen.config import load_config
from pipeline.real import live_simulation_preview, pipeline, simulation_preview


class RealSimulationPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load_environment_config(None)
        self.config['camera_overhead']['extrinsics']['verified'] = True
        self.base = RobotBaseFrame.from_config(self.config)
        self.points = np.array([[.59 + .01*i/2, -.43 + .02*j/2, .091 + .004*k/2]
                                for i in range(3) for j in range(3) for k in range(3)])
        self.cloud = self.root / 'real_deprojection_cloud.ply'
        write_ply(self.cloud, self.points)
        self.capture = StageReport('real_capture', data={'source_type': 'live_ros2_rgbd', 'motion_safe': True,
                                   'acquisition': {'camera_pose_from_tf': True, 'distortion_nonzero': False,
                                                   'depth_valid_fraction': .9}})
        self.dep = StageReport('real_deprojection', artifacts={'pointcloud_ply': str(self.cloud)},
                               data={'motion_safe': True, 'pointcloud_units': 'm',
                                     'coordinate_frame_id': self.base.frame_id,
                                     'robot_base_pose_world': self.base.to_dict()})
        cfg = load_config(None)
        metadata = {'gripper_name': cfg['model_gripper'], 'model_name': 'diffusion-discriminator',
                    'gripper_config': cfg['checkpoint_config'],
                    'ros2project': {'revision': cfg['expected_revision'], 'planner': 'diffusion',
                                    'config_sha256': 'a', 'generator_sha256': 'b', 'discriminator_sha256': 'c'}}
        self.grasp = StageReport('grasp', data={'source_pipeline': 'real', 'grasp_backend': 'graspgen',
                                 'configuration': cfg, 'server_metadata': metadata,
                                 'input_pointcloud': {'sha256': hashlib.sha256(self.cloud.read_bytes()).hexdigest()},
                                 'motion': {'allowed': False}, 'candidates': []})
        self.plan = StageReport('plan', data={'source_tray': 'tray_2', 'destination_tray': 'tray_1'})
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        for name, report in [('real_capture.json', self.capture), ('real_deprojection.json', self.dep),
                             ('grasp_graspgen.json', self.grasp), ('grasp.json', self.grasp),
                             ('plan.json', self.plan)]:
            report.save(self.root / name)

    def test_live_uses_calibrated_base_positions_without_relocation(self):
        out = self.root / 'live'
        out.mkdir()
        _, _, _, _, _, _, manifest, _ = live_simulation_preview.prepare_live(self.root, self.config, out)
        points, _ = read_ply(out / 'live_cloud_in_simulation.ply')
        np.testing.assert_allclose(points, self.points, atol=1e-6)
        np.testing.assert_array_equal(manifest['T_sim_base_from_original_base'], np.eye(4))
        self.assertFalse(manifest['hardware_motion_allowed'])

    def test_launcher_requires_graspgen_for_simulation(self):
        with patch('pipeline.plan.run') as planner:
            result = pipeline.main(['--source', 'clip', '--simulate-deposit',
                                    '--out-dir', str(self.root / 'new')])
        self.assertEqual(result, 1)
        planner.assert_not_called()

    def test_live_rejects_unverified_distorted_or_incompatible_capture(self):
        out = self.root / 'live'
        out.mkdir()
        for change in ({'motion_safe': False}, {'acquisition': {'camera_pose_from_tf': False,
                       'distortion_nonzero': False, 'depth_valid_fraction': .9}},
                       {'acquisition': {'camera_pose_from_tf': True, 'distortion_nonzero': True,
                                        'depth_valid_fraction': .9}}):
            with self.subTest(change=change):
                saved = copy.deepcopy(self.capture.data)
                self.capture.data.update(change)
                self.save()
                with self.assertRaises(ValueError):
                    live_simulation_preview.prepare_live(self.root, self.config, out)
                self.capture.data = saved
        self.save()
        other = copy.deepcopy(self.config)
        other['environment']['trays']['tray_2']['position_xyz_m'][0] += 2
        with self.assertRaisesRegex(ValueError, 'does not fit'):
            live_simulation_preview.prepare_live(self.root, other, out)
        other = copy.deepcopy(self.config)
        other['camera_overhead']['extrinsics']['verified'] = False
        with self.assertRaisesRegex(ValueError, 'extrinsics verified'):
            live_simulation_preview.prepare_live(self.root, other, out)

    def test_live_rejects_modified_cloud(self):
        out = self.root / 'live'
        out.mkdir()
        self.cloud.write_text(self.cloud.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hash'):
            live_simulation_preview.prepare_live(self.root, self.config, out)

    def test_preview_reports_success_only_when_child_completed(self):
        def completed(command, cwd, stdout, stderr):
            sim = Path(command[command.index('--out-dir') + 1])
            sim.mkdir()
            write_json(sim / 'live_simulation_result.json', {'status': 'COMPLETED',
                       'selected_server_index': 3, 'motion': {'tool_settled_on_destination': True}})
            return SimpleNamespace(returncode=0)
        with patch('pipeline.real.simulation_preview.subprocess.run', side_effect=completed):
            report = simulation_preview.run(out_dir=str(self.root), source='ros2', clip_dir='unused')
        self.assertEqual(report.data['simulation_status'], 'COMPLETED')
        self.assertFalse(report.data['hardware_motion_allowed'])
        self.assertTrue(report.data['tool_settled_on_destination'])

    def test_failure_does_not_claim_success_or_retry_on_calibration_error(self):
        def failed(command, cwd, stdout, stderr):
            sim = Path(command[command.index('--out-dir') + 1])
            sim.mkdir()
            write_json(sim / 'live_simulation_result.json', {'status': 'FAILED', 'error': 'camera not calibrated'})
            return SimpleNamespace(returncode=1)
        with patch('pipeline.real.simulation_preview.subprocess.run', side_effect=failed) as child:
            with self.assertRaises(StageError):
                simulation_preview.run(out_dir=str(self.root), source='ros2', clip_dir='unused', max_attempts=3)
        self.assertEqual(child.call_count, 1)
        report = StageReport.load(self.root / 'real_simulation_preview.json', require_completed=False)
        self.assertEqual(report.status, 'FAILED')
        self.assertFalse(report.data['hardware_motion_allowed'])

    def test_new_batch_only_after_candidate_rejection(self):
        calls = []
        def child(command, cwd, stdout, stderr):
            sim = Path(command[command.index('--out-dir') + 1])
            sim.mkdir()
            calls.append(sim)
            if len(calls) == 1:
                write_json(sim / 'live_simulation_result.json', {'status': 'FAILED', 'trajectory_executed': False,
                           'error': 'No executable GraspGen candidate: inspect report'})
                write_json(sim / 'graspgen_sim_motion_validation.json', {'attempts': []})
                return SimpleNamespace(returncode=1)
            write_json(sim / 'live_simulation_result.json', {'status': 'COMPLETED', 'selected_server_index': 9,
                       'motion': {'tool_settled_on_destination': True}})
            return SimpleNamespace(returncode=0)

        def new_grasps(**kwargs):
            self.grasp.save(Path(kwargs['out_dir']) / 'grasp_graspgen.json')
            return self.grasp

        with patch('pipeline.real.simulation_preview.subprocess.run', side_effect=child), \
             patch('pipeline.real.simulation_preview.grasp_stage.run', side_effect=new_grasps) as infer:
            report = simulation_preview.run(out_dir=str(self.root), source='ros2', clip_dir='unused',
                                            max_attempts=3, grasp_kwargs={'backend': 'graspgen'})
        self.assertEqual(infer.call_count, 1)
        self.assertEqual(report.data['selected_batch'], 2)
        self.assertEqual(len(report.data['attempts']), 2)
        self.assertEqual(Path(report.artifacts['selected_grasp_report']).parent.name, 'attempt_02')
        self.assertFalse(report.data['hardware_motion_allowed'])


if __name__ == '__main__':
    unittest.main()
