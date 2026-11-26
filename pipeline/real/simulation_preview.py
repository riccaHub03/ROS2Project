"""Run a simulation-only preview from real perception reports; never command hardware."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from pipeline import grasp as grasp_stage
from pipeline.common import paths
from pipeline.common.report import StageError, StageReport, stage_run

log = logging.getLogger('real_simulation_preview')


def run(*, out_dir: str, source: str, clip_dir: str, config_path: Optional[str] = None,
        omit_return: bool = False, video: bool = False, max_attempts: int = 1,
        grasp_kwargs: Optional[dict] = None) -> StageReport:
    out = Path(out_dir).resolve()
    with stage_run('real_simulation_preview', out / 'real_simulation_preview.json',
                   inputs={'capture': out / paths.REAL_CAPTURE, 'grasp': out / paths.GRASP,
                           'config': config_path}, logger=log) as report:
        captured = StageReport.load(out / paths.REAL_CAPTURE, expect_stage='real_capture')
        grasp = StageReport.load(out / paths.GRASP, expect_stage='grasp')
        if grasp.data.get('grasp_backend') != 'graspgen':
            raise StageError('real_simulation_preview', 'simulation requires actual GraspGen candidates')
        if not 1 <= max_attempts <= 10:
            raise StageError('real_simulation_preview', 'maximum GraspGen attempts must be 1–10')
        if source == 'clip' and captured.data.get('source_type') == 'recorded_clip_rgbd':
            if config_path:
                raise StageError('real_simulation_preview', 'clip preview currently uses the default cell configuration; '
                                 'custom --config is not supported in this preview')
            result_name, video_name = 'clip_simulation_result.json', 'clip_simulation.mp4'
        elif source == 'ros2' and captured.data.get('source_type') == 'live_ros2_rgbd':
            result_name, video_name = 'live_simulation_result.json', 'live_simulation.mp4'
        else:
            raise StageError('real_simulation_preview',
                             'simulation requires matching --source clip or a calibrated --source ros2')
        report.data.update(execution_scope='pybullet_only', hardware_motion_allowed=False,
                           physical_grasp_validated=False, source_type=captured.data.get('source_type'),
                           object_geometry='single-view convex approximation', attachment='ideal fixed constraint',
                           return_omitted=omit_return, attempts=[])
        for index in range(1, max_attempts + 1):
            run_dir = out if index == 1 else out / 'simulation_attempts' / f'attempt_{index:02d}'
            if index > 1:
                if grasp_kwargs is None:
                    raise StageError('real_simulation_preview', 'retry requires explicit GraspGen configuration')
                run_dir.mkdir(parents=True, exist_ok=False)
                for name in ('plan.json', 'real_capture.json', 'real_deprojection.json',
                             'real_deprojection_cloud.ply') + (('clip_input.json',) if source == 'clip' else ()):
                    (run_dir / name).symlink_to(out / name)
                log.info('new independent GraspGen batch %d/%d', index, max_attempts)
                grasp_stage.run(deprojection_path=str(out / paths.REAL_DEPROJECTION),
                                plan_path=str(out / paths.PLAN), out_dir=str(run_dir),
                                config_path=config_path, robot_description=grasp.data.get('robot_description'),
                                **grasp_kwargs)
            sim_out = run_dir / 'simulation'
            if sim_out.exists():
                raise StageError('real_simulation_preview', f'simulation output already exists: {sim_out}')
            if source == 'clip':
                command = [sys.executable, str(paths.project_root() / 'scripts/test_clip_sim.py'),
                           '--run-dir', str(run_dir), '--clip', str(Path(clip_dir).resolve()), '--out-dir', str(sim_out)]
            else:
                command = [sys.executable, '-m', 'pipeline.real.live_simulation_preview',
                           '--run-dir', str(run_dir), '--out-dir', str(sim_out)]
                if config_path:
                    command.extend(['--config', config_path])
            if omit_return:
                command.append('--omit-return')
            if video:
                command.append('--video')
            log_file = run_dir / 'simulation_preview.log'
            with log_file.open('w', encoding='utf-8') as stream:
                process = subprocess.run(command, cwd=paths.project_root(), stdout=stream, stderr=subprocess.STDOUT)
            result_file = sim_out / result_name
            result = json.loads(result_file.read_text()) if result_file.is_file() else {}
            validation = sim_out / 'graspgen_sim_motion_validation.json'
            attempt = {'batch': index, 'grasp_report': str(run_dir / 'grasp_graspgen.json'),
                       'status': result.get('status', 'NO_REPORT'), 'exit_code': process.returncode,
                       'simulation_result': str(result_file) if result_file.exists() else None,
                       'validation': str(validation) if validation.exists() else None,
                       'log': str(log_file)}
            report.data['attempts'].append(attempt)
            if process.returncode == 0 and result.get('status') == 'COMPLETED':
                report.artifacts.update(log=str(log_file), simulation_result=str(result_file),
                                        selected_grasp_report=str(run_dir / 'grasp_graspgen.json'))
                if validation.exists():
                    report.artifacts['validation'] = str(validation)
                preview = sim_out / video_name
                if preview.is_file():
                    report.artifacts['video'] = str(preview)
                report.data.update(simulation_status='COMPLETED', selected_batch=index,
                                   selected_server_index=result.get('selected_server_index'),
                                   tool_settled_on_destination=result.get('motion', {}).get('tool_settled_on_destination'))
                log.info('simulated deposit completed in batch %d; hardware motion remains blocked', index)
                return report
            reason = str(result.get('error', ''))
            retryable = (result.get('status') == 'FAILED' and not result.get('trajectory_executed')
                         and 'No executable GraspGen candidate' in reason and validation.exists())
            log.warning('simulation batch %d/%d failed: %s (details: %s)', index, max_attempts, reason, log_file)
            if not retryable:
                break
        raise StageError('real_simulation_preview',
                         f'simulation did not complete after {len(report.data["attempts"])} batch(es); '
                         f'see the attempt reports in {out}')
    return report
