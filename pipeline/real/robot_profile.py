"""Bind real-pipeline GraspGen runs to the configured iiwa 7 and Robotiq model.

This validates model assets and records nominal conventions. It does not read
robot state, calibrate the mounting, plan physical motion or command hardware.
"""
from __future__ import annotations

import logging

from pipeline.common import paths
from pipeline.common.report import stage_run
from pipeline.config_loader import load_environment_config
from pipeline.graspgen.config import load_config
from pipeline.robot_model import RobotModel

log = logging.getLogger('real_robot_profile')


def run(*, config_path=None, graspgen_config=None, out_dir=None):
    out = paths.output_dir(out_dir)
    with stage_run('real_robot_profile', out / 'real_robot_profile.json',
                   inputs={'config': config_path, 'graspgen_config': graspgen_config}, logger=log) as report:
        config = load_environment_config(config_path)
        model = RobotModel.from_config(config)
        if model.model_id != 'iiwa7_r800':
            raise ValueError('real pipeline requires the configured iiwa7_r800 model')
        base = config['environment']['kuka_base']['frame_id']
        if base != model.settings['base_link']:
            raise ValueError(f'robot base frame {base!r} differs from model base {model.settings["base_link"]!r}')
        cfg = load_config(graspgen_config)
        if cfg['model_gripper'] != 'robotiq_2f_140':
            raise ValueError('real pipeline requires the Robotiq 2F-140 GraspGen checkpoint')
        if cfg['robot']['motion_enabled']:
            raise ValueError('real pipeline has no validated physical executor; motion_enabled must be false')
        description = model.report()
        description.update(
            physical_gripper_model='robotiq_2f_140',
            graspgen_checkpoint=cfg['checkpoint_config'],
            T_graspgen_gripper_tcp=cfg['robot']['T_gripper_tcp'],
            graspgen_tcp_calibrated=cfg['robot']['tcp_transform_verified'],
            flange_gripper_mount_calibrated=False,
            note='GraspGen base-to-TCP convention; flange-to-gripper mounting is not established by this matrix')
        report.data = {'robot_description': description, 'hardware_motion_allowed': False,
                       'physical_grasp_validated': False,
                       'limitations': ['URDF joint limits are model limits, not measured controller limits',
                                       'flange-to-gripper mounting and TCP require laboratory validation',
                                       'table and tray configuration is nominal',
                                       'no physical KUKA or gripper executor is connected']}
        report.warn(log, 'model identity recorded; mounting, TCP and physical execution remain unverified')
    return report
