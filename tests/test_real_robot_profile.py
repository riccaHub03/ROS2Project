"""The real pipeline binds grasp inference to the robot without enabling motion."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.common.report import StageError, StageReport
from pipeline.real import pipeline, robot_profile


class RealRobotProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.robot = SimpleNamespace(model_id='iiwa7_r800', settings={'base_link': 'lbr_link_0'},
                                     report=lambda: {'model': 'iiwa7_r800', 'model_sha256': 'fixture'})

    def test_profile_records_gripper_and_blocks_hardware(self):
        with patch.object(robot_profile.RobotModel, 'from_config', return_value=self.robot):
            report = robot_profile.run(out_dir=self.tmp.name)
        data = report.data
        self.assertEqual(data['robot_description']['physical_gripper_model'], 'robotiq_2f_140')
        self.assertFalse(data['robot_description']['flange_gripper_mount_calibrated'])
        self.assertFalse(data['hardware_motion_allowed'])
        self.assertFalse(data['physical_grasp_validated'])
        self.assertTrue((Path(self.tmp.name) / 'real_robot_profile.json').is_file())

    def test_wrong_robot_or_base_and_enabled_motion_are_rejected(self):
        for robot in (SimpleNamespace(model_id='pybullet_iiwa14'),
                      SimpleNamespace(model_id='iiwa7_r800', settings={'base_link': 'wrong'})):
            with self.subTest(robot=robot), patch.object(robot_profile.RobotModel, 'from_config', return_value=robot):
                with self.assertRaises(StageError):
                    robot_profile.run(out_dir=self.tmp.name)
        from pipeline.graspgen.config import load_config
        cfg = load_config()
        cfg['robot']['motion_enabled'] = True
        with patch.object(robot_profile.RobotModel, 'from_config', return_value=self.robot), \
                patch.object(robot_profile, 'load_config', return_value=cfg), self.assertRaises(StageError):
            robot_profile.run(out_dir=self.tmp.name)

    def test_real_pipeline_defaults_to_graspgen_and_passes_robot_provenance(self):
        args = pipeline.build_parser().parse_args(['--out-dir', self.tmp.name])
        self.assertEqual(args.grasp_backend, 'graspgen')
        profile = StageReport('real_robot_profile', data={'robot_description': {'model': 'iiwa7_r800'}})
        with patch.object(robot_profile, 'run', return_value=profile), \
                patch.object(pipeline.grasp, 'run', return_value=StageReport('grasp')) as grasp:
            next(step for step in pipeline.build_steps(args) if step.name == 'grasp').run()
        self.assertEqual(grasp.call_args.kwargs['robot_description']['model'], 'iiwa7_r800')


if __name__ == '__main__':
    unittest.main()
