"""Speed scaling must shorten retiming without bypassing the existing dynamic bounds."""
import copy
import unittest
import numpy as np

from pipeline.config_loader import load_environment_config
from pipeline.system.motion import configure_motion, motion_speed
from pipeline.sim.checked_joint_planner import CheckedJointPlanner


class SpeedTests(unittest.TestCase):
    def test_invalid_factors_are_rejected(self):
        for value in (0, -.5, 2.01, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                motion_speed(value)

    def test_stricter_configuration_and_acceleration_jerk_are_preserved(self):
        control = dict(differential_ik=dict(joint_max_velocity_rad_s=[.1]*7,
                                           joint_max_acceleration_rad_s2=1., joint_max_jerk_rad_s3=5.),
                       velocities=dict(cartesian_linear_speed_m_s=.02, cartesian_angular_speed_rad_s=.1))
        original = copy.deepcopy(control)
        configure_motion(control, 2.)
        self.assertEqual(control['differential_ik']['joint_max_velocity_rad_s'], [.1]*7)
        self.assertEqual(control['differential_ik']['joint_max_acceleration_rad_s2'], 1.)
        self.assertEqual(control['differential_ik']['joint_max_jerk_rad_s3'], 5.)
        self.assertEqual(control['velocities'], original['velocities'])

    def test_faster_trajectory_respects_same_acceleration_and_jerk_bounds(self):
        durations = []
        for factor in (1., 2.):
            config = load_environment_config()
            control = config['motion_control']
            configure_motion(control, factor)
            planner = CheckedJointPlanner(None, config, lambda: None)
            path = np.array([np.zeros(7), np.ones(7)*.5])
            samples = planner.retime('Transfer', path, .4 / control['velocities']['cartesian_linear_speed_m_s'])
            self.assertTrue(np.allclose(samples[0], path[0]))
            self.assertTrue(np.allclose(samples[-1], path[-1]))
            limits = np.array([planner.cfg.max_velocity * planner.cfg.velocity_safety_factor,
                               planner.cfg.max_acceleration, planner.cfg.max_jerk])
            self.assertTrue(np.all(planner.peaks <= limits + 1e-6))
            durations.append(planner.phases[0]['duration_s'])
        self.assertLess(durations[1], durations[0])


if __name__ == '__main__':
    unittest.main()
