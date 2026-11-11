"""IK differenziale vincolata: nucleo numpy su un braccio a 7 giunti, senza PyBullet."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.sim import constrained_differential_ik as dik

AXES = [(0, 0, 1), (0, 1, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1)]
LENGTHS = [0.3, 0.0, 0.3, 0.0, 0.25, 0.0, 0.1]
Q0 = np.array([0.3, 0.6, -0.2, -1.0, 0.2, 0.8, 0.0])
Q_MIN, Q_MAX = np.full(7, -2.9), np.full(7, 2.9)


def fk(q):
    R, position = np.eye(3), np.zeros(3)
    for axis, length, angle in zip(AXES, LENGTHS, q):
        R = R @ dik.rot_exp(np.asarray(axis, float) * angle)
        position = position + R @ [0, 0, length]
    return position, R


def config(**options):
    return dik.DifferentialIKConfig.from_config({"motion_control": {"differential_ik": {"enabled": True, **options}}})


def controller(**options):
    return dik.ConstrainedDifferentialIK(fk, Q_MIN, Q_MAX, config(**options), Q0)


class Limits:
    EPS = 1e-6


class ConstrainedDifferentialIKTests(unittest.TestCase):
    def test_disabled_by_default(self):
        for cfg in ({}, {"motion_control": {}}, {"motion_control": {"differential_ik": {"enabled": False}}}):
            self.assertFalse(dik.DifferentialIKConfig.from_config(cfg).enabled)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            config(joint_max_velocity_rad_s=[1, 2])
        with self.assertRaises(ValueError):
            config(velocity_safety_factor=0)
        self.assertTrue(np.all(config(joint_max_jerk_rad_s3=25).max_jerk == 25))

    def test_rotation_log_exp_roundtrip(self):
        for vector in ([0.3, -0.2, 0.5], [0, 0, 0], [0, 0, 3.1], [1.2, 0, 0]):
            np.testing.assert_allclose(dik.rot_log(dik.rot_exp(vector)), vector, atol=1e-6)

    def test_damping_bounds_gain_near_singularity(self):
        J = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1e-6])[:, :6]
        pinv, smallest = dik.dls_pinv(J, 0.05, 0.05)
        self.assertLess(smallest, 0.05)
        self.assertLess(np.abs(pinv).max(), 25)            # la pseudoinversa pura darebbe 1e6

    def test_bounded_solver_matches_unconstrained_and_respects_bounds(self):
        J = np.random.default_rng(1).normal(size=(6, 7))
        task = np.array([0.05, 0, 0, 0, 0, 0])
        wide = np.full(7, 10.0)
        free = dik.bounded_dls(J, task, np.zeros(7), -wide, wide, 0.0, 0.0)
        np.testing.assert_allclose(J @ free, task, atol=1e-9)
        tight = np.full(7, 0.02)
        limited = dik.bounded_dls(J, task, np.zeros(7), -tight, tight, 0.0, 0.0)
        self.assertTrue(np.all(np.abs(limited) <= 0.02 + 1e-12))

    def test_phase_reaches_target_within_all_limits(self):
        c = controller()
        p0, R0 = fk(Q0)
        target = p0 + [0.08, -0.06, 0.05]
        target_R = R0 @ dik.rot_exp([0.0, 0.0, 0.4])
        samples = c.plan_phase("prova", target, target_R)
        position, R = fk(samples[-1])
        self.assertLess(np.linalg.norm(position - target), 0.003)
        self.assertLess(np.linalg.norm(dik.rot_log(target_R @ R.T)), 0.035)
        summary = c.summary()
        self.assertTrue(summary["applied_within_limits"], summary)
        np.testing.assert_allclose(c.dq, 0)

    def test_impossible_speed_is_slowed_not_violated(self):
        c = controller()
        p0, R0 = fk(Q0)
        target = p0 + [0.15, 0.1, -0.1]
        samples = c.plan_phase("veloce", target, R0, nominal_duration=0.15)   # molto piu' rapido dei limiti
        stats = c.phase_stats[0]
        self.assertGreater(stats["time_scale"], 1.5)
        self.assertLess(stats["min_override"], 1.0)
        summary = c.summary()
        self.assertTrue(summary["applied_within_limits"], summary)
        self.assertLess(np.linalg.norm(fk(samples[-1])[0] - target), 0.003)

    def test_motion_starts_and_ends_smoothly(self):
        c = controller()
        p0, R0 = fk(Q0)
        c.plan_phase("morbida", p0 + [0.1, 0, 0], R0)
        applied = np.asarray(c.log["applied"])
        first_step = c.cfg.max_acceleration * dik.CONTROL_DT
        self.assertTrue(np.all(np.abs(applied[0]) <= first_step + 1e-9))
        self.assertLess(np.abs(applied[-1]).max(), 2e-3)

    def test_joint_position_limits_are_never_crossed(self):
        q_max = Q0 + 0.12                          # un giunto lascia pochissimo spazio verso l'alto
        c = dik.ConstrainedDifferentialIK(fk, Q_MIN, q_max, config(), Q0)
        p0, R0 = fk(Q0)
        try:
            c.plan_phase("al limite", p0 + [0.12, 0.12, 0.0], R0, nominal_duration=2.0)
        except ValueError:
            pass                                    # raggiungere il target puo' essere impossibile: i limiti no
        trajectory = np.asarray([Q0] + [Q0 + np.cumsum(np.asarray(c.log["applied"]) * c.dt, axis=0)[i]
                                         for i in range(len(c.log["applied"]))])
        self.assertTrue(np.all(trajectory <= q_max + 1e-9))

    def test_phases_are_continuous_and_plot_is_written(self):
        c = controller()
        p0, R0 = fk(Q0)
        first = c.plan_phase("uno", p0 + [0.05, 0, 0.05], R0)
        second = c.plan_phase("due", p0 + [0.0, 0.05, 0.05], R0)
        np.testing.assert_allclose(first[-1], second[0])
        self.assertEqual([s["phase"] for s in c.phase_stats], ["uno", "due"])
        with tempfile.TemporaryDirectory() as tmp:
            plots = c.save_plot(tmp)
            for path in plots.values():
                self.assertGreater(Path(path).stat().st_size, 1000)
            self.assertEqual(set(plots), {"velocity", "dynamics"})


if __name__ == "__main__":
    unittest.main()
