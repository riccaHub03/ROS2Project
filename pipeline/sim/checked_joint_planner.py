"""CPU planning for the native simulation, with independent collision checks.

Uses the tutor's seeded IK / FK acceptance criteria. Free-space transfers use
joint-space search rather than forcing a straight Cartesian line from HOME.
This is not cuRobo or a ROS controller. No physics is advanced by this module.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.interpolate import make_interp_spline

from pipeline.sim.constrained_differential_ik import CONTROL_DT, DifferentialIKConfig


class CheckedJointPlanner:
    def __init__(self, cell, config, heartbeat):
        self.cell = cell
        self.cfg = DifferentialIKConfig.from_config(config)
        self.heartbeat = heartbeat
        self.rng = np.random.default_rng(42)
        self.phases = []
        self.peaks = np.zeros((3, 7))

    def check(self, q, clearance):
        self.heartbeat()
        if not np.isfinite(q).all() or self.cell.joint_margin(q) < .05:
            raise ValueError("Joint position outside planning margin (0.05 rad)")
        self.cell.set_joints(q)
        clearance()

    def solve(self, position, quaternion, seed, clearance, *, multistart=False):
        limits = np.asarray(self.cell.limits)
        seeds = [np.asarray(seed)]
        if multistart:
            seeds += [np.zeros(7)] + [self.rng.uniform(limits[:, 0] + .1,
                                                     limits[:, 1] - .1) for _ in range(6)]
        reasons = []
        for initial in seeds:
            self.heartbeat()
            solution = self.cell.solve_ik(position, quaternion, initial)
            if not solution.ok:
                reasons.append("IK pose tolerance / joint limits")
                continue
            q = np.asarray(solution.q)
            try:
                self.check(q, clearance)
            except ValueError as exc:
                reasons.append(str(exc))
                continue
            return q
        raise ValueError("No collision-free IK: " + (reasons[-1] if reasons else "no solution"))

    def cartesian(self, start, position, quaternion, clearance):
        self.cell.set_joints(start)
        origin, orientation = self.cell.tcp_pose()
        from scipy.spatial.transform import Rotation, Slerp
        rotations = Rotation.from_quat([orientation, quaternion])
        angle = (rotations[0].inv() * rotations[1]).magnitude()
        count = max(1, int(np.ceil(np.linalg.norm(np.asarray(position)-origin) / .004)),
                    int(np.ceil(angle / .03)))
        if count > 2000:
            raise ValueError("Cartesian path exceeds tutor's 2000 sample bound")
        interpolator = Slerp([0, 1], rotations)
        path = [np.asarray(start)]
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            point = np.asarray(origin) * (1-fraction) + np.asarray(position) * fraction
            q = self.solve(point, interpolator(fraction).as_quat(), path[-1], clearance)
            if np.max(np.abs(q-path[-1])) > .25:
                raise ValueError("Cartesian IK branch jump exceeds 0.25 rad")
            path.append(q)
        return np.asarray(path)

    def transit(self, start, goal, clearance):
        """Checked direct connection, then bounded bidirectional RRT-Connect."""
        start, goal = np.asarray(start), np.asarray(goal)

        def edge(a, b):
            for q in np.linspace(a, b, max(2, int(np.ceil(np.max(np.abs(b-a))/.025)) + 1)):
                try:
                    self.check(q, clearance)
                except ValueError:
                    return False
            return True

        self.check(start, clearance)
        self.check(goal, clearance)
        if edge(start, goal):
            return np.asarray([start, goal])
        trees = [([start], [-1]), ([goal], [-1])]
        limits = np.asarray(self.cell.limits)
        stop = time.monotonic() + 4.

        def extend(tree, target):
            nodes, parents = tree
            nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes)-target, axis=1)))
            delta = target - nodes[nearest]
            q = nodes[nearest] + delta * min(1., .18 / max(np.linalg.norm(delta), 1e-12))
            if not edge(nodes[nearest], q):
                return None
            nodes.append(q)
            parents.append(nearest)
            return len(nodes)-1

        def branch(tree, index):
            result = []
            while index >= 0:
                result.append(tree[0][index])
                index = tree[1][index]
            return result[::-1]

        for iteration in range(600):
            self.heartbeat()
            if time.monotonic() > stop:
                break
            side = iteration % 2
            first, second = trees[side], trees[1-side]
            target = second[0][0] if iteration % 5 == 0 else self.rng.uniform(
                limits[:, 0]+.05, limits[:, 1]-.05)
            a = extend(first, target)
            if a is None:
                continue
            for _ in range(50):
                b = extend(second, first[0][a])
                if b is None:
                    break
                if np.linalg.norm(second[0][b]-first[0][a]) < 1e-8:
                    path = branch(first, a) + branch(second, b)[-2::-1]
                    return np.asarray(path if side == 0 else path[::-1])
        raise ValueError("No collision-free joint transit within search budget")

    def retime(self, label, path, minimum_duration=0.6):
        """Quintic spline with zero endpoint velocity/acceleration; bounds on samples.

        Spline overshoot is NOT assumed safe: the caller checks the resulting
        trajectory for joint limits, collisions and endpoint FK before execution.
        """
        path = np.asarray(path)
        keep = np.r_[True, np.max(np.abs(np.diff(path, axis=0)), axis=1) > 1e-9]
        path = path[keep]
        if len(path) == 1:
            path = np.repeat(path, 2, axis=0)
        distances = np.maximum(np.linalg.norm(np.diff(path, axis=0), axis=1), 1e-8)
        u = np.r_[0., np.cumsum(distances)]
        u /= u[-1]
        zero = np.zeros(7)
        spline = make_interp_spline(u, path, k=5, bc_type=([(1, zero), (2, zero)],
                                                        [(1, zero), (2, zero)]))
        limits = np.asarray([self.cfg.max_velocity * self.cfg.velocity_safety_factor,
                             self.cfg.max_acceleration, self.cfg.max_jerk])
        probe = np.unique(np.r_[np.linspace(0, 1, 2001), u])
        duration = max(minimum_duration, .6)
        for order in (1, 2, 3):
            peak = np.max(np.abs(spline(probe, nu=order)), axis=0)
            duration = max(duration, float(np.max(peak/limits[order-1]))**(1/order) * 1.1)
        for _ in range(5):
            self.heartbeat()
            count = int(np.ceil(duration / CONTROL_DT)) + 1
            if count > 120 / CONTROL_DT:
                raise ValueError("Phase duration exceeds 120 seconds")
            samples = spline(np.linspace(0, 1, count))
            # Include the stationary boundary on each side to check joins too.
            padded = np.vstack([np.repeat(samples[:1], 3, axis=0), samples,
                                np.repeat(samples[-1:], 3, axis=0)])
            peaks = np.asarray([np.max(np.abs(np.diff(padded, n=n, axis=0)), axis=0)
                                / CONTROL_DT**n for n in (1, 2, 3)])
            if np.all(peaks <= limits + 1e-6):
                self.peaks = np.maximum(self.peaks, peaks)
                self.phases.append({"phase": label, "duration_s": (count-1)*CONTROL_DT,
                                    "ik_waypoints": len(path)})
                return samples
            duration *= 1.3
        raise ValueError("Trajectory dynamics could not be bounded")

    def summary(self):
        return {"enabled": True, "method": "seeded_ik_rrt_connect_quintic_retiming",
                "applied_within_limits": True, "phases": self.phases,
                "peak_applied_velocity_rad_s": self.peaks[0].tolist(),
                "peak_applied_acceleration_rad_s2": self.peaks[1].tolist(),
                "peak_applied_jerk_rad_s3": self.peaks[2].tolist()}

    def save_plot(self, directory):
        return {}
