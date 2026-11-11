"""Grasp candidates and feasibility validation: cell model, candidate generation, IK/limits/collisions, stage output."""

import copy
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pybullet as p
import trimesh
import yaml

from pipeline import feasibility, grasp, plan
from pipeline.common import paths
from pipeline.common import waypoints as wp
from pipeline.common.ply import write_ply
from pipeline.common.report import StageReport
from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import RobotBaseFrame
from pipeline.robot_cell import RobotCell
from pipeline.sim import robot_scene
from pipeline.sim.robot_motion_preview import MotionPreview
from pipeline.sim.robot_scene import HOME

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/surgical_environment.yaml"
CONFIG = load_environment_config(CONFIG_PATH)
DOWN = p.getQuaternionFromEuler([math.pi, 0, 0])
SUPPORT_TOP = 0.09          # top of the trays (0.07 + 0.04 / 2)


def flat_tool_cloud(center=(0.60, -0.42), yaw=0.3, length=0.14, width=0.012, thickness=0.004, step=0.002):
    """Top surface of a thin instrument lying on tray_2 (robot base frame = world frame)."""
    xs = np.arange(-length / 2, length / 2 + 1e-9, step)
    ys = np.arange(-width / 2, width / 2 + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys)
    local = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rotation.T + np.array([center[0], center[1], SUPPORT_TOP + 0.001 + thickness])


def make_context(points, config=CONFIG, base=None, source="tray_2", destination="tray_1", params=None):
    base = base or RobotBaseFrame()
    a = grasp.analyze(points, base)
    params = params or feasibility.load_params(config)
    candidates = grasp.generate_candidates(points, a, base, SUPPORT_TOP, params)
    ctx = feasibility.ValidationContext(
        config=config, base_frame=base, source_key=source, destination_key=destination,
        tool_box=grasp.tool_box(points, a, base, SUPPORT_TOP), support_top_world_z=SUPPORT_TOP,
        max_stroke=grasp.MAX_GRIPPER_STROKE_M, params=params)
    return a, candidates, ctx


class RobotCellTests(unittest.TestCase):
    def setUp(self):
        self.cell = RobotCell(CONFIG, RobotBaseFrame())

    def tearDown(self):
        self.cell.close()

    def test_home_pose_is_collision_free(self):
        self.cell.set_joints(HOME)
        self.assertTrue(self.cell.clearance(width=0.08).clear)

    def test_ik_converges_below_a_millimetre(self):
        target = [0.60, -0.42, 0.10]
        solution = self.cell.solve_ik(target, DOWN, feasibility.ik_seeds(target, HOME)[1])
        self.assertTrue(solution.ok)
        self.assertLess(solution.position_error_m, 0.001)
        np.testing.assert_allclose(self.cell.tcp_pose()[0], target, atol=0.001)

    def test_unreachable_pose_is_reported_not_hidden(self):
        solution = self.cell.solve_ik([1.3, 0.0, 0.1], DOWN, HOME)
        self.assertFalse(solution.converged)

    def test_fingers_below_the_tray_surface_collide(self):
        target = [0.60, -0.42, 0.06]
        self.cell.solve_ik(target, DOWN, feasibility.ik_seeds(target, HOME)[1])
        violations = self.cell.clearance(width=0.044).violations
        self.assertTrue(any("tray_2" in v for v in violations), violations)

    def test_fingertips_just_above_the_tray_need_the_relaxed_margin(self):
        target = [0.60, -0.42, 0.091]
        solution = self.cell.solve_ik(target, DOWN, feasibility.ik_seeds(target, HOME)[1])
        self.assertTrue(solution.ok)
        self.assertTrue(self.cell.clearance(width=0.044).violations)                # 1 mm: not a normal clearance
        relaxed = self.cell.clearance(width=0.044, relaxed=frozenset({("fingers", "tray_2")}))
        self.assertTrue(relaxed.clear, relaxed.violations)

    def test_a_moved_and_rotated_base_is_modelled_in_its_own_frame(self):
        base = RobotBaseFrame([0.04, 0.01, 0.0], [0.0, 0.0, 0.3])           # still clear of the table edge (x = 0.175)
        with RobotCell(CONFIG, base) as cell:
            position, orientation = base.pose_world_to_base([0.60, -0.42, 0.15], DOWN)
            solution = cell.solve_ik(position, orientation, feasibility.ik_seeds(position, HOME)[1])
            self.assertTrue(solution.ok)
            self.assertTrue(cell.clearance(width=0.08).clear)
            below, orientation_below = base.pose_world_to_base([0.60, -0.42, 0.06], DOWN)
            cell.solve_ik(below, orientation_below, solution.q)
            self.assertTrue(any("tray_2" in v for v in cell.clearance(width=0.044).violations))


class CandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        self.points = flat_tool_cloud()
        self.a, self.candidates, self.ctx = make_context(self.points)

    def test_five_positions_and_both_jaw_directions(self):
        self.assertEqual(len(self.candidates), 10)
        positions = sorted({round(c.rel_position, 3) for c in self.candidates})
        self.assertEqual(len(positions), 5)
        self.assertGreaterEqual(positions[0], 0.35)
        self.assertLessEqual(positions[-1], 0.65)
        for std, rev in zip(self.candidates[::2], self.candidates[1::2]):
            self.assertFalse(std.jaws_reversed)
            self.assertTrue(rev.jaws_reversed)
            np.testing.assert_allclose(std.position, rev.position)
            np.testing.assert_allclose(std.closing, -rev.closing, atol=1e-9)

    def test_vertical_approach_unit_orientation_and_jaw_opening(self):
        for cand in self.candidates:
            self.assertAlmostEqual(float(np.linalg.norm(cand.quaternion)), 1.0, places=6)
            np.testing.assert_allclose(cand.approach, [0, 0, -1], atol=1e-9)
            self.assertAlmostEqual(float(cand.closing[2]), 0.0, places=9)             # jaws close horizontally
            self.assertAlmostEqual(cand.opening, cand.local_width + grasp.CLEARANCE_M, places=9)
            self.assertAlmostEqual(cand.local_width, 0.012, delta=0.0025)

    def test_fingertips_reach_the_instrument_without_touching_the_tray(self):
        for cand in self.candidates:
            height = cand.position[2] - SUPPORT_TOP
            self.assertGreaterEqual(height, feasibility.DEFAULT_PARAMS["tip_clearance_m"] - 1e-9)
            self.assertLess(height, 0.005)
            self.assertAlmostEqual(cand.pinch_depth, 0.005 - height, places=6)         # top 5 mm above the tray

    def test_central_candidates_rank_higher(self):
        best = max(self.candidates, key=lambda c: c.quality)
        self.assertAlmostEqual(best.rel_position, 0.5, places=2)
        outer = [c for c in self.candidates if abs(c.rel_position - 0.5) > 0.1]
        self.assertTrue(all(c.quality < best.quality for c in outer))

    def test_the_tool_box_follows_the_instrument(self):
        box = self.ctx.tool_box
        self.assertAlmostEqual(box["half_extents"][0] * 2, 0.14, delta=0.004)
        self.assertAlmostEqual(box["half_extents"][2] * 2, 0.004, delta=0.0015)
        self.assertAlmostEqual(box["position"][2], SUPPORT_TOP + 0.001 + box["half_extents"][2], delta=0.0005)


class FeasibilityTests(unittest.TestCase):
    def test_a_flat_instrument_on_tray_two_has_feasible_candidates(self):
        _, candidates, ctx = make_context(flat_tool_cloud())
        result = feasibility.evaluate(candidates, ctx)
        summary = result.summary
        self.assertTrue(summary["ik_checked"] and summary["collision_checked"])
        self.assertGreater(summary["feasible"], 0)
        top = result.candidates[0]
        self.assertTrue(top["feasible"] and top["selected"])
        self.assertEqual(result.selected_id, top["id"])
        self.assertEqual(sum(e["selected"] for e in result.candidates), 1)
        self.assertEqual([e["rank"] for e in result.candidates], list(range(1, len(candidates) + 1)))
        for key in ("contact", "orientation", "kinematics", "collision", "score", "failure_reasons", "pose_base_frame"):
            self.assertIn(key, top)
        self.assertLess(top["kinematics"]["max_ik_error_mm"], 1.0)
        self.assertGreaterEqual(top["kinematics"]["min_joint_margin_rad"], 0.03)
        self.assertTrue(summary["limitations"])
        phases = [phase["phase"] for phase in top["kinematics"]["phases"]]
        self.assertEqual(phases, [wp.APPROACH, wp.DESCEND, wp.LIFT, wp.TRANSFER, wp.PLACE])

    def test_the_best_feasible_candidate_wins_over_better_geometry_that_fails(self):
        _, candidates, ctx = make_context(flat_tool_cloud())
        result = feasibility.evaluate(candidates, ctx)
        feasible = [e for e in result.candidates if e["feasible"]]
        self.assertEqual(result.selected_id, max(feasible, key=lambda e: e["score"])["id"])
        self.assertTrue(all(e["score"] == 0.0 for e in result.candidates if not e["feasible"]))

    def test_an_instrument_out_of_reach_is_rejected_with_ik_reasons(self):
        _, candidates, ctx = make_context(flat_tool_cloud(center=(0.95, -0.42)))
        result = feasibility.evaluate(candidates, ctx)
        self.assertEqual(result.summary["feasible"], 0)
        self.assertIsNone(result.selected_id)
        self.assertTrue(all(any("IK" in r or "limit" in r for r in e["failure_reasons"]) for e in result.candidates))

    def test_a_wall_between_the_trays_blocks_the_transfer(self):
        config = copy.deepcopy(CONFIG)
        config["environment"]["trays"]["wall"] = {"name": "wall", "position_xyz_m": [0.60, 0.0, 0.30],
                                                  "dimensions_xyz_m": [0.40, 0.04, 0.60]}
        _, candidates, ctx = make_context(flat_tool_cloud(), config=config)
        result = feasibility.evaluate(candidates, ctx)
        self.assertEqual(result.summary["feasible"], 0)
        reasons = [r for e in result.candidates for r in e["failure_reasons"]]
        self.assertTrue(any("wall" in r and wp.TRANSFER in r for r in reasons), reasons)
        collided = [e for e in result.candidates if e["collision"]["passed"] is False]
        self.assertTrue(collided)

    def test_mask_noise_tilting_the_cloud_slightly_does_not_reject_a_resting_instrument(self):
        """A 3 deg PCA tilt (SAM 3 mask edges) used to tilt the carried box and sink a corner into the tray."""
        points = flat_tool_cloud()
        centre = points.mean(axis=0)
        tilt = math.radians(3)
        axis = np.array([math.cos(0.3), math.sin(0.3), 0.0])
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + math.sin(tilt) * k + (1 - math.cos(tilt)) * (k @ k)
        noisy = (points - centre) @ rotation.T + centre
        _, candidates, ctx = make_context(noisy)
        np.testing.assert_allclose(feasibility._quat_to_matrix(ctx.tool_box["quaternion"])[:, 2], [0, 0, 1], atol=1e-9)
        result = feasibility.evaluate(candidates, ctx)
        self.assertGreater(result.summary["feasible"], 0, [e["failure_reasons"] for e in result.candidates][:2])

    def test_a_tilted_instrument_is_not_suited_to_a_top_down_pinch(self):
        points = flat_tool_cloud()
        centre = points.mean(axis=0)
        tilt = math.radians(60)                      # roll about the long axis (the cloud is yawed by 0.3 rad)
        axis = np.array([math.cos(0.3), math.sin(0.3), 0.0])
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + math.sin(tilt) * k + (1 - math.cos(tilt)) * (k @ k)
        tilted = (points - centre) @ rotation.T + centre
        _, candidates, ctx = make_context(tilted)
        result = feasibility.evaluate(candidates, ctx)
        self.assertEqual(result.summary["feasible"], 0)
        self.assertTrue(all(any("tilted" in r for r in e["failure_reasons"]) for e in result.candidates))
        self.assertTrue(all(e["kinematics"].get("skipped") for e in result.candidates))   # no IK spent on them

    def test_an_instrument_wider_than_the_stroke_is_rejected_by_the_contact_check(self):
        _, candidates, ctx = make_context(flat_tool_cloud(width=0.09))
        result = feasibility.evaluate(candidates, ctx)
        self.assertEqual(result.summary["feasible"], 0)
        self.assertTrue(all("exceeds the gripper stroke" in e["failure_reasons"][0] for e in result.candidates))

    def test_jaw_direction_matters_for_the_joint_limits(self):
        """Reversing the jaws turns the wrist by 180 deg: A7 may run out of range during the transfer."""
        differing, covered = 0, 0
        for yaw in (0.0, 0.4, 1.2, -1.2, 2.5, -2.5):
            _, candidates, ctx = make_context(flat_tool_cloud(yaw=yaw))
            centre = [c for c in candidates if abs(c.rel_position - 0.5) < 0.01]
            result = feasibility.evaluate(centre, ctx)
            outcome = {e["jaws_reversed"]: e["feasible"] for e in result.candidates}
            differing += outcome[False] != outcome[True]
            covered += outcome[False] or outcome[True]
            for entry in result.candidates:
                if not entry["feasible"]:
                    self.assertTrue(any("joint" in r or "IK" in r or "collision" in r.lower() or "<->" in r
                                        for r in entry["failure_reasons"]), entry["failure_reasons"])
        self.assertGreater(differing, 0, "reversed jaws never changed the outcome")
        self.assertGreaterEqual(covered, 4, "one of the two jaw directions should usually fit")

    def test_the_check_works_in_a_moved_and_rotated_base_frame(self):
        base = RobotBaseFrame([0.04, 0.01, 0.0], [0.0, 0.0, 0.3])
        world_points = flat_tool_cloud()
        points = np.asarray(base.world_to_base(world_points))
        _, candidates, ctx = make_context(points, base=base)
        result = feasibility.evaluate(candidates, ctx)
        self.assertGreater(result.summary["feasible"], 0)

    def test_a_base_that_overlaps_the_table_is_rejected_by_the_collision_check(self):
        base = RobotBaseFrame([0.10, -0.05, 0.0], [0.0, 0.0, 0.0])         # the base link would sink into the table
        points = np.asarray(base.world_to_base(flat_tool_cloud()))
        _, candidates, ctx = make_context(points, base=base)
        result = feasibility.evaluate(candidates, ctx)
        self.assertEqual(result.summary["feasible"], 0)
        self.assertTrue(any("table" in r for e in result.candidates for r in e["failure_reasons"]))

    def test_geometry_only_never_claims_feasibility(self):
        _, candidates, ctx = make_context(flat_tool_cloud())
        result = feasibility.geometry_only(candidates, ctx, "disabled")
        self.assertIsNone(result.selected_id)
        self.assertTrue(all(e["feasible"] is None and not e["selected"] for e in result.candidates))
        self.assertFalse(result.summary["ik_checked"] or result.summary["collision_checked"])


class GraspStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = patch.dict("os.environ", {paths.ROOT_ENV: str(self.tmp)})
        self.env.start()
        self.out = paths.output_dir()
        plan.run("Passami il bisturi dal vassoio 2 al vassoio 1", offline=True, out_dir=str(self.out))

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _deprojection(self, points):
        ply = self.out / "cloud.ply"
        write_ply(ply, points)
        StageReport("sim_deprojection", artifacts={"pointcloud_ply": str(ply)},
                    data={"motion_safe": True, "coordinate_frame_id": "lbr_link_0",
                          "robot_base_pose_world": RobotBaseFrame().to_dict()}).save(self.out / paths.SIM_DEPROJECTION)

    def _config(self, **validation):
        config = copy.deepcopy(CONFIG)
        config.setdefault("grasp_validation", {}).update(validation)
        path = self.tmp / "env.yaml"
        path.write_text(yaml.safe_dump(config))
        return str(path)

    def _run(self, config_path=None):
        return grasp.run(out_dir=str(self.out), config_path=config_path)

    def test_stage_output_lists_candidates_with_their_checks(self):
        self._deprojection(flat_tool_cloud())
        report = self._run()
        data = report.data
        self.assertEqual(report.status, "COMPLETED")
        self.assertEqual(data["readiness"], grasp.READY, data["readiness_blockers"])
        self.assertEqual(len(data["grasp_candidates"]), 10)
        self.assertEqual(data["designated_grasp_geometry"]["candidate_id"], data["selected_candidate_id"])
        self.assertTrue(data["feasibility"]["ik_checked"] and data["feasibility"]["collision_checked"])
        self.assertEqual((data["source_tray"], data["destination_tray"]), ("tray_2", "tray_1"))
        chosen = next(c for c in data["grasp_candidates"] if c["selected"])
        np.testing.assert_allclose(data["designated_grasp_geometry"]["grasp_centroid_base_frame_xyz_m"],
                                   chosen["pose_base_frame"]["position_xyz_m"])
        self.assertTrue(Path(report.artifacts["candidates_png"]).exists())

    def test_without_a_feasible_candidate_the_grasp_is_not_ready(self):
        self._deprojection(flat_tool_cloud(center=(0.95, -0.42)))
        data = self._run().data
        self.assertEqual(data["readiness"], grasp.NOT_READY)
        self.assertIsNone(data["selected_candidate_id"])
        self.assertTrue(data["readiness_blockers"][0].startswith("none of the 10 grasp candidates is feasible"))
        self.assertIn("designated_grasp_geometry", data)              # still visualisable

    def test_disabling_the_check_cannot_clear_a_grasp_when_it_is_required(self):
        self._deprojection(flat_tool_cloud())
        data = self._run(self._config(enabled=False)).data
        self.assertFalse(data["feasibility"]["ik_checked"])
        self.assertTrue(all(c["feasible"] is None for c in data["grasp_candidates"]))
        self.assertEqual(data["readiness"], grasp.NOT_READY)
        self.assertTrue(any("was not verified" in b for b in data["readiness_blockers"]))
        relaxed = self._run(self._config(enabled=False, require_feasibility=False)).data
        self.assertEqual(relaxed["readiness"], grasp.READY)           # old behaviour: reachable annulus only

    def test_a_failing_cell_model_is_reported_as_unverified(self):
        self._deprojection(flat_tool_cloud())
        with patch.object(feasibility, "evaluate", side_effect=RuntimeError("no pybullet")):
            report = self._run()
        self.assertEqual(report.data["readiness"], grasp.NOT_READY)
        self.assertTrue(any("could not run" in w for w in report.warnings))
        self.assertTrue(any("was not verified" in b for b in report.data["readiness_blockers"]))


class SceneIntegrationTests(unittest.TestCase):
    """What the check validates is what the preview executes, and the window can show the candidates."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.env = patch.dict("os.environ", {paths.ROOT_ENV: str(root)})
        self.env.start()
        mesh = root / "instrument.stl"
        trimesh.creation.box([0.14, 0.012, 0.004]).export(mesh)
        config = copy.deepcopy(CONFIG)
        config["surgical_instrument"]["stl_path"] = str(mesh)
        self.config_path = root / "env.yaml"
        self.config_path.write_text(yaml.safe_dump(config))
        self.plan = {"target_instrument": "Scalpel", "source_tray": "tray_2", "destination_tray": "tray_1"}
        self.scene = robot_scene.RobotScene(config, self.plan, seed=42)
        yaw = p.getEulerFromQuaternion(self.scene.tool_initial_orn)[2]
        x, y, _ = self.scene.tool_initial_pos
        out = paths.output_dir()
        plan.run("Passami il bisturi dal vassoio 2 al vassoio 1", offline=True, out_dir=str(out))
        ply = out / "cloud.ply"
        write_ply(ply, flat_tool_cloud(center=(x, y), yaw=yaw))
        StageReport("sim_deprojection", artifacts={"pointcloud_ply": str(ply)},
                    data={"motion_safe": True, "coordinate_frame_id": "lbr_link_0",
                          "robot_base_pose_world": self.scene.base_frame.to_dict()}).save(out / paths.SIM_DEPROJECTION)
        self.report = grasp.run(out_dir=str(out), config_path=str(self.config_path))

    def tearDown(self):
        self.scene.close()
        self.env.stop()
        self.temp.cleanup()

    def test_the_preview_walks_the_targets_the_check_validated(self):
        data = self.report.data
        self.assertEqual(data["readiness"], grasp.READY, data["readiness_blockers"])
        motion = MotionPreview(self.scene, data)
        point = np.asarray(self.scene.base_frame.base_to_world(
            data["designated_grasp_geometry"]["grasp_centroid_base_frame_xyz_m"]))
        expected = wp.pick_and_place_targets(
            point, destination_tray=self.scene.trays[self.scene.destination], tool_bottom_z=self.scene.tool_bottom,
            motion=self.scene.config.get("motion_control", {}), base_frame=self.scene.base_frame)
        self.assertEqual([w["phase"] for w in motion.waypoints], [t.label for t in expected])
        for waypoint, target in zip(motion.waypoints, expected):
            np.testing.assert_allclose(waypoint["tcp_position_m"], target.position, atol=1e-9)

    def test_the_selected_candidate_executes_to_completion(self):
        motion = MotionPreview(self.scene, self.report.data)
        for _ in range(30000):
            motion.tick()
            self.scene.step()
            if motion.finished:
                break
        self.assertTrue(motion.finished)
        self.assertEqual(motion.report()["status"], "COMPLETED")

    def test_the_window_draws_every_candidate_and_the_selected_grasp(self):
        before = len(self.scene.markers)
        self.scene.show_grasp(self.report.data)
        self.assertEqual(len(self.scene.markers), before + 1)          # the selected grasp marker
        broken = copy.deepcopy(self.report.data)
        for candidate in broken["grasp_candidates"]:
            candidate["feasible"] = None                              # unverified candidates are drawn too
        self.scene.show_grasp(broken)


if __name__ == "__main__":
    unittest.main()
