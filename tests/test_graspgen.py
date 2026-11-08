"""Mac contract tests: real official client, simulated ZMQ GPU server, no model import."""
import json
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import msgpack
import msgpack_numpy
import numpy as np
import yaml
import zmq

from pipeline import grasp
from pipeline.common.ply import read_ply, write_ply
from pipeline.common.report import StageError, StageReport
from pipeline.frame_transformer import RobotBaseFrame, matrix_from_quaternion
from pipeline.graspgen import backend, bundle, import_cloud
from pipeline.graspgen.config import DEFAULT_CONFIG, PINNED_REVISION, ROOT, load_config, rigid_transform
from pipeline.graspgen.prepare import build_dockerfile, check_models, check_source

msgpack_numpy.patch()


def metadata():
    return {"gripper_name": "robotiq_2f_140", "model_name": "diffusion-discriminator",
            "gripper_config": "/models/checkpoints/graspgen_robotiq_2f_140.yml",
            "ros2project": {"revision": PINNED_REVISION, "planner": "diffusion",
                            "config_sha256": "a" * 64, "generator_sha256": "b" * 64,
                            "discriminator_sha256": "c" * 64}}


class FakeGPU:
    """Wire-compatible REP server. Only predictions are fake; transport/client are real."""
    def __init__(self, response=None, meta=None, infer_delay=0):
        poses = np.repeat(np.eye(4)[None], 3, axis=0).astype(np.float32)
        poses[:, :3, 3] = [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [-0.01, 0, 0]]
        self.response = response if response is not None else {"grasps": poses, "confidences": np.array([0.2, 0.9, 0.6], np.float32)}
        self.meta = meta if meta is not None else metadata()
        self.delay = infer_delay
        self.requests = []
        self.stop = threading.Event()
        self.ready = queue.Queue()

    def __enter__(self):
        def serve():
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REP)
            sock.setsockopt(zmq.LINGER, 0)
            try:
                port = sock.bind_to_random_port("tcp://127.0.0.1")
                self.ready.put(port)
                while not self.stop.is_set():
                    if not sock.poll(30):
                        continue
                    req = msgpack.unpackb(sock.recv(), raw=False)
                    self.requests.append(req)
                    if req["action"] == "metadata":
                        reply = self.meta
                    else:
                        time.sleep(self.delay)
                        reply = self.response
                    sock.send(msgpack.packb(reply, use_bin_type=True))
            except Exception as exc:
                self.ready.put(exc)
            finally:
                sock.close()
                ctx.term()
        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        value = self.ready.get(timeout=3)
        if isinstance(value, Exception):
            raise value
        self.port = value
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("FakeGPU did not stop")


class GraspGenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.base = RobotBaseFrame([0.3, -0.25, 0.1], [0.17, -0.23, 0.61])
        self.points = np.random.default_rng(4).uniform([-0.07, -0.01, -0.003], [0.07, 0.01, 0.003], (100, 3))
        self.points += [0.5, -0.3, 0.2]
        self.dep = self.out / "sim_deprojection.json"
        write_ply(self.out / "cloud.ply", self.points)
        self.report = StageReport("sim_deprojection", artifacts={"pointcloud_ply": str(self.out / "cloud.ply")},
                                  data={"pointcloud_units": "m", "coordinate_frame_id": self.base.frame_id,
                                        "robot_base_pose_world": self.base.to_dict(),
                                        "geometry_verified_for_motion": False})
        self.report.save(self.dep)
        self.kw = dict(deprojection_path=str(self.dep), out_dir=str(self.out), backend="graspgen")

    def run_server(self, server, **kwargs):
        return grasp.run(**self.kw, graspgen_port=server.port, graspgen_timeout_ms=1000, **kwargs)

    def test_official_client_geometry_global_ranking_and_reports(self):
        with FakeGPU() as server:
            report = self.run_server(server)
        request = server.requests[1]
        sent = request["point_cloud"]
        cloud, _ = read_ply(self.out / "cloud.ply")
        center = cloud.mean(axis=0)
        np.testing.assert_allclose(sent, cloud-center, atol=1e-8)
        np.testing.assert_allclose(sent.mean(axis=0), 0, atol=1e-8)
        self.assertEqual(sent.dtype, np.float32)
        self.assertEqual(request["max_tries"], 6)
        data = report.data
        self.assertEqual([c["server_index"] for c in data["candidates"]], [1, 2, 0])
        self.assertAlmostEqual(data["selected"]["score"], 0.9, places=6)
        gg = np.eye(4)
        gg[:3, 3] = center + [0.04, 0.05, 0.06]
        tcp = gg @ np.array(load_config()["robot"]["T_gripper_tcp"])
        np.testing.assert_allclose(data["selected"]["T_base_gripper"], gg, atol=1e-7)
        np.testing.assert_allclose(data["selected"]["T_world_gripper"], backend.base_transform(self.base) @ gg, atol=1e-7)
        np.testing.assert_allclose(data["selected"]["T_world_tcp"], backend.base_transform(self.base) @ tcp, atol=1e-7)
        geometry = data["designated_grasp_geometry"]
        np.testing.assert_allclose(geometry["grasp_centroid_base_frame_xyz_m"], tcp[:3, 3])
        np.testing.assert_allclose(matrix_from_quaternion(geometry["recommended_gripper_orientation"]["quaternion_xyzw"]), tcp[:3, :3], atol=1e-12)
        self.assertFalse(data["motion"]["allowed"])
        self.assertIn("same_gripper", data["motion"]["blocked_reasons"])
        self.assertFalse(data["physical_grasp_validated"])
        self.assertFalse(data["checks"]["pca_filter_used"])
        self.assertNotIn("functional_zones_segmentation", data)
        for name in ("grasp.json", "grasp_graspgen.json"):
            loaded = StageReport.load(self.out / name)
            self.assertEqual(loaded.data["grasp_backend"], "graspgen")
            self.assertEqual(loaded.status, "COMPLETED")
        for artifact in report.artifacts.values():
            self.assertTrue(Path(artifact).is_file())
        saved = json.loads(Path(report.artifacts["candidates_json"]).read_text())
        self.assertEqual(len(saved["candidates"]), 3)

    def test_full_rotation_and_tcp_composition(self):
        pose = backend.base_transform(RobotBaseFrame([0.02, 0.03, 0.04], [0.4, -0.6, 1.1]))
        transform = np.array(load_config()["robot"]["T_gripper_tcp"])
        c = backend.rank_candidates([pose], [0.7], [0.4, -0.2, 0.5], self.base, transform)[0]
        expected = pose.copy()
        expected[:3, 3] += [0.4, -0.2, 0.5]
        np.testing.assert_allclose(c["T_base_tcp"], expected @ transform)
        np.testing.assert_allclose(c["T_world_tcp"], backend.base_transform(self.base) @ expected @ transform)

    def test_deterministic_cap_records_the_actual_center(self):
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
        cfg["input"]["max_points"] = 32
        config = self.out / "client.yaml"
        config.write_text(yaml.safe_dump(cfg))
        with FakeGPU() as server:
            first = self.run_server(server, graspgen_config=str(config))
            second = self.run_server(server, graspgen_config=str(config))
        sends = [r["point_cloud"] for r in server.requests if r["action"] == "infer"]
        self.assertEqual(sends[0].shape, (32, 3))
        np.testing.assert_array_equal(sends[0], sends[1])
        center = first.data["input_pointcloud"]["center_base_xyz_m"]
        np.testing.assert_allclose(np.asarray(first.data["selected"]["T_base_gripper"])[:3, 3],
                                   np.array(center) + [0.04, 0.05, 0.06], atol=1e-7)
        self.assertEqual(second.data["input_pointcloud"]["total_points"], 100)

    def test_config_flags_cannot_override_unverified_input_geometry(self):
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
        cfg["robot"].update(gripper="robotiq_2f_140", motion_enabled=True, tcp_transform_verified=True,
                            compatibility_verified=True, verification_evidence="synthetic fixture")
        config = self.out / "client.yaml"
        config.write_text(yaml.safe_dump(cfg))
        with FakeGPU() as server:
            report = self.run_server(server, graspgen_config=str(config))
        self.assertFalse(report.data["motion"]["allowed"])
        self.assertEqual(report.data["motion"]["blocked_reasons"], ["metric_depth_and_extrinsics_verified"])

    def test_stage_cli_and_real_resume_use_backend(self):
        from pipeline.real.pipeline import main as real_main
        from pipeline.sim.pipeline import main as sim_main
        self.report.stage = "real_deprojection"
        self.report.save(self.out / "real_deprojection.json")
        for main in (sim_main, real_main):
            with self.subTest(main=main), FakeGPU() as server:
                rc = main(["--from-stage", "grasp", "--out-dir", str(self.out),
                           "--grasp-backend", "graspgen", "--graspgen-port", str(server.port),
                           "--graspgen-timeout-ms", "1000"])
                self.assertEqual(rc, 0)
                self.assertEqual(StageReport.load(self.out / "grasp.json").data["grasp_backend"], "graspgen")

    def test_world_and_camera_cloud_frames(self):
        world = self.base.base_to_world(self.points)
        result, _ = backend.points_to_base(world, {"coordinate_frame_id": "world"}, self.base)
        np.testing.assert_allclose(result, self.points, atol=1e-15)
        extrinsic = backend.base_transform(RobotBaseFrame([0.5, 0.1, 1], [0.1, 0.2, -0.3]))
        camera = (self.points-extrinsic[:3, 3]) @ extrinsic[:3, :3]
        result, _ = backend.points_to_base(camera, {"coordinate_frame_id": "camera_optical",
                                                   "T_base_from_pointcloud": extrinsic}, self.base)
        np.testing.assert_allclose(result, self.points, atol=1e-15)
        with self.assertRaisesRegex(ValueError, "Unknown pointcloud frame"):
            backend.points_to_base(camera, {"coordinate_frame_id": "camera_optical"}, self.base)

    def test_empty_response_fails_without_overwriting_pca_archive(self):
        grasp.run(deprojection_path=str(self.dep), out_dir=str(self.out))
        pca_before = (self.out / "grasp_pca.json").read_bytes()
        with FakeGPU({"grasps": np.empty((0, 4, 4)), "confidences": np.empty(0)}) as server:
            with self.assertRaisesRegex(StageError, "no grasp candidates"):
                self.run_server(server)
        self.assertEqual((self.out / "grasp_pca.json").read_bytes(), pca_before)
        for filename in ("grasp.json", "grasp_graspgen.json"):
            failed = StageReport.load(self.out / filename, require_completed=False)
            self.assertEqual(failed.status, "FAILED")
            self.assertFalse(failed.data["motion"]["allowed"])
        with self.assertRaises(StageError):
            StageReport.load(self.out / "grasp.json")

    def test_mismatch_is_rejected_before_inference(self):
        for key, value in (("gripper_name", "franka_panda"), ("model_name", "graspmoe"),
                           ("gripper_config", "/models/wrong.yml")):
            m = metadata()
            m[key] = value
            with self.subTest(key=key), FakeGPU(meta=m) as server:
                with self.assertRaises(StageError):
                    self.run_server(server)
                self.assertEqual([r["action"] for r in server.requests], ["metadata"])
        m = metadata()
        m.pop("ros2project")
        with FakeGPU(meta=m) as server:
            with self.assertRaisesRegex(StageError, "provenance missing"):
                self.run_server(server)

    def test_timeout_is_bounded_and_writes_failed_report(self):
        with FakeGPU(infer_delay=0.2) as server:
            started = time.monotonic()
            with self.assertRaisesRegex(StageError, "timeout"):
                grasp.run(**self.kw, graspgen_port=server.port, graspgen_timeout_ms=40)
            self.assertLess(time.monotonic()-started, 1)
        failed = StageReport.load(self.out / "grasp_graspgen.json", require_completed=False)
        self.assertEqual(failed.status, "FAILED")

    def test_offline_server_metadata_timeout_has_no_infinite_wait(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        started = time.monotonic()
        with self.assertRaisesRegex(StageError, "timeout"):
            grasp.run(**self.kw, graspgen_port=port, graspgen_timeout_ms=40)
        self.assertLess(time.monotonic()-started, 1)

    def test_server_error_is_explicit(self):
        with FakeGPU({"error": "CUDA out of memory"}) as server:
            with self.assertRaisesRegex(StageError, "CUDA out of memory"):
                self.run_server(server)

    def test_invalid_scores_poses_and_shapes_are_rejected(self):
        for transform, score in ((np.diag([1, 1, -1, 1]), 0.9), (np.eye(4), float("nan")),
                                 (np.eye(4), 1.1), (np.full((4, 4), np.nan), 0.5)):
            with self.subTest(score=score), self.assertRaises(ValueError):
                backend.rank_candidates([transform], [score], np.zeros(3), self.base, np.eye(4))
        with self.assertRaisesRegex(ValueError, "Malformed"):
            backend.rank_candidates([np.eye(4)], [0.3, 0.4], np.zeros(3), self.base, np.eye(4))

    def test_units_extent_and_nonfinite_cloud_rejected_before_network(self):
        self.report.data["pointcloud_units"] = "mm"
        self.report.save(self.dep)
        with self.assertRaisesRegex(StageError, "units must be m"):
            grasp.run(**self.kw)
        self.report.data["pointcloud_units"] = "m"
        self.report.save(self.dep)
        write_ply(self.out / "cloud.ply", self.points * 1000)
        with self.assertRaisesRegex(StageError, "extent"):
            grasp.run(**self.kw)
        self.points[0, 0] = float("nan")
        write_ply(self.out / "cloud.ply", self.points)
        with self.assertRaisesRegex(StageError, "finite"):
            grasp.run(**self.kw)

    def test_motion_gate_even_when_requesting_preview_directly(self):
        with FakeGPU() as server:
            data = self.run_server(server).data
        from pipeline.sim.robot_motion_preview import MotionPreview
        with self.assertRaisesRegex(ValueError, "motion blocked"):
            MotionPreview(object(), data)
        data["motion"]["allowed"] = True
        data["configuration"]["robot"]["gripper"] = "robotiq_2f_140"
        with self.assertRaisesRegex(ValueError, "executor gripper"):
            backend.require_motion_allowed(data, gripper="visual_parallel_jaws")

    def test_import_and_portable_bundle(self):
        sample = self.out / "author.json"
        sample.write_text(json.dumps({"pc": self.points.tolist()}))
        imported = self.out / "imported"
        self.assertEqual(import_cloud.main(["--input", str(sample), "--units", "m", "--frame", "base",
                                           "--out-dir", str(imported)]), 0)
        portable = self.out / "portable"
        bundle.main(["--deprojection", str(imported / "imported_deprojection.json"), "--destination", str(portable)])
        # Delete the original: operative cloud now resolves relative to portable report.
        (imported / "imported_cloud.ply").unlink()
        with FakeGPU() as server:
            result = grasp.run(deprojection_path=str(portable / "deprojection.json"), out_dir=str(self.out / "result"),
                               backend="graspgen", graspgen_port=server.port, graspgen_timeout_ms=1000)
        self.assertEqual(result.data["input_pointcloud"]["total_points"], len(self.points))

    def test_launcher_flags_defaults_and_forwarding(self):
        from pipeline.sim.pipeline import build_parser as sim_parser, build_steps as sim_steps
        from pipeline.real.pipeline import build_parser as real_parser, build_steps as real_steps
        for parser in (sim_parser, real_parser):
            self.assertEqual(parser().parse_args([]).grasp_backend, "pca")
            args = parser().parse_args(["--grasp-backend", "graspgen", "--graspgen-host", "gpu.example",
                                        "--graspgen-port", "5558", "--graspgen-timeout-ms", "80"])
            steps = sim_steps(args, "scissors", "single") if parser == sim_parser else real_steps(args)
            with patch("pipeline.grasp.run") as run:
                steps[-1].run()
                self.assertEqual(run.call_args.kwargs["backend"], "graspgen")
                self.assertEqual(run.call_args.kwargs["graspgen_port"], 5558)

    def test_complete_headless_pipeline_and_motion_block(self):
        import trimesh
        mesh = self.out / "instrument.stl"
        trimesh.creation.box([0.14, 0.012, 0.004]).export(mesh)
        cfg = yaml.safe_load((ROOT / "config/surgical_environment.yaml").read_text())
        cfg["surgical_instrument"]["stl_path"] = str(mesh)
        cfg["environment"]["kuka_base"].update(self.base.to_dict())
        config = self.out / "environment.yaml"
        config.write_text(yaml.safe_dump(cfg))
        for no_motion in (True, False):
            out = self.out / ("visualize" if no_motion else "blocked")
            with self.subTest(no_motion=no_motion), FakeGPU() as server:
                args = [sys.executable, "run_pipeline_sim.py", "--headless", "--vision", "ground-truth",
                        "--tool", "Scalpel", "--config", str(config), "--out-dir", str(out),
                        "--grasp-backend", "graspgen", "--graspgen-port", str(server.port),
                        "--graspgen-timeout-ms", "1000"]
                if no_motion:
                    args.append("--no-motion")
                env = dict(os.environ, PIPELINE_ROOT=str(self.out))
                result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, 0 if no_motion else 1, result.stdout + result.stderr)
            manifest = StageReport.load(out / "graphical_run.json", require_completed=False)
            self.assertEqual(manifest.data["state"], "GRASP_VISUALIZED" if no_motion else "FAILED")
            self.assertFalse(manifest.data["physical_grasp_validated"])
            self.assertFalse((out / "sim_robot_motion.json").exists())
            self.assertEqual(StageReport.load(out / "grasp_graspgen.json").data["grasp_backend"], "graspgen")

    def test_official_source_and_cpu_import_without_torch(self):
        code = ROOT / "third_party/GraspGen"
        self.assertEqual(check_source(code), PINNED_REVISION)
        subprocess.run([sys.executable, "-c", "from pipeline.graspgen.client import official_client_class; "
                        "from pipeline.graspgen.config import load_config; import sys; "
                        "official_client_class(load_config()['submodule']); assert 'torch' not in sys.modules"],
                       cwd=ROOT, check=True, timeout=10)
        recipe = build_dockerfile(code)
        self.assertNotIn("pickle5", recipe)
        self.assertIn("--no-build-isolation ./pointnet2_ops", recipe)
        self.assertIn("torch-2.1.0+cu121.html", recipe)
        official = yaml.safe_load((code / "config/grippers/robotiq_2f_140.yaml").read_text())
        self.assertEqual(official["depth"], load_config()["robot"]["T_gripper_tcp"][2][3])

    def test_model_preflight_checks_both_checkpoints_and_grippers(self):
        model = self.out / "graspgen_robotiq_2f_140.yml"
        cfg = {"data": {"gripper_name": "robotiq_2f_140"},
               "diffusion": {"gripper_name": "robotiq_2f_140", "obs_backbone": "pointnet"},
               "discriminator": {"gripper_name": "robotiq_2f_140", "obs_backbone": "pointnet",
                                 "checkpoint": "graspgen_robotiq_2f_140_dis.pth"},
               "eval": {"model_name": "diffusion-discriminator", "checkpoint": "graspgen_robotiq_2f_140_gen.pth"}}
        model.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, "Missing checkpoint"):
            check_models(model)
        for kind in ("gen", "dis"):
            (self.out / f"graspgen_robotiq_2f_140_{kind}.pth").write_bytes(b"fake test data" * 100)
        result = check_models(model, hashes=True)
        self.assertEqual(len(result["generator_sha256"]), 64)
        cfg["discriminator"]["gripper_name"] = "franka_panda"
        model.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, "gripper names"):
            check_models(model)


if __name__ == "__main__":
    unittest.main()
