"""Interactive/headless KUKA run: keeps the PyBullet window alive on the main thread while a
worker process runs segmentation -> deprojection -> grasp analysis through the regular stages.

Outputs (in addition to the stage reports): ``graphical_run.json`` (run manifest) and
``sim_robot_motion.json`` (measured motion report), both in the standard report envelope.
"""

from __future__ import annotations

import fcntl
import logging
import multiprocessing
import os
import queue
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
import pybullet as p

from pipeline import grasp, plan
from pipeline.common import paths
from pipeline.common.ply import read_ply
from pipeline.common.report import CANCELLED, COMPLETED, FAILED, RUNNING, StageReport
from pipeline.config_loader import load_environment_config
from pipeline.sim import deproject, segment
from pipeline.sim.robot_motion_preview import MotionPreview
from pipeline.sim.robot_scene import RobotScene

log = logging.getLogger("graphical")
MIN_CONFIDENCE = 0.15
CENTROID_TOLERANCE_M = 0.00015
OLLAMA_LOCAL_URL = "http://127.0.0.1:11434/v1/chat/completions"


class RunManifest:
    """``graphical_run.json``: coarse envelope status plus a fine-grained ``data.state``."""

    def __init__(self, path, **data: Any):
        self.path = path
        self.report = StageReport("graphical_run", status=RUNNING, data={"state": "STARTING", **data})
        self.report.save(path)

    def update(self, state: Optional[str] = None, status: Optional[str] = None,
               error: Optional[str] = None, **data: Any) -> None:
        if state:
            self.report.data["state"] = state
        if status:
            self.report.status = status
        if error:
            self.report.error = {"message": error}
        self.report.data.update(data)
        self.report.save(self.path)


def _compute(vision: str, events, config_path: Optional[str] = None) -> None:
    """Worker process: keeps the GUI responsive while SAM 3 runs on CPU."""
    try:
        out = paths.output_dir()
        reference_pose = out / paths.SIM_REFERENCE_POSE
        if vision == "sam3":
            events.put(("status", "Stage segment: loading and running SAM3 on CPU"))
            import torch
            torch.set_num_threads(3)
        else:
            events.put(("status", "Diagnostic segment: mask from PyBullet IDs, SAM3 bypassed"))
        mask = segment.run(backend=vision)
        confidence = mask.data["confidence_score"]
        if confidence is not None and confidence < MIN_CONFIDENCE:
            raise RuntimeError(f"SAM3 confidence below {MIN_CONFIDENCE}: inspect the overlay; motion cancelled")
        if mask.data["active_pixel_count"] < segment.MIN_MASK_PIXELS:
            raise RuntimeError("Mask empty or too small: cannot compute the grasp")

        events.put(("status", "Stage deproject: mask to robot base frame"))
        deprojection = deproject.run(reference_pose_path=str(reference_pose), config_path=config_path)
        points, _ = read_ply(deprojection.artifact("pointcloud_ply"))
        if (deprojection.data["centroid_source"] == "optical" and not np.allclose(
                points.mean(axis=0), deprojection.data["frames"]["robot_base_frame"], atol=CENTROID_TOLERANCE_M)):
            raise RuntimeError("Centroid and point cloud are inconsistent: motion cancelled")

        events.put(("status", "Stage grasp: PCA and grasp proposal"))
        report = grasp.run(deprojection_path=str(out / paths.SIM_DEPROJECTION), config_path=config_path)
        if report.data.get("readiness_blockers"):
            raise RuntimeError("Grasp not cleared: " + "; ".join(report.data["readiness_blockers"]))
        report.data.update(readiness="GRASP_CANDIDATE", segmentation_backend=mask.data["segmentation_backend"],
                           feasibility={**report.data.get("feasibility", {}), "ik_checked": False,
                                        "collision_checked": False, "contact_checked": False})
        report.save(out / paths.GRASP)
        events.put(("result", report.data))
    except BaseException as exc:
        traceback.print_exc()
        events.put(("error", f"{type(exc).__name__}: {exc}"))


def _acquire_lock(out):
    lock = (out / ".graphical.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("A graphical run is already active. Close it before starting another")
    return lock


def run_graphical_pipeline(args, instruction: str) -> int:
    if args.out_dir:
        os.environ[paths.OUTPUT_ENV] = str(args.out_dir)   # inherited by the spawned worker
    out = paths.output_dir()
    lock = _acquire_lock(out)
    scene = worker = events = None
    manifest = RunManifest(out / "graphical_run.json", vision=args.vision,
                           display="headless" if args.headless else "native_pybullet_gui",
                           physical_grasp_validated=False)
    try:
        diagnostic = args.vision == "ground-truth"
        log.info("diagnostic mode: deterministic parser and simulator mask" if diagnostic
                 else "pipeline with Ollama/SAM3; demonstrative PyBullet motion")
        plan_report = plan.run(instruction, tool_hint=args.tool, offline=diagnostic, ollama_url=OLLAMA_LOCAL_URL,
                               config_path=args.config)
        scene = RobotScene(load_environment_config(args.config), plan_report.data, gui=not args.headless,
                           seed=args.seed)
        scene.status("Scene ready. R: repeat motion; P: pause; Q: quit")
        manifest.update("PROCESSING", plan_report=str(out / paths.PLAN), capture_report=str(out / paths.SIM_CAPTURE))
        context = multiprocessing.get_context("spawn")
        events = context.Queue()
        worker = context.Process(target=_compute, args=(args.vision, events, args.config), daemon=True)
        worker.start()
        motion = grasp_data = None
        paused = done = False
        exit_code = 0
        dead_since = None
        tick = 0
        error_color = [0.8, 0.05, 0.05]

        def fail(message: str, **data: Any) -> None:
            nonlocal done, exit_code
            scene.status(message, error_color)
            manifest.update("FAILED", status=FAILED, error=message, **data)
            done, exit_code = True, 1

        while p.isConnected(scene.cid):
            frame_start = time.monotonic()
            if scene.gui:
                keys = p.getKeyboardEvents(physicsClientId=scene.cid)
                pressed = lambda key: keys.get(ord(key), 0) & p.KEY_WAS_TRIGGERED
                if pressed("q") or keys.get(27, 0) & p.KEY_WAS_TRIGGERED:
                    break
                if pressed("p"):
                    paused = not paused
                    scene.status("Paused" if paused else "Resumed")
                if pressed("r") and done and grasp_data is not None and not args.no_motion:
                    scene.restore()
                    motion = MotionPreview(scene, grasp_data)
                    done, exit_code = False, 0
                    manifest.update("ANIMATING", status=RUNNING)
            while True:
                try:
                    kind, payload = events.get_nowait()
                except queue.Empty:
                    break
                if kind == "status":
                    scene.status(payload)
                elif kind == "error":
                    fail(payload)
                elif kind == "result":
                    grasp_data = payload
                    scene.show_grasp(grasp_data)
                    scene.save_overview(out / "sim_robot_grasp.png")
                    manifest.update(grasp_report=str(out / paths.GRASP))
                    if args.no_motion:
                        manifest.update("GRASP_VISUALIZED", status=COMPLETED)
                        done = True
                    else:
                        scene.status("Checking waypoint IK and joint limits")
                        try:
                            motion = MotionPreview(scene, grasp_data)
                            manifest.update("ANIMATING")
                        except (ValueError, RuntimeError) as exc:
                            fail(str(exc))
            if not done and worker.exitcode is not None and grasp_data is None:
                dead_since = dead_since or time.monotonic()
                if time.monotonic() - dead_since > 2:
                    fail(f"Analysis process terminated: exit code {worker.exitcode}")
            for _ in range(4):
                if motion is not None and not done and not paused:
                    try:
                        motion.tick()
                    except (ValueError, RuntimeError) as exc:
                        scene.release_tool()
                        fail(str(exc), motion=motion.report())
                if not paused:
                    scene.step()
                if motion is not None and motion.finished and not done:
                    for _ in range(120):   # let the tool settle after release
                        scene.step()
                    motion_report = motion.report()
                    StageReport("sim_robot_motion", status=motion_report["status"], data=motion_report
                                ).save(out / "sim_robot_motion.json")
                    exit_code = 0 if motion_report["status"] == COMPLETED else 1
                    manifest.update(motion_report["status"], status=motion_report["status"], motion=motion_report)
                    scene.save_overview(out / "sim_robot_result.png")
                    scene.status("Preview complete (ideal fixed-constraint grasp). R: repeat; Q: quit")
                    done = True
            tick += 1
            if done and (args.headless or args.exit_after_run):
                return exit_code
            if scene.gui:
                time.sleep(max(0, 1 / 60 - (time.monotonic() - frame_start)))
            elif tick % 60 == 0:
                time.sleep(0.005)
        if not done:
            manifest.update("CANCELLED", status=CANCELLED)
        return exit_code
    except p.error as exc:
        if scene is not None and scene.gui and not p.isConnected(scene.cid):
            manifest.update("CANCELLED", status=CANCELLED)
            return 130
        manifest.update("FAILED", status=FAILED, error=f"{type(exc).__name__}: {exc}")
        raise
    except BaseException as exc:
        manifest.update("FAILED", status=FAILED, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if worker is not None:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=3)
        if events is not None:
            events.close()
        if scene is not None:
            scene.close()
        lock.close()
