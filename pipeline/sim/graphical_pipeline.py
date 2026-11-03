"""Esegue la pipeline mantenendo viva la finestra PyBullet sul thread principale."""

from __future__ import annotations

import datetime
import fcntl
import multiprocessing
import queue
import time
import traceback

import numpy as np
from PIL import Image
import pybullet as p

from pipeline.config_loader import load_environment_config
from pipeline.sim.robot_scene import ROOT, RobotScene, write_json
from pipeline.sim.robot_motion_preview import MotionPreview
from pipeline.step1_llm_planner import run_llm_planner


def _compute(plan, camera, vision, events):
    """Il processo separato mantiene reattiva la GUI anche durante SAM3 su CPU."""
    try:
        out = ROOT / "data/outputs"
        if vision == "sam3":
            events.put(("status", "Stage 3: caricamento e inferenza SAM3 su CPU"))
            import torch
            torch.set_num_threads(3)
            from pipeline.sim.sim_step3_sam3_camera import run_sim_sam3_segmentation
            mask = run_sim_sam3_segmentation()
            if mask["confidence_score"] < 0.15:
                raise RuntimeError("SAM3 ha confidenza inferiore a 0.15: ispeziona l'overlay; movimento annullato")
            mask["segmentation_backend"] = "sam3"
        else:
            events.put(("status", "Stage 3 diagnostico: maschera da ID PyBullet, SAM3 escluso"))
            binary = np.asarray(Image.open(camera["files"]["ground_truth_mask"]))
            Image.fromarray(binary).save(out / "sim_camera_mask.png")
            rgb = np.asarray(Image.open(camera["files"]["camera_rgb_image"])).copy()
            selected = binary > 128
            rgb[selected] = (rgb[selected] * 0.35 + np.array([0, 255, 60]) * 0.65).astype(np.uint8)
            Image.fromarray(rgb).save(out / "sim_camera_overlay.png")
            mask = {
                "stage": "sim_stage3_vision_mask", "segmentation_backend": "simulator_object_id",
                "confidence_score": None, "active_pixel_count": int(selected.sum()),
                "files": {"camera_mask_png": str(out / "sim_camera_mask.png"),
                          "camera_overlay_png": str(out / "sim_camera_overlay.png")},
                "status": "DIAGNOSTIC_GROUND_TRUTH",
            }
        if mask["active_pixel_count"] < 15:
            raise RuntimeError("Maschera vuota o troppo piccola: impossibile calcolare la presa")
        write_json(out / "sim_stage3_vision_mask.json", mask)
        events.put(("status", "Stage 4: deproiezione della maschera nel riferimento robot"))
        from pipeline.sim.sim_step4_transforms import run_sim_transforms
        from pipeline.step5_grasp_geometry import parse_ply, run_grasp_geometry
        transforms = run_sim_transforms()
        points, _ = parse_ply(transforms["pointcloud_file"])
        centroid = np.mean(points, axis=0)
        if not np.allclose(centroid, transforms["frames"]["robot_base_frame"], atol=0.00015):
            raise RuntimeError("Centroide e nuvola di punti incoerenti: movimento annullato")
        events.put(("status", "Stage 5: PCA e proposta geometrica della presa"))
        grasp = run_grasp_geometry(source_type="sim")
        grasp["status"] = "GRASP_CANDIDATE"
        grasp["segmentation_backend"] = mask["segmentation_backend"]
        grasp["feasibility"] = {"ik_checked": False, "collision_checked": False, "contact_checked": False}
        write_json(out / "stage5_grasp_geometry.json", grasp)
        events.put(("result", grasp))
    except BaseException as exc:
        traceback.print_exc()
        events.put(("error", f"{type(exc).__name__}: {exc}"))


def run_graphical_pipeline(args, instruction):
    out = ROOT / "data/outputs"
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".graphical.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Una run grafica e' gia' attiva. Chiudila prima di avviarne un'altra")
    scene = worker = events = None
    result = {
        "started_at": datetime.datetime.now().isoformat(), "status": "STARTING",
        "vision": args.vision, "display": "headless" if args.headless else "native_pybullet_gui",
        "physical_grasp_validated": False,
    }
    manifest = out / "graphical_run.json"
    try:
        diagnostic = args.vision == "ground-truth"
        print("Modalita diagnostica: parser deterministico e maschera da simulatore" if diagnostic
              else "Pipeline con Ollama/SAM3; movimento PyBullet dimostrativo", flush=True)
        plan = run_llm_planner(instruction_text=instruction, target_tool_hint=args.tool, offline=diagnostic,
                               ollama_url="http://127.0.0.1:11434/v1/chat/completions")
        scene = RobotScene(load_environment_config(), plan, gui=not args.headless, seed=args.seed)
        scene.status("Scena pronta. R: ripeti movimento; P: pausa; Q: chiudi")
        result.update(status="PROCESSING", task_plan=plan, camera_report=scene.camera_report)
        write_json(manifest, result)
        context = multiprocessing.get_context("spawn")
        events = context.Queue()
        worker = context.Process(target=_compute, args=(plan, scene.camera_report, args.vision, events), daemon=True)
        worker.start()
        motion = grasp = None
        paused = False
        done = False
        exit_code = 0
        dead_since = None
        tick = 0
        while p.isConnected(scene.cid):
            frame_start = time.monotonic()
            if scene.gui:
                keys = p.getKeyboardEvents(physicsClientId=scene.cid)
                pressed = lambda key: keys.get(ord(key), 0) & p.KEY_WAS_TRIGGERED
                if pressed("q") or keys.get(27, 0) & p.KEY_WAS_TRIGGERED:
                    break
                if pressed("p"):
                    paused = not paused
                    scene.status("Pausa" if paused else "Ripresa")
                if pressed("r") and done and grasp is not None and not args.no_motion:
                    scene.restore()
                    motion = MotionPreview(scene, grasp)
                    done = False
                    exit_code = 0
                    result["status"] = "ANIMATING"
                    write_json(manifest, result)
            while True:
                try:
                    kind, data = events.get_nowait()
                except queue.Empty:
                    break
                if kind == "status":
                    scene.status(data)
                elif kind == "error":
                    scene.status(data, [0.8, 0.05, 0.05])
                    result.update(status="FAILED", error=data)
                    done, exit_code = True, 1
                elif kind == "result":
                    grasp = data
                    scene.show_grasp(grasp)
                    scene.save_overview(out / "sim_robot_grasp.png")
                    result["grasp_report"] = str(out / "stage5_grasp_geometry.json")
                    if args.no_motion:
                        result["status"] = "GRASP_VISUALIZED"
                        done = True
                    else:
                        scene.status("Verifica IK dei waypoint e dei limiti articolari")
                        try:
                            motion = MotionPreview(scene, grasp)
                            result["status"] = "ANIMATING"
                        except (ValueError, RuntimeError) as exc:
                            scene.status(str(exc), [0.8, 0.05, 0.05])
                            result.update(status="FAILED", error=str(exc))
                            done, exit_code = True, 1
                write_json(manifest, result)
            if not done and worker.exitcode is not None and grasp is None:
                dead_since = dead_since or time.monotonic()
                if time.monotonic() - dead_since > 2:
                    result.update(status="FAILED", error=f"Processo di analisi terminato: codice {worker.exitcode}")
                    scene.status(result["error"], [0.8, 0.05, 0.05])
                    done, exit_code = True, 1
                    write_json(manifest, result)
            for _ in range(4):
                if motion is not None and not done and not paused:
                    try:
                        motion.tick()
                    except (ValueError, RuntimeError) as exc:
                        scene.release_tool()
                        scene.status(str(exc), [0.8, 0.05, 0.05])
                        result.update(status="FAILED", error=str(exc), motion=motion.report())
                        done, exit_code = True, 1
                        write_json(manifest, result)
                if not paused:
                    scene.step()
                if motion is not None and motion.finished and not done:
                    # Lascia assestare lo strumento dopo il rilascio.
                    for _ in range(120):
                        scene.step()
                    report = motion.report()
                    write_json(out / "sim_robot_motion.json", report)
                    result.update(status=report["status"], motion=report)
                    exit_code = 0 if report["status"] == "COMPLETED" else 1
                    scene.save_overview(out / "sim_robot_result.png")
                    scene.status("Anteprima completata. Presa con vincolo ideale. R: ripeti; Q: chiudi")
                    done = True
                    write_json(manifest, result)
            tick += 1
            if done and (args.headless or args.exit_after_run):
                return exit_code
            if scene.gui:
                time.sleep(max(0, 1 / 60 - (time.monotonic() - frame_start)))
            elif tick % 60 == 0:
                time.sleep(0.005)
        if not done:
            result["status"] = "CANCELLED"
            write_json(manifest, result)
        return exit_code
    except p.error as exc:
        if scene is not None and scene.gui and not p.isConnected(scene.cid):
            result["status"] = "CANCELLED"
            write_json(manifest, result)
            return 130
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_json(manifest, result)
        raise
    except BaseException as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_json(manifest, result)
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
