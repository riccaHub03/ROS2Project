"""Portable launcher. ROS bindings are imported only after argument/preflight checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from uuid import uuid4

import yaml

from pipeline.common import paths
from pipeline.common.report import StageReport, write_json
from pipeline.config_loader import load_environment_config
from pipeline.system.motion import configure_motion, motion_speed
from pipeline.system.registry import load_profile


def parser():
    p = argparse.ArgumentParser(description="LLM/SAM3/GraspGen with ROS 2 RGB-D, arm and gripper actions (simulation only)")
    p.add_argument("--profile", choices=["full_system", "graspgen_only"], default="full_system")
    p.add_argument("--instruction", default=None)
    p.add_argument("--config")
    p.add_argument("--tool")
    p.add_argument("--ollama-url")
    p.add_argument("--llm-model")
    p.add_argument("--sam3-model")
    p.add_argument("--out-dir", help="New directory; existing runs are never overwritten")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--motion-speed", type=motion_speed, default=1.)
    display = p.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true")
    display.add_argument("--headless", action="store_true")
    p.add_argument("--check", action="store_true", help="Check ROS, models, services and assets; no motion")
    return p


def check(settings):
    from pipeline.system.preflight import check as ai_check
    try:
        import rclpy
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from sensor_msgs.msg import CameraInfo, Image
        import tf2_ros
        ros = dict(name="ros2_runtime", ok=True, detail=dict(distro=os.environ.get("ROS_DISTRO"),
                                                            python=sys.version.split()[0]))
    except ImportError as exc:
        ros = dict(name="ros2_runtime", ok=False, detail=f"{exc}. Use bash scripts/run_ros2_system.sh --check")
    return [ros] + ai_check(settings)


def run(settings, args, checks):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import String
    from pipeline.ros2.coordinator import Coordinator
    from pipeline.ros2.simulation import snapshot_qos
    from pipeline.system.orchestrator import finalize

    out = Path(args.out_dir).resolve() if args.out_dir else paths.project_root() / "data/outputs" / (
        "ros2_" + args.profile + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    # A UUID namespace prevents accidental cross-talk with another run or hardware.
    namespace = "/surgical_sim/run_" + uuid4().hex[:12]
    environment = load_environment_config(settings["environment"])
    environment["simulation_gripper"] = "robotiq_2f_140_official_meshes"
    configure_motion(environment["motion_control"], args.motion_speed)
    snapshot = out / "environment.yaml"
    snapshot.write_text(yaml.safe_dump(environment, sort_keys=False))
    settings = dict(settings, environment=str(snapshot), motion_speed=args.motion_speed)
    from pipeline.robot_model import RobotModel
    from pipeline.sim.gripper_geometry import for_config
    robot_description = RobotModel.from_config(environment).report(for_config(environment))
    instruction = args.instruction or f"Pick up the {settings['scene_tool']} from tray 2 and place it in tray 1."
    spec = dict(out_dir=str(out), namespace=namespace, settings=settings, instruction=instruction,
                gui=not args.headless, seed=args.seed,
                base_frame_id=environment["environment"]["kuka_base"].get("frame_id", "lbr_link_0"))
    run_file = write_json(out / "ros2_request.json", spec)
    try:
        commit = subprocess.check_output(["git", "-C", str(paths.project_root()), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(paths.project_root()), "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = dict(schema_version=1, run_id=out.name, request_id=uuid4().hex,
                    profile=args.profile, transport="ros2", status="RUNNING", settings=settings,
                    instruction=instruction, seed=args.seed, namespace=namespace,
                    started_at=datetime.now().astimezone().isoformat(), preflight=checks,
                    source_commit=commit, source_worktree_dirty=dirty,
                    robot_description=robot_description,
                    validation_mode=settings.get("validation_mode", "live"),
                    environment_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "runtime_versions.json", dict(python=sys.version, ros_distro=os.environ.get("ROS_DISTRO"),
                 packages=subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()))
    old_output = os.environ.get(paths.OUTPUT_ENV)
    os.environ[paths.OUTPUT_ENV] = str(out)
    process = node = sender = executor = thread = None
    code, error = 1, None
    try:
        rclpy.init(args=[])
        # Inherit the exact interpreter/environment; never import Humble bindings
        # into an unrelated Python 3.11 venv.
        with (out / "ros2_simulation.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "pipeline.ros2.simulation", "--run-file", str(run_file)],
                                       stdout=log, stderr=subprocess.STDOUT)
            node = Coordinator(spec, process)
            sender = Node("surgical_request", namespace=namespace)
            publisher = sender.create_publisher(String, "instruction", snapshot_qos())
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            executor.add_node(sender)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            print(f"ROS namespace: {namespace}\nResults: {out}", flush=True)
            deadline = time.monotonic() + 15
            while publisher.get_subscription_count() == 0:
                node.healthy()
                if time.monotonic() > deadline:
                    raise TimeoutError("ROS instruction subscriber was not discovered")
                time.sleep(.05)
            publisher.publish(String(data=instruction))
            while node.instruction is None:
                node.healthy()
                if time.monotonic() > deadline:
                    raise TimeoutError("Instruction was not delivered through ROS 2")
                time.sleep(.02)
            node.execute_task()
            code = 0
    except KeyboardInterrupt:
        code, error = 130, "Interrupted by user"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if code != 0:
            StageReport("ros2_run", status="CANCELLED" if code == 130 else "FAILED",
                        data=dict(execution_scope="pybullet_only", physical_grasp_validated=False,
                                  namespace=namespace, failed_stage=getattr(node, "current_stage", "STARTUP"),
                                  completed_actions=getattr(node, "action_results", [])),
                        error={"message": error}).save(out / "ros2_run.json")
            if node:
                node.stop()
        if process and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if executor:
            executor.shutdown(timeout_sec=3)
        if thread:
            thread.join(timeout=3)
        if node:
            node.destroy_node()
        if sender:
            sender.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if old_output is None:
            os.environ.pop(paths.OUTPUT_ENV, None)
        else:
            os.environ[paths.OUTPUT_ENV] = old_output
    result = finalize(out, manifest, code, error)
    print(f"{manifest['status']}: {out}", flush=True)
    if manifest.get("error"):
        print(manifest["error"], file=sys.stderr)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in argv:
        from rclpy.utilities import remove_ros_args
        argv = remove_ros_args(args=["full_system", *argv])[1:]
    args = parser().parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", str(paths.project_root() / ".cache/matplotlib"))
    if (args.config and not args.tool) or (args.tool and args.tool.casefold() != "bone forceps" and not args.config):
        parser().error("Custom scenes require both --config and --tool")
    try:
        settings = load_profile(args.profile, environment=args.config, scene_tool=args.tool,
                                ollama_url=args.ollama_url, llm_model=args.llm_model, sam3_model=args.sam3_model)
        checks = check(settings)
        if args.check or not all(c["ok"] for c in checks):
            print(json.dumps(checks, indent=2))
            return 0 if all(c["ok"] for c in checks) else 1
        return run(settings, args, checks)
    except (ValueError, OSError) as exc:
        print(f"ROS system configuration: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
