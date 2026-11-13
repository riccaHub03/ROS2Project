"""Preflight and checkpoint validation shared by GPU scripts and server wrapper."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import yaml

from pipeline.graspgen.config import PINNED_REVISION, ROOT


def git_output(code, *args):
    return subprocess.check_output(["git", "-c", f"safe.directory={Path(code).resolve()}",
                                    "-C", str(code), *args], text=True).strip()


def check_source(code):
    code = Path(code)
    revision = git_output(code, "rev-parse", "HEAD")
    if revision != PINNED_REVISION:
        raise ValueError(f"Submodule revision {revision} differs from pin {PINNED_REVISION}")
    if git_output(code, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Official submodule is modified; keep it unchanged")
    required = ["client-server/graspgen_server.py", "grasp_gen/serving/zmq_client.py",
                "grasp_gen/serving/zmq_server.py", "scripts/demo_object_pc.py",
                "docker/graspgen_cuda121.dockerfile", "config/grippers/robotiq_2f_140.yaml"]
    for name in required:
        if not (code / name).is_file():
            raise ValueError(f"Missing official path: {code / name}")
    return revision


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_dockerfile(code):
    """Fix an ephemeral copy, never the official tree. Python 3.10 needs no pickle5."""
    source = (Path(code) / "docker/graspgen_cuda121.dockerfile").read_text()
    fixes = {
        "FROM nvcr.io/nvidia/pytorch:23.07-py3 AS base":
            "FROM nvcr.io/nvidia/pytorch:23.07-py3 AS base\n"
            "# The retired NGC pip mirror causes DNS retries for every package.\n"
            "ENV PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1",
        "RUN pip install --upgrade pip": "RUN pip install 'pip<25.3' 'setuptools<78' wheel numpy==1.26.4",
        "RUN pip install torch==2.1.0 torchvision": "RUN pip install torch==2.1.0 torchvision==0.16.0 numpy==1.26.4",
        "RUN pip install imageio pickle5 opencv-python python-fcl":
            "RUN pip install imageio 'opencv-python<4.12' python-fcl",
        # Ninja otherwise starts CPU-count + 2 compiler jobs, which can exhaust
        # RAM on a 16 GB desktop while compiling CUDA for several architectures.
        "RUN pip install ./pointnet2_ops": "RUN MAX_JOBS=1 TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;9.0' pip install --no-build-isolation ./pointnet2_ops",
        "torch-2.1.2+cu121.html": "torch-2.1.0+cu121.html",
    }
    for old, new in fixes.items():
        if source.count(old) != 1:
            raise ValueError(f"Official Dockerfile changed; review compatibility fix: {old}")
        source = source.replace(old, new)
    return source


def check_models(config, *, hashes=False):
    config = Path(config).resolve()
    cfg = yaml.safe_load(config.read_text())
    gripper = "robotiq_2f_140"
    if config.name != f"graspgen_{gripper}.yml":
        raise ValueError("Expected the official Robotiq 2F-140 checkpoint config")
    if cfg["eval"]["model_name"] != "diffusion-discriminator":
        raise ValueError("Checkpoint must use diffusion-discriminator")
    if any(cfg[section]["gripper_name"] != gripper for section in ("data", "diffusion", "discriminator")):
        raise ValueError("data/diffusion/discriminator gripper names must all match Robotiq 2F-140")
    result = {"config": str(config), "config_sha256": sha256(config), "gripper": gripper,
              "backbones": {s: cfg[s]["obs_backbone"] for s in ("diffusion", "discriminator")}}
    for section, kind in (("eval", "generator"), ("discriminator", "discriminator")):
        file = (config.parent / cfg[section]["checkpoint"]).resolve()
        if file.name != f"graspgen_{gripper}_{'gen' if kind == 'generator' else 'dis'}.pth":
            raise ValueError(f"Unexpected {kind} checkpoint filename: {file.name}")
        if not file.is_file() or file.stat().st_size < 1024:
            raise ValueError(f"Missing checkpoint / Git LFS pointer: {file}; run models on the NVIDIA PC")
        with file.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"Git LFS pointer instead of weights: {file}")
        result[f"{kind}_checkpoint"] = str(file)
        if hashes:
            result[f"{kind}_sha256"] = sha256(file)
    models_root = config.parent.parent
    if (models_root / ".git").exists():
        result["models_revision"] = git_output(models_root, "rev-parse", "HEAD")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also check Linux, NVIDIA driver and Docker GPU passthrough")
    parser.add_argument("--code", type=Path, default=ROOT / "third_party/GraspGen")
    parser.add_argument("--models", type=Path, help="Validate downloaded weights and author sample data")
    parser.add_argument("--write-base-dockerfile", type=Path, help="Write a corrected temporary official base recipe")
    args = parser.parse_args(argv)
    result = {"revision": check_source(args.code), "platform": platform.platform(),
              "client_source": str(args.code / "grasp_gen/serving/zmq_client.py")}
    if args.write_base_dockerfile:
        args.write_base_dockerfile.write_text(build_dockerfile(args.code))
    if args.gpu:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            parser.error("GPU setup targets Linux x86_64 on an NVIDIA PC; Mac is client only")
        for command in ("nvidia-smi", "docker", "git"):
            if not shutil.which(command):
                parser.error(f"Missing prerequisite: {command}")
        subprocess.run(["nvidia-smi"], check=True)
        capabilities = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True)
        if any(float(v.strip()) >= 10 for v in capabilities.splitlines()):
            parser.error("Blackwell requires a separately verified CUDA 12.8+ profile; this recipe uses CUDA 12.1 / torch 2.1")
        subprocess.run(["git", "lfs", "version"], check=True)
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "run", "--rm", "--gpus", "all", "ubuntu:22.04", "nvidia-smi"], check=True)
    if args.models:
        result["models"] = check_models(args.models / "checkpoints/graspgen_robotiq_2f_140.yml")
        samples = sorted((args.models / "sample_data/real_object_pc").glob("*.json"))
        if not samples:
            parser.error("No author point clouds found in sample_data/real_object_pc")
        result["sample_count"] = len(samples)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
