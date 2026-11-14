"""Read-only readiness checks. Does not generate grasps or command motion."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import urllib.request


def check(settings):
    results = []

    def probe(name, fn):
        try:
            detail = fn()
            results.append(dict(name=name, ok=True, detail=detail))
        except Exception as exc:
            results.append(dict(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}"))

    def assets():
        from pipeline.config_loader import load_environment_config
        env = load_environment_config(settings["environment"])
        from pipeline.robot_model import RobotModel
        robot = RobotModel.from_config(env)
        mesh = Path(env["surgical_instrument"]["stl_path"])
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        from pipeline.sim.robotiq_description import prepared_description
        prepared_description()
        # Import the same nominal gripper description used by the motion adapter.
        from pipeline.sim.gripper_geometry import ROBOTIQ_PROXY
        return {"mesh": str(mesh), "gripper": ROBOTIQ_PROXY.name, "robot": robot.report(ROBOTIQ_PROXY)}

    def dependencies():
        for name in ("pybullet", "trimesh", "scipy", "zmq", "msgpack", "matplotlib"):
            importlib.import_module(name)
        return "Simulation and client imports available"

    def ollama():
        base = settings["ollama_url"].removesuffix("/v1/chat/completions")
        with urllib.request.urlopen(base + "/api/tags", timeout=5) as response:
            models = json.load(response)["models"]
        names = {m.get("name") for m in models} | {m.get("model") for m in models}
        if settings["llm_model"] not in names:
            raise RuntimeError(f"Model {settings['llm_model']} absent; run ollama pull {settings['llm_model']}")
        return {"model": settings["llm_model"], "endpoint": settings["ollama_url"]}

    def sam3():
        from transformers import Sam3Model, Sam3Processor, Sam3ImageProcessorFast
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        model = settings["sam3_model"]
        try:
            directory = Path(model) if Path(model).is_dir() else Path(snapshot_download(model, local_files_only=True))
        except LocalEntryNotFoundError as exc:
            raise RuntimeError("SAM3 weights are not downloaded. After HF login run: "
                               ".venv/bin/python scripts/download_sam3.py") from exc
        for name in ("config.json", "processor_config.json", "tokenizer_config.json"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        indexes = list(directory.glob("*.safetensors.index.json")) + list(directory.glob("pytorch_model.bin.index.json"))
        weights = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
        if not weights:
            raise RuntimeError("SAM3 weights not cached; prepare facebook/sam3 before running")
        for index in indexes:
            for shard in set(json.loads(index.read_text())["weight_map"].values()):
                if not (directory / shard).is_file():
                    raise FileNotFoundError(directory / shard)
        return {"model": model, "device": "cpu", "validation": "cache and imports; actual load/inference at segmentation"}

    def graspgen():
        from pipeline.graspgen.config import load_config
        from pipeline.graspgen.client import official_client_class
        from pipeline.graspgen.backend import validate_metadata
        cfg = load_config(settings["graspgen_config"], timeout_ms=5000)
        server = cfg["server"]
        cls = official_client_class(cfg["submodule"])
        with cls(host=server["host"], port=server["port"], timeout_ms=server["timeout_ms"], wait_for_server=False) as client:
            metadata = client.get_metadata()
        validate_metadata(metadata, cfg)
        return metadata

    probe("simulation_dependencies", dependencies)
    probe("scene_assets", assets)
    if settings["planner"] == "ollama":
        probe("ollama", ollama)
    if settings["vision"] == "sam3":
        probe("sam3", sam3)
    probe("graspgen", graspgen)
    return results
