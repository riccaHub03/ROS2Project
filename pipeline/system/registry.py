"""Resolve profiles independently of the working directory; preserve backend invariants."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_profile(name, *, environment=None, scene_tool=None, ollama_url=None, llm_model=None, sam3_model=None):
    source = ROOT / "config/system" / f"{name}.yaml"
    settings = yaml.safe_load(source.read_text())
    for key, value in dict(environment=environment, scene_tool=scene_tool, ollama_url=ollama_url,
                           llm_model=llm_model, sam3_model=sam3_model).items():
        if value is not None:
            settings[key] = value
    expected = {"full_system": ("ollama", "sam3", True),
                "graspgen_only": ("deterministic", "ground-truth", False)}[name]
    if (settings["planner"], settings["vision"], settings["strict"]) != expected:
        raise ValueError(f"Profile {name} requires planner/vision/strict={expected}")
    for key in ("scene_tool", "ollama_url", "llm_model", "sam3_model"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"Profile {key} must be a nonempty string")
    for key in ("environment", "graspgen_config"):
        path = Path(settings[key])
        settings[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
        if not Path(settings[key]).is_file():
            raise FileNotFoundError(settings[key])
    # Relative local model paths are relative to the project, HF IDs remain unchanged.
    model_path = ROOT / settings["sam3_model"]
    prepared = ROOT / "data/external/sam3-prepared"
    if settings["sam3_model"] == "facebook/sam3" and (prepared / "model.safetensors").is_file():
        model_path = prepared
    if model_path.is_dir():
        settings["sam3_model"] = str(model_path.resolve())
    settings["profile"] = name
    return settings
