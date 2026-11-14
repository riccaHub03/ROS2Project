"""Loading of the centralized environment configuration (config/surgical_environment.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


CONFIG_FILENAME = "surgical_environment.yaml"


def default_config_path() -> Path:
    """The versioned YAML next to the code (independent of the data root used for inputs/outputs)."""
    return Path(__file__).resolve().parents[1] / "config" / CONFIG_FILENAME


def load_environment_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the YAML config (default: ``config/surgical_environment.yaml`` of the project).

    The YAML is the single source of truth: there are no in-code defaults to drift out of sync.
    """
    path = Path(config_path) if config_path is not None else default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    from pipeline.robot_model import robot_settings
    config["robot"] = robot_settings(config)
    # Versioned asset paths are relative to the repository, not to /workspace
    # or to the caller's working directory (native Linux and Docker agree).
    instrument = config.get("surgical_instrument", {})
    mesh = instrument.get("stl_path")
    if mesh and mesh != "auto" and not Path(mesh).is_absolute():
        instrument["stl_path"] = str((Path(__file__).resolve().parents[1] / mesh).resolve())
    return config
