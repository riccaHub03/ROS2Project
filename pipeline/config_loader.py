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
    return yaml.safe_load(path.read_text(encoding="utf-8"))
