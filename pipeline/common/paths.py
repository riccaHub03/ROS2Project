"""Single source of truth for project paths and default report/artifact names.

The project root can be redirected with ``PIPELINE_ROOT`` (used by tests) and the
output directory with ``PIPELINE_OUTPUT_DIR``; otherwise everything lives under
``<repo>/data``.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "PIPELINE_ROOT"
OUTPUT_ENV = "PIPELINE_OUTPUT_DIR"

# Default report file names (one JSON per stage, all inside the output directory).
PLAN = "plan.json"
SIM_CAPTURE = "sim_capture.json"
SIM_SEGMENTATION = "sim_segmentation.json"
SIM_REFERENCE_POSE = "sim_reference_pose.json"
SIM_OVERVIEW = "sim_overview.json"
SIM_DEPROJECTION = "sim_deprojection.json"
REAL_CAPTURE = "real_capture.json"
REAL_SEGMENTATION = "real_segmentation.json"
REAL_DEPROJECTION = "real_deprojection.json"
GRASP = "grasp.json"
TRAJECTORY = "robot_trajectory.json"
VIDEO = "medical_pick_and_place.mp4"


def project_root() -> Path:
    env = os.environ.get(ROOT_ENV)
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


def input_dir() -> Path:
    return project_root() / "data" / "inputs"


def cache_dir() -> Path:
    return input_dir() / ".cache"


def output_dir(override: str | os.PathLike | None = None, *, create: bool = True) -> Path:
    """Resolve the output directory: explicit argument > env var > ``data/outputs``."""
    if override:
        path = Path(override).resolve()
    elif os.environ.get(OUTPUT_ENV):
        path = Path(os.environ[OUTPUT_ENV]).resolve()
    else:
        path = project_root() / "data" / "outputs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def report_path(name: str, out_dir: str | os.PathLike | None = None) -> Path:
    """Default location of a stage report inside the output directory."""
    return output_dir(out_dir) / name


def resolve(explicit: str | os.PathLike | None, default_name: str,
            out_dir: str | os.PathLike | None = None) -> Path:
    """Return ``explicit`` if given, else the default report ``default_name`` in ``out_dir``."""
    return Path(explicit) if explicit else report_path(default_name, out_dir)


def output_report(explicit: str | os.PathLike | None, default_name: str,
                  out_dir: str | os.PathLike | None = None) -> Path:
    """Where a stage WRITES its report: always inside the output directory.

    ``--report NAME`` is a file name relative to the output directory; only an absolute
    path can place the report elsewhere.
    """
    base = output_dir(out_dir)
    if not explicit:
        return base / default_name
    explicit = Path(explicit)
    return explicit if explicit.is_absolute() else base / explicit
