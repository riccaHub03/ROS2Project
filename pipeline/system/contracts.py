"""Validation at the boundaries of the complete single-object simulation."""
from __future__ import annotations

import numpy as np

from pipeline.common.report import StageError
from pipeline.config_loader import load_environment_config


def validate_task(task, config, expected_instrument=None):
    if not isinstance(task, dict):
        raise StageError("plan", "LLM response must be a JSON object")
    keys = ("action", "target_instrument", "target_keyword", "source_tray", "destination_tray")
    if any(not isinstance(task.get(k), str) or not task[k].strip() for k in keys):
        raise StageError("plan", "LLM response requires nonempty strings: " + ", ".join(keys))
    if task["action"] != "pick_and_place":
        raise StageError("plan", "This simulation supports only action=pick_and_place")
    trays = config["environment"]["trays"]
    if any(task[k] not in trays for k in ("source_tray", "destination_tray")):
        raise StageError("plan", "LLM selected an unknown tray; automatic normalization disabled")
    if task["source_tray"] == task["destination_tray"]:
        raise StageError("plan", "Source and destination must differ")
    # A configured CAD scene must never silently substitute a different requested tool.
    if expected_instrument and task["target_instrument"].strip().casefold() != expected_instrument.casefold():
        raise StageError("plan", f"Scene contains {expected_instrument!r}, but LLM selected {task['target_instrument']!r}")
    from pipeline.plan import extract_keyword
    if task["target_keyword"].strip().casefold() != extract_keyword(task["target_instrument"]):
        raise StageError("plan", "LLM keyword does not match the selected instrument")


def validate_reconstruction(data, config_path):
    if data.get("centroid_source") != "optical" or data.get("pointcloud_units") != "m":
        raise StageError("reconstruction", "Full system requires optical reconstruction in metres")
    from pipeline.common.scene import get_tray
    # Source tray is checked from the actual task, not the instrument reference pose.
    from pipeline.common import paths
    from pipeline.common.report import StageReport
    task = StageReport.load(paths.output_dir() / paths.PLAN).data
    point = np.asarray(data["frames"]["world_frame"], dtype=float)
    tray = get_tray(load_environment_config(config_path), task["source_tray"])
    if not np.isfinite(point).all() or not tray.contains_xy(point, 0.05):
        raise StageError("reconstruction", "Optical centroid outside source tray; reference-pose fallback disabled")


def validate_segmentation(data):
    score = data.get("confidence_score")
    if (data.get("segmentation_backend") != "sam3" or not isinstance(score, (int, float))
            or not np.isfinite(score) or not 0.15 <= score <= 1
            or data.get("active_pixel_count", 0) < 15):
        raise StageError("segmentation", "Full system requires a nonempty SAM3 mask and a finite accepted score")
