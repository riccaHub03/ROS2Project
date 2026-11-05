"""Uniform JSON report envelope shared by every pipeline stage.

Every stage reads its upstream reports with :meth:`StageReport.load` and writes its own
through the :func:`stage_run` context manager, which also records timing and, on failure,
persists a ``FAILED`` report (with the error and traceback) before re-raising.

Envelope::

    {
      "schema_version": 2, "stage": "...", "status": "COMPLETED | FAILED (| RUNNING | CANCELLED for run manifests)",
      "timestamp": "...", "duration_s": 1.2,
      "inputs":    {"plan": "<path of upstream report>", ...},
      "artifacts": {"mask_png": "<generated file>", ...},
      "data":      {... stage specific payload ...},
      "warnings":  [...], "error": null | {"type", "message", "traceback"}
    }
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

SCHEMA_VERSION = 2
COMPLETED = "COMPLETED"
FAILED = "FAILED"
RUNNING = "RUNNING"        # long-lived reports (e.g. the graphical run manifest)
CANCELLED = "CANCELLED"


class StageError(RuntimeError):
    """A stage failed; ``stage`` and ``report_path`` identify where to look."""

    def __init__(self, stage: str, message: str, report_path: Optional[Path] = None):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.report_path = report_path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class StageReport:
    stage: str
    status: str = COMPLETED
    inputs: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[Dict[str, str]] = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    duration_s: Optional[float] = None
    schema_version: int = SCHEMA_VERSION

    # -- persistence -------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version, "stage": self.stage, "status": self.status,
            "timestamp": self.timestamp, "duration_s": self.duration_s,
            "inputs": self.inputs, "artifacts": self.artifacts, "data": self.data,
            "warnings": self.warnings, "error": self.error,
        }

    def save(self, path: Path | str) -> Path:
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path | str, *, expect_stage: Optional[str] = None,
             require_completed: bool = True) -> "StageReport":
        """Load an upstream report, failing with an actionable message if unusable."""
        path = Path(path)
        label = expect_stage or path.stem
        if not path.exists():
            raise StageError(label, f"upstream report not found: {path} (run that stage first)")
        raw = read_json(path)
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise StageError(label, f"{path} has schema_version {raw.get('schema_version')!r}, "
                                    f"expected {SCHEMA_VERSION}; regenerate it with the current pipeline")
        report = cls(stage=raw["stage"], status=raw["status"], inputs=raw.get("inputs", {}),
                     artifacts=raw.get("artifacts", {}), data=raw.get("data", {}),
                     warnings=raw.get("warnings", []), error=raw.get("error"),
                     timestamp=raw.get("timestamp", ""), duration_s=raw.get("duration_s"))
        if expect_stage and report.stage != expect_stage:
            raise StageError(expect_stage, f"{path} is a '{report.stage}' report, expected '{expect_stage}'")
        if require_completed and report.status != COMPLETED:
            raise StageError(report.stage, f"upstream stage did not complete (status={report.status}); see {path}")
        return report

    # -- typed accessors -----------------------------------------------------------
    def require(self, *keys: str) -> Any:
        """Fetch a (nested) key from ``data`` with a clear error if it is missing."""
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise StageError(self.stage, f"report is missing data.{'.'.join(keys)}")
            node = node[key]
        return node

    def artifact(self, name: str, *, must_exist: bool = True) -> Path:
        if name not in self.artifacts:
            raise StageError(self.stage, f"report has no artifact '{name}'")
        path = Path(self.artifacts[name])
        if must_exist and not path.exists():
            raise StageError(self.stage, f"artifact '{name}' is missing on disk: {path}")
        return path

    def warn(self, logger: logging.Logger, message: str) -> None:
        logger.warning(message)
        self.warnings.append(message)


@contextmanager
def stage_run(stage: str, report_path: Path | str, *, inputs: Optional[Dict[str, Any]] = None,
              logger: Optional[logging.Logger] = None) -> Iterator[StageReport]:
    """Run a stage body, always leaving a report behind (COMPLETED or FAILED)."""
    log = logger or logging.getLogger(stage)
    report_path = Path(report_path)
    report = StageReport(stage=stage, inputs={k: str(v) for k, v in (inputs or {}).items() if v is not None})
    started = time.perf_counter()
    log.info("start")
    try:
        yield report
    except Exception as exc:
        report.status = FAILED
        report.error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        report.duration_s = round(time.perf_counter() - started, 3)
        report.save(report_path)
        log.error("failed: %s (report: %s)", exc, report_path)
        if isinstance(exc, StageError):
            exc.report_path = exc.report_path or report_path
            raise
        raise StageError(stage, f"{type(exc).__name__}: {exc}", report_path) from exc
    report.duration_s = round(time.perf_counter() - started, 3)
    report.save(report_path)
    log.info("done in %.2fs -> %s", report.duration_s, report_path)
