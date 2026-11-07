"""Tiny sequential runner shared by the sim and real pipelines (chain, resume, stop early)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from pipeline.common.report import StageError, StageReport

log = logging.getLogger("pipeline")


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[], StageReport]


def execute(steps: Sequence[Step], *, start_at: Optional[str] = None,
            stop_after: Optional[str] = None) -> List[StageReport]:
    """Run steps in order. Skipped upstream steps must already have reports on disk."""
    names = [s.name for s in steps]
    for label, value in (("--from-stage", start_at), ("--stop-after", stop_after)):
        if value and value not in names:
            raise StageError("pipeline", f"{label} '{value}' is not a stage; choose from {names}")
    first = names.index(start_at) if start_at else 0
    last = names.index(stop_after) if stop_after else len(steps) - 1
    if first > last:
        raise StageError("pipeline", "--from-stage comes after --stop-after")
    reports: List[StageReport] = []
    started = time.perf_counter()
    for step in steps[first:last + 1]:
        log.info("=== stage: %s ===", step.name)
        reports.append(step.run())
    log.info("pipeline completed in %.1fs (%s)", time.perf_counter() - started,
             " -> ".join(r.stage for r in reports))
    return reports
