"""Shared CLI scaffolding so every stage is runnable on its own."""

from __future__ import annotations

import argparse
import logging
from typing import Callable, Optional, Sequence

from pipeline.common.report import StageError, StageReport

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=LOG_FORMAT,
                        datefmt="%H:%M:%S", force=True)


def stage_parser(description: str, *, config: bool = True) -> argparse.ArgumentParser:
    """Parser with the options common to all stages."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-dir", default=None, help="Directory for reports and artifacts "
                        "(default: $PIPELINE_OUTPUT_DIR or data/outputs)")
    parser.add_argument("--report", default=None, help="Name of the JSON report this stage writes, "
                        "created inside --out-dir (default: the stage's standard name)")
    if config:
        parser.add_argument("--config", default=None, help="surgical_environment.yaml path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def run_cli(parser: argparse.ArgumentParser, run: Callable[[argparse.Namespace], StageReport],
            argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, run the stage and map failures to a non-zero exit code."""
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    try:
        run(args)
    except StageError as exc:
        logging.getLogger("cli").error("%s", exc)
        return 1
    return 0
