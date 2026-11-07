"""Remove generated files from the output directory (reports, images, clouds, videos).

Dry run by default: lists what would be deleted. Add ``--yes`` to delete. ``.gitkeep`` and
sub-directories are never touched; a stale ``.graphical.lock`` is removed only if no
graphical run holds it.

Run:  python -m pipeline.clean            # preview
      python -m pipeline.clean --yes      # delete
"""

from __future__ import annotations

import argparse
import fcntl
import logging
from pathlib import Path
from typing import List, Optional

from pipeline.common import paths
from pipeline.common.cli import setup_logging

log = logging.getLogger("clean")
KEEP = {".gitkeep", "README.md"}


def lock_is_free(lock: Path) -> bool:
    try:
        with lock.open("w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
    except BlockingIOError:
        return False


def generated_files(out_dir: Path) -> List[Path]:
    files = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.name in KEEP:
            continue
        if path.name == ".graphical.lock" and not lock_is_free(path):
            log.warning("a graphical run is active: keeping %s", path.name)
            continue
        files.append(path)
    return files


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--yes", action="store_true", help="Actually delete (default: dry run)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = paths.output_dir(args.out_dir)
    files = generated_files(out_dir)
    size = sum(f.stat().st_size for f in files)
    for f in files:
        log.info("%s %s", "delete" if args.yes else "would delete", f.name)
        if args.yes:
            f.unlink()
    log.info("%s %d files (%.1f MB) in %s", "removed" if args.yes else "would remove", len(files), size / 1e6, out_dir)
    if files and not args.yes:
        log.info("dry run: re-run with --yes to delete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
