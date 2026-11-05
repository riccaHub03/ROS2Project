"""ASCII PLY reading/writing for point clouds (single implementation for all stages)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def write_ply(path: Path | str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if colors is not None and len(colors) != len(points):
        raise ValueError(f"colors ({len(colors)}) and points ({len(points)}) must have the same length")
    header = ["ply", "format ascii 1.0", f"element vertex {len(points)}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        if colors is None:
            np.savetxt(fh, points, fmt="%.6f")
        else:
            rows = np.hstack([points, np.asarray(colors, dtype=np.float64).reshape(-1, 3)])
            np.savetxt(fh, rows, fmt="%.6f %.6f %.6f %d %d %d")
    return path


def read_ply(path: Path | str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read vertices (and optional uchar RGB) from an ASCII PLY."""
    pts, cols, in_header = [], [], True
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if in_header:
                in_header = line.strip() != "end_header"
                continue
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(parts) >= 6:
                cols.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return (np.array(pts, dtype=np.float64).reshape(-1, 3),
            np.array(cols, dtype=np.uint8) if cols else None)
