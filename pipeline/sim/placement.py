"""Posizionamento casuale dello strumento nel vassoio, dentro l'area raggiungibile dal KUKA."""

from __future__ import annotations

import math

import numpy as np

# Corona orizzontale (attorno alla base del KUKA) in cui la presa dall'alto e' stata
# verificata con IK. Margini interni rispetto ai limiti misurati (0.41 - 0.78 m).
GRASP_R_MIN = 0.45
GRASP_R_MAX = 0.74
TRAY_WALL_MARGIN = 0.02  # m di spazio libero tra l'ingombro dello strumento e il bordo del vassoio
MAX_ATTEMPTS = 500


def rotated_half_extents(dims_xy, yaw):
    """Semi-ingombro in x e y del rettangolo dello strumento ruotato di yaw."""
    hx, hy = dims_xy[0] / 2.0, dims_xy[1] / 2.0
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return hx * c + hy * s, hx * s + hy * c


def sample_tool_pose(tray_position, tray_dimensions, tool_dims_xy, seed=None,
                     yaw_range=(-math.pi / 2, math.pi / 2)):
    """Ritorna (x, y, yaw): posa casuale con tutto l'ingombro dentro il vassoio e centro raggiungibile.

    Se non esiste una posa valida (strumento troppo grande) ritorna il centro del vassoio.
    """
    rng = np.random.default_rng(seed)
    cx, cy = float(tray_position[0]), float(tray_position[1])
    half_x, half_y = tray_dimensions[0] / 2.0, tray_dimensions[1] / 2.0
    for _ in range(MAX_ATTEMPTS):
        yaw = float(rng.uniform(*yaw_range))
        ex, ey = rotated_half_extents(tool_dims_xy, yaw)
        lim_x = half_x - ex - TRAY_WALL_MARGIN
        lim_y = half_y - ey - TRAY_WALL_MARGIN
        if lim_x <= 0 or lim_y <= 0:
            continue
        x = cx + float(rng.uniform(-lim_x, lim_x))
        y = cy + float(rng.uniform(-lim_y, lim_y))
        if GRASP_R_MIN <= math.hypot(x, y) <= GRASP_R_MAX:
            return x, y, yaw
    return cx, cy, 0.0
