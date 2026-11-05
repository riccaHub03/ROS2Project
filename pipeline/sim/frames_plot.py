"""Diagnostic two-panel figure: camera view with projected frames + 3D workspace diagram."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.common.scene import Config, all_trays
from pipeline.frame_transformer import FrameTransformer

log = logging.getLogger("sim_deprojection")


def _rect_xy(center, dims):
    """Closed 2D outline (x, y lists) of a box footprint."""
    cx, cy, dx, dy = center[0], center[1], dims[0] / 2, dims[1] / 2
    return [cx - dx, cx + dx, cx + dx, cx - dx, cx - dx], [cy - dy, cy - dy, cy + dy, cy + dy, cy - dy]


def render_frames_figure(path: Path, rgb: np.ndarray, transformer: FrameTransformer, centroid_base: np.ndarray,
                         cam_pos, config: Config, tool_frame_label: Optional[str] = "Target Instrument Frame") -> None:
    env = config.get("environment", {})
    trays = all_trays(config)
    base = np.array(env.get("kuka_base", {}).get("position_xyz_m", [0.0, 0.0, 0.0]))
    cx, cy, fx, fy = transformer.cx, transformer.cy, transformer.fx, transformer.fy

    def project(world_pt):
        pc = transformer.transform_world_to_camera(world_pt)
        return None if pc[2] <= 0.01 else np.array([fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy])

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rgb)

    def draw_frame_2d(origin, label, axis_len, color):
        p0 = project(origin)
        if p0 is None:
            return
        for i, (name, col) in enumerate(zip("XYZ", ("red", "lime", "cyan"))):
            end = project(origin + np.eye(3)[:, i] * axis_len)
            if end is not None:
                ax1.annotate("", xy=tuple(end), xytext=tuple(p0), arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
                ax1.text(end[0], end[1], name, color=col, fontsize=9, fontweight="bold", ha="center", va="center")
        ax1.scatter([p0[0]], [p0[1]], c=color, s=80, edgecolors="black", zorder=6)
        ax1.text(p0[0] + 10, p0[1] - 10, label, color="white", fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75))

    for tray in trays:
        draw_frame_2d(np.array(tray.position), f"{tray.name} Frame", 0.10, "gold")
    draw_frame_2d(centroid_base, tool_frame_label, 0.09, "orange")

    ox, oy = 65, 65  # camera optical frame widget
    ax1.annotate("", xy=(ox + 45, oy), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", color="red", lw=3))
    ax1.text(ox + 50, oy, "X_opt (right)", color="red", fontsize=8, fontweight="bold")
    ax1.annotate("", xy=(ox, oy + 45), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", color="lime", lw=3))
    ax1.text(ox, oy + 55, "Y_opt (down)", color="lime", fontsize=8, fontweight="bold")
    ax1.scatter([ox], [oy], c="cyan", s=90, edgecolors="black", zorder=7)
    ax1.text(ox - 55, oy - 12, "Camera Optical\nFrame", color="white", fontsize=8, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="navy", alpha=0.85))
    tool_px = project(centroid_base)
    if tool_px is not None:
        ax1.annotate("", xy=(int(cx), 20), xytext=tuple(tool_px),
                     arrowprops=dict(arrowstyle="->", color="magenta", lw=2, linestyle="--"))
        ax1.text(int(cx), 10, "Towards robot base (lbr_link_0)", color="magenta", fontsize=9, fontweight="bold",
                 ha="center", bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.8))
    ax1.set_title("CAMERA VIEW WITH FRAME REFERENCES", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    def draw_frame_3d(origin, R, label, scale):
        for i, col in enumerate(("red", "lime", "blue")):
            v = R[:, i] * scale
            ax2.quiver(*origin, *v, color=col, lw=2)
        ax2.scatter(*[[c] for c in origin], s=40, c="black")
        ax2.text(origin[0], origin[1], origin[2] + scale * 0.25, label, fontsize=8, fontweight="bold")

    eye = np.eye(3)
    draw_frame_3d(base, eye, "Robot base (lbr_link_0)", 0.18)
    draw_frame_3d(np.asarray(cam_pos), transformer.R_base_opt, "Camera Optical Frame", 0.16)
    for tray in trays:
        draw_frame_3d(np.array(tray.position), eye, f"{tray.key} Frame", 0.12)
        bx, by = _rect_xy(tray.position, tray.dimensions)
        ax2.plot(bx, by, [tray.top_z] * 5, color="gray", lw=1.2, linestyle="--", label=f"Tray {tray.key}")
    draw_frame_3d(centroid_base, eye, "Instrument Frame", 0.10)
    ax2.plot(*zip(base, cam_pos), "k--", alpha=0.4, label="T_base_cam")
    ax2.plot(*zip(cam_pos, centroid_base), "c-", lw=2, label="Deprojection optical axis")
    ax2.plot(*zip(base, centroid_base), "m-.", lw=1.5, label="T_base_tool")

    table = env.get("surgical_table", {})
    tx, ty = _rect_xy(table.get("center_xyz_m", [0.60, 0.0, -0.05]), table.get("dimensions_xyz_m", [0.85, 1.60, 0.20]))
    ax2.plot(tx, ty, [float(table.get("surface_z_m", 0.05))] * 5, "b-", lw=1.8, alpha=0.6, label="Operating table")
    ax2.set(xlabel="X (base) [m]", ylabel="Y (base) [m]", zlabel="Z (base) [m]")
    ax2.set_title("3D WORKSPACE FRAME DIAGRAM", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.view_init(elev=28, azim=-55)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("frames figure saved: %s", path)
