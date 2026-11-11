#!/usr/bin/env python3
"""Rende la vista overhead del layout (tavolo, vassoi, collezione completa) per un controllo visivo.

Non e' un test automatico: esegui  python3 scripts/render_layout_check.py
Output: <data/outputs>/layout_check_render.png
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image

cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Tavolo
tab_pos = [0.60, 0.0, -0.05]
tab_dims = [0.85, 1.60, 0.20]
c_tab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tab_dims[0]/2, tab_dims[1]/2, tab_dims[2]/2])
v_tab = p.createVisualShape(p.GEOM_BOX, halfExtents=[tab_dims[0]/2, tab_dims[1]/2, tab_dims[2]/2],
                            rgbaColor=[0.15, 0.35, 0.60, 1.0])
p.createMultiBody(0, c_tab, v_tab, tab_pos)

# Tray 1 (Right: +Y)
tr1_pos = [0.60, 0.42, 0.07]
tr1_dims = [0.62, 0.68, 0.04]
c_tr1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tr1_dims[0]/2, tr1_dims[1]/2, tr1_dims[2]/2])
v_tr1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[tr1_dims[0]/2, tr1_dims[1]/2, tr1_dims[2]/2],
                            rgbaColor=[0.82, 0.84, 0.88, 1.0])
p.createMultiBody(0, c_tr1, v_tr1, tr1_pos)

# Tray 2 (Left: -Y)
tr2_pos = [0.60, -0.42, 0.07]
tr2_dims = [0.62, 0.68, 0.04]
c_tr2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tr2_dims[0]/2, tr2_dims[1]/2, tr2_dims[2]/2])
v_tr2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[tr2_dims[0]/2, tr2_dims[1]/2, tr2_dims[2]/2],
                            rgbaColor=[0.82, 0.84, 0.88, 1.0])
p.createMultiBody(0, c_tr2, v_tr2, tr2_pos)

# All Instruments OBJ inside Tray 2
mesh_file = os.path.join(ROOT, "data", "inputs", "All_Instruments_combined.obj")
# Instruments rotated to lay flat:
# Local Z became -Y, local Y became +Z
# Rotation matrix: col0=[1,0,0], col1=[0,0,1], col2=[0,-1,0]
# Quaternion for this rotation:
# Trace = 1 + 0 + 0 = 1, so w = 0.5 * sqrt(2) = 0.7071, x = (1 - (-1))*s = 2 * (0.25/0.7071) = 0.7071
q_rot = [0.7071068, 0.0, 0.0, 0.7071068]
# Place exactly on top surface of tray 2:
# Tray top is at 0.07 + 0.04/2 = 0.09m
tool_pos = [0.60, -0.42, 0.095]
v_tool = p.createVisualShape(p.GEOM_MESH, fileName=mesh_file, meshScale=[0.01, 0.01, 0.01],
                             rgbaColor=[0.25, 0.25, 0.28, 1.0])
p.createMultiBody(0, -1, v_tool, tool_pos, q_rot)

# Camera directly overhead tray 2
cam_pos = [0.60, -0.42, 1.15]
cam_target = [0.60, -0.42, 0.07]
up_vector = [-1.0, 0.0, 0.0] # Top of image is -X (towards robot base)

view_mat = p.computeViewMatrix(cam_pos, cam_target, up_vector)
proj_mat = p.computeProjectionMatrixFOV(fov=48.0, aspect=640/480, nearVal=0.1, farVal=2.5)

w, h, rgb, depth, seg = p.getCameraImage(640, 480, view_mat, proj_mat, renderer=p.ER_TINY_RENDERER)
img = Image.fromarray(rgb[:, :, :3])
from pipeline.common import paths
out_png = paths.output_dir() / "layout_check_render.png"
img.save(out_png)
print(f"Rendered to {out_png}")
p.disconnect()
