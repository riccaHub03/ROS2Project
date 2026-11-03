#!/usr/bin/env python3
"""
Generatore di Modelli 3D STL Parametrici per Ferri Chirurgici
Crea mesh STL pulite e watertight di:
  1. Forbici chirurgiche Mayo/Metzenbaum (surgical_scissors.stl)
  2. Bisturi chirurgico con lama #10 (scalpel.stl)
"""

import os
import math
import struct
import numpy as np

def write_binary_stl(filename, triangles):
    """
    Scrive un file binario STL standard.
    triangles: array-like di forma (N, 3, 3) contenente per ogni triangolo i 3 vertici [v0, v1, v2].
    """
    triangles = np.asarray(triangles, dtype=np.float32)
    n_triangles = len(triangles)
    
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "wb") as f:
        # Header di 80 byte
        header = f"Binary STL Surgical Instrument - {os.path.basename(filename)}".ljust(80)[:80].encode('ascii')
        f.write(header)
        # Numero di triangoli (uint32)
        f.write(struct.pack("<I", n_triangles))
        
        # Calcolo normali
        v0 = triangles[:, 0, :]
        v1 = triangles[:, 1, :]
        v2 = triangles[:, 2, :]
        normals = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        normals = normals / norm
        
        # Buffer per triangoli: ogni triangolo = 12 float (normale + 3 vertici) + uint16 attribute
        for i in range(n_triangles):
            n = normals[i]
            pts = triangles[i]
            # normal (3f), v0 (3f), v1 (3f), v2 (3f)
            data = struct.pack("<12fH", n[0], n[1], n[2],
                               pts[0, 0], pts[0, 1], pts[0, 2],
                               pts[1, 0], pts[1, 1], pts[1, 2],
                               pts[2, 0], pts[2, 1], pts[2, 2],
                               0)
            f.write(data)
    print(f"Mesh STL salvata: {filename} ({n_triangles:,} triangoli)")

def create_box_triangles(center, size):
    """Genera 12 triangoli per un parallelepipedo centrato in center con dimensioni size [dx, dy, dz]."""
    cx, cy, cz = center
    dx, dy, dz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    
    # 8 vertici
    v = [
        [cx - dx, cy - dy, cz - dz], # 0
        [cx + dx, cy - dy, cz - dz], # 1
        [cx + dx, cy + dy, cz - dz], # 2
        [cx - dx, cy + dy, cz - dz], # 3
        [cx - dx, cy - dy, cz + dz], # 4
        [cx + dx, cy - dy, cz + dz], # 5
        [cx + dx, cy + dy, cz + dz], # 6
        [cx - dx, cy + dy, cz + dz], # 7
    ]
    # 6 facce x 2 triangoli
    faces = [
        [0, 1, 2], [0, 2, 3], # Bottom
        [4, 6, 5], [4, 7, 6], # Top
        [0, 5, 1], [0, 4, 5], # Front (-y)
        [2, 7, 3], [2, 6, 7], # Back (+y)
        [0, 3, 7], [0, 7, 4], # Left (-x)
        [1, 5, 6], [1, 6, 2], # Right (+x)
    ]
    return [[v[idx[0]], v[idx[1]], v[idx[2]]] for idx in faces]

def create_cylinder_triangles(center, radius, length, axis='z', num_segments=16):
    """Genera triangoli per un cilindro pieno."""
    cx, cy, cz = center
    half_l = length / 2.0
    triangles = []
    
    angles = np.linspace(0, 2 * math.pi, num_segments, endpoint=False)
    for i in range(num_segments):
        a1 = angles[i]
        a2 = angles[(i + 1) % num_segments]
        
        c1, s1 = math.cos(a1) * radius, math.sin(a1) * radius
        c2, s2 = math.cos(a2) * radius, math.sin(a2) * radius
        
        if axis == 'z':
            p1_bot, p2_bot = [cx + c1, cy + s1, cz - half_l], [cx + c2, cy + s2, cz - half_l]
            p1_top, p2_top = [cx + c1, cy + s1, cz + half_l], [cx + c2, cy + s2, cz + half_l]
            center_bot, center_top = [cx, cy, cz - half_l], [cx, cy, cz + half_l]
        elif axis == 'x':
            p1_bot, p2_bot = [cx - half_l, cy + c1, cz + s1], [cx - half_l, cy + c2, cz + s2]
            p1_top, p2_top = [cx + half_l, cy + c1, cz + s1], [cx + half_l, cy + c2, cz + s2]
            center_bot, center_top = [cx - half_l, cy, cz], [cx + half_l, cy, cz]
        else: # 'y'
            p1_bot, p2_bot = [cx + c1, cy - half_l, cz + s1], [cx + c2, cy - half_l, cz + s2]
            p1_top, p2_top = [cx + c1, cy + half_l, cz + s1], [cx + c2, cy + half_l, cz + s2]
            center_bot, center_top = [cx, cy - half_l, cz], [cx, cy + half_l, cz]
            
        # Faccia laterale (2 triangoli)
        triangles.append([p1_bot, p2_bot, p2_top])
        triangles.append([p1_bot, p2_top, p1_top])
        # Tappi
        triangles.append([center_bot, p2_bot, p1_bot])
        triangles.append([center_top, p1_top, p2_top])
        
    return triangles

def create_ring_triangles(center, r_outer, r_inner, thickness, num_segments=24):
    """Genera triangoli per un anello cavo (impugnatura dita forbici chirurgiche)."""
    cx, cy, cz = center
    half_t = thickness / 2.0
    triangles = []
    
    angles = np.linspace(0, 2 * math.pi, num_segments, endpoint=False)
    for i in range(num_segments):
        a1 = angles[i]
        a2 = angles[(i + 1) % num_segments]
        
        co1, so1 = math.cos(a1) * r_outer, math.sin(a1) * r_outer
        co2, so2 = math.cos(a2) * r_outer, math.sin(a2) * r_outer
        ci1, si1 = math.cos(a1) * r_inner, math.sin(a1) * r_inner
        ci2, si2 = math.cos(a2) * r_inner, math.sin(a2) * r_inner
        
        # Vertici superiori (+z) e inferiori (-z)
        p_out_t1 = [cx + co1, cy + so1, cz + half_t]
        p_out_t2 = [cx + co2, cy + so2, cz + half_t]
        p_out_b1 = [cx + co1, cy + so1, cz - half_t]
        p_out_b2 = [cx + co2, cy + so2, cz - half_t]
        
        p_in_t1 = [cx + ci1, cy + si1, cz + half_t]
        p_in_t2 = [cx + ci2, cy + si2, cz + half_t]
        p_in_b1 = [cx + ci1, cy + si1, cz - half_t]
        p_in_b2 = [cx + ci2, cy + si2, cz - half_t]
        
        # Tappo superiore (+z)
        triangles.append([p_out_t1, p_in_t1, p_in_t2])
        triangles.append([p_out_t1, p_in_t2, p_out_t2])
        
        # Tappo inferiore (-z)
        triangles.append([p_out_b1, p_in_b2, p_in_b1])
        triangles.append([p_out_b1, p_out_b2, p_in_b2])
        
        # Parete esterna
        triangles.append([p_out_b1, p_out_b2, p_out_t2])
        triangles.append([p_out_b1, p_out_t2, p_out_t1])
        
        # Parete interna
        triangles.append([p_in_b1, p_in_t2, p_in_b2])
        triangles.append([p_in_b1, p_in_t1, p_in_t2])
        
    return triangles

def generate_surgical_scissors_stl(output_path="models/surgical_instruments/surgical_scissors.stl"):
    """
    Genera il modello 3D realistico di una forbice chirurgica (Mayo/Metzenbaum).
    Dimensioni reali:
      - Lunghezza totale: 140 mm (da -65mm anelli a +75mm punte lame)
      - Larghezza massima anelli: 42 mm (+/- 21mm)
      - Spessore medio corpo: 4.0 mm
      - Perno centrale di rotazione: a x = 0.0 mm
      - Zona di presa ottimale (shank/fusto): x = -20mm a -5mm
    """
    triangles = []
    
    # 1. Anello impugnatura destro (finger ring 1) a x = -52mm, y = +15mm
    triangles.extend(create_ring_triangles(center=[-0.052, 0.015, 0.0], r_outer=0.013, r_inner=0.009, thickness=0.0035))
    
    # 2. Anello impugnatura sinistro (finger ring 2) a x = -52mm, y = -15mm
    triangles.extend(create_ring_triangles(center=[-0.052, -0.015, 0.0], r_outer=0.013, r_inner=0.009, thickness=0.0035))
    
    # 3. Fusti di raccordo / Shanks (dall'anello al perno centrale)
    # Shank 1 (destro)
    triangles.extend(create_box_triangles(center=[-0.024, 0.008, 0.0], size=[0.045, 0.004, 0.0035]))
    # Shank 2 (sinistro)
    triangles.extend(create_box_triangles(center=[-0.024, -0.008, 0.0], size=[0.045, 0.004, 0.0035]))
    
    # 4. Perno centrale (pivot joint screw)
    triangles.extend(create_cylinder_triangles(center=[0.0, 0.0, 0.0], radius=0.0055, length=0.0055, axis='z', num_segments=20))
    
    # 5. Lame chirurgiche rastremate (blades) da x = 0 a x = +0.065 m
    # Lama 1 (superiore)
    triangles.extend(create_box_triangles(center=[0.020, 0.0035, 0.001], size=[0.038, 0.0045, 0.0028]))
    triangles.extend(create_box_triangles(center=[0.050, 0.0020, 0.001], size=[0.028, 0.0028, 0.0020]))
    # Lama 2 (inferiore)
    triangles.extend(create_box_triangles(center=[0.020, -0.0035, -0.001], size=[0.038, 0.0045, 0.0028]))
    triangles.extend(create_box_triangles(center=[0.050, -0.0020, -0.001], size=[0.028, 0.0028, 0.0020]))
    
    # Punta delle lame (tips)
    triangles.extend(create_box_triangles(center=[0.066, 0.000, 0.0], size=[0.006, 0.0025, 0.0020]))
    
    write_binary_stl(output_path, triangles)
    return output_path

def generate_scalpel_stl(output_path="models/surgical_instruments/scalpel.stl"):
    """
    Genera il modello 3D di un bisturi chirurgico (Manico n.3 + Lama n.10).
    Lunghezza totale: 135 mm, spessore 3 mm.
    """
    triangles = []
    # Manico scanalato
    triangles.extend(create_box_triangles(center=[-0.025, 0.0, 0.0], size=[0.085, 0.010, 0.0028]))
    # Collo bisturi
    triangles.extend(create_box_triangles(center=[0.022, 0.0, 0.0], size=[0.018, 0.006, 0.0022]))
    # Lama curvilinea
    triangles.extend(create_box_triangles(center=[0.040, 0.0015, 0.0], size=[0.025, 0.008, 0.0012]))
    triangles.extend(create_box_triangles(center=[0.055, 0.0010, 0.0], size=[0.012, 0.004, 0.0010]))
    
    write_binary_stl(output_path, triangles)
    return output_path

if __name__ == "__main__":
    base_dir = "/workspace" if os.path.exists("/workspace") else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sciss_path = os.path.join(base_dir, "models/surgical_instruments/surgical_scissors.stl")
    scalp_path = os.path.join(base_dir, "models/surgical_instruments/scalpel.stl")
    
    generate_surgical_scissors_stl(sciss_path)
    generate_scalpel_stl(scalp_path)
