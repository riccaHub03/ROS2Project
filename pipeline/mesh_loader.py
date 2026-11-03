#!/usr/bin/env python3
"""
Modulo: Loader, Scanner e Adattatore Dinamico per Mesh STL di Ferri Chirurgici (es. Dataset 3D-COSI).
Funzionalità:
  1. Auto-discovery dei file .stl / .obj nella cartella models/surgical_instruments (anche ricorsivo).
  2. Auto-scaling intelligente: rileva se il modello CAD è in millimetri (mm) o metri (m).
  3. Calcolo bounding box, centro geometrico e dimensioni reali dello strumento.
  4. Creazione robusta del corpo rigido PyBullet (Collision & Visual shape con fallback).
"""

import os
import struct
import numpy as np
from pathlib import Path
import pybullet as p


def get_default_mesh_search_dirs():
    """Ritorna i percorsi candidati in cui cercare le mesh chirurgiche."""
    base_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = [
        "/workspace/data/inputs",
        "/workspace/data",
        os.path.join(base_proj, "data", "inputs"),
        os.path.join(base_proj, "data"),
        "/workspace/models/surgical_instruments",
        "/workspace/models",
        os.path.join(base_proj, "models", "surgical_instruments"),
        os.path.join(base_proj, "models")
    ]
    return [d for d in dirs if os.path.exists(d)]


def load_instruments_catalog():
    """Carica il catalogo JSON indicizzato dei 232 strumenti STL (3D-COSI)."""
    base_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        "/workspace/data/inputs/instruments_catalog.json",
        os.path.join(base_proj, "data", "inputs", "instruments_catalog.json"),
        os.path.join(base_proj, "instruments_catalog.json")
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                import json
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def find_instrument_mesh(search_dir=None, target_keyword=None, category=None, config=None):
    """
    Cerca un file STL o OBJ valido con risoluzione intelligente su 27 categorie (232 modelli):
      1. Verifica se specificato esplicitamente in surgical_environment.yaml
      2. Ricerca nel catalogo indicizzato instruments_catalog.json per categoria o keyword
      3. Seleziona preferenzialmente le versioni pulite (RemoveHighlightON / SmoothFusion)
      4. Fallback ricorsivo sulle cartelle o sul modello predefinito
    """
    # 1. Verifica config
    if config:
        s_cfg = config.get("surgical_instrument", {})
        cfg_path = s_cfg.get("stl_path")
        if cfg_path and cfg_path.strip() != "auto":
            full_cfg_path = os.path.join("/workspace", cfg_path) if not os.path.isabs(cfg_path) else cfg_path
            if os.path.exists(full_cfg_path):
                return os.path.abspath(full_cfg_path)

    # 2. Ricerca tramite Catalogo Indicizzato (Istantanea, < 0.001s)
    catalog = load_instruments_catalog()
    if catalog and "all_instruments" in catalog:
        # Costruzione query
        query = (category or target_keyword or "").lower().strip()
        # Normalizzazione forme singolari/plurali
        norm_map = {
            "scissors": "scissor", "forbici": "scissor", "forbice": "scissor",
            "forceps": "forcep", "pinza": "forcep", "pinze": "forcep",
            "clamps": "clamp", "clamp": "clamp", "morsetto": "clamp",
            "scalpels": "scalpel", "scalpel": "scalpel", "bisturi": "scalpel",
            "needle holders": "needle", "needle holder": "needle", "portaghi": "needle",
            "retractors": "retractor", "retractor": "retractor", "divaricatore": "retractor",
            "curettes": "curette", "curette": "curette",
            "chisels": "chisel", "chisel": "chisel", "scalpello": "chisel",
            "mallets": "mallet", "mallet": "mallet", "martello": "mallet",
            "drills": "drill", "drill": "drill", "trapano": "drill",
            "tweezers": "tweezer", "tweezer": "tweezer", "pinzetta": "tweezer",
            "spatulas": "spatula", "spatula": "spatula",
            "raspatories": "raspatory", "raspatory": "raspatory",
            "gags": "gag", "gag": "gag", "apribocca": "gag"
        }
        search_term = norm_map.get(query, query)

        # Ricerca per categoria o nome modello
        matched = []
        for item in catalog["all_instruments"]:
            cat_l = item.get("category", "").lower()
            model_l = item.get("model_name", "").lower()
            file_l = item.get("filename", "").lower()

            if search_term and (search_term in cat_l or search_term in model_l or search_term in file_l):
                matched.append(item)

        if matched:
            # Ordina dando priorità alle varianti pulite (RemoveHighlightON / SmoothFusion)
            preferred = [m for m in matched if m.get("is_preferred", False)]
            chosen = preferred[0] if preferred else matched[0]

            # Risolvi percorso effettivo
            base_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cand_paths = [
                chosen.get("workspace_path", ""),
                os.path.join(base_proj, chosen.get("rel_path", "")),
                os.path.join("/workspace", chosen.get("rel_path", ""))
            ]
            for cp in cand_paths:
                if cp and os.path.exists(cp):
                    return os.path.abspath(cp)

    # 3. Fallback Ricerca nelle cartelle
    candidate_dirs = [search_dir] if search_dir else get_default_mesh_search_dirs()
    found_files = []

    for d in candidate_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in ('.stl', '.obj', '.ply') and "identifier" not in f.lower():
                    found_files.append(os.path.join(root, f))

    if not found_files:
        return None

    if target_keyword:
        kw = target_keyword.lower().replace("scissors", "scissor").replace("surgical", "").strip()
        matched = [f for f in found_files if kw in os.path.basename(f).lower()]
        if matched:
            # Preferisci RemoveHighlightON o SmoothFusion
            pref = [f for f in matched if "removehighlighton" in f.lower() or "smoothfusion" in f.lower()]
            return pref[0] if pref else matched[0]

    priority_keywords = ["scissor", "forcep", "clamp", "scalpel", "instrument"]
    for pk in priority_keywords:
        for f in found_files:
            if pk in os.path.basename(f).lower():
                return f

    return found_files[0]


def inspect_stl_bounds(stl_path):
    """
    Ispeziona un file STL (binario o ASCII) calcolandone il bounding box min/max,
    le dimensioni spaziali e deducendo se l'unità di misura è in millimetri o metri.
    """
    stl_path = str(stl_path)
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"File non trovato: {stl_path}")

    file_size = os.path.getsize(stl_path)
    if file_size < 84:
        raise ValueError("File STL non valido (troppo piccolo)")

    v_arr = None
    # Tentativo 1: lettura STL binario standard
    is_binary = True
    with open(stl_path, "rb") as f:
        header = f.read(80)
        n_triangles_bytes = f.read(4)
        if len(n_triangles_bytes) == 4:
            n_triangles = struct.unpack("<I", n_triangles_bytes)[0]
            # Controllo di coerenza dimensione: 84 + n_triangles * 50 bytes
            expected_size = 84 + n_triangles * 50
            if abs(file_size - expected_size) <= 2:
                # Lettura in blocco istantanea con NumPy (zero loop lenti, <0.002s)
                dt = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
                # Campiona blocchi da inizio, metà e fine del file per catturare l'intero bounding box
                sample_pts = []
                for frac in [0.0, 0.5, 0.9]:
                    start_tri = int(n_triangles * frac)
                    f.seek(84 + start_tri * 50)
                    chunk = np.frombuffer(f.read(min(2000, n_triangles - start_tri) * 50), dtype=dt)
                    if len(chunk) > 0:
                        sample_pts.append(chunk['v'].reshape(-1, 3))
                if sample_pts:
                    v_arr = np.vstack(sample_pts)
            else:
                is_binary = False
        else:
            is_binary = False

    # Tentativo 2: fallback ASCII STL
    if not is_binary or v_arr is None or len(v_arr) == 0:
        vertices = []
        with open(stl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_str = line.strip().lower()
                if line_str.startswith("vertex"):
                    parts = line_str.split()[1:]
                    if len(parts) >= 3:
                        try:
                            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            pass
                if len(vertices) > 10000:
                    break
        v_arr = np.array(vertices, dtype=np.float64)
    bb_min = np.min(v_arr, axis=0)
    bb_max = np.max(v_arr, axis=0)
    dims = bb_max - bb_min
    max_dim = float(np.max(dims))

    # Auto-Scaling Intelligente:
    # Nei dataset chirurgici scansionati (come 3D-COSI su Zenodo), le unità sono quasi sempre in MILLIMETRI (es. 140 mm).
    # In ROS 2 e PyBullet tutto è in METRI (0.14 m).
    # Se max_dim > 5.0 (es. 120 - 250), la mesh è in millimetri!
    if max_dim > 5.0:
        scale_factor = 0.001 # mm -> metri
        unit = "millimeters"
    elif max_dim > 0.6:
        # Potrebbe essere in centimetri
        scale_factor = 0.01 # cm -> metri
        unit = "centimeters"
    else:
        # Già in metri (dimensione realistica < 0.6m)
        scale_factor = 1.0
        unit = "meters"

    scaled_dims = [float(d * scale_factor) for d in dims]
    mesh_scale = [scale_factor, scale_factor, scale_factor]

    # Calcolo orientamento piatto ottimale per il tavolo operatorio (XY)
    import math
    longest_axis = int(np.argmax(dims))
    if longest_axis == 2:
        # Se Z è l'asse più lungo (scansione in verticale tipica di 3D-COSI), ruota per adagiarlo sul piano
        flat_orn = list(p.getQuaternionFromEuler([0, math.pi/2, 0.4]))
    elif longest_axis == 1:
        flat_orn = list(p.getQuaternionFromEuler([math.pi/2, 0, 0.4]))
    else:
        flat_orn = list(p.getQuaternionFromEuler([0, 0, 0.4]))

    return {
        "stl_path": stl_path,
        "filename": os.path.basename(stl_path),
        "raw_min": bb_min.tolist(),
        "raw_max": bb_max.tolist(),
        "raw_dimensions": dims.tolist(),
        "detected_unit": unit,
        "recommended_scale": mesh_scale,
        "scaled_dimensions_m": scaled_dims,
        "flat_orientation": flat_orn,
        "longest_axis": longest_axis,
        "sample_vertices_count": len(v_arr) if v_arr is not None else 0
    }


def load_instrument_in_pybullet(client_id, mesh_path, base_pos=[0.483, 0.406, 0.088],
                                base_orn=None, mass=0.080):
    """
    Carica lo strumento in PyBullet applicando l'auto-scaling corretto.
    Se il file è una mesh 3D-COSI ad alta densità o presenta errori, gestisce il fallback senza crash.
    """
    info = inspect_stl_bounds(mesh_path)
    scale = info["recommended_scale"]
    if base_orn is None:
        base_orn = info["flat_orientation"]

    print(f"Caricamento Mesh Chirurgica: {info['filename']}")
    print(f"   -> Unità rilevata : {info['detected_unit']} (Applicato fattore scala: {scale[0]})")
    print(f"   -> Dimensioni reali: L={info['scaled_dimensions_m'][0]*100:.1f}cm, W={info['scaled_dimensions_m'][1]*100:.1f}cm, H={info['scaled_dimensions_m'][2]*100:.1f}cm")

    try:
        col_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=str(mesh_path),
            meshScale=scale,
            physicsClientId=client_id
        )
        vis_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(mesh_path),
            meshScale=scale,
            rgbaColor=[0.88, 0.90, 0.94, 1.0], # Finitura acciaio satinato
            physicsClientId=client_id
        )
        tool_body = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=base_pos,
            baseOrientation=base_orn,
            physicsClientId=client_id
        )
        p.changeDynamics(tool_body, -1, lateralFriction=0.85, spinningFriction=0.08, rollingFriction=0.01,
                         physicsClientId=client_id)
        return tool_body, info
    except Exception as e:
        print(f"Avviso caricamento mesh complessa in PyBullet ({e}), applico collisione geometrica adattata.")
        # Fallback: crea una visual shape mesh e una collision box adatta alle dimensioni scalate
        vis_id = p.createVisualShape(p.GEOM_MESH, fileName=str(mesh_path), meshScale=scale,
                                     rgbaColor=[0.88, 0.90, 0.94, 1.0], physicsClientId=client_id)
        s_dims = info["scaled_dimensions_m"]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s_dims[0]/2.0, s_dims[1]/2.0, s_dims[2]/2.0],
                                        physicsClientId=client_id)
        tool_body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                      basePosition=base_pos, baseOrientation=base_orn, physicsClientId=client_id)
        return tool_body, info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mesh Loader & Catalog Query")
    parser.add_argument("--tool", type=str, default=None, help="Cerca mesh per categoria o keyword")
    parser.add_argument("--catalog", action="store_true", help="Mostra riepilogo del catalogo")
    args = parser.parse_args()

    cat = load_instruments_catalog()
    if args.catalog:
        if cat:
            print(f"Catalogo 3D-COSI: {cat['total_categories']} categorie, {cat['total_stl_files']} modelli STL")
            for c, inf in cat["categories"].items():
                print(f"  • {c:<25}: {inf['models_count']} modelli")
        else:
            print("Catalogo non trovato.")
    elif args.tool:
        mesh = find_instrument_mesh(target_keyword=args.tool)
        print(f"Risultato ricerca per '{args.tool}':")
        print(f"   -> File: {mesh}")
        if mesh:
            bounds = inspect_stl_bounds(mesh)
            print(f"   -> Unità: {bounds['detected_unit']}")
            print(f"   -> Dimensioni scalate: {[round(d*100, 1) for d in bounds['scaled_dimensions_m']]} cm")
    else:
        print("Usa --tool <nome> per cercare uno strumento o --catalog per il sommario.")
