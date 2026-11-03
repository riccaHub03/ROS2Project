#!/usr/bin/env python3
"""
Costruttore del Catalogo Strumenti Chirurgici 3D (3D-COSI Dataset).
Scansiona ricorsivamente data/inputs/ e indicizza tutti i 232 file STL suddivisi per:
  - 27 Categorie Cliniche
  - Modelli e Varianti (RemoveHighlightON / SmoothFusion / Sample)
  - Dimensioni file e percorsi relativi/assoluti

Genera: data/inputs/instruments_catalog.json
"""

import os
import sys
import json
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def build_catalog(base_dir=None, output_json_path=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    inputs_dir = os.path.join(base_dir, "data", "inputs")
    ds_root = os.path.join(inputs_dir, "c_Surgical Instruments", "Surgical Instruments")
    if not os.path.exists(ds_root):
        # Fallback se la cartella è direttamente in inputs
        ds_root = os.path.join(inputs_dir, "Surgical Instruments")

    catalog = {
        "dataset_name": "3D-COSI Surgical Instruments Collection",
        "total_categories": 0,
        "total_stl_files": 0,
        "categories": {},
        "all_instruments": []
    }

    if not os.path.exists(ds_root):
        print(f"Cartella dataset non trovata: {ds_root}")
        return catalog

    categories = sorted([d for d in os.listdir(ds_root) if os.path.isdir(os.path.join(ds_root, d))])
    catalog["total_categories"] = len(categories)

    total_files = 0

    for cat in categories:
        cat_path = os.path.join(ds_root, cat)
        catalog["categories"][cat] = {
            "category_name": cat,
            "models_count": 0,
            "files": []
        }

        for root, _, files in os.walk(cat_path):
            stl_files = [f for f in files if f.lower().endswith(".stl") and "identifier" not in f.lower()]
            for stl in sorted(stl_files):
                full_path = os.path.join(root, stl)
                rel_path = os.path.relpath(full_path, base_dir).replace("\\", "/")
                file_size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 2)

                # Identifica se è la versione preferita (filtrata / antiriflesso o smooth)
                stl_lower = stl.lower()
                is_preferred = (
                    "removehighlighton" in stl_lower or
                    "smoothfusion" in stl_lower or
                    "sample" in stl_lower or
                    ("removehighlight" not in stl_lower and "fusion" not in stl_lower)
                )

                # Nome descrittivo dello strumento (dalla cartella padre o dal file)
                parent_name = os.path.basename(root)
                model_name = parent_name if parent_name != cat else os.path.splitext(stl)[0]

                entry = {
                    "category": cat,
                    "model_name": model_name,
                    "filename": stl,
                    "rel_path": rel_path,
                    "workspace_path": f"/workspace/{rel_path}",
                    "file_size_mb": file_size_mb,
                    "is_preferred": is_preferred
                }

                catalog["categories"][cat]["files"].append(entry)
                catalog["all_instruments"].append(entry)
                total_files += 1

        catalog["categories"][cat]["models_count"] = len(catalog["categories"][cat]["files"])

    catalog["total_stl_files"] = total_files

    # Aggiungi anche gli STL/OBJ singoli presenti direttamente in data/inputs
    standalone_files = [
        f for f in os.listdir(inputs_dir)
        if (f.lower().endswith(".stl") or f.lower().endswith(".obj")) and "identifier" not in f.lower()
    ]
    for s_file in standalone_files:
        full_path = os.path.join(inputs_dir, s_file)
        rel_path = os.path.relpath(full_path, base_dir).replace("\\", "/")
        file_size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 2)
        catalog["all_instruments"].append({
            "category": "Standalone / Custom",
            "model_name": os.path.splitext(s_file)[0],
            "filename": s_file,
            "rel_path": rel_path,
            "workspace_path": f"/workspace/{rel_path}",
            "file_size_mb": file_size_mb,
            "is_preferred": True
        })

    if output_json_path is None:
        output_json_path = os.path.join(inputs_dir, "instruments_catalog.json")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print("\n" + "="*70)
    print("CATALOGO STRUMENTI CHIRURGICI 3D GENERATO CON SUCCESSO")
    print("="*70)
    print(f"Percorso JSON       : {output_json_path}")
    print(f"Totale Categorie   : {catalog['total_categories']}")
    print(f"Totale Modelli STL : {catalog['total_stl_files']}")
    print("-" * 70)
    print(f"{'CATEGORIA':<28} | {'FILE STL':<10} | {'MODELLO DI ESEMPIO'}")
    print("-" * 70)
    for cat, info in catalog["categories"].items():
        sample_model = info["files"][0]["model_name"] if info["files"] else "N/A"
        print(f"{cat:<28} | {info['models_count']:<10} | {sample_model[:30]}")
    print("="*70 + "\n")

    return catalog


if __name__ == "__main__":
    build_catalog()
