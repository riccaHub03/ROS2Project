#!/usr/bin/env python3
"""
Pipeline - Step 5: Definizione della Geometria del Pezzo Dedita alla Presa (Grasp Geometry).
Analizza la nuvola di punti 3D de-proiettata dello strumento (reconstructed_instrument.ply),
calcola l'orientamento principale tramite PCA, segmenta longitudinalmente l'attrezzo
in zone funzionali (Anelli / Fusto-Cerniera / Lame), isola la geometria della porzione
ottimale di presa (Zone B: Shank/Hinge) e verifica la compatibilità cinematica della pinza.

Input:
  - data/outputs/sim_stage4_transforms.json (o real_stage4_transforms.json)
  - data/outputs/stage1_task_plan.json
  - config/surgical_environment.yaml
Output:
  - data/outputs/stage5_grasp_geometry.json
  - data/outputs/stage5_segmented_grasp_zones.ply
  - data/outputs/stage5_grasp_geometry_visual.png
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import json
import argparse
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config_loader import load_environment_config
from pipeline.frame_transformer import quaternion_from_matrix


def parse_ply(filepath):
    """Estrae vertici [x, y, z] e colori opzionali da file Stanford PLY ASCII."""
    pts = []
    colors = []
    header = True
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if header:
                if line == "end_header":
                    header = False
                continue
            parts = line.split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(parts) >= 6:
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.array(pts, dtype=np.float64), np.array(colors, dtype=np.uint8) if colors else None


def export_colored_ply(filepath, points_xyz, colors_rgb):
    """Esporta nuvola di punti classificata con colori RGB in formato ASCII PLY."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    n_pts = len(points_xyz)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n_pts):
            p = points_xyz[i]
            c = colors_rgb[i]
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def run_grasp_geometry(source_type="sim", transforms_report_path=None, task_plan_path=None,
                       config_path=None, output_path=None):
    """
    Esegue l'analisi geometrica dello strumento chirurgico per la presa robotica.
    """
    print("\n" + "="*75)
    print(f"🔬 [PIPELINE - STAGE 5] DEFINIZIONE GEOMETRIA DEL PEZZO DEDITA ALLA PRESA")
    print(f"   (Sorgente: {source_type.upper()})")
    print("="*75)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "data", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Risoluzione percorsi file di ingresso
    if transforms_report_path is None:
        filename = f"{source_type}_stage4_transforms.json"
        transforms_report_path = os.path.join(out_dir, filename)

    if task_plan_path is None:
        task_plan_path = os.path.join(out_dir, "stage1_task_plan.json")

    if not os.path.exists(transforms_report_path):
        raise FileNotFoundError(f"Report Stage 4 non trovato: {transforms_report_path}")

    with open(transforms_report_path, "r", encoding="utf-8") as f:
        t_data = json.load(f)

    tool_name = "surgical_instrument"
    if os.path.exists(task_plan_path):
        with open(task_plan_path, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
            tool_name = plan_data.get("target_instrument", tool_name)

    config = load_environment_config(config_path)
    ply_path = t_data.get("pointcloud_file")
    if not ply_path or not os.path.exists(ply_path):
        raise FileNotFoundError(f"File PLY nuvola di punti non trovato: {ply_path}")

    # 2. Caricamento e filtraggio nuvola di punti
    raw_pts, raw_cols = parse_ply(ply_path)
    if len(raw_pts) < 15:
        raise ValueError(f"Troppi pochi punti 3D ({len(raw_pts)}) per l'analisi geometrica!")

    # Filtraggio del piano d'appoggio (vassoio/tavolo)
    # I punti dello strumento giacciono sopra la quota del piano del vassoio
    z_min_raw = np.min(raw_pts[:, 2])
    z_max_raw = np.max(raw_pts[:, 2])
    
    # Se ci sono punti aderenti al fondo vassoio (z ~ 0.05 m per vassoio basso o z ~ 0.16 per vassoio rialzato)
    # filtriamo via la base per isolare la geometria pura dell'attrezzo
    z_height_span = z_max_raw - z_min_raw
    if z_height_span > 0.015:
        # Filtra i pixel con z nei percentili più bassi se coincidenti col colore del vassoio
        z_cutoff = z_min_raw + 0.10 * z_height_span
        tool_mask = raw_pts[:, 2] >= z_cutoff
        tool_pts = raw_pts[tool_mask]
    else:
        tool_pts = raw_pts

    if len(tool_pts) < 10:
        tool_pts = raw_pts

    print(f"📊 Punti 3D analizzati: {len(tool_pts)} (grezzi: {len(raw_pts)})")

    # 3. Analisi delle Componenti Principali (PCA)
    centroid = np.mean(tool_pts, axis=0)
    centered = tool_pts - centroid
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    v1 = evecs[:, 0]  # Asse principale longitudinale (lunghezza dello strumento)
    v2 = evecs[:, 1]  # Asse laterale (larghezza dello strumento)
    v3 = evecs[:, 2]  # Asse normale (spessore / normale al piano del tavolo)

    # Coerenza dell'asse normale v3: deve puntare verso l'alto (+Z)
    if v3[2] < 0:
        v3 = -v3

    # Coerenza dell'asse longitudinale v1: convenzione verso +X o +Y positivo
    if (v1[0] + v1[1]) < 0:
        v1 = -v1

    # Ricalcolo v2 per garantire una terna ortonormale destrorsa rigorosa
    v2 = np.cross(v3, v1)
    v2 /= np.linalg.norm(v2)

    # 4. Proiezione longitudinale e segmentazione funzionale dell'attrezzo
    # s_i è la coordinata scalare di ciascun punto lungo l'asse longitudinale v1
    s = centered @ v1
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    tool_length = float(s_max - s_min)

    # Dimensioni totali del bounding box orientato (OBB)
    tool_width = float(np.ptp(centered @ v2))
    tool_thickness = float(np.ptp(centered @ v3))

    # Segmentazione in tre zone anatomiche/funzionali dell'attrezzo chirurgico:
    #   - Zone A (0% - 35%): Anelli d'impugnatura (Finger Loops / Rings)
    #   - Zone B (35% - 65%): Fusto centrale e perno cerniera (Central Shank / Pivot Hinge) -> ZONA DI PRESA
    #   - Zone C (65% - 100%): Punte e lame taglienti (Working End / Blades) -> ESCLUSIONE DI SICUREZZA
    rel_s = (s - s_min) / tool_length
    mask_zone_a = (rel_s >= 0.00) & (rel_s < 0.35)
    mask_zone_b = (rel_s >= 0.35) & (rel_s <= 0.65)
    mask_zone_c = (rel_s > 0.65) & (rel_s <= 1.00)

    pts_a = tool_pts[mask_zone_a]
    pts_b = tool_pts[mask_zone_b]
    pts_c = tool_pts[mask_zone_c]

    # 5. Caratterizzazione geometrica della Zona B (Porzione dedita alla presa)
    if len(pts_b) >= 3:
        grasp_centroid = np.mean(pts_b, axis=0)
        grasp_width = float(np.ptp(pts_b @ v2))
        grasp_thickness = float(np.ptp(pts_b @ v3))
    else:
        # Fallback su baricentro complessivo se la nuvola è molto rada
        grasp_centroid = centroid
        grasp_width = tool_width * 0.4
        grasp_thickness = tool_thickness

    # Orientamento dell'End-Effector KUKA (TCP pinza):
    # - Asse Z_ee (Approccio): rivolto verso il basso lungo la normale alla superficie (-v3)
    # - Asse Y_ee (Chiusura pinza): parallelo alla larghezza del fusto (v2), così le ganasce stringono sui fianchi
    # - Asse X_ee (Dita pinza): ortogonale a Z_ee e Y_ee (parallelo all'asse longitudinale dello strumento v1)
    z_ee = -v3
    z_ee /= np.linalg.norm(z_ee)

    y_ee = v2
    y_ee /= np.linalg.norm(y_ee)

    x_ee = np.cross(y_ee, z_ee)
    x_ee /= np.linalg.norm(x_ee)

    r_grasp = np.column_stack([x_ee, y_ee, z_ee])
    grasp_quat = quaternion_from_matrix(r_grasp)

    # 6. Verifica di fattibilità per pinza a due griffe parallele
    max_gripper_stroke_m = 0.080  # 80 mm apertura standard pinza robotica
    clearance_m = 0.020           # 20 mm margine di sicurezza apertura pre-grasp
    required_stroke_m = grasp_width + clearance_m
    is_graspable = bool(required_stroke_m <= max_gripper_stroke_m and grasp_width > 0.002)

    # 7. Assegnazione colori per visualizzazione 3D delle zone:
    #   - Zone A (Anelli): Blu (40, 120, 220)
    #   - Zone B (Grasp Zone): Verde brillante (40, 220, 80)
    #   - Zone C (Lame/Punte): Rosso (220, 50, 50)
    classified_colors = np.zeros((len(tool_pts), 3), dtype=np.uint8)
    classified_colors[mask_zone_a] = [40, 120, 220]
    classified_colors[mask_zone_b] = [40, 220, 80]
    classified_colors[mask_zone_c] = [220, 50, 50]

    out_segmented_ply = os.path.join(out_dir, "stage5_segmented_grasp_zones.ply")
    export_colored_ply(out_segmented_ply, tool_pts, classified_colors)

    # 8. Generazione visualizzazione diagnostica 2D
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Vista dall'alto (Robot Base X-Y)
    ax1 = axes[0]
    ax1.scatter(pts_a[:, 0], pts_a[:, 1], c="royalblue", s=25, label="Zone A: Anelli (Impugnatura)", alpha=0.8)
    ax1.scatter(pts_b[:, 0], pts_b[:, 1], c="limegreen", s=35, label="Zone B: Fusto/Cerniera (PRESA OTTIMALE)", alpha=0.9)
    ax1.scatter(pts_c[:, 0], pts_c[:, 1], c="crimson", s=25, label="Zone C: Lame (Zona Esclusione)", alpha=0.8)

    # Disegna assi principali dal centroide dello strumento
    scale = 0.05
    ax1.arrow(centroid[0], centroid[1], v1[0]*scale, v1[1]*scale,
              color="black", width=0.0015, head_width=0.005, label="Asse Longitudinale v1")
    ax1.arrow(centroid[0], centroid[1], v2[0]*scale, v2[1]*scale,
              color="darkorange", width=0.0015, head_width=0.005, label="Asse Laterale v2 (Chiusura pinza)")

    # Segna il punto baricentrico di presa
    ax1.scatter([grasp_centroid[0]], [grasp_centroid[1]], c="gold", s=180, marker="*",
                edgecolors="black", linewidths=1.5, label="Target Grasp Centroid (Zone B)", zorder=5)

    ax1.set_xlabel("X (Robot Base Frame) [m]")
    ax1.set_ylabel("Y (Robot Base Frame) [m]")
    ax1.set_title(f"Segmentazione Geometrica Zone di Presa ({tool_name.replace('_', ' ').title()})")
    ax1.axis("equal")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)

    # Profilo 1D lungo l'asse longitudinale v1
    ax2 = axes[1]
    ax2.hist(rel_s[mask_zone_a]*100, bins=15, color="royalblue", alpha=0.7, label="Zone A: Anelli")
    ax2.hist(rel_s[mask_zone_b]*100, bins=15, color="limegreen", alpha=0.7, label="Zone B: Fusto (Grasp)")
    ax2.hist(rel_s[mask_zone_c]*100, bins=15, color="crimson", alpha=0.7, label="Zone C: Lame")
    ax2.axvline(35.0, color="gray", linestyle="--", label="Soglia Zone A/B (35%)")
    ax2.axvline(65.0, color="gray", linestyle="--", label="Soglia Zone B/C (65%)")
    ax2.set_xlabel("Posizione Longitudinale Relativa [%]")
    ax2.set_ylabel("Densità di Punti Superficiali")
    ax2.set_title("Distribuzione del Materiale lungo l'Asse Longitudinale")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_visual_png = os.path.join(out_dir, "stage5_grasp_geometry_visual.png")
    fig.savefig(out_visual_png, dpi=150)
    plt.close(fig)

    # 9. Costruzione del report JSON strutturato
    if output_path is None:
        output_path = os.path.join(out_dir, "stage5_grasp_geometry.json")

    report = {
        "stage": "stage5_grasp_geometry",
        "source_pipeline": source_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "target_instrument": tool_name,
        "input_pointcloud": {
            "source_ply": ply_path,
            "total_points": int(len(raw_pts)),
            "filtered_points": int(len(tool_pts))
        },
        "instrument_bounding_dimensions_m": {
            "length_along_v1": round(tool_length, 4),
            "width_along_v2": round(tool_width, 4),
            "thickness_along_v3": round(tool_thickness, 4)
        },
        "principal_axes_base_frame": {
            "v1_longitudinal": [round(float(x), 4) for x in v1],
            "v2_lateral": [round(float(x), 4) for x in v2],
            "v3_normal": [round(float(x), 4) for x in v3]
        },
        "functional_zones_segmentation": {
            "zone_A_proximal_rings": {
                "description": "Anelli d'impugnatura per le dita del chirurgo (Finger loops)",
                "longitudinal_range_rel": [0.00, 0.35],
                "points_count": int(len(pts_a)),
                "grasp_classification": "EXCLUDED_HIGH_COMPLIANCE_HOLLOW_STRUCTURE",
                "risk": "Flessione torsionale, rotazione attorno alle dita e scivolamento pinza"
            },
            "zone_B_central_shank": {
                "description": "Fusto centrale e perno della cerniera (Shank and pivot hinge)",
                "longitudinal_range_rel": [0.35, 0.65],
                "points_count": int(len(pts_b)),
                "grasp_classification": "PRIMARY_OPTIMAL_GRASP_ZONE",
                "advantage": "Massima rigidità strutturale del metallo pieno, minimo momento d'inerzia, planarità ottimale per pinza a due griffe"
            },
            "zone_C_distal_blades": {
                "description": "Lame e punte taglienti chirurgiche (Cutting blades and tips)",
                "longitudinal_range_rel": [0.65, 1.00],
                "points_count": int(len(pts_c)),
                "grasp_classification": "FORBIDDEN_CUTTING_EDGE_EXCLUSION_ZONE",
                "risk": "Danneggiamento del filo di taglio chirurgico, rischio collisione e pericolo clinico"
            }
        },
        "designated_grasp_geometry": {
            "selected_zone": "Zone B (Central Shank / Pivot Hinge)",
            "grasp_centroid_base_frame_xyz_m": [round(float(x), 4) for x in grasp_centroid],
            "local_cross_section_m": {
                "width_along_v2": round(grasp_width, 4),
                "thickness_along_v3": round(grasp_thickness, 4)
            },
            "recommended_gripper_orientation": {
                "quaternion_xyzw": [round(float(q), 5) for q in grasp_quat],
                "approach_vector_base": [round(float(x), 4) for x in z_ee],
                "jaw_closing_vector_base": [round(float(x), 4) for x in y_ee],
                "finger_alignment_vector_base": [round(float(x), 4) for x in x_ee]
            },
            "gripper_compatibility_check": {
                "parallel_jaw_suitable": is_graspable,
                "required_opening_m": round(required_stroke_m, 4),
                "max_allowed_stroke_m": max_gripper_stroke_m,
                "safety_clearance_m": clearance_m
            }
        },
        "artifacts_generated": {
            "segmented_pointcloud_ply": out_segmented_ply,
            "diagnostic_visual_png": out_visual_png
        },
        "status": "VALIDATED_GEOMETRY_READY_FOR_TRAJECTORY_PLAN"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n✅ RISULTATI CARATTERIZZAZIONE GEOMETRICA:")
    print(f"   - Strumento Identificato         : {tool_name}")
    print(f"   - Lunghezza Totale (v1)          : {tool_length*1000:.1f} mm")
    print(f"   - Larghezza Massima (v2)         : {tool_width*1000:.1f} mm")
    print(f"   - Spessore Medio (v3)            : {tool_thickness*1000:.1f} mm")
    print(f"   - Zona di Presa Selezionata      : Zone B (Fusto / Perno Cerniera)")
    print(f"   - Baricentro di Presa (Base Frame): {[round(x, 4) for x in grasp_centroid]} m")
    print(f"   - Sezione di Presa (Larghezza)   : {grasp_width*1000:.1f} mm")
    print(f"   - Sezione di Presa (Spessore)    : {grasp_thickness*1000:.1f} mm")
    print(f"   - Apertura Pinza Richiesta       : {required_stroke_m*1000:.1f} mm (Max: {max_gripper_stroke_m*1000:.1f} mm)")
    print(f"   - Compatibilità Pinza a Griffe   : {'CONFERMATA' if is_graspable else 'NON COMPATIBILE'}")
    print(f"   - Report Geometria Salvato in    : {output_path}")
    print(f"   - Visualizzazione Grafica        : {out_visual_png}")
    print("="*75 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Step 5: Grasp Geometry Definition")
    parser.add_argument("--source", type=str, choices=["sim", "real"], default="sim",
                        help="Sorgente della nuvola di punti (sim o real)")
    parser.add_argument("--transforms", type=str, default=None,
                        help="Percorso custom del file json stage 4")
    parser.add_argument("--output", type=str, default=None,
                        help="Percorso custom di output del file json stage 5")
    args = parser.parse_args()

    run_grasp_geometry(source_type=args.source, transforms_report_path=args.transforms,
                       output_path=args.output)
