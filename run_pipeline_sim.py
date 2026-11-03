#!/usr/bin/env python3
"""
Orchestratore Pipeline di Simulazione 3D (All_Instruments / Single-Tool).
Esegue la catena di montaggio in simulazione PyBullet con telecamera RGB-D reale:
  1. LLM Task Planner         -> data/outputs/stage1_task_plan.json
  2. Spawning 3D & Camera     -> data/outputs/sim_stage2_camera.json
  3. SAM 3 su Vista Camera    -> data/outputs/sim_stage3_vision_mask.json
  4. 3D Deprojection & Frames -> data/outputs/sim_stage4_transforms.json
  5. Grasp Geometry & Zones   -> data/outputs/stage5_grasp_geometry.json
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import time

from pipeline.step1_llm_planner import run_llm_planner, DEFAULT_INSTRUCTION
from pipeline.sim.sim_step2_camera_render import run_sim_camera_render
from pipeline.sim.sim_step3b_stl_placement import run_sim_stl_placement
from pipeline.sim.sim_step4_transforms import run_sim_transforms
from pipeline.step5_grasp_geometry import run_grasp_geometry
from pipeline.mesh_loader import load_instruments_catalog


def main():
    parser = argparse.ArgumentParser(description="Pipeline di Simulazione 3D")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Comando clinico per il task planner LLM")
    parser.add_argument("--tool", type=str, default=None,
                        help="Strumento chirurgico specifico da testare (es. 'Scalpel', 'Bone Forceps', 'Needle Holders', 'Bulldog Clamps', 'Mallet', 'Scissors')")
    parser.add_argument("--list-tools", action="store_true",
                        help="Visualizza il catalogo completo di tutte le 27 categorie e strumenti disponibili")
    parser.add_argument("--mode", type=str, choices=["all", "single"], default=None,
                        help="'all' carica la scena 3D completa All_Instruments.obj; 'single' carica un solo STL")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed di posizionamento casuale nel vassoio")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="Finestra 3D nativa con KUKA e anteprima del trasferimento")
    display.add_argument("--headless", action="store_true", help="Stessa run con robot senza finestra; termina dopo il risultato")
    parser.add_argument("--vision", choices=["sam3", "ground-truth"], default="sam3",
                        help="SAM3 oppure test diagnostico esplicito con maschera da ID PyBullet")
    parser.add_argument("--no-motion", action="store_true", help="Visualizza la presa senza muovere il robot")
    parser.add_argument("--exit-after-run", action="store_true", help="Chiudi la finestra al termine della run")
    args = parser.parse_args()

    if args.list_tools:
        cat = load_instruments_catalog()
        print("\n" + "="*70)
        print("CATALOGO COMPLETO STRUMENTI CHIRURGICI 3D DISPONIBILI (3D-COSI)")
        print("="*70)
        if cat:
            print(f"Totale Categorie: {cat['total_categories']} | Totale Modelli STL: {cat['total_stl_files']}\n")
            print(f"{'#':<3} | {'CATEGORIA':<26} | {'MODELLI':<8} | {'ESEMPIO'}")
            print("-" * 70)
            for idx, (c_name, info) in enumerate(cat["categories"].items(), 1):
                sample_m = info["files"][0]["model_name"] if info["files"] else "N/A"
                print(f"{idx:<3} | {c_name:<26} | {info['models_count']:<8} | {sample_m[:30]}")
            print("="*70)
            print("Per testare una categoria o strumento, esegui:")
            print("   python3 run_pipeline_sim.py --tool \"Bone Forceps\"")
            print("   python3 run_pipeline_sim.py --tool \"Scalpel\"")
            print("   python3 run_pipeline_sim.py --tool \"Needle Holders\"\n")
        else:
            print("Catalogo non trovato. Generalo con scripts/build_instruments_catalog.py")
        return

    # Determina l'istruzione clinica
    instruction = args.instruction
    if not instruction:
        if args.tool:
            instruction = f"Passami lo strumento {args.tool} posizionato sul vassoio 2 e mettilo nel vassoio 1 per l'intervento chirurgico."
        else:
            instruction = DEFAULT_INSTRUCTION

    # Se è specificato uno strumento bersaglio dal catalogo STL, impostiamo la modalità 'single'
    # in modo che la telecamera inquadri ESATTAMENTE quel modello 3D, garantendo coerenza ottica al 100%
    mode = args.mode or ("single" if args.tool or args.gui or args.headless else "all")
    if args.gui or args.headless:
        if mode != "single":
            parser.error("La run con robot richiede --mode single: la collezione combinata non ha corpi separati")
        from pipeline.sim.graphical_pipeline import run_graphical_pipeline
        return run_graphical_pipeline(args, instruction)
    if args.vision != "sam3" or args.no_motion or args.exit_after_run:
        parser.error("--vision ground-truth, --no-motion e --exit-after-run richiedono --gui o --headless")

    t_start = time.time()
    print("\n" + "="*75)
    print(f"AVVIO PIPELINE DI SIMULAZIONE 3D (SCENA: {mode.upper()})")
    print("   (Zero fotografie 2D usate / 100% Raycasting ottico da Geometria 3D)")
    if args.tool:
        print(f"   Strumento Bersaglio Selezionato: {args.tool} (Modalità: {mode.upper()})")
    print("="*75)

    # 1. LLM Planner
    run_llm_planner(instruction_text=instruction, target_tool_hint=args.tool)

    # 2. Spawn 3D & Render Camera (All_Instruments o Single STL)
    run_sim_camera_render(mode=mode, seed=args.seed)

    # 3. SAM 3 sulla Vista della Telecamera
    from pipeline.sim.sim_step3_sam3_camera import run_sim_sam3_segmentation
    run_sim_sam3_segmentation()

    # 3b. Posizionamento STL nel Vassoio con Report Visivo PNG
    run_sim_stl_placement(tool_override=args.tool, seed=args.seed)

    # 4. De-proiezione e Frame Transformations
    run_sim_transforms()

    # 5. Definizione Geometria del Grasping e Zone Funzionali
    run_grasp_geometry(source_type="sim")

    duration = time.time() - t_start
    print("\n" + "="*75)
    print(f"PIPELINE DI SIMULAZIONE 3D COMPLETATA CON SUCCESSO in {duration:.2f} secondi!")
    print("="*75)
    print("REPORT E ARTEFATTI PRODOTTI:")
    print("   1. [Stage 1] data/outputs/stage1_task_plan.json")
    print("   2. [Stage 2] data/outputs/sim_stage2_camera.json (+ sim_camera_rgb.png, sim_camera_depth.npy)")
    print("   3. [Stage 3] data/outputs/sim_stage3_vision_mask.json (+ sim_camera_mask.png, sim_camera_overlay.png)")
    print("   4. [Stage 3b] data/outputs/stage3_stl_placement.json (+ sim_stl_tray_placement.png)")
    print("   5. [Stage 4] data/outputs/sim_stage4_transforms.json (+ sim_reconstructed_instrument.ply, sim_stage4_frames_visual.png)")
    print("   6. [Stage 5] data/outputs/stage5_grasp_geometry.json (+ stage5_segmented_grasp_zones.ply, stage5_grasp_geometry_visual.png)")
    print("="*75 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
