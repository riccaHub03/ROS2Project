#!/usr/bin/env python3
"""
Orchestratore Pipeline Reale (Multi-Strumento - Zero STL nel Docker).
Esegue la catena di percezione reale per il set chirurgico completo:
  1. LLM Task Planner                 -> data/outputs/stage1_task_plan.json
  2. Acquisizione Scena Vassoio       -> data/outputs/real_stage2_scene.json
  3. SAM 3 Multi-Tool Segmentation    -> data/outputs/real_stage3_vision_mask.json
  4. De-proiezione Metrica 3D         -> data/outputs/real_stage4_transforms.json
  5. Grasp Geometry & Zones           -> data/outputs/stage5_grasp_geometry.json
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import time

from pipeline.step1_llm_planner import run_llm_planner, DEFAULT_INSTRUCTION
from pipeline.real.real_step2_scene_capture import run_real_scene_capture
from pipeline.real.real_step3_sam3_multi import run_real_sam3_segmentation
from pipeline.real.real_step4_transforms import run_real_transforms
from pipeline.step5_grasp_geometry import run_grasp_geometry


def main():
    parser = argparse.ArgumentParser(description="Pipeline Reale (Multi-Strumento)")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION,
                        help="Comando clinico per il task planner LLM")
    parser.add_argument("--image", type=str, default=None,
                        help="Immagine reale del vassoio contenente tutti gli strumenti")
    parser.add_argument("--config", type=str, default=None, help="File YAML dell'ambiente")
    args = parser.parse_args()
    if args.config and not os.path.isfile(args.config):
        parser.error(f"Configurazione non trovata: {args.config}")

    t_start = time.time()
    print("\n" + "="*75)
    print("AVVIO PIPELINE REALE (VASSOIO COMPLETO MULTI-STRUMENTO)")
    print("   (Zero STL trasferiti nel Docker / 100% percezione ottica reale)")
    print("="*75)

    # 1. LLM Planner
    run_llm_planner(instruction_text=args.instruction, config_path=args.config)

    # 2. Acquisizione Scena Reale
    run_real_scene_capture(image_file=args.image, config_path=args.config)

    # 3. SAM 3 Instance Selection
    run_real_sam3_segmentation()

    # 4. De-proiezione e Frame Transformations
    run_real_transforms(config_path=args.config)

    # 5. Definizione Geometria del Grasping e Zone Funzionali
    run_grasp_geometry(source_type="real", config_path=args.config)

    duration = time.time() - t_start
    print("\n" + "="*75)
    print(f"PIPELINE REALE COMPLETATA CON SUCCESSO in {duration:.2f} secondi!")
    print("="*75)
    print("REPORT E ARTEFATTI PRODOTTI:")
    print("   1. [Stage 1] data/outputs/stage1_task_plan.json")
    print("   2. [Stage 2] data/outputs/real_stage2_scene.json (+ real_scene_rgb.png)")
    print("   3. [Stage 3] data/outputs/real_stage3_vision_mask.json (+ real_tool_mask.png, real_tool_overlay.png)")
    print("   4. [Stage 4] data/outputs/real_stage4_transforms.json (+ real_reconstructed_instrument.ply)")
    print("   5. [Stage 5] data/outputs/stage5_grasp_geometry.json (+ stage5_segmented_grasp_zones.ply, stage5_grasp_geometry_visual.png)")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
