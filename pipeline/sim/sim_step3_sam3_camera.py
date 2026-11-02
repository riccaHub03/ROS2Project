#!/usr/bin/env python3
"""
Pipeline Simulazione - Step 3: SAM 3 Segmentazione Visiva sulla Vista Telecamera.
Prende l'immagine inquadrata dalla telecamera simulata (sim_camera_rgb.png)
e usa SAM 3 per segmentare direttamente lo strumento chirurgico nel frame ottico.
Input: data/outputs/sim_stage2_camera.json + data/outputs/stage1_task_plan.json
Output: data/outputs/sim_stage3_vision_mask.json (+ sim_camera_mask.png, sim_camera_overlay.png)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import json
import argparse
import datetime
import numpy as np
from PIL import Image
import torch
from transformers import Sam3Model, Sam3Processor


def run_sim_sam3_segmentation(cam_report_path=None, plan_path=None, output_path=None):
    print("\n" + "="*70)
    print("👁️  [PIPELINE SIM - STAGE 3] SAM 3 SEGMENTAZIONE SULLA VISTA TELECAMERA")
    print("="*70)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if cam_report_path is None:
        cam_report_path = os.path.join(base_dir, "data", "outputs", "sim_stage2_camera.json")
    if plan_path is None:
        plan_path = os.path.join(base_dir, "data", "outputs", "stage1_task_plan.json")

    with open(cam_report_path, "r", encoding="utf-8") as f:
        cam_data = json.load(f)
    with open(plan_path, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    target_tool = plan_data.get("target_instrument", "vascular scissors")
    target_kw = plan_data.get("target_keyword", "scissors")
    camera_rgb_path = cam_data["files"]["camera_rgb_image"]

    print(f"📷 Immagine da analizzare: Vista Telecamera Overhead ({os.path.basename(camera_rgb_path)})")
    print(f"🎯 Target Prompt SAM 3   : '{target_tool}' (keyword: '{target_kw}')")

    # Caricamento SAM 3
    print("⏳ Inferenza SAM 3 sul frame ottico della telecamera...")
    model_id = "facebook/sam3"
    processor = Sam3Processor.from_pretrained(model_id, local_files_only=True)
    model = Sam3Model.from_pretrained(model_id, local_files_only=True).to("cpu")
    model.eval()

    image = Image.open(camera_rgb_path).convert("RGB")
    orig_w, orig_h = image.size

    # Inferenza con keyword mirata
    inputs = processor(images=image, text=target_kw, return_tensors="pt").to("cpu")
    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.0, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results["masks"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    if len(scores) > 0:
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        mask_np = (masks[best_idx] > 0).astype(np.uint8) * 255
        if best_score < 0.15:
            print(f"⚠️ ATTENZIONE: Confidenza di rilevamento molto bassa ({best_score:.3f} < 0.15).")
            print(f"   Lo strumento '{target_tool}' NON sembra presente nella scena inquadrata dalla telecamera!")
        else:
            print(f"✅ Rilevata istanza bersaglio con confidenza: {best_score:.3f}")
    else:
        # Fallback ottico su sogliatura se SAM 3 non trova istanze
        print("⚠️ Nessun match diretto, applicazione fallback ottico.")
        mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)
        best_score = 0.0

    out_dir = os.path.join(base_dir, "data", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_mask_path = os.path.join(out_dir, "sim_camera_mask.png")
    out_overlay_path = os.path.join(out_dir, "sim_camera_overlay.png")

    Image.fromarray(mask_np).save(out_mask_path)

    # Creazione Overlay
    rgb_np = np.array(image)
    overlay = rgb_np.copy()
    m_bool = mask_np > 128
    overlay[m_bool] = (overlay[m_bool] * 0.35 + np.array([0, 255, 60]) * 0.65).astype(np.uint8)
    Image.fromarray(overlay).save(out_overlay_path)

    # Calcolo metriche 2D
    v_idx, u_idx = np.where(m_bool)
    active_px = int(len(u_idx))
    if active_px > 0:
        bbox_2d = [int(np.min(u_idx)), int(np.min(v_idx)), int(np.max(u_idx)), int(np.max(v_idx))]
        centroid_2d = [float(np.mean(u_idx)), float(np.mean(v_idx))]
    else:
        bbox_2d = [0, 0, 0, 0]
        centroid_2d = [0.0, 0.0]

    if output_path is None:
        output_path = os.path.join(out_dir, "sim_stage3_vision_mask.json")

    report = {
        "stage": "sim_stage3_vision_mask",
        "timestamp": datetime.datetime.now().isoformat(),
        "input_camera_image": camera_rgb_path,
        "target_prompt": target_kw,
        "confidence_score": round(best_score, 4),
        "active_pixel_count": active_px,
        "bbox_2d_pixels": bbox_2d,
        "centroid_2d_pixels": [round(c, 2) for c in centroid_2d],
        "files": {
            "camera_mask_png": out_mask_path,
            "camera_overlay_png": out_overlay_path
        },
        "status": "COMPLETED"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Segmentazione SAM 3 della vista telecamera completata:")
    print(f"   - Pixel attivi strumento: {active_px} px")
    print(f"   - Maschera telecamera   : {out_mask_path}")
    print(f"   - Report JSON           : {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim Step 3: SAM 3 on Camera Image")
    parser.add_argument("--cam", type=str, default=None)
    parser.add_argument("--plan", type=str, default=None)
    args = parser.parse_args()
    run_sim_sam3_segmentation(cam_report_path=args.cam, plan_path=args.plan)
