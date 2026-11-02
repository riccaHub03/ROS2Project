#!/usr/bin/env python3
"""
Pipeline Reale - Step 3: SAM 3 Instance Selection nel Vassoio Multi-Strumento.
Analizza la scena reale contenente decine di strumenti chirurgici sul vassoio
e usa SAM 3 con text-prompt per isolare UNICAMENTE l'istanza dello strumento bersaglio.
Input: data/outputs/real_stage2_scene.json + data/outputs/stage1_task_plan.json
Output: data/outputs/real_stage3_vision_mask.json (+ real_tool_mask.png, real_tool_overlay.png)
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


def run_real_sam3_segmentation(scene_report_path=None, plan_path=None, output_path=None):
    print("\n" + "="*70)
    print("👁️  [PIPELINE REALE - STAGE 3] SAM 3 INSTANCE SELECTION NEL VASSOIO REALE")
    print("="*70)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if scene_report_path is None:
        scene_report_path = os.path.join(base_dir, "data", "outputs", "real_stage2_scene.json")
    if plan_path is None:
        plan_path = os.path.join(base_dir, "data", "outputs", "stage1_task_plan.json")

    with open(scene_report_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)
    with open(plan_path, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    target_prompt = plan_data.get("target_instrument", "vascular scissors")
    target_kw = plan_data.get("target_keyword", "scissors")
    scene_image_path = scene_data["files"]["scene_rgb_image"]

    print(f"📷 Scena multi-strumento: {scene_image_path}")
    print(f"🎯 Strumento da isolare tra tutti i ferri: '{target_prompt}' (keyword: '{target_kw}')")

    print("⏳ Caricamento pesi SAM 3 per segmentazione multi-istanza...")
    model_id = "facebook/sam3"
    processor = Sam3Processor.from_pretrained(model_id, local_files_only=True)
    model = Sam3Model.from_pretrained(model_id, local_files_only=True).to("cpu")
    model.eval()

    image = Image.open(scene_image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Inferenza SAM 3 con instance selection
    inputs = processor(images=image, text=target_prompt, return_tensors="pt").to("cpu")
    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.0, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results["masks"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    if len(scores) == 0:
        print(f"⚠️ Riprovo con keyword semplificata: '{target_kw}'...")
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
        print(f" -> Istanze totali rilevate nel set reale: {len(scores)}")
        for idx, sc in enumerate(scores):
            star = "⭐ [SELEZIONATA PER LA PRESA]" if idx == best_idx else ""
            print(f"    - Istanza {idx}: Confidenza {sc*100:.1f}% {star}")
        selected_mask = (masks[best_idx] > 0).astype(np.uint8) * 255
    else:
        raise RuntimeError(f"Nessuno strumento rilevato per il prompt: {target_prompt}")

    out_dir = os.path.join(base_dir, "data", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_mask_path = os.path.join(out_dir, "real_tool_mask.png")
    out_overlay_path = os.path.join(out_dir, "real_tool_overlay.png")

    mask_img = Image.fromarray(selected_mask).resize((orig_w, orig_h), Image.Resampling.NEAREST)
    mask_img.save(out_mask_path)
    mask_np = np.array(mask_img)

    # Overlay: verde brillante per lo strumento selezionato, azzurro per gli altri ferri
    orig_np = np.array(image)
    overlay = orig_np.copy()
    m_bool = mask_np > 128

    if len(scores) > 1:
        other_idxs = [i for i in range(len(scores)) if i != best_idx and scores[i] >= 0.15]
        if other_idxs:
            other_comb = np.any(masks[other_idxs] > 0, axis=0)
            other_res = np.array(Image.fromarray(other_comb.astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.NEAREST)) > 0
            other_only = other_res & (~m_bool)
            overlay[other_only] = (overlay[other_only] * 0.5 + np.array([0, 160, 255]) * 0.5).astype(np.uint8)

    overlay[m_bool] = (overlay[m_bool] * 0.35 + np.array([0, 255, 60]) * 0.65).astype(np.uint8)
    Image.fromarray(overlay).save(out_overlay_path)

    v_idx, u_idx = np.where(m_bool)
    active_px = int(len(u_idx))
    centroid_2d = [float(np.mean(u_idx)), float(np.mean(v_idx))] if active_px > 0 else [0.0, 0.0]

    if output_path is None:
        output_path = os.path.join(out_dir, "real_stage3_vision_mask.json")

    report = {
        "stage": "real_stage3_vision_mask",
        "timestamp": datetime.datetime.now().isoformat(),
        "input_scene": scene_image_path,
        "target_instrument": target_prompt,
        "confidence_score": round(best_score, 4),
        "active_pixel_count": active_px,
        "centroid_2d_pixels": [round(c, 2) for c in centroid_2d],
        "files": {
            "binary_mask": out_mask_path,
            "visual_overlay": out_overlay_path
        },
        "status": "COMPLETED"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Strumento isolato con successo dal vassoio multi-ferro:")
    print(f"   - Confidenza SAM 3: {best_score*100:.1f}%")
    print(f"   - Overlay salvato : {out_overlay_path}")
    print(f"   - Report JSON     : {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Step 3: SAM 3 Multi-Tool Instance Selection")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--plan", type=str, default=None)
    args = parser.parse_args()
    run_real_sam3_segmentation(scene_report_path=args.scene, plan_path=args.plan)
