"""
Modulo 2: Segmentazione Visiva Zero-Shot con SAM 3 (Segment Anything Model 3).
Esegue la segmentazione a singola istanza per identificare e isolare esattamente
lo strumento chirurgico bersaglio sul vassoio / tavolo operatorio.
"""

import os
import shutil
import numpy as np
from PIL import Image
import torch
from transformers import Sam3Model, Sam3Processor

def detect_instrument_mask(image_path=None, target_prompt="surgical scissors", config=None):
    print("\n" + "="*65)
    print("👁️  [FASE 2/4] SEGMENTAZIONE VISIVA SAM 3 (INSTANCE SELECTION)")
    print(f"   Target Prompt: '{target_prompt}'")
    print("="*65)

    if image_path is None or not os.path.exists(image_path):
        candidates = [
            "/workspace/data/inputs/blog-ferri-chirurgici.jpg",
            "/workspace/data/inputs/surgical_scene.jpg",
            "/workspace/blog-ferri-chirurgici.jpg",
            "/workspace/surgical_scene.jpg",
        ]
        image_path = None
        for c in candidates:
            if os.path.exists(c):
                image_path = c
                break
        if image_path is None:
            raise FileNotFoundError("Nessuna immagine chirurgica trovata in data/inputs/ o /workspace/")

    print(f"📷 Immagine di input: {image_path}")

    # Caricamento del modello SAM 3 (pesi locali HuggingFace su CPU)
    print("⏳ Caricamento pesi facebook/sam3...")
    model_id = "facebook/sam3"
    processor = Sam3Processor.from_pretrained(model_id, local_files_only=True)
    model = Sam3Model.from_pretrained(model_id, local_files_only=True).to("cpu")
    model.eval()

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Ottimizzazione scala per inferenza veloce e precisa
    max_side = 800
    scale = min(max_side / orig_w, max_side / orig_h, 1.0)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    image_small = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    inputs = processor(images=image_small, text=target_prompt, return_tensors="pt").to("cpu")
    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.0,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results["masks"].detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy()

    if len(scores) > 0:
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        print(f" -> Istanze totali identificate: {len(scores)}")
        for idx, sc in enumerate(scores):
            status_tag = "⭐ [SELEZIONATA PER LA PRESA]" if idx == best_idx else "  [Candidata alternativa]"
            print(f"    - Istanza {idx}: Confidenza {sc * 100:.1f}% {status_tag}")

        # Estrae SOLO la singola istanza con massima confidenza (Instance Selection)
        selected_mask = (masks[best_idx] > 0).astype(np.uint8) * 255
    else:
        print("⚠️ Nessuna istanza trovata per il prompt specificato.")
        selected_mask = np.zeros((new_h, new_w), dtype=np.uint8)

    mask_img = Image.fromarray(selected_mask).resize((orig_w, orig_h), Image.Resampling.NEAREST)

    # Salvataggio Maschera Binaria in data/outputs/
    out_dir = "/workspace/data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_mask_path = os.path.join(out_dir, "surgical_mask.png")
    mask_img.save(out_mask_path)

    # Creazione Overlay Visivo ad alta distinzione (Verde = Target, Azzurro = Altri ferri)
    orig_np = np.array(image)
    target_mask_np = np.array(mask_img) > 0
    overlay_np = orig_np.copy()

    if len(scores) > 1:
        other_indices = [i for i in range(len(scores)) if i != best_idx and scores[i] >= 0.20]
        if other_indices:
            other_combined = np.any(masks[other_indices] > 0, axis=0)
            other_resized = np.array(Image.fromarray(other_combined.astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.NEAREST)) > 0
            other_only = other_resized & (~target_mask_np)
            overlay_np[other_only] = (overlay_np[other_only] * 0.5 + np.array([0, 160, 255]) * 0.5).astype(np.uint8)

    overlay_np[target_mask_np] = (overlay_np[target_mask_np] * 0.35 + np.array([0, 255, 60]) * 0.65).astype(np.uint8)

    out_overlay_path = os.path.join(out_dir, "surgical_overlay.png")
    Image.fromarray(overlay_np).save(out_overlay_path)

    print(f"✅ Maschera binaria salvata: {out_mask_path}")
    print(f"✅ Overlay a colori salvato : {out_overlay_path}")
    return out_mask_path
