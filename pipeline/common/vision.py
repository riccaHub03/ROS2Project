"""Text-prompted instance segmentation (SAM 3) plus mask/overlay helpers shared by sim and real."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np
from PIL import Image

from pipeline.common.report import StageError

log = logging.getLogger("vision")

SAM3_MODEL_ID = "facebook/sam3"
TARGET_COLOR = np.array([0, 255, 60])      # selected instrument
OTHERS_COLOR = np.array([0, 160, 255])     # other detected instruments
LOW_CONFIDENCE = 0.15
REAL_MIN_CONFIDENCE = 0.30                 # default acceptance threshold on the real robot (as sam3_ros score_th)
MAX_MASK_FRACTION = 0.25                   # a tool never covers more than this fraction of the image


@dataclass
class Instances:
    """Candidate instances returned by a segmenter (masks: N x H x W, scores: N)."""
    masks: np.ndarray
    scores: np.ndarray

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def best(self) -> int:
        return int(np.argmax(self.scores))


class Segmenter(Protocol):
    def segment(self, image: Image.Image, prompt: str) -> Instances: ...


class Sam3Segmenter:
    """Local SAM 3 (HuggingFace weights, CPU). Weights are loaded on first use."""

    def __init__(self, model_id: str = SAM3_MODEL_ID):
        self.model_id = model_id
        self._processor = None
        self._model = None

    def _load(self) -> None:
        import json
        from huggingface_hub import hf_hub_download
        from transformers import CLIPTokenizer, Sam3ImageProcessorFast, Sam3Model, Sam3Processor

        local = Path(self.model_id)
        cfg_path = (local / "processor_config.json" if local.is_dir() else
                    Path(hf_hub_download(self.model_id, "processor_config.json", local_files_only=True)))
        cfg = json.loads(cfg_path.read_text())
        if isinstance(cfg.get("image_processor"), dict):
            # Combined image/video checkpoint keeps the preprocessor nested here.
            image_processor = Sam3ImageProcessorFast.from_dict(cfg["image_processor"])
            tokenizer = CLIPTokenizer.from_pretrained(self.model_id, local_files_only=True)
            self._processor = Sam3Processor(image_processor, tokenizer, target_size=cfg.get("target_size"))
        else:
            self._processor = Sam3Processor.from_pretrained(self.model_id, local_files_only=True)
        self._model = Sam3Model.from_pretrained(self.model_id, local_files_only=True).to("cpu")
        self._model.eval()

    def segment(self, image: Image.Image, prompt: str) -> Instances:
        import torch
        if self._model is None:
            log.info("loading %s weights", self.model_id)
            self._load()
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            outputs = self._model(**inputs)
        res = self._processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist())[0]
        return Instances(res["masks"].detach().cpu().numpy(), res["scores"].detach().cpu().numpy())


@lru_cache(maxsize=1)
def get_segmenter(model_id: str = SAM3_MODEL_ID) -> Sam3Segmenter:
    return Sam3Segmenter(model_id)


# --------------------------------------------------------------------------- masks
def blend(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, weight: float) -> None:
    """In-place alpha blend of ``color`` over ``rgb`` where ``mask`` is True."""
    rgb[mask] = (rgb[mask] * (1.0 - weight) + color * weight).astype(np.uint8)


def make_overlay(rgb: np.ndarray, target: np.ndarray, others: Optional[np.ndarray] = None) -> np.ndarray:
    overlay = rgb.copy()
    if others is not None:
        blend(overlay, others & ~target, OTHERS_COLOR, 0.5)
    blend(overlay, target, TARGET_COLOR, 0.65)
    return overlay


def mask_metrics(mask: np.ndarray) -> Dict[str, Any]:
    v, u = np.where(mask)
    if len(u) == 0:
        return {"active_pixel_count": 0, "bbox_2d_pixels": [0, 0, 0, 0], "centroid_2d_pixels": [0.0, 0.0]}
    return {"active_pixel_count": int(len(u)),
            "bbox_2d_pixels": [int(u.min()), int(v.min()), int(u.max()), int(v.max())],
            "centroid_2d_pixels": [round(float(u.mean()), 2), round(float(v.mean()), 2)]}


def save_mask_and_overlay(mask: np.ndarray, overlay: np.ndarray, mask_path: Path, overlay_path: Path) -> None:
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)


# --------------------------------------------------------------------------- segmentation
@dataclass
class Segmentation:
    mask: np.ndarray            # uint8 0/255, H x W
    overlay: np.ndarray         # uint8 H x W x 3
    data: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def segment_image(image: Image.Image, prompts: Sequence[str], segmenter: Segmenter, *,
                  require_match: bool = False, others_min_score: Optional[float] = None,
                  min_confidence: Optional[float] = None) -> Segmentation:
    """Segment the best instance for the first prompt that yields any candidates.

    ``require_match`` raises if no prompt matches (otherwise an empty mask is returned);
    ``others_min_score`` also tints other candidates above that score in the overlay;
    ``min_confidence`` makes the stage FAIL (instead of only warning) when the best instance
    scores below it or when the mask is implausibly large for a single instrument.
    """
    rgb = np.array(image.convert("RGB"))
    instances: Optional[Instances] = None
    used: Optional[str] = None
    for prompt in prompts:
        log.info("SAM3 prompt: '%s'", prompt)
        instances = segmenter.segment(image, prompt)
        used = prompt
        if len(instances):
            break
    warnings: List[str] = []
    h, w = rgb.shape[:2]
    if instances is None or not len(instances):
        if require_match:
            raise StageError("segmentation", f"no instrument detected for prompts {list(prompts)}")
        warnings.append(f"no instance found for prompts {list(prompts)}")
        target = np.zeros((h, w), dtype=bool)
        score, others = 0.0, None
        n = 0
    else:
        best = instances.best
        score, n = float(instances.scores[best]), len(instances)
        for idx, sc in enumerate(instances.scores):
            log.info("instance %d: %.1f%%%s", idx, sc * 100, "  <- selected" if idx == best else "")
        target = instances.masks[best] > 0
        others = None
        if others_min_score is not None:
            ids = [i for i in range(n) if i != best and instances.scores[i] >= others_min_score]
            others = np.any(instances.masks[ids] > 0, axis=0) if ids else None
        if score < LOW_CONFIDENCE:
            warnings.append(f"very low detection confidence ({score:.3f} < {LOW_CONFIDENCE}): "
                            "the instrument may not be in the scene")
    if min_confidence is not None and instances is not None and len(instances):
        if score < min_confidence:
            raise StageError("segmentation", f"detection confidence {score:.3f} is below the acceptance "
                                             f"threshold {min_confidence:.2f} for prompt '{used}': refusing to continue")
        fraction = float(target.mean())
        if fraction > MAX_MASK_FRACTION:
            raise StageError("segmentation", f"mask covers {fraction:.0%} of the image (> {MAX_MASK_FRACTION:.0%}): "
                                             "not a single instrument, refusing to continue")
    data = {"segmentation_backend": "sam3", "prompts_tried": list(prompts), "prompt_used": used,
            "num_instances": n, "confidence_score": round(score, 4), **mask_metrics(target)}
    return Segmentation((target.astype(np.uint8) * 255), make_overlay(rgb, target, others), data, warnings)
