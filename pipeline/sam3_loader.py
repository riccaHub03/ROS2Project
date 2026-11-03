"""Caricamento locale SAM3 con configurazioni immagine autonome o annidate."""

import json
from pathlib import Path


def load_sam3(model_id="facebook/sam3"):
    from huggingface_hub import hf_hub_download
    from transformers import CLIPTokenizer, Sam3ImageProcessorFast, Sam3Model, Sam3Processor

    local = Path(model_id)
    config_path = (local / "processor_config.json" if local.is_dir() else
                   Path(hf_hub_download(model_id, "processor_config.json", local_files_only=True)))
    config = json.loads(config_path.read_text())
    if isinstance(config.get("image_processor"), dict):
        # Il checkpoint combinato immagine/video conserva qui il preprocessore.
        image_processor = Sam3ImageProcessorFast.from_dict(config["image_processor"])
        tokenizer = CLIPTokenizer.from_pretrained(model_id, local_files_only=True)
        processor = Sam3Processor(image_processor, tokenizer, target_size=config.get("target_size"))
    else:
        processor = Sam3Processor.from_pretrained(model_id, local_files_only=True)
    model = Sam3Model.from_pretrained(model_id, local_files_only=True).to("cpu")
    model.eval()
    return processor, model
