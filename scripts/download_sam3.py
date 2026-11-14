#!/usr/bin/env python3
"""Download only Transformers SAM3 assets, using the user's authorized HF login."""
from pathlib import Path
import shutil
import os
import hashlib
import json

# Avoid a second on-disk cache of Xet chunks on this space-constrained workstation.
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/external/sam3-prepared"
PATTERNS = ["*.json", "*.txt", "*.safetensors", "LICENSE"]


def main():
    # Check access before allocating the large model, never print credentials.
    api = HfApi()
    try:
        info = api.model_info("facebook/sam3", files_metadata=True)
        api.auth_check("facebook/sam3")
    except Exception as exc:
        raise SystemExit(f"SAM3 access unavailable ({type(exc).__name__}). Request facebook/sam3 access and run .venv/bin/hf auth login.")
    needed = sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors"))
    existing = sum(f.stat().st_size for f in DESTINATION.glob("*.safetensors")) if DESTINATION.exists() else 0
    if shutil.disk_usage(ROOT).free < max(0, needed - existing) + 512 * 1024**2:
        raise SystemExit("Insufficient disk space for SAM3 weights plus a 512 MiB reserve.")
    path = snapshot_download("facebook/sam3", revision=info.sha, local_dir=str(DESTINATION),
                             allow_patterns=PATTERNS, max_workers=1)
    verified = {}
    for entry in info.siblings:
        if not entry.rfilename.endswith(".safetensors"):
            continue
        weight = DESTINATION / entry.rfilename
        with weight.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lfs = entry.lfs
        expected = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if not expected or digest != expected:
            raise SystemExit(f"SAM3 checksum mismatch for {entry.rfilename}; do not run inference with these weights.")
        verified[entry.rfilename] = digest
    (DESTINATION / "download_verification.json").write_text(json.dumps(
        dict(repository="facebook/sam3", revision=info.sha, sha256=verified), indent=2) + "\n")
    print(f"SAM3 Transformers assets: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
