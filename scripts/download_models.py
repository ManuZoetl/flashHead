#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/workspace/models"))
FLASHHEAD_DIR = MODEL_ROOT / "SoulX-FlashHead-1_3B"
WAV2VEC_DIR = MODEL_ROOT / "wav2vec2-base-960h"
MAX_WORKERS = int(os.environ.get("HF_MAX_WORKERS", "8"))

FLASHHEAD_REPO = "Soul-AILab/SoulX-FlashHead-1_3B"
WAV2VEC_REPO = "facebook/wav2vec2-base-960h"

LITE_FILES = (
    "Model_Lite/config.json",
    "Model_Lite/diffusion_pytorch_model.safetensors",
    "VAE_LTX/config.json",
    "VAE_LTX/diffusion_pytorch_model.safetensors",
)

LITE_PATTERNS = [
    "Model_Lite/*",
    "VAE_LTX/*",
    "README.md",
    "LICENSE*",
]


def files_present(root: Path, relative_paths: tuple[str, ...]) -> bool:
    return all((root / relative_path).is_file() for relative_path in relative_paths)


def download_with_retry(
    *,
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str] | None = None,
    attempts: int = 4,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            print(
                f"Downloading {repo_id} to {local_dir} "
                f"(attempt {attempt}/{attempts})",
                flush=True,
            )
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                allow_patterns=allow_patterns,
                max_workers=MAX_WORKERS,
                token=os.environ.get("HF_TOKEN") or None,
            )
            return
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc}", file=sys.stderr, flush=True)
            if attempt == attempts:
                raise
            time.sleep(min(30, attempt * 5))


def ensure_flashhead_lite() -> None:
    if files_present(FLASHHEAD_DIR, LITE_FILES):
        print(f"FlashHead Lite weights already present: {FLASHHEAD_DIR}", flush=True)
        return

    print("FlashHead Lite weights are missing.", flush=True)
    download_with_retry(
        repo_id=FLASHHEAD_REPO,
        local_dir=FLASHHEAD_DIR,
        allow_patterns=LITE_PATTERNS,
    )

    missing_files = [
        str(FLASHHEAD_DIR / relative_path)
        for relative_path in LITE_FILES
        if not (FLASHHEAD_DIR / relative_path).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "FlashHead Lite download completed but required files are missing:\n- "
            + "\n- ".join(missing_files)
        )


def ensure_wav2vec() -> None:
    common_files = (
        "config.json",
        "preprocessor_config.json",
    )
    weights_present = any(
        (WAV2VEC_DIR / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    wav2vec_ready = files_present(WAV2VEC_DIR, common_files) and weights_present

    if wav2vec_ready:
        print(f"Wav2Vec2 weights already present: {WAV2VEC_DIR}", flush=True)
        return

    download_with_retry(repo_id=WAV2VEC_REPO, local_dir=WAV2VEC_DIR)

    weights_present = any(
        (WAV2VEC_DIR / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    if not files_present(WAV2VEC_DIR, common_files) or not weights_present:
        raise FileNotFoundError(
            f"Wav2Vec2 download is incomplete under {WAV2VEC_DIR}"
        )


def main() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    print("Preparing FlashHead Lite model", flush=True)
    ensure_flashhead_lite()
    ensure_wav2vec()
    print("Model preparation complete.", flush=True)


if __name__ == "__main__":
    main()
