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


def has_all(path: Path, patterns: tuple[str, ...]) -> bool:
    return path.exists() and all(any(path.glob(pattern)) for pattern in patterns)


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


def main() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    flashhead_ready = has_all(
        FLASHHEAD_DIR,
        (
            "Model_Lite/*.safetensors",
            "Model_Lite/config.json",
            "VAE_LTX/*",
        ),
    )
    if flashhead_ready:
        print(f"FlashHead Lite weights already present: {FLASHHEAD_DIR}", flush=True)
    else:
        download_with_retry(
            repo_id="Soul-AILab/SoulX-FlashHead-1_3B",
            local_dir=FLASHHEAD_DIR,
            allow_patterns=[
                "Model_Lite/*",
                "VAE_LTX/*",
                "README.md",
                "LICENSE*",
            ],
        )

    wav2vec_ready = has_all(
        WAV2VEC_DIR,
        (
            "config.json",
            "preprocessor_config.json",
            "*.safetensors",
        ),
    ) or has_all(
        WAV2VEC_DIR,
        (
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
        ),
    )

    if wav2vec_ready:
        print(f"Wav2Vec2 weights already present: {WAV2VEC_DIR}", flush=True)
    else:
        download_with_retry(
            repo_id="facebook/wav2vec2-base-960h",
            local_dir=WAV2VEC_DIR,
        )

    print("Model preparation complete.", flush=True)


if __name__ == "__main__":
    main()
