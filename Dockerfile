# syntax=docker/dockerfile:1.7
FROM python:3.10-slim-bookworm

ARG FLASHHEAD_REPO=https://github.com/Soul-AILab/SoulX-FlashHead.git
ARG FLASHHEAD_REF=9bc03de06bb0de82cd6bc477804512ae06144bf2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG FLASH_ATTN_VERSION=2.8.0.post2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    FLASHHEAD_HOME=/opt/flashhead \
    MODEL_ROOT=/workspace/models \
    FLASHHEAD_MODEL_VARIANT=lite \
    PURGE_PRO_MODELS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    RUN_MODE=gradio \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# TorchInductor compiles native kernels during the first inference. Keep the
# compiler toolchain in the runtime image instead of installing it on every Pod.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        torch==2.7.1 \
        torchvision==0.22.1 \
        torchaudio==2.7.1 \
        --index-url "${TORCH_INDEX_URL}"

RUN git clone "${FLASHHEAD_REPO}" "${FLASHHEAD_HOME}" \
    && cd "${FLASHHEAD_HOME}" \
    && git checkout "${FLASHHEAD_REF}" \
    && git submodule update --init --recursive

WORKDIR /opt/flashhead

# The Jarvis runtime uses only FlashHead Lite. Remove Pro from the Gradio model
# selector so a slow Pro run cannot be started accidentally.
RUN python - <<'PY'
from pathlib import Path

path = Path("/opt/flashhead/gradio_app_streaming.py")
text = path.read_text(encoding="utf-8")
old = 'choices=["pro", "lite"],\n                    value="lite",'
new = 'choices=["lite"],\n                    value="lite",\n                    interactive=False,'
if old not in text:
    raise RuntimeError("Could not locate FlashHead model selector to patch")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
PY

# Avoid upstream dependency conflicts:
# - only headless OpenCV is needed in the container
# - torch 2.7.1 already pins the matching NCCL package
# - install MediaPipe separately with a NumPy/OpenCV combination that has wheels for Python 3.10
RUN sed \
      -e '/^opencv-python>=/d' \
      -e '/^opencv-python-headless>=/d' \
      -e '/^mediapipe==/d' \
      -e '/^nvidia-nccl-cu12==/d' \
      requirements.txt > /tmp/flashhead-requirements.txt \
    && python -m pip install \
        numpy==1.26.4 \
        opencv-python-headless==4.11.0.86 \
        mediapipe==0.10.9 \
    && python -m pip install -r /tmp/flashhead-requirements.txt \
    && python -m pip install "huggingface_hub>=0.34.0,<1.0"

# Use the official prebuilt wheel instead of compiling FlashAttention.
RUN python -m pip install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

COPY scripts/download_models.py /opt/flashhead-container/download_models.py
COPY scripts/start.sh /opt/flashhead-container/start.sh

RUN chmod +x /opt/flashhead-container/start.sh \
    && mkdir -p /workspace/models /workspace/.cache/huggingface \
    && python - <<'PY'
import shutil
import sys

import flash_attn
import mediapipe
import torch

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
assert shutil.which("gcc"), "gcc is missing"
assert shutil.which("g++"), "g++ is missing"
assert shutil.which("ninja"), "ninja is missing"
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA wheel:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("MediaPipe:", mediapipe.__version__)
print("C compiler:", shutil.which("gcc"))
print("C++ compiler:", shutil.which("g++"))
print("Ninja:", shutil.which("ninja"))
PY

EXPOSE 7860
VOLUME ["/workspace"]
ENTRYPOINT ["/opt/flashhead-container/start.sh"]