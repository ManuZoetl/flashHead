#!/usr/bin/env bash
set -Eeuo pipefail

export MODEL_ROOT="${MODEL_ROOT:-/workspace/models}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export FLASHHEAD_MODEL_VARIANT="lite"
export PURGE_PRO_MODELS="${PURGE_PRO_MODELS:-1}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

mkdir -p \
  "${MODEL_ROOT}" \
  "${HF_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}"

echo "RUN_MODE=${RUN_MODE:-gradio}"
echo "FLASHHEAD_MODEL_VARIANT=${FLASHHEAD_MODEL_VARIANT}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "CC=${CC}"
echo "CXX=${CXX}"

if [[ "${PURGE_PRO_MODELS}" == "1" ]]; then
  pro_model_dir="${MODEL_ROOT}/SoulX-FlashHead-1_3B/Model_Pro"
  pro_vae_dir="${MODEL_ROOT}/SoulX-FlashHead-1_3B/VAE_Wan"
  if [[ -e "${pro_model_dir}" || -e "${pro_vae_dir}" ]]; then
    echo "Removing unused FlashHead Pro weights from persistent storage"
    rm -rf "${pro_model_dir}" "${pro_vae_dir}"
  fi
fi

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  python /opt/flashhead-container/download_models.py
fi

# Upstream examples expect ./models relative to the repository.
ln -sfn "${MODEL_ROOT}" /opt/flashhead/models
cd /opt/flashhead

python - <<'PY'
import shutil
import torch

print("Torch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("gcc:", shutil.which("gcc"), flush=True)
print("g++:", shutil.which("g++"), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(
        "VRAM GB:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        flush=True,
    )
PY

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

case "${RUN_MODE:-gradio}" in
  gradio)
    echo "Starting FlashHead Lite Gradio on ${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}"
    exec python gradio_app_streaming.py
    ;;
  smoke)
    echo "Running the official FlashHead Lite smoke test"
    exec python generate_video.py \
      --ckpt_dir "${MODEL_ROOT}/SoulX-FlashHead-1_3B" \
      --wav2vec_dir "${MODEL_ROOT}/wav2vec2-base-960h" \
      --model_type lite \
      --cond_image examples/girl.png \
      --audio_path examples/podcast_sichuan_16k.wav \
      --audio_encode_mode stream \
      --save_file "/workspace/flashhead-lite-smoke.mp4"
    ;;
  idle|shell)
    echo "Container ready. RUN_MODE=${RUN_MODE}. Keeping the Pod alive."
    exec sleep infinity
    ;;
  *)
    echo "Unknown RUN_MODE: ${RUN_MODE}. Use gradio, smoke, idle, or pass a command." >&2
    exit 2
    ;;
esac
