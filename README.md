# FlashHead Lite Docker image

Reproducible Docker image for running [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) Lite on a RunPod GPU Pod.

The image contains the complete Python/CUDA runtime. Model weights are downloaded on the first container start into `/workspace/models`, so they survive Pod restarts when `/workspace` is mounted as a persistent RunPod volume.

## Docker Hub image

GitHub Actions publishes:

```text
manuztl/flashhead:latest
```

It also publishes immutable tags such as `sha-abc1234`.

## Included in the image

- Python 3.10
- PyTorch 2.7.1 with CUDA 12.8 wheels
- FlashAttention 2.8.0.post2 from the official prebuilt wheel
- FlashHead source pinned to commit `9bc03de06bb0de82cd6bc477804512ae06144bf2`
- FlashHead dependencies and a Lite-only Gradio test UI
- compiler toolchain required by TorchInductor/Triton
- automatic download of only `Model_Lite`, `VAE_LTX`, and Wav2Vec2

FlashHead Pro is intentionally disabled because the Jarvis avatar runtime targets real-time generation.

Conda is not used.

## Required GitHub Actions secrets

Open **Settings → Secrets and variables → Actions** in this repository and create:

```text
DOCKERHUB_USERNAME=manuztl
DOCKERHUB_TOKEN=<Docker Hub access token with read/write permission>
```

Then open **Actions → Build and push FlashHead image → Run workflow**. Every push to `main` also triggers a build.

## RunPod template

| Setting | Value |
|---|---|
| Template type | Pods |
| GPU | RTX 4090 24 GB recommended; A40 also works after warm-up |
| Container image | `manuztl/flashhead:latest` |
| Start command | empty |
| Container disk | 30 GB |
| Persistent volume | 30 GB minimum |
| Volume mount path | `/workspace` |
| HTTP port | `7860` |
| TCP port | none required |

Environment variables:

```text
RUN_MODE=gradio
MODEL_ROOT=/workspace/models
HF_HOME=/workspace/.cache/huggingface
HF_MAX_WORKERS=8
```

Optional:

```text
HF_TOKEN=<Hugging Face token>
PURGE_PRO_MODELS=1
```

`PURGE_PRO_MODELS=1` is the default. On the first start after upgrading from a Pro-capable image, the unused `Model_Pro` and `VAE_Wan` directories are removed from `/workspace/models`.

The model repositories are public, so an HF token is normally not required.

## Startup modes

### Gradio UI

```text
RUN_MODE=gradio
```

Starts the FlashHead streaming demo on port `7860`. The model selector is locked to `lite`.

### Smoke test

```text
RUN_MODE=smoke
```

Runs the official Lite example and writes:

```text
/workspace/flashhead-lite-smoke.mp4
```

### Keep the Pod alive

```text
RUN_MODE=idle
```

Useful for terminal debugging.

### Skip automatic model download

```text
SKIP_MODEL_DOWNLOAD=1
```

## Local build

```bash
docker build -t manuztl/flashhead:local .

docker run --rm --gpus all \
  -p 7860:7860 \
  -v "$PWD/.workspace:/workspace" \
  -e RUN_MODE=gradio \
  manuztl/flashhead:local
```

## First start

On the first start, the container downloads the Lite model, LTX VAE, and Wav2Vec2 into `/workspace/models`. Later starts reuse these files. TorchInductor and Triton caches are also stored below `/workspace/.cache`.

The first inference on a new GPU/runtime combination performs a one-time TorchInductor/Triton compile warm-up. Later Lite generations reuse the cache and run near real time on suitable GPUs.
