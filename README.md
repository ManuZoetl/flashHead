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
- FlashHead dependencies and Gradio test UI
- automatic download of only `Model_Lite`, `VAE_LTX`, and Wav2Vec2

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
| GPU | RTX 4090 24 GB recommended |
| Container image | `manuztl/flashhead:latest` |
| Start command | empty |
| Container disk | 30 GB |
| Persistent volume | 30 GB minimum |
| Volume mount path | `/workspace` |
| HTTP port | `7860` |
| TCP port | `22` optional |

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
```

The model repositories are public, so an HF token is normally not required.

## Startup modes

### Gradio UI

```text
RUN_MODE=gradio
```

Starts the upstream FlashHead streaming demo on port `7860`.

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
