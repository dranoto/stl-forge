# STL Forge — image-to-3D-printable-STL on RunPod Serverless
# Build target: linux/amd64 (Apple Silicon must pass --platform linux/amd64)
#
# Build:   docker build --platform linux/amd64 -t thankfulcarp/stl-forge:runpod-latest .
# Test:    docker run --rm -it --gpus all -e MODE_TO_RUN=pod thankfulcarp/stl-forge:runpod-latest
# Push:    docker push thankfulcarp/stl-forge:runpod-latest

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Default args for the model — can be overridden by env vars at runtime.
# LOW_VRAM defaults to 0 because L4 (24 GB) has plenty of headroom for
# shape-only (10 GB official). Setting LOW_VRAM=1 is only useful when
# targeting 16 GB GPUs (A4000) or going BELOW the official floor.
ENV HF_MODEL=tencent/Hunyuan3D-2.1
ENV DEFAULT_TARGET_FACES=100000
ENV DEFAULT_MC_RESOLUTION=256
ENV LOW_VRAM=0
# Offline mode: the model is BAKED into the image at /root/.cache/hy3dgen/...
# (see RUN below), so HF_HUB_OFFLINE=1 keeps us from accidentally hitting
# the network. No need to point HF_HOME at a RunPod pre-cache anymore.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
# Pin the pip index to the cu128 torch repo so Hunyuan3D's `pip install -r
# requirements.txt` finds a torch wheel via the build-isolated resolver.
# cu128 wheels are needed for Blackwell (sm_120 / RTX PRO 6000) support;
# cu124 wheels only go up to sm_90 and fail the RunPod fitness check on
# newer GPUs.
ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple

# System deps: Python 3.10 (Hunyuan3D 2.1 tested), build tools for any wheel compiles
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv python3.10-dev \
    git wget curl ca-certificates \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch 2.7.0 + CUDA 12.8 wheels. Required for Blackwell (sm_120) support
# on newer RunPod worker GPUs (RTX PRO 6000 Blackwell Server Edition, etc.).
# cu124 wheels only support sm_50-sm_90 and fail the RunPod fitness check on Blackwell.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Hunyuan3D 2.1 — shallow clone of the official repo. We use the in-tree
# `hy3dshape` package directly (PYTHONPATH), not a pip install — that matches
# the project's own import pattern and lets us skip the Paint custom_rasterizer
# compile (which is texture-synthesis only, irrelevant for shape-only mode).
ARG HUNYUAN3D_REF=main
RUN git clone --depth 1 --branch ${HUNYUAN3D_REF} \
        https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app/hunyuan3d-2.1

# Install Hunyuan3D's full requirements.txt EXCEPT the slow source-build packages
# (basicsr, tb_nightly, deepspeed, cupy, bpy) and demo-only packages
# (gradio, fastapi, uvicorn, pymeshlab's heavier cousins). Keep the rest so
# transitive deps (numpy, pymeshlab, scipy, opencv, etc.) come along for free
# — they're imported at module load in hy3dshape, even for shape-only inference.
#
# Verified with import test on ania: hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline
# imports cleanly with this filter set. Without pymeshlab or numpy, the import
# fails with ModuleNotFoundError. --no-deps was a bad call; this is the fix.
RUN grep -v -E '^(basicsr|tb_nightly|deepspeed|cupy-cuda12x|bpy|realesrgan|rembg$|gradio|fastapi|uvicorn|pytorch-lightning|pandas|pygltflib|open3d|xatlas|pythreejs|configargparse|custom_rasterizer|hy3d|svgelements|--extra-index-url)' \
        /app/hunyuan3d-2.1/requirements.txt > /tmp/hy3d-filtered.txt \
    && pip install --no-cache-dir -r /tmp/hy3d-filtered.txt

# Our application deps — pinned, minimal, no gradio (we don't need a UI in the container)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code (placed BEFORE the bake step so app-code changes don't
# invalidate the model-bake cache layer)
COPY src/ /app/src/

# Bake the Hunyuan3D 2.1 shape-only model into the image at the path
# hy3dgen.shapgen expects (/root/.cache/hy3dgen/tencent/Hunyuan3D-2.1/).
# This eliminates cold-start model downloads across ALL workers + hosts.
#
# We grab ONLY the shape-related subfolders:
#   - hunyuan3d-dit-v2-1/   (DiT, 7.03 GB)
#   - hunyuan3d-vae-v2-1/   (Shape VAE, 0.63 GB)
# We SKIP hunyuan3d-paintpbr-v2-1/ (~50 GB of texture models) and
# hy3dpaint/ (texture pipeline code).
#
# Total baked: ~7.7 GB. Image grows from ~12 GB to ~20 GB.
# Build time impact: +3-5 min for the model download at build time.
#
# tencent/Hunyuan3D-2.1 is NOT gated (community license), no HF_TOKEN needed.
#
# Why a script file instead of `python -c "..."`: Dockerfile parser treats
# lines starting with `from` as a FROM instruction, even inside a python -c
# string. A separate script file avoids that gotcha.
COPY scripts/bake_model.py /tmp/bake_model.py
# Note: the apt-installed python3.10 has no `python` symlink, so use `python3`.
# (Earlier builds failed with exit 127 "python: not found" because of this.)
# Also: HF_HUB_OFFLINE=1 is set above for runtime safety, but the bake step
# itself needs to download the model — override HF_HUB_OFFLINE=0 just for
# this RUN. (Without the override, snapshot_download fails with
# "outgoing traffic has been disabled".)
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 /tmp/bake_model.py

# Default args for the model — can be overridden by env vars at runtime.
# LOW_VRAM defaults to 0 because L4 (24 GB) has plenty of headroom for
# shape-only (10 GB official). Setting LOW_VRAM=1 is only useful when
# targeting 16 GB GPUs (A4000) or going BELOW the official floor.
ENV HF_MODEL=tencent/Hunyuan3D-2.1
ENV DEFAULT_TARGET_FACES=100000
ENV DEFAULT_MC_RESOLUTION=256
ENV LOW_VRAM=0

# Healthcheck — RunPod uses the handler's importable state, not HTTP, but a
# quick python import check catches the most common "image won't even start" failures
HEALTHCHECK NONE

# Entrypoint — RunPod serverless handler
CMD ["python3", "-u", "/app/src/handler.py"]
