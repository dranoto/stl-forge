#!/usr/bin/env python3
"""Bake the Hunyuan3D 2.1 SHAPE-ONLY model into the Docker image at the path
hy3dgen.shapgen expects (/root/.cache/hy3dgen/tencent/Hunyuan3D-2.1/).

We grab ONLY:
  - hunyuan3d-dit-v2-1/   (DiT, 7.03 GB)
  - hunyuan3d-vae-v2-1/   (Shape VAE, 0.63 GB)

We SKIP hunyuan3d-paintpbr-v2-1/ (~50 GB of PBR texture models) and
hy3dpaint/ (texture pipeline code).

Total baked: ~7.7 GB.
"""
import os
import sys

from huggingface_hub import snapshot_download

LOCAL_DIR = "/root/.cache/hy3dgen/tencent/Hunyuan3D-2.1"

def main() -> int:
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"[bake] downloading tencent/Hunyuan3D-2.1 (shape-only) to {LOCAL_DIR}", flush=True)
    try:
        snapshot_download(
            repo_id="tencent/Hunyuan3D-2.1",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=["hunyuan3d-dit-v2-1/*", "hunyuan3d-vae-v2-1/*"],
        )
    except Exception as e:
        print(f"[bake] FAILED: {e}", file=sys.stderr, flush=True)
        return 1

    # Sanity check: the files we expected should be there
    expected = [
        os.path.join(LOCAL_DIR, "hunyuan3d-dit-v2-1", "model.fp16.ckpt"),
        os.path.join(LOCAL_DIR, "hunyuan3d-dit-v2-1", "config.yaml"),
        os.path.join(LOCAL_DIR, "hunyuan3d-vae-v2-1", "model.fp16.ckpt"),
        os.path.join(LOCAL_DIR, "hunyuan3d-vae-v2-1", "config.yaml"),
    ]
    for path in expected:
        if not os.path.exists(path):
            print(f"[bake] MISSING expected file: {path}", file=sys.stderr, flush=True)
            return 1

    # Report size
    total_bytes = 0
    for root, _, files in os.walk(LOCAL_DIR):
        for f in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    print(f"[bake] DONE: shape-only model baked ({total_bytes / 1024**3:.2f} GB)", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
