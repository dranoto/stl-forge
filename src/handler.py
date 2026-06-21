"""
STL Forge — RunPod serverless handler.

Input event:
  input.image: str         — base64, data URL (data:image/...;base64,...), or http(s) URL. Required.
  input.target_faces: int  — decimation target. Default 100000. Range 10k-500k.
  input.mc_resolution: int — Marching Cubes voxel grid. Default 256. Lower for less VRAM.
  input.quant: str         — "fp16" (default), "fp8", or "int8" — when set, quantize the DiT.

Output (wrapped by RunPod under "output" — see rp_format_response):
  For files that fit in the 20 MB /runsync cap (<= ~5 MB binary, <= ~100k faces):
    stl_b64: str           — binary STL, base64-encoded (backward-compatible path)
    report: dict           — vertices, faces, watertight, bbox, stl_bytes
  For files that exceed it (>= ~5 MB binary):
    stl_url: str           — presigned HTTPS URL on R2 (24h TTL); client should GET this
    stl_bytes: int         — size of the STL on disk (for client display)
    report: dict           — same fields as above

The 5 MB / 100k-face threshold is conservative. Actual /runsync response cap is
20 MB (7 MB base64) so the threshold leaves headroom for the JSON envelope.

If BUCKET_ENDPOINT_URL / BUCKET_ACCESS_KEY_ID / BUCKET_SECRET_ACCESS_KEY are not
set on the endpoint, large files fall back to stl_b64 (which will 400 at the
RunPod job-done callback) and log a clear warning. Set those env vars to enable
the upload path.
"""

import os
import io
import sys
import base64
import tempfile
import time
import traceback
from urllib.request import urlopen

import runpod
import numpy as np
import trimesh
from PIL import Image

# ---- Constants ---------------------------------------------------------------

HF_MODEL = os.environ.get("HF_MODEL", "tencent/Hunyuan3D-2.1")
DEFAULT_TARGET_FACES = int(os.environ.get("DEFAULT_TARGET_FACES", 100_000))
DEFAULT_MC_RESOLUTION = int(os.environ.get("DEFAULT_MC_RESOLUTION", 256))
LOW_VRAM = os.environ.get("LOW_VRAM", "0") == "1"

# R2 / S3-compatible storage for >20 MB outputs. When all four env vars are
# set, files that exceed STL_B64_INLINE_MAX_BYTES get uploaded and the
# response returns a presigned URL instead of a base64 blob.
BUCKET_ENDPOINT_URL = os.environ.get("BUCKET_ENDPOINT_URL", "").strip()
BUCKET_ACCESS_KEY_ID = os.environ.get("BUCKET_ACCESS_KEY_ID", "").strip()
BUCKET_SECRET_ACCESS_KEY = os.environ.get("BUCKET_SECRET_ACCESS_KEY", "").strip()
BUCKET_NAME = os.environ.get("BUCKET_NAME", "<bucket-name>").strip()
R2_CONFIGURED = bool(BUCKET_ENDPOINT_URL and BUCKET_ACCESS_KEY_ID and BUCKET_SECRET_ACCESS_KEY)

# Threshold (in bytes of base64) above which we upload instead of inlining.
# /runsync caps the response at 20 MB; 7 MB base64 leaves ~13 MB headroom for
# the JSON envelope + report + URL string.
STL_B64_INLINE_MAX_BYTES = 7 * 1024 * 1024  # 7 MB of base64 → ~5 MB binary

# ---- Lazy model loader -------------------------------------------------------

_pipeline = None
_load_time = None


# ---- Cached-snapshot resolver -------------------------------------------------

def _resolve_cached_snapshot(model_id: str) -> str:
    """Resolve the local path to RunPod's pre-cached snapshot of `model_id`.

    RunPod Serverless pre-caches models to::

        $HF_HOME/hub/models--{org}--{name}/snapshots/{commit_hash}/

    where $HF_HOME defaults to ``/runpod-volume/huggingface-cache`` (set in
    the Dockerfile). The commit hash is auto-pinned by RunPod when you save
    the endpoint's Model field — we never hardcode it; we just scan for
    whatever's on disk and return the first snapshot.

    Falls back to ``model_id`` (which will fail under HF_HUB_OFFLINE=1) if
    no snapshot is found, surfacing a clear "Model field not set" error
    to the user instead of silently downloading.
    """
    from pathlib import Path
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    parts = model_id.split("/")
    if len(parts) != 2:
        return model_id
    cache_root = Path(hf_home) / "hub" / f"models--{parts[0]}--{parts[1]}" / "snapshots"
    if cache_root.is_dir():
        snapshots = sorted(d for d in cache_root.iterdir() if d.is_dir())
        if snapshots:
            return str(snapshots[0])
    return model_id


# ---- Lazy model loader -------------------------------------------------------

def load_pipeline():
    """Load the Hunyuan3D shape pipeline once per worker. The trick from
    Wan2GP — model sits in module scope, never in handler scope."""
    global _pipeline, _load_time
    if _pipeline is not None:
        return _pipeline

    print(f"[stl-forge] loading {HF_MODEL} (low_vram={LOW_VRAM})", flush=True)
    t0 = time.time()
    # Hunyuan3D 2.1 repo is nested: /app/hunyuan3d-2.1/hy3dshape/hy3dshape/pipelines.py
    # We need the OUTER hy3dshape/ on sys.path so Python finds the INNER one
    # as the `hy3dshape` package. (PYTHONPATH in the Dockerfile points at the
    # repo root, not this level, so we insert explicitly.)
    sys.path.insert(0, "/app/hunyuan3d-2.1/hy3dshape")
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except ImportError:
        # Fallback in case the user replaces the cloned repo with the
        # pip-distributed `hy3dgen` package instead.
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    # Load from RunPod's pre-cached snapshot. We don't pass a revision
    # because RunPod auto-pins to a commit hash; the resolver finds
    # whatever's actually on disk. local_files_only=True keeps HF from
    # trying to verify against the Hub API (which HF_HUB_OFFLINE=1
    # would block anyway, but this also gives a clearer error path).
    local_path = _resolve_cached_snapshot(HF_MODEL)
    print(f"[stl-forge] local snapshot path: {local_path}", flush=True)
    _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        local_path,
        local_files_only=True,
    )
    # Hunyuan3D exposes `low_vram_mode` only on some pipeline variants; try safely
    if LOW_VRAM and hasattr(_pipeline, "enable_low_vram"):
        _pipeline.enable_low_vram()
    _load_time = time.time() - t0
    print(f"[stl-forge] loaded in {_load_time:.1f}s", flush=True)
    return _pipeline


# ---- Image I/O ---------------------------------------------------------------

def load_image(image_input: str) -> Image.Image:
    """Resolve an image from base64 / data URL / http(s) URL into a PIL Image."""
    if image_input.startswith(("http://", "https://")):
        with urlopen(image_input, timeout=30) as r:
            data = r.read()
    elif image_input.startswith("data:image"):
        b64 = image_input.split(",", 1)[1]
        data = base64.b64decode(b64)
    else:
        # assume raw base64
        data = base64.b64decode(image_input)
    return Image.open(io.BytesIO(data))


def flatten_alpha(img: Image.Image) -> Image.Image:
    """Flatten RGBA onto white. Hunyuan3D expects RGB."""
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


# ---- Printability pipeline ---------------------------------------------------

def make_printable(mesh: trimesh.Trimesh, target_faces: int = 100_000) -> trimesh.Trimesh:
    """Clean, repair, decimate, centre — turns a generated mesh into something
    a slicer can chew on. See [[STL Forge - Project Plan]] §3 for rationale."""
    # 1. basic cleanup
    mesh.process(validate=True)
    mesh.merge_vertices()
    # trimesh 4.x: remove_duplicate_faces() AND remove_degenerate_faces()
    # were both removed entirely. Replacements:
    #   mesh.update_faces(mesh.unique_faces())        # dedup
    #   mesh.update_faces(mesh.nondegenerate_faces()) # drop zero-area
    # (Both removed in trimesh 4.0; no module-level replacement either.)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    # 2. fix holes if any (best-effort, swallow errors)
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception as e:
        print(f"[stl-forge] fill_holes: {e}", flush=True)

    # 3. fix winding + normals + inversion
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)

    # 4. keep the largest connected component — drop floating fragments
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.faces))

    # 5. decimate to target face count
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception as e:
            print(f"[stl-forge] decimation failed, keeping original: {e}", flush=True)

    # 6. centre on origin (slicer-friendly)
    mesh.rezero()
    return mesh


# ---- R2 upload ----------------------------------------------------------------

def upload_to_r2(stl_bytes: bytes, job_id: str) -> str | None:
    """Upload an STL to Cloudflare R2 (or any S3-compatible storage) and return
    a presigned HTTPS URL. Returns None on any failure (the caller falls back
    to inline stl_b64 and accepts that the response may 400 at the job-done
    callback if the file is too large).

    Uses boto3 directly rather than `runpod.serverless.utils.rp_upload` — the
    runpod SDK's function names change between versions and the upload helper
    bakes in its own bucket_name convention that doesn't match our R2 setup.
    """
    if not R2_CONFIGURED:
        print("[stl-forge] R2 not configured (BUCKET_* env vars missing)", flush=True)
        return None
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        print(f"[stl-forge] boto3 not installed, can't upload: {e}", flush=True)
        return None
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=BUCKET_ENDPOINT_URL,
            aws_access_key_id=BUCKET_ACCESS_KEY_ID,
            aws_secret_access_key=BUCKET_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
        )
        key = f"stl-forge/{job_id}.stl"
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=stl_bytes)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET_NAME, "Key": key},
            ExpiresIn=86400,  # 24h; lifecycle rule cleans up the actual object after 7 days
        )
        print(f"[stl-forge] uploaded {len(stl_bytes)/1024/1024:.1f} MB STL → R2: {url}", flush=True)
        return url
    except Exception as e:
        print(f"[stl-forge] R2 upload failed: {e}", flush=True)
        return None


# ---- Handler -----------------------------------------------------------------

# Default behaviour: every STL goes to the S3-compatible bucket and the
# response returns a presigned URL. The client downloads the bytes from that
# URL. This keeps the handler simple (one code path) and uniform (no surprise
# 502s when an edge-timeout drops a >20MB /runsync response).
#
# Pass input.force_inline=True to opt back into the b64-in-the-response path
# for a single request. Only safe when the resulting STL is <7MB base64
# (≤100k faces); the client will still work for larger files but RunPod's
# 20MB /runsync cap will reject the response at job-done time.
FORCE_INLINE_KEY = "force_inline"


def handler(event):
    """RunPod serverless handler. Returns a dict with stl_url (and stl_bytes
    + report) for every output, or an error. RunPod wraps this in
    `{"output": ...}` automatically.

    By default the handler uploads the STL to the S3-compatible bucket
    configured via BUCKET_* env vars and returns a 24h presigned URL. Pass
    `force_inline=True` in the input to get the base64-in-response path
    instead (only safe for ~100k faces or fewer).
    """
    from runpod.serverless.utils.rp_cleanup import clean

    tmp_path = None
    try:
        inp = event.get("input") or {}
        if "image" not in inp:
            return {"error": "missing 'image' in input (base64, data URL, or http(s) URL)"}

        target_faces = int(inp.get("target_faces", DEFAULT_TARGET_FACES))
        target_faces = max(10_000, min(500_000, target_faces))
        mc_resolution = int(inp.get("mc_resolution", DEFAULT_MC_RESOLUTION))
        mc_resolution = max(64, min(512, mc_resolution))
        force_inline = bool(inp.get(FORCE_INLINE_KEY, False))

        print(
            f"[stl-forge] request: target_faces={target_faces} mc_resolution={mc_resolution} "
            f"force_inline={force_inline}",
            flush=True,
        )

        # 1. image → PIL → flattened RGB
        img = load_image(inp["image"])
        img = flatten_alpha(img)
        print(f"[stl-forge] image: {img.size} {img.mode}", flush=True)

        # 2. write to temp file (Hunyuan3D pipeline expects a file path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, "PNG")
            tmp_path = f.name

        # 3. load model (first call only)
        pipe = load_pipeline()

        # 4. generate
        t0 = time.time()
        kwargs = {"num_inference_steps": 30}
        if hasattr(pipe, "octree_resolution"):
            kwargs["octree_resolution"] = mc_resolution
        mesh_result = pipe(image=tmp_path, **kwargs)[0]
        gen_time = time.time() - t0
        print(f"[stl-forge] mesh generated in {gen_time:.1f}s", flush=True)

        # 5. trimesh + printability
        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh_result.vertices),
            faces=np.asarray(mesh_result.faces),
        )
        raw_faces = len(mesh.faces)
        cleaned = make_printable(mesh, target_faces=target_faces)
        print(
            f"[stl-forge] cleaned: {len(cleaned.vertices)} verts, "
            f"{len(cleaned.faces)}/{raw_faces} faces",
            flush=True,
        )

        # 6. export STL (binary, little-endian)
        buf = io.BytesIO()
        cleaned.export(buf, file_type="stl")
        stl_bytes = buf.getvalue()

        # 7. report (always included)
        report = {
            "vertices": int(len(cleaned.vertices)),
            "faces": int(len(cleaned.faces)),
            "watertight": bool(cleaned.is_watertight),
            "bbox_extents": cleaned.extents.tolist() if cleaned.extents is not None else None,
            "stl_bytes": int(len(stl_bytes)),
            "generation_time_s": round(gen_time, 2),
            "model_load_time_s": round(_load_time, 2) if _load_time else None,
            "model": HF_MODEL,
        }

        # 8. choose return path: always S3, unless force_inline=True
        stl_b64 = base64.b64encode(stl_bytes).decode("ascii")
        if force_inline or len(stl_b64) <= STL_B64_INLINE_MAX_BYTES:
            # Inline path (small files, or caller explicitly opted in).
            return {
                "stl_b64": stl_b64,
                "stl_bytes": int(len(stl_bytes)),
                "report": report,
            }

        # Default: upload to S3-compatible storage and return a presigned URL.
        job_id = event.get("id", f"unknown-{int(time.time())}")
        stl_url = upload_to_r2(stl_bytes, job_id)
        if stl_url is None:
            # R2 not configured or upload failed — fall back to inline. If the
            # resulting base64 is too big for /runsync the job-done callback
            # will 502, but that's better than failing the whole request.
            print(
                f"[stl-forge] WARNING: STL is {len(stl_b64)/1024/1024:.1f} MB base64 "
                f"(> {STL_B64_INLINE_MAX_BYTES/1024/1024:.1f} MB inline limit) and R2 upload "
                f"failed. Returning stl_b64 anyway — expect a job-done 400.",
                flush=True,
            )
            return {
                "stl_b64": stl_b64,
                "stl_bytes": int(len(stl_bytes)),
                "report": report,
            }

        return {
            "stl_url": stl_url,
            "stl_bytes": int(len(stl_bytes)),
            "report": report,
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        try:
            clean()
        except Exception:
            pass


# ---- Local test entry point --------------------------------------------------
# `python src/handler.py --test_input '{"input": {"image": "..."}}'`
# `python src/handler.py --rp_serve_api` for a local FastAPI on :8000

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
