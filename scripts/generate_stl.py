#!/usr/bin/env python3
"""
STL Forge — image → 3D-printable STL via the RunPod serverless endpoint.

Usage:
    python generate_stl.py <image_path> [-o output.stl] [--target-faces N] [--mc-resolution N]
        [--bg-removal {auto,always,never}]

Env vars:
    RUNPOD_API_KEY         required
    STL_FORGE_ENDPOINT_ID  required

This is the JARVIS skill's worker script. The agent invokes it when the user
asks for an image-to-STL conversion. The script:
  0. (optional) Pre-process: remove the background via DeepInfra Bria
     (see [[remove-background]]). Default is `auto`: unconditional for JPGs
     (no alpha), auto-skip for PNGs that already have transparent corners.
  1. Encodes the (possibly pre-processed) image to base64
  2. POSTs to /runsync
  3. Decodes the returned stl_b64
  4. Writes the STL bytes to disk
  5. Prints a JSON summary on stdout for the agent to consume
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("STL_FORGE_ENDPOINT_ID")
BASE_URL = "https://api.runpod.ai/v2"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_TIMEOUT = 600  # generous; cold start can be 60–90s

# Canonical home for STL Forge outputs. Every STL the script produces goes
# here by default, and the starting PNG gets copied alongside it so the
# (input, output) pair is always together. Override with `-o` for one-off
# destinations.
DEFAULT_STL_DIR = Path.home() / "workspace" / "STLs"
DEFAULT_STL_DIR.mkdir(parents=True, exist_ok=True)

# Optional: background-removal pre-processing via the remove-background skill.
# The skill script is at ~/.hermes/skills/creative/remove-background/scripts/.
_RB_SCRIPT_DIR = Path.home() / ".hermes/skills/creative/remove-background/scripts"
if str(_RB_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_RB_SCRIPT_DIR))
try:
    from remove_background import remove_background as _rb_remove_background  # noqa: E402
    from remove_background import has_clean_alpha as _rb_has_clean_alpha      # noqa: E402
    _RB_AVAILABLE = True
except Exception as _rb_err:  # ImportError if missing deps, or other module-load issues
    _RB_AVAILABLE = False
    _RB_IMPORT_ERROR = _rb_err


def encode_image(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(
            f"unsupported image format: {p.suffix} (use one of {sorted(SUPPORTED_EXTS)})"
        )
    return base64.b64encode(p.read_bytes()).decode("ascii")


def call_endpoint(
    image_b64: str,
    target_faces: int,
    mc_resolution: int,
    timeout: int,
    sync: bool = False,
) -> dict:
    """Submit a job to the STL Forge endpoint.

    By default uses `/run` (async) + `/status/<job_id>` polling. This is more
    reliable than `/runsync` because Cloudflare's edge has an idle timeout
    (~100s) that can drop the `/runsync` response even when `wait=300000` is
    set, especially for high-poly jobs that take ~90s to generate. With
    `/run`, the POST returns a job_id immediately and the poll request is
    tiny so no edge timeout ever bites.

    Pass sync=True to fall back to `/runsync` (the inline path; only safe
    for jobs that complete in <100s).
    """
    if not API_KEY:
        raise RuntimeError("RUNPOD_API_KEY env var not set — see [[STL Forge - Deploy]] Step 3")
    if not ENDPOINT_ID:
        raise RuntimeError("STL_FORGE_ENDPOINT_ID env var not set — see [[STL Forge - Deploy]] Step 3")

    payload = {
        "input": {
            "image": image_b64,
            "target_faces": target_faces,
            "mc_resolution": mc_resolution,
        },
    }

    if sync:
        # /runsync path (legacy / inline). Capped at wait=300000 (5 min cap per RunPod API).
        url = f"{BASE_URL}/{ENDPOINT_ID}/runsync?wait=300000"
        print(f"[stl-forge] POST {url} (sync; {len(image_b64) / 1024:.0f} KB base64)", file=sys.stderr)
        t0 = time.time()
        r = requests.post(url, headers={"authorization": f"Bearer {API_KEY}", "content-type": "application/json"},
                           json=payload, timeout=timeout)
        r.raise_for_status()
        print(f"[stl-forge] request done in {time.time() - t0:.1f}s", file=sys.stderr)
        return r.json()

    # /run + /status polling (default)
    submit_url = f"{BASE_URL}/{ENDPOINT_ID}/run"
    status_url_tpl = f"{BASE_URL}/{ENDPOINT_ID}/status/{{job_id}}"
    headers = {"authorization": f"Bearer {API_KEY}", "content-type": "application/json"}

    print(f"[stl-forge] POST {submit_url} (async; {len(image_b64) / 1024:.0f} KB base64)", file=sys.stderr)
    t0 = time.time()
    r = requests.post(submit_url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    submit = r.json()
    job_id = submit.get("id")
    if not job_id:
        raise RuntimeError(f"no job_id in /run response: {submit}")
    print(f"[stl-forge] job_id: {job_id}", file=sys.stderr)

    # Poll /status until the job is terminal.
    poll_start = time.time()
    poll_interval = 15
    while True:
        if time.time() - poll_start > timeout:
            raise RuntimeError(f"job {job_id} did not complete within {timeout}s")
        s = requests.get(status_url_tpl.format(job_id=job_id), headers=headers, timeout=30).json()
        status = s.get("status")
        elapsed = time.time() - poll_start
        delay = s.get("delayTime")
        exec_t = s.get("executionTime")
        print(
            f"[stl-forge] [{elapsed:.0f}s] status={status}  delay={delay}s  exec={exec_t}s",
            file=sys.stderr,
        )
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            if status != "COMPLETED":
                raise RuntimeError(f"job {job_id} ended with status={status}: {s}")
            print(f"[stl-forge] job done in {time.time() - t0:.1f}s total", file=sys.stderr)
            return s
        time.sleep(poll_interval)


def download_stl_from_url(stl_url: str, timeout: int = 60) -> bytes:
    """Download an STL from a presigned URL (used when the handler returned
    `stl_url` because the file was too big for the inline `stl_b64` path)."""
    print(f"[stl-forge] downloading STL from {stl_url[:80]}...", file=sys.stderr)
    t0 = time.time()
    r = requests.get(stl_url, timeout=timeout)
    r.raise_for_status()
    elapsed = time.time() - t0
    print(f"[stl-forge] download done in {elapsed:.1f}s ({len(r.content)/1024/1024:.1f} MB)", file=sys.stderr)
    return r.content


def maybe_remove_background(input_path: str, mode: str) -> str:
    """
    Pre-process the image by removing the background.

    mode: 'auto' (default), 'always', or 'never'.
        - 'auto':  unconditional for JPG/JPEG (no alpha possible), auto-skip
                   for PNGs that already have transparent corners, run for
                   PNGs that don't.
        - 'always': always call the API (even on already-clean PNGs).
        - 'never': skip entirely.

    Returns the path to the image to actually use (either the input or a
    newly-saved _no_bg.png). Never raises on a no-op skip.
    """
    if mode == "never":
        print(f"[stl-forge] bg-removal: skipped (--bg-removal=never)", file=sys.stderr)
        return input_path

    if not _RB_AVAILABLE:
        print(
            f"[stl-forge] bg-removal: WARNING requested but remove-background skill "
            f"is not importable: {_RB_IMPORT_ERROR}. Continuing with original image.",
            file=sys.stderr,
        )
        return input_path

    ext = Path(input_path).suffix.lower()
    is_jpg = ext in {".jpg", ".jpeg"}
    is_png = ext == ".png"

    if mode == "auto":
        if is_jpg:
            should_run = True
            reason = "jpg (no alpha possible)"
        elif is_png:
            clean = _rb_has_clean_alpha(input_path)
            should_run = not clean
            reason = "png (auto-skip: already clean alpha)" if clean else "png (no clean alpha)"
        else:
            should_run = False
            reason = f"{ext} (no auto behavior for this extension)"
    elif mode == "always":
        should_run = True
        reason = "--bg-removal=always"
    else:
        raise ValueError(f"invalid --bg-removal mode: {mode!r} (use auto|always|never)")

    if not should_run:
        print(f"[stl-forge] bg-removal: skipped ({reason})", file=sys.stderr)
        return input_path

    print(f"[stl-forge] bg-removal: running ({reason})", file=sys.stderr)
    result = _rb_remove_background(input_path)
    print(
        f"[stl-forge] bg-removal: saved {result['output_path']} "
        f"(${result['cost_usd']:.4f}, {result['elapsed_s']:.1f}s)",
        file=sys.stderr,
    )
    return result["output_path"]


def stage_input(input_path: str) -> str:
    """
    Make sure the input image lives in the canonical STL workspace folder so
    the (input, output) pair is always together. If `input_path` is already
    in the folder, return it unchanged. Otherwise, copy it in (preserving
    extension) and return the new path.

    The bg-removal script and the RunPod step both read from disk, so this
    guarantees they see a stable path.
    """
    import shutil
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {input_path}")
    target = DEFAULT_STL_DIR / p.name
    if p.resolve() == target.resolve():
        return str(p)
    shutil.copy2(p, target)
    print(f"[stl-forge] copied input → {target}", file=sys.stderr)
    return str(target)


def default_stl_path(input_path: str) -> str:
    """Return the canonical STL output path: <DEFAULT_STL_DIR>/<stem>.stl"""
    p = Path(input_path)
    return str(DEFAULT_STL_DIR / f"{p.stem}.stl")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate 3D-printable STL from an image via STL Forge"
    )
    ap.add_argument("image", help="path to input image (PNG/JPG/JPEG/WebP)")
    ap.add_argument(
        "-o", "--output",
        help=f"output STL path (default: {DEFAULT_STL_DIR}/<image-stem>.stl)",
    )
    ap.add_argument(
        "--target-faces",
        type=int,
        default=100_000,
        help="decimation target (10k–500k, default 100k). Higher = more detail, larger STL.",
    )
    ap.add_argument(
        "--mc-resolution",
        type=int,
        default=256,
        help="Marching Cubes voxel grid (64–512, default 256). Lower = less VRAM, coarser mesh.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"total wait time in seconds (default {DEFAULT_TIMEOUT})",
    )
    ap.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Use the legacy /runsync (inline) path instead of /run + polling. "
            "Only safe for jobs that complete in <100s — Cloudflare's edge has "
            "an idle timeout that can drop the response on slow jobs. Default: "
            "/run + /status polling (uniformly reliable for any job size)."
        ),
    )
    ap.add_argument(
        "--bg-removal",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Pre-process the image by removing the background via DeepInfra Bria "
            "($0.018/image). 'auto' (default) = unconditional for JPG (no alpha "
            "possible), auto-skip for PNGs with already-clean alpha. 'always' = "
            "always call the API. 'never' = skip entirely."
        ),
    )
    args = ap.parse_args()

    try:
        # Step 0a: stage the input into the canonical STL workspace folder so
        # the (input, output) pair is always together.
        staged_input = stage_input(args.image)
        # Step 0b: optional background-removal pre-processing
        processed_image = maybe_remove_background(staged_input, args.bg_removal)
        # Step 1: encode (base64) — read from disk to keep the pipeline simple
        image_b64 = encode_image(processed_image)
        # Step 2: hit RunPod
        result = call_endpoint(
            image_b64, args.target_faces, args.mc_resolution, args.timeout,
            sync=args.sync,
        )
    except Exception as e:
        print(f"[stl-forge] error: {e}", file=sys.stderr)
        return 1

    if "error" in result:
        print(f"[stl-forge] endpoint error: {result['error']}", file=sys.stderr)
        return 2

    # RunPod wraps the handler's return value in `output`. Be tolerant of both
    # RunPod wraps the handler's return in `output` — read both shapes for compat.
    if "stl_url" in result:
        # R2 presigned URL path (handler uploaded because the file was too big
        # for inline stl_b64)
        stl_url = result["stl_url"]
        stl_bytes = download_stl_from_url(stl_url)
        report = result.get("report", {})
    elif isinstance(result.get("output"), dict) and "stl_url" in result["output"]:
        # Same as above but nested under output (some runpod SDK versions)
        out = result["output"]
        stl_bytes = download_stl_from_url(out["stl_url"])
        report = out.get("report", {})
    elif "stl_b64" in result:
        stl_bytes = base64.b64decode(result["stl_b64"])
        report = result.get("report", {})
    elif isinstance(result.get("output"), dict) and "stl_b64" in result["output"]:
        out = result["output"]
        stl_bytes = base64.b64decode(out["stl_b64"])
        report = out.get("report", {})
    else:
        print(
            f"[stl-forge] unexpected response (no stl_url or stl_b64 at top level or in 'output'): "
            f"top-level keys={list(result.keys())}",
            file=sys.stderr,
        )
        return 3
    out_path = args.output or default_stl_path(args.image)
    Path(out_path).write_bytes(stl_bytes)

    report = result.get("report", {})
    print(f"[stl-forge] saved: {out_path} ({len(stl_bytes) / 1024:.1f} KB)", file=sys.stderr)
    print(
        f"[stl-forge] vertices={report.get('vertices')}, "
        f"faces={report.get('faces')}, "
        f"watertight={report.get('watertight')}",
        file=sys.stderr,
    )
    if report.get("generation_time_s"):
        print(
            f"[stl-forge] generation_time={report['generation_time_s']}s",
            file=sys.stderr,
        )

    # Final JSON summary — this is what the agent parses.
    print(
        json.dumps(
            {
                "stl_path": str(out_path),
                "stl_bytes": len(stl_bytes),
                "report": report,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
