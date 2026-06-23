# STL Forge — Root-level entrypoint for RunPod Hub compatibility.
#
# The real implementation lives in src/handler.py. This file exists so
# RunPod Hub's static check finds a top-level handler.py with the expected
# `runpod.serverless.start({"handler": handler})` pattern at module bottom.
#
# Behavior is identical to running `python src/handler.py` directly: the
# imported `handler` function and its module-level dependencies (model load,
# env vars, etc.) are the same. Our Dockerfile's CMD can point at either
# /app/handler.py (this file) or /app/src/handler.py — both are equivalent.
import os
import sys

import runpod

# Make `src` importable as a sibling package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.handler import handler  # noqa: E402

runpod.serverless.start({"handler": handler})
