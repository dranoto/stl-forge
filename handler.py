# STL Forge — Root-level entrypoint for RunPod Hub compatibility.
#
# The real implementation lives in src/handler.py. This file exists at the
# repo root so RunPod Hub's static analysis finds it (Hub greps for a
# module-level `def handler(event)` and `runpod.serverless.start(...)` in
# `handler.py` at the repo root).
#
# Behavior is identical to running `python src/handler.py` directly: the
# imported handler function and its module-level dependencies are the same.
import os
import sys

import runpod

# Make `src` importable as a sibling package so `from src.handler import ...`
# resolves whether the file is invoked as `/app/handler.py` (Dockerfile CMD)
# or `python handler.py` (Hub's local check).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.handler import handler as _src_handler  # noqa: E402  real impl

# Module-level `def handler` — RunPod Hub's static check matches on this.
def handler(event):
    """RunPod entrypoint — thin delegate to src/handler.py.handler."""
    return _src_handler(event)


runpod.serverless.start({"handler": handler})
