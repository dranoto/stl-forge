# STL Forge — RunPod Serverless wrapper at repo root.
#
# Why this file exists: RunPod Hub's "Handler script" UI field expects
# `handler.py` at the repo root (or a path you specify there). The real
# implementation lives in `.runpod/handler.py` and is kept there to satisfy
# the "bundle everything in .runpod/" preference.
#
# The real impl has an `if __name__ == "__main__":` guard so importing it
# here does NOT auto-start the worker — we start it explicitly below.

import os
import sys

import runpod

# Make `.runpod/` importable so `from handler import handler` resolves the
# full implementation that lives next to this wrapper.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runpod"),
)

from handler import handler as _real_handler  # noqa: E402  full impl


# Module-level `def handler` required by Hub's static analysis + RunPod SDK.
def handler(event):
    """Thin delegate to .runpod/handler.py.handler."""
    return _real_handler(event)


# Single worker-boot call. The real impl's `__name__` guard means this is
# the only place the worker actually starts — no double-start risk.
runpod.serverless.start({"handler": handler})
