# STL Forge — Hub-side handler entrypoint.
#
# This file exists at .runpod/handler.py so RunPod Hub's static analysis
# finds it (the docs say ".runpod directory takes precedence over the
# root directory"). Hub imports the module to verify `def handler` AND
# `runpod.serverless.start(...)` are both reachable at module load —
# not gated by `if __name__ == "__main__":` (which only fires when the
# file is run directly, not when imported).
#
# The real implementation lives in src/handler.py and is imported here.
# The Docker runtime invokes `python src/handler.py` directly (see the
# Dockerfile), which still goes through the `if __name__ == "__main__":`
# guard in src/handler.py for testability.
import os
import sys

import runpod

# Make `src` importable whether Hub runs this from the repo root or from
# the .runpod/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.handler import handler as _src_handler  # noqa: E402  real impl


# Module-level `def handler` — RunPod Hub's static check matches on this.
def handler(event):
    """RunPod entrypoint — thin delegate to src/handler.py.handler."""
    return _src_handler(event)


# Module-bottom call (NOT gated by `if __name__ == "__main__":`) so Hub's
# import-based validator can confirm the runpod worker wires up correctly.
runpod.serverless.start({"handler": handler})
