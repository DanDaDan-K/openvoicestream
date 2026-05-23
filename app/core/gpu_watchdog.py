"""GPU/NPU health watchdog stub (Week 1).

Always returns OK in Week 1. Week 2 will plug in real hardware probes
(GPU memory pressure, NPU runtime status, thermal throttling, CUDA
context health, device fault). Do not import CUDA/RKNN/Hailo/NVML
here — the stub must stay importable on every target.
"""

from __future__ import annotations


def is_ok() -> bool:
    """Return True when hardware is healthy for new traffic.

    Week 1: hard-coded True.
    TODO(week2): real GPU/NPU telemetry — see Deliverable 3 spec.
    """
    return True
