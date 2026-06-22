# DEPRECATED: canonical implementation moved to voxedge.backends.jetson._deploy_paths
# This module is a backward-compatibility shim.
# All existing imports (TTS_BINARY, ASR_BINARY, resolve_tts_worker_binary, etc.)
# continue to work unchanged. test_trt_edge_llm_ipc_paths.py reloads this module
# to exercise env-driven path resolution — the shim re-imports (not just re-exports)
# the canonical module so that monkeypatched env vars take effect on reload.
import importlib
import sys

# Force a fresh import of the canonical module every time this shim is (re)loaded,
# so that module-level constants re-evaluate against the current os.environ.
# This preserves the ``importlib.reload(server.core.deploy_paths)`` pattern used
# by test_trt_edge_llm_ipc_paths.py.
_canonical_mod = "voxedge.backends.jetson._deploy_paths"
if _canonical_mod in sys.modules:
    del sys.modules[_canonical_mod]

from voxedge.backends.jetson._deploy_paths import *  # noqa: F401,F403
from voxedge.backends.jetson._deploy_paths import __all__  # noqa: F401
