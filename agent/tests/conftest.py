"""Test bootstrap: expose DialogueApp under a stable import path.

The dialogue app lives under `agent/apps/dialogue/`, which is outside
the `openvoicestream_agent` namespace. To keep tests independent of cwd,
we register it at `openvoicestream_agent.apps_dialogue_shim`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `apps.*` importable for the CLI loader too.
_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from apps.dialogue.app import DialogueApp  # noqa: E402

import openvoicestream_agent as _ovs  # noqa: E402

# Stash the class on a stable attribute name for tests.
shim_mod_name = "openvoicestream_agent.apps_dialogue_shim"
import types as _types  # noqa: E402

_shim = _types.ModuleType(shim_mod_name)
_shim.DialogueApp = DialogueApp
sys.modules[shim_mod_name] = _shim


@pytest.fixture
def dialogue_cls():
    return DialogueApp
