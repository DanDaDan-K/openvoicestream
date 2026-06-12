"""Tests for server.core.leaf_composition (TRACK 1 SLICE 1, standalone).

Covers:
  * union-pull ASR-invariance under TTS choice (+ shared sub-leaf de-dup);
  * precision default resolution + flip (model default re-resolves leaves);
  * validator rejects nano FP16-TTS-n2 (memory), unknown leaf id, double-TTS;
  * env precedence (leaf < overrides) + ${VAR} expansion.
"""

from __future__ import annotations

import pytest

from server.core import leaf_composition as lc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> lc.Registry:
    """The real on-disk registry under configs/leaves/."""
    return lc.load_registry()


ASR_N1 = "asr.qwen3_asr.orin-nx.n1"
ASR_N2 = "asr.qwen3_asr.orin-nx.n2"
TTS_N1 = "tts.qwen3_tts.orin-nx.n1"
TTS_N2 = "tts.qwen3_tts.orin-nx.n2"
TTS_SHARED = "tts.qwen3_tts.shared"


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def test_registry_loads_expected_leaves(registry):
    for lid in (ASR_N1, ASR_N2, TTS_N1, TTS_N2, TTS_SHARED):
        assert lid in registry.leaves, lid
    assert "orin-nx" in registry.devices
    assert "orin-nano" in registry.devices
    assert "qwen3-tts-customvoice" in registry.models


# ---------------------------------------------------------------------------
# Union-pull ASR invariance under TTS choice
# ---------------------------------------------------------------------------

def test_asr_pull_invariant_under_tts_choice(registry):
    asr_only = set(resolve_asr_files(registry))
    with_n1 = set(resolve_asr_files(registry, TTS_N1))
    with_n2 = set(resolve_asr_files(registry, TTS_N2))
    assert asr_only == with_n1 == with_n2
    assert asr_only  # non-empty


def resolve_asr_files(registry, tts_leaf=None):
    selected = [ASR_N2] + ([tts_leaf] if tts_leaf else [])
    all_files = lc.resolve_pull(selected, registry)
    asr_leaf = registry.get_leaf(ASR_N2)
    asr_files = set(asr_leaf.artifacts.files)
    return [f for f in all_files if f in asr_files]


def test_asr_n1_n2_same_engine_files(registry):
    assert lc.resolve_pull([ASR_N1], registry) == lc.resolve_pull([ASR_N2], registry)


def test_pull_dedups_shared_subleaf(registry):
    # Both TTS leaves require the same shared sub-leaf; its files appear once.
    files = lc.resolve_pull([TTS_N1, TTS_N2], registry)
    assert len(files) == len(set(files))
    shared = registry.get_leaf(TTS_SHARED)
    for f in shared.artifacts.files:
        assert files.count(f) == 1


def test_pull_is_deterministic(registry):
    a = lc.resolve_pull([ASR_N2, TTS_N1], registry)
    b = lc.resolve_pull([ASR_N2, TTS_N1], registry)
    assert a == b


# ---------------------------------------------------------------------------
# Precision resolution + flip
# ---------------------------------------------------------------------------

def test_precision_resolves_from_model_default(registry):
    leaf = registry.get_leaf(TTS_N2)
    assert leaf.precision is None  # unset on the leaf
    assert registry.resolve_precision(leaf, "orin-nx") == "fp16"


def test_precision_flip_reresolves_all_leaves(registry):
    # Flip the model default jetson fp16 -> w8a16; every leaf re-resolves.
    flipped = lc.ModelSpec(
        id="qwen3-tts-customvoice",
        default_precision={"jetson": "w8a16"},
    )
    registry.models["qwen3-tts-customvoice"] = flipped
    for lid in (TTS_N1, TTS_N2, TTS_SHARED):
        leaf = registry.get_leaf(lid)
        assert registry.resolve_precision(leaf, "orin-nx") == "w8a16"


def test_explicit_leaf_precision_wins(registry):
    leaf = lc.Leaf(
        id="tts.x.test", capability="tts", model="qwen3-tts-customvoice",
        precision="w4a16",
    )
    assert registry.resolve_precision(leaf, "orin-nx") == "w4a16"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_validate_ok_on_nx(registry):
    # DELTA semantics: base_reservation(4400) + ASR-n2 delta(1925)
    # + TTS-n1 kv1024 delta(3039) + shared sub-leaf(0) = 9364 <= 15656*0.85=13307.
    plan = lc.validate_composition("orin-nx", [ASR_N2, TTS_N1], registry)
    assert plan.device == "orin-nx"
    assert set(plan.leaf_ids) == {ASR_N2, TTS_N1}
    # peak = device base reservation + leaf deltas (shared sub-leaf 0 MB once)
    assert plan.peak_unified_mb == 4400 + 1925 + 3039
    # precision resolved in the plan
    by_id = {r.leaf.id: r.precision for r in plan.resolved}
    assert by_id[TTS_N1] == "fp16"


def test_validate_ok_on_nx_asr_n2_tts_n2(registry):
    # The bug-motivating combo: absolute-sum (6335+9057=15392) falsely rejected
    # this; DELTA math accepts it.
    # base(4400) + ASR-n2 delta(1925) + TTS-n2 delta(4636) = 10961
    # <= 15656*0.85 = 13307 → PASS.
    plan = lc.validate_composition("orin-nx", [ASR_N2, TTS_N2], registry)
    assert plan.peak_unified_mb == 4400 + 1925 + 4636
    assert plan.peak_unified_mb <= plan.headroom_mb


def test_validate_ok_on_nx_asr_n2_tts_n1(registry):
    # base(4400) + ASR-n2 delta(1925) + TTS-n1 kv1024 delta(3039) = 9364 → PASS.
    plan = lc.validate_composition("orin-nx", [ASR_N2, TTS_N1], registry)
    assert plan.peak_unified_mb == 9364
    assert plan.peak_unified_mb <= plan.headroom_mb


def test_validate_rejects_nano_fp16_tts_n1_memory(registry):
    # Conservative product composition still rejects Nano: a standalone
    # 2026-06-12 kv1024 smoke can dual-open ASR-ready + TTS on a cleaned host,
    # but it used swap and is not a wide-margin default.
    # base(4400) + TTS-n1 delta(3039) = 7439 > 7864*0.85 = 6684 → REJECT.
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("orin-nano", [TTS_N1], registry)
    assert "memory budget" in str(exc.value)
    assert "orin-nano" in str(exc.value)


def test_validate_rejects_nano_fp16_tts_n2_memory(registry):
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("orin-nano", [ASR_N2, TTS_N2], registry)
    assert "memory budget" in str(exc.value)
    assert "orin-nano" in str(exc.value)


def test_validate_rejects_unknown_leaf_id(registry):
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("orin-nx", ["tts.qwen3_tts.orin-nano.n2"], registry)
    assert "not built" in str(exc.value)


def test_validate_rejects_double_tts(registry):
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("orin-nx", [TTS_N1, TTS_N2], registry)
    assert "illegal pairing" in str(exc.value)
    assert "tts" in str(exc.value)


def test_validate_rejects_unknown_device(registry):
    with pytest.raises(lc.CompositionError) as exc:
        lc.validate_composition("rk3588", [ASR_N2], registry)
    assert "unknown device" in str(exc.value)


# ---------------------------------------------------------------------------
# Env precedence + ${VAR} expansion
# ---------------------------------------------------------------------------

def test_env_merges_leaf_and_shared(registry):
    env = lc.resolve_env([TTS_N1], registry)
    # from the concrete leaf
    assert "EDGE_LLM_TTS_TALKER_ENGINE" in env
    # from the shared sub-leaf (pulled via requires)
    assert env["EDGE_LLM_TTS_TALKER_BACKEND"] == "qwen3_tts_explicit_kv"


def test_env_override_wins(registry):
    env = lc.resolve_env(
        [TTS_N1], registry,
        overrides={"EDGE_LLM_TTS_TALKER_BACKEND": "custom_override"},
    )
    assert env["EDGE_LLM_TTS_TALKER_BACKEND"] == "custom_override"


def test_env_var_expansion(registry):
    env = lc.resolve_env(
        [ASR_N2], registry,
        overrides={"QWEN3_ARTIFACT_ROOT": "/opt/models/qwen3-edgellm"},
    )
    assert env["EDGE_LLM_ASR_ENGINE_DIR"] == (
        "/opt/models/qwen3-edgellm/engines/orin-nx/highperf-v2/"
        "asr_thinker_full_fp8embed"
    )


def test_env_unknown_var_preserved(registry):
    # No QWEN3_ARTIFACT_ROOT override → the ${VAR} is preserved verbatim.
    env = lc.resolve_env([ASR_N2], registry)
    assert "${QWEN3_ARTIFACT_ROOT}" in env["EDGE_LLM_ASR_ENGINE_DIR"]


def test_resolve_env_standalone_module_level(registry):
    # Standalone: resolve_env does NOT read os.environ (env layer is later).
    import os
    os.environ["QWEN3_ARTIFACT_ROOT"] = "/should/not/leak"
    try:
        env = lc.resolve_env([ASR_N2], registry)
        assert "/should/not/leak" not in env["EDGE_LLM_ASR_ENGINE_DIR"]
        assert "${QWEN3_ARTIFACT_ROOT}" in env["EDGE_LLM_ASR_ENGINE_DIR"]
    finally:
        os.environ.pop("QWEN3_ARTIFACT_ROOT", None)
