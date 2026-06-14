"""Tests for TextPromptEncoder against the REAL text_encoder_pe.onnx.

Skipped when the artifact is absent (CI without the 254MB-ish encoder still
passes). Asserts shape [1,16,512], active rows L2-normalised, pad rows zero,
and that the npy cache round-trips (second call hits the cache without re-running
the ONNX session).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ovs_agent.apps.voice_rebot_arm.perception.text_pe import (
    EMBED_DIM,
    TextPromptEncoder,
)

ENCODER = Path(
    "/Users/harvest/project/_scratch/yolo-onnx-probe/embin_out/text_encoder_pe.onnx"
)

pytestmark = pytest.mark.skipif(
    not ENCODER.exists(), reason=f"text encoder onnx absent: {ENCODER}"
)

VOCAB = ["box", "cardboard box", "carton", "yellow banana"]


def test_shape_and_norms(tmp_path):
    enc = TextPromptEncoder(str(ENCODER), pad_slots=16, cache_dir=str(tmp_path))
    emb = enc.encode(VOCAB)
    assert emb.shape == (1, 16, EMBED_DIM)
    assert emb.dtype == np.float32
    assert enc.active_n == len(VOCAB)
    # active rows are unit-norm
    for i in range(len(VOCAB)):
        n = float(np.linalg.norm(emb[0, i]))
        assert abs(n - 1.0) < 1e-4, f"row {i} norm {n}"
    # pad rows are exactly zero
    for i in range(len(VOCAB), 16):
        assert np.all(emb[0, i] == 0.0), f"pad row {i} nonzero"


def test_cache_roundtrip(tmp_path):
    enc = TextPromptEncoder(str(ENCODER), pad_slots=16, cache_dir=str(tmp_path))
    emb1 = enc.encode(VOCAB)
    # a .npy cache file should now exist
    caches = list(Path(tmp_path).glob("text_pe_*.npy"))
    assert len(caches) == 1

    # second encoder, same vocab + cache_dir → must hit cache. Sabotage the ORT
    # session so a recompute would blow up; cache hit means it never runs.
    enc2 = TextPromptEncoder(str(ENCODER), pad_slots=16, cache_dir=str(tmp_path))

    def _boom():
        raise AssertionError("cache miss: ONNX session was created")

    enc2._ensure_session = _boom  # type: ignore[assignment]
    emb2 = enc2.encode(VOCAB)
    assert np.array_equal(emb1, emb2)


def test_cache_key_changes_with_vocab_order(tmp_path):
    enc = TextPromptEncoder(str(ENCODER), pad_slots=16, cache_dir=str(tmp_path))
    enc.encode(["box", "carton"])
    enc.encode(["carton", "box"])  # different order → different slot meaning
    caches = list(Path(tmp_path).glob("text_pe_*.npy"))
    assert len(caches) == 2  # distinct cache keys


def test_too_many_classes_raises(tmp_path):
    enc = TextPromptEncoder(str(ENCODER), pad_slots=2, cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="exceed pad_slots"):
        enc.encode(["a", "b", "c"])
