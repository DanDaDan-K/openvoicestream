"""Shared helpers for the v0.8.0 regression harness.

Reuses normalization / error-rate / energy / MD5 logic from
``bench/perf/asr_stream_ws_bench.py`` and ``bench/perf/stats.py`` rather than
reinventing it. Audio loading uses the stdlib ``wave`` module so the harness
runs on a bare device python without ``soundfile`` — the perf corpus is already
16 kHz mono int16 PCM (see ``corpus/manifest.json`` audio_spec), so no resample
is needed for the streaming-ASR feed path.
"""
from __future__ import annotations

import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np

# --- text normalization (verbatim from asr_stream_ws_bench.py:17-53) ----------


def norm_zh(text: str) -> str:
    return re.sub(r"[\s，。、“”\"'（）()：:；;,.!?！？-]", "", text).lower()


def norm_en(text: str) -> list[str]:
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text).lower()
    return [w for w in text.split() if w]


def norm_en_chars(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def err_rate(ref: str, hyp: str, lang: str) -> float:
    if lang == "zh":
        r, h = norm_zh(ref), norm_zh(hyp)
        return edit_distance(r, h) / max(1, len(r))
    r, h = norm_en(ref), norm_en(hyp)
    return edit_distance(r, h) / max(1, len(r))


def char_err_rate(ref: str, hyp: str, lang: str) -> float:
    if lang == "zh":
        r, h = norm_zh(ref), norm_zh(hyp)
    else:
        r, h = norm_en_chars(ref), norm_en_chars(hyp)
    return edit_distance(r, h) / max(1, len(r))


# --- audio loading / energy ---------------------------------------------------


def load_wav_i16(path: Path) -> tuple[np.ndarray, int]:
    """Load a 16-bit PCM WAV as int16 mono samples + sample rate (stdlib only)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got sampwidth={sw}")
    audio = np.frombuffer(raw, dtype="<i2")
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return np.asarray(audio, dtype=np.int16), sr


def pcm_energy(pcm_s16: bytes) -> dict:
    """RMS + peak energy of raw int16 little-endian PCM, normalized to [-1, 1].

    A non-empty buffer is NEVER proof of speech (project history: empty-valid
    audio = silence). RMS/peak make silence detectable.
    """
    if not pcm_s16:
        return {"rms": 0.0, "peak": 0.0, "samples": 0}
    a = np.frombuffer(pcm_s16, dtype="<i2").astype(np.float32) / 32768.0
    return {
        "rms": float(np.sqrt(np.mean(a * a))),
        "peak": float(np.max(np.abs(a))),
        "samples": int(a.shape[0]),
    }


def wav_to_pcm_s16(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Extract raw PCM + (sr, channels) from an in-memory WAV (RIFF) blob."""
    import io

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        pcm = w.readframes(w.getnframes())
    return pcm, sr, ch


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# --- corpus -------------------------------------------------------------------


def load_manifest(corpus: Path) -> dict:
    return json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))


def load_tts_prompts(corpus: Path) -> list[dict]:
    return json.loads((corpus / "tts_prompts.json").read_text(encoding="utf-8"))["prompts"]
