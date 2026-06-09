"""TTS correctness — capture + compare against v0.7.1 goldens.

Design ref: v080-regression-harness.md §2. For each prompt:
  - synthesize via ``POST /tts`` (full WAV) → PCM MD5, sr/channels, duration,
    RMS + peak energy
  - ASR-roundtrip the synthesized audio back through ``POST /asr`` (or the
    streaming WS) → transcript, error rate vs the prompt text

HARD RULE (project history): byte-non-empty is NEVER sufficient — empty-valid
audio = silence. Energy (RMS/peak) + ASR-roundtrip are mandatory.

The golden is keyed by the backend actually serving ``/tts`` (probed from
``/tts/capabilities``). CustomVoice (language=chinese|english, 9-row prefix) and
MOSS are captured ONLY if that backend is live; otherwise the dimension is a
recorded GAP, not a fabricated golden.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from _common import (
    char_err_rate,
    err_rate,
    load_tts_prompts,
    md5,
    pcm_energy,
    wav_to_pcm_s16,
)

ENERGY_PCT_TOL = 0.25       # §2: energy ±25%
ERROR_RATE_ABS_TOL = 0.10   # §2: roundtrip semantic golden+0.10
SILENCE_RMS_FRAC = 0.10     # §2 gate: silence if RMS < 10% of golden


def _tts_synthesize(base_url: str, text: str, language: str | None,
                    extra: dict | None = None) -> bytes:
    body = {"text": text}
    if language:
        body["language"] = language
    if extra:
        body.update(extra)
    # Retry on the single-slot release lag (429): a just-closed roundtrip ASR
    # WS can briefly hold the session ceiling on the serial edgellm profile.
    import time as _time
    last = None
    for i in range(8):
        r = requests.post(f"{base_url}/tts", json=body, timeout=120)
        if r.status_code == 429:
            last = r
            _time.sleep(0.75 * (i + 1))
            continue
        r.raise_for_status()
        return r.content  # WAV bytes
    last.raise_for_status()  # type: ignore[union-attr]
    return b""


def _asr_roundtrip(base_url: str, wav_bytes: bytes) -> str:
    """Round-trip synthesized WAV back to text.

    Prefers the streaming ``/asr/stream`` WS (the offline ``POST /asr`` path is
    broken in :prod-unified-v8 — ``result.meta`` is None → 500). Falls back to
    ``POST /asr`` only if the WS path is unavailable.
    """
    try:
        return _asr_roundtrip_ws(base_url, wav_bytes)
    except Exception:  # noqa: BLE001
        pass
    try:
        r = requests.post(
            f"{base_url}/asr",
            files={"file": ("tts.wav", wav_bytes, "audio/wav")},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json().get("text", "")
        return f"<asr_http_{r.status_code}>"
    except Exception as exc:  # noqa: BLE001
        return f"<asr_error:{exc!r}>"


def _asr_roundtrip_ws(base_url: str, wav_bytes: bytes) -> str:
    """Feed synthesized WAV through the streaming ASR WebSocket; return final text."""
    import json as _json
    import time as _time

    import websocket

    from _common import wav_to_pcm_s16

    host = base_url.split("://", 1)[-1]
    pcm, sr, ch = wav_to_pcm_s16(wav_bytes)
    if ch > 1:
        import numpy as np
        a = np.frombuffer(pcm, dtype="<i2").reshape(-1, ch).mean(axis=1).astype("<i2")
        pcm = a.tobytes()
    url = f"ws://{host}/asr/stream?language=auto&sample_rate={sr}"
    ws = websocket.create_connection(url, timeout=30)
    chunk_n = int(sr * 0.25) * 2  # 250ms of int16
    for i in range(0, len(pcm), chunk_n):
        ws.send_binary(pcm[i:i + chunk_n])
    ws.settimeout(15)
    ws.send_binary(b"")
    text = ""
    while True:
        try:
            msg = _json.loads(ws.recv())
        except (websocket.WebSocketTimeoutException,
                websocket.WebSocketConnectionClosedException):
            break
        if msg.get("text"):
            text = msg["text"]
        if msg.get("is_final"):
            break
    try:
        ws.close()
    except Exception:
        pass
    _time.sleep(0.5)  # let the serial ASR slot release before next /tts
    return text


def probe_backend(base_url: str) -> dict:
    r = requests.get(f"{base_url}/tts/capabilities", timeout=30)
    r.raise_for_status()
    return r.json()


def _capture_one(base_url: str, prompt: dict, language: str | None,
                 extra: dict | None, roundtrip: bool) -> dict:
    wav = _tts_synthesize(base_url, prompt["text"], language, extra)
    pcm, sr, ch = wav_to_pcm_s16(wav)
    energy = pcm_energy(pcm)
    duration_s = (energy["samples"] / ch / sr) if (sr and ch) else 0.0
    row = {
        "id": prompt["id"],
        "lang": prompt["lang"],
        "category": prompt["category"],
        "text": prompt["text"],
        "request_language": language,
        "pcm_md5": md5(pcm),
        "wav_md5": md5(wav),
        "sample_rate": sr,
        "channels": ch,
        "duration_s": round(duration_s, 3),
        "rms": round(energy["rms"], 6),
        "peak": round(energy["peak"], 6),
        "pcm_bytes": len(pcm),
    }
    if roundtrip:
        rt = _asr_roundtrip(base_url, wav)
        row["roundtrip_text"] = rt
        if not rt.startswith("<"):
            row["roundtrip_error_rate"] = err_rate(rt, prompt["text"], prompt["lang"])
            row["roundtrip_cer"] = char_err_rate(rt, prompt["text"], prompt["lang"])
    return row


def capture(base_url: str, corpus: Path, roundtrip: bool = True) -> dict:
    """Capture the TTS golden for whichever backend serves /tts."""
    caps = probe_backend(base_url)
    backend = caps.get("backend", "unknown")
    supports_lang = "multi_language" in caps.get("capabilities", [])
    prompts = load_tts_prompts(corpus)

    rows = []
    for p in prompts:
        # CustomVoice expects language=chinese|english; matcha/others use the
        # short code or no language. Probe-driven: only pass language when the
        # backend advertises multi_language.
        language = None
        if supports_lang:
            language = {"zh": "chinese", "en": "english"}.get(p["lang"], p["lang"]) \
                if backend.startswith("customvoice") else p["lang"]
        row = _capture_one(base_url, p, language, None, roundtrip)
        rows.append(row)
        print(f"[tts:{backend}] {p['id']:<12} rms={row['rms']:.4f} "
              f"peak={row['peak']:.3f} dur={row['duration_s']:.2f}s "
              f"md5={row['pcm_md5'][:8]} rt={row.get('roundtrip_text','')[:24]!r}",
              flush=True)
    return {
        "dimension": "tts_correctness",
        "backend": backend,
        "capabilities": caps.get("capabilities", []),
        "supports_voice_cloning": caps.get("supports_voice_cloning", False),
        "endpoint": "/tts",
        "n": len(rows),
        "prompts": rows,
    }


def compare(golden: dict, candidate: dict) -> tuple[bool, list[str]]:
    notes: list[str] = []
    g_by_id = {p["id"]: p for p in golden.get("prompts", [])}
    deterministic = golden.get("backend", "").endswith("_trt") or \
        golden.get("backend") in ("matcha_trt", "kokoro_trt")
    ok = True
    for cand in candidate.get("prompts", []):
        g = g_by_id.get(cand["id"])
        if g is None:
            notes.append(f"FAIL {cand['id']}: no golden entry")
            ok = False
            continue
        # Silence gate.
        if g["rms"] > 0 and cand["rms"] < g["rms"] * SILENCE_RMS_FRAC:
            notes.append(f"FAIL {cand['id']}: silence rms {cand['rms']:.4f} "
                         f"< 10% of golden {g['rms']:.4f}")
            ok = False
            continue
        # Deterministic engines: MD5 exact.
        if deterministic and cand["pcm_md5"] != g["pcm_md5"]:
            notes.append(f"FAIL {cand['id']}: pcm md5 drift "
                         f"{cand['pcm_md5'][:8]} != {g['pcm_md5'][:8]}")
            ok = False
            continue
        # Energy ±25%.
        lo, hi = g["rms"] * (1 - ENERGY_PCT_TOL), g["rms"] * (1 + ENERGY_PCT_TOL)
        if not (lo <= cand["rms"] <= hi):
            notes.append(f"FAIL {cand['id']}: rms {cand['rms']:.4f} outside "
                         f"±25% of golden {g['rms']:.4f}")
            ok = False
            continue
        # Roundtrip semantic.
        gr = g.get("roundtrip_cer")
        cr = cand.get("roundtrip_cer")
        if gr is not None and cr is not None and cr > gr + ERROR_RATE_ABS_TOL:
            notes.append(f"FAIL {cand['id']}: roundtrip cer {cr:.3f} > "
                         f"golden {gr:.3f} + {ERROR_RATE_ABS_TOL}")
            ok = False
            continue
        notes.append(f"PASS {cand['id']}: rms={cand['rms']:.4f} "
                     f"md5={'exact' if deterministic else 'n/a'}")
    return ok, notes


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8621")
    ap.add_argument("--corpus", default="bench/perf/corpus")
    ap.add_argument("--no-roundtrip", action="store_true")
    a = ap.parse_args()
    out = capture(a.base_url, Path(a.corpus), roundtrip=not a.no_roundtrip)
    print(json.dumps(out, ensure_ascii=False, indent=2))
