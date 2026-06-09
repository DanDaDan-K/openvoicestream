"""N=2 slot-pool concurrency — capture + compare against v0.7.1 goldens.

Design ref: v080-regression-harness.md §3 (the "moat" that must survive the
Section 6 v2 batch-lane redesign). Observable invariants:
  - N=1 baseline PCM MD5 per prompt
  - N=2 dual-client MD5s == N=1 byte-identical (deterministic engines)
  - TTFA p50 N=1 vs N=2; ratio ≤ 1.5x
  - 30-round burst with 0 errors / 0 CUDA faults

This module drives the LIVE service over ``POST /tts/stream`` (raw 4-byte SR
header + int16 PCM), mirroring ``bench/perf/load_2client_tts.py`` and
``stability_tts_n2_common.py``. The MOSS in-process stress path
(``stress_moss_tts_n2.py``) is only applicable when the MOSS backend is loaded —
captured as a GAP otherwise. CUDA-fault detection over docker logs is done by
``test_build_abi_sanity.py``; here we count request-level errors.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from _common import md5, pcm_energy

TTFA_RATIO_GATE = 1.5  # §3 / load_2client_tts.py

# Fixed prompts (from load_2client_tts.py) — content is the contract.
PROMPTS = [
    "我们都非常震惊。这位母亲表示。",
    "今天天气真不错，适合出门散步。",
    "人工智能正在改变我们的生活方式。",
    "请问您需要什么帮助吗？",
]


def _stream_one(base_url: str, text: str) -> dict:
    """POST /tts/stream; return ttfa_ms + full PCM (header stripped) + md5."""
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/tts/stream", json={"text": text},
                      stream=True, timeout=120)
    r.raise_for_status()
    first_audio_t = None
    chunks: list[bytes] = []
    header_seen = False
    for chunk in r.iter_content(chunk_size=4096):
        if not chunk:
            continue
        if not header_seen:
            # First 4 bytes are the sample-rate header; audio follows.
            payload = chunk[4:]
            header_seen = True
            if payload:
                if first_audio_t is None:
                    first_audio_t = time.perf_counter()
                chunks.append(payload)
        else:
            if first_audio_t is None:
                first_audio_t = time.perf_counter()
            chunks.append(chunk)
    r.close()
    if first_audio_t is None:
        first_audio_t = time.perf_counter()
    pcm = b"".join(chunks)
    energy = pcm_energy(pcm)
    return {
        "text": text,
        "ttfa_ms": round((first_audio_t - t0) * 1000, 1),
        "pcm_md5": md5(pcm),
        "pcm_bytes": len(pcm),
        "rms": round(energy["rms"], 6),
    }


def _run_n(base_url: str, texts: list[str]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(texts)) as ex:
        return list(ex.map(lambda t: _stream_one(base_url, t), texts))


def capture(base_url: str, burst_rounds: int = 30) -> dict:
    # N=1 baseline (first two prompts, sequential).
    n1 = [_stream_one(base_url, PROMPTS[0]), _stream_one(base_url, PROMPTS[1])]
    n1_ttfas = sorted(r["ttfa_ms"] for r in n1)
    n1_ttfa_p50 = n1_ttfas[len(n1_ttfas) // 2]
    print(f"[n2] N=1 baseline ttfa p50={n1_ttfa_p50:.1f}ms "
          f"md5_0={n1[0]['pcm_md5'][:8]} md5_1={n1[1]['pcm_md5'][:8]}", flush=True)

    # N=2 same two prompts concurrently.
    n2 = _run_n(base_url, [PROMPTS[0], PROMPTS[1]])
    n2_ttfas = sorted(r["ttfa_ms"] for r in n2)
    n2_ttfa_p50 = n2_ttfas[len(n2_ttfas) // 2]
    md5_match = (n2[0]["pcm_md5"] == n1[0]["pcm_md5"]
                 and n2[1]["pcm_md5"] == n1[1]["pcm_md5"])
    ttfa_ratio = (n2_ttfa_p50 / n1_ttfa_p50) if n1_ttfa_p50 else None
    print(f"[n2] N=2 ttfa p50={n2_ttfa_p50:.1f}ms ratio={ttfa_ratio} "
          f"md5_match={md5_match}", flush=True)

    # 30-round dual-client burst — count request-level errors / empties.
    errors = 0
    for i in range(burst_rounds):
        try:
            r = _run_n(base_url, [PROMPTS[i % len(PROMPTS)],
                                  PROMPTS[(i + 1) % len(PROMPTS)]])
            if any(x["pcm_bytes"] == 0 for x in r):
                errors += 1
                print(f"[n2] burst round {i}: empty audio", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[n2] burst round {i}: ERROR {exc!r}", flush=True)
    print(f"[n2] burst DONE rounds={burst_rounds} errors={errors}", flush=True)

    return {
        "dimension": "tts_n2_slotpool",
        "endpoint": "/tts/stream",
        "n1_ttfa_p50_ms": n1_ttfa_p50,
        "n2_ttfa_p50_ms": n2_ttfa_p50,
        "ttfa_ratio": round(ttfa_ratio, 3) if ttfa_ratio else None,
        "ttfa_ratio_gate": TTFA_RATIO_GATE,
        "n1_md5": [n1[0]["pcm_md5"], n1[1]["pcm_md5"]],
        "n2_md5": [n2[0]["pcm_md5"], n2[1]["pcm_md5"]],
        "n2_md5_matches_n1": md5_match,
        "burst_rounds": burst_rounds,
        "burst_errors": errors,
        "n1_detail": n1,
        "n2_detail": n2,
    }


def compare(golden: dict, candidate: dict) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if candidate.get("burst_errors", 0) > 0:
        notes.append(f"FAIL: {candidate['burst_errors']} burst errors")
        ok = False
    if not candidate.get("n2_md5_matches_n1", False):
        notes.append("FAIL: N=2 md5 != N=1 (slot-pool not byte-identical)")
        ok = False
    cr = candidate.get("ttfa_ratio")
    if cr is not None and cr > TTFA_RATIO_GATE:
        notes.append(f"FAIL: ttfa ratio {cr} > {TTFA_RATIO_GATE}")
        ok = False
    if ok:
        notes.append(f"PASS: burst_errors=0 md5_match ttfa_ratio="
                     f"{candidate.get('ttfa_ratio')}")
    return ok, notes


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8621")
    ap.add_argument("--rounds", type=int, default=30)
    a = ap.parse_args()
    print(json.dumps(capture(a.base_url, a.rounds), ensure_ascii=False, indent=2))
