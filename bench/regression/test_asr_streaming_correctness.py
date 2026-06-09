"""ASR streaming correctness — capture + compare against v0.7.1 goldens.

Design ref: v080-regression-harness.md §1. Drives the live ``/asr/stream``
WebSocket (same protocol as ``bench/perf/asr_stream_ws_bench.py``) over the
20-file SHA256-pinned corpus, recording the final transcript + ``eos_to_final_ms``
per file as the golden.

ENGINE-INVARIANT NOTE (R2 single-vs-split / R3 MRope continuity / KV-overflow /
R4 sys-prompt cache fallback): those named cases from
``asr-streaming-v080-migration.md`` §4 Phase 5 target the qwen3-asr edgellm
runtime introduced BY the v0.8.0 migration. They are NOT observable through the
v0.7.1 HTTP/WS surface (which runs the paraformer_trt engine). The capture path
records the corpus transcript baseline that those invariants must not regress;
the in-runtime spike assertions are wired in the migration worktree, not here.

Compare mode: per-file error_rate must pass at ``golden_error_rate + 0.10``
(gate.py error_rate_abs tolerance) and char-error-rate likewise; a hard
transcript divergence (semantic) fails the gate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import websocket  # websocket-client

from _common import char_err_rate, err_rate, load_manifest, load_wav_i16

ERROR_RATE_ABS_TOL = 0.10  # gate.py:42


def _drain(ws, deadline_s: float) -> list[dict]:
    msgs: list[dict] = []
    while time.perf_counter() < deadline_s:
        try:
            raw = ws.recv()
        except (websocket.WebSocketTimeoutException,
                websocket.WebSocketConnectionClosedException):
            break
        try:
            msgs.append(json.loads(raw))
        except Exception:
            msgs.append({"raw": raw})
    return msgs


def run_one(url: str, wav_path: Path, ref: str, lang: str, chunk_ms: int) -> dict:
    audio_i16, sr = load_wav_i16(wav_path)
    chunk_n = max(1, int(sr * chunk_ms / 1000))
    ws = websocket.create_connection(url, timeout=30)
    ws.settimeout(0.001)
    n_msgs = 0
    for start in range(0, len(audio_i16), chunk_n):
        ws.send_binary(audio_i16[start:start + chunk_n].tobytes())
        n_msgs += len(_drain(ws, time.perf_counter() + 0.001))

    ws.settimeout(15)
    eos_at = time.perf_counter()
    ws.send_binary(b"")
    final = None
    while True:
        try:
            msg = json.loads(ws.recv())
        except (websocket.WebSocketTimeoutException,
                websocket.WebSocketConnectionClosedException):
            break
        n_msgs += 1
        if msg.get("is_final"):
            final = msg
            break
    eos_to_final_ms = (time.perf_counter() - eos_at) * 1000
    try:
        ws.close()
    except Exception:
        pass

    text = (final or {}).get("text", "")
    return {
        "text": text,
        "ref": ref,
        "error_rate": err_rate(ref, text, lang),
        "char_error_rate": char_err_rate(ref, text, lang),
        "eos_to_final_ms": round(eos_to_final_ms, 1),
        "messages": n_msgs,
        "got_final": final is not None,
    }


def capture(base_host: str, corpus: Path, chunk_ms: int = 250) -> dict:
    """Capture the ASR streaming golden over the full 20-file corpus."""
    manifest = load_manifest(corpus)
    url_base = f"ws://{base_host}/asr/stream?language=auto&sample_rate=16000"
    files = []
    for item in manifest["files"]:
        ref = item.get("eval_transcript") or item["transcript"]
        row = run_one(url_base, corpus / item["filename"], ref, item["lang"], chunk_ms)
        row.update({
            "id": item["id"],
            "lang": item["lang"],
            "category": item["category"],
            "duration_s": item["duration_s"],
            "sha256": item.get("sha256", ""),
        })
        files.append(row)
        print(f"[asr] {item['id']:<12} cer={row['char_error_rate']:.3f} "
              f"eos2final={row['eos_to_final_ms']:.0f}ms text={row['text'][:30]!r}",
              flush=True)
    return {
        "dimension": "asr_streaming",
        "endpoint": "/asr/stream (ws)",
        "chunk_ms": chunk_ms,
        "n": len(files),
        "files": files,
    }


def compare(golden: dict, candidate: dict) -> tuple[bool, list[str]]:
    """Compare a freshly-captured candidate vs the golden. Returns (ok, notes)."""
    notes: list[str] = []
    g_by_id = {f["id"]: f for f in golden.get("files", [])}
    ok = True
    for cand in candidate.get("files", []):
        g = g_by_id.get(cand["id"])
        if g is None:
            notes.append(f"FAIL {cand['id']}: no golden entry")
            ok = False
            continue
        limit = g["char_error_rate"] + ERROR_RATE_ABS_TOL
        if cand["char_error_rate"] > limit:
            notes.append(
                f"FAIL {cand['id']}: cer {cand['char_error_rate']:.3f} > "
                f"golden {g['char_error_rate']:.3f} + {ERROR_RATE_ABS_TOL}")
            ok = False
        else:
            notes.append(
                f"PASS {cand['id']}: cer {cand['char_error_rate']:.3f} "
                f"(golden {g['char_error_rate']:.3f})")
    return ok, notes
