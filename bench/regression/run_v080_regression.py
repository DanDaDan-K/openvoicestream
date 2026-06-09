#!/usr/bin/env python3
"""v0.8.0 migration regression runner — capture goldens / check a candidate.

Design ref: v080-regression-harness.md §6 (capture) + §7 (CI runner).

Two modes:

  CAPTURE (record goldens from a known-good v0.7.1 service):
    python3 run_v080_regression.py --capture \\
        --base-url http://localhost:8621 --container seeed-voice \\
        --device orin-nx --out goldens/v071/

  CHECK (compare a candidate service vs the goldens, emit pass/fail):
    python3 run_v080_regression.py --check \\
        --base-url http://localhost:8621 --container seeed-voice \\
        --device orin-nx --golden goldens/v071/

The CAPTURE path is fully exercised in this task. The CHECK path is coherent
but exercised later (against the v0.8.0 candidate). Dimensions whose backend is
NOT exposed by the live service are recorded as GAPS in meta.json — never faked.

Capture order (§7): build/ABI → ASR → TTS → N=2. perf.py + gate.py are run
separately by ``bench/perf/run_on_device.sh`` (§6 step 2) and their results
feed gate.py; this runner records the correctness goldens.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import requests  # noqa: E402

import test_asr_streaming_correctness as asr_mod  # noqa: E402
import test_build_abi_sanity as build_mod  # noqa: E402
import test_tts_correctness as tts_mod  # noqa: E402
import test_tts_n2_slotpool as n2_mod  # noqa: E402


# --- backend → engine family hints --------------------------------------------
ASR_BACKEND_FAMILY = {
    "paraformer_trt": "paraformer",
    "qwen3_asr_trt": "qwen3_asr",
    "qwen3_asr": "qwen3_asr",
}
TTS_BACKEND_FAMILY = {
    "matcha_trt": "matcha",
    "customvoice": "customvoice",
    "customvoice_trt": "customvoice",
    "moss_tts_nano": "moss",
}


def _probe(base_url: str) -> dict:
    asr = requests.get(f"{base_url}/asr/capabilities", timeout=30).json()
    tts = requests.get(f"{base_url}/tts/capabilities", timeout=30).json()
    return {"asr": asr, "tts": tts}


def _device_date(container: str | None) -> str:
    """Use the device clock (§6 step 4)."""
    try:
        if container:
            p = subprocess.run(["docker", "exec", container, "date", "-u",
                                "+%Y-%m-%dT%H:%M:%SZ"],
                               capture_output=True, text=True, timeout=30)
            if p.returncode == 0 and p.stdout.strip():
                return p.stdout.strip()
        p = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                           capture_output=True, text=True, timeout=30)
        return p.stdout.strip()
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _image_sha(container: str | None) -> str | None:
    if not container:
        return None
    try:
        p = subprocess.run(
            ["docker", "inspect", "--format", "{{.Image}} {{.Config.Image}}",
             container],
            capture_output=True, text=True, timeout=30)
        return p.stdout.strip() or None
    except Exception:
        return None


def _write(out_dir: Path, rel: str, payload: dict) -> None:
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"  wrote {path}")


def do_capture(args) -> int:
    out = Path(args.out)
    corpus = Path(args.corpus)
    base = args.base_url
    caps = _probe(base)
    asr_backend = caps["asr"].get("backend", "unknown")
    tts_backend = caps["tts"].get("backend", "unknown")
    print(f"== live backends: ASR={asr_backend} TTS={tts_backend} ==")

    gaps: list[str] = []
    captured: list[str] = []

    # (1) build / ABI — fail-fast in CI; informational here.
    backend_hints = []
    if asr_backend in ASR_BACKEND_FAMILY:
        backend_hints.append(ASR_BACKEND_FAMILY[asr_backend])
    if tts_backend in TTS_BACKEND_FAMILY:
        backend_hints.append(TTS_BACKEND_FAMILY[tts_backend])
    if args.container:
        print("[1/4] build/ABI ...")
        build = build_mod.capture(args.container, backend_hints)
        _write(out, "build/engine_md5s.json", build)
        captured.append("build")
        gaps += [f"build: {g}" for g in build.get("gaps", [])]
    else:
        gaps.append("build: no --container, engine md5s + log scan skipped")

    # (2) ASR streaming correctness.
    print("[2/4] ASR streaming ...")
    try:
        asr_host = base.split("://", 1)[-1]
        asr = asr_mod.capture(asr_host, corpus, chunk_ms=args.chunk_ms)
        asr["asr_backend"] = asr_backend
        _write(out, "asr/streaming.json", asr)
        captured.append("asr_streaming")
        # The qwen3-asr engine invariants (R2/R3/R4/KV-overflow) are migration-
        # runtime only; record the gap explicitly.
        if asr_backend != "qwen3_asr_trt":
            gaps.append(
                f"asr: engine invariants R2/R3/R4/KV-overflow not observable on "
                f"'{asr_backend}' (qwen3-asr edgellm runtime is v0.8.0-only); "
                f"corpus transcript baseline captured instead")
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"asr: capture failed: {exc!r}")

    # (3) TTS correctness.
    print("[3/4] TTS correctness ...")
    try:
        tts = tts_mod.capture(base, corpus, roundtrip=not args.no_roundtrip)
        # Route the golden under the backend's own subdir.
        sub = {"customvoice": "customvoice", "customvoice_trt": "customvoice",
               "moss_tts_nano": "moss"}.get(tts_backend, "matcha")
        _write(out, f"tts/{sub}/correctness.json", tts)
        captured.append(f"tts/{sub}")
        if tts_backend not in ("customvoice", "customvoice_trt"):
            gaps.append(
                f"tts/customvoice: CustomVoice (zh/en 9-row prefix) not deployed "
                f"(live backend is '{tts_backend}'); no CustomVoice golden")
        if tts_backend != "moss_tts_nano":
            gaps.append(
                f"tts/moss: MOSS-TTS-Nano not deployed (live backend is "
                f"'{tts_backend}'); no MOSS golden")
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"tts: capture failed: {exc!r}")

    # (4) N=2 slot-pool.
    print("[4/4] N=2 slot-pool ...")
    try:
        n2 = n2_mod.capture(base, burst_rounds=args.rounds)
        n2["tts_backend"] = tts_backend
        _write(out, "tts/n2_slotpool.json", n2)
        captured.append("tts_n2_slotpool")
        if tts_backend == "moss_tts_nano":
            captured.append("moss in-process stress applicable (run stress_moss_tts_n2.py)")
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"n2: capture failed: {exc!r}")

    # meta.json (§6 step 4).
    meta = {
        "harness_version": "v080-regression-1",
        "captured_from": "v0.7.1 voice-engine (GOLDEN baseline)",
        "device": args.device,
        "client_host": args.device,
        "base_url": base,
        "container": args.container,
        "image_sha": _image_sha(args.container),
        "image_tag_expected": "prod-unified-v8",
        "asr_backend": asr_backend,
        "tts_backend": tts_backend,
        "date_device_utc": _device_date(args.container),
        "captured_dimensions": captured,
        "gaps": gaps,
    }
    _write(out, "meta.json", meta)

    print("\n== CAPTURE SUMMARY ==")
    print(f"captured: {captured}")
    print(f"gaps ({len(gaps)}):")
    for g in gaps:
        print(f"  - {g}")
    return 0


def _load(golden: Path, rel: str) -> dict | None:
    p = golden / rel
    if not p.exists():
        return None
    return json.loads(p.read_text())


def do_check(args) -> int:
    """Compare a candidate live service against the captured goldens."""
    golden = Path(args.golden)
    base = args.base_url
    corpus = Path(args.corpus)
    caps = _probe(base)
    asr_backend = caps["asr"].get("backend", "unknown")
    tts_backend = caps["tts"].get("backend", "unknown")

    all_ok = True
    results: list[tuple[str, bool, list[str]]] = []

    # build / ABI.
    if args.container:
        g = _load(golden, "build/engine_md5s.json")
        if g:
            hints = g.get("backend_hints", [])
            cand = build_mod.capture(args.container, hints)
            ok, notes = build_mod.compare(g, cand)
            results.append(("build", ok, notes))
            all_ok &= ok

    # ASR.
    g = _load(golden, "asr/streaming.json")
    if g:
        cand = asr_mod.capture(base.split("://", 1)[-1], corpus,
                               chunk_ms=args.chunk_ms)
        ok, notes = asr_mod.compare(g, cand)
        results.append(("asr_streaming", ok, notes))
        all_ok &= ok

    # TTS (try matcha/customvoice/moss subdirs).
    for sub in ("matcha", "customvoice", "moss"):
        g = _load(golden, f"tts/{sub}/correctness.json")
        if g:
            cand = tts_mod.capture(base, corpus, roundtrip=not args.no_roundtrip)
            ok, notes = tts_mod.compare(g, cand)
            results.append((f"tts/{sub}", ok, notes))
            all_ok &= ok
            break

    # N=2.
    g = _load(golden, "tts/n2_slotpool.json")
    if g:
        cand = n2_mod.capture(base, burst_rounds=args.rounds)
        ok, notes = n2_mod.compare(g, cand)
        results.append(("tts_n2_slotpool", ok, notes))
        all_ok &= ok

    print("\n== CHECK RESULTS ==")
    for name, ok, notes in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for n in notes:
            print(f"    {n}")
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'} "
          f"(candidate ASR={asr_backend} TTS={tts_backend})")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true",
                      help="record goldens from a known-good service")
    mode.add_argument("--check", action="store_true",
                      help="compare a candidate service vs goldens")
    ap.add_argument("--base-url", default="http://localhost:8621")
    ap.add_argument("--container", default=None,
                    help="docker container name for engine md5 + log scan")
    ap.add_argument("--device", default=None, help="client_host / fleet node name")
    ap.add_argument("--corpus", default="bench/perf/corpus")
    ap.add_argument("--out", default="bench/regression/goldens/v071/",
                    help="capture output dir")
    ap.add_argument("--golden", default="bench/regression/goldens/v071/",
                    help="golden dir for --check")
    ap.add_argument("--chunk-ms", type=int, default=250)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--no-roundtrip", action="store_true")
    args = ap.parse_args()
    return do_capture(args) if args.capture else do_check(args)


if __name__ == "__main__":
    sys.exit(main())
