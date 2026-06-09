"""Build / ABI sanity — capture + compare against v0.7.1 goldens.

Design ref: v080-regression-harness.md §5. Captures:
  - MD5 of the deployed TRT engines / model artifacts the live backend uses
    (ASR thinker/audio-enc, TTS Talker/CodePredictor/Code2Wav, MOSS worker bin
    — whichever exist for the deployed backend)
  - docker-log error scan: ``error|crash|fail`` + CUDA regex (per
    stability_tts_n2_common.py:50-60), expected count 0

Engine discovery is backend-aware: ``:prod-unified-v8`` ships paraformer_trt +
matcha_trt, so it has paraformer/matcha engines, NOT qwen3-asr or MOSS. Missing
artifacts for a backend that is NOT deployed are recorded as GAPS, not failures.

This module shells out to ``docker exec``/``docker logs`` and is therefore
invoked by the runner via ``--container``; the md5/grep logic lives here so the
gate is a pure function of captured maps.
"""
from __future__ import annotations

import re
import subprocess

# CUDA / crash regex (stability_tts_n2_common.py:50-60).
ERROR_RE = re.compile(
    r"(error|crash|fail|cuda\s+error|illegal\s+memory|out\s+of\s+memory|"
    r"segmentation\s+fault|core\s+dumped|assert)",
    re.IGNORECASE,
)

# Candidate engine/artifact globs to md5 inside the container, by backend family.
ENGINE_GLOBS = {
    "paraformer": ["/opt/**/paraformer*/**/*.trt", "/opt/**/paraformer*/**/*.onnx",
                   "/opt/**/paraformer*/**/*.engine"],
    "matcha": ["/opt/**/matcha*/**/*.trt", "/opt/**/matcha*/**/*.onnx",
               "/opt/**/matcha*/**/*.engine"],
    "qwen3_asr": ["/opt/**/qwen3*asr*/**/*.engine", "/opt/**/thinker*.engine",
                  "/opt/**/audio_enc*.engine"],
    "customvoice": ["/opt/**/customvoice/**/*.engine", "/opt/**/talker*.engine",
                    "/opt/**/code_predictor*.engine", "/opt/**/code2wav*.engine"],
    "moss": ["/opt/**/moss*/**/*.engine", "/opt/**/moss_tts*worker*"],
}


def _docker(container: str, cmd: str) -> tuple[int, str]:
    p = subprocess.run(
        ["docker", "exec", container, "bash", "-lc", cmd],
        capture_output=True, text=True, timeout=120,
    )
    return p.returncode, (p.stdout + p.stderr)


def capture(container: str, backend_hints: list[str]) -> dict:
    """Capture engine md5s for the deployed backends + a docker-log error scan."""
    engines: dict[str, str] = {}
    gaps: list[str] = []
    # md5 every artifact matching the globs for the deployed backend families.
    for fam in backend_hints:
        globs = ENGINE_GLOBS.get(fam, [])
        found_any = False
        for g in globs:
            # shopt globstar for ** support; md5sum each match.
            rc, out = _docker(
                container,
                f"shopt -s globstar nullglob; for f in {g}; do "
                f"[ -f \"$f\" ] && md5sum \"$f\"; done",
            )
            for line in out.splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2 and len(parts[0]) == 32:
                    engines[parts[1].strip()] = parts[0]
                    found_any = True
        if not found_any:
            gaps.append(f"no engine artifacts found for backend family '{fam}'")

    # docker logs error scan.
    p = subprocess.run(
        ["docker", "logs", "--tail", "500", container],
        capture_output=True, text=True, timeout=60,
    )
    log_text = p.stdout + p.stderr
    matches = [ln for ln in log_text.splitlines() if ERROR_RE.search(ln)]

    print(f"[build] engines md5'd: {len(engines)}; log error-lines: {len(matches)}",
          flush=True)
    for path, h in engines.items():
        print(f"[build]   {h}  {path}", flush=True)
    for g in gaps:
        print(f"[build]   GAP: {g}", flush=True)

    return {
        "dimension": "build_abi",
        "container": container,
        "backend_hints": backend_hints,
        "engine_md5s": engines,
        "log_error_count": len(matches),
        "log_error_sample": matches[:10],
        "gaps": gaps,
    }


def compare(golden: dict, candidate: dict) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    if candidate.get("log_error_count", 0) > 0:
        notes.append(f"FAIL: {candidate['log_error_count']} docker-log error lines")
        ok = False
    g_eng = golden.get("engine_md5s", {})
    c_eng = candidate.get("engine_md5s", {})
    for path, h in g_eng.items():
        if path not in c_eng:
            notes.append(f"FAIL: engine missing in candidate: {path}")
            ok = False
        elif c_eng[path] != h:
            notes.append(f"FAIL: engine md5 drift {path}: "
                         f"{c_eng[path][:8]} != {h[:8]}")
            ok = False
    if ok:
        notes.append(f"PASS: {len(g_eng)} engine md5s match, 0 log errors")
    return ok, notes
