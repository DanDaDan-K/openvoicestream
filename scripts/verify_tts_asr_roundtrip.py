#!/usr/bin/env python3
"""HTTP TTS -> ASR round-trip verifier for OpenVoiceStream services.

The TTS and ASR endpoints may be served by the same process or by separate
local services. This is useful for profiles such as Kokoro TRT that intentionally
ship TTS-only but can still be quality-gated by a resident ASR service.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

DEFAULT_CASES = [
    {
        "name": "short_en",
        "text": "Hello, this is a TensorRT Kokoro test.",
        "language": "en",
        "min_similarity": 0.55,
    },
    {
        "name": "control_en",
        "text": "Please close the bedroom window and turn on the desk light.",
        "language": "en",
        "min_similarity": 0.55,
        "keywords": ["bedroom", "window", "desk", "light"],
    },
    {
        "name": "multi_clause_en",
        "text": (
            "Please close the bedroom window, turn on the desk light, and turn "
            "off the kitchen fan."
        ),
        "language": "en",
        "min_similarity": 0.5,
        "keywords": ["bedroom", "window", "light", "kitchen", "fan"],
    },
]


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    return text


def _lcs_similarity(expected: str, actual: str) -> float:
    a = _normalize(expected)
    b = _normalize(actual)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _request_json(url: str, timeout: float) -> dict[str, Any]:
    with OPENER.open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[bytes, dict[str, str]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(req, timeout=timeout) as resp:
        return resp.read(), dict(resp.headers)


def _post_multipart_file(url: str, field: str, path: Path, timeout: float) -> dict[str, Any]:
    boundary = f"openvoicestream-{time.time_ns()}"
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        url,
        data=header + path.read_bytes() + footer,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width": width,
        "frames": frames,
        "duration_s": round(frames / rate, 3) if rate else 0,
        "bytes": path.stat().st_size,
    }


def _stream_to_wav(stream_bytes: bytes, path: Path) -> dict[str, Any]:
    if len(stream_bytes) < 4:
        raise RuntimeError("stream response is missing the 4-byte sample-rate header")
    sample_rate = struct.unpack("<I", stream_bytes[:4])[0]
    pcm = stream_bytes[4:]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return _wav_info(path)


def _load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("cases JSON must be a list")
    return data


def _check_keywords(expected_keywords: list[str], actual: str) -> dict[str, Any]:
    actual_norm = actual.lower()
    missing = [kw for kw in expected_keywords if kw.lower() not in actual_norm]
    return {
        "expected": expected_keywords,
        "missing": missing,
        "passed": not missing,
    }


def _run_case(args: argparse.Namespace, case: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    name = str(case.get("name") or f"case_{time.time_ns()}")
    text = str(case["text"])
    language = str(case.get("language") or args.language)
    speaker_id = case.get("speaker_id", args.speaker_id)
    speed = case.get("speed", args.speed)
    min_similarity = float(case.get("min_similarity", args.min_similarity))

    payload: dict[str, Any] = {"text": text}
    if speaker_id is not None:
        payload["speaker_id"] = int(speaker_id)
        payload["sid"] = int(speaker_id)
    if speed is not None:
        payload["speed"] = float(speed)
    if language:
        payload["language"] = language

    tts_endpoint = "/tts/stream" if args.streaming else "/tts"
    t0 = time.time()
    audio_bytes, headers = _post_json(f"{args.tts_url.rstrip('/')}{tts_endpoint}", payload, args.timeout_sec)
    tts_wall_s = time.time() - t0

    wav_path = out_dir / f"{name}.wav"
    if args.streaming:
        wav = _stream_to_wav(audio_bytes, wav_path)
    else:
        wav_path.write_bytes(audio_bytes)
        wav = _wav_info(wav_path)

    asr_url = f"{args.asr_url.rstrip('/')}/asr?language={language}"
    t0 = time.time()
    asr = _post_multipart_file(asr_url, "file", wav_path, args.timeout_sec)
    asr_wall_s = time.time() - t0
    actual = str(asr.get("text", "")).strip()
    similarity = _lcs_similarity(text, actual)
    keyword_result = _check_keywords(list(case.get("keywords") or []), actual)
    passed = bool(actual) and similarity >= min_similarity and keyword_result["passed"]

    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "text": text,
        "asr_text": actual,
        "language": language,
        "similarity": round(similarity, 4),
        "min_similarity": min_similarity,
        "keywords": keyword_result,
        "wav": wav,
        "wav_path": str(wav_path),
        "tts": {
            "endpoint": tts_endpoint,
            "wall_s": round(tts_wall_s, 3),
            "rtf": headers.get("X-RTF"),
            "inference_time": headers.get("X-Inference-Time"),
        },
        "asr": {
            "backend": asr.get("backend"),
            "wall_s": round(asr_wall_s, 3),
            "raw": asr,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-url", default="http://127.0.0.1:8621")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8621")
    parser.add_argument("--cases-json", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/openvoicestream-tts-asr-roundtrip"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--speaker-id", type=int, default=52)
    parser.add_argument("--speed", type=float, default=0.85)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--streaming", action="store_true", help="call /tts/stream and wrap raw PCM as WAV for ASR")
    args = parser.parse_args()

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cases = _load_cases(args.cases_json)
        report = {
            "status": "PASS",
            "tts_url": args.tts_url.rstrip("/"),
            "asr_url": args.asr_url.rstrip("/"),
            "streaming": args.streaming,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "health": {
                "tts": _request_json(f"{args.tts_url.rstrip('/')}/health", args.timeout_sec),
                "asr": _request_json(f"{args.asr_url.rstrip('/')}/health", args.timeout_sec),
            },
            "cases": [],
        }
        if not report["health"]["tts"].get("tts"):
            raise RuntimeError(f"TTS service is not ready: {report['health']['tts']}")
        if not report["health"]["asr"].get("asr"):
            raise RuntimeError(f"ASR service is not ready: {report['health']['asr']}")

        for case in cases:
            result = _run_case(args, case, args.out_dir)
            report["cases"].append(result)
            if result["status"] != "PASS":
                report["status"] = "FAIL"

        report_path = args.report or (args.out_dir / "report.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 1
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"round-trip verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
