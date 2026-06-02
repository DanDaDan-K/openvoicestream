#!/usr/bin/env python3
"""Build a dialogue-style RKLLM quantization dataset for Qwen3 ASR decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPTS = (
    "Human: Please transcribe the audio.\nAssistant: ",
    "Human: Provide the speech-to-text result.\nAssistant: ",
    "Human: Convert speech to text.\nAssistant: ",
    "Human: Transcribe in Chinese.\nAssistant: ",
    "Human: Transcribe in English.\nAssistant: ",
    "Human: Return only the transcript without explanation.\nAssistant: ",
)


EXTRA_TARGETS = (
    "好的，已经为您安排好了。",
    "你好，今天天气真不错。",
    "请问还有其他需要帮助的吗？",
    "明天上午十点的会议已经提醒您。",
    "音量已经调到合适的大小。",
    "Hello, how can I help you today?",
    "Your order has been confirmed.",
    "The timer is set for ten minutes.",
    "Lights in the living room are now off.",
    "Please say that again, I did not catch it.",
)


def _load_targets(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for entry in data.get("files", []):
        text = (entry.get("eval_transcript") or entry.get("transcript") or "").strip()
        if text:
            targets.append(text)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    targets = _load_targets(args.manifest)
    targets.extend(EXTRA_TARGETS)

    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    for target_index, target in enumerate(targets):
        for offset in range(args.repeat):
            prompt = PROMPTS[(target_index + offset) % len(PROMPTS)]
            key = (prompt, target)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"input": prompt, "target": target})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
