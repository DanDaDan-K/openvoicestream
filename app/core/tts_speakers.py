"""TTS speaker registry.

Public APIs expose numeric ``speaker_id`` values. Each id resolves to either a
preset speaker supported by the backend, or a precomputed voice-clone embedding.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Literal


SpeakerType = Literal["preset", "embedding"]


@dataclass(frozen=True)
class SpeakerSpec:
    id: int
    type: SpeakerType
    speaker: str | None = None
    speaker_embedding_b64: str | None = None


_DEFAULT_SPEAKERS: dict[int, SpeakerSpec] = {
    0: SpeakerSpec(id=0, type="preset", speaker=""),
    # Qwen3-TTS highperf preset speakers validated on Orin NX.
    2301: SpeakerSpec(id=2301, type="preset", speaker="2301"),
    2302: SpeakerSpec(id=2302, type="preset", speaker="2302"),
}


def _spec_from_value(sid: int, value: Any) -> SpeakerSpec:
    if isinstance(value, dict):
        typ = str(value.get("type", "preset")).strip().lower()
        if typ == "embedding":
            emb = value.get("speaker_embedding_b64") or value.get("embedding_b64")
            if not emb:
                raise ValueError(f"speaker_id {sid} embedding entry missing speaker_embedding_b64")
            return SpeakerSpec(id=sid, type="embedding", speaker_embedding_b64=str(emb))
        if typ != "preset":
            raise ValueError(f"speaker_id {sid} has unsupported type {typ!r}")
        speaker = value.get("speaker", value.get("name", sid))
        return SpeakerSpec(id=sid, type="preset", speaker="" if speaker is None else str(speaker))
    return SpeakerSpec(id=sid, type="preset", speaker="" if value is None else str(value))


def _load_speaker_map() -> dict[int, SpeakerSpec]:
    mapping = dict(_DEFAULT_SPEAKERS)
    raw = os.environ.get("OVS_TTS_SPEAKERS_JSON")
    if raw:
        data: Any = json.loads(raw)
        if isinstance(data, dict):
            for key, value in data.items():
                sid = int(key)
                mapping[sid] = _spec_from_value(sid, value)
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                sid = int(item["id"])
                mapping[sid] = _spec_from_value(sid, item)
        else:
            raise ValueError("OVS_TTS_SPEAKERS_JSON must be a JSON object or list")
    return mapping


def speaker_spec_for_id(speaker_id: int | None) -> SpeakerSpec | None:
    if speaker_id is None:
        return None
    sid = int(speaker_id)
    mapping = _load_speaker_map()
    if sid in mapping:
        return mapping[sid]
    if os.environ.get("OVS_TTS_ALLOW_UNMAPPED_SPEAKER_ID", "1").lower() in ("0", "false", "no", "off"):
        raise ValueError(f"Unknown TTS speaker_id: {sid}")
    return SpeakerSpec(id=sid, type="preset", speaker=str(sid))


def speaker_kwargs_for_id(speaker_id: int | None) -> dict[str, object]:
    spec = speaker_spec_for_id(speaker_id)
    if spec is None:
        return {}
    if spec.type == "embedding":
        assert spec.speaker_embedding_b64 is not None
        return {
            "speaker_id": spec.id,
            "speaker_embedding": base64.b64decode(spec.speaker_embedding_b64),
        }
    return {"speaker_id": spec.id, "speaker": spec.speaker or ""}


def available_speakers() -> list[dict[str, int | str]]:
    speakers: list[dict[str, int | str]] = []
    for sid, spec in sorted(_load_speaker_map().items()):
        item: dict[str, int | str] = {"id": sid, "type": spec.type}
        if spec.type == "preset":
            item["speaker"] = spec.speaker or ""
        else:
            item["speaker_embedding_b64"] = "<configured>"
        speakers.append(item)
    return speakers
