"""Config dataclass + YAML loader with ${VAR} env substitution."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("openvoicestream-agent requires PyYAML (uv add pyyaml)") from exc


_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _default_slv_config() -> dict[str, Any]:
    return {
        "asr_language": "zh",
        "tts_language": "zh",
        "tts_voice": "default",
        "tts_speed": 1.0,
        "sample_rate": 16000,
        "vad": "silero",
        "vad_silence_ms": 400,
        "multi_utterance": True,
    }


@dataclass
class Config:
    """Top-level agent config."""

    slv_url: str = "ws://localhost:8621/v2v/stream"
    slv_config: dict[str, Any] = field(default_factory=_default_slv_config)
    llm_backend: str = "edge_llm"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "qwen2.5-3b-instruct"
    system_prompt: str = "You are a helpful, concise voice assistant."
    audio_input_device: str | int | None = None
    audio_output_device: str | int | None = None
    audio_input_sample_rate: int = 16000
    audio_output_sample_rate: int = 24000
    log_level: str = "INFO"


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in strings."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    """Load YAML config, apply env substitution, return a Config."""
    p = Path(path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping; got {type(raw).__name__}")

    # SLV config sub-block: merge with defaults so users don't have to
    # restate every key.
    slv_cfg = _default_slv_config()
    slv_cfg.update(raw.get("slv_config", {}) or {})
    # Force the framework invariant: persistent WS across utterances.
    slv_cfg["multi_utterance"] = True

    fields = {k: v for k, v in raw.items() if k != "slv_config"}
    return Config(slv_config=slv_cfg, **fields)
