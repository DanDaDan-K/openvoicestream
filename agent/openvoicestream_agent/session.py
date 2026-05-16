"""Per-conversation Session state.

INVARIANT: NO trimming, NO truncation, NO max_history. Edge-LLM's
prefix cache is the optimization; client-side history rewriting would
break it.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    sid: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    history: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    locale: str = "zh"
    cache_warmed: bool = False

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Return OpenAI-format messages with system prompt prepended.

        The full history is returned; trimming is forbidden by invariant.
        """
        return [{"role": "system", "content": system_prompt}, *self.history]
