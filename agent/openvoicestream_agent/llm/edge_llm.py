"""edge-llm-chat-service backend (OpenAI-compatible + prefix-cache hooks)."""
from __future__ import annotations

from typing import Any, AsyncIterator

from ..session import Session
from .openai_compat import OpenAICompatBackend


class EdgeLLMBackend(OpenAICompatBackend):
    """Adds edge-llm's `save_system_prompt_kv_cache` / `prefix_cache` flags."""

    async def stream(  # type: ignore[override]
        self,
        messages: list[dict[str, str]],
        session: Session | None = None,
        **kw: Any,
    ) -> AsyncIterator[str]:
        if session is None or not session.cache_warmed:
            cache_flags: dict[str, Any] = {
                "save_system_prompt_kv_cache": True,
                "return_cache_metrics": True,
            }
        else:
            cache_flags = {
                "prefix_cache": True,
                "return_cache_metrics": True,
            }

        caller_extra = dict(kw.pop("extra_body", None) or {})
        cache_flags.update(caller_extra)
        kw["extra_body"] = cache_flags

        async for delta in super().stream(messages, **kw):
            yield delta

        if session is not None:
            session.cache_warmed = True
