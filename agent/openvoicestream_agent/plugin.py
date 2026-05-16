"""Plugin base class for the OpenVoiceStream Agent.

Lifecycle:
  1. __init__(app) -- store reference to shared BaseApp context
  2. setup() -- check deps/hardware, return False to skip gracefully
  3. async start() -- run alongside the app event loop (no-op default)
  4. async stop() -- shutdown (called in reverse registration order)

Semantic hooks (observer broadcasts -- they do NOT route the
conversation; BaseApp.on_user_utterance is the single router):
  - async on_user_speech_start()
  - async on_user_partial(text: str)
  - async on_user_utterance(text: str)
  - async on_assistant_token(token: str)
  - async on_assistant_sentence(sentence: str)
  - async on_assistant_done()
  - async on_error(exc: BaseException)

All hooks default to no-op; override only what you need.
"""
from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app_base import BaseApp

logger = logging.getLogger(__name__)


class Plugin(ABC):
    name: str = "unnamed"

    def __init__(self, app: "BaseApp") -> None:
        self.app = app
        self._running = False

    def setup(self) -> bool:
        """Sync prerequisite check. Return False to skip the plugin."""
        return True

    async def start(self) -> None:
        """Optional async startup. Default: no-op."""
        self._running = True

    async def stop(self) -> None:
        """Optional async shutdown. Default: no-op."""
        self._running = False

    # ── observer hooks ─────────────────────────────────────────────

    async def on_user_speech_start(self) -> None:
        """VAD detected the user started talking. Default: no-op."""

    async def on_user_partial(self, text: str) -> None:
        """ASR partial transcript update. Default: no-op."""

    async def on_user_utterance(self, text: str) -> None:
        """ASR final transcript for one utterance. Default: no-op."""

    async def on_assistant_token(self, token: str) -> None:
        """One LLM streaming token. Default: no-op."""

    async def on_assistant_sentence(self, sentence: str) -> None:
        """SLV finished synthesizing one sentence. Default: no-op."""

    async def on_assistant_done(self) -> None:
        """SLV emitted tts_done. Default: no-op."""

    async def on_error(self, exc: BaseException) -> None:
        """Any V2V transport/protocol error. Default: no-op."""
