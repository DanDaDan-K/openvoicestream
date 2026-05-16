"""DialogueApp -- the canonical Phase-1 App: simple voice chat.

INVARIANT: tokens stream DIRECTLY to SLV. No client-side sentence batching.
"""
from __future__ import annotations

from openvoicestream_agent import BaseApp


class DialogueApp(BaseApp):
    async def on_user_utterance(self, text: str) -> None:
        if not text.strip():
            return
        self.session.add_user(text)
        chunks: list[str] = []
        async for token in self.llm.stream(
            self.session.messages(self.config.system_prompt),
            session=self.session,
        ):
            chunks.append(token)
            self.events.emit("assistant_token", token)
            await self.slv.send_text(token)
        await self.slv.flush_tts()
        self.session.add_assistant("".join(chunks))


__all__ = ["DialogueApp"]
