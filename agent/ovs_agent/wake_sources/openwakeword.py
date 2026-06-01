"""OpenWakeWordSource — local keyword spotter as an ovs-agent WakeSource.

Reads PCM chunks from a TappedAudioIO capture tap, accumulates 80ms
windows (1280 samples @ 16k), feeds them to openwakeword, and fires
``app.wake(source="openwakeword")`` whenever any model score crosses
the threshold (subject to cooldown).

Inputs:
  * AudioIO writes mono int16 PCM at 16 kHz (AudioIO hardcodes channels=1
    and the sounddevice layer asks PortAudio to down-mix the reSpeaker
    XVF3800 multi-channel device on our behalf).
  * The model expects 80ms frames at 16 kHz (1280 samples).
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

try:
    from openwakeword import Model as _OWWModel
except Exception as _exc:  # pragma: no cover - import-time check
    _OWWModel = None  # type: ignore[assignment]
    _OWW_IMPORT_ERR: Exception | None = _exc
else:
    _OWW_IMPORT_ERR = None

from ovs_agent.wake_source import WakeSource

logger = logging.getLogger(__name__)


class OpenWakeWordSource(WakeSource):
    name = "openwakeword"

    # 80ms @ 16k = 1280 samples; openwakeword's documented chunk size.
    CHUNK_SAMPLES = 1280

    def __init__(
        self,
        app,
        model_name: str = "hey jarvis",
        threshold: float = 0.5,
        cooldown_s: float = 2.0,
        vad_threshold: float = 0.0,
    ) -> None:
        super().__init__(app)
        self._model_name = model_name
        self._threshold = float(threshold)
        self._cooldown_s = float(cooldown_s)
        self._vad_threshold = float(vad_threshold)
        self._task: asyncio.Task | None = None
        self._last_wake_ts: float = 0.0
        self._buffer = np.zeros(0, dtype=np.int16)
        self._model = None  # lazily constructed in setup()

    # ── plugin lifecycle ───────────────────────────────────────────
    def setup(self) -> bool:
        if _OWWModel is None:
            logger.error(
                "openwakeword not importable: %r; wake source disabled",
                _OWW_IMPORT_ERR,
            )
            return False
        model_id = self._model_name.replace(" ", "_") + "_v0.1"
        try:
            self._model = _OWWModel(
                wakeword_models=[model_id],
                inference_framework="onnx",
                vad_threshold=self._vad_threshold,
            )
        except Exception:
            # Fall back to bare model name (openwakeword's loader also
            # accepts the path-less alias for bundled models).
            try:
                self._model = _OWWModel(
                    wakeword_models=[self._model_name],
                    inference_framework="onnx",
                    vad_threshold=self._vad_threshold,
                )
            except Exception:
                logger.exception(
                    "openwakeword model load failed (model=%r); disabling",
                    self._model_name,
                )
                return False
        logger.info(
            "OpenWakeWordSource: model=%s threshold=%.2f cooldown=%.1fs",
            self._model_name, self._threshold, self._cooldown_s,
        )
        return True

    async def start(self) -> None:
        await super().start()
        self._task = asyncio.create_task(self._loop(), name="openwakeword")

    async def stop(self) -> None:
        await super().stop()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ── inference loop ─────────────────────────────────────────────
    async def _loop(self) -> None:
        # AudioIO opens its streams lazily; wait a beat so start_capture
        # has had a chance to allocate the queue + start the input stream.
        await asyncio.sleep(0.5)
        try:
            tap = await self.app.audio.start_capture_tap()
        except AttributeError:
            logger.error(
                "audio object %r is not a TappedAudioIO; wake source idle",
                type(self.app.audio).__name__,
            )
            return

        logger.info("OpenWakeWordSource: loop entered, waiting for audio")
        try:
            while True:
                chunk = await tap.get()
                arr = np.frombuffer(chunk, dtype=np.int16)
                if arr.size == 0:
                    continue
                self._buffer = np.concatenate([self._buffer, arr])
                while len(self._buffer) >= self.CHUNK_SAMPLES:
                    window = self._buffer[: self.CHUNK_SAMPLES]
                    self._buffer = self._buffer[self.CHUNK_SAMPLES :]
                    try:
                        scores = self._model.predict(window)
                    except Exception:
                        logger.exception("openwakeword.predict raised")
                        continue
                    if not scores:
                        continue
                    triggered: str | None = None
                    top_score = 0.0
                    for wname, score in scores.items():
                        if score > top_score:
                            top_score = float(score)
                        if score > self._threshold and self._cooldown_ok():
                            triggered = wname
                            break
                    if triggered is not None:
                        logger.info(
                            "WAKE detected: model=%s score=%.3f",
                            triggered, top_score,
                        )
                        self._last_wake_ts = time.monotonic()
                        # Discard stale samples so the next wake doesn't
                        # immediately re-trigger on the trailing audio.
                        self._buffer = np.zeros(0, dtype=np.int16)
                        try:
                            await self.app.wake(source=self.name)
                        except Exception:
                            logger.exception("app.wake() failed")
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("OpenWakeWordSource loop crashed")

    def _cooldown_ok(self) -> bool:
        return (time.monotonic() - self._last_wake_ts) > self._cooldown_s


__all__ = ["OpenWakeWordSource"]
