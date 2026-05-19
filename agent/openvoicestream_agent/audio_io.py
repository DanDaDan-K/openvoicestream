"""sounddevice-backed mic capture + speaker playback.

Single persistent InputStream + OutputStream; sounddevice callbacks run
on background threads and push into asyncio.Queue via run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncIterator

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError) as _sd_exc:  # pragma: no cover - hardware-dependent
    sd = None  # type: ignore[assignment]
    _SD_IMPORT_ERR: Exception | None = _sd_exc
else:
    _SD_IMPORT_ERR = None

logger = logging.getLogger(__name__)


class AudioIO:
    """Mic in / speaker out, backed by sounddevice."""

    def __init__(
        self,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        input_sr: int = 16000,
        output_sr: int = 24000,
        chunk_ms: int = 100,
    ) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.chunk_ms = chunk_ms
        self._chunk_frames = int(input_sr * chunk_ms / 1000)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._in_queue: asyncio.Queue[bytes] | None = None
        self._out_queue: asyncio.Queue[bytes] | None = None
        self._input_stream: "sd.RawInputStream | None" = None
        self._output_stream: "sd.RawOutputStream | None" = None
        self._playback_task: asyncio.Task | None = None
        self._playback_buffer = bytearray()
        self._playback_lock = threading.Lock()
        self._is_playing = False
        # When True, play() drops incoming TTS pcm instead of queueing it.
        # Set by stop_playback (barge-in / sleep / stop-intent) so that
        # already-buffered TTS frames SLV keeps streaming over the WS
        # don't resume audible playback after we've silenced the speaker.
        # Cleared by arm_for_next_turn() at the start of the next utterance.
        self._discard_playback = False

    @property
    def is_playing(self) -> bool:
        self._ensure_playback_buffer()
        with self._playback_lock:
            has_buffered_audio = bool(self._playback_buffer)
        return self._is_playing or has_buffered_audio

    # ── capture ─────────────────────────────────────────────────────

    async def start_capture(self) -> AsyncIterator[bytes]:
        if sd is None:  # pragma: no cover - hardware-dependent
            raise RuntimeError(
                f"sounddevice unavailable: {_SD_IMPORT_ERR!r}. Install libportaudio2."
            )
        self._loop = asyncio.get_running_loop()
        self._in_queue = asyncio.Queue(maxsize=64)

        def _cb(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                logger.debug("input status: %s", status)
            buf = bytes(indata)
            try:
                assert self._loop is not None and self._in_queue is not None
                # IMPORTANT: schedule _safe_put on the loop thread, not
                # put_nowait directly -- QueueFull would otherwise be
                # raised on the loop thread where this try/except cannot
                # catch it.
                self._loop.call_soon_threadsafe(self._safe_put, buf)
            except Exception as e:  # pragma: no cover
                logger.warning("mic cb error: %s", e)

        self._input_stream = sd.RawInputStream(
            samplerate=self.input_sr,
            blocksize=self._chunk_frames,
            device=self.input_device,
            channels=1,
            dtype="int16",
            callback=_cb,
        )
        self._input_stream.start()

        try:
            while True:
                chunk = await self._in_queue.get()
                yield chunk
        finally:
            self._stop_input_stream()

    def _safe_put(self, data: bytes) -> None:
        """Runs on the asyncio loop thread; drops the chunk if the queue is full."""
        if self._in_queue is None:
            return
        try:
            self._in_queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.warning("mic queue full -- dropping chunk")

    def _stop_input_stream(self) -> None:
        if self._input_stream is not None:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:  # pragma: no cover
                pass
            self._input_stream = None

    # ── playback ────────────────────────────────────────────────────

    def _ensure_playback_buffer(self) -> None:
        if not hasattr(self, "_playback_buffer"):
            self._playback_buffer = bytearray()
        if not hasattr(self, "_playback_lock"):
            self._playback_lock = threading.Lock()

    def _output_callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("output status: %s", status)
        self._ensure_playback_buffer()
        needed = len(outdata)
        with self._playback_lock:
            n = min(needed, len(self._playback_buffer))
            if n:
                outdata[:n] = self._playback_buffer[:n]
                del self._playback_buffer[:n]
            if n < needed:
                outdata[n:needed] = b"\x00" * (needed - n)

    def _ensure_output(self) -> None:
        if sd is None:  # pragma: no cover - hardware-dependent
            raise RuntimeError(
                f"sounddevice unavailable: {_SD_IMPORT_ERR!r}. Install libportaudio2."
            )
        if self._output_stream is not None:
            return
        self._ensure_playback_buffer()
        self._output_stream = sd.RawOutputStream(
            samplerate=self.output_sr,
            blocksize=max(1, int(self.output_sr * 0.02)),
            device=self.output_device,
            channels=1,
            dtype="int16",
            callback=self._output_callback,
        )
        self._output_stream.start()

    async def _playback_loop(self) -> None:
        assert self._out_queue is not None
        try:
            while True:
                pcm = await self._out_queue.get()
                if pcm is None:
                    continue
                # NB: do NOT toggle _is_playing here. SLV streams TTS chunks
                # with variable inter-frame timing, so the queue can be
                # transiently empty between two chunks of the same utterance.
                # If we flipped to False during those gaps, barge-in checks
                # (`if audio.is_playing`) would race and miss real interrupts.
                # is_playing is owned by BaseApp dispatch:
                #   first TTSAudio frame → play() sets True
                #   TTSDone               → mark_playback_done() sets False
                #   barge-in / shutdown   → stop_playback() sets False
                try:
                    if self._output_stream is not None:
                        await asyncio.to_thread(self._output_stream.write, pcm)
                except Exception as e:  # pragma: no cover
                    logger.warning("playback write error: %s", e)
        except asyncio.CancelledError:
            raise

    def mark_playback_done(self) -> None:
        """Called by BaseApp when SLV emits TTSDone.

        This marks the remote TTS stream done, but local PortAudio may still
        have buffered PCM to play. `is_playing` therefore stays true until the
        callback drains `_playback_buffer`; otherwise barge-in during audible
        tail audio is missed.
        """
        self._is_playing = False

    async def play(self, pcm: bytes) -> None:
        # After barge-in (or sleep / stop-intent), SLV may keep streaming
        # the rest of the in-flight TTS for several hundred ms. Drop those
        # so the speaker actually stays silent until the next user turn.
        if self._discard_playback:
            return
        self._ensure_output()
        self._ensure_playback_buffer()
        self._is_playing = True
        with self._playback_lock:
            self._playback_buffer.extend(pcm)

    def arm_for_next_turn(self) -> None:
        """Re-enable playback for the next turn after a barge-in / sleep /
        stop-intent.  Called from BaseApp when a new ASR final arrives."""
        self._discard_playback = False

    def set_output_sample_rate(self, sr: int) -> None:
        if sr == self.output_sr and self._output_stream is not None:
            return
        self.output_sr = sr
        # Reconfigure: drop existing stream, recreate lazily next play().
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:  # pragma: no cover
                pass
            self._output_stream = None
        if self._playback_task is not None:
            self._playback_task.cancel()
            self._playback_task = None
        self._out_queue = None
        self._ensure_playback_buffer()
        with self._playback_lock:
            self._playback_buffer.clear()

    async def stop_playback(self) -> None:
        """Drain queued audio (barge-in / sleep / stop-intent).

        Also arms `_discard_playback` so any TTS chunks SLV keeps streaming
        over the WS for the rest of the in-flight utterance are dropped on
        arrival instead of being re-queued by play().  arm_for_next_turn()
        clears the latch at the start of the next user turn.
        """
        self._ensure_playback_buffer()
        with self._playback_lock:
            self._playback_buffer.clear()
        self._is_playing = False
        self._discard_playback = True

    async def close(self) -> None:
        self._stop_input_stream()
        if self._playback_task is not None:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except (asyncio.CancelledError, Exception):
                pass
            self._playback_task = None
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:  # pragma: no cover
                pass
            self._output_stream = None


__all__ = ["AudioIO"]


# Helper to keep numpy import alive (some platforms need it loaded for
# sounddevice's CFFI bindings to find PortAudio's int16 path).
_ = np.zeros(1, dtype=np.int16)
