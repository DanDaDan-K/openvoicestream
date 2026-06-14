"""ArmDashboardPlugin — read-only web view of the arm + vision pipeline.

Serves a single static page plus a polling JSON API:

  GET /                 the dashboard page (plain html/js, no build step)
  GET /api/state        {arm, frame_seq, frame_meta, frame_history, events,
                         place_bounds, busy}
  GET /api/frame.jpg    latest annotated decision/idle frame
  GET /api/depth.jpg    matching depth colormap

Frame/event content comes from :mod:`dashboard_bus` (fed by GraspPlugin's
frame_sink tee + idle observer). Arm state is proxied server-side from the
existing observation server (FastAPI, :8775) so the browser never needs CORS.

Read-only by design — no control endpoints — so binding beyond loopback is
safe; default bind is 0.0.0.0 for LAN demo viewing (override with
OVS_ARM_DASHBOARD_BIND).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Optional

from ovs_agent.plugin import Plugin

logger = logging.getLogger(__name__)

_PAGE = Path(__file__).with_name("static_dashboard.html")


def _wav_bytes_to_pcm16_mono(data: bytes, target_sr: int = 16000) -> bytes:
    """Decode WAV bytes → mono int16 PCM at ``target_sr`` (linear resample).
    Used by the debug inject endpoint; mirrors tests/e2e/fake_audio."""
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
    elif sw == 4:
        arr = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    else:
        arr = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1).astype(np.int16)
    if sr != target_sr and arr.size:
        n_out = int(len(arr) * target_sr / sr)
        x0 = np.linspace(0, 1, len(arr), endpoint=False)
        x1 = np.linspace(0, 1, n_out, endpoint=False)
        arr = np.interp(x1, x0, arr.astype(np.float32)).astype(np.int16)
    return arr.tobytes()


class ArmDashboardPlugin(Plugin):
    name = "arm_dashboard"

    def __init__(self, app: Any, config: Optional[dict] = None) -> None:
        super().__init__(app)
        self.cfg = dict(config or {})
        self._runner = None
        self._site = None
        self._started = False

    async def start(self) -> None:
        if self.cfg.get("enabled", True) is False or self._started:
            return
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp not installed — arm_dashboard disabled")
            return
        self._started = True
        web_app = web.Application()
        web_app.router.add_get("/", self._handle_index)
        web_app.router.add_get("/api/state", self._api_state)
        web_app.router.add_get("/api/frame.jpg", self._api_frame)
        web_app.router.add_get("/api/depth.jpg", self._api_depth)
        # DEBUG-ONLY remote audio inject (mic-less e2e). Always registered, but
        # the handler refuses unless OVS_REBOT_DEBUG_INJECT=1 — so production is
        # inert until explicitly enabled (+ restart). Works in server-loop: the
        # injected PCM is forwarded to SLV by the normal mic pump.
        web_app.router.add_post("/api/control/inject_wav", self._api_inject_wav)
        self._runner = web.AppRunner(web_app)
        await self._runner.setup()
        bind = os.environ.get("OVS_ARM_DASHBOARD_BIND", "0.0.0.0").strip() or "0.0.0.0"
        port = int(self.cfg.get("port", 8776))
        self._site = web.TCPSite(self._runner, bind, port)
        await self._site.start()
        logger.info("arm_dashboard listening on http://%s:%d (read-only)", bind, port)
        await super().start()

    async def stop(self) -> None:
        await super().stop()
        if not self._started:
            return
        self._started = False
        try:
            if self._runner is not None:
                await self._runner.cleanup()
        except Exception:
            logger.debug("arm_dashboard cleanup failed", exc_info=True)
        self._runner = None
        self._site = None

    # ── handlers ─────────────────────────────────────────────────────
    async def _handle_index(self, request):  # noqa: ANN001
        from aiohttp import web

        if _PAGE.exists():
            return web.FileResponse(path=str(_PAGE))
        return web.Response(text="dashboard page missing", status=500)

    async def _api_inject_wav(self, request):  # noqa: ANN001
        """DEBUG-ONLY: POST a WAV body → fed into the mic capture queue as a
        spoken utterance, mic-less. In server-loop the normal mic pump forwards
        it to SLV → ASR → LLM → tool_call → arm. Gated behind
        OVS_REBOT_DEBUG_INJECT=1 (default OFF) because a remote caller could
        otherwise move the physical arm. Forces asr_eos after feeding so the
        SLV finalizes regardless of the VAD/endpoint config."""
        from aiohttp import web

        if os.environ.get("OVS_REBOT_DEBUG_INJECT") != "1":
            return web.json_response(
                {"ok": False, "error": "inject disabled; set "
                 "OVS_REBOT_DEBUG_INJECT=1 and restart the agent"},
                status=403,
            )
        data = await request.read()
        if not data:
            return web.json_response(
                {"ok": False, "error": "empty body; POST raw WAV bytes"}, status=400
            )
        audio = getattr(self.app, "audio", None)
        if audio is None or getattr(audio, "_in_queue", None) is None:
            return web.json_response(
                {"ok": False, "error": "mic capture not active"}, status=409
            )
        try:
            sr = int(getattr(audio, "input_sr", 16000))
            pcm = _wav_bytes_to_pcm16_mono(data, target_sr=sr)
        except Exception as e:  # noqa: BLE001
            return web.json_response(
                {"ok": False, "error": f"bad wav: {e}"}, status=400
            )
        logger.warning(
            "inject_wav: feeding %d PCM bytes (%.2fs @ %dHz) as a spoken utterance",
            len(pcm), len(pcm) / 2 / sr, sr,
        )
        # Wake → clear wake-tone suppression → feed audio → generous trailing
        # silence so the SLV's (silero) server VAD detects speech-end and
        # finalizes the utterance NATURALLY — exactly like a real spoken command.
        #
        # We deliberately do NOT force asr_eos: a forced EOS preempts the
        # streaming/offline-segment ASR before it produces a final for SHORT
        # utterances (observed 2026-06-14: short English inject → empty final,
        # while the offline /asr transcribes the very same audio cleanly). Real
        # voice never force-EOSes; letting the server VAD endpoint routes short
        # commands (<6s) through the offline-segment path, which transcribes
        # English fine. The THINKING watchdog recovers if the VAD never fires.
        try:
            await self.app.wake(source="inject_wav")
        except Exception:
            logger.debug("inject_wav: wake failed", exc_info=True)
        await asyncio.sleep(0.4)
        await audio.inject_pcm(pcm)
        await audio.inject_pcm(b"\x00\x00" * int(sr * 1.0))  # 1.0s trailing silence
        return web.json_response({"ok": True, "pcm_bytes": len(pcm), "sr": sr})

    async def _api_state(self, request):  # noqa: ANN001
        from aiohttp import web

        from .dashboard_bus import BUS

        state = BUS.snapshot()
        state["arm"] = await asyncio.to_thread(self._fetch_observation)
        state["busy"] = self._busy_motion()
        state["place_bounds"] = self._place_bounds()
        return web.json_response(state)

    async def _api_frame(self, request):  # noqa: ANN001
        return self._jpg_response((await self._bus()).latest_jpg())

    async def _api_depth(self, request):  # noqa: ANN001
        return self._jpg_response((await self._bus()).latest_depth_jpg())

    @staticmethod
    async def _bus():
        from .dashboard_bus import BUS

        return BUS

    @staticmethod
    def _jpg_response(data: Optional[bytes]):
        from aiohttp import web

        if not data:
            return web.Response(status=404, text="no frame yet")
        return web.Response(
            body=data, content_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    # ── data sources ─────────────────────────────────────────────────
    def _fetch_observation(self) -> Optional[dict]:
        """Server-side proxy to the observation server (loopback, no CORS)."""
        port = int(self.cfg.get("observation_port", 8775))
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/observation", timeout=1.5
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def _busy_motion(self) -> Optional[str]:
        for plugin in getattr(self.app, "plugins", []) or []:
            if plugin.__class__.__name__ == "GraspPlugin":
                try:
                    return plugin._busy_motion_name()  # noqa: SLF001
                except Exception:
                    return None
        return None

    def _place_bounds(self) -> Optional[list]:
        pb = self.cfg.get("place_bounds")
        if pb:
            try:
                vals = [float(v) for v in pb]
                if len(vals) == 4:
                    return vals
            except (TypeError, ValueError):
                pass
        return None
