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
