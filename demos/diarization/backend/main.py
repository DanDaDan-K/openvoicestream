"""diarization demo backend — speaker-colored live captions on /asr/stream.

Endpoints:
    GET /healthz      liveness of the demo app itself
    GET /api/config   where SLV's WebSocket lives, for the browser to connect
    /                 static frontend (this demo)
    /common/*         shared frontend assets (ui.css / ui.js / mic-capture.js ...)

Environment:
    SLV_URL   SLV server base URL (default http://127.0.0.1:8621)
    PORT      listen port for direct execution (default 8704)

The browser talks to SLV's ``/asr/stream`` WebSocket DIRECTLY (WebSocket is
not subject to CORS); this backend only serves static files and tells the
frontend where SLV lives. Diarization is a per-connection opt-in: the
frontend appends ``?diarize=true`` (server/main.py, same convention as
``?punctuate=``), which makes every *final* event carry ``speaker`` /
``speaker_conf`` plus segment ``start``/``end``, and — after the client
sends the empty end-of-stream frame — a ``diarization_summary`` event with
globally relabeled segments. ``/api/config`` carries the query params as
``asr_query`` so the wiring is visible in one place.

When SLV_URL points at loopback, ``/api/config`` flags it so the frontend
substitutes ``location.hostname`` — a browser on another machine then
reaches the device instead of itself.

Run locally (module path contains a hyphen, so run the file directly):
    uv run python diarization/backend/main.py
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_HERE = Path(__file__).resolve().parent          # demos/diarization/backend
_APP_DIR = _HERE.parent                           # demos/diarization
_DEMOS_DIR = _APP_DIR.parent                      # demos/

DEFAULT_SLV_URL = "http://127.0.0.1:8621"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _ws_endpoint(slv_url: str) -> dict:
    """Derive the browser-facing WebSocket endpoint from SLV_URL.

    ``loopback=True`` tells the frontend to replace ``host`` with
    ``location.hostname`` so external browsers hit the device, not themselves.
    """
    u = urlsplit(slv_url)
    scheme = "wss" if u.scheme == "https" else "ws"
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    return {
        "scheme": scheme,
        "host": host,
        "port": port,
        "loopback": host in _LOOPBACK_HOSTS,
    }


def create_app(slv_url: str | None = None) -> FastAPI:
    slv_url = (slv_url or os.environ.get("SLV_URL") or DEFAULT_SLV_URL).rstrip("/")

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # No pooled resources (the browser connects to SLV directly); kept
        # for parity with the gallery app factory.
        yield

    app = FastAPI(title="slv-demo-diarization", docs_url=None, redoc_url=None,
                  lifespan=_lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "service": "slv-demo-diarization"}

    @app.get("/api/config")
    async def api_config() -> dict:
        return {
            "slv_url": slv_url,
            "ws": _ws_endpoint(slv_url),
            "asr_path": "/asr/stream",
            # Per-connection opt-in: appended to the /asr/stream query string.
            # Diarize implies speaker embedding server-side, so this single
            # flag is all the browser needs.
            "asr_query": {"diarize": "true"},
        }

    # Static frontend (mounted last so /api/* and /healthz win).
    common_frontend = _DEMOS_DIR / "common" / "frontend"
    if common_frontend.is_dir():  # pragma: no branch
        app.mount("/common", StaticFiles(directory=common_frontend), name="common")
    frontend = _APP_DIR / "frontend"
    if frontend.is_dir():  # pragma: no branch
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8704")))
