#!/usr/bin/env python3
"""Tiny same-origin gateway for the self-contained live-caption page.

Serves ``live_caption.html`` AND proxies ``/translate`` (+ ``/health``) to the
NLLB translator service. Because the browser then only ever talks to THIS
origin, there is no cross-origin call and no CORS requirement — the deployed
translator service is never touched.

The browser still streams mic audio directly to the SLV ASR WebSocket
(WebSockets are not subject to CORS); only the HTTP translate call is proxied.

Usage:
    uv run python serve_caption.py                 # :8080, NLLB on orin-nx
    uv run python serve_caption.py --port 9000 --nllb http://host:9001
    python3 serve_caption.py                       # stdlib only, no deps

Then open http://localhost:8080 and click 开始录音.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HTML_PATH = Path(__file__).with_name("live_caption.html")
# Proxy upstream calls without inheriting any system HTTP proxy (LAN/Tailscale).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Handler(BaseHTTPRequestHandler):
    nllb = "http://100.82.225.102:9001"  # overridden in main()

    def log_message(self, fmt, *args):  # quieter logs
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Permissive CORS too, so the page also works if opened from file://.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        if self.path in ("/", "/index.html", "/caption"):
            try:
                html = HTML_PATH.read_bytes()
            except FileNotFoundError:
                self._send(500, b"live_caption.html not found", "text/plain")
                return
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._proxy_get("/health")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/translate":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            self._proxy_post("/translate", body)
        else:
            self._send(404, b"not found", "text/plain")

    # ── upstream proxy ────────────────────────────────────────────
    def _proxy_post(self, path: str, body: bytes):
        req = urllib.request.Request(
            self.nllb.rstrip("/") + path, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        self._forward(req)

    def _proxy_get(self, path: str):
        req = urllib.request.Request(self.nllb.rstrip("/") + path, method="GET")
        self._forward(req)

    def _forward(self, req):
        try:
            with _OPENER.open(req, timeout=15) as r:
                self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read() or b"{}", "application/json")
        except Exception as e:  # noqa: BLE001
            self._send(502, json.dumps({"error": f"upstream: {e}"}).encode(), "application/json")


def main() -> None:
    ap = argparse.ArgumentParser(description="live-caption gateway (serve page + proxy translate)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--nllb", default="http://100.82.225.102:9001",
                    help="NLLB translator base URL to proxy /translate to")
    args = ap.parse_args()
    Handler.nllb = args.nllb
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"live-caption gateway → http://localhost:{args.port}")
    print(f"  translate proxied to {args.nllb}")
    print("  open the URL above, pick languages, click 开始录音")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
