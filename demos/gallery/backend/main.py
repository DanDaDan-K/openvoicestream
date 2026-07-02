"""Gallery portal backend.

Endpoints:
    GET  /healthz       liveness of the gallery app itself
    GET  /api/status    aggregated SLV health + backend status (+ kiosk flag)
    GET  /api/catalog   demo cards from registry.json annotated with SLV capabilities
    GET  /api/profiles  profiles from DEMO_PROFILES_DIR filtered by device platform
    POST /api/switch    validated proxy to SLV /admin/backend/reload

Environment:
    SLV_URL            SLV server base URL (default http://127.0.0.1:8621)
    SLV_ADMIN_KEY      forwarded as X-Admin-Key on admin calls (optional)
    DEMO_PROFILES_DIR  directory of profile JSONs offered in the switch panel
                       (default: <repo>/configs/profiles when run from the repo)
    DEMO_KIOSK         truthy => kiosk mode (frontend hides debug details)

Run locally:  uv run uvicorn gallery.backend.main:app --port 8700
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from common.backend.slv_proxy import SLVProxy, ProbeResult

_HERE = Path(__file__).resolve().parent          # demos/gallery/backend
_GALLERY_DIR = _HERE.parent                       # demos/gallery
_DEMOS_DIR = _GALLERY_DIR.parent                  # demos/
_REGISTRY_PATH = _DEMOS_DIR / "registry.json"
_DEFAULT_PROFILES_DIR = _DEMOS_DIR.parent / "configs" / "profiles"

_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY


# ── profile listing / platform filtering ────────────────────────────────────


def _platform_tokens(probe: ProbeResult) -> set[str]:
    """Derive platform tokens from SLV-reported backend names.

    Backend names follow ``<platform>.<engine>`` (e.g. ``jetson.trt_edge_llm``);
    the profile files follow ``<platform>-<combo>.json``. We keep this
    deliberately simple: collect the dotted prefixes; a profile matches when
    its filename prefix equals a token or starts with one (``rk`` matches
    ``rk3576``). No tokens derivable => no filtering.
    """
    tokens: set[str] = set()
    sources = []
    if probe.health:
        sources += [probe.health.get("asr_backend"), probe.health.get("tts_backend")]
    if probe.backend_status:
        for kind in ("asr", "tts"):
            entry = probe.backend_status.get(kind) or {}
            sources.append(entry.get("backend_name"))
    for name in sources:
        if isinstance(name, str) and "." in name:
            prefix = name.split(".", 1)[0].strip().lower()
            if prefix:
                tokens.add(prefix)
    return tokens


def _list_profiles(profiles_dir: Path, probe: ProbeResult) -> dict:
    if not profiles_dir.is_dir():
        return {"profiles": [], "filtered": False, "platforms": [],
                "error": f"profiles dir not found: {profiles_dir}"}

    tokens = _platform_tokens(probe)
    profiles = []
    for path in sorted(profiles_dir.glob("*.json")):
        prefix = path.stem.split("-", 1)[0].lower()
        if tokens and not any(prefix == t or prefix.startswith(t) for t in tokens):
            continue
        entry = {"name": path.stem, "file": path.name}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry["name"] = data.get("name", path.stem)
            entry["description"] = data.get("description", "")
            entry["asr_backend"] = data.get("asr_backend")
            entry["tts_backend"] = data.get("tts_backend")
        except Exception as exc:  # noqa: BLE001 — a broken profile shouldn't hide the rest
            entry["error"] = f"unreadable: {type(exc).__name__}"
        profiles.append(entry)
    return {"profiles": profiles, "filtered": bool(tokens), "platforms": sorted(tokens)}


def _allowed_profile_names(profiles_dir: Path) -> set[str]:
    """Names accepted by /api/switch: every profile JSON in the directory,
    by logical name and by filename stem (SLV resolves both)."""
    allowed: set[str] = set()
    if not profiles_dir.is_dir():
        return allowed
    for path in profiles_dir.glob("*.json"):
        allowed.add(path.stem)
        try:
            name = json.loads(path.read_text(encoding="utf-8")).get("name")
            if isinstance(name, str) and name:
                allowed.add(name)
        except Exception:  # noqa: BLE001
            pass
    return allowed


# ── catalog ─────────────────────────────────────────────────────────────────


def _load_registry(registry_path: Path) -> list[dict]:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return list(data.get("demos", []))


def _check_need(need: str, probe: ProbeResult) -> tuple[Optional[bool], str]:
    """Return (satisfied, reason_key). satisfied=None => unknown (SLV down)."""
    health = probe.health or {}
    if need in ("asr", "tts"):
        if probe.health is None:
            return None, "slv_unreachable"
        return bool(health.get(need)), f"{need}_not_ready"

    if "." in need:
        kind, cap = need.split(".", 1)
        if probe.health is None:
            return None, "slv_unreachable"
        caps = health.get(f"{kind}_capabilities") or []
        if kind == "tts" and cap == "voice_clone":
            tc = probe.tts_capabilities or {}
            if tc.get("supports_voice_cloning") or cap in caps:
                return True, ""
            return False, "no_voice_clone"
        return cap in caps, f"missing_{kind}_{cap}"

    return None, "unknown_need"


_REASON_TEXT = {
    "slv_unreachable": {"zh": "语音服务不可达", "en": "voice server unreachable"},
    "asr_not_ready": {"zh": "ASR 后端未就绪", "en": "ASR backend not ready"},
    "tts_not_ready": {"zh": "TTS 后端未就绪", "en": "TTS backend not ready"},
    "no_voice_clone": {"zh": "当前 TTS 引擎不支持声音克隆", "en": "current TTS engine has no voice cloning"},
}


def _reason_text(key: str) -> dict:
    if key in _REASON_TEXT:
        return _REASON_TEXT[key]
    if key.startswith("missing_"):
        _, kind, cap = key.split("_", 2)
        return {
            "zh": f"当前 {kind.upper()} 引擎缺少能力：{cap}",
            "en": f"current {kind.upper()} engine lacks capability: {cap}",
        }
    return {"zh": "设备能力未知", "en": "device capability unknown"}


def _annotate_card(demo: dict, probe: ProbeResult) -> dict:
    card = dict(demo)
    needs = demo.get("needs") or []
    unmet_reason: Optional[str] = None
    unknown = False
    for need in needs:
        ok, reason = _check_need(need, probe)
        if ok is False:
            unmet_reason = reason
            break
        if ok is None:
            unknown = True

    if unmet_reason:
        card["state"] = "unsupported"
        card["reason"] = _reason_text(unmet_reason)
    elif demo.get("status") == "coming-soon":
        card["state"] = "coming-soon"
        card["reason"] = {"zh": "即将上线", "en": "coming soon"}
    elif unknown:
        card["state"] = "unknown"
        card["reason"] = _reason_text("slv_unreachable")
    else:
        card["state"] = "available"
        card["reason"] = {"zh": "", "en": ""}
    card["available"] = card["state"] == "available"
    return card


# ── app factory ──────────────────────────────────────────────────────────────


class SwitchRequest(BaseModel):
    kind: Literal["tts", "asr"]
    profile: str
    drain_timeout_s: Optional[float] = None


def create_app(
    proxy: SLVProxy | None = None,
    registry_path: Path | None = None,
    profiles_dir: Path | None = None,
    kiosk: bool | None = None,
) -> FastAPI:
    slv = proxy or SLVProxy()

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        await slv.aclose()

    app = FastAPI(title="slv-demo-gallery", docs_url=None, redoc_url=None,
                  lifespan=_lifespan)
    registry_path = registry_path or _REGISTRY_PATH
    profiles_dir = Path(
        profiles_dir
        or os.environ.get("DEMO_PROFILES_DIR")
        or _DEFAULT_PROFILES_DIR
    )
    kiosk_mode = _env_truthy("DEMO_KIOSK") if kiosk is None else kiosk

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "service": "slv-demo-gallery", "kiosk": kiosk_mode}

    @app.get("/api/status")
    async def api_status() -> dict:
        probe = await slv.probe()
        return {
            "kiosk": kiosk_mode,
            "slv_url": slv.base_url,
            "degraded": bool(probe.errors),
            "slv": probe.to_dict(),
        }

    @app.get("/api/catalog")
    async def api_catalog() -> dict:
        try:
            demos = _load_registry(registry_path)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {"error": "registry_unreadable", "message": str(exc)}, status_code=500
            )
        probe = await slv.probe()
        return {
            "kiosk": kiosk_mode,
            "slv_reachable": probe.reachable,
            "degraded": bool(probe.errors),
            "errors": probe.errors,
            "demos": [_annotate_card(d, probe) for d in demos],
        }

    @app.get("/api/profiles")
    async def api_profiles() -> dict:
        probe = await slv.probe()
        result = _list_profiles(profiles_dir, probe)
        result["slv_reachable"] = probe.reachable
        return result

    @app.post("/api/switch")
    async def api_switch(req: SwitchRequest):
        allowed = _allowed_profile_names(profiles_dir)
        if req.profile not in allowed:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "profile_not_allowed",
                    "profile": req.profile,
                    "hint": "profile must be one of GET /api/profiles",
                },
            )
        try:
            resp = await slv.reload_backend(
                req.kind, req.profile, drain_timeout_s=req.drain_timeout_s
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error": "slv_unreachable", "message": str(exc)},
            ) from exc
        # Pass SLV's verdict (reloaded / rolled_back / 4xx detail) through as-is.
        try:
            body = resp.json()
        except ValueError:
            body = {"error": "invalid_slv_response", "text": resp.text[:500]}
        return JSONResponse(body, status_code=resp.status_code)

    # Static frontend (mounted last so /api/* wins).
    common_frontend = _DEMOS_DIR / "common" / "frontend"
    if common_frontend.is_dir():  # pragma: no branch
        app.mount("/common", StaticFiles(directory=common_frontend), name="common")
    frontend = _GALLERY_DIR / "frontend"
    if frontend.is_dir():  # pragma: no branch
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8700")))
