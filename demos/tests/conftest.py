from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

DEMOS_DIR = Path(__file__).resolve().parent.parent
if str(DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(DEMOS_DIR))

from common.backend.slv_proxy import SLVProxy  # noqa: E402
from gallery.backend.main import create_app  # noqa: E402
from tests.mock_slv import create_mock_slv, default_state  # noqa: E402


@pytest.fixture()
def mock_slv():
    """(app, state) of a controllable fake SLV."""
    return create_mock_slv(default_state())


@pytest.fixture()
def profiles_dir(tmp_path: Path) -> Path:
    """Tiny profiles dir with two platforms for filter tests."""
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "jetson-qwen3asr-moss-nx.json").write_text(
        '{"name": "jetson-qwen3asr-moss-nx", "description": "Qwen3 ASR + MOSS TTS", '
        '"asr_backend": "jetson.trt_edge_llm", "tts_backend": "jetson.moss_tts_nano"}'
    )
    (d / "jetson-kokoro-trt.json").write_text(
        '{"name": "jetson-kokoro-trt", "description": "Kokoro TTS", '
        '"tts_backend": "jetson.kokoro_trt"}'
    )
    (d / "rk3576-default.json").write_text(
        '{"name": "rk3576-default", "description": "RK3576 default", '
        '"asr_backend": "rk.paraformer", "tts_backend": "rk.kokoro"}'
    )
    return d


def gallery_client(mock_app, profiles_dir: Path, admin_key: str | None = None,
                   kiosk: bool | None = None) -> httpx.AsyncClient:
    """Gallery app wired to the mock SLV through in-process ASGI transports."""
    proxy = SLVProxy(
        base_url="http://mock-slv",
        admin_key=admin_key,
        transport=httpx.ASGITransport(app=mock_app),
    )
    app = create_app(proxy=proxy, profiles_dir=profiles_dir, kiosk=kiosk)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gallery"
    )
