"""Optional punctuation restoration (CT-Transformer via sherpa-onnx).

Opt-in, default-OFF, lazy-loaded. When disabled the model is never loaded and
the feature costs nothing (no import side effects, no memory). Used by the
``POST /punctuate`` endpoint and — when a stream is opened with
``?punctuate=true`` — by the streaming *finalize* path only (never on partials).

Stateless: pure text-in / text-out. The CT-Transformer tokenizer + 272727-token
vocab are embedded in the ONNX file; we drive it through sherpa-onnx's
``OfflinePunctuation`` so tokenization matches the upstream model exactly
(reproducing it by hand is fragile). CPU-only — backend/device independent, so
the same code path works on Jetson / RK / RPi.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Stable identifier surfaced in payloads / capabilities so consumers can detect
# a model swap. Bump if the underlying model file changes.
PUNCT_MODEL_NAME = "ct_transformer_zh_en_vocab272727_2024-04-12"

_HF_REPO = "csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"

_punct = None           # cached sherpa_onnx.OfflinePunctuation
_lock = threading.Lock()
_load_failed = False     # sticky: don't retry a hard load failure every request


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def punctuation_enabled() -> bool:
    """Global default for the feature, from the ``OVS_PUNCT`` env (default off).

    A per-connection ``?punctuate=`` query overrides this; see the stream
    handlers. Mirrors the ``OVS_VAD_BACKEND`` env-default + query-override
    convention.
    """
    return _truthy(os.environ.get("OVS_PUNCT", ""))


def _model_path() -> str:
    explicit = os.environ.get("OVS_PUNCT_MODEL")
    if explicit:
        return explicit
    base = os.environ.get("MODEL_DIR", "/opt/models")
    return os.path.join(base, "punctuation", "model.onnx")


def _hf_url() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{_HF_REPO}/resolve/main/model.onnx"


def _ensure_model(path: str) -> None:
    """Download the model to ``path`` on first use if missing (idempotent).

    Honors HF_ENDPOINT mirrors (most edge devices can't reach hf.co directly).
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    import shutil
    import subprocess

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = _hf_url()
    logger.info("Punctuation model missing; downloading %s -> %s", url, path)
    tmp = path + ".part"
    if shutil.which("curl"):
        subprocess.run(
            ["curl", "-fSL", "--retry", "3", "-o", tmp, url], check=True
        )
    else:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    os.replace(tmp, path)
    logger.info("Punctuation model ready (%d bytes).", os.path.getsize(path))


def _get_punct():
    """Return the cached OfflinePunctuation, loading it once on first use."""
    global _punct, _load_failed
    if _punct is not None:
        return _punct
    if _load_failed:
        return None
    with _lock:
        if _punct is not None:
            return _punct
        if _load_failed:
            return None
        try:
            import sherpa_onnx

            path = _model_path()
            _ensure_model(path)
            num_threads = int(os.environ.get("OVS_PUNCT_THREADS", "2"))
            config = sherpa_onnx.OfflinePunctuationConfig(
                model=sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=path,
                    num_threads=num_threads,
                    provider="cpu",
                    debug=False,
                ),
            )
            _punct = sherpa_onnx.OfflinePunctuation(config)
            logger.info("Punctuation model loaded (%s, threads=%d).", path, num_threads)
        except Exception:
            _load_failed = True
            logger.exception("Failed to load punctuation model; feature disabled.")
            return None
    return _punct


def preload() -> bool:
    """Eagerly load the model (call at startup only when the feature is on) so
    the first request doesn't pay download + init latency. Returns readiness.
    """
    return _get_punct() is not None


def add_punctuation(text: str) -> str:
    """Return ``text`` with restored punctuation, or the input unchanged if the
    model is unavailable or the text is empty. Never raises to callers.
    """
    if not text or not text.strip():
        return text
    punct = _get_punct()
    if punct is None:
        return text
    try:
        return punct.add_punctuation(text)
    except Exception:
        logger.exception("add_punctuation failed; returning original text.")
        return text
