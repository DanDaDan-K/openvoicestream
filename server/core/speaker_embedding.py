"""Optional speaker-embedding extraction (CAM++ / 3D-Speaker via sherpa-onnx).

Opt-in, default-OFF, lazy-loaded. When disabled the model is never loaded and
the feature costs nothing. Used by the ``POST /speaker/embedding`` endpoint and
— when a stream is opened with ``?speaker_embedding=true`` — by the streaming
*finalize* path only (once per utterance, never per-frame / per-partial).

OVS is **stateless**: it only emits the raw embedding vector + metadata. All
identity logic (registration DB, cosine matching, threshold, speaker_id/name)
lives on the consumer side. The embedding is a cross-service contract, so we
always ship ``embedding_model`` + ``dim`` + ``normalized`` so the consumer can
detect a model swap (which invalidates previously-registered vectors).

We drive the model through sherpa-onnx's ``SpeakerEmbeddingExtractor`` so the
kaldi-native-fbank front-end matches upstream exactly — enrollment and query
embeddings stay comparable only if produced by the identical extractor. CPU-only,
backend/device independent (same path on Jetson / RK / RPi).
"""

from __future__ import annotations

import base64
import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Stable identifier surfaced in payloads so consumers can detect a model swap.
SPEAKER_MODEL_NAME = "campplus_sv_zh_en_3dspeaker"

_HF_URL_DEFAULT = (
    "{endpoint}/csukuangfj/speaker-embedding-models/resolve/main/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)

_extractor = None       # cached sherpa_onnx.SpeakerEmbeddingExtractor
_dim = 0
_lock = threading.Lock()
_load_failed = False     # sticky: don't retry a hard load failure every request


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def speaker_embedding_enabled() -> bool:
    """Global default for the feature, from ``OVS_SPEAKER_EMB`` (default off).

    A per-connection ``?speaker_embedding=`` query overrides this. Mirrors the
    ``OVS_VAD_BACKEND`` env-default + query-override convention.
    """
    return _truthy(os.environ.get("OVS_SPEAKER_EMB", ""))


def _model_path() -> str:
    explicit = os.environ.get("OVS_SPEAKER_EMB_MODEL")
    if explicit:
        return explicit
    base = os.environ.get("MODEL_DIR", "/opt/models")
    return os.path.join(base, "speaker", "campplus.onnx")


def _hf_url() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return _HF_URL_DEFAULT.format(endpoint=endpoint)


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
    logger.info("Speaker model missing; downloading %s -> %s", url, path)
    tmp = path + ".part"
    if shutil.which("curl"):
        subprocess.run(["curl", "-fSL", "--retry", "3", "-o", tmp, url], check=True)
    else:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    os.replace(tmp, path)
    logger.info("Speaker model ready (%d bytes).", os.path.getsize(path))


def _get_extractor():
    """Return the cached extractor, loading it once on first use."""
    global _extractor, _dim, _load_failed
    if _extractor is not None:
        return _extractor
    if _load_failed:
        return None
    with _lock:
        if _extractor is not None:
            return _extractor
        if _load_failed:
            return None
        try:
            import sherpa_onnx

            path = _model_path()
            _ensure_model(path)
            num_threads = int(os.environ.get("OVS_SPEAKER_THREADS", "2"))
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=path,
                num_threads=num_threads,
                provider="cpu",
                debug=False,
            )
            ext = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            _extractor = ext
            _dim = int(ext.dim)
            logger.info(
                "Speaker model loaded (%s, dim=%d, threads=%d).",
                path, _dim, num_threads,
            )
        except Exception:
            _load_failed = True
            logger.exception("Failed to load speaker model; feature disabled.")
            return None
    return _extractor


def preload() -> bool:
    """Eagerly load the model (call at startup only when the feature is on).
    Returns readiness.
    """
    return _get_extractor() is not None


def embedding_dim() -> int:
    _get_extractor()
    return _dim


def compute_embedding(samples: np.ndarray, sample_rate: int):
    """Compute the speaker embedding for one utterance.

    ``samples`` must be mono float32 in [-1, 1]. Returns an L2-normalized
    float32 vector (np.ndarray, shape [dim]) or ``None`` if the model is
    unavailable / the audio is too short. Never raises to callers.
    """
    ext = _get_extractor()
    if ext is None:
        return None
    if samples is None or len(samples) == 0:
        return None
    try:
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        stream = ext.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()
        if not ext.is_ready(stream):
            # Too little audio for the front-end to produce features.
            return None
        emb = np.array(ext.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception:
        logger.exception("compute_embedding failed.")
        return None


def encode_embedding(emb: np.ndarray) -> str:
    """Little-endian float32 bytes, base64-encoded (consumer: np.frombuffer(..,'<f4'))."""
    return base64.b64encode(np.asarray(emb, dtype="<f4").tobytes()).decode("ascii")


def embedding_payload(emb: np.ndarray) -> dict:
    """Build the cross-service contract fields for a final payload."""
    return {
        "speaker_embedding": encode_embedding(emb),
        "embedding_model": SPEAKER_MODEL_NAME,
        "dim": int(len(emb)),
        "normalized": True,
    }


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw int16 little-endian PCM bytes to float32 in [-1, 1]."""
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0


_TARGET_SR = 16000


def _resample_linear(samples: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample mono float32 to 16 kHz with linear interpolation (dependency-free).

    Good enough for speaker embedding (CAM++ is robust to mild resampling
    artifacts); enrollment and query stay comparable as long as both go through
    this same path.
    """
    if src_sr == _TARGET_SR or len(samples) == 0:
        return samples
    n_out = int(round(len(samples) * _TARGET_SR / src_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def decode_audio_to_16k_mono(data: bytes, fallback_sr: int = _TARGET_SR) -> np.ndarray:
    """Decode an uploaded audio blob to mono float32 @ 16 kHz.

    Accepts a PCM16 WAV (RIFF, parsed via stdlib ``wave`` — no soundfile/scipy
    dependency so it works on every image) or, if the blob isn't a parseable
    WAV, treats it as raw little-endian int16 PCM at ``fallback_sr``.
    """
    import io
    import wave

    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            src_sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if sampwidth != 2:
            # Only PCM16 is supported via the stdlib path; fall back to raw.
            raise wave.Error("non-pcm16 wav")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        return _resample_linear(samples, src_sr)
    except (wave.Error, EOFError, ValueError):
        # Not a WAV we can parse — assume raw PCM16 at the declared rate.
        samples = pcm16_to_float32(data)
        return _resample_linear(samples, fallback_sr)
