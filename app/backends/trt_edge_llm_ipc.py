"""Common subprocess and I/O utilities for TRT-Edge-LLM backends.

Provides:
  - Path constants (override via env vars)
  - run_binary()  — one-shot subprocess invocation
  - write_safetensors() — numpy -> safetensors file (no PyPI dep needed)
  - audio_bytes_to_mel() — WAV bytes -> log-mel spectrogram (scipy-only)
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# GPU subprocess gate: serialise binary launches to avoid concurrent GPU init OOM
_gpu_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Paths — all overridable via environment variables
# ---------------------------------------------------------------------------

_EDGE_LLM_BASE = os.environ.get(
    "EDGE_LLM_BASE", os.path.expanduser("~/project/tensorrt-edge-llm")
)
_EDGE_LLM_BUILD = os.path.join(
    _EDGE_LLM_BASE,
    os.environ.get("EDGE_LLM_BUILD_DIR", "build_sm87"),
)

# Binaries
TTS_BINARY = os.environ.get(
    "EDGE_LLM_TTS_BIN",
    os.path.join(_EDGE_LLM_BUILD, "examples/omni/qwen3_tts_inference"),
)
ASR_BINARY = os.environ.get(
    "EDGE_LLM_ASR_BIN",
    os.path.join(_EDGE_LLM_BUILD, "examples/llm/llm_inference"),
)
PLUGIN_PATH = os.environ.get(
    "EDGELLM_PLUGIN_PATH",
    os.path.join(_EDGE_LLM_BUILD, "libNvInfer_edgellm_plugin.so"),
)

# TTS engine directories
TTS_TALKER_DIR = os.environ.get(
    "EDGE_LLM_TTS_TALKER_DIR",
    os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/talker"),
)
TTS_CODE2WAV_DIR = os.environ.get(
    "EDGE_LLM_TTS_CODE2WAV_DIR",
    os.path.expanduser(
        "~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav"
    ),
)
TTS_TOKENIZER_DIR = os.environ.get(
    "EDGE_LLM_TTS_TOKENIZER_DIR",
    os.path.expanduser("~/qwen3-tts-trt-edge-llm-export"),
)

# ASR engine directories
ASR_ENGINE_DIR = os.environ.get(
    "EDGE_LLM_ASR_ENGINE_DIR",
    os.path.expanduser("~/qwen3-asr-trt-edge-llm-export/engines/thinker"),
)
ASR_AUDIO_ENC_DIR = os.environ.get(
    "EDGE_LLM_ASR_AUDIO_ENC_DIR",
    os.path.expanduser(
        "~/qwen3-asr-trt-edge-llm-export/engines/audio_encoder"
    ),
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _build_env() -> dict:
    """Return a copy of os.environ with EDGELLM_PLUGIN_PATH set."""
    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = PLUGIN_PATH
    return env


# ---------------------------------------------------------------------------
# Binary runner
# ---------------------------------------------------------------------------


def run_binary(
    binary_path: str,
    args: list[str],
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a TRT-Edge-LLM binary and return the CompletedProcess.

    Raises RuntimeError on non-zero exit (unless ``check=False``).
    """
    cmd = [binary_path] + args
    logger.info("Running (acquiring GPU lock): %s", " ".join(cmd[:4]))
    with _gpu_lock:
        logger.info("GPU lock acquired, launching: %s", os.path.basename(binary_path))
        try:
            result = subprocess.run(
                cmd,
                env=_build_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"{os.path.basename(binary_path)} timed out after {timeout}s"
            ) from e
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{os.path.basename(binary_path)} timed out after {timeout}s"
        )

    if check and result.returncode != 0:
        stderr_snip = result.stderr[:1000] if result.stderr else "(empty)"
        raise RuntimeError(
            f"{os.path.basename(binary_path)} failed (exit={result.returncode}): "
            f"{stderr_snip}"
        )
    return result


# ---------------------------------------------------------------------------
# Safetensors writer (zero external deps)
# ---------------------------------------------------------------------------

_SAFETENSORS_DTYPE_MAP = {
    np.float16: "F16",
    np.float32: "F32",
    np.int32: "I32",
    np.int64: "I64",
    np.int8: "I8",
    np.uint8: "U8",
    np.bool_: "BOOL",
}


def write_safetensors(tensor: np.ndarray, name: str, path: str) -> None:
    """Write a single numpy array to a standard safetensors file.

    The tensor is written as-is (caller must cast to desired dtype first).
    """
    header = {
        name: {
            "dtype": _SAFETENSORS_DTYPE_MAP.get(
                tensor.dtype.type, str(tensor.dtype)
            ),
            "shape": list(tensor.shape),
            "data_offsets": [0, tensor.nbytes],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(tensor.tobytes())


# ---------------------------------------------------------------------------
# Mel-spectrogram computation (scipy + numpy, no librosa needed)
# ---------------------------------------------------------------------------

# Whisper / Qwen3 ASR constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
MEL_FLOOR = 1e-10


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank() -> np.ndarray:
    """Build Slaney-norm mel filterbank [n_mels, n_fft//2+1]."""
    n_freq = N_FFT // 2 + 1
    low_mel = _hz_to_mel(np.float64(FMIN))
    high_mel = _hz_to_mel(np.float64(FMAX))
    mel_points = np.linspace(low_mel, high_mel, N_MELS + 2, dtype=np.float64)
    hz_points = _mel_to_hz(mel_points)

    bin = np.floor((n_freq - 1) * hz_points / FMAX).astype(np.int32)
    bin = np.clip(bin, 0, n_freq - 1)

    fb = np.zeros((N_MELS, n_freq), dtype=np.float64)
    for m in range(1, N_MELS + 1):
        left = int(bin[m - 1])
        center = int(bin[m])
        right = int(bin[m + 1])
        if left != center:
            for i in range(left, center):
                fb[m - 1, i] = (i - left) / (center - left)
        if center != right:
            for i in range(center, right):
                fb[m - 1, i] = (right - i) / (right - center)

    # Slaney norm: normalize each filter to unit area
    widths = hz_points[2:] - hz_points[:-2]
    fb *= (2.0 / widths)[:, np.newaxis]
    return fb.astype(np.float32)


# Build once at module level (cache)
_MEL_FILTERBANK = _build_mel_filterbank()


def audio_bytes_to_mel(
    audio_bytes: bytes,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert WAV bytes to log-mel spectrogram.

    Returns float32 array of shape ``[1, 128, T]`` (batch, mel, time),
    using Whisper-compatible parameters (n_fft=400, hop=160, 128 mel bins).

    Uses only scipy + numpy (no librosa or soundfile dependency).
    """
    from scipy.io import wavfile
    from scipy import signal as scipy_signal

    # -- Read WAV --
    sr, audio = wavfile.read(io.BytesIO(audio_bytes))

    # Convert integer PCM to float32 [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        new_len = int(round(len(audio) * target_sr / sr))
        audio = scipy_signal.resample(audio, new_len).astype(np.float32)

    # -- STFT (Whisper-compatible: periodic Hann, centered, drop last frame) --
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic
    pad_len = N_FFT // 2
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="reflect")

    _, _, stft = scipy_signal.stft(
        audio_padded,
        fs=target_sr,
        window=window,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        boundary=None,
        padded=False,
    )
    stft = stft[:, :-1]  # drop last frame (Whisper convention)

    # -- Power spectrum --
    power = np.abs(stft) ** 2  # [n_freq, T]

    # -- Mel filterbank --
    with np.errstate(all='ignore'):
        mel_spec = _MEL_FILTERBANK @ power  # [128, T]

    # -- Log compression (Whisper-style) --
    mel_spec = np.nan_to_num(mel_spec, nan=0.0)
    mel_spec = np.maximum(mel_spec, MEL_FLOOR)
    log_mel = np.log10(mel_spec)
    log_mel = np.clip(log_mel, -4.0, None)
    log_mel = (log_mel + 4.0) / 4.0

    # Add batch dimension
    return np.expand_dims(log_mel, 0).astype(np.float32)  # [1, 128, T]


# ---------------------------------------------------------------------------
# Temp-file helpers
# ---------------------------------------------------------------------------


def write_temp_json(data: dict, suffix: str = ".json") -> str:
    """Write a JSON dict to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False
    )
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def write_temp_wav(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Write audio bytes to a temporary WAV file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, delete=False
    )
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name
