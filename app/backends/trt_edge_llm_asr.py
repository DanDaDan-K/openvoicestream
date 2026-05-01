"""ASR backend via TRT-Edge-LLM C++ binary (llm_inference).

Audio is converted to a Whisper-compatible log-mel spectrogram in Python
(scipy + numpy, no librosa), saved as a safetensors file, and passed to the
LLM binary via ``--multimodalEngineDir`` for the audio encoder.

Supports: OFFLINE, MULTI_LANGUAGE
Streaming: planned (Phase 2, requires llm_stream binary).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Optional

import numpy as np

from asr_backend import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

from backends.trt_edge_llm_ipc import (
    ASR_BINARY,
    ASR_ENGINE_DIR,
    ASR_AUDIO_ENC_DIR,
    PLUGIN_PATH,
    audio_bytes_to_mel,
    run_binary,
    write_safetensors,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_GENERATE_LENGTH = int(
    os.environ.get("ASR_MAX_GENERATE_LENGTH", "200")
)
_DEFAULT_TEMPERATURE = float(os.environ.get("ASR_TEMPERATURE", "1.0"))
_DEFAULT_TOP_P = float(os.environ.get("ASR_TOP_P", "0.8"))
_DEFAULT_TOP_K = int(os.environ.get("ASR_TOP_K", "50"))


class TRTEdgeLLMASRBackend(ASRBackend):
    """ASR via TRT-Edge-LLM llm_inference subprocess."""

    def __init__(self):
        self._ready = False

    # -- ASRBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.MULTI_LANGUAGE, ASRCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        """Verify all required files exist."""
        required = [
            (ASR_BINARY, "ASR binary"),
            (PLUGIN_PATH, "TRT-Edge-LLM plugin"),
            (os.path.join(ASR_ENGINE_DIR, "config.json"), "LLM config"),
            (os.path.join(ASR_ENGINE_DIR, "llm.engine"), "LLM engine"),
            (os.path.join(
                ASR_AUDIO_ENC_DIR, "audio", "config.json"
            ), "audio encoder config"),
            (os.path.join(
                ASR_AUDIO_ENC_DIR, "audio", "audio_encoder.engine"
            ), "audio encoder engine"),
        ]
        missing = []
        for path, label in required:
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")
        if missing:
            raise FileNotFoundError(
                "ASR preload failed — missing:\n  " + "\n  ".join(missing)
            )

        logger.info(
            "ASR backend preload OK (binary=%s engine=%s audio_enc=%s)",
            ASR_BINARY,
            ASR_ENGINE_DIR,
            ASR_AUDIO_ENC_DIR,
        )
        self._ready = True

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe audio via subprocess.

        Workflow:
          1. Write incoming audio to a temp WAV file.
          2. Compute log-mel spectrogram (numpy+scipy).
          3. Save mel as FP16 safetensors.
          4. Build input JSON referencing the mel file.
          5. Run ``llm_inference --multimodalEngineDir ...``.
          6. Parse output JSON for transcribed text.
        """
        if not self._ready:
            raise RuntimeError("ASR backend not preloaded")

        with tempfile.TemporaryDirectory(
            prefix="trt_edgellm_asr_"
        ) as tmpdir:
            # -- 1. Compute mel spectrogram (with duration guard) --
            mel = audio_bytes_to_mel(audio_bytes)  # [1, 128, T] float32
            if mel.shape[2] > 6000:  # ~60 seconds at 16kHz 10ms hop
                raise ValueError(
                    f"Audio too long: {mel.shape[2]} frames (~{mel.shape[2]*0.01:.0f}s). "
                    "Max 6000 frames (60s). Split into smaller chunks."
                )

            # Convert to FP16 for TRT
            mel_fp16 = mel.astype(np.float16)

            mel_path = os.path.join(tmpdir, "mel.safetensors")
            write_safetensors(mel_fp16, "mel", mel_path)
            logger.info(
                "Mel computed: shape=%s size=%s -> %s",
                list(mel_fp16.shape),
                mel_fp16.nbytes,
                mel_path,
            )

            # -- 2. Build input JSON --
            input_data = {
                "requests": [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "audio",
                                        "audio": mel_path,
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "batch_size": 1,
                "temperature": _DEFAULT_TEMPERATURE,
                "top_p": _DEFAULT_TOP_P,
                "top_k": _DEFAULT_TOP_K,
                "max_generate_length": _DEFAULT_MAX_GENERATE_LENGTH,
                "apply_chat_template": True,
                "add_generation_prompt": True,
            }

            input_path = os.path.join(tmpdir, "input.json")
            with open(input_path, "w") as f:
                json.dump(input_data, f)

            output_path = os.path.join(tmpdir, "output.json")

            # -- 3. Run binary --
            cli_args = [
                "--engineDir",
                ASR_ENGINE_DIR,
                "--multimodalEngineDir",
                ASR_AUDIO_ENC_DIR,
                "--inputFile",
                input_path,
                "--outputFile",
                output_path,
            ]

            t0 = time.time()
            result = run_binary(ASR_BINARY, cli_args, timeout=60)
            elapsed = time.time() - t0

            # -- 4. Parse output — fail loudly on errors
            if result.returncode != 0 or not os.path.exists(output_path):
                raise RuntimeError(
                    f"ASR subprocess failed (exit={result.returncode}): "
                    f"stdout={result.stdout[-300:]}, stderr={result.stderr[-300:]}"
                )

            with open(output_path) as f:
                output_data = json.load(f)

            responses = output_data.get("responses", [])
            if not responses:
                raise RuntimeError(f"ASR produced no responses: {output_data}")

            r = responses[0]
            text = r.get("output_text", "")
            if text == "TensorRT Edge LLM cannot handle this request. Fails.":
                raise RuntimeError(
                    f"ASR inference failed (model returned error): {r}"
                )

            language_detected = None
            if text and len(text) > 9 and text[:9] == "language ":
                space = text.find(" ", 9)
                if space > 0:
                    language_detected = text[9:space]
                    text = text[space + 1 :].lstrip()

            meta = {
                "inference_time_s": round(elapsed, 3),
            }
            return TranscriptionResult(
                text=text, language=language_detected, **meta
            )

    def create_stream(self, language: str = "auto") -> ASRStream:
        """Streaming ASR is not yet implemented for the TRT-Edge-LLM backend.

        Phase 2 will use ``llm_stream --streamInterval N`` with per-token
        stdout parsing.
        """
        raise NotImplementedError(
            f"{self.name} does not support streaming yet; "
            "use llm_inference for offline transcription."
        )
