"""TTS backend via TRT-Edge-LLM C++ binary (qwen3_tts_inference).

Calls the binary per-request with temp-file I/O.
Supports: BASIC_TTS, MULTI_LANGUAGE
Audio output: WAV via Code2Wav (vocoder) engine.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Optional

from tts_backend import TTSBackend, TTSCapability

from backends.trt_edge_llm_ipc import (
    TTS_BINARY,
    TTS_TALKER_DIR,
    TTS_CODE2WAV_DIR,
    TTS_TOKENIZER_DIR,
    PLUGIN_PATH,
    run_binary,
    write_temp_json,
)

logger = logging.getLogger(__name__)


def _detect_language(text: str) -> str:
    """Simple language detection — returns config-compatible language strings."""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            return "chinese"
        if 0x3040 <= cp <= 0x30FF:
            return "japanese"
        if 0xAC00 <= cp <= 0xD7AF:
            return "korean"
    return "english"


# Default sampling parameters
_DEFAULT_TEMPERATURE = float(os.environ.get("TTS_TALKER_TEMPERATURE", "0.9"))
_DEFAULT_TOP_K = int(os.environ.get("TTS_TALKER_TOP_K", "50"))
_DEFAULT_TOP_P = float(os.environ.get("TTS_TOP_P", "1.0"))
_DEFAULT_MAX_AUDIO_LENGTH = int(os.environ.get("TTS_MAX_AUDIO_LENGTH", "1024"))
_DEFAULT_REPETITION_PENALTY = float(os.environ.get("TTS_REPETITION_PENALTY", "1.05"))


class TRTEdgeLLMTTSBackend(TTSBackend):
    """TTS via TRT-Edge-LLM qwen3_tts_inference subprocess."""

    def __init__(self):
        self._ready = False

    # -- TTSBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def capabilities(self) -> set[TTSCapability]:
        return {TTSCapability.BASIC_TTS, TTSCapability.MULTI_LANGUAGE, TTSCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        """Verify all required files exist."""
        required = [
            (TTS_BINARY, "TTS binary"),
            (PLUGIN_PATH, "TRT-Edge-LLM plugin"),
            (os.path.join(TTS_TALKER_DIR, "config.json"), "talker config"),
            (os.path.join(TTS_TALKER_DIR, "llm.engine"), "talker engine"),
            (os.path.join(TTS_TOKENIZER_DIR, "tokenizer.json"), "tokenizer"),
        ]
        missing = []
        for path, label in required:
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")
        if missing:
            raise FileNotFoundError(
                "TTS preload failed — missing:\n  " + "\n  ".join(missing)
            )

        # Code2Wav is optional (graceful fallback)
        c2w_path = os.path.join(TTS_CODE2WAV_DIR, "code2wav.engine")
        if os.path.exists(c2w_path):
            logger.info("Code2Wav engine found at %s", c2w_path)
        else:
            logger.warning(
                "Code2Wav not found at %s — will output RVQ codes only",
                c2w_path,
            )

        logger.info(
            "TTS backend preload OK (binary=%s talker=%s)",
            TTS_BINARY,
            TTS_TALKER_DIR,
        )
        self._ready = True

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Run TTS inference via subprocess.

        Returns (wav_bytes, meta_dict).  ``wav_bytes`` will be empty if the
        Code2Wav engine is unavailable (the backend produced RVQ codes only).
        """
        if not self._ready:
            raise RuntimeError("TTS backend not preloaded")

        # Build input JSON
        input_data = {
            "requests": [
                {
                    "messages": [{"role": "user", "content": text}],
                    "speaker": "",
                }
            ],
            "batch_size": 1,
            "apply_chat_template": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "talker_temperature": _DEFAULT_TEMPERATURE,
            "talker_top_k": _DEFAULT_TOP_K,
            "talker_top_p": _DEFAULT_TOP_P,
            "repetition_penalty": _DEFAULT_REPETITION_PENALTY,
            "max_audio_length": kwargs.get(
                "max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH
            ),
        }

        with tempfile.TemporaryDirectory(prefix="trt_edgellm_tts_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            output_path = os.path.join(tmpdir, "output.json")
            audio_dir = os.path.join(tmpdir, "audio_out")
            os.makedirs(audio_dir, exist_ok=True)

            with open(input_path, "w") as f:
                json.dump(input_data, f)

            # Build CLI args
            cli_args = [
                "--inputFile",
                input_path,
                "--talkerEngineDir",
                TTS_TALKER_DIR,
                "--tokenizerDir",
                TTS_TOKENIZER_DIR,
                "--outputFile",
                output_path,
                "--outputAudioDir",
                audio_dir,
            ]

            # Add code2wav if engine exists
            c2w_path = os.path.join(TTS_CODE2WAV_DIR, "code2wav.engine")
            if os.path.exists(c2w_path):
                cli_args += ["--code2wavEngineDir", TTS_CODE2WAV_DIR]

            t0 = time.time()
            result = run_binary(TTS_BINARY, cli_args, timeout=120)
            elapsed = time.time() - t0

            # Parse output — fail loudly on errors
            if result.returncode != 0 or not os.path.exists(output_path):
                raise RuntimeError(
                    f"TTS subprocess failed (exit={result.returncode}): "
                    f"stdout={result.stdout[-300:]}, stderr={result.stderr[-300:]}"
                )

            with open(output_path) as f:
                output_data = json.load(f)

            responses = output_data.get("responses", [])
            if not responses:
                raise RuntimeError(f"TTS produced no responses: {output_data}")

            r = responses[0]
            audio_file = r.get("audio_file")
            wav_bytes = b""
            meta = {"inference_time_s": round(elapsed, 3), "sample_rate": 24000}

            if audio_file and os.path.exists(audio_file):
                with open(audio_file, "rb") as f:
                    wav_bytes = f.read()
                meta["duration_s"] = r.get("audio_duration_ms", 0) / 1000.0
                meta["samples"] = r.get("audio_samples", 0)
            else:
                logger.warning("No audio WAV in output, returning RVQ codes only")
                meta["rvq_file"] = r.get("rvq_file")
                if not meta.get("rvq_file"):
                    raise RuntimeError(
                        f"TTS output has neither audio nor RVQ: {list(r.keys())}"
                    )

            return wav_bytes, meta
