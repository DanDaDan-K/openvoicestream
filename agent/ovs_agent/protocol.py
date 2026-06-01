"""V2V WebSocket protocol message-type constants (agent-side copy).

These mirror the server's ``app/core/v2v.py`` (the SLV /v2v/stream protocol).
They are vendored here so ``ovs_agent`` is a self-contained package with NO
import dependency on the SLV server source tree — the agent image ships only
``agent/`` (see docs/BUILD_IMAGES.md), and the server-side ``app/`` is being
renamed to ``server/``, which would break a ``from app.core.v2v import`` here.

⚠️  MUST stay byte-for-byte in sync with ``app/core/v2v.py`` on the server.
    These are stable wire-protocol identifiers; changing one without changing
    the matching server constant silently breaks the session. If you add a new
    message type to the server protocol, add it here too.
"""
from __future__ import annotations

# ── Client → Server JSON message types ──────────────────────────────────
CLIENT_CONFIG = "config"          # initial setup, must be first message
CLIENT_TEXT = "text"              # streaming text input for TTS
CLIENT_ASR_EOS = "asr_eos"        # manually finalize ASR (overrides VAD)
CLIENT_TTS_FLUSH = "tts_flush"    # flush remaining TTS buffer
CLIENT_ABORT = "abort"            # barge-in: cancel current TTS
CLIENT_TOOL_RESULT = "tool_result"        # device returns a remote-tool result
CLIENT_TOOL_ADVERTISE = "tool_advertise"  # device uploads local tool schemas

# ── Server → Client JSON message types ──────────────────────────────────
SERVER_ASR_PARTIAL = "asr_partial"
SERVER_ASR_ENDPOINT = "asr_endpoint"        # VAD detected end of speech
SERVER_ASR_FINAL = "asr_final"
SERVER_TTS_STARTED = "tts_started"          # first audio frame about to ship
SERVER_TTS_SENTENCE_DONE = "tts_sentence_done"  # one sentence finished
SERVER_TTS_DONE = "tts_done"                # flush complete, no more audio
SERVER_VAD_EVENT = "vad_event"              # server-side VAD speech_start/end
SERVER_ERROR = "error"
SERVER_TOOL_CALL = "tool_call"              # server asks device to run a tool

# ── vad_event "event" field values ──────────────────────────────────────
VAD_EVENT_SPEECH_START = "speech_start"
VAD_EVENT_SPEECH_END = "speech_end"
