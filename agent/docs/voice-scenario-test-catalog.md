# reBot Voice Agent — E2E Scenario Test Catalog

Comprehensive scenario roadmap for the reBot voice agent, designed against the
real architecture (server-loop tool-calling + dialogue). Drives the build-out of
`tests/e2e/` and the deterministic `tests/sim_pump.py` layer. (Generated 2026-06-14.)

## Two-layer rule (which layer tests what)
- **deterministic-sim** (`tests/sim_pump.py`, no device, CI-fast): agent-side
  timing mechanics — gate/preroll, reconnect-window drop, tone suppression,
  state-machine dispatch (empty/duplicate/low-signal final), wake failure.
- **real-SLV** (`tests/e2e/*`, live orin-nx ASR): real recognition accuracy,
  onset/homophone, multi-turn ASR.
- **server-loop-tool**: the production tool-calling path (LLM + tools on SLV).
  The voice/ASR layer is shared between client-loop and server-loop (both use
  `/v2v/stream`), so ONLY LLM/tool/dialogue scenarios need server-loop.

## Server-loop tool-call harness (the key enabler — currently missing)
Most P0 tool-call/dialogue scenarios need a harness the repo lacks. Preferred:
extend `MultiModeApp` with **stub tools** (`{"ok":True,"started":True}` handlers),
launch with `OVS_V2V_SERVER_LOOP=1`, drive `ServerToolCall` events via a
`FakeSLVClient`. Fallback: launch `voice_rebot_arm` app with a **fake actuator**
(no physical arm). Full real-SLV+LLM only for semantic-reasoning scenarios
(coreference / clarification, e.g. MT-009/010/015).

## Implementation batches (P0 first)
**Batch 1 (P0 — highest demo risk):**
- TC-001/002/003 — server-loop tool-call basics: command→preamble→action; preamble-TTS barge dropped; action-body (IDLE) speech captured. *(needs the stub-tool harness)*
- TC-006/008/009/010/010b/012/014 — tool failure+retry / sequential / wrong-tool-corrected / slow-non-blocking / async-timeout→ok=False / concurrent / dead-WS-drop. **(DONE — `test_server_loop_tool_scenarios.py`)**
- TC-007/011/013 — interrupt-during-tool (barge), 4429-during-tool. TC-013's agent-side behavior == TC-014 (both the `connect_if_dead=False` drop). TC-007/011 need barge mechanics → sim_pump or real-SLV.
- MT-006/008/011/013 — N>4 dialogue; dialogue×tool interleave; coreference "放回去"; one-sentence chat+command.
- ER-001/009/011 — empty final ignored; **arm unavailable/perception-fail → clean ok=False once (DONE)**; **wake failure (DONE — test_wake_reconnect_policy)**. (ER-001/009 in `test_server_loop_tool_scenarios.py`.)
- ER-005/008 — mid-speech disconnect; SLV-side LLM timeout. Both need real-SLV (server owns the LLM/timeout in server-loop) → not deterministically faithful agent-side.
- LC-002/003/005/006/008 — idle→sleep→wake; long idle>30s reconnect; 4429 boot/runtime; explicit sleep cancels in-flight tool.
- AC-003/005/009 — noise prefix; far-field/low-gain; mid-utterance VAD split.

**Batch 2 (P1):** TC-014/015/016; MT-007/009/010/012/014/015; ER-002/007/010/012/014/015/018; LC-007/009; AC-004/006/007/008/010/011/014; SP-005/006.

**Batch 3 (P2):** MT-016/018; ER-016; LC-010/011/012; AC-017/018; SP-007.

## Already covered (do not re-implement)
- Timing mechanics: T1 preroll, T2 reconnect-window drop, T3/T4 tone suppress
  (`test_mic_pump_preroll`, `test_reconnect_window_drop`, `test_tone_suppress_onset`).
- reBot capture/corpus: clean/low onset, wake-tone, done-tone sweep, Qwen +
  macOS command corpus, wake-tail bleed, echo gate
  (`test_rebot_voice_capture`, `test_command_during_speaking_is_dropped`).
- Scenario sweep: baseline, after_wake_fast, donetone react sweep, during_reply,
  after_action_idle, turn3/turn4 consecutive, after_rewake, soft_onset
  (`test_scenario_accuracy`).
- reBot-param classics: stop_intent, empty_final, idle_stability
  (`test_rebot_scenarios`).
- server-loop protocol units: advertise/readvertise, unknown-tool, failure-result
  (started=False→ok=False), background dispatch, env quote-strip, no-local-LLM.
- server-loop tool-flow scenarios: TC-006/008/009/010/010b/012/014, ER-001/009
  (`test_server_loop_tool_scenarios.py`).

## PRODUCT-OBSERVATION correction (per-tool timeout)
There IS an agent-side per-tool timeout, contrary to the earlier note:
`ToolRegistry.dispatch` wraps **coroutine** handlers in
`asyncio.wait_for(..., timeout_s)` (default 10s, `tools/registry.py:299`) →
overrun returns `{"success": False, "error": "...timed out..."}` → ok=False
(asserted by TC-010b). The real gap is narrower: a **blocking sync** handler is
run unwrapped (`t.fn(**clean)` not awaited through wait_for), so it would not
time out and would tie up its dispatch task. Arm motion handlers should be async
(or internally bounded) to inherit the timeout.
- Wake policy + failure: reconnect decision matrix + wake-failure stays-SLEEPING
  (`test_wake_reconnect_policy`).

## Coverage gaps (zero coverage today, by category)
- **Tool-call flow (server-loop)**: preamble barge / IDLE action-body capture /
  failure-retry / sequential / timeout / wrong-tool / concurrent / 4429 e2e.
- **Multi-turn dialog**: N>4 server-loop dialogue, dialogue×tool, topic switch,
  coreference, mixed chat+command.
- **Error/recovery**: empty/duplicate/low-signal final, mid-speech disconnect
  real e2e, server-side LLM/TTS timeout, arm unavailable/perception-fail,
  wake false-positive.
- **Lifecycle**: full idle→sleep→wake, long idle>30s, boot/runtime 4429,
  explicit-sleep-vs-command-sleep, watchdog restarts.
- **Acoustic**: noise prefix, far-field/low-gain sweep, fast/slow speech, VAD
  mid-split, two-speaker overlap, background noise SNR, clipping.

## Observability note
The dashboard `/ws` does not expose `ServerToolCall` / `CLIENT_TOOL_RESULT`
directly. Tool-flow e2e assertions need a stub-tool counter or a
`FakeSLVClient` result spy (scenarios marked `[VERIFY]` in the design need an
observability hook before they can assert).
