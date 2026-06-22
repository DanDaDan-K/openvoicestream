# Deploy: Qwen3-TTS CustomVoice (v0.8.0, int4+fp8)

CustomVoice (CV) Qwen3-TTS-12Hz-0.6B with an **int4 talker + fp8 text_embedding**,
named speakers (e.g. `serena=3066`), served by the v0.8.0 streaming worker
(`qwen3_tts_streaming_worker`). One worker binary serves **both** Base (8-row
prefix) and CustomVoice (9-row) via a runtime-if on `langId` (overlay
`UPSTREAM_PIN=c48c0de`, fork branch `wip/cv-9row-v080-n1n2`).

Profile: [`configs/profiles/jetson-edgellm-v080-customvoice.json`](../configs/profiles/jetson-edgellm-v080-customvoice.json).

## Status — through-service gate PASS (orin-nano, non-production)

| case | HTTP | faster-whisper roundtrip |
|------|------|--------------------------|
| zh (`language=chinese`) | 200 | `你好，很高兴见到你。` (exact) |
| en (`language=english`) | 200 | `Hello! Nice to meet you.` (exact) |
| Base (no `language`) | 200 | `你好，很高兴见到你。` (exact, no regression) |

Worker reached ready, no `tts_manager_start_failed`, 0 CUDA errors. Standalone
worker gate (langId 2055/2050 → 9-row, langId -1 → 8-row, all EOS) also PASS.

## Image

Overlay the CV worker + a **dedicated CV plugin** onto the `:v0.8.0-n1n2-rebake`
base (do not rebuild other workers — the base ASR/MOSS workers + their plugins
are untouched):

| in-image path | md5 | what |
|---|---|---|
| `/opt/jv-workers/qwen3_tts_streaming_worker` | `50d586de…` | CV (Base+CV) streaming worker, from `c48c0de` |
| `/opt/jv-workers/libNvInfer_edgellm_plugin_cv.so` | `e723ffc7…` | CV-matched plugin (dedicated path) |
| `/opt/edgellm-v080/plugins/libNvInfer_edgellm_plugin.so` | `0c058bee…` | ASR + Base TTS plugin — **untouched** |
| `/opt/jv-workers/libNvInfer_edgellm_plugin.so` | `09ad20a8…` | MOSS plugin — **untouched** |

The CV model bundle (int4 talker + fp8 text_embedding + code_predictor +
code2wav + tokenizer) bakes to `/opt/models/qwen3-tts-customvoice/`.

## Three load-bearing gotchas (why the first gate failed)

1. **The plugin must be the CV-matched one.** The streaming worker built from
   `c48c0de` core-dumps at TRT engine deserialize (*"Using an engine plan file
   across different models of devices"*) against any other plugin. It is installed
   at a **dedicated path** so the shared ASR/MOSS plugins stay untouched, and
   `EDGELLM_PLUGIN_PATH` points at it.

2. **`EDGELLM_PLUGIN_PATH` is NOT operator-prefixed.** `profile_loader.py`'s
   `_OPERATOR_KEY_PREFIXES` lists `EDGE_LLM_` (with underscore), not `EDGELLM_`.
   So a container `-e EDGELLM_PLUGIN_PATH=…` is **overwritten** by the profile's
   value. The CV plugin path therefore MUST live in the **profile env** (it does,
   above) — not just `-e`. *(See "Open items" — this asymmetry is a footgun.)*

3. **The baked entrypoint pre-stamps `EDGE_LLM_TTS_*` + `OVS_TTS_WORKER_CONCURRENCY`.**
   Those ARE operator-prefixed, so they win over the profile and point the talker
   at the wrong (baked Base) path → `tts_manager_start_failed`. Until the
   entrypoint is made conditional, pass the CV paths as explicit `-e` operator
   overrides. Also `EDGE_LLM_TTS_STATEFUL_CODE2WAV=1` is REQUIRED — the streaming
   worker uses `_synthesize_worker_via_stream` (gated on `stateful_code2wav_enabled()`);
   `=0` takes the non-streaming branch that raises `KeyError: 'output_file'`.

## Run recipe (until the entrypoint is fixed)

Bring up the CV image with the profile **and** these operator `-e` overrides (the
entrypoint pre-stamp defeats the profile for `EDGE_LLM_TTS_*`/`OVS_TTS_*`):

```bash
-e OVS_PROFILE=jetson-edgellm-v080-customvoice \
-e EDGE_LLM_TTS_WORKER_BIN=/opt/jv-workers/qwen3_tts_streaming_worker \
-e EDGE_LLM_TTS_TALKER_DIR=/opt/models/qwen3-tts-customvoice/talker_assembled_dir \
-e EDGE_LLM_TTS_CP_DIR=/opt/models/qwen3-tts-customvoice/code_predictor \
-e EDGE_LLM_TTS_CODE2WAV_DIR=/opt/models/qwen3-tts-customvoice/code2wav \
-e EDGE_LLM_TTS_TOKENIZER_DIR=/opt/models/qwen3-tts-customvoice/tokenizer \
-e EDGE_LLM_TTS_STATEFUL_CODE2WAV=1 \
-e OVS_TTS_MODEL_ID=qwen3-tts-customvoice \
-e OVS_TTS_WORKER_CONCURRENCY=1
# EDGELLM_PLUGIN_PATH comes from the profile (NOT operator-prefixed — see gotcha 2).
# The :v0.8.0-n1n2-rebake base is a SLIM image: also bind-mount host CUDA/TRT libs
# (/host-cuda, /host-nvidia-libs, /host-libs) per deploy/jetson-release-highperf.sh.
```

## Open items (for review / follow-up)

- **Fix the `EDGELLM_` vs `EDGE_LLM_` operator-prefix asymmetry** in
  `profile_loader.py` (rename the var to `EDGE_LLM_PLUGIN_PATH`, or add `EDGELLM_`
  to the operator prefixes). Today the gap *happens* to help (profile carries the
  CV plugin) but it is a silent footgun.
- **Make the image entrypoint not pre-stamp `EDGE_LLM_TTS_*`/`OVS_TTS_*`** when a
  profile supplies them, so the profile alone drives CV (drops the `-e` recipe).
- **Bake the CV bundle to `/opt/models/qwen3-tts-customvoice/`** and re-run the
  through-service gate against the baked `/opt` paths (the PASS above used a
  RO-mounted `/cv-bundle`).
- **Image push** is held — not pushed to any registry yet.
