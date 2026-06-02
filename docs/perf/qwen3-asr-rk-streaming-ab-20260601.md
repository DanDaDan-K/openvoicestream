# Qwen3 ASR RK Streaming A/B (2026-06-01/02)

This note captures real `/v2v/stream` measurements on RK3576, RK3588, and
Jetson Orin Nano for dialogue-style ASR. It focuses on end-of-utterance
latency and recognition quality, not just offline finalize speed.

## Scope

- Client: Mac over Tailscale, `bench/perf/perf.py v2v-stream`
- Protocol: `/v2v/stream`, ASR-only, `--eos client`, realtime audio pacing
- Corpus: `bench/perf/corpus`, `category=short`, `lang=zh|en`
- Steady rows: 3 per language after 1 warmup
- Devices:
  - `cat-remote` RK3576, container `slv-rk3576-probe`
  - `radxa` RK3588, container `openvoicestream-kokoro`
  - `orin-nano` Jetson Orin Nano, container `speech-customvoice`

`zh_short_02` is a known weak row for strict CER because its original label
contains an English parenthetical gloss:
`AlsothePacificTsunamiWarningCenter`. The harness now keeps that original label
auditable as `strict_error_rate`, but the headline `error_rate` uses the
manifest's `eval_transcript`, which matches the spoken Chinese content that all
tested ASR backends actually emit.

## Results

| Device | Decoder | English WER mean | English total latency mean | Chinese CER mean | Chinese total latency mean | Status |
|---|---:|---:|---:|---:|---:|---|
| Jetson Orin Nano `orin-nano` | TRT-EdgeLLM highperf | 6.1% | 327 ms | 22.0% | 353 ms | same streaming protocol baseline |
| RK3576 `cat-remote` | RKLLM `W4A16_G128` | 19.8% | 820 ms | 27.3% | 1328 ms | current usable baseline |
| RK3588 `radxa` | RKLLM `FP16` | 16.1% | 481 ms | 22.0% | 898 ms | best multilingual accuracy |
| RK3588 `radxa` | RKLLM `W8A8` | 21.2% | 361 ms | 22.0% | 575 ms | faster; English quality regression |
| RK3588 `radxa` | RKLLM `W8A8_G128` | 24.6% | 521 ms | 22.0% | 1020 ms | not better than FP16/W8A8 |
| RK3576 `cat-remote` | RKLLM `W4A16` | 84.3% | 549 ms | 82.3% | 927 ms | reject |
| RK3576 `cat-remote` | RKNN matmul `w8a16` | 84.3% | 13506 ms | 89.9% | 14249 ms | reject |

Additional chunk-size A/B on 2026-06-02 used the same client-EOS protocol but
changed the client audio packet size from 250 ms to 100 ms:

| Device | Decoder | Chunk | English WER mean | English total latency mean | Chinese CER mean | Chinese total latency mean | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| RK3576 `cat-remote` | RKLLM `W4A16_G128` | 100 ms | 19.8% | 943 ms | 25.5% | 1425 ms | slower than 250 ms |
| RK3588 `radxa` | RKLLM `FP16` | 100 ms | 16.1% | 954 ms | 22.0% | 1381 ms | slower than 250 ms |

Raw result files are under `bench/perf/results/`:

- RK3576 RKLLM baseline: `v2v_stream_remote_20260601-231707-348225.json`,
  `v2v_stream_remote_20260601-231756-139212.json`
- RK3588 FP16 baseline: `v2v_stream_remote_20260601-231706-051548.json`,
  `v2v_stream_remote_20260601-231832-200019.json`
- RK3588 W8A8: `v2v_stream_remote_20260601-232326-896890.json`,
  `v2v_stream_remote_20260601-232403-419254.json`
- RK3588 W8A8_G128: `v2v_stream_remote_20260601-232614-887600.json`,
  `v2v_stream_remote_20260601-232648-929502.json`
- RK3576 W4A16: `v2v_stream_remote_20260602-000331-784833-p78417-c8e5ea41.json`,
  `v2v_stream_remote_20260602-000409-808073-p79935-40302300.json`
- RK3576 matmul W8A16: `v2v_stream_remote_20260601-233440-050713.json`,
  `v2v_stream_remote_20260601-233615-461290.json`
- Jetson Orin Nano TRT highperf: `v2v_stream_remote_20260602-001824-978982-p18500-62543788.json`,
  `v2v_stream_remote_20260602-001902-675148-p19909-0da88878.json`
- RK3576 client-EOS fast-cancel hot patch check:
  `v2v_stream_remote_20260602-003054-000116-p48690-2a89cee8.json`,
  `v2v_stream_remote_20260602-003132-305238-p50050-53ddbc07.json`
- 100 ms chunk-size A/B after fixing in-flight endpoint/final accounting:
  - RK3576:
    `v2v_stream_remote_20260602-004059-626849-p71591-7404051a.json`,
    `v2v_stream_remote_20260602-004141-501079-p73002-86ec3d35.json`
  - RK3588:
    `v2v_stream_remote_20260602-004058-795706-p71590-db8cab44.json`,
    `v2v_stream_remote_20260602-004140-981500-p73003-93c0ed0a.json`

Use the comparison helper to regenerate the headline table from raw JSON:

```bash
uv run python bench/perf/compare_v2v_stream_results.py --best \
  orin_en=bench/perf/results/v2v_stream_remote_20260602-001824-978982-p18500-62543788.json \
  orin_zh=bench/perf/results/v2v_stream_remote_20260602-001902-675148-p19909-0da88878.json \
  rk3576_w4a16_g128_en=bench/perf/results/v2v_stream_remote_20260601-231707-348225.json \
  rk3576_w4a16_g128_zh=bench/perf/results/v2v_stream_remote_20260601-231756-139212.json \
  rk3588_fp16_en=bench/perf/results/v2v_stream_remote_20260601-231706-051548.json \
  rk3588_fp16_zh=bench/perf/results/v2v_stream_remote_20260601-231832-200019.json \
  rk3588_w8a8_en=bench/perf/results/v2v_stream_remote_20260601-232326-896890.json \
  rk3588_w8a8_zh=bench/perf/results/v2v_stream_remote_20260601-232403-419254.json \
  rk3588_w8a8_g128_en=bench/perf/results/v2v_stream_remote_20260601-232614-887600.json \
  rk3588_w8a8_g128_zh=bench/perf/results/v2v_stream_remote_20260601-232648-929502.json \
  rk3576_w4a16_en=bench/perf/results/v2v_stream_remote_20260602-000331-784833-p78417-c8e5ea41.json \
  rk3576_w4a16_zh=bench/perf/results/v2v_stream_remote_20260602-000409-808073-p79935-40302300.json
```

This sorts candidates by quality first, then latency. That ordering is
intentional for ASR: a low-latency decoder with a large WER/CER regression is a
non-starter for dialogue.

Use the gate mode for candidate acceptance. It fails the run if the candidate
regresses CER/WER beyond the allowed delta, even when latency improves:

```bash
uv run python bench/perf/compare_v2v_stream_results.py --gate \
  --baseline-label rk3588_fp16_en \
  --candidate-label rk3588_w8a8_en \
  --group short/en \
  rk3588_fp16_en=bench/perf/results/v2v_stream_remote_20260601-231706-051548.json \
  rk3588_w8a8_en=bench/perf/results/v2v_stream_remote_20260601-232326-896890.json
```

On the current RK3588 data this rejects W8A8 for English: WER regresses from
16.1% to 21.2%, even though total latency improves from 481 ms to 361 ms.

```bash
uv run python bench/perf/compare_v2v_stream_results.py --gate \
  --baseline-label rk3588_fp16_zh \
  --candidate-label rk3588_w8a8_zh \
  --group short/zh \
  rk3588_fp16_zh=bench/perf/results/v2v_stream_remote_20260601-231832-200019.json \
  rk3588_w8a8_zh=bench/perf/results/v2v_stream_remote_20260601-232403-419254.json
```

On the current RK3588 data this accepts W8A8 for Chinese: mean CER stays at
22.0% while total latency improves from 898 ms to 575 ms.

For dialogue streaming, add a TFD guard when evaluating a candidate that changes
chunking, stream mode, or decoder quantization. This keeps a candidate from
improving post-EOS final latency while delaying the first visible partial:

```bash
uv run python bench/perf/compare_v2v_stream_results.py --gate \
  --baseline-label rk3588_fp16_zh \
  --candidate-label rk3588_w8a8_zh \
  --group short/zh \
  --max-tfd-ratio 1.10 \
  rk3588_fp16_zh=bench/perf/results/v2v_stream_remote_20260601-231832-200019.json \
  rk3588_w8a8_zh=bench/perf/results/v2v_stream_remote_20260601-232403-419254.json
```

On the current RK3588 Chinese data, W8A8 passes with a 10% TFD guard
(`1276 ms -> 1360 ms`) but fails a stricter 5% guard. Treat that as an
explicit product tradeoff: lower final latency, slightly slower first partial.

The raw `/v2v/stream` rows now also record `partial_before_client_eos` so a
streaming run can distinguish true pre-EOU partials from a final-only backend.

When a benchmark JSON contains both languages under the same label, gate them
together:

```bash
uv run python bench/perf/compare_v2v_stream_results.py --gate \
  --baseline-label rk3576_w4a16_g128 \
  --candidate-label rk3576_w8a8 \
  --groups short/en,short/zh \
  --max-tfd-ratio 1.10 \
  rk3576_w4a16_g128=bench/perf/results/<baseline>.json \
  rk3576_w8a8=bench/perf/results/<candidate>.json
```

## Findings

1. RK3576 accuracy is primarily decoder-artifact limited.

   The live RK3576 artifact is:

   ```text
   /opt/asr/models/rkllm/decoder_qwen3.w4a16_g128.rk3576.rkllm
   model_dtype: W4A16_G128
   ```

   RK3588 has FP16 and W8A8-family RKLLM artifacts available. RK3576 does not.
   The older RK3576 `W4A16` artifact loads and is faster, but accuracy collapses
   on both English and Chinese. Runtime parameter A/B (`ASR_REPEAT_PENALTY=1.0`
   vs default `1.15`) also did not materially improve quality, so the next
   useful step is generating or obtaining a less aggressive RK3576 RKLLM
   decoder, ideally W8A8 first and FP16 if memory allows.

   Follow-up on 2026-06-02: WSL2 builder successfully exported
   `decoder_qwen3.w8a8.rk3576.rkllm` (945,098,540 bytes) from
   `/home/harve/qwen3-asr-rknn/decoder_hf` using RKLLM toolkit 1.2.3 and
   `data_quant.json`. Artifact preflight passed on WSL2 and on the RK3576 host.
   The file was copied to the live Docker volume `rk-asr-models` and loaded by
   a temporary non-production container:

   ```text
   loading rkllm model from /opt/asr/models/decoder/decoder_qwen3.w8a8.rk3576.rkllm
   target_platform: RK3576
   model_dtype: W8A8
   ```

   The deployed RK3576 image still has the older runtime lookup that only scans
   `model_dir/decoder`, so a temporary symlink from `decoder/` to `rkllm/` was
   used for the live A/B. The repository fix already makes the runtime scan both
   `decoder/` and `rkllm/` by exact quant/platform tokens; deploy that before
   relying on the cleaner `rkllm/` layout in production images.

   W8A8 improves short-corpus quality but does not pass the dialogue latency
   gate with partial streaming enabled:

   | RK3576 profile | short/en WER | short/en total | short/en TFD | short/zh CER | short/zh total | short/zh TFD |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | W4A16_G128 baseline | 19.8% | 883 ms | 1446 ms | 25.5% | 1358 ms | 1617 ms |
   | W8A8 default partials | 17.6% | 1302 ms | 1580 ms | 17.2% | 1801 ms | 1578 ms |
   | W8A8 throttled partials | 17.6% | 1013 ms | 1888 ms | 17.2% | 2007 ms | 1888 ms |
   | W8A8 no partials diagnostic | 17.6% | 734 ms | 4746 ms | 17.2% | 1471 ms | 3469 ms |

   Gate result vs W4A16_G128 with streaming partials: 0/2 passed. W8A8 is a
   quality probe, not the RK3576 default yet. The no-partial diagnostic shows
   the next performance lever: final latency can improve when stale partial
   decodes are out of the way, but disabling partials destroys dialogue TFD.
   The right follow-up is a true streaming scheduler fix: partial decode must be
   cancellable/preemptible on client EOU and should not monopolize the single
   RKLLM/NPU lane when finalization is pending.

   Follow-up deployment check on 2026-06-02: the migrated voxedge RK image
   `openvoicestream:rk-voxedge-profile` was started on `cat-remote` with
   `OVS_PROFILE=rk3576-w8a8-probe`. The service loaded the expected artifact:

   ```text
   loading rkllm model from /opt/asr/models/decoder/decoder_qwen3.w8a8.rk3576.rkllm
   target_platform: RK3576
   model_dtype: W8A8
   ```

   Three same-image local `/v2v/stream --eos client --chunk-ms 250 --realtime`
   probes were then run on-device: stock voxedge, immediate client-EOS cancel
   hot patch, and the framework-level ASR ingest queue + opt-in partial-abort
   hook.

   | RK3576 voxedge W8A8 probe | short/en total | short/en ASR finalize | short/zh total | short/zh ASR finalize | Notes |
   | --- | ---: | ---: | ---: | ---: | --- |
   | stock image | 321 ms | 14 ms | 2701 ms | 2701 ms | one `zh_short_02` WS loss |
   | immediate cancel hot patch | 382 ms | 19 ms | 2956 ms | 2956 ms | no improvement |
   | ASR ingest queue + partial-abort hook | 377 ms | 19 ms | 2689 ms | 2689 ms | no WS loss; Chinese latency still high |

   Raw local result files on `cat-remote`:
   `v2v_stream_local_20260602-070926-756544.json`,
   `v2v_stream_local_20260602-071337-312606.json`, and
   `v2v_stream_local_20260602-072548-961811.json`.

   This disproves the narrower hypothesis that only `asr_out_task` was queuing
   cancel too late. The earlier blocker is that the WebSocket dispatcher itself
   awaits `asr_manager.accept_audio()`, which offloads
   `stream.accept_waveform()` to the ASR executor. When a synchronous RK partial
   decode is running inside `accept_waveform()`, the dispatcher cannot read the
   following client `asr_eos` frame yet, so a `CLIENT_ASR_EOS`-branch cancel is
   still too late. The framework fix now decouples WS receive from ASR ingest:
   audio/control frames are queued quickly, a dedicated ASR-ingest task runs
   `accept_audio()` in order, and client EOU can invoke an opt-in
   `abort_partial_decode()` hook without forcing backends that do worker IPC
   through an unsafe cancel path. This is the right reusable architecture and
   removes the observed WS-loss class, but it does not make RK3576 W8A8 Chinese
   final decode fast enough. The remaining W8A8 Chinese bottleneck is in the
   decoder/finalization path itself, so the next optimization should target
   final decode budget and prompt/runtime policy (`ASR_MAX_NEW_TOKENS`,
   Chinese early-stop/repetition, rolling-buffer length, and language-specific
   low-latency profiles), not more transport scheduling.

   Follow-up token-budget A/B on 2026-06-02 used the same migrated voxedge
   W8A8 probe container on `cat-remote`. The profile was temporarily changed
   from `ASR_MAX_NEW_TOKENS=80` to 64 and 48, then restored to 80 after the
   run. These were on-device loopback `/v2v/stream --eos client --chunk-ms 250
   --realtime` tests.

   | RK3576 W8A8 token budget | Scope | short/en total | short/en ASR finalize | short/zh total | short/zh ASR finalize | Result |
   | --- | --- | ---: | ---: | ---: | ---: | --- |
   | 64 | mixed short subset | 154 ms | 23 ms | 2946 ms | 2946 ms | no Chinese gain |
   | 48 | mixed short subset | 408 ms | 23 ms | 2786 ms | 2786 ms | no Chinese gain |
   | 80 | zh-only, 3 samples | - | - | 1987 ms mean | 1987 ms mean | restored baseline |
   | 48 | zh-only, same 3 samples | - | - | 2305 ms mean | 2305 ms mean | worse than 80 |

   Raw local result files on `cat-remote`:
   `v2v_stream_local_20260602-073329-660548.json` (64 mixed),
   `v2v_stream_local_20260602-073500-783825.json` (48 mixed),
   `v2v_stream_local_20260602-073705-763111.json` (80 zh-only), and
   `v2v_stream_local_20260602-073826-852636.json` (48 zh-only).

   The zh-only rows are the most useful signal: 48 tokens kept `zh_short_01`
   and `zh_short_02` essentially unchanged but made `zh_short_03` slower
   (`1762 ms -> 2762 ms`). This rules out a simple lower
   `ASR_MAX_NEW_TOKENS` default as the RK3576 fix. The next useful lever is
   decoder early-stop/repetition behavior and why Chinese final decode does
   not hit an endpoint/stop as quickly as English on the same artifact.

   A follow-up decoder-perf smoke added per-final logging for
   `input_tokens`, generated callback chunks, RKLLM prefill time, and RKLLM
   generate time. With the restored 80-token profile:

   | Sample | Input tokens | Generated chunks | RKLLM prefill | RKLLM generate | ASR finalize |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | `zh_short_01` | 72 | 8 | 433-445 ms | 709-735 ms | 1303-1328 ms |
   | `zh_short_02` | 83 | 16 | 468 ms | 1431 ms | 2010 ms |

   This is the clearest current root-cause signal: the slow Chinese sample is
   not hitting the global `max_new_tokens` ceiling, but it generates roughly
   twice as many callback chunks and spends roughly twice as long in RKLLM
   generation. RK3576 W8A8 tuning should therefore target final-output stop
   policy and redundant Chinese continuation, with an accuracy gate, rather
   than lowering the global token budget.

   A low-risk decoder policy A/B on 2026-06-02 then used:

   - `ASR_MAX_NEW_TOKENS=64`
   - `ASR_FINAL_STOP_ON_PUNCT=1`
   - `ASR_FINAL_STOP_MIN_CHARS=8`
   - `ASR_FINAL_STOP_MIN_CHUNKS=2`
   - `QWEN3_ASR_VAD_FINAL_ASYNC=1`

   This is now the preferred RK dialogue profile policy. It keeps the same
   short-zh recognition output in the measured rows and slightly reduces final
   decode time:

   | Device/profile | `zh_short_01` total / CER | `zh_short_02` total / CER | `zh_short_03` total / CER | Mean total | Mean CER |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | RK3588 previous hybrid | 614 ms / 20.0% | 1836 ms / 0.0% | 1839 ms / 5.3% | 1430 ms | 8.4% |
   | RK3588 low-risk policy | 511 ms / 20.0% | 1733 ms / 0.0% | 1836 ms / 5.3% | 1357 ms | 8.4% |
   | RK3576 W8A8 previous hybrid | 1107 ms / 20.0% | 2642 ms / 22.7% | 2794 ms / 21.1% | 2514 ms | 21.3% |
   | RK3576 W8A8 low-risk policy | 1006 ms / 20.0% | 2584 ms / 22.7% | 2755 ms / 21.1% | 2448 ms | 21.3% |

   The policy is a latency improvement, not a RK3576 accuracy fix. RK3576 W8A8
   remains the best available RK3576 quality probe from the tested artifacts
   (better than W4A16/W4A16_G128 on the earlier short corpus), but its Chinese
   CER is still too high for parity with RK3588/Jetson.

   Follow-up W4A16_G128 recalibration A/B on 2026-06-02 generated
   `decoder_qwen3_recalib.w4a16_g128.rk3576.rkllm` on `wsl2-local` using a
   dialogue-style 60-row calibration JSON instead of the previous 6-row generic
   prompt dataset. The candidate was tested on `cat-remote` with the same
   low-risk dialogue policy as W8A8:

   - `ASR_MAX_NEW_TOKENS=64`
   - `ASR_FINAL_STOP_ON_PUNCT=1`
   - `ASR_FINAL_STOP_MIN_CHARS=8`
   - `ASR_FINAL_STOP_MIN_CHUNKS=2`

   Same-device `/v2v/stream --eos client --chunk-ms 250 --realtime` runs used
   one warmup row and three steady rows, so each language below covers rows
   `*_short_02` through `*_short_04`.

   | RK3576 decoder | short/zh mean total | short/zh mean CER | short/en mean total | short/en mean WER | Tail truncation | Decision |
   | --- | ---: | ---: | ---: | ---: | ---: | --- |
   | W4A16_G128 old | 1659 ms | 11.1% | 1357 ms | 19.8% | 0.0% | baseline |
   | W4A16_G128 recalib | 1585 ms | 6.3% | 1326 ms | 16.5% | 0.0% | keep as candidate |
   | W8A8 | 1687 ms | 1.8% | 1597 ms | 24.6% | 0.0% | best zh accuracy, slower and worse en |

   Recalibrated W4A16_G128 is therefore a useful RK3576 candidate: it keeps the
   W4 efficiency advantage and improves quality versus the old W4 artifact. It
   does not fully match W8A8 Chinese accuracy, but it is faster than W8A8 and
   better than W8A8 on this English short subset. It should be expanded to a
   larger mixed-language gate before replacing the W8A8 probe profile.

   Expanded short-set gate on the same day then ran all five short rows per
   language with `warmup=0` and `runs=5`. This is the better decision signal
   than the first 3-row probe:

   | RK3576 decoder | short/zh mean total | short/zh mean CER | short/en mean total | short/en mean WER | Tail truncation | Decision |
   | --- | ---: | ---: | ---: | ---: | ---: | --- |
   | W8A8 | 1614 ms | 5.1% | 1066 ms | 17.6% | 0.0% | stronger Chinese profile |
   | W4A16_G128 recalib | 1700 ms | 7.8% | 1001 ms | 12.7% | 0.0% | stronger English profile |

   The expanded gate changes the candidate decision: recalibrated W4A16_G128 is
   not a clean global replacement for W8A8. W8A8 remains better for Chinese
   accuracy and is also slightly faster on the 5-row Chinese short set.
   Recalibrated W4A16_G128 is better for English accuracy and slightly faster
   on the 5-row English short set. The practical next step is either
   language-aware decoder profiles or a larger mixed-language gate before
   changing the RK3576 default.

   Historical note: an earlier punctuation-stop policy was tested as an opt-in
   decoder setting:
   `ASR_FINAL_STOP_ON_PUNCT=1`, with `ASR_FINAL_STOP_MIN_CHARS=8` and
   `ASR_FINAL_STOP_MIN_CHUNKS=4`. That older policy was measured before the
   hybrid frontend/backend EOU path and async backend endpoint split above. On
   the same zh-only short run it did trigger `abort_reason=final_punctuation`,
   but the abort happened close to the natural end of generation:

   | RK3576 W8A8 setting | `zh_short_01` | `zh_short_02` | `zh_short_03` | Mean | Decision |
   | --- | ---: | ---: | ---: | ---: | --- |
   | Baseline, 80 tokens | 1347 ms | 2852 ms | 1762 ms | 1987 ms | current probe default |
   | Final punctuation stop | 1303 ms | 2715 ms | 2690 ms | 2236 ms | reject as default |

   Raw result: `v2v_stream_local_20260602-075620-698135.json`. Decoder logs
   showed `final_punctuation` aborts on all three samples, but generated
   callback chunks were still 8/16/14 respectively, so the policy did not
   materially reduce the long Chinese generation path and made the third sample
   slower end-to-end. Keep this policy available for future model variants, but
   do not enable it in RK3576/RK3588 defaults.

   VAD-mode follow-up is more promising for the dialogue path. The client-EOS
   benchmark disables the outer `/v2v/stream` VAD and sends `asr_eos`
   immediately after upload, so it measures post-client-EOU finalization. In a
   real dialogue path where the service owns EOU detection, `/v2v/stream
   --eos vad --vad-silence-ms 400` let the service overlap more work before
   the close-out EOS:

   | RK3576 W8A8 VAD mode | `zh_short_01` total/finalize | `zh_short_02` total/finalize | Mean total | Decision |
   | --- | ---: | ---: | ---: | --- |
   | client EOS baseline | 1347 / 1347 ms | 2852 / 2852 ms | 2099 ms | diagnostic baseline |
   | service VAD, 400 ms | 786 / 29 ms | 1551 / 795 ms | 1169 ms | promising dialogue mode |
   | service VAD, 250 ms | truncated outputs | truncated outputs | n/a | reject |

   Raw 400 ms VAD result: `v2v_stream_local_20260602-075858-466080.json`.
   Reducing VAD silence to 250 ms caused premature endpoints and severe
   transcript truncation in logs (`0.80s audio`, texts like `这还是迹象。` and
   `传统上，往往。`), so it is not an acceptable accuracy/performance tradeoff.
   The actionable direction is not lower debounce; it is making the VAD-mode
   path robust and measuring it as the product dialogue default, while keeping
   client-EOS as a lower-level finalization diagnostic.

   RK3588 follow-up shows why that accuracy gate is mandatory. On
   `radxa/openvoicestream-kokoro` (port 8621), same zh short corpus:

   | RK3588 mode | Samples | Mean total | Accuracy note | Decision |
   | --- | ---: | ---: | --- | --- |
   | client EOS | n=2 | 923 ms | `zh_short_01`, `zh_short_03` not truncated | stable diagnostic |
   | service VAD, 400 ms | n=3 | 703 ms | `zh_short_03` truncated to `传统上，王位继承人。` | reject as default |

   Raw results:
   `v2v_stream_local_20260602-000546-879149.json` (client EOS) and
   `v2v_stream_local_20260602-000434-373684.json` (VAD 400 ms). The VAD row
   confirms that "lower total latency" alone is not a valid optimization for
   dialogue ASR; endpoint policy changes must be gated by CER/WER and by a
   no-truncation check on tail-sensitive utterances.

   The perf harness now records `tail_truncated`, `tail_missing_units`, and
   `tail_truncation_rate` for `/v2v/stream` rows. The comparison gate can also
   backfill `error_rate` and `tail_truncation_rate` from older JSON files using
   the local corpus manifest, so historical VAD/client-EOS runs can be audited:

   ```bash
   uv run python bench/perf/compare_v2v_stream_results.py --gate \
     --baseline-label rk3588_client \
     --candidate-label rk3588_vad \
     --group short/zh \
     --forbid-tail-truncation \
     rk3588_client=/private/tmp/rk3588_client_20260602-000546.json \
     rk3588_vad=/private/tmp/rk3588_vad400_20260602-000434.json
   ```

   Result: `FAIL`; RK3588 VAD CER/CER-like error backfill regressed from
   `12.6%` to `46.2%`, and tail truncation was `33.3%`. The same gate on
   RK3576 VAD 400 ms also failed: no tail truncation (`0.0%`), but quality
   regressed from `28.7%` to `40.4%`. This gives a reusable acceptance rule:
   endpoint changes must improve latency while preserving quality and keeping
   tail truncation at zero.

   The same tail-quality harness was then synced to both RK hosts and verified
   in-place, so new device-side JSON now contains `error_rate`,
   `tail_truncated`, `tail_missing_units`, and `tail_truncation_rate` directly
   in each raw row. A later harness update also records `coverage_rate` and
   `short_output_rate` so non-prefix early stops like `这还是迹象。` are visible
   even when they happen to include the reference tail word. Fresh on-device
   loopback runs on 2026-06-02 produced:

   | Device | Mode | Mean CER | Tail truncation | Mean total | Gate decision |
   | --- | --- | ---: | ---: | ---: | --- |
   | RK3576 `cat-remote` | client EOS | 28.7% | 0.0% | 2314 ms | baseline |
   | RK3576 `cat-remote` | service VAD, 400 ms | 55.0% | 0.0% | 840 ms | reject: quality regression |
   | RK3588 `radxa` | client EOS | 28.7% | 0.0% | 1093 ms | baseline |
   | RK3588 `radxa` | service VAD, 400 ms | 46.2% | 33.3% | 558 ms | reject: quality + tail truncation |

   Raw result files:
   - RK3576 client EOS:
     `v2v_stream_local_20260602-081827-772882-p1463600-9a43f51d.json`
   - RK3576 VAD 400 ms:
     `v2v_stream_local_20260602-081915-163987-p1466736-f0e5bcdc.json`
   - RK3588 client EOS:
     `v2v_stream_local_20260602-001823-197730-p2445045-07360f5f.json`
   - RK3588 VAD 400 ms:
     `v2v_stream_local_20260602-001913-853349-p2445220-c8342a98.json`

   These results are the current product-facing conclusion for dialogue EOU:
   VAD can hide finalization latency, but the tested 400 ms policy is not
   acceptable as a default until it passes both CER/WER and no-tail-truncation
   gates.

   Follow-up on the migrated voxedge RK3576 W8A8 probe container found a
   framework/backend mismatch in that rejected VAD path:

   - `/v2v/stream --eos vad --vad-silence-ms ...` configured the outer VAD, but
     the per-session silence setting did not reach Qwen3 true-streaming's
     internal endpoint detector.
   - The voxedge RK adapter also hid stream capability flags from the product
     server, so `/v2v/stream` could not tell that Qwen3 wanted backend-owned
     endpointing.
   - Outer VAD could therefore force `finalize()` before Qwen3 internal VAD had
     accumulated enough tail context, especially on `zh_short_03`.

   The fix is generic and opt-in. `ASRSessionManager` now accepts
   session-scoped `stream_options` and only passes them to backends whose
   `create_stream()` signature supports them. Streams can also expose
   `prefer_backend_endpoint_vad=True`; `/v2v/stream` still sends outer VAD
   events but no longer finalizes solely on outer `speech_end` for those
   streams. Legacy backends keep the old call path and old VAD finalization
   behavior.

   RK3576 hot-patch validation on `cat-remote`, container
   `slv-rk3576-w8a8-probe`, confirmed the new path in logs:

   ```text
   VAD endpoint: ... (silence=500-1000ms speech=...)
   Qwen3-true-stream finalize: ... finalize=0ms
   ASR finalize: mode=true_streaming (no fallback) ms=0
   ```

   Fresh on-device loopback results after the fix:

   | RK3576 W8A8 endpoint mode | Mean CER | Tail truncation | Mean total | Mean endpoint | Mean finalize | Gate vs client EOS |
   | --- | ---: | ---: | ---: | ---: | ---: | --- |
   | client EOS baseline | 28.7% | 0.0% | 2314 ms | 0 ms | 2314 ms | baseline |
   | backend-owned VAD, 800 ms | 32.2% | 0.0% | 3103 ms | 3100 ms | 3 ms | fail: CER + latency |
   | backend-owned VAD, 600 ms | 32.2% | 0.0% | 2583 ms | 2580 ms | 3 ms | fail: CER + latency |
   | backend-owned VAD, 400 ms | 32.2% | 0.0% | 2525 ms | 2521 ms | 4 ms | fail: CER + latency |

   Raw result files on `cat-remote`:
   `v2v_stream_local_20260602-081827-772882-p1463600-9a43f51d.json`
   (client EOS),
   `v2v_stream_local_20260602-091353-521742-p1656879-5cb45a06.json`
   (VAD 800),
   `v2v_stream_local_20260602-091828-124692-p1673325-8825fd25.json`
   (VAD 600), and
   `v2v_stream_local_20260602-091922-744215-p1676823-3bfe11a4.json`
   (VAD 400).

   This is a correctness fix, not yet a default performance win. It removes the
   premature-finalize failure class: `zh_short_03` improved from the earlier
   VAD-truncated `传统上，王维继承人在完成学业。` / `36.8%` CER to
   `传统上，往往继承人在完成学业后会直接入。` / `15.8%` CER. However,
   the candidate still fails the full gate because `zh_short_02` is a
   stable corpus/model mismatch: the reference contains an English parenthetical
   `Also the Pacific Tsunami Warning Center`, while both client-EOS and VAD
   outputs omit that English phrase and score `60.7%` CER. After this fix, the
   remaining RK3576 accuracy/performance work is backend/model work: improve the
   Qwen3 decoder artifact/stop policy and validate with a less pathological
   mixed-language row, not lower the outer VAD debounce further.

   The same minimal backend-owned VAD patch was then applied to the RK3588
   `radxa/openvoicestream-kokoro` container. One namespace issue was found
   during hot patching: do not copy local `server/core/asr_backend.py` into the
   container as `app/core/asr_backend.py`, because the container imports
   `app.*`, not `server.*`. The container was recovered by restoring
   `app/core/asr_backend.py` from the original image
   `openvoicestream:rk-kokoro-2026-05-23-rebuilt`. For RK Qwen3, `qwen3_rk.py`
   must also be deployed with the matching `qwen3/engine.py` and
   `qwen3/decoder.py`; otherwise preload fails with an unexpected
   `final_stop_on_punctuation` argument.

   RK3588 validation after the matched hot patch:

   | RK3588 endpoint mode | Mean CER | Tail truncation | Mean total | Mean endpoint | Mean finalize | Gate vs client EOS |
   | --- | ---: | ---: | ---: | ---: | ---: | --- |
   | client EOS baseline | 28.7% | 0.0% | 1093 ms | 0 ms | 1093 ms | baseline |
   | backend-owned VAD, 400 ms | 32.2% | 0.0% | 1254 ms | 1245 ms | 9 ms | fail: CER + latency |
   | backend-owned VAD, 300 ms | 44.1% | 0.0% | 1153 ms | 1151 ms | 2 ms | reject: early stop |

   Raw result files on `radxa`:
   `v2v_stream_local_20260602-001823-197730-p2445045-07360f5f.json`
   (client EOS),
   `v2v_stream_local_20260602-012935-756701-p2459416-2f0d4b97.json`
   (VAD 400), and
   `v2v_stream_local_20260602-013030-649133-p2459661-12d2385b.json`
   (VAD 300).

   The 400 ms RK3588 run proves the framework fix transfers across RK3576 and
   RK3588: the old `zh_short_03` truncation (`传统上，王位继承人。`) is gone, and
   logs show Qwen3 internal `VAD endpoint` followed by near-zero `finalize`.
   But it still does not pass the product gate because it is slower than
   client-EOS on this corpus and has the same `zh_short_02` mixed-language
   accuracy issue as RK3576. The 300 ms run is explicitly rejected: `zh_short_02`
   endpointed after only `0.80s` of audio and returned `这还是迹象。`
   (`96.4%` row CER). This sets a practical lower bound: for current Qwen3 true
   streaming on RK3588, pushing backend endpoint silence below 400 ms is not a
   safe accuracy/performance optimization.

   A follow-up tried to make that lower threshold safe by adding a generic
   per-session backend endpoint option:
   `stream_options["vad_min_utterance_s"]`, exposed from `/v2v/stream` as
   config field `asr_endpoint_min_speech_s` and environment default
   `OVS_ASR_ENDPOINT_MIN_SPEECH_S`. This is reusable backend infrastructure,
   not a RK-specific default.

   Two additional harness fixes were needed before the A/B was meaningful:

   - VAD-mode `/v2v/stream` benchmark now keeps sending trailing silence until
     final or a bounded tail window instead of sending only
     `vad_silence_ms + chunk_ms`. Real dialogue microphones continue producing
     silence until endpoint; a too-short synthetic tail can create false
     timeouts when backend endpointing is more conservative.
   - `compare_v2v_stream_results.py --gate` now fails when the candidate has
     more error rows than the baseline. Previously a candidate with one timeout
     could still look good because CER/latency were averaged over successful
     rows only.

   RK3588 `vad_silence=300 ms` with min-speech guards was rejected:

   | RK3588 backend endpoint setting | Successful rows | Error rows | Mean total | Result |
   | --- | ---: | ---: | ---: | --- |
   | 300 ms silence, 1.2 s min speech | 2 | 1 | 1763 ms | reject: `zh_short_02` timeout |
   | 300 ms silence, 0.8 s min speech | 2 | 1 | 1764 ms | reject: `zh_short_02` timeout |

   Raw result files:
   `v2v_stream_local_20260602-013945-858460-p2461947-61bb0210.json`
   and
   `v2v_stream_local_20260602-014337-239252-p2462806-ccd099ae.json`.

   The useful conclusion is negative: increasing minimum utterance duration
   blocks the `0.80s` false endpoint, but for `zh_short_02` it prevents backend
   endpointing entirely under the current internal VAD accumulator. Do not use
   min-speech as a default RK3588 optimization. The next backend-level direction
   should inspect the internal webrtc/silero VAD accumulator itself, or use a
   model-aware endpoint confidence/coverage signal, instead of a single global
   min-speech scalar.

   A second guard based on received audio length was then added:
   `stream_options["vad_min_audio_s"]`, exposed as `/v2v/stream` config field
   `asr_endpoint_min_audio_s` and environment default
   `OVS_ASR_ENDPOINT_MIN_AUDIO_S`. Unlike `vad_min_utterance_s`, this does not
   depend on the backend VAD's speech accumulator, so it can delay an early
   endpoint without preventing endpoint forever.

   RK3588 `vad_silence=300 ms, min_audio=1.6 s` still failed:

   | Row | Endpoint/finalize | CER | Text | Backend log |
   | --- | ---: | ---: | --- | --- |
   | `zh_short_01` | 1765 / 251 ms | 20.0% | `我们的非常震惊，这位母亲表示。` | 2.80s audio |
   | `zh_short_02` | 1513 / 251 ms | 96.4% | `这还是迹象。` | 1.60s audio, speech=0.72s |
   | `zh_short_03` | 1512 / 251 ms | 15.8% | `传统上，往往继承人在完成学业后会直接入。` | 3.60s audio |

   Raw result:
   `v2v_stream_local_20260602-015104-721868-p2464721-9298b93d.json`.
   Gate result vs client-EOS: `FAIL` (`CER 44.1%` vs `28.7%`, total `1848 ms`
   vs `1093 ms`). Raising `min_audio_s` further would only move the endpoint
   toward full-audio client-EOS behavior while adding latency, so this is not a
   viable default either.

   A frame-level probe was then added to mirror Qwen3 true-streaming's VAD
   accumulator on the raw WAV. On both RK3588 and RK3576, `zh_short_02` showed
   the expected full utterance rather than the service log's earlier
   `speech=0.72s` fragment:

   ```text
   audio_s=3.90 speech_s=3.06 silence_ms=360 endpoint_s=3.84
   transitions: false@0.02 true@0.38 false@0.46 true@0.58 false@3.56
   ```

   This changed the root-cause diagnosis. The problem was not that Qwen3's
   internal webrtc VAD classified the original audio as only 0.72s speech. The
   product `/v2v/stream` outer VAD could emit a second `SPEECH_START` while the
   backend-owned endpoint stream was already active, and the legacy barge-in
   path always called `asr_manager.on_speech_start()`. That preempted the
   active Qwen3 stream and discarded the earlier audio, so the backend only saw
   the tail fragment. `/v2v/stream` now checks
   `stream.prefer_backend_endpoint_vad`; for those opt-in streams, an outer
   `SPEECH_START` during an active ASR turn still notifies the client/cancels
   TTS, but it does not reopen the ASR stream.

   Fresh no-preempt validation on RK3588:

   | RK3588 endpoint mode | Mean CER | Tail truncation | Mean total | Mean endpoint | Mean finalize | Gate vs client EOS |
   | --- | ---: | ---: | ---: | ---: | ---: | --- |
   | client EOS baseline | 28.7% | 0.0% | 1093 ms | 0 ms | 1093 ms | baseline |
   | backend-owned VAD, 300 ms, no preempt | 32.2% | 0.0% | 1845 ms | 1594 ms | 251 ms | fail: CER + latency |
   | backend-owned VAD, 400 ms, no preempt | 32.2% | 0.0% | 1845 ms | 1594 ms | 251 ms | fail: CER + latency |

   Raw results on `radxa`:
   `v2v_stream_local_20260602-020243-234679-p2467507-7737dd1c.json`
   (VAD 300) and
   `v2v_stream_local_20260602-020342-507000-p2467763-9539199c.json`
   (VAD 400).

   Fresh no-preempt validation on RK3576:

   | RK3576 endpoint mode | Mean CER | Tail truncation | Mean total | Mean endpoint | Mean finalize | Gate vs client EOS |
   | --- | ---: | ---: | ---: | ---: | ---: | --- |
   | client EOS baseline | 28.7% | 0.0% | 2314 ms | 0 ms | 2314 ms | baseline |
   | backend-owned VAD, 400 ms, no preempt | 32.2% | 0.0% | 2897 ms | 2811 ms | 86 ms | fail: CER + latency |

   Raw result on `cat-remote`:
   `v2v_stream_local_20260602-100555-630760-p1837737-09aa05be.json`.

   The fix removes the severe false-fragment failure: `zh_short_02` now decodes
   as `而且太平洋海啸预警中心也表示，并未发现海啸迹象。` with backend logs
   showing `speech=2.98s` and `3.20-3.60s` accumulated audio, instead of
   `这还是迹象。` from a 1.60s fragment. It does not make service VAD pass the
   product gate yet. The remaining RK accuracy delta is mostly model/decoder
   quality: `zh_short_02` has `60.7%` CER under client-EOS too, and no-preempt
   VAD's extra quality loss is mainly `zh_short_03` changing from
   `王位继承人` to `往往继承人`. The remaining performance delta is backend
   endpoint timing/final decode policy: 3576 averages ~2.9s total, 3588 ~1.85s,
   versus client-EOS ~2.31s and ~1.09s respectively.

   A follow-up split backend endpoint detection from final decode completion.
   Qwen3 true-streaming can now expose `is_final=True` to the product server as
   soon as backend VAD endpoint is detected, while the RKLLM final decode runs
   in a short-lived background thread and `finish()` joins it before returning
   final text. This is controlled by `QWEN3_ASR_VAD_FINAL_ASYNC`; it defaults
   off because RK3576 did not benefit, and is enabled only in the RK3588
   Kokoro / Chinese-low-latency profiles that were measured to improve.

   RK3588 `radxa/openvoicestream-kokoro` with `QWEN3_ASR_VAD_FINAL_ASYNC=1`:

   | RK3588 VAD 400 variant | Mean CER | Tail truncation | Mean total | Mean endpoint | Mean finalize | Gate vs client EOS |
   | --- | ---: | ---: | ---: | ---: | ---: | --- |
   | no-preempt sync final | 32.2% | 0.0% | 1845 ms | 1594 ms | 251 ms | fail: CER + latency |
   | no-preempt async final | 32.2% | 0.0% | 1426 ms | 1175 ms | 251 ms | fail: CER + latency |

   Raw results:
   `v2v_stream_local_20260602-020342-507000-p2467763-9539199c.json`
   (sync final),
   `v2v_stream_local_20260602-021510-111396-p2470055-22f547e5.json` and
   `v2v_stream_local_20260602-022039-916765-p2471578-9a9136eb.json`
   (async final).

   RK3576 `cat-remote/slv-rk3576-w8a8-probe` did not benefit from enabling the
   same async final path:

   | RK3576 VAD 400 variant | Mean CER | Tail truncation | Mean total | Mean endpoint | Mean finalize | Decision |
   | --- | ---: | ---: | ---: | ---: | ---: | --- |
   | no-preempt sync final | 32.2% | 0.0% | 2897 ms | 2811 ms | 86 ms | keep |
   | async final forced on | 32.2% | 0.0% | 3035 ms | 2099 ms | 936 ms | reject for RK3576 |
   | default-off after code/profile split | 32.2% | 0.0% | 2769 ms | 2684 ms | 84 ms | no async regression |

   Raw results:
   `v2v_stream_local_20260602-100555-630760-p1837737-09aa05be.json`
   (sync final),
   `v2v_stream_local_20260602-101657-267882-p1876404-b8f3b6d7.json`
   (async forced on), and
   `v2v_stream_local_20260602-102121-255441-p1892367-ef00c34e.json`
   (default off after split).

   This is an example of the current acceptance rule in practice: a generic
   backend capability is useful, but it must be opt-in by profile/device until
   each hardware path proves no regression. RK3588 gets earlier endpoint and
   lower VAD total; RK3576 keeps the synchronous path while decoder/model work
   remains the main optimization lever.

   The evaluation-label fix above was then synced to both RK devices and the
   latest client-EOS and VAD400 runs were repeated on 2026-06-02. This removes
   the misleading `zh_short_02` English-gloss penalty from the headline metric
   while retaining `strict_error_rate` for audit:

   | Device | Mode | Headline CER | Strict CER | Tail truncation | Mean total | Mean endpoint | Mean finalize | Gate vs client EOS |
   | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
   | RK3588 `radxa` | client EOS | 8.4% | 28.7% | 0.0% | 1064 ms | 0 ms | 1064 ms | baseline |
   | RK3588 `radxa` | VAD400 async final | 11.9% | 32.2% | 0.0% | 1426 ms | 1091 ms | 335 ms | fail: quality + latency |
   | RK3576 `cat-remote` | client EOS | 8.4% | 28.7% | 0.0% | 2127 ms | 0 ms | 2127 ms | baseline |
   | RK3576 `cat-remote` | VAD400 sync final | 11.9% | 32.2% | 0.0% | 2874 ms | 2789 ms | 85 ms | fail: quality + latency |

   Raw result files:
   - RK3588 VAD400 async:
     `v2v_stream_local_20260602-022828-366182-p2473080-4d36137a.json`
   - RK3588 client EOS:
     `v2v_stream_local_20260602-022934-110369-p2473327-04fc6683.json`
   - RK3576 VAD400 sync:
     `v2v_stream_local_20260602-102832-362358-p1917744-c3dbbd43.json`
   - RK3576 client EOS:
     `v2v_stream_local_20260602-102938-436986-p1921830-9ae26f4f.json`

   Per-row outputs after the new metric show the remaining real quality gap is
   small but reproducible:

   | Device/mode | `zh_short_01` | `zh_short_02` | `zh_short_03` |
   | --- | --- | --- | --- |
   | RK3588 client EOS | 20.0%, `我们的非常震惊...` | 0.0%, full Chinese tsunami row | 5.3%, `王位继承人...` |
   | RK3588 VAD400 | 20.0%, `我们的非常震惊...` | 0.0%, full Chinese tsunami row | 15.8%, `往往继承人...` |
   | RK3576 client EOS | 20.0%, `我们的非常震惊...` | 0.0%, full Chinese tsunami row | 5.3%, `王位继承人...` |
   | RK3576 VAD400 | 20.0%, `我们的非常震惊...` | 0.0%, full Chinese tsunami row | 15.8%, `往往继承人...` |

   The latest service logs are important: for `zh_short_03`, VAD final and
   client-EOS both decode `3.60s` of audio with `input_tokens=78`. The VAD path
   still produces `传统上，往往继承人在完成学业后会直接入。`, while client-EOS
   produces `传统上，王位继承人在完成学业后会直接入。`. That means the remaining
   VAD accuracy delta is no longer explained by missing audio or an early VAD
   cutoff. The next investigation should compare final decoder input/state
   between the two paths and make RKLLM final decode deterministic/stable, not
   keep lowering endpoint silence.

   Follow-up final-input fingerprinting confirmed that diagnosis. A temporary
   device hot patch enabled `QWEN3_ASR_DEBUG_FINAL_INPUT=1` for one client-EOS
   and one VAD400 pass, then the containers were restored to the default-off
   code. For `zh_short_03`, both paths still had the same high-level size
   (`9` chunks, `3.60s` audio, `(52, 1024)` frames, `78` input tokens), but the
   actual encoder-frame and full-embedding hashes differed:

   | Device | Path | Frame SHA1 | Embed SHA1 | Output |
   | --- | --- | --- | --- | --- |
   | RK3588 `radxa` | client EOS | `88144c89e780e0b3` | `f31b54d61ac39dc3` | `王位继承人` |
   | RK3588 `radxa` | VAD400 | `cf1331264666dde0` | `04590cadeb0fd972` | `往往继承人` |
   | RK3576 `cat-remote` | client EOS | `b3e649663df7b26c` | `bb29ee84715a5519` | `王位继承人` |
   | RK3576 `cat-remote` | VAD400 | `3618389e0687b31e` | `3e062c622cfc40aa` | `往往继承人` |

   This narrows the next fix: VAD finalization is entering final decode with a
   different rolling encoder buffer than client-EOS, despite matching token
   counts. The highest-value optimization is to align/freeze the final encoder
   buffer policy between client-EOS and backend-VAD (for example, endpoint
   should include the same final chunk/context window before setting
   `_finalizing`), then re-run the gate. Decoder stop-policy tuning is secondary
   until the final input hash matches.

   Current endpoint diagnosis after the probe/no-preempt fix: do not lower the
   outer VAD debounce further. The next useful optimizations are backend/model
   work: improve Qwen3 RKLLM decoder quality, reduce final decode budget without
   harming coverage, and add a model-aware endpoint/coverage signal so backend
   VAD can finish earlier only when the decoded utterance is complete.

2. RK3588 W8A8 is a valid Chinese latency option, but not a multilingual default.

   W8A8 reduced Chinese total latency from 898 ms to 575 ms with the same mean
   CER on this corpus. English WER worsened from 16.1% to 21.2%, so FP16 remains
   the default for bilingual dialogue. This has been codified as an explicit
   opt-in profile, `rk3588-zh-lowlatency`, which sets
   `ASR_DECODER_QUANT=w8a8` while leaving `rk3588-default` and
   `rk3588-multilang` on FP16. The opt-in profile also sets
   `QWEN3_ASR_TRUE_ROLL_SEC=5` to align with the true-streaming rolling-buffer
   design and reduce decoder prefill work for dialogue turns; default RK
   profiles explicitly keep `QWEN3_ASR_TRUE_ROLL_SEC=15` for behavior
   continuity until a long-utterance A/B says otherwise.

   Follow-up fix on 2026-06-02: true-streaming runtime knobs are now read when
   each `Qwen3TrueStreamingASRStream` instance is created, not when
   `streaming.py` is imported. This matters for dialogue A/B and
   `/admin/backend/reload`: `QWEN3_ASR_TRUE_ROLL_SEC`,
   `QWEN3_ASR_TRUE_PARTIAL_*`, `VAD_ENDPOINT_SILENCE_MS`,
   `VAD_MIN_UTTERANCE_S`, `QWEN3_ASR_VAD_SUSTAIN_FRAMES`, and the
   `QWEN3_ASR_VAD_BACKEND`/webrtc parameters now follow the active profile
   inside the same Python process.

   `QWEN3_ASR_VAD_FINAL_ASYNC` follows the same runtime-profile rule. It is
   deliberately default-off in code and enabled only by measured RK3588
   profiles (`rk3588-kokoro-rknn`, `rk3588-kokoro-rknn-34pct`, and
   `rk3588-zh-lowlatency`) because RK3576 W8A8 did not improve with the same
   setting.

3. Orin Nano remains the reference latency/English-quality target for dialogue.

   With the same `/v2v/stream --eos client` protocol, Orin Nano highperf returns
   short English finals in ~327 ms mean vs RK3588 FP16 ~481 ms and RK3576
   W4A16_G128 ~820 ms. English WER on this short set is also lower (6.1% vs
   16.1%/19.8%). Chinese mean CER ties RK3588 at 22.0% because `zh_short_02`
   dominates the small sample; median CER is 5.3%, so the Chinese result should
   be read with that corpus caveat.

4. RK3576 matmul W8A16 is not a replacement path in its current form.

   The matmul decoder loaded and ran, but produced short/truncated text and
   12-15 second post-EOU latency. It should not be used for Qwen3 ASR dialogue
   until the matmul weights/runtime are independently debugged against decoder
   logits and token parity.

5. Streaming benchmark reliability matters.

   `/v2v/stream` has a single-session limiter on the tested RK and Jetson
   profiles. Back-to-back benchmark iterations can briefly collide with the
   previous WS slot before the server releases it, producing 4429/session-limiter
   closes. The harness now probes `/v2v/stream` on open, retries immediate slot
   rejects, records remaining errors, and uses unique result filenames to avoid
   parallel process collisions.

   An initial Orin Nano run before this retry fix produced two
   `WebSocketConnectionClosedException` rows with matching server-side
   `session_limiter: WS 4429` logs. The retry fix removed those false errors
   and produced stable Orin Nano rows.

   A second harness issue showed up in the 100 ms chunk-size A/B: while sending
   audio, `/v2v/stream` can already emit `asr_endpoint` or `asr_final` for
   short clips with trailing silence. The client previously only sampled
   `asr_partial` during upload, so it could discard endpoint/final timestamps
   and later report misleading near-zero post-EOU latency. The harness now
   consumes endpoint/final frames during audio upload and records
   `endpoint_before_client_eos` / `final_before_client_eos` in raw rows. The
   headline latency fields are post-client-EOS wait time, so frames that arrive
   before client EOS are clamped to zero instead of producing negative endpoint
   latency.

6. RKLLM artifact lookup must check both layouts.

   Adding `model_dir/decoder/matmul_*` created a `decoder/` directory on RK3576,
   which previously masked the valid RKLLM artifact in `model_dir/rkllm/`. The
   engine now searches both `decoder/` and `rkllm/` for RKLLM files.

7. RKLLM artifact lookup must match quantization exactly.

   `ASR_DECODER_QUANT=w4a16` previously matched `w4a16_g128` because the engine
   searched with substring patterns like `*w4a16*`. This made A/B runs load the
   wrong artifact. The engine now matches filename tokens exactly, so `w4a16`,
   `w4a16_g128`, `w8a8`, and `w8a8_g128` remain distinct.

8. Client-EOS fast-cancel is semantically correct but not a RK3576 cure.

   `/v2v/stream` now calls `stream.cancel_and_finalize()` before
   `finalize_with_status()` for `asr_eos`, allowing backends to abort stale
   partial decodes and skip residual sub-chunk tail encoding. A RK3576 hot patch
   against `slv-rk3576-probe` showed no stable end-to-end improvement on the
   current `W4A16_G128` artifact: English mean total latency was ~883 ms
   vs the earlier ~820 ms baseline, and Chinese was ~1358 ms vs ~1328 ms.
   Logs show samples with VAD pre-fire finalize near-instantly, while client-EOS
   samples still spend ~1.0-1.6 s in the final RKLLM decode. This reinforces
   that RK3576's main fix is a better decoder artifact/quantization, not a
   server EOU sequencing tweak.

9. Smaller client chunks are not a default performance win for RK Qwen3-ASR.

   With 100 ms chunks, RK3576 English total latency rose from ~820 ms to
   ~943 ms and Chinese from ~1328 ms to ~1425 ms. RK3588 FP16 English rose from
   ~481 ms to ~954 ms and Chinese from ~898 ms to ~1381 ms. Accuracy was
   essentially unchanged on this small corpus. The likely reason is that more
   frequent small packets increase Python/WebSocket/backend scheduling overhead
   without changing the final RKLLM decode cost. Keep 250 ms as the current
   default for RK Qwen3-ASR dialogue until a backend-side chunk scheduler or
   true incremental decoder path is proven.

10. There are no remaining untested RK3576 RKLLM artifacts on the current device.

   `cat-remote` currently exposes only:

   ```text
   /opt/asr/models/rkllm/decoder_hf.w4a16.rk3576.rkllm
   /opt/asr/models/rkllm/decoder_qwen3.w4a16_g128.rk3576.rkllm
   /opt/asr/models/decoder/matmul_w8a16/
   ```

   All three have been tested above and are either the current baseline
   (`W4A16_G128`) or rejected (`W4A16`, matmul W8A16). `radxa` has the expected
   RK3588 FP16/W8A8/W8A8_G128 artifacts under `/opt/asr/models/decoder/`, all
   already tested. Further RK3576 quality/performance improvement requires
   generating a new RK3576 RKLLM artifact rather than only switching runtime
   environment variables on the deployed container.

## Recommended Next Steps

1. Treat the generated RK3576 W8A8 artifact as a probe, not the default. It
   improved the old strict short-corpus quality versus W4A16_G128, but still
   does not pass the dialogue gate once streaming/VAD behavior is included.

2. Add an opt-in RK Qwen3 final-decode diagnostic that records comparable final
   input/state fingerprints for client-EOS and backend-VAD paths: accumulated
   audio duration, token count, embedding/frame checksum or norm, generated
   chunk count, abort reason, and output text. Use it to explain why
   `zh_short_03` changes from `王位继承人` to `往往继承人` with the same
   `input_tokens=78`.

3. Keep the generic stream framework changes backend-safe: session
   `stream_options`, backend-owned endpoint preference, no-preempt during an
   active backend endpoint stream, and opt-in async final. Defaults must stay
   compatible with Jetson TRT-EdgeLLM and non-RK backends.

4. Keep RK3588 default on FP16 for multilingual mode. Add a Chinese-low-latency
   profile that sets `ASR_DECODER_QUANT=w8a8` only if the product accepts the
   English regression.

   Implemented profile:

   ```bash
   OVS_PROFILE=rk3588-zh-lowlatency \
   docker compose -f deploy/docker-compose.radxa.yml up -d
   ```

   Before using it on a fresh device, confirm
   `decoder_qwen3.w8a8.rk3588.rkllm` exists under `/opt/asr/models/decoder/`
   or `/opt/asr/models/rkllm/`, and confirm startup logs show the W8A8 artifact
   and `model_dtype`. `deploy/docker-compose.radxa.yml` intentionally leaves
   `ASR_DECODER_QUANT` and `QWEN3_ASR_TRUE_ROLL_SEC` empty by default so the
   selected profile owns decoder quant and streaming rolling-buffer length. If
   an operator sets a conflicting `ASR_DECODER_QUANT` in the shell or `.env`,
   startup now fails loudly instead of silently shadowing the profile.
   Profile reload/preflight also checks that the selected
   `ASR_DECODER_QUANT` has an exact-token RKLLM file in `ASR_MODEL_DIR/decoder`
   or `ASR_MODEL_DIR/rkllm`; a missing W8A8 artifact is reported as
   `ASR_RKLLM_DECODER` before backend preload.
   On code with the 2026-06-02 stream-config fix, hot reload also applies
   rolling-buffer/partial/VAD tuning to newly created streams without requiring
   a Python process restart.

## Builder Notes

Checked the currently authorized RK devices on 2026-06-02:

- `cat-remote` has deployed runtime assets under `/home/cat/qwen3-asr-models`,
  but only the existing `W4A16_G128` RKLLM artifact plus matmul decoder weights.
  Its service container does not have `rkllm`, `rknn`, `torch`, or
  `transformers` installed.
- `radxa` has the RK3588 runtime artifacts already tested above. Its service
  container also lacks `rkllm`, `rknn`, `torch`, and `transformers`; the old
  `/home/radxa/rkvoice-stream/.venv/bin/python` is not executable and there is
  no local quantization dataset.

So the authorized RK devices are suitable for runtime A/B, not for producing
new RKLLM decoder artifacts. The next artifact-generation step needs an x86
Linux builder with RKLLM toolkit and the prepared HF decoder directory.

There is an existing x86 Linux builder on `wsl2-local`:

```text
/home/harve/qwen3-asr-rknn/decoder_hf
/home/harve/qwen3-asr-rknn/data_quant.json
/home/harve/qwen3-asr-rknn/rkllm/decoder_qwen3.fp16.rk3588.rkllm
/home/harve/qwen3-asr-rknn/rkllm/decoder_qwen3.w8a8.rk3588.rkllm
/home/harve/qwen3-asr-rknn/rkllm/decoder_qwen3.w8a8_g128.rk3588.rkllm
```

The default WSL2 Python currently fails to import RKLLM toolkit because its
Torch/CUDA/NCCL libraries are mismatched, but the existing environment below
does import RKLLM successfully:

```text
/home/harve/qwen3-tts-export/.venv/bin/python
```

Use `scripts/convert_qwen3_asr_rkllm.py` from this repository with that Python
environment to generate RK3576 candidates once WSL2 writes are allowed.

Planned builder commands:

```bash
python /home/harve/convert_qwen3_asr_rkllm.py \
  --decoder-hf /home/harve/qwen3-asr-rknn/decoder_hf \
  --dataset /home/harve/qwen3-asr-rknn/data_quant.json \
  --out-dir /home/harve/qwen3-asr-rknn/rkllm \
  --target-platform rk3576 \
  --quant w8a8 \
  --quant-algorithm normal \
  --npu-cores 2 \
  --overwrite

python /home/harve/convert_qwen3_asr_rkllm.py \
  --decoder-hf /home/harve/qwen3-asr-rknn/decoder_hf \
  --dataset /home/harve/qwen3-asr-rknn/data_quant.json \
  --out-dir /home/harve/qwen3-asr-rknn/rkllm \
  --target-platform rk3576 \
  --quant w8a8_g128 \
  --quant-algorithm normal \
  --npu-cores 2 \
  --overwrite
```

Before copying a generated artifact to a RK board, run the local preflight gate
to catch token/name collisions:

```bash
python scripts/check_rkllm_artifact.py \
  /path/to/decoder_qwen3.w8a8.rk3576.rkllm \
  --target-platform rk3576 \
  --quant w8a8 \
  --model-dir /path/to/local-or-mounted/opt/asr/models

python scripts/check_rkllm_artifact.py \
  /path/to/decoder_qwen3.w8a8_g128.rk3576.rkllm \
  --target-platform rk3576 \
  --quant w8a8_g128 \
  --model-dir /path/to/local-or-mounted/opt/asr/models
```

The gate intentionally rejects substring-only matches, so
`ASR_DECODER_QUANT=w8a8` cannot accidentally validate a
`decoder_qwen3.w8a8_g128.rk3576.rkllm` file. After upload, the runtime log must
still be checked for both `loading rkllm model from ...` and `model_dtype`.

Use the runtime log verifier after restarting a candidate profile:

```bash
docker logs <container> 2>&1 | python scripts/check_rkllm_runtime_log.py \
  --quant w8a8 \
  --target-platform rk3576 \
  --artifact-basename decoder_qwen3.w8a8.rk3576.rkllm
```

This verifies the loaded artifact basename, exact quant/platform filename
tokens, `target_platform`, and `model_dtype`. It catches the common failure
mode where the env requests `w8a8`, but the runtime actually loads
`w8a8_g128` or the previous `w4a16_g128` decoder.

The verifier has been checked against the current live containers:

```text
radxa/openvoicestream-kokoro:
  model : /opt/asr/models/decoder/decoder_qwen3.fp16.rk3588.rkllm
  dtype : FP16
  target: RK3588

cat-remote/slv-rk3576-probe:
  model : /opt/asr/models/rkllm/decoder_qwen3.w4a16_g128.rk3576.rkllm
  dtype : W4A16_G128
  target: RK3576
```

Both logs contained two RKLLM load events. When `--artifact-basename` is
provided, the verifier selects the matching basename even if other RKLLM models
are also present in the same log; otherwise it warns and checks the last load
event.
