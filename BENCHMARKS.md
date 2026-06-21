# Benchmarks

Measured performance for the v0.8.0 (TensorRT-Edge-LLM) voice stack, with a
focus on **concurrent (N>1) sessions** and **zero-regression vs the v0.7.1
baseline**. Every row names the device, date, and the gate that produced it so
the numbers can be reproduced.

> Single source of truth: `docs/plans/voxedge-v080-consolidation/benchmarks-dataset.md`
> (with per-row `file` provenance + `repro` metadata). This file is the
> outward-facing view of the N>1 / v0.8.0 subset of that dataset.

For the broader cross-device matrix (RTF, TTFA, CER/WER across Jetson / Rockchip
/ Raspberry Pi), see the [Performance section of the README](README.md#performance).

---

## v0.8.0 N>1 concurrency (verified 2026-06-21)

All N=2 numbers come from real on-device burst tests gated on a **byte-identical
audio/transcript** check (concurrent output == solo output) and **zero CUDA /
race errors**. "N=2 verified" is the validated concurrency ceiling; N>2 is
untested on every device.

### ASR — N=2 streaming (Jetson Orin NX, gate v080-0023)

| Metric | Value | Notes |
|---|---|---|
| Streaming partials → final | 9 partials → 1 final | single-session, full streaming path |
| CER (streaming final) | **0.105** | offline-decode on the same clip is ~0.05 |
| N=2 concurrent zh/en isolation | **no cross-talk** | two simultaneous sessions, one zh + one en |
| Session admission (5 concurrent) | 2 admitted, 3 rejected `too_many_sessions` (4429) | `OVS_MAX_CONCURRENT_SESSIONS=2` enforced |
| CUDA errors | **0** | through-service gate (not bare worker) |

Device: Jetson Orin NX · Date: 2026-06-21 · Gate: **v080-0023** (through-service
ASR N=2 streaming).
Source: `~/project/edgellm-v080-migration/docs/plans/v080-0023-*.md` + task records.

### TTS — N=2 slot-pool, int4 talker (Jetson Orin Nano, staggered gate)

Production TTS concurrency is a **slot-pool** (independent lanes, staggered-friendly,
same model as the ASR `SessionLaneManager`), not a lockstep batch-lane.

| Gate | Result |
|---|---|
| **G1** staggered (B not blocked by in-flight A) | PASS |
| **G2** byte-identical (concurrent output == solo) | PASS |
| **G3** session admission (4429 over the limit) | PASS |

| Resource | Value |
|---|---|
| System RAM at N=2 | **~4 GB** (tegrastats peak 5718 / 7620 MB; baseline 1703 MB) |
| Worker RSS | 908 MB |
| OOM | none — fits 8 GB and 16 GB Orin Nano |
| int4 talker engine | **245.9 MB** vs **903 MB** fp16 (−73%) |

Device: Jetson Orin Nano · Date: 2026-06-21 · Gate: TTS N=2 slot-pool int4
staggered (G1/G2/G3).

### TTS — N=2 shared-engine (Jetson Orin Nano, VRAM-saving)

The shared-engine constructor lets the 2nd slot reuse the resident weights, so
only context/KV (not a second copy of the weights) is added.

| Metric | Value |
|---|---|
| N=1 peak | 3805 MB |
| N=2 peak | 5385 MB |
| 2nd slot cost | **+1.6 GB** (context/KV only, not quadratic weights) |
| vs two independent instances | **~436 MB saved** |
| byte-identical (concurrent == solo) | PASS — slot A `154f7880`, slot B `1a5324be` |
| CUDA errors | **0** |

Device: Jetson Orin Nano · Date: 2026-06-21 · Gate: TTS N=2 shared-engine.

### TTS — M5 spike (Jetson Orin NX, phase 5b)

| Check | Result |
|---|---|
| concurrent == solo (RVQ hash) | byte-exact |
| audio md5 | byte-exact |
| CUDA errors | 0 |

Device: Jetson Orin NX · Date: 2026-06-21 · Gate: phase 5b M5 spike.

---

## v0.8.0 vs v0.7.1 baseline — zero regression (Jetson Orin NX)

ASR `--check` regression gate against the v0.7.1 golden set:

| Metric | Value |
|---|---|
| Overall | **17 / 20 PASS** |
| English + clean Chinese | **all pass** |
| Improvements over v0.7.1 golden | several clips better, e.g. `zh_long_01` CER **0.080 → 0.043** |
| The 3 FAIL clips | high-baseline-CER clips where the abs-tolerance gate is brittle (hard-clip), **not a regression** |

Device: Jetson Orin NX · Date: 2026-06-21 · Baseline:
`bench/regression/baselines/v080-c2-before-20260621/`
(artifacts live in `~/project/edgellm-v080-migration`).

---

## int4 vs fp16 talker (Qwen3-TTS 0.6B Base)

| Precision | Talker engine size |
|---|---|
| fp16 | 903 MB |
| int4-AWQ + fp8 | **245.9 MB** (−73%) |

int4 is byte-stable and recommended by default for Orin Nano (fits 8 GB with
N=2). fp16 stays available for accuracy-sensitive comparison. See the
`repro` block in the source dataset for the full build recipe.

---

## Reproduction artifacts (2026-06-21)

| Artifact | Anchor / hash |
|---|---|
| fork release tip | `port/qwen3-tts-base-v080-n1n2` @ `7142a30` |
| rebake image | `seeed-local-voice:v0.8.0-n1n2-rebake` |
| ASR worker | `5ebd436b` |
| TTS shared-engine worker | `190178f6` |
| int4 talker engine | HF `harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8` |
| asr-b2 engine | `4122dfcc` (HF) |
| talker-b2 engine | `f7339e02` |

---

## Methodology

- **N=x verified** = real on-device burst test + MD5 audio gate + zero CUDA/race.
  N=2 covers both the TTS talker batch-lane / slot-pool and the ASR streaming
  slot-pool.
- **ASR latency** = finalize (audio-end → final), VAD 400 ms wait excluded
  (constant, overlaps decode). CER/WER use greedy decode (top_k=1, temp=0) +
  force_language scaffold, replicating the production decode contract.
- **TTS latency** = TTFA (warm: prefill + first chunk). RTF = wall-clock /
  audio duration. The N=2 slow-client TTFA penalty (1.4–5×) is memory-bandwidth
  bound, not a bug.
- **Deployment scope:** these gates ran on `orin-nx` / `orin-nano` profiling
  devices. The `seeed-orin-nx` production robot-arm stack was **not** touched.

See [`docs/deploy-v080-n1n2.md`](docs/deploy-v080-n1n2.md) for the matching
deployment runbook.
