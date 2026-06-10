# v080-0015 — Multi-chunk (K>1) streaming-audio transcript parity on Orin NX

**Date:** 2026-06-10 · **Device:** orin-nx (`orinnx`, Linux aarch64, JetPack 6.2 / TRT 10.3,
15656 MB unified) · **edgellm:** 0.8.0 · **Branch:** `feat/edgellm-v080-migration`

**Goal (FINAL acceptance of the v0.8.0 ASR migration):** verify that feeding **real audio
in MULTIPLE mel chunks** (KV accumulating across chunk boundaries — the true production
streaming scenario) produces the **same transcript** as the one-shot / K=1 path. So far
only K=1 (single chunk through the streaming wrapper == one-shot) had been validated.

## VERDICT

**Multi-chunk (K>1) streaming-audio DECODE is transcript-correct — PASS.** With
encoder-VALID chunks, K=2 (KV accumulating across the audio-chunk boundary) produces a
**byte-identical** transcript to one-shot K=1.

The real-corpus K>1 runs (zh_long_01/02/03 split into K=2/3/5) are **gated by the audio
encoder's TRT optimization-profile window**, NOT by any decode bug. The encoder requires a
windowed feature dim **≥ 10 blocks (≈ ≥ 9 s of audio) per `encodeAudioChunk` call**;
splitting a 10–15 s WAV into smaller mel chunks puts each chunk below that floor and
`setInputShape` refuses it (`satisfyProfile`). This is an **encoder-feeding engineering
item**, not a multi-chunk decode bug.

- No decode bug found → no patch in this commit (acceptance doc only).
- Recommendation: an **accumulating-window re-encode** (or a wider encoder opt profile with
  min-window=1) is required to make sub-9 s streaming chunks viable. See §4.

---

## 0. Host identity

```
$ uname -srm; hostname
Linux 5.15.148-tegra aarch64
orinnx
```

## 1. Per-WAV per-K transcript + CER (raw)

Driver: `spike_v080_m6_audio_streaming --mel <full.safetensors> --chunks K
--llm .../engines-v080/llm --audio .../engines-v080/audio`. For K>1 the full mel is split
into K contiguous time windows (`<mel>.c0..c{K-1}`, even split). Goldens are the v0.7.1 ==
v080-0012/0013 one-shot transcripts. CER via the offline harness `normalize_text`/`cer`
(`qwen3-asr-hlm-validate/validate_qwen3_asr_hlm_offline.py`).

Mel lengths: zh_long_01 T=1038 (≈10.4 s) · zh_long_02 T=1386 (≈13.9 s) · zh_long_03 T=1542 (≈15.4 s).

| WAV | K | status | CER vs golden | CER vs K=1 | match K=1 |
|---|---|---|---|---|---|
| zh_long_01 | 1 | OK | **0.0000** | 0.0000 | true |
| zh_long_01 | 2 | ENCODE_FAIL (W=6 < 10) | — | — | — |
| zh_long_01 | 3 | ENCODE_FAIL | — | — | — |
| zh_long_01 | 5 | ENCODE_FAIL | — | — | — |
| zh_long_02 | 1 | OK | **0.0000** | 0.0000 | true |
| zh_long_02 | 2/3/5 | ENCODE_FAIL | — | — | — |
| zh_long_03 | 1 | OK | **0.0000** | 0.0000 | true |
| zh_long_03 | 2 | ENCODE_FAIL (W=8 < 10) | — | — | — |
| zh_long_03 | 3/5 | ENCODE_FAIL | — | — | — |

K=1 transcripts (raw, == v0.7.1 golden, CER 0):
```
zh_long_01  language Chinese这并不是告别，这是一个篇章的结束，也是新篇章的开始。
zh_long_02  language Chinese桥下垂直净空十五米。该项目于二零一一年八月完工，但直到二零一七年三月才开始通车。
zh_long_03  language Chinese适当使用博客可以使学生变得更善于分析和进行思辨。通过积极回忆网络材料，学生们可以在他人的文章的上下文语境中找到自己的立场，并能够针对特定问题提出自己的观点。Work 二零零二。
```

## 2. The multi-chunk DECODE-correctness test (encoder-valid chunks) — the real gate

The real corpus can't be split into ≥2 encoder-valid chunks (each WAV's *full* window is
only W=10–15, so any split drops chunks below W=10). To isolate the **decode path** (KV
accumulation + MRope continuity + audio_pad boundaries across audio chunks) from the
encoder-window refusal, a 2076-frame mel was built by self-concatenating zh_long_01
(`zh_dup2`, T=2076), then run at K=1 (single 2076-frame chunk) vs K=2 (two 1038-frame
chunks, **each W=10 → encoder-valid**, KV accumulating across the boundary):

```
zh_dup2 K=1  total audio tokens = 270 (single chunk)
=== TRANSCRIPT (chunks=1) ===
language Chinese这并不是告别，这是一个篇章的结束，也是新篇章的开始。这并不是告别，这是一个篇章的结束，也是新篇章的开始。

zh_dup2 K=2  chunk0 audioTokens=135 + chunk1 audioTokens=135 = 270 (KV accumulating)
=== TRANSCRIPT (chunks=2) ===
language Chinese这并不是告别，这是一个篇章的结束，也是新篇章的开始。这并不是告别，这是一个篇章的结束，也是新篇章的开始。
```

**K=2 transcript == K=1 transcript, byte-identical (CER 0.0000).** This proves multi-chunk
AUDIO decode is correct: KV accumulates correctly across the audio-chunk boundary, MRope
positions stay continuous for audio chunks, and the audio_pad token bindings are right.
The remaining gap from v080-0012 (K=1 only) is closed for the decode path.

## 3. Encoder-window constraint — empirical characterization

`encodeAudioChunk` binds the mel as encoder input `padded_feature` with shape
`[W, 128, 100]`, where W = number of windowed feature blocks (≈ ceil over ~100 mel frames
each). The audio encoder engine's opt profile is **`[10,128,100] .. [30,128,100]`** →
`W ∈ [10, 30]`. Below W=10, `setInputShape` fails:

```
[ERROR] [TensorRT] IExecutionContext::setInputShape: Error Code 3: API Usage Error
(Parameter check failed, condition: satisfyProfile. Set dimension [6,128,100] for tensor
padded_feature does not satisfy any optimization profiles.
Valid range for profile 0: [10,128,100]..[30,128,100].)
[ERROR] [audioRunner.cpp:420:encodeOneAudioImpl] Failed to set padded features input shape
probeAudioTokens: encodeAudioChunk failed for .../zh_long_01.safetensors.c0
```

(zh_long_01 K=2 → 519-frame chunks → W=6; zh_long_03 K=2 → 771-frame chunks → W=8 — both < 10.)

**Minimum viable streaming chunk size (single-chunk probe sweep on a zh_long_03 mel slice):**

| mel frames | W attempted | result |
|---|---|---|
| 1050 | 11 | OK (137 audio tokens) |
| 1000 | 11 | OK (130) |
| 950  | 10 | OK (124) |
| 940  | 10 | OK (122) |
| 910  | 10 | OK (119) |
| 905  | 10 | OK (118) |
| **901** | **10** | **OK (118) ← floor** |
| 900  | 9 | **FAIL** (`[9,128,100]` < min 10) |

→ **Minimum viable chunk = 901 mel frames ≈ 9.01 s of audio** (901 × 10 ms hop). 900 frames
(W=9) is the first refusal. The encoder cannot encode a window smaller than ~9 s.

### Why every real-corpus K>1 split fails
Each WAV's *whole* utterance is only W=10–15 (zh_long_01 W=10, zh_long_02 W≈13, zh_long_03
W≈15). Any contiguous split therefore drops at least one chunk below W=10. There is no K>1
contiguous split of a 10–15 s WAV that keeps every chunk ≥ 9 s.

## 4. Recommendation: accumulating-window vs wider profile

This is an **encoder-feeding engineering item**, not a decode bug. Two options:

- **(a) Accumulating-window re-encode (recommended for the current engine).** Each streaming
  step re-encodes the *growing* audio prefix (always ≥ 9 s once enough has arrived), and the
  decode hook consumes only the *newly produced* audio tokens. Pros: works with the existing
  encoder engine, no rebuild. Cons: O(K) re-encode cost on the full window each step;
  requires the runtime to diff audio-token deltas across re-encodes and not double-bind the
  overlapping pads. The v080-0012 chunked driver assumes **disjoint, non-overlapping** chunks
  (each chunk's pads appended additively), so accumulating-window needs new runtime logic to
  emit only the incremental pads — it cannot be tested with the current `appendChunk` contract.
- **(b) Rebuild the encoder with a wider opt profile (min-window = 1).** Lets `encodeAudioChunk`
  accept short windows directly → true low-latency disjoint streaming chunks. Pros: clean
  disjoint streaming, lowest per-step cost. Cons: requires an encoder re-export/rebuild, and
  short windows may shift encoder accuracy (the model was trained/profiled for ≥ ~9 s windows —
  needs an accuracy re-check after rebuild). **Not rebuilt in this task** per scope.

For an edge voice assistant where the production path already accumulates a full utterance
before ASR finalize, **(a) accumulating-window** is the lower-risk path; **(b)** is only
worth it if true sub-9 s incremental decode latency becomes a product requirement.

---

## Evidence artifacts (device: `~/asr_v080_e2e/v080_0015/`)
- `m6_zh_long_0{1,2,3}_k{1,2,3,5}.log` — per-WAV per-K runs (K=1 OK + transcript; K>1 ENCODE_FAIL).
- `m6_zh_dup2_k{1,2}.log` — encoder-valid multi-chunk decode-correctness test (K=2 == K=1).
- `floor_{900..1050}.log` — encoder-window floor sweep (W=10 floor = 901 frames).
- `run_multichunk.sh`, `chunk_mel.py`, `probe_floor.sh`, `probe_threshold.py`, `make_long2.py`,
  `score.py` — drivers (score.py reuses the offline harness `cer`/`normalize_text`).

## Reproduce
```
cd ~/asr_v080_e2e/v080_0015
bash run_multichunk.sh zh_long_01 1     # K=1 reference == golden
bash run_multichunk.sh zh_long_01 2     # K=2 -> ENCODE_FAIL (W=6 < 10)
python3 make_long2.py && bash run_multichunk.sh zh_dup2 1 && bash run_multichunk.sh zh_dup2 2
bash probe_floor.sh 901 900              # W=10 floor: 901 OK, 900 FAIL
python3 score.py zh_long_01,zh_long_02,zh_long_03 1,2,3,5
```
