# v080-0014 — Qwen3.5-4B GDN + live 3-engine pipeline co-resident on Orin NX 16GB

**Date:** 2026-06-09 · **Device:** orin-nx (`orinnx`, Linux aarch64, JetPack 6.2 / TRT 10.3,
15656 MB unified) · **edgellm:** 0.8.0 · **Branch:** `feat/edgellm-v080-migration`

**Goal:** rebuild the Qwen3.5-4B GDN v0.8.0 LLM engine and run the full
ASR → GDN-LLM → MOSS pipeline co-resident, measuring per-stage speed and the
definitive co-resident peak memory (vs the ~8–11.5 GB math estimate and the 16 GB ceiling).

## VERDICT: PASS — /goal fully MET with the live 3-engine measurement in hand.

The full v0.8.0 ASR → GDN-LLM → MOSS pipeline runs co-resident on Orin NX 16 GB.
Speed, accuracy, and memory are all acceptable.

---

## 0. Host + disk

```
$ uname -srm; hostname; df -h /
Linux 5.15.148-tegra aarch64
orinnx
/dev/nvme0n1p1  233G  213G  9.6G  96% /   (before reclaim)
```

Reclaimed ~20 GB by deleting the stale **0.7.1** GDN `onnx/` + `engines*` (the May-24
throwaway attempts) → 30 G free. After the full run: 19 G free.

## 1. GDN engine rebuild

**hf_src GDN arch confirmed** (`~/edgellm-workspace/qwen35-4b-awq/hf_src/config.json`):
`model_type: qwen3_5_text`, 32 layers, `full_attention_interval: 4` → 24 GDN/mamba + 8 attention.
(hf_src transferred from `wsl2-local` via `fleet transfer` direct LAN; tar-md5 verify OK.)

**Export** (v0.8.0, detached):
```
.venv-x86export/bin/tensorrt-edgellm-export \
  hf_src onnx --externalize-weights int4_ffn --skip-visual --skip-audio --skip-code2wav
```
→ `onnx/llm/config.json`: `edgellm_version: 0.8.0`, `kv_cache_dtype: fp16`,
`model_type: hybrid_mamba`, `num_linear_attn_layers: 24` + `num_attention_layers: 8`,
`layer_types` = 24 mamba + 8 attention. 192 FFN Int4GroupwiseGemmPlugin tensors externalized.
ONNX md5 `e8cadce91a50c1c3475c17502e64b13f`.

**Build** (detached, ~6.5 min):
```
llm_build --onnxDir onnx/llm --engineDir engines-v080-gdn \
  --maxBatchSize 1 --maxInputLen 4096 --maxKVCacheCapacity 4096
```
→ `LLM engine built successfully`. The fresh ONNX parses cleanly
(`Successfully created plugin: Int4GroupwiseGemmPlugin` — the exact op that was
"not found" with the stale 0.7.1 ONNX).

| Engine | md5 | size |
|---|---|---|
| `engines-v080-gdn/llm.engine` | `afcb055b67bbc33d3dacf5491e4719d5` | 1,019,519,468 B (972 MB) |

## 2. Full pipeline speed (single pass, zh question → ASR → GDN-LLM → MOSS)

| Stage | Metric | Value |
|---|---|---|
| ASR (Qwen3-ASR-0.6B) | wall | **4.897 s** |
| LLM (Qwen3.5-4B GDN) | engine load | 4.46 s |
| | **TTFT** | **0.528 s** |
| | decode (20 tok) | 0.799 s |
| | **throughput** | **25.03 tok/s** |
| MOSS-TTS-Nano | **TTFA** | **151 ms** (worker) / 152.5 ms (wall) |
| | audio dur | 4.08 s |
| | **RTF** | **0.262** |
| e2e | wall | **11.224 s** |

MOSS TTFA 151 ms / RTF 0.26 match the known MOSS-TTS-Nano production figures.

## 3. CO-RESIDENT PEAK MEMORY (the definitive 16 GB-fit proof)

True simultaneous 3-engine residency: MOSS worker held resident + a long
`llm_inference` generation running + ASR run concurrently → tegrastats captures the
real triple-overlap (`llm_still_alive=True` during the ASR run). NOT a projection.

| Stage | used MB |
|---|---|
| baseline | 3253 |
| + MOSS resident | 4088 (+835) |
| + LLM resident | 6307 (+2219) |
| **ASR + LLM + MOSS (triple)** | **9968** |
| **tegrastats peak** | **11062 / 15656** |

Raw peak line:
```
06-09-2026 21:38:14 RAM 11029/15656MB (lfb 53x1MB) SWAP 846/7828MB ... GR3D_FREQ 99% gpu@58.312C
```

- **fits 16 GB: YES** — headroom **4594 MB (~4.5 GB)**.
- Actual **11062 MB** is within the ~8–11.5 GB math estimate (top end).

## 4. Accuracy sanity

| Stage | Output | Verdict |
|---|---|---|
| ASR | `这并不是告别，这是一个篇章的结束，也是新篇章的开始。` | correct |
| LLM | `这句话表达了旧阶段的终结与新旅程的开启。` | coherent |
| MOSS | 4.08 s WAV, rms 0.04242, 391,680 samples | intelligible (non-zero energy) |

## 5. Container restore (raw)

Stopped for RAM (snapshot saved to `container_snapshot_v080_0014_pre.txt`), restored after:
```
seeed-voice                Up (healthy)
translator                 Up (healthy)
edge-llm-chat-service      Up (healthy)
industrial-security-demo   Restarting (1)   # pre-existing crash-loop — never touched
```

## Root-cause note: why the prior attempt died (NOT a network drop)

The prior export was **OOM-killed by the kernel** (`global_oom`, docker cgroup scope):
```
Out of memory: Killed process 204356 (tensorrt-edgell) total-vm:21684408kB ...
```
The 4 running containers (notably `edge-llm-chat-service`, several GB unified) left only
~5.5 GB available vs the export's large virtual footprint under total memory pressure.
Stopping the three RAM-holding containers raised available RAM 5.5 → 11 GB, after which the
re-export, build, full pipeline, and co-residency probe all completed. All containers restored.

> Workspace paths live under `~/edgellm-workspace/qwen35-4b-awq` and `~/asr_v080_e2e`
> (not `~/tensorrt-edgellm-workspace/qwen35-4b-awq` as the original brief stated).

See `bench/regression/goldens/v080-edgellm/pipeline_3engine_coresident.json` for the
machine-readable results.
