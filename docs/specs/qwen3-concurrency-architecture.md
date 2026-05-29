# Qwen3 ASR + TTS 并发流式架构定稿（Phase C）

**状态**: 决策已定，进入 Phase D 实施
**目标**: qwen3 ASR + qwen3 TTS(Custom) 并发流式，内存尽可能小、性能尽可能好
**设备**: Orin Nano 8GB（可用 ~6.5GB）+ Orin NX 16GB
**TensorRT 分支**: `v071/customvoice-product`

> 本文档取代 `qwen3-asr-tts-batch-concurrency.md` 中"ASR 走官方 batch"的结论。Phase A→D 深度调查证明 ASR 异步 session 不适配官方 static batch（见 §2.1），改用 slot-pool。

---

## 1. 首要原则：改动尽量产品侧，保 EdgeLLM fork 干净

每次拉 NVIDIA 上游，**改过的上游文件（M）= merge 冲突债**。因此：

- **并发/调度逻辑放产品侧**：worker（`qwen3-edgellm-jetson/native/`）+ OVS 服务层（`seeed-local-voice`）
- **fork 改动只允许 minimal + additive**：新增方法/构造，不改上游算法逻辑，旧 API 保留为 wrapper
- **能上游的通用原语走 PR**（如批量 prefill API），让 NVIDIA 维护

---

## 2. 三栈最终架构（各有依据，不强求统一）

| 栈 | 并发机制 | 依据 | fork 改动 |
|---|---|---|---|
| **qwen3 ASR** | slot-pool（共享 engine + C-optimized 单 encoder context + per-slot decoder context）| 异步稀疏长命 session 不适配 static batch | minimal：shared-engine 构造路径（additive）|
| **qwen3 TTS** | 官方 cohort batch + 外层 micro-batcher | 已有批量路径 + 大模型省权重 + request-scoped 适配 cohort | 复用已做的 batched prefill（additive）|
| **MOSS TTS** | slot-pool（现状不动）| 已工作 + 零债 + 0.1B 小模型 batch 收益弱 + 无批量路径 | 无 |

### 2.1 为什么 ASR 不能用官方 batch（核心结论）

官方 batch 是 **static/dense** 模型：
- `HybridCacheManager` 无单 lane 操作，KV slot 死绑 batchIdx（`resetForNewSequences` 只 copy 长度元数据不搬 KV 实体，`hybridCacheManager.cpp:247-273` vs compactBatch `:325-337`）
- batched append 要求 N 个 lane 全有非空 chunk，不支持稀疏 ready-subset

ASR 是**异步稀疏长命 session**（N 个 session 异步到达/离开、各自 KV 跨 hop 持续、每 hop 只有部分 session 有 ready chunk）→ 本质需要 continuous batching，官方不提供，自建要新增 4 类 API（单 lane KV evict/reset、MRope 单 lane reset、per-lane begin/endAsrSession、稀疏参与 append/decode）= 高难度。

**slot-pool 用 N 个独立 context 规避此问题**：每 session 独立生命周期，天然匹配异步。

### 2.2 为什么 slot-pool 内存不输 batch（C-optimized）

实测 encoder 利用率 N=4 仅 16%，故 encoder 走**单 context 串行共享**，只 decoder per-slot：

| 方案 | N=4 内存 |
|---|---|
| slot-pool 朴素（每 slot 独立 encoder context）| ~3571MB |
| **slot-pool C-optimized（共享 engine + 单 encoder context + per-slot 小 decoder context）** | **~2155MB** |
| 官方 cohort batch | ~2131MB |

C-optimized slot-pool 内存 ≈ batch，且匹配异步、规避 4 类新 API。

### 2.3 为什么 TTS 用 cohort batch（而非 slot-pool）

- `qwen3OmniTTSRuntime` **已有**批量路径 `runTalkerGenerationLoop(activeBatchSize, vector<PerBatchTalkerState>)`（`:1405,:1791`）
- qwen3 是大模型（1507MB 权重），batch 共享权重省内存收益大
- TTS request-scoped（一段文本→一段音频，起止成对），适配 cohort

**但 cohort batch 不是免费**（codex 第二意见）：
- TTS batch 入口只接收已组好的 vector，**无运行时队列/补位** → 需外层 micro-batcher 组 cohort（`:1405`）
- 共享 workspace 并发不安全，需外层串行（`qwen3OmniTTSRuntime.h:655`）
- 采样参数取 `requests[0]` 套全 batch → 异构请求需同参或走 batch=1 fallback（`:1427`）
- 混批空转：finished lane idle until all done → 长短文本混批短文本空转（`:1819,:1975`）

→ TTS 方案 = batch + **外层 bounded micro-batcher**：极短窗口攒同参 cohort，超时/异构走 batch=1 fallback（~100 行 Python/worker 调度）

### 2.4 为什么 MOSS 保持 slot-pool

MOSS 是 TTS，cohort batch 概念适用，但：MOSS runtime（`mossTtsNanoRuntime`）从头就是 slot-pool，**无批量路径**（要新写）；已 N=2 验证、零债；0.1B 小模型 batch 省权重收益弱；slot-pool 本就共享 engine（权重不重复）。**改 = 返工换微小收益，不做。**

---

## 3. 分层归属

```
OVS 服务层 (seeed-local-voice)
  ├── ASR: admission(N slot) + session→slot 映射 + 背压(429)
  └── TTS: micro-batcher(cohort 组装 + 同参约束 + batch=1 fallback)
        │
产品 worker (qwen3-edgellm-jetson/native/edgellm_voice_worker)
  ├── qwen3_asr_worker.cpp: N runtime 实例(共享 engine) + session 路由   [slot-pool]
  └── qwen3_tts_worker.cpp: 接收 cohort vector → 批量 prefill+generate    [batch]
        │ #include (build 时, 只读消费)
        ▼
EdgeLLM fork (TensorRT-Edge-LLM, v071/customvoice-product)
  ├── [additive] shared-engine 构造路径 (ASR slot-pool 共享权重用)
  ├── [additive] appendPrefillEmbedsBatched (TTS cohort 用; 已实现)
  └── [不改] llmEngineRunner / HybridCacheManager / streaming / 上游算法
```

---

## 4. fork 改动清单（minimal additive，逐项评债）

| 改动 | 类型 | 债 | 可上游 |
|---|---|---|---|
| shared-engine 构造路径（runtime 接受预加载 ICudaEngine）| additive 构造重载 | 低（kUSER_MANAGED 已预留共享语义）| 是（多 context 共享 engine 是合理能力）|
| `appendPrefillEmbedsBatched` 单 context-N（已实现）| additive 方法 + N=1 wrapper | 低 | 是（批量 prefill API）|
| MRope `initializeMRopeForSession` 传 activeBatchSize（已实现）| additive 默认参 | 极低 | 是 |

**禁止触碰**：`llmEngineRunner.{cpp,h}`、`hybridCacheManager.{cpp,h}`、`streaming.{h,cpp}` 的上游算法逻辑。

---

## 5. Phase D 实施顺序

1. **D-1 ASR slot-pool**（先做，匹配异步、MOSS 模式验证过）
   - fork: shared-engine 构造路径（additive）
   - worker: `qwen3_asr_worker.cpp` 从单 AsrSessionState → N 实例池 + session 路由
   - OVS: ASR admission(N) + 背压
   - engine: 无需重 build（slot-pool 用 max_batch_size=1 engine 即可，N 个实例各自单批）
   - 验收: CER 一致 + N=4 30min 0 error + 内存 ≤ 2.2GB + 异步 TTFA

2. **D-2 TTS cohort batch**（复用已做的 batched prefill）
   - fork: 已有 `appendPrefillEmbedsBatched`；TTS 收口走批量 prefill 路径
   - worker/OVS: micro-batcher（cohort 组装 + 同参 + batch=1 fallback + workspace 串行保护）
   - engine: TTS talker 重 build max_batch_size=N
   - 验收: 音质一致 + cohort TTFA + 混批退化可控 + 取消仍 per-request

3. **D-3 收尾**: profile 字段 + runbook + 技术债抽取（Phase B 并入）

---

## 6. 关键风险

| 风险 | 缓解 |
|---|---|
| ASR slot-pool N 实例共享 engine 的 context memory 管理 | kUSER_MANAGED + setContextMemory 已预留；验证多 context 共享不串数据 |
| ASR 单 encoder context 串行成瓶颈 | 实测 N=4 利用率 16%，安全；break-even 在 N≈25 |
| TTS cohort 混批空转 | micro-batcher 同长度/同参分组；超时 batch=1 fallback |
| TTS 共享 workspace 并发不安全 | micro-batcher 外层串行单 runtime；或 per-cohort 串行执行 |
| Orin Nano 8GB ASR+TTS 同时并发 | 分别 N=2 试，实测峰值；必要时 ASR N=2 + TTS N=2 |
