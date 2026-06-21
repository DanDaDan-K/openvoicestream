# VoxEdge v0.8.0 整合 — 执行入口 (INDEX)

> 给**执行者**的总入口。本目录是 VoxEdge 边缘语音栈统一到 TensorRT-Edge-LLM **v0.8.0** + 开源传播的完整规划。
> 经 3 轮 codex 独立审核 + 真机实证 + 多 agent 调研收敛。最后更新 **2026-06-21**。

---

## 0. 怎么读(顺序)

| # | 文档 | 作用 | 谁先读 |
|---|---|---|---|
| 1 | **本 INDEX** | 状态 / 决策 / 环境 / 从哪开始 | 所有人 |
| 2 | [consolidation-plan.md](consolidation-plan.md) | 主计划:现状图 A、目标架构 B(+B.1 五仓)、工作流 C0–C7 / D / E / F(测试 gate)/ G(清理)/ H(回归策略)+ dispatch 护栏 | 执行者必读 |
| 3 | [c0-patch-ownership-manifest.md](c0-patch-ownership-manifest.md) | **迁移 day-1 依据**:20 项 runtime/patch 归属(C1/C2/C3)+ 双写风险 + 源vs封装边界 | 动 fork/jve 前必读 |
| 4 | [benchmarks-dataset.md](benchmarks-dataset.md) | 性能/能力唯一真相(表A性能 + 表B最佳实践 + GAPS + repro 元数据) | 写 README/BENCHMARKS 前 |
| 5 | [positioning-and-propagation.md](positioning-and-propagation.md) | 对外定位/分层/传播/license 矩阵 | 做发布/文档时 |
| 6 | [competitor-research.md](competitor-research.md) | 10 个同类开源项目传播复盘 | 做发布时 |
| 7 | [code-structure/](code-structure/) | 五仓 AST 结构分析(在正确分支 @ ref 上做的) | 需要代码定位时 |

**角色分工(沿用 CTO/spec/executor 三段式):** 主线程定 spec 边界 + 串联 + 自验关键数字;codex 出设计/审核(只读带 file:line);general-purpose 照 spec 实施(build/deploy/test)。**执行体 prompt 必带护栏**(见 plan「DISPATCH NOTES」+ 「Preflight 防误触」)。

---

## 1. 当前状态(2026-06-21)

### ✅ 已完成(执行者不要重做)
- **数据丢失隐患全清零:**
  - `tensorrt-edge-llm`(小写副本)的 4 个生产关键 commit(SlotPool 抽取 / ASR-assistant prefix / N-instance slot-pool worker / shared-engine ctor)→ 已 push 到 `suharvest/v071/customvoice-product`(FF `1668470..893ba2a`)。
  - int4 export driver 分支:`wip/native-int4-talker`(ff2318e)+ `wip/asr-int4-decoder`(c80bcc0)**都在 suharvest fork**;未提交 working 文件(`quantize_talker_stage1_bigcalib.py` + `cv-int4-derisk/*`)已拉回 Mac。
- **全仓本地备份**:`~/project-backups/20260621-133543/`(6.6GB,97 bundle 含全部未推送 + 35 diff + untracked 内容 + 本规划文档 + wsl2-recovered/ 的 int4 drivers)。
- **stateful code2wav × N=2** 真机+源码定论:flag 是 no-op(没接线,非不安全),收益增量,**backlog**(详见 benchmarks-dataset GAPS#10 + c0 manifest W2 旁注)。
- **模型 license 调研完成**(见 positioning §六 license 矩阵)。
- 规划经 3 轮 codex 审 + must-fix 全闭合。

### ⏳ 待执行(本计划的工作流,尚未开工)
- C0 manifest 已产出(本目录);**C1–C7 / D / E / F / G / H 全部待执行**。
- 两个外部确认:核 MOSS-TTS-Nano 仓库根 LICENSE 文件是否落地 Apache 全文;(CV license 已定=跟随 Qwen 官方不改)。

---

## 2. 决策记录(都已拍板,执行者按此办,勿再议)

| 决策 | 结论 | 依据 |
|---|---|---|
| 最终仓库结构 | 3 层 **5 职责单元 / 6 物理仓**(RK=2 仓算 1 后端);对外只一个品牌 VoxEdge | plan B.1 |
| fork vs jetson-voice-engine | 三类变更模型:C1 上游bug + C2 本地runtime扩展 → **源在 fork**;C3 overlay/recipes → jve。**任何 runtime 改动只在 fork 落地,jve patch 自动生成** | plan B.1 / c0 manifest |
| int4/fp8 export drivers 归属 | **`jetson-voice-engine/recipes/`**(不放 fork),pin fork commit,只调 fork export API | plan C6 / c0 |
| CustomVoice | **一等公民**(用户定):走 int4,移植 P5 9-row 条件到 v0.8.0,**DROP W1 w8a16**(EOS 破不可行);过 F4+F5-CV 后可进默认 | plan C1b/B / c0 §4 |
| stateful code2wav N>1 | **backlog**(没接线,收益增量,RTF 已<1 性价比低) | benchmarks GAPS#10 |
| 默认精度翻转 | **C1 只加 opt-in,C1b 才翻默认**,且挂 F-gate 全绿;CustomVoice 默认前过 F4+F5-CV | plan C1/C1b |
| 开源 license | 主体 Apache/MIT 可开+商用;**NLLB=CC-BY-NC 剥成可选非商用**;FunASR(Paraformer/SenseVoice)需署名+附协议;**CV 跟随 Qwen 官方不加料**;MOSS 待核 LICENSE | positioning §六 |
| N≤0 prefill guard | **尚未落地**(分支里只有 empty-fullText LOG_ERROR),需新写并提交 fork(C1) | plan C3(a) / c0 §4 |
| stateful 等性能优化定性 | B2(M=1 GEMV)/A2(streaming worker)= C2(我们路线,可日后 PR) | c0 §4 |
| qwen3asr_rk | 外部库(非我们),**移出整合范围,不删不并** | plan G |

---

## 3. 环境与访问(执行者必须遵守)

### 设备(用 fleet,完整路径见下)
- **可用(profiling,可 stop/测)**:`orin-nano`(v0.8.0 build 树 `~/project/edgellm-v080-build`,ENABLE_CUTE_DSL=OFF)、`orin-nx`、`wsl2-local`(RTX,有 modelopt,做 int4/fp8 导出;flap-prone,跑长任务先 mask 定时器+tmux)。
- **🚫 绝对不碰**:`seeed-orin-nx`(生产机械臂栈)+ 生产 `speech-models` 卷。测试一律用隔离卷 `~/comp-e2e-models`。
- Fleet 全路径(子 shell alias 不生效):`uv run --project ~/project/_hub python ~/project/_hub/fleet.py`。**flags(--sudo/--timeout/--json)放设备名之前**;`--sudo` 用于 apt/systemctl/docker/写 $HOME 外。坑:zsh 里别把 fleet 路径塞进带空格的变量(词分裂)。

### fork remote(git push 高危,看清楚)
- fork 工作区里 **`origin` = NVIDIA 官方仓**!推到我们 fork 必须用 **`suharvest`**(或 `origin-claude`,同 URL)。**绝不 `git push origin`**。
- canonical fork = `https://github.com/suharvest/TensorRT-Edge-LLM.git`。v0.8.0 主分支 `port/qwen3-tts-base-v080`;int4 drivers `wip/native-int4-talker` + `wip/asr-int4-decoder`;CV/v071 在 `v071/customvoice-product`。

### HF 预编译产物(终端用户消费的东西)
`harvestsu/qwen3-asr-0.6b-int4-v080`、`harvestsu/qwen3-tts-0.6b-base-jetson-trtllm-int4fp8`、`harvestsu/qwen3-tts-0.6b-customvoice-jetson-trtllm-int4fp8`。

### 备份与已恢复资产
- 全仓备份:`~/project-backups/20260621-133543/`(bundles/ + uncommitted/ + untracked-content/ + scratch/ + planning-docs/ + wsl2-recovered/)。
- 恢复方式见该目录 `MANIFEST.txt`。

---

## 4. 从哪开始(day-1,无破坏性)

1. **建 C0 patch 清单的可执行版**:照 [c0-patch-ownership-manifest.md](c0-patch-ownership-manifest.md),逐个 v0.7.1 patch 对 v0.8.0 base 跑 `git apply --check`,确认哪些已被上游吸收(归零)、哪些需保留。
2. **C6 收尾(recovery 已做)**:把已恢复的 int4 drivers 落进 `jetson-voice-engine/recipes/`,写 recipes README + pin fork commit。
3. 之后按 plan 执行顺序:**C(统一 v0.8.0)→ D(一键部署)→ E(README/BENCHMARKS/Recipes)**,每步过对应 F-gate(H 节回归策略是硬约束:F 是各 workstream 的 exit-criteria,不是收尾)。

**所有 build/deploy/test 派发必带**:EVIDENCE 段(md5 + 原始验证输出 + before/after + `docker logs|grep -iE error|crash|fail`)、Fleet 规则块、Preflight 防误触、禁破坏性操作(见 plan「DISPATCH NOTES」)。
