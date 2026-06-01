# OpenVoiceStream Agent — Debug Handoff (2026-05-18)

> 接力人请直接从 **§3 当前未解决问题** 开始读。前面是 context。

本次实机调试入口：**Mac 本机 agent + Orin (`orin-nx`, Tailscale IP 100.82.225.102) 上的 SLV + edge-llm-chat-service**。
进入这个 session 前请先扫一眼 `HANDOFF.md`（老的总览）+ `MEMORY.md`（项目记忆，特别是 OVS / edge-llm / Jetson 部分）。

---

## 1. 已部署修复（已通过测试 / 已在线）

按时间顺序，全部已合入 `main` 分支（本地未 push 远端）。

### 1.1 Agent (`seeed-local-voice/agent/`)

| # | 修复 | 关键文件:行 | 测试 |
|---|---|---|---|
| A | **barge-in discard latch** — SLV barge-in 后还会继续推 ~几百 ms TTS 余音，audio_io 把它们当新 chunk 播 + dispatch 翻回 SPEAKING；加 `_discard_playback` flag，`stop_playback()` 锁，`arm_for_next_turn()` 在新 ASRFinal / wake / dashboard send_text 时解锁 | `audio_io.py:48-58,164-178,187-202`<br>`app_base.py:607-627,734-741,869-873`<br>`plugins/debug_dashboard.py:_api_send_text,_api_reconnect` | `tests/test_bargein_discard_latch.py` (5) |
| B | **TTSDone in BARGED_IN 不降回 IDLE** — 否则 PTT 模式还会启动 auto-sleep 计时器 | `app_base.py:909-919` | 同上 |
| C | **SLV WS 自动重连** — SLV 在每轮 asr_eos 后会关 WS，老代码靠 `ASRFinal session_complete=True` 触发 reconnect；如果 SLV 没回 final（空/丢），dispatch loop `events()` 自然返回 → 任务退出 → 后续所有 asr_eos 打到死 WS（"send_json: WS closed mid-send" 刷屏）。改成 dispatch 外层 while 循环，reader 死了就 reconnect + 重新拿 events 迭代器，带指数退避 | `app_base.py:_slv_dispatch (675~)`<br>`slv_client.py:_send_json/send_audio: ConnectionClosed → self._ws=None` | (no new test, 实机验证) |
| D | **低信号 ASR 过滤** — 单字、纯感叹/填充词（"嗯"、"啊"、"you"、"uh" 等 20+ 个）的 ASRFinal 直接丢弃不进 LLM；标点/空白自动剥离再判断 | `app_base.py:76-105 (_INTERJECTIONS,_strip_for_signal)`,`827-862` | `tests/test_low_signal_filter.py` (16) |
| E | **echo auto-recover** — 连续 3 条 ≤40 字的 assistant 输出完全相同 → 自动清 history + 清 cache_warmed + 清 prefix_cache_disabled，event bus 发 `on_echo_recovery` | `session.py:add_assistant + _maybe_recover_from_echo` | `tests/test_echo_recovery.py` (6) |
| F | **mic_pump 不阻塞** — `send_audio` 套 0.5s 超时避免 SLV reconnect 期间 mic 输入端堆满 | `app_base.py:476-505 (_send_audio_nonblocking)` | `tests/test_send_audio_nonblocking.py` (3) |
| G | **/api/session/clear** — 手动清 history 的 endpoint，dashboard 备用 | `plugins/debug_dashboard.py:_api_session_clear` | — |
| H | **VAD-driven barge-in 加日志** — 之前只有 ASRPartial 路径有 `BARGE-IN fired`，VAD 路径静默没法诊断 | `app_base.py:608` | — |

**测试**: `uv run pytest tests/ --ignore=tests/e2e/test_*` ⇒ **227 passed, 1 skipped**。

### 1.2 edge-llm-chat-service (wrapper, 已部署 Orin 上)

| # | 修复 | 关键文件:行 |
|---|---|---|
| I | **SingleFlight middleware** — 序列化 `/v1/chat/completions`，避免 TRT 单引擎并发崩溃 ("Myelin: already loaded binary graph")。默认等 30s，超时返回结构化 `503 engine_busy` | `edge_llm_chat_service/guard.py:386-485` (新) <br> `edge_llm_chat_service/server.py:111-119` |

**测试**: `uv run --extra tokenizers pytest tests/ -q` ⇒ **45 passed**。

### 1.3 TensorRT-Edge-LLM upstream patches（已 push 到 Orin `/home/harvest/TensorRT-Edge-LLM/`，已在容器内）

| # | 修复 | 关键文件:行 |
|---|---|---|
| J | **`engine.py:generate_stream` finally 加 `channel.cancel()`** + worker.join timeout 5s→30s。consumer 关闭（client disconnect、barge-in）时让 C++ runtime 立刻撤回 | `experimental/server/engine.py:765-790` |
| K | **`_generate_stream_sse` 重写为 async generator** + 后台 drain 线程 + watcher 协程轮询 `request.is_disconnected()`；轮询触发时 `stop_flag.set()` → drain 线程下个 chunk 边界 break → 同线程 `gen.close()` → engine.py finally 跑 `channel.cancel()` | `experimental/server/api_server.py:418-545` |

记录在 `edge-llm-chat-service/UPSTREAM_PATCHES.md`。

### 1.5 2026-05-18 晚间 barge-in 固化补丁

新增结论：SLV `CLIENT_ABORT` 旧实现不只是取消 TTS，还调用了
`asr_stream.cancel_and_finalize()`，导致 barge-in 后“声音停了，但打断语音
没有 ASR final”。已修成 **abort 只取消 TTS / 清 TTS 队列，不取消 ASR**。

固化状态：
- 本地源码已改：`seeed-local-voice/app/main.py`
- 远端 orin-nx 源码已改：`/home/harvest/project/seeed-local-voice/app/main.py`
- 远端运行镜像已 commit：
  `seeed-local-voice:jetson-v1.12-highperf-bargein-asrfix-20260518`
- 远端 compose 已改：
  `/tmp/seeed-local-voice-release/deploy/docker-compose.yml`
  的 `speech.image` 指向上述本地修复镜像
- 当前运行容器：
  `seeed-local-voice-latest-speech-1`
  已由该修复镜像重建并监听 `100.82.225.102:8621`

备份：
- 远端容器内旧文件：
  `/opt/speech/app/main.py.bak-bargein-asr`
- 远端源码旧文件：
  `/home/harvest/project/seeed-local-voice/app/main.py.bak-bargein-asrfix-20260518`
- 远端 compose 旧文件：
  `/tmp/seeed-local-voice-release/deploy/docker-compose.yml.bak-bargein-asrfix-20260518`

### 1.4 SingleFlight 必要性证据

之前 LLMStreamError 实测重现：
```
14:51:33  POST /v1/chat/completions  ← Turn 1 长故事，stream 开始
14:51:39  BARGE-IN fired              ← agent cancel SSE 连接
14:51:40  POST /v1/chat/completions  ← Turn 2 (555ms 后)
14:51:40  → Myelin: already loaded binary graph
```

修复后：Turn 1 client disconnect → channel.cancel() → C++ worker 退出 → SingleFlight 放 Turn 2 → 不再撞引擎状态。

---

## 2. 用户使用启动方式

```bash
cd /Users/harvest/project/seeed-local-voice/agent
NO_PROXY='100.82.225.102,localhost,127.0.0.1' \
no_proxy='100.82.225.102,localhost,127.0.0.1' \
OVS_SLV_URL='ws://100.82.225.102:8621/v2v/stream' \
OVS_LLM_URL='http://100.82.225.102:8000/v1' \
OVS_LLM_MODEL='Qwen/Qwen3-4B-AWQ' \
nohup uv run ovs-agent run multi_mode > /tmp/ovs-agent.log 2>&1 &
```

Dashboard: <http://localhost:18000>。

---

## 3. 当前未解决问题（接力人请重点看这里）

### 3.1 ⚠ 第二次 barge-in 不触发（最重要）

**实测时序**（agent log `/tmp/ovs-agent.log`，2026-05-18 16:17~16:18）：

```
16:17:55.023  ConvState: thinking → speaking            (第 1 个故事开始播)
16:17:57.353  client VAD: speech started
16:17:57.354  BARGE-IN fired (VAD-driven, state=speaking)   ✅ 第 1 次 barge-in 成功
16:17:58.654  client VAD: speech ended -> asr_eos
16:17:58.919  asr_final received: '别说了等一下'
16:17:59.329  ConvState: thinking → speaking

16:18:03.330  asr_final received: '换一个故事吧'         (用户要新故事)
16:18:03.899  ConvState: thinking → speaking            (第 2 个故事开始播，26 秒长)
...
                                                          ← 用户在此期间想 barge-in，但没触发 ❌
                                                          ← 整个 26s 内没有 "speech started" 也没有 "BARGE-IN fired"
...
16:18:29.060  ⚠ "mic queue full -- dropping chunk" × 52 条
16:18:29.061  ConvState: speaking → idle                (故事自然播完)
```

**冒烟**：长故事播放期间 mic_pump 没消费 input queue → 26s 全部音频被 drop。VAD 没机会跑。

**已尝试的修复**（fix F）：mic_pump 的 `slv.send_audio` 套 0.5s 超时——但**不够**。

**怀疑根因（未验证）**：

1. **`_broadcast("on_mic_rms", ...)` 是否阻塞**？这个 broadcast 走 dashboard WS，浏览器卡 / WS 慢可能拖住 mic_pump（broadcast 没 timeout）。`app_base.py:486-501` 已经限频到 200ms/次，但每次还是 await 全部 plugin 的回调。
2. **TTS 播放期间 audio_io 占用 event loop**？`_playback_loop` 调 `self._output_stream.write(pcm)` 是 sounddevice 的同步阻塞调用——如果 PortAudio 输出 buffer 紧，几十 ms~几百 ms 阻塞，event loop 转不过来给 mic_pump 时间。
3. **VAD 计算本身耗时**？`_update_vad` 同步运算每个 chunk 100ms。

**建议下一步排查**：
- 在 `_send_audio_nonblocking` 调用后加 `logger.debug("mic chunk processed t=%dms")`，看长故事期间 mic 处理频率
- 注释掉 `on_mic_rms` broadcast 看是否就好了（隔离 dashboard 嫌疑）
- 把 `_playback_loop` 的 `output_stream.write` 套 `await asyncio.to_thread(...)`（变成不阻塞 event loop）
- 看 `audio_io.py:97-103` _safe_put 的 `mic queue full` 是 sounddevice callback 线程的告警——只在 _in_queue 满时打。`maxsize=64`，每 chunk 100ms = 6.4s 容量。26s 故事远超容量。改大 maxsize 不解决根因，但能确认是否消费侧慢。

### 3.2 ⚠ MIC RMS 柱形图渲染不对

见用户截图：柱形条在阈值线**下方**（绿色，说明 RMS > 阈值），但阈值横线被画在图表上方靠近顶部位置。代码上柱条和阈值线用同一 y 缩放公式 `Math.min(1, v/0.2) * H`：

```js
// dashboard.js:553-569
const h = Math.min(1, rmsSamples[i] / 0.2) * H;     // bar height
ctx.fillRect(i * bw, H - h, ..., h);
ctx.fillStyle = rmsSamples[i] >= thr ? "#82c91e" : "#7a8597";
const ty = H - Math.min(1, thr / 0.2) * H;          // threshold y
```

按数学，thr=0.030 + H=60 → ty = 60 - 9 = 51（接近底部）。但截图显示线在中上部，**实际位置与公式不一致**。

**怀疑**：
- canvas devicePixelRatio / CSS height 跟 width/height 属性不一致导致 y 缩放错（HTML `width=240 height=60`，CSS `width:100% height:60px`）
- `rmsThr.textContent` 被异常字符污染（`parseFloat` 拿到非预期值）
- 或者某次 broadcast 把 `d.threshold` 发成更大值

**复现路径**：开 dashboard 说点话能看出条/线相对位置。F12 console 跑 `parseFloat(document.getElementById('rmsThr').textContent)` 验证实际值。

### 3.3 ⚠ SLV WS keepalive timeout 时 mic queue full 刷屏

跟 3.1 部分重叠。每 ~10 分钟用户 idle，SLV 会主动 `1011 keepalive ping timeout` 关 WS。已加 dispatch 外层自动 reconnect（fix C），reconnect 成功了。但 reconnect 走 `ws_connect` 偶发 opening handshake timeout（要几秒~十几秒），期间 mic 输入端被 sounddevice 持续灌，"mic queue full" 大量刷屏。

fix F 应该缓解了，但 3.1 的故事场景说明还有别的阻塞点。

---

## 4. Dashboard 相关代码位置（接力人需要的地图）

### 4.1 后端

- **`agent/ovs_agent/plugins/debug_dashboard.py`** — 整个 dashboard plugin
  - 路由注册：line 95~120
  - WS 连接处理（`_handle_ws`）+ 浏览器客户端管理（`_browser_clients` set）
  - Plugin hook 实现：`on_*` 方法（被 `event_bus.emit(...)` 触发）
  - 控制 API：`_api_reconnect / _api_abort / _api_send_text / _api_session_clear / _api_wake / _api_sleep / _api_ptt_*`
  - `_schedule_broadcast(event, data)` → 给所有 browser ws 客户端推消息
  - EventBus 订阅：`bus.subscribe("on_session_trimmed" / "on_prefix_cache_disabled" / "on_echo_recovery", ...)` 见 line 142~

### 4.2 前端

- **`agent/ovs_agent/plugins/static/dashboard.html`** — 布局骨架
  - line 97: `<canvas id="rmsChart" width="240" height="60">` （§3.2 渲染 bug 的 canvas）
  - line 98: `<span id="rmsThr">` threshold 显示
- **`agent/ovs_agent/plugins/static/dashboard.css`** — 样式
  - line 269: `#rmsChart { width: 100%; height: 60px; ... }`
  - state 颜色：line 19~63（`--state-barged` 等）
- **`agent/ovs_agent/plugins/static/dashboard.js`** — 主逻辑
  - WS 连接 + 重连：`connectWS()`（line ~600）
  - 事件分发：单个大 `if (ev === "...")` 分支链，line ~700+
  - `drawRms()` MIC RMS canvas 绘制：line 553~569（§3.2 bug 在这）
  - `pushRms(v)` 推 sample + 排 rAF：line 549
  - `on_mic_rms` 接收：line 739~745
  - `applyState(s)` 状态 pill：line 543
  - Barge-in / barged_in 视觉：dashboard.css `.state-barged_in` line 63

### 4.3 协议（agent → 浏览器）

事件名约定（`event_bus.emit(name, data)` → broadcast 到所有浏览器 WS）：

| 事件 | data 形状 | 触发处 |
|---|---|---|
| `on_state_change` | `{state: "speaking"}` | `_set_state` |
| `on_mic_rms` | `{rms: float, threshold: float, state: "idle/speech"}` | mic_pump 每 200ms |
| `on_user_partial` | `text: str` | ASRPartial |
| `on_user_utterance` | `text: str` | ASRFinal |
| `on_assistant_token` | `text: str` | LLM stream |
| `on_assistant_done` | `null` | TTSDone |
| `on_session_trimmed` | `{dropped_turns, kept_turns, approx_tokens, ...}` | Session.messages 触发 trim |
| `on_echo_recovery` ✨ | `{window, echo_text, sid}` | 本次新加（fix E） |
| `on_error` | `TypedLLMError(type, message, exc_class)` | 异常路径 |
| `on_slv_reconnect` | `{count: N}` | SLV reconnect |
| `on_llm_availability_change` | `{state: "healthy/degraded/down/recovering/unknown"}` | 探活 |

---

## 5. 远端 Orin 相关速查

```bash
# 看 edge-llm log，关注 cancel / disconnect / single-flight / Myelin
fleet exec orin-nx -- 'docker logs --since 5m edge-llm-chat-service 2>&1 \
  | grep -iE "disconnect|cancel|single-flight|Myelin|error" \
  | grep -ivE "Initializing plugin|MS\]"'

# SLV log
fleet exec orin-nx -- 'docker logs --tail 30 seeed-local-voice-latest-speech-1 2>&1'

# Orin 端 TRT 源码（已 push 过 patch）：
#   /home/harvest/TensorRT-Edge-LLM/experimental/server/{engine.py,api_server.py}
# wrapper 仓库（build 入口）：
#   /home/harvest/edge-llm-chat-service/
# wrapper 重 build:
fleet exec orin-nx -- 'cd /home/harvest/edge-llm-chat-service && bash ./deploy/edge-llm/build.sh > /tmp/build.log 2>&1 && cd deploy/edge-llm && docker compose up -d'
```

---

## 6. 关键 commit 索引（本地 main 分支，未 push）

```bash
cd /Users/harvest/project/seeed-local-voice/agent && git log --oneline -25
cd /Users/harvest/project/edge-llm-chat-service     && git log --oneline -5
cd /Users/harvest/project/tensorrt-edge-llm         && git status   # 改动未 commit
```

upstream TRT 那两个文件的 patch 还**没 commit**，只是 push 到 Orin 跑着，需要正式 commit 到 `highperf/runtime-service` 分支并写明动机。

---

## 7. 给接力人的建议优先级

1. **§3.1 第二次 barge-in 不触发** — 用户体感最差，先解。优先验证是 `_broadcast` 阻塞还是 `_playback_loop` 阻塞 event loop（注释一个看另一个）。
2. **§3.2 RMS 柱形图渲染** — 调试期间影响判断 VAD 是否工作，优先级中等。canvas 渲染逻辑就 16 行代码，直接在浏览器 console 复现看。
3. **§3.3 mic queue full 刷屏** — 跟 §3.1 同根，§3.1 解了大概率这条也好。
4. 把 TRT upstream patch commit 进 `highperf/runtime-service` 分支并 push。

测试 / 启动 / 部署在 §1、§2、§5。需要联系 agent 端的人可以直接拿这个文档 + `/tmp/ovs-agent.log` + dashboard 截图开干。
