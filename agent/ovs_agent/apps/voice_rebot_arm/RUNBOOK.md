# voice_rebot_arm — reBot B601-DM 运维 / 验收 Runbook

语音控制 reBot B601-DM 机械臂的 app。复用我们的语音栈（wake / ASR / LLM 工具 / TTS）
+ Actuator ABC 框架；Phase A = 笛卡尔预设动作，Phase B = 去 torch 视觉抓取。

设备：**seeed-orin-nx**（Jetson Orin NX 16G, JetPack 6.2.1, CUDA 12.6, Python 3.10）。
机械臂在 **`/dev/ttyACM1`**（Damiao CAN via DM-serial）。**`/dev/ttyACM0` 是 SO-ARM，勿碰。**

---

## 0. 当前状态（2026-06-03）

- **Phase A — 部署 LIVE 已验**：容器 `voice-rebot-arm` 运行中（server-loop，8 工具 advertise 给 SLV）。
  actuator 连 ttyACM1 读到真实位姿、唤醒词 listening、go_home 通电动作 PASS。
  **未验**：真人对麦说命令（硬件麦无法软件注入，需到场，见 §2）。
- **Phase B — 感知半条链真机验通（dry-run）**：Orbbec 采帧 → YOLOE-seg ONNX 检测
  → 短轴抓取位姿（相机系）全跑通。**未做**：手眼标定 + 监督式真抓取（见 §3）。

---

## 1. 镜像 / 部署速查

- 镜像：`voice-rebot-arm:dev`（设备本地构建；`agent/Dockerfile.rebot-arm`，context=repo root）。
- 重新构建（设备上，前台）：
  ```
  # 在 device：把最新 agent/ 放到 /home/seeed/vra_build/agent/ 后
  docker build -f /home/seeed/vra_build/agent/Dockerfile.rebot-arm -t voice-rebot-arm:dev /home/seeed/vra_build
  ```
- **B601-DM 与 SO-ARM 二选一**（共用唯一 reSpeaker 麦 + 16G 内存，且内存余量极小）：
  ```
  # 上 B601-DM（停 SO-ARM）
  docker stop voice-arm && docker start voice-rebot-arm
  # 切回 SO-ARM
  docker rm -f voice-rebot-arm && docker start voice-arm   # 或 docker stop voice-rebot-arm
  ```
  `seeed-voice`(ASR/TTS) / `edge-llm`(LLM) 常驻共享，**绝不 stop/down**。
- 首次/换 image 起容器（完整 docker run，已含全部 env）：见 §5。
- 日志：`docker logs --tail 80 voice-rebot-arm`（fleet：`fleet exec --sudo seeed-orin-nx -- docker logs --tail 80 voice-rebot-arm`）。

---

## 2. Phase A 验收（语音 → 预设动作）— 需到场

1. 确认容器在跑且 SO-ARM 已停：`docker ps`（voice-rebot-arm Up、voice-arm 不在）。
2. **桌面清空、人在急停旁。**
3. 对 reSpeaker 麦说：**“Hey Jarvis”**（唤醒）→ 应答后 → 命令：
   - **“回到原位”** → `go_home`
   - **“挥手”** → `wave`（左右摆几下）
   - **“指一下”** → `point_at`
   - **“张开夹爪”** → `open_gripper`（开 6cm）
   - **“夹紧”/“抓住”** → `close_gripper`（grasp 0.2 N·m）
4. 不灵排查：`docker logs --tail 80 voice-rebot-arm`（看 wake 触发 / ASR 文本 / SLV tool_call / actuator 执行）。
5. 也可绕过语音直接测动作（HTTP，仍会动臂，需盯着）：
   `curl -X POST http://localhost:8775/actions/go_home/test`

动作坐标已在真机 IK 标定（`actions.yaml`，安全盒 x∈[0.20,0.34] y∈[±0.14] z∈[0.12,0.34]）。

---

## 3. Phase B 完成步骤（视觉抓取）— 含一次性手眼标定（需到场）

感知链已真机验通（见 §0）。到“完整抓取”还差以下，全是纯工程、无新未知：

### 3.1 重新导出 grasp 词表的 ONNX（离线，几分钟）
验证用的 ONNX 只 bake 了 `["person","bus"]`。真抓取要 bake 抓取物体词表：
```python
from ultralytics import YOLOE
m = YOLOE("yoloe-26s-seg.pt")
m.set_classes(["cup","water bottle","banana","light blue coffee cup","tool","red object","green object"])
m.export(format="onnx", opset=12, simplify=True, imgsz=640)   # 词表+NMS 烤进图
```
词表顺序要与 `YoloOnnxSegmenter` 的 `names` 一致。

### 3.2 手眼标定（eye-in-hand）→ `hand_eye.npz` ★硬门槛，必须到场
当前**没有** `config/calibration/orbbec_gemini2/hand_eye.npz`，没它相机系位姿转不到机械臂基座系 → **无法抓**。
- 准备：打印 ArUco 板（reBot 仓 `aruco100x100.pdf`），固定在工作台。
- 用 reBot-DevArm-Grasp 仓的 `scripts/collect_handeye_eih.py` + `calibration/hand_eye.py`（TSAI）：
  机械臂带相机走 N 个姿态，每姿态记 (TCP 位姿, 板的相机系位姿) → 解 eye-in-hand → 存 `hand_eye.npz`。
- 这一步**机械臂会动 + 需人监督 + 需标定板**，无法远程自动化。
- 产物放：`config/calibration/orbbec_gemini2/hand_eye.npz`（`utils/camera_utils.load_hand_eye` 读这里）。

### 3.3 把 perception 接入 grasp_service / 容器
- 容器镜像已含 cv2/onnxruntime/scipy；**缺 `pyorbbecsdk2` + ONNX engine + hand_eye.npz**，需加进镜像或挂载。
- 相机 SDK：`pip install pyorbbecsdk2`（cp310/aarch64 wheel，无需编译；**导入名是 `pyorbbecsdk`**）。
- 把 grasp 词表 ONNX + hand_eye.npz 挂进 `/opt/seeed/voice_rebot_arm/config/` 或烤进镜像。
- `grasp_object(object_name)` 工具已注册（Phase B），走 `grasp_service.run_grasp_once`。

### 3.4 监督式执行抓取（机械臂伸手，需人在场盯急停）
语音：“Hey Jarvis” → “抓那个杯子” → 相机检测 → 相机系 grasp pose → hand_eye 变换到基座
→ move_to(pregrasp→grasp) → grasp(force) → lift。barge-in/“停” 会 cancel 并安全张爪。

### 3.5 GPU 推理（可选，提速）
- onnxruntime-gpu 暴露 [TensorRT, CUDA, CPU] EP，但**生产语音栈占满 16G 内存 → TRT 建 engine 会 OOM**。
  要用 GPU/TRT：先腾内存（停语音栈），或接受 CPU EP（dry-run 实测 1.18s/帧，单次抓取够用）。
- dry-run 脚本 `tools/perception_dryrun.py` 有 `OVS_ORT_PROVIDERS=cpu|cuda` 开关。

---

## 4. 已知坑（全部踩过，勿重蹈）

**镜像构建**
- base 必须 `python:3.10-slim`（Pinocchio `pin` 4.0.0 aarch64 wheel 只在 cp310 验过；3.11 无保证）。`requires-python>=3.10`。
- 缺 `build-essential` + `portaudio19-dev` → pyaudio 源码编译失败（无 aarch64 wheel）。
- `motorbridge` 不 pin 会版本漂（见过 0.2.2 / 0.4.2）→ pin `==0.2.8`（真机验证版本）。
- 唤醒词模型不在镜像、运行时也不下（无 entrypoint）→ Dockerfile `openwakeword.utils.download_models()` 构建时烧入。

**运行 config**
- SDK 默认 `channel=/dev/ttyACM0`（SO-ARM 口！）→ 必须传 **realpath `/dev/ttyACM1`**（不能 by-id 软链，SDK 用 `startswith("/dev/tty")` 判串口/CAN）。
- `grasp_force`/`open_distance_m` 等空串（`${VAR:-}` 未设）→ 旧代码 `float('')` 崩 → ArmPlugin 禁用。已修：空串当未设。
- `MIC_INDEX=auto` 是**死代码**（`audio/devices.py:resolve_input_index` 从没被调用）→ 用名称子串 `MIC_INDEX=reSpeaker`。
- 缺 `OVS_AGENT_SERVER_LOOP=1` → 跑 legacy client-loop（skip advertise）。生产用 server-loop（SLV 跑 LLM+工具循环）。已设为镜像默认。
- 夹爪“只动夹爪”的帧要**省略位姿字段**（带 x/y/z 会先 move_to 让臂乱动）。

**Phase B 设备**
- Orbbec USB 节点默认 root-only → `sudo chmod a+rw /dev/bus/usb/<bus>/<dev>`（单节点，非破坏）后 SDK 才能开。
- `pyorbbecsdk`(v1) 无 cp310/aarch64 wheel；用 `pyorbbecsdk2`（pip 名），导入名 `pyorbbecsdk`。

**Fleet**
- `fleet exec` 直接 exec argv（无 shell）：`cd`/管道/`python -c` 等需 `--literal` + `sh -c '...'`。
- 带空格的 env 值（如 `WAKEWORD_MODEL="hey jarvis"`）经 fleet 会被拆 → 用 `--literal` + `-e "WAKEWORD_MODEL=hey jarvis"`。

---

## 5. 完整 docker run（参考）

```bash
docker run -d --name voice-rebot-arm --restart no --runtime nvidia \
  --network voice-arm_default --user 0 \
  --device /dev/snd --device /dev/ttyACM1 \
  -v /run/user:/run/user:ro -v /home/seeed/.config/pulse:/host-pulse-config:ro \
  -v /opt/seeed/voice_rebot_arm/config:/opt/seeed/voice_rebot_arm/config \
  -e CONFIG_DIR=/opt/seeed/voice_rebot_arm/config \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  -e VOICE_SERVICE_HOST=seeed-voice -e VOICE_SERVICE_PORT=8000 -e VOICE_SERVICE_URL=http://seeed-voice:8000 \
  -e LLM_SERVICE_HOST=edge-llm -e LLM_SERVICE_PORT=8000 -e LLM_SERVICE_URL=http://edge-llm:8000 \
  -e MIC_INDEX=reSpeaker -e MIC_CHANNEL_SELECT=0 -e WAKEWORD_MODEL="hey jarvis" \
  -e REBOT_CHANNEL=/dev/ttyACM1 -e REBOT_REPO_ROOT=/opt/rebot \
  -e HF_ENDPOINT=https://hf-mirror.com -e OBSERVATION_PORT=8775 \
  -e OVS_AGENT_SERVER_LOOP=1 \
  -p 8775:8775 voice-rebot-arm:dev
```
（镜像已把大部分 env 设为默认；上面显式列全便于排查。）

---

## 6. 源码地图

| 路径 | 作用 |
|---|---|
| `rebot_arm.py` | vendored RebotArm 封装（SDK 延迟 import、channel 覆盖、夹爪力控） |
| `rebot_actuator.py` | `RebotArmActuator(Actuator)`：frame→笛卡尔 waypoint，夹爪带符号幅度 |
| `actions.yaml` | 5 个动作（真机 IK 标定的笛卡尔坐标） |
| `config.yaml` | app 配置（${VAR} 由 ovs-agent 自身 env 替换） |
| `app.py` | `VoiceRebotArmApp`：注册 ArmPlugin + GraspPlugin |
| `grasp_plugin.py` / `grasp_service.py` | Phase B：`grasp_object` 工具 + 抓取流水线（cancel 感知） |
| `perception/yolo_onnx.py` | 去 torch YOLOE-seg（onnxruntime + numpy/cv2 后处理） |
| `perception/ordinary_grasp.py` / `transforms.py` / `camera/` | 短轴抓取 / 坐标变换 / 相机驱动（vendored 去 torch） |
| `tools/perception_dryrun.py` | Phase B 感知 dry-run 参考脚本（不动臂；路径为 seeed-orin-nx 设置） |
