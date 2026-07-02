# SLV Demo Gallery / 演示门户

**EN** — Browser-based demo portal for seeed-local-voice. Opens on port `:8700`,
shows what the device can do (live device status + demo cards), and hot-swaps
ASR/TTS model profiles at runtime through the SLV admin API. Each demo is a thin
FastAPI backend + build-free static frontend; the admin key never reaches the
browser.

**中文** — seeed-local-voice 的浏览器演示门户。打开 `:8700` 即可看到设备状态与
演示卡片，并可通过 SLV admin API 运行时热切换 ASR/TTS 模型组合。每个 demo 都是
「薄 FastAPI backend + 无构建纯静态前端」，admin 密钥只存在于 demo 后端，浏览器
永远拿不到。

## Quick start / 快速启动

Compose（推荐 / recommended）:

```bash
docker compose -f demos/docker-compose.demos.yml --profile gallery up -d
# open http://<device>:8700
```

Local dev without Docker（本地开发免 Docker）:

```bash
cd demos
uv sync
uv run uvicorn gallery.backend.main:app --host 0.0.0.0 --port 8700
```

Tests / 测试:

```bash
cd demos && uv run pytest tests/ -q
```

Need a fake SLV while developing? / 开发时可起假 SLV：

```bash
uv run python tests/mock_slv.py     # 127.0.0.1:8629, admin key "test-key"
SLV_URL=http://127.0.0.1:8629 SLV_ADMIN_KEY=test-key \
  uv run uvicorn gallery.backend.main:app --port 8700
```

## Environment / 环境变量

| Variable | Default | Description |
|---|---|---|
| `SLV_URL` | `http://127.0.0.1:8621` | SLV server base URL / SLV 服务地址 |
| `SLV_ADMIN_KEY` | *(empty)* | Forwarded as `X-Admin-Key` on admin calls; optional on loopback / 管理接口密钥，环回部署可省略 |
| `DEMO_PROFILES_DIR` | `<repo>/configs/profiles` | Profile JSONs offered in the switch panel / 切换面板可选的 profile 目录 |
| `DEMO_KIOSK` | `0` | Truthy = kiosk mode: hide debug details / 展会 kiosk 模式，隐藏调试信息 |
| `PORT` | `8700` | Gallery listen port (when run as `python main.py`) |

## Layout / 目录

```
demos/
  common/backend/slv_proxy.py    # shared SLV probe + admin proxy
  common/frontend/ui.css|ui.js   # design tokens + shared components
  common/frontend/slv-client.js  # browser SDK (ASR WS, TTS stream player; mic in P1)
  gallery/                       # portal app (:8700)
  registry.json                  # demo catalog (5 cards land in P1/P2)
  docker-compose.demos.yml
  tests/                         # pytest + controllable mock SLV
```
