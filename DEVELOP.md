# Development setup

> For the full picture of how the three repos fit together, read
> [ARCHITECTURE.md](ARCHITECTURE.md). This file is the practical "get a dev box
> working" checklist.

## The `voxedge` dependency (open-core split, 2026-05-30)

The voice library was extracted into its own repo at `../voxedge`
(`/Users/harvest/project/voxedge`). The product (`server/` + `agent/`) imports
`voxedge.*` but does **not** vendor it, so a fresh checkout has no `voxedge` on
`sys.path` and will fail to import until you install it.

### Local dev — editable install

```bash
scripts/dev-setup.sh        # installs voxedge editable + server reqs + agent[dev]
```

or by hand:

```bash
uv pip install -e ../voxedge
```

`import voxedge` then resolves to the standalone repo. The product's backend
registry (`server/core/asr_backend.py` / `tts_backend.py`) points at
`voxedge.backends.*`; `server/core/voxedge_backend_config.py` builds each
backend's config from env/profile (voxedge backends are env-free).

### Deployment (docker) — wheel install

The images do **not** bind-mount voxedge. Every device Dockerfile
(`deploy/docker/Dockerfile.jetson{,.slim}`, `Dockerfile.rk`, `Dockerfile.rpi`)
`pip install`s a pre-built wheel staged at
`deploy/wheels/voxedge-0.0.1a0-py3-none-any.whl`. The thin "diff" images
(`Dockerfile.*.voxedge-patch`) `--force-reinstall` just that wheel onto a base
image for fast Python-only iteration.

**The wheel is a build artifact of `../voxedge`.** Rebuild it whenever voxedge
source changes, before building/deploying any image:

```bash
scripts/build_voxedge_wheel.sh
```

This regenerates the wheel from `../voxedge` and writes
`deploy/wheels/voxedge.BUILD.txt` recording the source git SHA, dirty flag, and
build date — so you can always tell which voxedge a deployed wheel was built
from. Commit the wheel and its `.BUILD.txt` together.

> The wheel filename is version-stable (`0.0.1a0`, pinned in
> `voxedge/pyproject.toml`) so the Dockerfiles never need editing on a rebuild;
> provenance lives in `voxedge.BUILD.txt`, not the filename.

## Running things locally

- **No-GPU smoke (mock backends):** see ARCHITECTURE.md → "Run it locally".
- **Server:** `python -m uvicorn server.main:app --port 8000`
- **Agent:** `ovs-agent run multi_mode --config <cfg.yaml>`

## Tests

```bash
pytest tests/                      # server integration tests (27)
pytest agent/tests/                # agent framework tests (94)
( cd ../voxedge && pytest )        # library tests (28, mock-based, no GPU)
```
