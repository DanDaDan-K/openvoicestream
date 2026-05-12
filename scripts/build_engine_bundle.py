#!/usr/bin/env python3
"""Build a host-signature-specific TRT engine bundle and a matching
HuggingFace manifest for upload.

Workflow:
  1. Detect host signature (same logic as app/core/engine_resolver).
  2. For each engine declared in a profile's required_engines:
       - call its build_script via env (WS auto-picked per device tier)
       - read the produced .engine / .plan
  3. Pack engines per-model into models/<m>/engines/<host_sig>.tar.gz
  4. Compute SHA-256 of each artifact + write models/<m>/manifest.json

Output layout under --out:
  <out>/models/<model_id>/manifest.json
  <out>/models/<model_id>/engines/<host_sig>.tar.gz
  <out>/models/<model_id>/onnx/<file>.onnx          (optional, copied for cold deploys)

Usage:
  uv run --project . scripts/build_engine_bundle.py \\
      --profile configs/profiles/jetson-zh-en.json \\
      --out /tmp/seeed-local-voice-artifacts

Upload to HuggingFace (after running this):
  huggingface-cli upload harvestsu/seeed-local-voice-artifacts \\
      /tmp/seeed-local-voice-artifacts .

The resulting tree on HF is consumed by app/core/engine_resolver.py at
runtime via HF_ENDPOINT (defaults to huggingface.co; set hf-mirror.com
for China).

This script is intended to be run on the target Jetson SKU (Nano/NX/AGX)
itself — the produced engines bake in tactics for that specific SM 8.7 +
TRT 10.3 + JetPack 6.x combination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.core import engine_resolver  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_engine_bundle")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _compile_one(spec: engine_resolver.EngineSpec, host: engine_resolver.HostSignature) -> None:
    """Compile an engine via its build_script, even if a stale file exists."""
    if spec.hf_only:
        logger.info("[%s] hf_only — skipping (caller must ship a prebuilt engine)", spec.engine_file)
        return
    if spec.engine_path.exists():
        spec.engine_path.unlink()
    engine_resolver._meta_path(spec.engine_path).unlink(missing_ok=True)
    engine_resolver._compile_locally(spec, host)


def _bundle_per_model(
    profile: dict,
    host: engine_resolver.HostSignature,
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list[engine_resolver.EngineSpec]] = {}
    for raw in profile.get("required_engines") or []:
        spec = engine_resolver.EngineSpec.from_dict(raw)
        by_model.setdefault(spec.model_id, []).append(spec)

    for model_id, specs in by_model.items():
        model_dir = out_root / "models" / model_id
        engines_dir = model_dir / "engines"
        engines_dir.mkdir(parents=True, exist_ok=True)

        # ── Build each engine in-place (target path = profile.engine_path) ──
        for spec in specs:
            _compile_one(spec, host)

        # ── Tar.gz the engines belonging to this model ──
        bundle_path = engines_dir / f"{host.key}.tar.gz"
        if bundle_path.exists():
            bundle_path.unlink()
        with tarfile.open(bundle_path, "w:gz") as tf:
            for spec in specs:
                if spec.hf_only or not spec.engine_path.exists():
                    continue
                tf.add(spec.engine_path, arcname=spec.engine_path.name)
                meta = engine_resolver._meta_path(spec.engine_path)
                if meta.exists():
                    tf.add(meta, arcname=meta.name)
        logger.info("packed %s → %s (%.1f MB)", model_id, bundle_path,
                    bundle_path.stat().st_size / (1024 * 1024))

        # ── Manifest with SHA-256s ──
        files: dict[str, dict] = {}
        rel_bundle = f"engines/{bundle_path.name}"
        files[rel_bundle] = {
            "sha256": _sha256(bundle_path),
            "size": bundle_path.stat().st_size,
        }
        # Also include ONNX inputs if present and requested.
        for spec in specs:
            if not spec.onnx_input:
                continue
            onnx_src = spec.engine_path.parent.parent / "onnx" / spec.onnx_input
            if not onnx_src.exists():
                logger.warning(
                    "onnx not on host for %s — skipping in manifest (cold deploys will need fallback path)",
                    spec.onnx_input,
                )
                continue
            onnx_dst_dir = model_dir / "onnx"
            onnx_dst_dir.mkdir(parents=True, exist_ok=True)
            onnx_dst = onnx_dst_dir / spec.onnx_input
            shutil.copy2(onnx_src, onnx_dst)
            rel_onnx = f"onnx/{spec.onnx_input}"
            files[rel_onnx] = {
                "sha256": _sha256(onnx_dst),
                "size": onnx_dst.stat().st_size,
            }

        manifest = {
            "model_id": model_id,
            "host_signatures": [host.to_dict()],
            "files": files,
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("manifest written: %s", model_dir / "manifest.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="profile JSON path")
    ap.add_argument("--out", required=True, help="output root for the artifact tree")
    args = ap.parse_args()

    profile = json.loads(Path(args.profile).read_text())
    host = engine_resolver.detect_host_signature()
    logger.info("host signature: %s", host.key)
    logger.info("profile: %s", profile.get("name"))

    _bundle_per_model(profile, host, Path(args.out))
    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
