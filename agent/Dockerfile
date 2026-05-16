# OpenVoiceStream Agent runtime image.
#
# IMPORTANT: build context MUST be the SLV repo root (one level up),
# because the agent imports SLV's `app.core.v2v` constants (invariant 6).
#
# Build:
#   docker build -f agent/Dockerfile -t ovs-agent:latest .
#
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/slv

# SLV source (read-only) -- agent imports from app.core.v2v.
COPY app /opt/slv/app

# Agent source.
COPY agent /opt/slv/agent

ENV PYTHONPATH=/opt/slv:/opt/slv/agent

RUN pip install --no-cache-dir \
        "websockets>=12" \
        "sounddevice>=0.4" \
        "numpy>=1.24" \
        "openai>=1.40" \
        "pyyaml>=6.0" \
        "pysbd>=0.3.4" \
    && pip install --no-cache-dir /opt/slv/agent

CMD ["python", "-m", "openvoicestream_agent.cli", "run", "dialogue"]
