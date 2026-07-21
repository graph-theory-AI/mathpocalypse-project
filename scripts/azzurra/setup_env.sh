#!/bin/bash
# Build the env ON THE AZZURRA LOGIN NODE (it has internet; compute nodes may not).
# Creates a uv venv on /workspace with vLLM (to serve) + this harness's deps.
#   ssh azzurra 'cd /workspace/$USER/mathpocalypse && bash scripts/azzurra/setup_env.sh'
set -euo pipefail

PROJ="/workspace/$USER/mathpocalypse"
cd "$PROJ"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$HOME/.local/bin/uv"; command -v uv >/dev/null 2>&1 && UV=uv

$UV venv --python 3.11
# vLLM brings its own torch + CUDA wheels. openai = client to the served model;
# pyyaml = registry. Pin the working vLLM combo once a run confirms it (see docs/log.md).
$UV pip install vllm openai "huggingface-hub[hf_transfer]>=0.25" pyyaml

# Repopulate gitignored paper sources from the registry (login node has internet).
.venv/bin/python scripts/fetch_sources.py

echo "== versions =="
.venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)"
echo "OK. Env at $PROJ/.venv"
