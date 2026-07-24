#!/bin/bash
# One-time Jean Zay setup for the Laguna-S-2.1 run. RUN ON A LOGIN OR prepost NODE
# (these have internet via the proxy; compute nodes do NOT). Everything lands on $WORK
# (5 TiB) — never $HOME (3 GiB cap).
#
#   ssh jeanzay
#   cd $WORK/mathpocalypse && bash scripts/jeanzay/setup_env_laguna.sh
#
# Laguna-S-2.1 (Poolside, released 2026-07-22) is a 118B-total / 8.5B-active MoE, BF16 ~236 GB
# of weights. Unlike GLM-5.2-FP8 (~744 GB, multi-node), it FITS ONE gpu_p6 node (4x H100 80G =
# 320 GB) at TP=4 — so serving is single-node (job_verify_laguna.sh), no Ray/pipeline-parallel.
#
# Produces: $WORK/mathpoc-venv-laguna (vLLM + openai + pyyaml) and the Laguna weights in
# $HF_HOME. After this, jobs run fully offline (HF_HUB_OFFLINE=1).
set -euo pipefail

VENV="${VENV:-$WORK/mathpoc-venv-laguna}"      # dedicated venv: Laguna needs a vLLM new enough
                                               # to know the `laguna` arch + poolside_v1 parsers,
                                               # which may differ from the GLM venv's pin.
export HF_HOME="${HF_HOME:-$WORK/hf}"
export PIP_CACHE_DIR="$WORK/.pip_cache"        # keep pip's cache off the 3 GiB $HOME
export TMPDIR="$WORK/.tmp"; mkdir -p "$TMPDIR" "$HF_HOME"
# FP8 (~118 GB) is the default: it MATCHES how GLM was served (GLM-5.2-FP8), so it's the
# apples-to-apples quant for the head-to-head, and it downloads/loads ~2x faster than BF16 and
# fits one node with room to spare. Higher-quality fallback: MODEL=poolside/Laguna-S-2.1 (BF16,
# ~236 GB, still fits one node at TP=4).
MODEL="${MODEL:-poolside/Laguna-S-2.1-FP8}"

# Build modules for the H100 target so any compiled bits match gpu_p6.
module purge
module load arch/h100
module load python/3.12.7 2>/dev/null || module load python 2>/dev/null || true
echo "python: $(python3 --version) @ $(command -v python3)"

# --- venv on $WORK -------------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip wheel
# vLLM must be recent enough to register the `laguna` architecture and the `poolside_v1`
# reasoning + tool-call parsers (model card: --reasoning-parser poolside_v1). Install the latest
# release; if it does NOT list `laguna` support, fall back to the vLLM nightly that the poolside
# model card pins. PIN the working version here once the first serve is healthy (mirrors the GLM
# setup's "pin once confirmed against recipes.vllm.ai" note).
pip install --upgrade "vllm" "openai" "pyyaml"
python -c "import vllm; print('vllm', vllm.__version__)"

# --- pre-stage the weights (the slow part: ~236 GB BF16 over the proxy) ---------------------
echo "=== downloading $MODEL into $HF_HOME (~236 GB BF16 — leave it running) ==="
hf download "$MODEL" --quiet || huggingface-cli download "$MODEL"

# --- paper sources (arxiv .tex; needs internet, so do it here too) -------------------------
# The three GLM-flagged pilot papers live in the JGT registry, not the Phase-1 registry.
export MATHPOC_REGISTRY="papers/jgt_registry.yaml"
python scripts/fetch_sources.py \
  balogh-2025-2112.13277 bangjensen-2024-2302.06177 zhu-2025-2208.02050 || true

cat <<EOF

Setup done.
  venv     : $VENV
  weights  : $HF_HOME  (model $MODEL)
Next: submit the single-node run with
  sbatch scripts/jeanzay/job_verify_laguna.sh
EOF
