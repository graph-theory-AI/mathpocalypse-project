#!/bin/bash
# One-time Jean Zay setup for the GLM-5.2 multi-node run. RUN ON A LOGIN OR prepost NODE
# (these have internet via the proxy; compute nodes do NOT). Everything lands on $WORK
# (5 TiB) — never $HOME (3 GiB cap).
#
#   ssh jeanzay
#   cd $WORK/mathpocalypse && bash scripts/jeanzay/setup_env.sh
#
# Produces: $WORK/mathpoc-venv (vLLM + openai + pyyaml) and the GLM-5.2-FP8 weights in
# $HF_HOME (~744 GB). After this, jobs run fully offline (HF_HUB_OFFLINE=1).
set -euo pipefail

VENV="${VENV:-$WORK/mathpoc-venv}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
export PIP_CACHE_DIR="$WORK/.pip_cache"        # keep pip's cache off the 3 GiB $HOME
export TMPDIR="$WORK/.tmp"; mkdir -p "$TMPDIR" "$HF_HOME"
MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"

# Build modules for the H100 target so any compiled bits match gpu_p6.
module purge
module load arch/h100
# Pick a Python; adjust to whatever `module avail python` shows on the day.
module load python/3.12.7 2>/dev/null || module load python 2>/dev/null || true
echo "python: $(python3 --version) @ $(command -v python3)"

# --- venv on $WORK -------------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip wheel
# vLLM must be recent enough to support GLM-5.2-FP8 (reasoning-parser glm45, sparse attention,
# pipeline-parallel). Pin once you confirm a working version against recipes.vllm.ai/zai-org/GLM-5.2.
pip install --upgrade "vllm" "openai" "pyyaml" "ray"
python -c "import vllm, ray; print('vllm', vllm.__version__, '| ray', ray.__version__)"

# --- pre-stage the weights (the slow part: ~744 GB over the proxy) -------------------------
echo "=== downloading $MODEL into $HF_HOME (this is ~744 GB — leave it running) ==="
hf download "$MODEL" --quiet || huggingface-cli download "$MODEL"

# --- paper sources (arxiv .tex; needs internet, so do it here too) -------------------------
python scripts/fetch_sources.py aubian-2025-2510.01791 bonamy-2014-1408.1964 || true

cat <<EOF

Setup done.
  venv     : $VENV
  weights  : $HF_HOME  (model $MODEL)
Next: submit the multi-node run with
  sbatch scripts/jeanzay/job_verify_glm52.sh

OPTIONAL kernel pre-warm: if the first job spends a long time JIT-compiling a sparse-attention
kernel, run a short boot once on the compil_h100 / a dev-QoS GPU job with TRITON_CACHE_DIR and
VLLM_CACHE_ROOT pointed at \$WORK (as the job script sets them) so the cache persists on \$WORK
and later jobs start warm. Compute nodes can't fetch headers, so the compile must succeed where
a CUDA toolkit + internet are available (compil_h100 / front-end).
EOF
