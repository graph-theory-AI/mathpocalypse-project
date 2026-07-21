#!/bin/bash
# Verify papers with DeepSeek-V4-Flash on ONE Azzurra node (4x H100, TP=4):
# serve the model -> wait health -> run the mathpoc harness over PAPERS -> release.
# Serving recipe proven in ../bigLLM-azzurra; reasoning flags per the vLLM V4 recipe.
#
#   ssh azzurra 'cd /workspace/$USER/mathpocalypse && sbatch scripts/azzurra/job_verify.sh'
#   PAPERS="bonamy-2014-1408.1964" sbatch scripts/azzurra/job_verify.sh
# Watch:   squeue -u $USER   (job name mathpoc-verify)
# Result:  reports/*.json    server log: logs/vllm_verify.log
#SBATCH --job-name=mathpoc-verify
#SBATCH --account=coati
#SBATCH --partition=gpu
#SBATCH --gpus=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/%u/logs/%x-%j.out
#SBATCH --error=/workspace/%u/logs/%x-%j.err
set -euo pipefail

# DeepSeek-V4 JIT-compiles a fused attention kernel via TileLang->nvcc at startup; load the
# coherent system CUDA so nvcc + headers agree (see ../bigLLM-azzurra docs/log.md).
module purge
module load cuda/13.0
export CUDA_HOME="${CUDA_HOME:-/softs/cuda-13.0}"

PROJ="/workspace/$USER/mathpocalypse"; cd "$PROJ"
# Reuse the proven bigLLM venv (vllm 0.23.0 + openai + pyyaml) for this first run; a
# dedicated venv (scripts/azzurra/setup_env.sh) can replace it later via VENV=.
VENV="${VENV:-/workspace/$USER/bigLLM/.venv}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
TP="${TP:-4}"
MAXLEN="${MAXLEN:-393216}"          # Think Max REQUIRES >= 393216 (384K) or it truncates
KVDTYPE="${KVDTYPE:-fp8}"           # V4 FlashMLA fp8 attention REQUIRES an fp8 kv-cache
MAXSEQS="${MAXSEQS:-4}"             # we verify sequentially; keep KV/headroom generous
PORT="${PORT:-8000}"
EFFORT="${EFFORT:-max}"             # Think Max
MAXTOK="${MAXTOK:-131072}"          # cap generated reasoning+answer per paper
PAPERS="${PAPERS:-aubian-2025-2510.01791 bonamy-2014-1408.1964}"  # the two short ones
export HF_HOME="/workspace/$USER/hf"; export HF_HUB_OFFLINE=1
mkdir -p logs reports

echo "=== node ==="; hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv
source "$VENV/bin/activate"
python -c "import vllm; print('vllm', vllm.__version__)"
python scripts/fetch_sources.py $PAPERS   # no-op if sources already staged

echo "=== serve $MODEL (TP=$TP, max_len=$MAXLEN, kv=$KVDTYPE) — first load of 159.6 GB is slow ==="
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" \
  --kv-cache-dtype "$KVDTYPE" --max-num-seqs "$MAXSEQS" \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --trust-remote-code --port "$PORT" \
  > logs/vllm_verify.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "=== wait for health (init can take ~15-30 min: load + TileLang compile + cudagraph) ==="
for i in $(seq 1 600); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "healthy ~$((i*5))s"; break; }
  grep -qiE "unrecognized arguments|invalid choice|headers are incompatible|Engine core initialization failed|out of memory|CUDA error" logs/vllm_verify.log 2>/dev/null \
    && { echo "!! server failed early — see logs/vllm_verify.log"; tail -40 logs/vllm_verify.log; exit 1; }
  sleep 5
done

echo "=== verify (reasoning_effort=$EFFORT, temp=1.0, top_p=1.0): $PAPERS ==="
export MATHPOC_BASE_URL="http://localhost:$PORT/v1"
python -m mathpoc verify $PAPERS \
  --reasoning-effort "$EFFORT" --temperature 1.0 --top-p 1.0 --max-tokens "$MAXTOK"
echo "=== done; reports in $PROJ/reports/ ; releasing GPUs ==="
