#!/bin/bash
# Run the agent pipeline on ONE Azzurra node (4x H100, TP=4): serve DeepSeek-V4-Flash, then
# run survey->verify->refute->aggregate on the paper. The single-pass BASELINE is not re-run
# here — we compare against the existing reports/<paper>.verify_v0.*.json from the 06-17 run
# (same model, same Think-Max knobs). Built on the proven job_verify.sh recipe.
#
#   ssh azzurra 'cd /workspace/$USER/mathpocalypse && sbatch scripts/azzurra/job_pipeline.sh'
#   PAPER="bonamy-2014-1408.1964" sbatch scripts/azzurra/job_pipeline.sh
# Watch:   squeue -u $USER   (job name mathpoc-pipeline)
# Result:  reports/<paper>.pipeline_v0.*.json     (survey->verify->refute->aggregate)
#          server log: logs/mathpoc-pipeline-*.out
#SBATCH --job-name=mathpoc-pipeline
#SBATCH --account=coati
#SBATCH --partition=gpu
#SBATCH --gpus=h100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=/workspace/%u/logs/%x-%j.out
#SBATCH --error=/workspace/%u/logs/%x-%j.err
set -euo pipefail

# DeepSeek-V4 JIT-compiles a fused attention kernel via TileLang->nvcc at startup; load the
# coherent system CUDA so nvcc + headers agree (see ../bigLLM-azzurra docs/log.md).
module purge
module load cuda/13.0
export CUDA_HOME="${CUDA_HOME:-/softs/cuda-13.0}"

PROJ="/workspace/$USER/mathpocalypse"; cd "$PROJ"
VENV="${VENV:-/workspace/$USER/bigLLM/.venv}"   # reuse the proven vllm 0.23.0 venv
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
TP="${TP:-4}"
MAXLEN="${MAXLEN:-393216}"          # Think Max REQUIRES >= 393216 (384K) or it truncates
KVDTYPE="${KVDTYPE:-fp8}"
MAXSEQS="${MAXSEQS:-8}"             # pipeline issues several requests; keep a few concurrent slots
PORT="${PORT:-8000}"
EFFORT="${EFFORT:-max}"             # Think Max
MAXTOK="${MAXTOK:-131072}"          # cap generated reasoning+answer per call
PAPER="${PAPER:-bonamy-2014-1408.1964}"   # the Erdos-Hajnal paper: the Lemma 2.1 question
export HF_HOME="/workspace/$USER/hf"; export HF_HUB_OFFLINE=1
mkdir -p logs reports

echo "=== node ==="; hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv
source "$VENV/bin/activate"
python -c "import vllm; print('vllm', vllm.__version__)"
python scripts/fetch_sources.py "$PAPER"   # no-op if source already staged

echo "=== serve $MODEL (TP=$TP, max_len=$MAXLEN, kv=$KVDTYPE) — first load of 159.6 GB is slow ==="
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" \
  --kv-cache-dtype "$KVDTYPE" --max-num-seqs "$MAXSEQS" \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --trust-remote-code --port "$PORT" \
  > logs/vllm_pipeline.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "=== wait for health (init can take ~15-30 min: load + TileLang compile + cudagraph) ==="
for i in $(seq 1 600); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "healthy ~$((i*5))s"; break; }
  grep -qiE "unrecognized arguments|invalid choice|headers are incompatible|Engine core initialization failed|out of memory|CUDA error" logs/vllm_pipeline.log 2>/dev/null \
    && { echo "!! server failed early — see logs/vllm_pipeline.log"; tail -40 logs/vllm_pipeline.log; exit 1; }
  sleep 5
done

export MATHPOC_BASE_URL="http://localhost:$PORT/v1"

echo "=== agent pipeline (survey->verify->refute->aggregate) on $PAPER ==="
echo "    baseline for comparison = existing reports/$PAPER.verify_v0.*.json (not re-run)"
python -m mathpoc verify-pipeline "$PAPER" \
  --reasoning-effort "$EFFORT" --temperature 1.0 --top-p 1.0 --max-tokens "$MAXTOK"

echo "=== done; pipeline report in $PROJ/reports/ ; releasing GPUs ==="
