#!/bin/bash
# Verify papers with Laguna-S-2.1 (Poolside, 118B MoE / 8.5B active) on Jean Zay, ONE gpu_p6
# node (4x H100 80G = 320 GB, TP=4): serve -> wait health -> run the mathpoc harness over
# PAPERS against localhost -> release. Single-node (FP8 weights ~118 GB fit one node with room to spare), so NO
# Ray and NO pipeline-parallel — the simple Azzurra-style shape, not the GLM multi-node one.
#
# Default PAPERS = the three GLM-flagged pilot papers with adjudicated ground truth, so we can
# score Laguna vs GLM head-to-head on the SAME verify_v0 prompt:
#   balogh-2025-2112.13277  — GLM FALSE POSITIVE (probabilistic sign error); theorem stands
#   bangjensen-2024-2302.06177 — GLM FALSE POSITIVE (misread TikZ figure); theorem stands
#   zhu-2025-2208.02050     — GLM REAL gap (reducible-tuple a_2=-1); theorem-vs-proof open
# --self-verify is ON (the FP guard) — a key test is whether Laguna's self-verify dissolves the
# two FPs that GLM's confirmed.
#
# PREREQS (run once on a login/prepost node — compute nodes are OFFLINE):
#   scripts/jeanzay/setup_env_laguna.sh   # builds $WORK venv-laguna + downloads Laguna-S-2.1
#
# Submit (from a login node, in $WORK/mathpocalypse):
#   sbatch scripts/jeanzay/job_verify_laguna.sh
#   PAPERS="zhu-2025-2208.02050" sbatch scripts/jeanzay/job_verify_laguna.sh
# Watch:   squeue -u $USER        (job name mathpoc-laguna)
# Result:  reports/*.json         server log: logs/vllm_laguna.<jobid>.log
#
#SBATCH --job-name=mathpoc-laguna
#SBATCH --account=amv@h100          # same grant as the GLM run (proj amv). myv@h100 = fallback.
#SBATCH --constraint=h100           # gpu_p6 — NOT a --partition flag on Jean Zay
#SBATCH --nodes=1                    # single node: ~118 GB FP8 weights fit 4x H100 (320 GB) at TP=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                 # full gpu_p6 node = 4x H100 80 GB
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00              # 4 GPU x 3h = 12 GPU-h; single-node schedules far easier than GLM's 3-node
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

module purge
module load arch/h100                # REQUIRED before H100-built modules are visible
module load cuda/13.0.3              # matches the venv torch cu130; present in case a kernel JIT-compiles via nvcc
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # compute nodes are offline — proxy would hang curl/NCCL

PROJ="$WORK/mathpocalypse"; cd "$PROJ"
VENV="${VENV:-$WORK/mathpoc-venv-laguna}"
MODEL="${MODEL:-poolside/Laguna-S-2.1-FP8}"   # FP8 (~118 GB) — MATCHES GLM's own FP8 serving, so
                                              # it's the apples-to-apples quant for the head-to-head,
                                              # and downloads/loads ~2x faster than BF16. BF16
                                              # (poolside/Laguna-S-2.1) is the higher-quality fallback.
SERVED="${SERVED:-laguna-s-2.1}"          # --served-model-name; harness auto-detects it
TP="${TP:-4}"
PORT="${PORT:-8000}"
# FP8 weights are ~118 GB of 320 GB, so KV+activation headroom is huge (~170 GB across 4 GPUs).
# GMU 0.90 is comfortable; KV in fp8, 1-2 seqs (we verify sequentially).
GMU="${GMU:-0.90}"
KVDTYPE="${KVDTYPE:-fp8}"
MAXSEQS="${MAXSEQS:-2}"
MAXLEN="${MAXLEN:-163840}"           # 160K window. Ample for these 3 short papers (<50K input) +
                                     # Laguna thinking traces; small enough to fit the tight KV pool.
MAXTOK="${MAXTOK:-131072}"           # cap generated thinking+answer per paper (128K)
EFFORT="${EFFORT:-max}"              # laguna style: any non-"off" => enable_thinking:true (binary; no levels)
# Laguna has no recommended sampling on the model card; use mild-reasoning defaults (temp 0.7,
# top_p 0.95) rather than GLM's 1.0/1.0. JSON is prompt-instructed + leniently parsed, so exact
# values are not load-bearing; kept a knob for reproducibility.
TEMP="${TEMP:-0.7}"
TOPP="${TOPP:-0.95}"
SELF_VERIFY="${SELF_VERIFY:-1}"      # 1 = adversarially re-check each major/critical finding (FP guard)
SV_FLAG=""; [ "$SELF_VERIFY" = "1" ] && SV_FLAG="--self-verify"
PAPERS="${PAPERS:-balogh-2025-2112.13277 bangjensen-2024-2302.06177 zhu-2025-2208.02050}"
# The three papers are in the JGT registry, not the Phase-1 registry — point the whole harness at it.
export MATHPOC_REGISTRY="papers/jgt_registry.yaml"
export HF_HOME="${HF_HOME:-$WORK/hf}"; export HF_HUB_OFFLINE=1   # weights pre-staged by setup_env_laguna.sh
export TRITON_CACHE_DIR="$WORK/.triton_cache"
export VLLM_CACHE_ROOT="$WORK/.vllm_cache"
mkdir -p logs reports "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
# Unique per-job log: the health loop greps it for failure strings; a reused log could let a stale
# error from a prior job trip a false early-abort (GLM job 845371 lesson).
VLOG="logs/vllm_laguna.${SLURM_JOB_ID}.log"

echo "=== node ==="; hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv
source "$VENV/bin/activate"
python -c "import vllm; print('vllm', vllm.__version__)"
python scripts/fetch_sources.py $PAPERS 2>/dev/null || true   # no-op if sources already staged

echo "=== serve $MODEL (TP=$TP, max_len=$MAXLEN, kv=$KVDTYPE, gmu=$GMU) — first load of ~118 GB (FP8) is slow ==="
# poolside_v1 reasoning parser (confirmed registered in vLLM 0.25.1; arch LagunaForCausalLM is
# natively supported). No --tool-call-parser: we never call tools (verification is prompt-
# instructed JSON), so it would only add a failure surface. Thinking ON at server level (the
# harness also sends enable_thinking:true via --thinking-style laguna, so request and server agree).
# --enforce-eager: on release-day (2026-07-22) the Laguna FP8 torch.compile/inductor + CUDA-graph
# capture path crashes with `CUDA error: illegal memory access` during warmup (job 116689). Eager
# skips compile+capture entirely — robust for a brand-new arch; inference is slower but irrelevant
# for 3 papers. Drop this flag once vLLM ships a compile fix for LagunaForCausalLM.
vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size "$TP" \
  --enforce-eager \
  --kv-cache-dtype "$KVDTYPE" --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$GMU" \
  --reasoning-parser poolside_v1 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --trust-remote-code --port "$PORT" \
  > "$VLOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

echo "=== wait for health (init can take ~15-30 min: 118 GB FP8 load over Lustre + cudagraph); log=$VLOG ==="
for i in $(seq 1 600); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "healthy ~$((i*5))s"; break; }
  grep -qiE "unrecognized arguments|invalid choice|out of memory|CUDA error|Engine core init.*failed|No available memory|ValueError.*architecture" "$VLOG" 2>/dev/null \
    && { echo "!! server failed early — see $VLOG"; tail -60 "$VLOG"; exit 1; }
  [ "$i" -eq 600 ] && { echo "!! health timeout (50 min)"; tail -60 "$VLOG"; exit 1; }
  sleep 5
done

echo "=== verify (model=$SERVED, thinking=laguna/on, self_verify=$SELF_VERIFY): $PAPERS ==="
export MATHPOC_BASE_URL="http://localhost:$PORT/v1" MATHPOC_MODEL="$SERVED"
python -m mathpoc verify $PAPERS \
  --backend http --thinking-style laguna $SV_FLAG \
  --reasoning-effort "$EFFORT" --temperature "$TEMP" --top-p "$TOPP" --max-tokens "$MAXTOK"

echo "=== done; reports in $PROJ/reports/ ; releasing GPUs ==="
