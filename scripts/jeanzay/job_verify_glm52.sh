#!/bin/bash
# Verify papers with GLM-5.2-FP8 (744B MoE) on Jean Zay, MULTI-NODE H100.
# GLM-5.2-FP8 weights (~744 GB) don't fit one gpu_p6 node (4x H100 80G = 320 GB), so we
# spread them: TP=4 (in-node, NVLink) x PP=NODES (across nodes, InfiniBand). 3 nodes = 12
# H100 = 960 GB is the tight-but-workable floor; 4 nodes (16 GPU) is comfortable.
#
# Pattern (default, BACKEND=http): boot a Ray cluster over the SLURM allocation -> `vllm serve`
# on the head node (places TPxPP ranks via Ray) -> wait health -> run the mathpoc harness over
# PAPERS against the localhost server -> release. Same proven shape as scripts/azzurra/job_verify.sh
# + the Ray bootstrap. The localhost server is internal IPC on an offline node, not an exposed
# endpoint; we use it because the recipe is proven, the glm45 reasoning parser runs server-side
# for free, and the 744 GB model loads ONCE and can be hit repeatedly while iterating.
#
# Alternative (BACKEND=vllm, EXPERIMENTAL): no server — the harness loads the model in-process via
# vLLM's offline API and runs the same papers as a pure batch. Cleaner single-process shape, but a
# less-trodden multi-node code path and reasoning is split heuristically (no server-side glm45).
# Use once the http path is proven; see docs/log.md 2026-06-23.
#
# PREREQS (run once on a login/prepost node — compute nodes are OFFLINE):
#   scripts/jeanzay/setup_env.sh   # builds $WORK venv + pre-downloads GLM-5.2-FP8 to $HF_HOME
#
# Submit (from a login node, in $WORK/mathpocalypse):
#   sbatch scripts/jeanzay/job_verify_glm52.sh
#   NODES=4 PAPERS="aubian-2025-2510.01791" sbatch scripts/jeanzay/job_verify_glm52.sh
# Watch:   squeue -u $USER        (job name mathpoc-glm52)
# Result:  reports/*.json         server log: logs/vllm_glm52.log
#
#SBATCH --job-name=mathpoc-glm52
#SBATCH --account=amv@h100          # 7,500 h.gpu grant AD011018098 (proj amv, 105842). myv@h100 = 500 h fallback.
#SBATCH --constraint=h100           # gpu_p6 — NOT a --partition flag on Jean Zay
#SBATCH --nodes=3                    # override via env: NODES (keep #SBATCH and the var in sync — see note below)
#SBATCH --ntasks-per-node=1          # one Ray launcher per node
#SBATCH --gres=gpu:4                 # full gpu_p6 node = 4x H100 80 GB
#SBATCH --cpus-per-task=96           # ~physical cores of a gpu_p6 node
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00              # <= qos_gpu_h100-t3 (20h). 12 GPU x 4h = 48 GPU-h of the 500 budget
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

# NOTE: #SBATCH --nodes is read by Slurm BEFORE the shell runs, so the NODES env override only
# takes effect when you pass it on the CLI (`NODES=4 sbatch --nodes=4 ...`) or keep them equal.
# We derive PP from the actually-allocated node count ($SLURM_JOB_NUM_NODES), so PP is always right.
NODES="${SLURM_JOB_NUM_NODES}"
GPUS_PER_NODE=4
TP="${TP:-$GPUS_PER_NODE}"           # tensor-parallel within a node (NVLink)
PP="${PP:-$NODES}"                   # pipeline-parallel across nodes (TP*PP must = total GPUs)

module purge
module load arch/h100                # REQUIRED before H100-built modules are visible
# GLM-5.2-FP8 (like DeepSeek) uses DeepGEMM, which JIT-COMPILES its fp8 MoE GEMM kernels with nvcc at
# runtime (first forward). Without a CUDA toolkit module nvcc is absent and DeepGEMM asserts
# `std::filesystem::exists(nvcc_path)` right after the 744 GB weight load (job 878700, 2026-06-25).
# cuda/13.0.3 matches the venv's torch 2.11.0+cu130 and sets CUDA_HOME so DeepGEMM finds nvcc. This is
# the Jean Zay analogue of the Azzurra recipe's "CUDA 13.0 module for the TileLang kernel".
module load cuda/13.0.3
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # compute nodes are offline — proxy would hang NCCL/curl

PROJ="$WORK/mathpocalypse"; cd "$PROJ"
BACKEND="${BACKEND:-http}"           # http = proven server recipe (default); vllm = experimental in-process batch
VENV="${VENV:-$WORK/mathpoc-venv}"   # built by setup_env.sh (recent vLLM that supports GLM-5.2 + sparse attn)
MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"   # HF id; resolves from the offline cache under $HF_HOME
SERVED="${SERVED:-glm-5.2-fp8}"      # --served-model-name (http mode); harness auto-detects it
PORT="${PORT:-8000}"                 # http mode: localhost-only server port
MAXSEQS="${MAXSEQS:-2}"              # http mode: we verify sequentially; small KV footprint
MAXLEN="${MAXLEN:-262144}"           # 256K. HARD-CAPPED by the KV pool, NOT by GMU headroom: with the 744B weights
                                     # nearly filling 12 GPUs, only ~5.25 GiB/worker is left for KV, and vLLM refuses
                                     # to start unless one max-len seq fits in it. Job 1281047 (2026-07-03) died at
                                     # engine-core init with MAXLEN=327680: "6.25 GiB KV needed > 5.25 GiB available,
                                     # estimated max model length is 275072". So 262144 (< that ceiling) is the safe,
                                     # proven value (served fine on jobs 1148396/1240111). Do NOT raise it; fix the
                                     # input+output-overflow the other way, by capping MAXTOK below. (Was 327680.)
KVDTYPE="${KVDTYPE:-fp8}"
GMU="${GMU:-0.90}"                   # 0.95 OOM'd by 0.16 GiB during the FIRST forward (MoE expert-activation
                                     # spike at prefill; job 879092, 2026-06-25). 0.90 frees ~4 GiB/GPU; we
                                     # only verify 1 seq at a time so the smaller KV (was 3.47x concurrency)
                                     # is fine. Pair with expandable_segments below.
EFFORT="${EFFORT:-max}"              # GLM-5.2 Think-Max (chat_template_kwargs.reasoning_effort; --thinking-style glm)
MAXTOK="${MAXTOK:-163840}"           # 160K. Two constraints bracket this: (LOWER) job 1110148 (2026-06-30) lost
                                     # 8/15 papers to truncation at EXACTLY 65536 completion tokens (GLM Think-Max
                                     # overran the cap before closing the JSON; successes were 53K-64K), so MAXTOK
                                     # must sit well above ~65K. (UPPER) input+MAXTOK must fit MAXLEN=262144, and
                                     # aharoni's input alone is 65537 tok, so 196608 overflowed by 1 token and 400'd.
                                     # 160K threads both: 2.5x the observed 64K output ceiling, yet leaves 98K tokens
                                     # for input (1.5x aharoni). Residual oversized-input papers are caught by the
                                     # per-paper try/except in __main__.py (skipped, not fatal). (Was 196608.)
SELF_VERIFY="${SELF_VERIFY:-1}"      # 1 = re-check each major/critical finding adversarially (FP guard); 0 = off
SV_FLAG=""; [ "$SELF_VERIFY" = "1" ] && SV_FLAG="--self-verify"
PAPERS="${PAPERS:-aubian-2025-2510.01791 bonamy-2014-1408.1964}"  # the two short ones, as on Azzurra
export HF_HOME="${HF_HOME:-$WORK/hf}"; export HF_HUB_OFFLINE=1   # weights pre-staged by setup_env.sh
# Persist Triton/torch.compile kernel caches on $WORK so a sparse-attention JIT compile (if any) is
# paid once, not every job (compute nodes can't fetch headers — see setup_env.sh kernel-warm note).
export TRITON_CACHE_DIR="$WORK/.triton_cache"
export VLLM_CACHE_ROOT="$WORK/.vllm_cache"
# DeepGEMM writes its JIT-compiled kernels to a cache; keep it off the 3 GiB $HOME (Jean Zay gotcha)
# and on $WORK. Belt-and-suspenders across DeepGEMM versions (var name has changed); harmless if unused.
export DG_JIT_CACHE_DIR="$WORK/.deepgemm_cache"
export DG_CACHE_DIR="$WORK/.deepgemm_cache"
# Reduce CUDA allocator fragmentation: the GMU=0.95 OOM had 294 MiB reserved-but-unallocated +
# CUDA-graph private pools, and the alloc was only 0.16 GiB short. expandable_segments reclaims that.
# NOTE: expandable_segments (cuMemMap virtual memory) is MUTUALLY EXCLUSIVE with vLLM's custom
# all-reduce (needs CUDA IPC handles, which cuMemMap allocations can't yield) — job 900448 died with
# `custom_all_reduce.cuh:455 'invalid argument'` during KV profiling. Hence --disable-custom-all-reduce
# on the serve cmd below (TP all-reduce falls back to PYNCCL). Keep these two paired.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs reports "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$DG_JIT_CACHE_DIR"
# Per-job server log. MUST be unique per job: the health loop greps it for failure strings, and a
# REUSED logs/vllm_glm52.log let a stale "Engine core init failed" from a prior job trip a FALSE
# early-abort that killed a healthy serve before the model loaded (job 845371, 2026-06-24).
VLOG="logs/vllm_glm52.${SLURM_JOB_ID}.log"

source "$VENV/bin/activate"
echo "=== allocation: $NODES nodes x $GPUS_PER_NODE H100 = $((NODES*GPUS_PER_NODE)) GPUs | TP=$TP PP=$PP ==="
python -c "import vllm; print('vllm', vllm.__version__)"

# --- Ray cluster over the SLURM allocation -------------------------------------------------
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head="${nodes[0]}"
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head" hostname --ip-address | awk '{print $1}')
rport=6379
# Ray plasma/raylet sockets are AF_UNIX paths, hard-capped at 107 bytes. $JOBSCRATCH on Jean Zay
# is a long Lustre path (/lustre/fsn1/jobscratch/<user>_<jobid>); Ray nests
# session_<timestamp>/sockets/plasma_store under RAY_TMPDIR, which overflowed 107 bytes and killed
# EngineCore at init (job 843820, 2026-06-24). Use a SHORT node-local path instead — the job is
# exclusive so there's no collision, and Ray only keeps small sockets/logs here.
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID}"
echo "=== ray head @ $head ($head_ip:$rport) ==="
# --overlap: these per-node `ray start --block` srun steps run concurrently inside one allocation;
# without --overlap a step can block on the job's step-resources and the worker never joins, leaving
# Ray with only the head's GPUs (job 853597, 2026-06-24: vLLM saw 4/12 GPUs, then a 30-min
# placement-group timeout). --overlap lets all node-steps coexist.
srun --overlap --nodes=1 --ntasks=1 -w "$head" \
  ray start --head --node-ip-address="$head_ip" --port=$rport \
            --num-gpus=$GPUS_PER_NODE --temp-dir="$RAY_TMPDIR" --block &
sleep 20
for ((i=1; i<NODES; i++)); do
  echo "=== ray worker @ ${nodes[$i]} -> $head_ip:$rport ==="
  srun --overlap --nodes=1 --ntasks=1 -w "${nodes[$i]}" \
    ray start --address="$head_ip:$rport" --num-gpus=$GPUS_PER_NODE --block &
  sleep 8
done
SERVER=""
# Teardown must NOT hang: a plain `srun ray stop` can block on step-resources and zombie the job for
# the rest of its walltime after a failure (job 853597 ran ~70 min dead). timeout + --overlap caps it;
# SLURM reaps the backgrounded --block steps once this script exits.
trap 'kill ${SERVER:-} 2>/dev/null || true; timeout 45 srun --overlap --ntasks=$NODES --ntasks-per-node=1 ray stop 2>/dev/null || true' EXIT

# CRITICAL for multi-node: without RAY_ADDRESS, vLLM's ray.init() does NOT attach to the cluster we
# just built — it spins up its OWN local 4-GPU instance on the head, then waits 1800 s for a 12-GPU
# placement group that can never form, and dies (job 869660, 2026-06-25: log says "Started a local
# Ray instance"). Pin the head's GCS address so vLLM (and `ray status`) join OUR cluster. VLLM_HOST_IP
# pins the node IP too: compute nodes are offline, so vLLM's get_ip() (which probes 8.8.8.8) fails and
# defaults to 0.0.0.0, which breaks driver/worker addressing on multi-node.
export RAY_ADDRESS="${head_ip}:${rport}"
export VLLM_HOST_IP="${head_ip}"
echo "=== RAY_ADDRESS=$RAY_ADDRESS  VLLM_HOST_IP=$VLLM_HOST_IP ==="

# Gate on Ray actually registering ALL GPUs before serving. The old code slept a fixed time then ran
# `vllm serve` blindly; if workers were slow/failed to join, vLLM ate a 1800 s placement-group timeout
# instead of failing fast. Poll `ray status` until it reports the full count, else abort with a clear msg.
EXPECT_GPUS=$((NODES*GPUS_PER_NODE))
echo "=== waiting for Ray to register $EXPECT_GPUS GPUs across $NODES nodes ==="
ngpu=0
for i in $(seq 1 60); do
  # MUST stay set -e-safe: early on `ray status` (or the grep) fails because the cluster isn't up,
  # and a bare `ngpu=$(failing pipeline)` trips `set -euo pipefail` and KILLS the job on iteration 1
  # before the gate can do its job (job 859580, 2026-06-24: died at ~15 s, exit 1, no gate verdict).
  # The trailing `|| echo 0` makes the whole pipeline exit 0 and yields a numeric default.
  ngpu=$(ray status 2>/dev/null | grep -oE '[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ GPU' | head -1 | sed -E 's#.*/([0-9]+)\.[0-9]+ GPU#\1#' || echo 0)
  ngpu="${ngpu:-0}"
  if [ "$ngpu" -ge "$EXPECT_GPUS" ] 2>/dev/null; then
    echo "Ray registered $ngpu/$EXPECT_GPUS GPUs (~$((i*3))s)"; break
  fi
  if [ "$i" -eq 60 ]; then
    echo "!! Ray registered only $ngpu/$EXPECT_GPUS GPUs after 180s — worker node(s) failed to join; aborting before the 30-min vLLM PG timeout"
    ray status || true; exit 1
  fi
  sleep 3
done
ray status || true

if [ "$BACKEND" = "http" ]; then
  # --- DEFAULT: serve on localhost across the Ray cluster, then drive the harness over it -----
  # The server is bound to the node only (internal IPC, not an exposed endpoint). glm45 splits
  # reasoning server-side; the 744 GB weights load once and the harness hits them per paper.
  echo "=== serve $MODEL (TP=$TP, PP=$PP, max_len=$MAXLEN, kv=$KVDTYPE) — first load of ~744 GB is slow ==="
  vllm serve "$MODEL" \
    --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --disable-custom-all-reduce \
    --kv-cache-dtype "$KVDTYPE" --max-model-len "$MAXLEN" \
    --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$GMU" \
    --reasoning-parser glm45 --trust-remote-code --port "$PORT" \
    > "$VLOG" 2>&1 &
  SERVER=$!

  echo "=== wait for health (init can take ~30-45 min: 744 GB load over Lustre + PP graph + cudagraph); log=$VLOG ==="
  for i in $(seq 1 720); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "healthy ~$((i*5))s"; break; }
    grep -qiE "unrecognized arguments|invalid choice|out of memory|CUDA error|Engine core init.*failed|RayActorError|No available memory" "$VLOG" 2>/dev/null \
      && { echo "!! server failed early — see $VLOG"; tail -60 "$VLOG"; exit 1; }
    [ "$i" -eq 720 ] && { echo "!! health timeout (60 min)"; tail -60 "$VLOG"; exit 1; }
    sleep 5
  done

  echo "=== verify (model=$SERVED, reasoning_effort=$EFFORT, self_verify=$SELF_VERIFY): $PAPERS ==="
  export MATHPOC_BASE_URL="http://localhost:$PORT/v1" MATHPOC_MODEL="$SERVED"
  python -m mathpoc verify $PAPERS \
    --backend http --thinking-style glm $SV_FLAG \
    --reasoning-effort "$EFFORT" --temperature 1.0 --top-p 1.0 --max-tokens "$MAXTOK"

else
  # --- EXPERIMENTAL (BACKEND=vllm): in-process offline batch, no server ----------------------
  # Harness loads the model in-process and places TP×PP over the Ray cluster; reasoning is split
  # heuristically (no server-side glm45 parser). Use once the http path above is proven.
  echo "=== GLM-5.2 offline batch (in-process, TP=$TP PP=$PP, max_len=$MAXLEN, kv=$KVDTYPE): $PAPERS ==="
  export MATHPOC_MODEL="$MODEL" MATHPOC_TP="$TP" MATHPOC_PP="$PP"
  export MATHPOC_MAX_MODEL_LEN="$MAXLEN" MATHPOC_KV_DTYPE="$KVDTYPE" MATHPOC_GPU_MEM_UTIL="$GMU"
  export MATHPOC_DIST_BACKEND=ray
  python -m mathpoc verify $PAPERS \
    --backend vllm --thinking-style glm $SV_FLAG \
    --reasoning-effort "$EFFORT" --temperature 1.0 --top-p 1.0 --max-tokens "$MAXTOK"
fi
echo "=== done; reports in $PROJ/reports/ ; releasing GPUs ==="
