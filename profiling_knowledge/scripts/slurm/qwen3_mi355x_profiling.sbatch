#!/bin/bash
#SBATCH --job-name=q3-prof
#SBATCH --partition=XAI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/%x-%j.out
# -t is passed per submission (pilot 00:30:00, linear_op 06:00:00, attention cells sized from the pilot).
#
# Qwen3-30B-A3B MI355X profiling — one job per STAGE:
#   STAGE=pilot      attention, TP=1, 2048 ctx, block 16, own trees -> seconds per attempt
#   STAGE=linear_op  attn_pre_proj/attn_post_proj (+rope, norms, emb) on the 16k token grid
#   STAGE=attention  one AITER cell: AITER_BLOCK_SIZES=1 or 16 (AITER_TPS/WORK_DIR/COLLECT_DIR overridable)
# Submit from the VM after rsync, e.g.
#   ssh cluster "sudo -u dn sbatch -t 00:30:00 -J q3-pilot --export=ALL,STAGE=pilot <this file>"
# See profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/01_plan.md, Step 3.
set -euo pipefail

STAGE="${STAGE:?set STAGE=pilot|linear_op|attention}"
FRONTIER_ROOT="${FRONTIER_ROOT:-/opt/shared/frontier-qwen3-profiling/Frontier}"
# v0.5.11-rocm700's aiter (Apr 2026) faults in mha_batch_prefill for head_dim 128 with paged KV (pilot jobs 21300/21303, repro 21302/21304);
# v0.5.17-rocm720 runs the same grid clean at block 1 and 16.
IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.17-rocm720-mi35x}"
# linear_op needs vllm, which v0.5.17 no longer ships; v0.5.11 has it and linear_op never touches the faulty prefill kernel.
LINEAR_IMAGE="${LINEAR_IMAGE:-lmsysorg/sglang:v0.5.11-rocm700-mi35x}"
MODEL="${MODEL:-qwen3-a3b-30b-moe}"
GPU_IDS="0,1,2,3,4,5,6,7"
# aiter JIT cache is per node; /mnt/data/aiter-cache exists on some nodes only, and dn cannot create it (job 21283)
AITER_CACHE_DIR=/mnt/data/aiter-cache
[ -d "$AITER_CACHE_DIR" ] && [ -w "$AITER_CACHE_DIR" ] || AITER_CACHE_DIR="/tmp/aiter-cache-$(id -un)"
DOCKER_GPU_FLAGS=(--device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G)

cd "$FRONTIER_ROOT"
# /opt/shared is NFS with root_squash: the container runs as root, which the server maps to nobody, so every
# directory the profiler writes into must exist and be world-writable before docker starts (pilot job 21280 died on mkdir).
mkdir -p "$AITER_CACHE_DIR" data/profiling/sweep_work/logs data/profiling/sweep_work_pilot data/profiling_pilot \
  "${WORK_DIR:-data/profiling/sweep_work}" "${COLLECT_DIR:-data/profiling}/compute/mi355x/$MODEL"
chmod -R a+rwX data/profiling data/profiling_pilot "${WORK_DIR:-data/profiling/sweep_work}" "${COLLECT_DIR:-data/profiling}" 2>/dev/null || true   # dirs a previous container created belong to nobody; leave them
[ "$STAGE" = linear_op ] && IMAGE="$LINEAR_IMAGE"
docker image inspect "$IMAGE" >/dev/null || { echo "image $IMAGE not on $(hostname)" >&2; exit 1; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} stage=$STAGE aiter_cache=$AITER_CACHE_DIR commit=${FRONTIER_COMMIT:-unknown (pass --export=ALL,FRONTIER_COMMIT=<sha>)} image=$IMAGE"
rocm-smi --showmemuse 2>/dev/null | grep -i "used" || true

run_attention() {
  MODELS="$MODEL" BACKENDS=AITER AITER_NUM_GPUS="${AITER_NUM_GPUS:-4}" \
  PREWARM_NQ=32 PREWARM_NKV=4 PREWARM_HD=128 \
  DOCKER_IMAGE="$IMAGE" AITER_CACHE_DIR="$AITER_CACHE_DIR" GPU_IDS="$GPU_IDS" \
  "$@" bash profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh --docker
}

case "$STAGE" in
  pilot)
    run_attention env AITER_TPS=1 MAX_SEQ_LEN=2048 MAX_MODEL_LEN=2048 AITER_BLOCK_SIZES=16 \
      DECODE_KV_CACHE_SIZE_LIST="128 512 1024 2048" TRUE_MIXED_PREFILL_CHUNK_SIZES=1024 \
      TRUE_MIXED_DECODE_KV_CACHE_SIZES="128 512 1024 2048" \
      WORK_DIR=data/profiling/sweep_work_pilot COLLECT_DIR=data/profiling_pilot
    # per-cell logs land in $WORK_DIR/logs (sweep default), so no job can truncate another's
    test -s "data/profiling_pilot/compute/mi355x/$MODEL/attention_combined_aiter_block16.csv" \
      || { echo "pilot produced no collected CSV" >&2; exit 1; }
    ;;
  attention)
    # MAX_SEQ_LEN, DECODE_KV_CACHE_SIZE_LIST, TRUE_MIXED_* are taken from the job environment when set (--export=ALL)
    run_attention env AITER_TPS="${AITER_TPS:-1 2 4 8}" MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}" MAX_MODEL_LEN="${MAX_SEQ_LEN:-16384}" \
      AITER_BLOCK_SIZES="${AITER_BLOCK_SIZES:?set AITER_BLOCK_SIZES=1 or 16}" \
      WORK_DIR="${WORK_DIR:-data/profiling/sweep_work}" COLLECT_DIR="${COLLECT_DIR:-data/profiling}"
    # the sweep is deliberately non-fatal per cell; make Slurm report the failure
    test -s "${COLLECT_DIR:-data/profiling}/compute/mi355x/$MODEL/attention_combined_aiter_block${AITER_BLOCK_SIZES}.csv" \
      || { echo "cell block${AITER_BLOCK_SIZES} produced no collected CSV" >&2; exit 1; }
    ;;
  linear_op)
    # Token grid: dense by default (every count to 2048, every 8 to 8192, every 16 to 16384; 4000 excluded = known GPU fault),
    # 3,327 values; a strict superset of the profiler's 386-value grid. Override with LINEAR_TOKENS_PY (a python expression).
    TOKENS_PY="${LINEAR_TOKENS_PY:-sorted((set(range(1, 2049)) | set(range(2056, 8193, 8)) | set(range(8208, 16385, 16))) - set([4000]))}"  # no braces: they end the \${var:-} default early
    TOKENS=$(docker run --rm -v "$PWD":/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier "$IMAGE" \
      python -c "print(*$TOKENS_PY)")
    docker run --rm "${DOCKER_GPU_FLAGS[@]}" \
      -v "$PWD":/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier \
      -e FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1 -e HSA_ENABLE_COREDUMP=0 -e CUDA_VISIBLE_DEVICES="$GPU_IDS" -e HIP_VISIBLE_DEVICES="$GPU_IDS" \
      "$IMAGE" python -m frontier.profiling.linear_op.main --disable_ray --yes --device mi355x --models "$MODEL" --is_moe \
        --num_gpus "${LINEAR_NUM_GPUS:-8}" --num_tensor_parallel_workers 1 2 4 8 --precision BF16 --profile_method cuda_event \
        --max_tokens 16384 --num_tokens_list $TOKENS --output_dir "${COLLECT_DIR:-data/profiling}" \
      > data/profiling/sweep_work/logs/linear_op.log 2>&1
    ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
echo "stage $STAGE done"
