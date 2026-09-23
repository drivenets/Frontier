#!/bin/bash
#SBATCH --job-name=q3-posprobe6
#SBATCH --partition=XAI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/%x-%j.out
# Effect-1 fix validation, second version (13_run_position_anomalies_root_cause.md s7): GEMM work backlog instead of the _sleep spin
# (linear_op_wrapper.BACKLOG_KIND=gemm) + 3 settle forwards, with the Effect-2 flag on. Same-job control: sleep spin, no settle. GPU 0.
set -euo pipefail
FRONTIER_ROOT="${FRONTIER_ROOT:-/opt/shared/frontier-qwen3-profiling/Frontier}"
IMAGE="${LINEAR_IMAGE:-lmsysorg/sglang:v0.5.11-rocm700-mi35x}"
OUT="${PP_OUT:-data/profiling_posprobe}"
cd "$FRONTIER_ROOT"
mkdir -p "$OUT/rocprof" .rocprofv3; chmod -R a+rwX "$OUT" .rocprofv3 2>/dev/null || true
echo "host=$(hostname) job=${SLURM_JOB_ID} image=$IMAGE out=$OUT wrapper_md5=$(md5sum frontier/profiling/linear_op/linear_op_wrapper.py | cut -c1-8)"
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G \
  -v "$PWD":/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier \
  -e HSA_ENABLE_COREDUMP=0 -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e OUT="$OUT" -e DEBUG_CLR_MAX_BATCH_SIZE=1000000 \
  "$IMAGE" bash -c '
    set -uo pipefail
    P=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/posprobe.py  # (the cluster job ran it from $OUT/posprobe.py, byte-identical)
    run() { local name=$1; local envs=$2; shift 2
      echo "=== $name $(date +%s.%N)"
      env $envs python $P --out $OUT/$name.jsonl "$@" 2>$OUT/stderr_$name.log | grep -E "^task|gpu_backlog" || echo "!!! $name failed (see stderr_$name.log)"; }
    TOK="32 64 256 1024 2048 4096 6144"
    run R35_gemm_s25_tp8      "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 8 --tokens $TOK
    run R36_gemm_s25_tp1      "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 1 --tokens $TOK
    run R37_gemm_clock_tp8    "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 8 --tokens $TOK --mode clockprobe
    run R38_gemm_clock_tp1    "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 1 --tokens $TOK --mode clockprobe
    run R39_sleep_control_tp8 "FRONTIER_LINEAR_ACTIVE_STEPS=25 FRONTIER_GPU_BACKLOG_KIND=sleep FRONTIER_LINEAR_SETTLE_STEPS=0" --tp 8 --tokens $TOK
    run R40_sleep_control_tp1 "FRONTIER_LINEAR_ACTIVE_STEPS=25 FRONTIER_GPU_BACKLOG_KIND=sleep FRONTIER_LINEAR_SETTLE_STEPS=0" --tp 1 --tokens $TOK
    run R41_gemm_s200_tp8     "FRONTIER_LINEAR_ACTIVE_STEPS=200" --tp 8 --tokens 64 2048
    echo "=== done $(date +%s.%N)"
  '
echo "posprobe6 job done"
