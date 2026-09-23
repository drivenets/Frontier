#!/bin/bash
#SBATCH --job-name=q3-posprobe3
#SBATCH --partition=XAI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/%x-%j.out
# Third posprobe job: (E1) device-side shader-clock trajectory across the spun block, (E2) GEMM backlog with a corrected
# calibration, both at 25 timed forwards (the dense_fixed collection's setting) and at 50; TP8 and TP1; GPU 0.
set -euo pipefail
FRONTIER_ROOT="${FRONTIER_ROOT:-/opt/shared/frontier-qwen3-profiling/Frontier}"
IMAGE="${LINEAR_IMAGE:-lmsysorg/sglang:v0.5.11-rocm700-mi35x}"
OUT="${PP_OUT:-data/profiling_posprobe}"
cd "$FRONTIER_ROOT"
mkdir -p "$OUT/rocprof" .rocprofv3; chmod -R a+rwX "$OUT" .rocprofv3 2>/dev/null || true  # files a previous container created belong to nobody
echo "host=$(hostname) job=${SLURM_JOB_ID} image=$IMAGE out=$OUT"
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G \
  -v "$PWD":/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier \
  -e HSA_ENABLE_COREDUMP=0 -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e OUT="$OUT" \
  "$IMAGE" bash -c '
    set -uo pipefail
    P=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/posprobe.py  # (the 2026-09-22 jobs ran it from $OUT/posprobe.py, byte-identical)
    run() { local name=$1; local envs=$2; shift 2
      echo "=== $name $(date +%s.%N)"
      env $envs python $P --out $OUT/$name.jsonl "$@" 2>$OUT/stderr_$name.log | grep -E "^task|gpu_backlog|busy_backlog" || echo "!!! $name failed (see stderr_$name.log)"; }
    TOK="32 64 256 1024 2048 4096 6144"
    run R15_clock_s25_tp8  "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 8 --tokens $TOK --mode clockprobe
    run R16_clock_s25_tp1  "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 1 --tokens $TOK --mode clockprobe
    run R17_busy_s25_tp8   "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 8 --tokens $TOK --mode busy_backlog
    run R18_busy_s25_tp1   "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 1 --tokens $TOK --mode busy_backlog
    run R19_default_s25_tp8 "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 8 --tokens $TOK
    run R20_default_s25_tp1 "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 1 --tokens $TOK
    run R21_clock_s50_tp8  "A=1" --tp 8 --tokens 64 1024 4096 --mode clockprobe
    run R22_busy_s50_tp8   "A=1" --tp 8 --tokens 64 1024 4096 --mode busy_backlog
    echo "=== done $(date +%s.%N)"; ls -la $OUT | head -60
  '
echo "posprobe3 job done"
