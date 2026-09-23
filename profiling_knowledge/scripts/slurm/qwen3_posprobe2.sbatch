#!/bin/bash
#SBATCH --job-name=q3-posprobe2
#SBATCH --partition=XAI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/%x-%j.out
# Follow-up to posprobe.sbatch: rocprofv3 traces of the 200-step (8 spun blocks of 25) loop, TP8 and TP2, GPU 0.
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
    echo "=== R13 trace steps200 tp8 $(date +%s.%N)"
    FRONTIER_LINEAR_ACTIVE_STEPS=200 rocprofv3 --kernel-trace --hip-trace --marker-trace --output-format csv -d /workspace/frontier/$OUT/rocprof -o R13_s200_tp8 -- \
      python $P --out $OUT/R13_trace_s200_tp8.jsonl --tp 8 --tokens 64 512 --marks 2>$OUT/stderr_R13.log | grep -E "^task|gpu_backlog" || echo "!!! R13 failed"
    echo "=== R14 trace steps200 tp2 $(date +%s.%N)"
    FRONTIER_LINEAR_ACTIVE_STEPS=200 rocprofv3 --kernel-trace --hip-trace --marker-trace --output-format csv -d /workspace/frontier/$OUT/rocprof -o R14_s200_tp2 -- \
      python $P --out $OUT/R14_trace_s200_tp2.jsonl --tp 2 --tokens 64 512 --marks 2>$OUT/stderr_R14.log | grep -E "^task|gpu_backlog" || echo "!!! R14 failed"
    echo "=== done $(date +%s.%N)"; ls -la $OUT/rocprof | head -40
  '
echo "posprobe2 job done"
