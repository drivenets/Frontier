#!/bin/bash
#SBATCH --job-name=q3-posprobe
#SBATCH --partition=XAI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/%x-%j.out
# Positional-anomaly investigation (run-25 block drift / run-20 attn_pre_proj bump), 2026-09-22.
# Single-process runs of one worker's task loop on GPU 0 with interventions; see posprobe.py for the modes.
set -euo pipefail
FRONTIER_ROOT="${FRONTIER_ROOT:-/opt/shared/frontier-qwen3-profiling/Frontier}"
IMAGE="${LINEAR_IMAGE:-lmsysorg/sglang:v0.5.11-rocm700-mi35x}"
OUT="${PP_OUT:-data/profiling_posprobe}"
cd "$FRONTIER_ROOT"
mkdir -p "$OUT/rocprof" .rocprofv3 && chmod -R a+rwX "$OUT" .rocprofv3
echo "host=$(hostname) job=${SLURM_JOB_ID} image=$IMAGE out=$OUT"
rocm-smi --showclocks -d 0 2>/dev/null | grep -iE 'sclk|mclk|fclk' | tr -s ' ' || true
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G \
  -v "$PWD":/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier \
  -e HSA_ENABLE_COREDUMP=0 -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e OUT="$OUT" -e JOB="${SLURM_JOB_ID}" \
  "$IMAGE" bash -c '
    set -uo pipefail
    P=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/posprobe.py  # (the 2026-09-22 jobs ran it from $OUT/posprobe.py, byte-identical)
    run() { # name, extra env (string), python args...
      local name=$1; local envs=$2; shift 2
      echo "=== $name $(date +%s.%N)"
      env $envs python $P --out $OUT/$name.jsonl "$@" 2>$OUT/stderr_$name.log | grep -E "^task|gpu_backlog|busy_backlog" || echo "!!! $name failed (see stderr_$name.log)"
    }
    TOK10="32 64 128 256 512 1024 2048 3072 4096 6144"
    TOK5="64 256 1024 2048 4096"
    run R1_default_tp8   "A=1" --tp 8 --tokens $TOK10
    run R2_default_tp1   "A=1" --tp 1 --tokens $TOK10
    run R3_default_tp2   "A=1" --tp 2 --tokens 64 512 2048
    run R4_busy_tp8      "A=1" --tp 8 --tokens $TOK5 --mode busy_backlog
    run R5_backlog10_tp8 "FRONTIER_GPU_BACKLOG_MS=10" --tp 8 --tokens $TOK5
    run R6_backlog400_tp8 "FRONTIER_GPU_BACKLOG_MS=400" --tp 8 --tokens $TOK5
    run R7_warmup6_tp8   "A=1" --tp 8 --tokens $TOK5 --mode warmup6
    run R8_hostsleep_tp8 "A=1" --tp 8 --tokens $TOK5 --mode hostsleep
    run R12_steps200_tp8 "FRONTIER_LINEAR_ACTIVE_STEPS=200" --tp 8 --tokens 64 512 2048
    echo "=== R9 trace tp8 $(date +%s.%N)"
    rocprofv3 --kernel-trace --hip-trace --marker-trace --output-format csv -d /workspace/frontier/$OUT/rocprof -o R9_tp8 -- \
      python $P --out $OUT/R9_trace_tp8.jsonl --tp 8 --tokens 64 512 2048 --marks 2>$OUT/stderr_R9.log | grep -E "^task|gpu_backlog" || echo "!!! R9 failed"
    echo "=== R10 trace tp1 $(date +%s.%N)"
    rocprofv3 --kernel-trace --hip-trace --marker-trace --output-format csv -d /workspace/frontier/$OUT/rocprof -o R10_tp1 -- \
      python $P --out $OUT/R10_trace_tp1.jsonl --tp 1 --tokens 64 512 2048 --marks 2>$OUT/stderr_R10.log | grep -E "^task|gpu_backlog" || echo "!!! R10 failed"
    echo "=== R11 amdlog tp8 $(date +%s.%N)"
    AMD_LOG_LEVEL=4 python $P --out $OUT/R11_amdlog_tp8.jsonl --tp 8 --tokens 64 --marks 2>$OUT/amdlog_R11_tp8.log | grep -E "^task|gpu_backlog" || echo "!!! R11 failed"
    echo "=== done $(date +%s.%N)"; ls -la $OUT $OUT/rocprof | head -60
  '
echo "posprobe job done"
