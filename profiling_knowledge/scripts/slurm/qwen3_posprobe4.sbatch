#!/bin/bash
#SBATCH --job-name=q3-posprobe4
#SBATCH --partition=XAI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/%x-%j.out
# Effect-2 fix validation (13_run_position_anomalies_root_cause.md s5): DEBUG_CLR_MAX_BATCH_SIZE raised from the ROCclr default
# of 1000 so the runtime's batch-flush barrier never falls inside a timed block. Same posprobe.py as jobs 21491/21494/21495;
# a same-job control run without the flag. GPU 0 of the node.
set -euo pipefail
FRONTIER_ROOT="${FRONTIER_ROOT:-/opt/shared/frontier-qwen3-profiling/Frontier}"
IMAGE="${LINEAR_IMAGE:-lmsysorg/sglang:v0.5.11-rocm700-mi35x}"
OUT="${PP_OUT:-data/profiling_posprobe}"
BATCH="${CLR_BATCH:-1000000}"
cd "$FRONTIER_ROOT"
mkdir -p "$OUT/rocprof" .rocprofv3; chmod -R a+rwX "$OUT" .rocprofv3 2>/dev/null || true
echo "host=$(hostname) job=${SLURM_JOB_ID} image=$IMAGE out=$OUT DEBUG_CLR_MAX_BATCH_SIZE=$BATCH"
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G \
  -v "$PWD":/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier \
  -e HSA_ENABLE_COREDUMP=0 -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e OUT="$OUT" -e BATCH="$BATCH" \
  "$IMAGE" bash -c '
    set -uo pipefail
    P=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/posprobe.py  # (the cluster job ran it from $OUT/posprobe.py, byte-identical)
    run() { local name=$1; local envs=$2; shift 2
      echo "=== $name $(date +%s.%N)"
      env $envs python $P --out $OUT/$name.jsonl "$@" 2>$OUT/stderr_$name.log | grep -E "^task|gpu_backlog" || echo "!!! $name failed (see stderr_$name.log)"; }
    TOK="32 64 256 1024 2048 4096 6144"
    run R23_batchfix_s25_tp8  "DEBUG_CLR_MAX_BATCH_SIZE=$BATCH FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 8 --tokens $TOK
    run R24_batchfix_s25_tp1  "DEBUG_CLR_MAX_BATCH_SIZE=$BATCH FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 1 --tokens $TOK
    run R25_batchfix_s200_tp8 "DEBUG_CLR_MAX_BATCH_SIZE=$BATCH FRONTIER_LINEAR_ACTIVE_STEPS=200" --tp 8 --tokens 64 512 2048
    run R27_control_s25_tp8   "FRONTIER_LINEAR_ACTIVE_STEPS=25" --tp 8 --tokens $TOK
    echo "=== R26 amdlog with fix $(date +%s.%N)"
    DEBUG_CLR_MAX_BATCH_SIZE=$BATCH AMD_LOG_LEVEL=4 python $P --out $OUT/R26_batchfix_amdlog_tp8.jsonl --tp 8 --tokens 64 --marks 2>$OUT/amdlog_R26_batchfix_tp8.log | grep -E "^task|gpu_backlog" || echo "!!! R26 failed"
    echo "InternalMarker-inside-launch count (fix):    $(grep -c "InternalMarker" $OUT/amdlog_R26_batchfix_tp8.log) (all internal markers incl. event records)"
    echo "BarrierAND packets (fix):                    $(grep -c "BarrierAND Header" $OUT/amdlog_R26_batchfix_tp8.log)"
    echo "BarrierAND packets (R11 without fix, ref):   $(grep -c "BarrierAND Header" $OUT/amdlog_R11_tp8.log)"
    echo "=== done $(date +%s.%N)"
  '
echo "posprobe4 job done"
