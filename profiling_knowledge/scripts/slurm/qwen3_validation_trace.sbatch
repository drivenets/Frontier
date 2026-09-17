#!/bin/bash
#SBATCH --job-name=q3-val-trace
#SBATCH --partition=XAI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/%x-%j.out
# 08_measurement_validation_preregistration.md, Phases 1-2: single-process runs of one worker's task loop on GPU 0 in four
# modes: a) event timing, b) event timing + FRONTIER_GPU_BACKLOG_MS=100, c) rocprofv3 kernel trace, c') trace + knob.
# rocm-smi sclk is sampled on the node every 0.5 s for the whole job (DVFS check D).
set -euo pipefail
FRONTIER_ROOT="${FRONTIER_ROOT:-/opt/shared/frontier-qwen3-profiling/Frontier}"
IMAGE="${LINEAR_IMAGE:-lmsysorg/sglang:v0.5.11-rocm700-mi35x}"
OUT="${VAL_OUT:-data/profiling_val_trace}"
TOK="${VAL_TOKENS:-1 8 64 3072 4096 4192 6144 8192}"
TPS="${VAL_TPS:-1 2 4 8}"
cd "$FRONTIER_ROOT"
mkdir -p "$OUT" .rocprofv3 && chmod -R a+rwX "$OUT" .rocprofv3
echo "host=$(hostname) job=${SLURM_JOB_ID} image=$IMAGE out=$OUT tokens=[$TOK] tps=[$TPS]"
( while true; do echo "$(date +%s.%N) $(rocm-smi --showclocks -d 0 2>/dev/null | grep -iE 'sclk|mclk' | tr -s ' ' | tr '\n' ' ')"; sleep 0.5; done ) > "$OUT/rocm_smi_clocks_${SLURM_JOB_ID}.log" &
SMI_PID=$!
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G \
  -v "$PWD":/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier \
  -e FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1 -e HSA_ENABLE_COREDUMP=0 -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  -e OUT="$OUT" -e TOK="$TOK" -e TPS="$TPS" \
  "$IMAGE" bash -c '
    set -uo pipefail
    T=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/trace_ops.py
    for TP in $TPS; do
      echo "=== mode a (event timing) TP$TP $(date +%s.%N)";  python $T --tp $TP --tokens $TOK 2>>$OUT/stderr_a_tp$TP.log | grep "^{" > $OUT/a_tp$TP.jsonl
      echo "=== mode b (event + backlog) TP$TP $(date +%s.%N)"; FRONTIER_GPU_BACKLOG_MS=100 python $T --tp $TP --tokens $TOK 2>>$OUT/stderr_b_tp$TP.log | grep "^{" > $OUT/b_tp$TP.jsonl
      echo "=== mode c (rocprofv3, no knob) TP$TP $(date +%s.%N)"; rocprofv3 --kernel-trace --marker-trace --output-format csv -d /workspace/frontier/$OUT/rocprof -o c_tp$TP -- python $T --tp $TP --tokens $TOK 2>>$OUT/stderr_c_tp$TP.log | grep "^{" > $OUT/c_tp$TP.jsonl
      echo "=== mode cb (rocprofv3 + backlog) TP$TP $(date +%s.%N)"; FRONTIER_GPU_BACKLOG_MS=100 rocprofv3 --kernel-trace --marker-trace --output-format csv -d /workspace/frontier/$OUT/rocprof -o cb_tp$TP -- python $T --tp $TP --tokens $TOK 2>>$OUT/stderr_cb_tp$TP.log | grep "^{" > $OUT/cb_tp$TP.jsonl
    done
    echo "=== done $(date +%s.%N)"; ls -la $OUT $OUT/rocprof | head -60
  '
kill $SMI_PID || true
echo "trace job done"
