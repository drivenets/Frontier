#!/bin/bash
#SBATCH --job-name=q3-fixedclk
#SBATCH --partition=XAI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/%x-%j.out
# Item 2 of the 2026-09-17 follow-ups: validation grid at a LOCKED shader clock (rocm-smi --setperfdeterminism), two-pass profiler
# (legacy + GPU-bound) plus rocprofv3 kernel traces with/without the spin, to separate the clock share of the no-knob/knob residual.
# The lock is per node and undone by the trap on any exit. Requires the whole node (gres gpu:8) so no other job sees the locked clocks.
set -uo pipefail
cd /opt/shared/frontier-qwen3-profiling/Frontier
CLK="${FIXED_SCLK_MHZ:-1900}"
reset_clocks() { for g in 0 1 2 3 4 5 6 7; do rocm-smi --resetperfdeterminism -d $g >/dev/null 2>&1; rocm-smi --setperflevel auto -d $g >/dev/null 2>&1; done; rocm-smi --showperflevel | grep -i level | head -8; }
trap reset_clocks EXIT
echo "host=$(hostname) job=$SLURM_JOB_ID lock=${CLK}MHz"
for g in 0 1 2 3 4 5 6 7; do rocm-smi --setperfdeterminism "$CLK" -d $g | grep -iE "success|error" ; done
rocm-smi --showperflevel | grep -i level | head -8
export STAGE=linear_op FRONTIER_COMMIT=d4469be+fixedclk LINEAR_TOKENS_PY="[1,8,64,3072,4096,4192,6144,8192]" COLLECT_DIR=data/profiling_fixedclk LINEAR_LOG_SUFFIX=_fixedclk
unset LINEAR_CMD_PREFIX LINEAR_TPS LINEAR_DOCKER_ENV
bash profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch   # two-pass profiler, 8 workers, locked clocks
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G \
  -v "$PWD":/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier \
  -e FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1 -e HSA_ENABLE_COREDUMP=0 -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  lmsysorg/sglang:v0.5.11-rocm700-mi35x bash -c '
    T=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/trace_ops.py; TOK="1 8 64 3072 4096 4192 6144 8192"
    for TP in 2 4 8; do
      rocprofv3 --kernel-trace --marker-trace --output-format csv -d /workspace/frontier/data/profiling_fixedclk/rocprof -o c_tp$TP -- python $T --tp $TP --tokens $TOK 2>/dev/null | grep "^{" > data/profiling_fixedclk/c_tp$TP.jsonl
    done; ls -la data/profiling_fixedclk/rocprof | head'
echo "fixed-clock job done"
