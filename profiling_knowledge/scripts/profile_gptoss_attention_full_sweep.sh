#!/bin/bash
# =============================================================================
# Full attention sweep for gpt-oss on MI355X: AITER *and* TORCH_SDPA
# =============================================================================
# One-shot, wide-coverage attention profiling intended to be run ONCE and not
# re-run: it sweeps backend x block_size x TP over a batch/KV/prefill grid wide
# enough that the trained predictor interpolates rather than extrapolates.
#
# Why both backends: AITER is what real sglang serving dispatches on MI355X
# (mha_batch_prefill_func / paged_attention_ragged), so it is the fidelity
# target. TORCH_SDPA is the portable reference the existing checked-in data was
# collected with. Keeping both makes the "how much did the kernel choice cost
# us" question answerable directly instead of by inference.
#
# Design follows GPTOSS_TRUE_MIXED_BATCH_PROFILING.md's "narrow what's known,
# widen only what's genuinely uncertain" rule, but deliberately widens further
# than that doc's workload-pinned sweep, because the point here is to avoid a
# second collection pass later.
#
# ---------------------------------------------------------------------------
# CHECKPOINTING -- read this before widening anything
# ---------------------------------------------------------------------------
# frontier.profiling.attention.main holds every result in memory and writes its
# CSVs once, at the very end. An interrupted run produces NOTHING, no matter how
# it was supervised. This script's only protection is that it splits the sweep
# into independent (model, backend, block_size) invocations, each with its own
# --output_dir, and skips any whose output already exists. A crash costs you one
# cell, not the sweep. Keep it that way: widening a single cell until it runs
# for hours re-creates exactly the failure mode that cost 86 hours once already.
#
# ---------------------------------------------------------------------------
# TWO REAL CONSTRAINTS ON THIS BOX (both measured, not assumed)
# ---------------------------------------------------------------------------
# 1. TORCH_SDPA MEMORY AT LOW TP. SDPA materialises the score matrix; aiter
#    (flash-style) does not. At max_seq_len 16384 with an 8192-token prefill
#    chunk the peak allocation scales with heads-per-GPU:
#
#        TP=8 -> ~4 GB    TP=4 -> ~8 GB    TP=2 -> ~16 GB    TP=1 -> ~32 GB
#
#    Confirmed failure with the sglang server up (~35 GB free/GPU):
#        torch.OutOfMemoryError: HIP out of memory. Tried to allocate 32.00 GiB.
#        GPU 1 has a total capacity of 287.98 GiB of which 4.31 GiB is free.
#    So SDPA_TPS defaults to TP>=4. Set SDPA_TPS="1 2 4 8" only if the GPUs are
#    actually free. AITER has no such limit and defaults to all four.
#
# 2. AITER JIT TEMPLATE COMPILES. paged_attention_ragged compiles a kernel
#    specialised on (gqa_ratio, head_size, block_size, dtype, partition_size).
#    gqa_ratio is 64/TP, so TP x block_size = 12 distinct templates here. Each
#    takes minutes, and the profiler's 8 GPU workers all block on one FileBaton
#    while it happens. prewarm_aiter() below forces each compile once, in a
#    single process, before the sweep -- and AITER_CACHE_DIR persists them
#    across runs. Without this you pay the compiles inside the sweep, 8-way
#    blocked, and an interrupted compile leaves a lock the next run waits on.
#
# ---------------------------------------------------------------------------
# ENVIRONMENT -- this must run in the sglang ROCm image, not on the host
# ---------------------------------------------------------------------------
# aiter needs /opt/rocm/bin/hipconfig for its runtime template compiles. On this
# host /opt/rocm -> /opt/rocm-7.0.1, which has no bin/ (the complete tree is
# /opt/rocm-7.2.4), and aiter hardcodes the path. The lmsysorg/sglang ROCm image
# has a working hipconfig, 60 prebuilt aiter modules, and every dep Frontier's
# profiler needs. See run_in_docker() at the bottom -- calling this script with
# --docker re-executes it inside that image for you.
# =============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- What to sweep -----------------------------------------------------------
MODELS="${MODELS:-openai/gpt-oss-120b}"
BACKENDS="${BACKENDS:-AITER TORCH_SDPA}"
BLOCK_SIZES="${BLOCK_SIZES:-1 16 32}"
# AITER cannot do block_size 32: aiter's mha_batch_prefill_func has no compiled
# kernel instance for it and fails with "invalid argument for batch_prefill" at
# every TP (its decode kernel does support 32; prefill does not). Verified by
# sweeping block_size x TP directly. 1 and 16 are the values that matter anyway
# -- sglang serves at page_size=1, vLLM defaults to 16; 32 exists only for
# backward compatibility with the old DeepSeek data.
AITER_BLOCK_SIZES="${AITER_BLOCK_SIZES:-1 16}"
# gqa_ratio = 64/TP for gpt-oss, so each TP is a separate aiter template.
AITER_TPS="${AITER_TPS:-1 2 4 8}"
SDPA_TPS="${SDPA_TPS:-4 8}"          # see constraint 1; widen only if GPUs are free

DEVICE="${DEVICE:-mi355x}"
NUM_GPUS="${NUM_GPUS:-8}"
# AITER faults with num_gpus=8 and a wide true-mixed grid:
#   Memory access fault by GPU node-{4,5,7} ... Reason: Unknown
# then BrokenProcessPool. Measured boundary on the full 492-point true-mixed
# grid: num_gpus 1/2/4 all complete 492/492; 8 dies at ~483/492. It is not
# per-GPU load (lower concurrency means MORE tasks per GPU and still passes),
# not one bad GPU (faults seen on three different ones), not a single bad shape
# (all 12 extreme combinations pass individually), and not the KV-cache memory
# budget (tested; see get_max_num_blocks in frontier/profiling/utils/__init__.py).
# Root cause unresolved -- suspected interaction with the sglang server resident
# on all 8 GPUs. 4 costs ~2x wall time on a run that takes minutes, so it is
# cheap insurance. Retest with 8 once the GPUs are actually free.
AITER_NUM_GPUS="${AITER_NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
PROFILE_METHOD="${PROFILE_METHOD:-cuda_event}"
PRECISION="${PRECISION:-BF16}"

# --- The grid ----------------------------------------------------------------
# max_seq_len bounds every context the predictor can be queried at without
# extrapolating (a RandomForest just returns its rightmost leaf beyond the
# training range). 16384 covers the real 8192+1024 workload with headroom.
# Raising it grows the standard grid superlinearly: 16384 -> 2,772 points,
# 32768 -> 5,485. Do not raise it "just in case".
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$MAX_SEQ_LEN}"

BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-1 2 4 8 16 24 32 48 64 96 128 160 192 256 320 384 448 512}"
DECODE_KV_CACHE_SIZE_LIST="${DECODE_KV_CACHE_SIZE_LIST:-128 512 1024 2048 4096 8192 16384}"

# Chunked-prefill grid search sweeps prefill chunk size across the whole
# max_seq_len range instead of pinning one value -- this is what makes the sweep
# cover "all relevant sequence lengths" on the prefill side.
ENABLE_CHUNKED_PREFILL_GRID_SEARCH="${ENABLE_CHUNKED_PREFILL_GRID_SEARCH:-true}"
FIXED_CHUNKED_PREFILL_SIZE="${FIXED_CHUNKED_PREFILL_SIZE:-8192}"  # used only if grid search is off

# True-mixed (prefill+decode in one scheduling step). Without these rows any
# concurrent chunked-prefill simulation dies on attn_decode_in_mixed.
TRUE_MIXED_PREFILL_BATCH_SIZES="${TRUE_MIXED_PREFILL_BATCH_SIZES:-1 2}"
TRUE_MIXED_PREFILL_CHUNK_SIZES="${TRUE_MIXED_PREFILL_CHUNK_SIZES:-1024 4096 8192}"
TRUE_MIXED_DECODE_BATCH_SIZES="${TRUE_MIXED_DECODE_BATCH_SIZES:-$BATCH_SIZE_LIST}"
TRUE_MIXED_DECODE_KV_CACHE_SIZES="${TRUE_MIXED_DECODE_KV_CACHE_SIZES:-$DECODE_KV_CACHE_SIZE_LIST}"
TRUE_MIXED_PREFILL_KV_CACHE_SIZE="${TRUE_MIXED_PREFILL_KV_CACHE_SIZE:-0}"

# --- Output ------------------------------------------------------------------
# Each cell writes to its own tree, so the shared-output-path overwrite trap
# (attention.csv keys only on device/model/profile_method -- not block_size, not
# TP, not backend) simply cannot fire. Collection into the canonical taxonomy is
# an explicit, separate step.
WORK_DIR="${WORK_DIR:-$REPO_ROOT/data/profiling/sweep_work}"
COLLECT_DIR="${COLLECT_DIR:-$REPO_ROOT/data/profiling}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
AITER_CACHE_DIR="${AITER_CACHE_DIR:-$HOME/.aiter}"
DRY_RUN="${DRY_RUN:-false}"
DOCKER_IMAGE="${DOCKER_IMAGE:-lmsysorg/sglang:v0.5.9-rocm700-mi35x}"

usage() {
  cat <<EOF
usage: $0 [--docker] [--dry-run] [--prewarm-only] [--collect-only]

  --docker        re-exec this script inside \$DOCKER_IMAGE (required on this
                  host: aiter needs a working /opt/rocm/bin/hipconfig)
  --dry-run       print every resolved command without touching a GPU
  --prewarm-only  only force the aiter JIT template compiles, then stop
  --collect-only  only copy finished per-cell CSVs into the canonical taxonomy

Every knob above is an env var, e.g.:
  SDPA_TPS="1 2 4 8" MAX_SEQ_LEN=32768 $0 --docker
EOF
}

PREWARM_ONLY=false
COLLECT_ONLY=false
USE_DOCKER=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker) USE_DOCKER=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --prewarm-only) PREWARM_ONLY=true; shift ;;
    --collect-only) COLLECT_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

tag_of() { echo "$1" | tr '/' '_'; }
lc() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

# ---------------------------------------------------------------------------
run_in_docker() {
  # --docker launches a container, so it only works from the host. Running it
  # from inside a container fails on `exec: docker: not found` (no docker CLI in
  # the image) and would in any case bind-mount container-local paths that do
  # not exist on the host.
  if [ -f /.dockerenv ] || ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: --docker must be run from the HOST, not from inside a container.

You appear to be inside a container already$([ -f /.dockerenv ] && echo " (/.dockerenv exists)").
REPO_ROOT resolved to '$REPO_ROOT', which is a path inside this container.

Do one of these instead:

  1. From the host (recommended):
       exit                       # leave this container
       cd /home/dn/frontier_work/drivenetsfrontier
       ./profiling_knowledge/scripts/$(basename "$0") --docker

  2. Already inside the RIGHT image ($DOCKER_IMAGE)?
     Drop --docker -- the script then runs the sweep directly:
       ./profiling_knowledge/scripts/$(basename "$0")

     Note the image matters: aiter needs a working /opt/rocm/bin/hipconfig.
     rocm/vllm images have no /opt/rocm and ship aiter with zero prebuilt
     modules, so AITER profiling will not work there.
EOF
    exit 2
  fi

  local args=()
  [ "$DRY_RUN" = "true" ] && args+=(--dry-run)
  [ "$PREWARM_ONLY" = "true" ] && args+=(--prewarm-only)
  [ "$COLLECT_ONLY" = "true" ] && args+=(--collect-only)

  # -it only when there is a real TTY: a multi-hour sweep is normally launched
  # under nohup/tmux, where `docker run -it` dies on "the input device is not a
  # TTY" before profiling anything.
  local -a tty_args=()
  [ -t 0 ] && [ -t 1 ] && tty_args=(-it)

  # --group-add 110 is the numeric GID owning /dev/kfd on this host; the image
  # has no "render" group entry, so the name form fails.
  set -x
  exec docker run --rm "${tty_args[@]}" \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 \
    --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G \
    -v "$REPO_ROOT":/workspace/frontier \
    -v "$AITER_CACHE_DIR":/root/.aiter \
    -w /workspace/frontier \
    -e PYTHONPATH=/workspace/frontier \
    -e CUDA_VISIBLE_DEVICES="$GPU_IDS" -e HIP_VISIBLE_DEVICES="$GPU_IDS" \
    -e MODELS="$MODELS" -e BACKENDS="$BACKENDS" -e BLOCK_SIZES="$BLOCK_SIZES" \
    -e AITER_BLOCK_SIZES="$AITER_BLOCK_SIZES" -e AITER_NUM_GPUS="$AITER_NUM_GPUS" \
    -e HSA_ENABLE_COREDUMP=0 \
    -e AITER_TPS="$AITER_TPS" -e SDPA_TPS="$SDPA_TPS" \
    -e MAX_SEQ_LEN="$MAX_SEQ_LEN" -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e BATCH_SIZE_LIST="$BATCH_SIZE_LIST" \
    -e DECODE_KV_CACHE_SIZE_LIST="$DECODE_KV_CACHE_SIZE_LIST" \
    -e ENABLE_CHUNKED_PREFILL_GRID_SEARCH="$ENABLE_CHUNKED_PREFILL_GRID_SEARCH" \
    -e FIXED_CHUNKED_PREFILL_SIZE="$FIXED_CHUNKED_PREFILL_SIZE" \
    -e TRUE_MIXED_PREFILL_BATCH_SIZES="$TRUE_MIXED_PREFILL_BATCH_SIZES" \
    -e TRUE_MIXED_PREFILL_CHUNK_SIZES="$TRUE_MIXED_PREFILL_CHUNK_SIZES" \
    -e TRUE_MIXED_DECODE_BATCH_SIZES="$TRUE_MIXED_DECODE_BATCH_SIZES" \
    -e TRUE_MIXED_DECODE_KV_CACHE_SIZES="$TRUE_MIXED_DECODE_KV_CACHE_SIZES" \
    -e PROFILE_METHOD="$PROFILE_METHOD" -e PRECISION="$PRECISION" \
    -e NUM_GPUS="$NUM_GPUS" -e DEVICE="$DEVICE" \
    -e AITER_CACHE_DIR=/root/.aiter \
    "$DOCKER_IMAGE" \
    bash -lc "profiling_knowledge/scripts/$(basename "$0") ${args[*]}"
}

# ---------------------------------------------------------------------------
# Force one aiter template compile per (TP, block_size), single-process, before
# the sweep. Each specialisation is keyed on gqa_ratio=num_q_heads/num_kv_heads
# and block_size, so both axes must be covered. Cheap once, painful inside a
# multi-worker run.
prewarm_aiter() {
  case " $BACKENDS " in *" AITER "*) ;; *) return 0 ;; esac
  echo "=== prewarming aiter JIT templates (TP x block_size) ==="
  if [ "$DRY_RUN" = "true" ]; then
    echo "  would prewarm: TP={$AITER_TPS} x block_size={$AITER_BLOCK_SIZES}"
    return 0
  fi
  # Stale locks from an interrupted compile make the next run block on a baton
  # that is never released.
  find "${AITER_CACHE_DIR:-$HOME/.aiter}/build" -name lock -type f -delete 2>/dev/null || true
  AITER_TPS="$AITER_TPS" BLOCK_SIZES="$AITER_BLOCK_SIZES" "$PYTHON_BIN" - <<'PYEOF'
import math, os, sys, torch
from aiter.ops.attention import paged_attention_ragged

DEV, DT, PART = "cuda:0", torch.bfloat16, 256
NQ_TOTAL, NKV_TOTAL, HD = 64, 8, 64   # gpt-oss attention dims (20b and 120b are identical)
for tp in os.environ["AITER_TPS"].split():
    for bs in os.environ["BLOCK_SIZES"].split():
        tp, bs = int(tp), int(bs)
        nq, nkv = NQ_TOTAL // tp, max(1, NKV_TOTAL // tp)
        ctx, B = 4 * bs, 1
        npages = (ctx + bs - 1) // bs
        kv = torch.randn(2, npages, bs, nkv, HD, dtype=DT, device=DEV)
        kc, vc = kv[0].contiguous(), kv[1].contiguous()
        q = torch.randn(B, nq, HD, dtype=DT, device=DEV)
        out = torch.empty(B, nq, HD, dtype=DT, device=DEV)
        mp = (ctx + PART - 1) // PART
        ws = torch.empty((B*nq*mp*HD)*4 + 2*(B*nq*mp)*4, dtype=torch.uint8, device=DEV)
        one = torch.tensor([1.0], dtype=torch.float32, device=DEV)
        paged_attention_ragged(
            out, ws, q, kc, vc, 1.0/math.sqrt(HD),
            torch.tensor([0, npages], dtype=torch.int32, device=DEV),
            torch.arange(npages, dtype=torch.int32, device=DEV),
            torch.tensor([bs], dtype=torch.int32, device=DEV),
            bs, mp, None, "auto", "NHD", 0.0, one, one, None, PART)
        torch.cuda.synchronize()
        print(f"  warm: TP={tp} gqa_ratio={nq//nkv} block_size={bs}", flush=True)
PYEOF
  echo "=== prewarm complete ==="
}

# ---------------------------------------------------------------------------
profile_cell() {
  local model="$1" backend="$2" block="$3" tps="$4" ngpus="$5"
  local cell="$(tag_of "$model")__$(lc "$backend")__block${block}"
  local outdir="$WORK_DIR/$cell"
  local csv="$outdir/compute/$DEVICE/$model/attention_combined.csv"
  local log="$LOG_DIR/$cell.log"

  if [ -s "$csv" ]; then
    echo "  skip  $cell (already collected: $(( $(wc -l < "$csv") - 1 )) rows)"
    return 0
  fi

  local -a cmd=(
    "$PYTHON_BIN" -m frontier.profiling.attention.main
    --disable_ray --yes
    --num_gpus "$ngpus" --device "$DEVICE"
    --models "$model"
    --num_tensor_parallel_workers $tps
    --max_pipeline_parallel_size 1
    --attention_backend "$backend"
    --profile_method "$PROFILE_METHOD" --precision "$PRECISION"
    --block_size "$block"
    --max_seq_len "$MAX_SEQ_LEN" --max_model_len "$MAX_MODEL_LEN"
    --min_batch_size 1 --max_batch_size 512
    --batch_size_list $BATCH_SIZE_LIST
    --decode_kv_cache_size_list $DECODE_KV_CACHE_SIZE_LIST
    --enable_true_mixed
    --true_mixed_prefill_batch_sizes $TRUE_MIXED_PREFILL_BATCH_SIZES
    --true_mixed_prefill_chunk_sizes $TRUE_MIXED_PREFILL_CHUNK_SIZES
    --true_mixed_decode_batch_sizes $TRUE_MIXED_DECODE_BATCH_SIZES
    --true_mixed_decode_kv_cache_sizes $TRUE_MIXED_DECODE_KV_CACHE_SIZES
    --true_mixed_prefill_kv_cache_size "$TRUE_MIXED_PREFILL_KV_CACHE_SIZE"
    --output_dir "$outdir"
  )
  if [ "$ENABLE_CHUNKED_PREFILL_GRID_SEARCH" = "true" ]; then
    cmd+=(--enable_chunked_prefill_grid_search)
  else
    cmd+=(--fixed_chunked_prefill_size "$FIXED_CHUNKED_PREFILL_SIZE")
  fi

  echo "------------------------------------------------------------"
  echo "  cell  $cell   (TP: $tps, num_gpus: $ngpus)"
  printf '  cmd  '; printf ' %q' "${cmd[@]}"; printf '\n'
  echo "  log  $log"
  [ "$DRY_RUN" = "true" ] && return 0

  mkdir -p "$outdir" "$LOG_DIR"
  local start=$(date +%s)
  if "${cmd[@]}" > "$log" 2>&1; then
    echo "  ok   $(( $(date +%s) - start ))s, $(( $(wc -l < "$csv") - 1 )) rows"
  else
    # Deliberately non-fatal: one failed cell must not abandon the others.
    echo "  FAIL $(( $(date +%s) - start ))s -- see $log" >&2
    grep -iE "OutOfMemoryError|RuntimeError|Error:" "$log" | tail -2 >&2
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Copy finished cells into the canonical taxonomy. TORCH_SDPA keeps the existing
# attention_combined_block{N}.csv names so it drops straight in where the current
# data sits; AITER gets an _aiter_ infix so both can coexist and be compared.
collect() {
  echo "=== collecting into $COLLECT_DIR ==="
  for model in $MODELS; do
    local dest="$COLLECT_DIR/compute/$DEVICE/$model"
    mkdir -p "$dest"
    for backend in $BACKENDS; do
      local cblocks="$BLOCK_SIZES"
      [ "$backend" = "AITER" ] && cblocks="$AITER_BLOCK_SIZES"
      for block in $cblocks; do
        local cell="$(tag_of "$model")__$(lc "$backend")__block${block}"
        local src="$WORK_DIR/$cell/compute/$DEVICE/$model"
        [ -s "$src/attention_combined.csv" ] || continue
        local infix=""
        [ "$backend" = "AITER" ] && infix="aiter_"
        for suffix in "" "_true_mixed" "_combined"; do
          [ -f "$src/attention${suffix}.csv" ] || continue
          local dst="$dest/attention${suffix}_${infix}block${block}.csv"
          if [ "$DRY_RUN" = "true" ]; then
            echo "  would copy $src/attention${suffix}.csv -> $dst"
          else
            cp "$src/attention${suffix}.csv" "$dst"
            echo "  $dst  ($(( $(wc -l < "$dst") - 1 )) rows)"
          fi
        done
      done
    done
  done
  cat <<EOF

Point the simulator at the *combined* file matching the backend and block_size
you are validating (it already merges standard + true-mixed rows):
  --random_forrest_execution_time_predictor_config_atten_input_file \\
      $COLLECT_DIR/compute/$DEVICE/<model>/attention_combined_aiter_block16.csv
or, on run_validation.py / frontier_cli_translator.py:
  --atten-input-file <same path> --block-size 16
EOF
}

# ---------------------------------------------------------------------------
[ "$USE_DOCKER" = "true" ] && run_in_docker

mkdir -p "$WORK_DIR" "$LOG_DIR"

cat <<EOF
============================================================
  gpt-oss attention full sweep -- AITER + TORCH_SDPA
============================================================
Models:        $MODELS
Backends:      $BACKENDS
Block sizes:   $BLOCK_SIZES   (AITER: $AITER_BLOCK_SIZES -- no block32 kernel)
TP (AITER):    $AITER_TPS
TP (SDPA):     $SDPA_TPS        <- capped by SDPA's materialised-score memory
Device/GPUs:   $DEVICE / $NUM_GPUS (AITER: $AITER_NUM_GPUS) (ids: $GPU_IDS)
Method:        $PROFILE_METHOD / $PRECISION
max_seq_len:   $MAX_SEQ_LEN   chunked-prefill grid search: $ENABLE_CHUNKED_PREFILL_GRID_SEARCH
batch sizes:   $BATCH_SIZE_LIST
decode KV:     $DECODE_KV_CACHE_SIZE_LIST
true-mixed:    prefill_bs={$TRUE_MIXED_PREFILL_BATCH_SIZES} chunk={$TRUE_MIXED_PREFILL_CHUNK_SIZES}
Work dir:      $WORK_DIR   (one independent tree per cell = the checkpointing)
Collect into:  $COLLECT_DIR
aiter cache:   $AITER_CACHE_DIR
Dry run:       $DRY_RUN
============================================================
EOF

if [ "$COLLECT_ONLY" = "true" ]; then collect; exit 0; fi

prewarm_aiter
if [ "$PREWARM_ONLY" = "true" ]; then exit 0; fi

failed=0
for model in $MODELS; do
  for backend in $BACKENDS; do
    case "$backend" in
      AITER)      tps="$AITER_TPS"; blocks="$AITER_BLOCK_SIZES"; ngpus="$AITER_NUM_GPUS" ;;
      TORCH_SDPA) tps="$SDPA_TPS";  blocks="$BLOCK_SIZES";       ngpus="$NUM_GPUS" ;;
      *)          tps="$AITER_TPS"; blocks="$BLOCK_SIZES";       ngpus="$NUM_GPUS" ;;
    esac
    for block in $blocks; do
      profile_cell "$model" "$backend" "$block" "$tps" "$ngpus" || failed=$((failed + 1))
    done
  done
done

echo "============================================================"
[ "$failed" -gt 0 ] && echo "WARNING: $failed cell(s) failed -- rerun this script, finished cells are skipped" >&2
[ "$DRY_RUN" = "true" ] || collect
echo "sweep complete (failed cells: $failed)"
