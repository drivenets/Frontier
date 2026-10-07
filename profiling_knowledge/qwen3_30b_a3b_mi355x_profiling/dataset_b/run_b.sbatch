#!/usr/bin/env bash
#SBATCH --job-name=qwen3-b-pilot
#SBATCH --partition=XAI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00
# Pass an existing log directory with sbatch --output; never run a login-node server.
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${B_TASK_DIR:?absolute directory containing collector and verified launch plan}"
: "${B_PLAN:?absolute verified launch plan path}"
: "${B_IMAGE_ID:?immutable sha256 image ID from approved image inspection}"
: "${B_MODEL:?absolute model directory}"
: "${B_RUN_PARENT:?existing writable shared-store run parent}"
# Bind each exact patched/new file individually at its inspected in-image path,
# rather than replacing the whole installed package tree (simpler: no need to
# stage a full ~3000-file sglang copy just to change 4 files). Set once the
# patch is applied to a local extracted copy of python/sglang/srt/{models,layers}:
#   frontier_b_markers.py, models/qwen3_moe.py, models/qwen2_moe.py, layers/communicator.py
: "${B_PATCHED_FRONTIER_B_MARKERS:?patched file: srt/frontier_b_markers.py}"
: "${B_PATCHED_QWEN3_MOE:?patched file: srt/models/qwen3_moe.py}"
: "${B_PATCHED_QWEN2_MOE:?patched file: srt/models/qwen2_moe.py}"
: "${B_PATCHED_COMMUNICATOR:?patched file: srt/layers/communicator.py}"
: "${B_PATCHED_SCHEDULE_BATCH:?patched file: srt/managers/schedule_batch.py}"
: "${B_PATCHED_IO_STRUCT:?patched file: srt/managers/io_struct.py}"
: "${B_PATCHED_SCHEDULER:?patched file: srt/managers/scheduler.py}"
: "${B_PATCHED_TOKENIZER_MANAGER:?patched file: srt/managers/tokenizer_manager.py}"
: "${B_PATCHED_HTTP_SERVER:?patched file: srt/entrypoints/http_server.py}"
: "${B_CONTAINER_SGLANG:?inspected absolute package path inside image, e.g. /usr/local/lib/python3.12/dist-packages/sglang}"
[[ "$B_IMAGE_ID" == sha256:* ]] || exit 2
run_dir="$B_RUN_PARENT/b-pilot-${SLURM_JOB_ID}"
mkdir "$run_dir"
# This new directory alone needs container writes on the root-squashed shared store.
chmod a+rwx "$run_dir"

# Resolve the sysfs sclk/mclk paths for the GPU Slurm actually allocated to
# this job (ROCR_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES are always "0" inside
# the cgroup regardless of which physical GPU that is -- correlate by PCI bus
# ID instead, matching rocm-smi's view against /sys/class/drm/*/device/uevent).
# telemetry_b.py runs as run_b.py's subprocess, which runs INSIDE the
# container, so it needs this discovered here, on the host, before docker run.
pci_bus=$(rocm-smi --showbus 2>/dev/null | grep -oE '[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]' | head -1)
[[ -n "$pci_bus" ]] || { echo "could not resolve allocated GPU's PCI bus id" >&2; exit 2; }
sclk_path="" mclk_path="" gpu_card=""
for dev in /sys/class/drm/card*/device; do
  if grep -qi "PCI_SLOT_NAME=0*${pci_bus#0000:}" "$dev/uevent" 2>/dev/null || \
     grep -qi "PCI_SLOT_NAME=${pci_bus}" "$dev/uevent" 2>/dev/null; then
    gpu_card="$dev"
    [[ -r "$dev/pp_dpm_sclk" ]] && sclk_path="$dev/pp_dpm_sclk"
    [[ -r "$dev/pp_dpm_mclk" ]] && mclk_path="$dev/pp_dpm_mclk"
    break
  fi
done
[[ -n "$gpu_card" ]] || { echo "could not map PCI bus $pci_bus to a /sys/class/drm card" >&2; exit 2; }
cat > "$run_dir/gpu_devices.json" <<EOF
[{"gpu_id": "pci:${pci_bus}", "sclk_path": "${sclk_path}", "mclk_path": "${mclk_path}", "sensor_resolution_ns": null}]
EOF
echo "resolved GPU telemetry device: pci=$pci_bus card=$gpu_card sclk=$sclk_path mclk=$mclk_path"

# AITER JIT-compiles its attention/GEMM kernels on first use (observed: single
# digit minutes cold, per-shape); --rm containers lose that cache every run
# unless bind-mounted from outside. Proven pattern from
# profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh: host
# $HOME/.aiter -> container /root/.aiter, with AITER_CACHE_DIR set to the
# *container* path.
#
# $HOME/.aiter is the SHARED dn account's cache, used concurrently by other
# users' jobs on other nodes (confirmed: docker ps on amd-mi355x-3 lists many
# other users' running/stopped containers under this same account). A lock
# file under build/ is not necessarily stale just because it is old or
# belongs to a different job -- it may be a live, in-progress compile from a
# concurrent job on another node sharing the same $HOME. Only ever remove a
# lock this flock -n (non-blocking) confirms is NOT currently held by any
# process; never remove by age, by job-id guess, or unconditionally.
HOST_AITER_CACHE_DIR="$HOME/.aiter"
mkdir -p "$HOST_AITER_CACHE_DIR"
if [ -d "$HOST_AITER_CACHE_DIR/build" ]; then
  while IFS= read -r -d '' lock; do
    if flock -n "$lock" true 2>/dev/null; then
      rm -f "$lock"
      echo "removed confirmed-stale AITER lock: $lock"
    else
      echo "AITER lock still held by a live process, leaving it alone: $lock"
    fi
  done < <(find "$HOST_AITER_CACHE_DIR/build" -name lock -type f -print0 2>/dev/null)
fi

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 \
  --ipc=host --network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32g \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" -e B_RUNNING_IMAGE_ID="$B_IMAGE_ID" \
  -e B_GPU_DEVICES_JSON="$run_dir/gpu_devices.json" \
  -e AITER_CACHE_DIR=/root/.aiter \
  -v "$HOST_AITER_CACHE_DIR":/root/.aiter \
  -v "$B_TASK_DIR:$B_TASK_DIR:ro" -v "$B_MODEL:$B_MODEL:ro" \
  -v "$B_PATCHED_FRONTIER_B_MARKERS:$B_CONTAINER_SGLANG/srt/frontier_b_markers.py:ro" \
  -v "$B_PATCHED_QWEN3_MOE:$B_CONTAINER_SGLANG/srt/models/qwen3_moe.py:ro" \
  -v "$B_PATCHED_QWEN2_MOE:$B_CONTAINER_SGLANG/srt/models/qwen2_moe.py:ro" \
  -v "$B_PATCHED_COMMUNICATOR:$B_CONTAINER_SGLANG/srt/layers/communicator.py:ro" \
  -v "$B_PATCHED_SCHEDULE_BATCH:$B_CONTAINER_SGLANG/srt/managers/schedule_batch.py:ro" \
  -v "$B_PATCHED_IO_STRUCT:$B_CONTAINER_SGLANG/srt/managers/io_struct.py:ro" \
  -v "$B_PATCHED_SCHEDULER:$B_CONTAINER_SGLANG/srt/managers/scheduler.py:ro" \
  -v "$B_PATCHED_TOKENIZER_MANAGER:$B_CONTAINER_SGLANG/srt/managers/tokenizer_manager.py:ro" \
  -v "$B_PATCHED_HTTP_SERVER:$B_CONTAINER_SGLANG/srt/entrypoints/http_server.py:ro" \
  -v "$run_dir:$run_dir" -v /sys/class/drm:/sys/class/drm:ro \
  --entrypoint python3 "$B_IMAGE_ID" "$B_TASK_DIR/run_b.py" "$B_PLAN" --output "$run_dir/collection"
