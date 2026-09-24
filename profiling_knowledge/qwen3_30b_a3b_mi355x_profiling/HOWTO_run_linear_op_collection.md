# How to collect, stage and verify a linear_op dataset (Qwen3-30B-A3B, MI355X)

The runbook for the current profiler (commit `4b8fce3` and later: two timing columns, spun blocks of 25 forwards, clock probe,
GC guard, fused RoPE kernel). For *why* it is built this way read `12_dip_investigation_summary.md`; for what each column means,
`10_measurement_contract.md` and the dataset README §5b. Everything below runs from the VM (`smatar-dev`); GPU work goes through
Slurm on the cluster, always as user `dn` (`sudo -u dn`), never from the VM itself.

## 0. Prerequisites

- Frontier worktree `/home/dn/Frontier-qwen3-profiling` on branch `smatar/qwen3-30b-mi355-profiling`.
- Cluster checkout `/opt/shared/frontier-qwen3-profiling/Frontier`, an rsync copy of the worktree owned by `dn`. It is **not** a git
  clone: sync the files you changed before every submission, by explicit path, never with `--delete`.
- Image `lmsysorg/sglang:v0.5.11-rocm700-mi35x` on the node (Slurm pulls it if absent; the first job on a node takes longer).
- `python3` with pandas on the VM for the checkers (the system interpreter is enough).

## 1. Sync code to the cluster

```bash
cd /home/dn/Frontier-qwen3-profiling
R=/opt/shared/frontier-qwen3-profiling/Frontier
rsync -a --rsync-path="sudo -u dn rsync" frontier/profiling/linear_op/linear_op_wrapper.py cluster:$R/frontier/profiling/linear_op/
rsync -a --rsync-path="sudo -u dn rsync" frontier/profiling/common/layers/rotary_embedding.py cluster:$R/frontier/profiling/common/layers/
rsync -a --rsync-path="sudo -u dn rsync" frontier/profiling/utils/record_function_tracer.py cluster:$R/frontier/profiling/utils/
rsync -a --rsync-path="sudo -u dn rsync" profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch cluster:$R/profiling_knowledge/scripts/slurm/
```
Check the copy matches before submitting (the collection is only as good as the code that ran):
```bash
for f in frontier/profiling/linear_op/linear_op_wrapper.py profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch; do
  echo "$(md5sum < $f | cut -c1-8) local  $f"; ssh cluster "sudo -u dn md5sum < $R/$f | cut -c1-8"; done
```

## 2. Submit the collection

The script is `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`, one job per `STAGE`. For linear ops:

```bash
ssh cluster "sudo -u dn bash -s" <<'EOF'
cd /opt/shared/frontier-qwen3-profiling/Frontier
TAG=dense_$(date +%Y%m%d_%H%M)                      # pick a tag; the scratch tree is data/profiling_$TAG
mkdir -p data/profiling_$TAG && chmod -R a+rwX data/profiling_$TAG
export STAGE=linear_op FRONTIER_COMMIT=$(git -C /home/dn/Frontier-qwen3-profiling rev-parse --short HEAD 2>/dev/null || echo unknown)
export COLLECT_DIR=data/profiling_$TAG LINEAR_LOG_SUFFIX=_$TAG
export LINEAR_PROFILE_METHODS=cuda_event                         # add "record_function" for the kernel-only file too (see §2.2)
export LINEAR_DOCKER_ENV="-e FRONTIER_LINEAR_ACTIVE_STEPS=200"   # timed forwards per shape; 25 for a quick pass
sbatch --parsable -t 03:00:00 -J q3-$TAG --export=ALL profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch
EOF
```
Never set `COLLECT_DIR=data/profiling` (that is the tracked dataset tree on the checkout); always a scratch tree.

### 2.1 Knobs (environment, all optional)

| variable | default | meaning |
|---|---|---|
| `LINEAR_TOKENS_PY` | dense grid: 1…2048 every 1, …8192 every 8, …16384 every 16, minus 4000 (3,327 values) | Python expression for the token list, e.g. `"[1,8,64,512,3072,4096,4192,6144,8192]"` for the 9-value validation grid (≈1–2 min) |
| `LINEAR_TPS` | `1 2 4 8` | tensor-parallel sizes |
| `LINEAR_NUM_GPUS` | 8 | worker processes (tasks are spread over GPUs) |
| `LINEAR_PROFILE_METHODS` | `cuda_event record_function` | each method runs into the same `COLLECT_DIR`: `cuda_event` → `linear_op.csv` (two timing columns), `record_function` → `linear_op_kernel_only.csv` (kineto per-scope kernel time) |
| `LINEAR_DOCKER_ENV` | empty | extra `-e VAR=value` flags for the container, e.g. `FRONTIER_LINEAR_ACTIVE_STEPS=200` |
| `FRONTIER_LINEAR_ACTIVE_STEPS` (container) | 50 | timed forwards per shape (3 warm-up forwards are always added) |
| `FRONTIER_LINEAR_BACKLOG_BLOCK_STEPS` (container) | 25 | forwards per spun block; the HIP queue holds ≈25 forwards of launches, do not raise without re-checking closure |
| `FRONTIER_GPU_BACKLOG_MS` (container) | 4× the legacy loop wall | total spin length; the default is what the coverage gate expects |
| `FRONTIER_RF_KEEP_TRACES` | 0 | 1 keeps the ≈4.9 MB kineto trace per shape (`profiler_traces/`); the dense grid would write ≈65 GB |
| `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK` | unset | **do not set** on this image: it forces the (now per-head, but slow) torch RoPE path and the checker rejects the rows |
| `LINEAR_CMD_PREFIX` | empty | wraps the profiler, e.g. `rocprofv3 --kernel-trace --output-format csv -d <dir> --` (needs a world-writable `.rocprofv3/` in the checkout) |
| `FRONTIER_COMMIT` | `unknown` | recorded in the job log; pass the worktree's HEAD |

Timings on one node, dense grid, `cuda_event` only: 3.5 min at 25 forwards, 19 min at 200. `record_function` roughly doubles it.

### 2.2 Locked clock (optional second dataset)

`profiling_knowledge/scripts/slurm/qwen3_fixed_clock_validation.sbatch` (cluster copy; the generalised collection script is a planned
task) locks all eight GPUs of the node with `rocm-smi --setperfdeterminism <MHz>` under an `EXIT` trap that resets them, then runs the
same stage. It needs the whole node (`--gres=gpu:8`). Never lock clocks outside a job.

## 3. Watch the job and fetch the result

```bash
ssh cluster "sacct -j <jobid> -n -P -o JobID,State,Elapsed,NodeList | grep -v batch"
ssh cluster "sudo -u dn tail -3 /opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/linear_op_<TAG>_cuda_event.log"
# expect: "✓ Saved linear-op profiling data to: data/profiling_<TAG>/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv"
mkdir -p /tmp/<TAG>
rsync -a --rsync-path="sudo -u dn rsync" cluster:/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling_<TAG>/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv /tmp/<TAG>/
```
The dense 200-forward file is ≈130–220 MB because every row stores all its samples.

## 4. Verify

Both checkers take the **directory / file path positionally** (an option in that place is a usage error; it once produced a vacuous PASS).
```bash
cd /home/dn/Frontier-qwen3-profiling
python3 profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py /tmp/<TAG>                       # dense grid
python3 profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py /tmp/<TAG> --tokens-grid "[1,8,64,512,3072,4096,4192,6144,8192]"   # validation grid
python3 profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/test_measurement_validity.py /tmp/<TAG>/linear_op.csv --expect green
```
What they check: row count = tokens × TP; two-column schema complete; spin covers ≥ 3× the legacy loop; GPU-bound ≤ 1.10× legacy on GEMM
scopes where both are kernel-bound; probe columns finite; `attn_rope_impl == vllm_kernel`; the pre-registered T1′/T2 signature
(legacy/GPU-bound ≥ 2 at 8 and 64 tokens; cost grows ≥ 2× from 64 to 4096 tokens). Flags: `--allow-legacy-schema` (pre-fix or
kineto files), `--allow-rope-fallback`, `--sclk-band LO,HI` (planned, for locked runs).

Things worth eyeballing that no gate enforces (monotonicity is deliberately not a criterion): per op × TP, plot the median against
tokens and compare with the previous run; the known real steps are the hipBLASLt tile switches near 4352, 4480, 4864, 5376 and
≈11.8k–12.3k tokens at TP 2/4/8. Check the probe columns (`sclk_mhz_*`) sit near 2.4 GHz; check `time_stats.forward_gpu_span` closure
(Σ ops / span ≈ 0.96–0.99).

## 5. Stage the data (shared storage, not git)

Datasets live on the shared filesystem; git carries a pointer `RUN.md` and the README rows.
```bash
D=/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/<YYYY-MM-DD_HHMM>_<what>
ssh cluster "sudo -u dn bash -c 'mkdir -p $D && cp /opt/shared/frontier-qwen3-profiling/Frontier/data/profiling_<TAG>/compute/mi355x/qwen3-a3b-30b-moe/linear_op*.csv $D/ \
  && cp /opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/linear_op_<TAG>_*.log $D/ && cd $D && sha256sum *.csv > SHA256SUMS && chmod -R a+rX .'"
```
Then in the worktree: create `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/<same name>/RUN.md` (job id, node, image, commit,
knobs, verification output, the shared path and sha256, what it supersedes), copy it next to the data as `README.md`, add a row to the
runs index in the dataset README, and commit **by explicit path** (the worktree holds other sessions' files). Template: the RUN.md of
`runs/2026-09-22_0849_linear_op_dense_200fwd_all_fixes/`. Never modify the root `linear_op.csv`.

## 6. Tests

```bash
python3 -m pytest tests/unit/test_linear_op_two_pass_orchestration.py tests/unit/test_linear_op_clock_probe.py \
  tests/unit/test_linear_op_timing_columns.py tests/unit/test_rotary_embedding_numerics.py \
  tests/unit/test_record_function_tracer_cleanup.py tests/unit/test_qwen3_sbatch_contract.py \
  tests/unit/test_qwen3_mi355x_sanity_check.py -q -p no:cacheprovider
```
CPU-only: the wrapper runs against a fake CUDA device. GPU-gated tests (closure, fused-kernel numerics) run in the container:
```bash
ssh cluster "sudo -u dn sbatch -p XAI --gres=gpu:1 -t 00:15:00 -J q3-gputests --wrap 'cd /opt/shared/frontier-qwen3-profiling/Frontier && docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host --shm-size 8G -v \$PWD:/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier -e HIP_VISIBLE_DEVICES=0 lmsysorg/sglang:v0.5.11-rocm700-mi35x python -m pytest tests/unit/test_linear_op_timing_columns.py tests/unit/test_rotary_embedding_numerics.py -v -p no:cacheprovider'"
```

## 7. Gotchas collected along the way

- The cluster checkout is root-squashed NFS: files written by the container belong to `nobody`; `chmod -R a+rwX` the scratch tree first.
- rocprofv3 writes nothing unless `.rocprofv3/` exists world-writable in the checkout.
- Three warm-up forwards do not ramp the clock from idle; only back-to-back tasks keep it at ≈2.4 GHz. A single small task starts at ≈0.8 GHz.
- 50 forwards behind one spin overflow the HIP queue at small shapes; hence blocks of 25.
- The other session found a 2–8 % within-block drift right after each spin (`13_run_position_anomalies_root_cause.md`) and added settle
  forwards; datasets collected before that fix (incl. `2026-09-22_0849`) carry it.
- `sanity_check.py <dir>` is positional; `--data-dir <dir>` used to pass vacuously.
