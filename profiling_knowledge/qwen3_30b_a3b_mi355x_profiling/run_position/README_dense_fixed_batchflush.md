# linear_op dense collection with the ROCclr batch-flush barrier moved out of the timed loop

**What this directory is.** A re-collection of the fixed dense linear_op grid for Qwen3-30B-A3B on MI355X (3,327 token counts ×
TP {1,2,4,8}, 3 warm-up + 25 timed forwards per shape, two timing columns, fused RoPE, GC disabled), identical to the
2026-09-22 collection in `data/profiling_dense_fixed/` (Slurm job 21483) in code, node type and settings, **except for one
environment variable in the container: `DEBUG_CLR_MAX_BATCH_SIZE=1000000`** (ROCm runtime default: 1000).

**Why.** The investigation in `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/13_run_position_anomalies_root_cause.md`
found that the ROCm runtime (ROCclr, `HostQueue::FlushSubmissionBatch`) enqueues an internal clean-up marker after every 1000
commands submitted since the last synchronize. On the device that marker is a barrier-AND packet with system-scope
acquire/release fences costing ≈5.5 µs, and it lands at a fixed position of every timed loop: at TP>1 (29 kernels + 22 event
records per forward) inside forward 20's `attn_pre_proj`, which made run 20 the row maximum in 42–60 % of TP2/4/8 rows of job
21483 (+3–4 µs on a 32–56 µs scope), and in the legacy pass at runs 21/22 as a +50–100 µs host stall. Raising the batch limit
above the number of commands in a block (25 forwards × 51 packets ≈ 1,300 plus the spin) means the marker is never issued
inside a timed loop; the runtime still cleans up at every `synchronize()` (which resets the batch), so nothing accumulates
beyond one pass. `DEBUG_CLR_MAX_BATCH_SIZE` is a release flag of ROCclr read from the environment
(`rocclr/utils/flags.hpp`, "Forces the callback to clean-up CPU submission queue"); it is not a documented user-facing knob.

**What is expected to change, and what is not.** Only the per-run sample distributions: no systematic `max` at run 20 for
`attn_pre_proj` at TP>1, no +50–100 µs at runs 21/22 in `time_stats_hostbound.*`. Medians should be unchanged within ordinary
run-to-run noise (job 21483 vs 21486 agreed to 1.000 median ratio). The within-block drift (Effect 1 in the same document, a
clock ramp after the `_sleep` backlog spin) is **not** addressed by this run and is still present.

**Provenance.**
- Code: worktree commit `4b8fce3` (`smatar/qwen3-30b-mi355-profiling`), byte-identical to the cluster copy for
  `frontier/profiling/linear_op/*.py`, `common/cuda_timer.py`, `common/timer_stats_store.py`, `common/layers/rotary_embedding.py`
  (md5 checked before submission).
- Image: `lmsysorg/sglang:v0.5.11-rocm700-mi35x`. sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`,
  `STAGE=linear_op COLLECT_DIR=data/profiling_dense_fixed_batchflush LINEAR_LOG_SUFFIX=_dense_batchflush`
  `LINEAR_DOCKER_ENV="-e FRONTIER_LINEAR_ACTIVE_STEPS=25 -e DEBUG_CLR_MAX_BATCH_SIZE=1000000"`.
- Validation of the flag on the same node, same day, before this collection: Slurm job `q3-posprobe4` (see run record),
  results in `data/profiling_posprobe/R23…R27*.jsonl` and `amdlog_R26_batchfix_tp8.log`.
- Logs: `data/profiling/sweep_work/logs/q3-dense-batchflush-<jobid>.out`, `linear_op_dense_batchflush_cuda_event.log`,
  `linear_op_dense_batchflush_record_function.log`.

**Files.** `compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` (cuda_event, two timing columns, same 148-column schema as
`profiling_dense_fixed`), `linear_op_kernel_only.csv` (record_function pass), this README.

**How to check the fix in the CSV** (from the worktree, venv `~/.virtualenvs/qwen3-profiling`):
```
python profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/run_position/run_position_hist.py <this dir>/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv time_stats
python profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/run_position/run_position_hist.py <...>/linear_op.csv time_stats_hostbound
```
Expected: `attn_pre_proj` argmax at run 20 ≈ 4 % (baseline) at every TP instead of 42–60 %; legacy argmax at runs 21/22 ≈ 4 %
instead of 62–76 %; argmin still concentrated at runs 22–25 (Effect 1, unchanged).

**Status.** Not staged into `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/`; the canonical file is still the 2026-09-15
collection. This directory is cluster scratch only.

## Run record

| item | value |
|---|---|
| collection job | Slurm **21500** (`q3-dense-batchflush`), node `amd-mi355x-1`, 2026-09-22 15:25:39–15:40:57 UTC (15 min: ≈4 min cuda_event pass, ≈11 min record_function pass) |
| validation job | Slurm **21499** (`q3-posprobe4`), same node, 2026-09-22 ≈15:24 UTC, 25 s |
| container env | `FRONTIER_LINEAR_ACTIVE_STEPS=25 DEBUG_CLR_MAX_BATCH_SIZE=1000000` (everything else as job 21483) |
| code | commit `4b8fce3`, md5 of the profiler files identical to the cluster copy |
| output | `compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` 45,073,269 B, md5 `cd405d64289b0b42e8916ba24574ba01`, 13,308 rows × 148 columns (schema identical to job 21483); `linear_op_kernel_only.csv` 5,705,704 B |

**Validation (job 21499, single process, GPU 0).** With the flag: `attn_pre_proj` position-20 sample = 997 permille of the row
median (same-job control without the flag: 1132; argmax at run 20 in 6 of 7 tasks vs 0 of 7 with the flag); the legacy pass has
no stall at runs 21/22 (control: 1907/1404 permille); 200-step run: no spikes at runs 20 and 118. Runtime log with the flag
(`amdlog_R26_batchfix_tp8.log`): the only barrier-AND packets are the 13 synchronize barriers and the 3 kernarg-chunk barriers
(at 1440, 502 and 2220 packets after a synchronize, exactly as without the flag); the barriers at 1003/2008/1003/2005 packets
are gone.

**Verification on this CSV** (`run_position_hist.py`, both columns; comparison with job 21483 by `num_tokens` × TP):

| check | job 21483 | this run |
|---|---|---|
| `time_stats.attn_pre_proj` argmax at run 20, TP2/4/8 | 42.2 / 53.6 / 59.5 % | not in the top 6 positions; top = runs 3–5 at 8–13 % |
| run-20 excess over row median, p50 (p90), TP2/4/8 | +3.0 (+4.4) / +3.7 (+4.6) / +3.9 (+4.7) µs | −0.6 (+0.1) / −0.4 (+0.2) / −0.2 (+0.2) µs |
| `time_stats_hostbound.attn_pre_proj` argmax at runs 21–22, TP2/4/8 | 62 / 75 / 76 % (run 22) | run 1 = 87 / 87 / 86 %; runs 21/22 at baseline |
| medians, this run / 21483, all ops, p1–p50–p99 | — | 0.96–1.00–1.05 (attention GEMMs 0.98–1.00–1.03) |
| Effect 1 (argmin ramp to run 25) | run 25 = 33 % (`attn_pre_proj` TP1) | run 25 = 34 %: unchanged, as expected |
| backlog delivered / requested | 0.97–1.03 on all rows | 2 rows outside 0.95–1.05 (TP1, tokens 1207/1215: 0.76 and 1.31, one clock re-fit on one worker); coverage ≥ 3.06× the legacy loop on every row |
| `sanity_check.py` | 875 rows trip the 1.05× GPU-bound/legacy gate | 60 (`attn_pre_proj`) and 57 (`attn_post_proj`) rows, max 1.071 — the known gate of `12_` §5; everything else passes |

Conclusion: the flag removes the runtime's batch-flush barrier from the timed loops without changing the medians. The
within-block drift after the `_sleep` spin (Effect 1) is untouched and still needs its own fix.
