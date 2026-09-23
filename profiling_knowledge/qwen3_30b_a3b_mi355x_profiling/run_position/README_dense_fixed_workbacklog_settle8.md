# linear_op dense collection: GEMM work backlog + 8 settle forwards + batch-flush flag (settle-8 variant)

**What this directory is.** The same collection as `data/profiling_dense_fixed_workbacklog/` (job 21519: GEMM work backlog
+ 3 settle forwards + `DEBUG_CLR_MAX_BATCH_SIZE=1000000`) with **one change: 8 settle forwards instead of 3**
(`FRONTIER_LINEAR_SETTLE_STEPS=8`). Purpose: job 21519 left a ≈1 % transient over timed runs 1–4 at every TP and a slow 0.7 %
decline over the block at TP1 (its README, run record); this run tests whether five more untimed forwards after the GEMM
backlog (≈0.2–3 ms per block, ≈145 more queue packets) absorb the transient. Everything below that is not about the settle
count is inherited from job 21519 and repeated here so this directory stands alone.

**Why (1).** `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/13_run_position_anomalies_root_cause.md`, Effect 1: the
single-wave `torch.cuda._sleep` spin leaves the GPU in a low-activity clock state; the 25 forwards after it run 2–8 % slow with
a monotone within-block ramp (run 25 = argmin in 12–33 % of rows vs 4 % expected), so job 21483's medians sit mid-ramp.
Alternatives tried on the same node the same day, all single-process probe runs (`posprobe.py`, TP1 and TP8, 7 token counts):

| variant | job | result |
|---|---|---|
| spin + 10 untimed forwards after it | 21513 | drift unchanged (early/late 1.025 vs 1.026 control at TP8; 1.066 vs 1.061 at TP1); the shader clock stays at its post-spin level through all 10 settle forwards, and a slower domain keeps ramping after it is flat |
| dense GEMM chain into a preallocated output (`gemm`) + 3 settle | 21515 | flat at heavy shapes, but small shapes noisy (TP8/64 tokens 27–49 µs around a 22.7 µs kernel) with the shader clock at 1.7–1.9 GHz right after the chain in that job; not reproduced in job 21517 (`gemm`, settle 0: flat) — job/thermal-state dependent |
| memory-streaming `add` over 1 GB operands (`stream`) + 3 settle | 21517 | flat at TP8 but small kernels 4–5 % slower than any spin-control run (23.8 vs 22.7 µs) and TP1 heavy shapes still drift 3.5 % with the shader clock flat |
| allocating GEMM chain (`gemm_alloc`) + 3 settle | 21517, 21494 (probe prototype), 21518 | **chosen**: flat position profile at every TP8 shape (early/late 0.995–1.017), medians 1–5 % below the spin control, i.e. the settled value from run 1; TP1 checked in job 21518 (run record) |

**What is expected to change, and what is not.** Per-run distributions: flat position profile (argmin ≈ 4 % per position, no
concentration at runs 22–25); `attn_pre_proj` argmax at run 20 and the legacy runs-21/22 stall absent (fix 2). Medians: kernel
time at the settled clock, ≈1–5 % **below** job 21483 for the attention GEMMs (largest at the heavy shapes) and within noise
for the norms and RoPE. `gpu_backlog_ms_actual` is the event-measured GEMM chain (it may deliver 85–105 % of the request; the
coverage gate is ≥ 3× the legacy loop wall). `host_wall_per_forward_ms_backlog` / `host_enqueue_per_forward_ms_backlog` still
divide by 25 but the block holds 28 forwards plus ≈400 GEMM launches; informational only.

**Provenance.**
- Code: worktree commit `3a8eb55` (the fixes committed; `frontier/profiling/linear_op/linear_op_wrapper.py`
  (md5 `713a8edc…`), `common/cuda_timer.py` (`50957b54…`), `common/timer_stats_store.py` (`1f508e70…`); cluster copy verified
  identical before submission.
- Image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`; sbatch `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` with
  `STAGE=linear_op COLLECT_DIR=data/profiling_dense_fixed_workbacklog_settle8 LINEAR_LOG_SUFFIX=_dense_workbacklog_settle8 FRONTIER_COMMIT=3a8eb55`
  `LINEAR_DOCKER_ENV="-e FRONTIER_LINEAR_ACTIVE_STEPS=25 -e DEBUG_CLR_MAX_BATCH_SIZE=1000000 -e FRONTIER_GPU_BACKLOG_KIND=gemm_alloc -e FRONTIER_LINEAR_SETTLE_STEPS=8"`.
- Validation on the same node just before: Slurm job 21518 (`q3-posprobe8`, `scripts/slurm/qwen3_posprobe8.sbatch`):
  gemm_alloc + settle 3 at TP1 and TP8 with device-side clock traces and a 200-step run (`data/profiling_posprobe/R50…R54*.jsonl`);
  earlier rounds: jobs 21513 (`posprobe5`), 21515 (`posprobe6`), 21517 (`posprobe7`).
- Logs: `data/profiling/sweep_work/logs/q3-dense-workbacklog-s8-<jobid>.out`, `linear_op_dense_workbacklog_settle8_{cuda_event,record_function}.log`.
- Abandoned sibling directories from the same day: `data/profiling_dense_fixed_settle/` (job 21514, cancelled) and
  `data/profiling_dense_fixed_gemmbacklog/` (job 21516, cancelled); their READMEs say why.

**Files.** `compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` (150 columns: the 148 of `profiling_dense_fixed` + `settle_steps`,
`backlog_kind`), `linear_op_kernel_only.csv`, this README.

**How to check** (worktree, venv `~/.virtualenvs/qwen3-profiling`):
```
python profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/run_position/run_position_hist.py <dir>/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv time_stats
python profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/run_position/run_position_profile.py <same csv> time_stats
```

**Status.** Cluster scratch only; not staged into `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/`.

## Run record

| item | value |
|---|---|
| collection job | Slurm **21539** (`q3-dense-workbacklog-s8`), node `amd-mi355x-1`, 2026-09-23 09:49:11–10:04:43 UTC (15.5 min) |
| container env | `FRONTIER_LINEAR_ACTIVE_STEPS=25 DEBUG_CLR_MAX_BATCH_SIZE=1000000 FRONTIER_GPU_BACKLOG_KIND=gemm_alloc FRONTIER_LINEAR_SETTLE_STEPS=8` |
| code | commit `3a8eb55` (profiler files identical on the cluster, md5 `713a8edc…` / `50957b54…` / `1f508e70…`) |
| output | `compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` 45,463,879 B, md5 `3635e4d305ae5cf84884e275d3ff37ef`, 13,308 rows × 150 columns (`settle_steps` = 8); `linear_op_kernel_only.csv` 5,870,513 B |
| GEMM calibration | `ms_per_call` 0.128–0.130 ms per worker |
| gates | `sanity_check.py` **PASS**; backlog delivered/requested p1 0.98 – p99 1.02 (32 rows below 0.95, none below 0.90), coverage ≥ 3.17× the legacy loop wall; host finishes enqueueing at ≤ 37 % of the loop wall |

**Verification** (`run_position_hist.py`, `run_position_profile.py`; `attn_pre_proj`, GPU-bound column; "early/late" = median
of runs 1–5 over median of runs 20–25, median over rows; "run1/med" = first timed run over the row median):

| | job 21483 (spin) | job 21519 (settle 3) | this run (settle 8) |
|---|---|---|---|
| early/late TP1 / TP2 / TP4 / TP8 | 1.028 / 1.026 / 1.020 / 1.016 | 1.014 / 1.010 / 1.007 / 1.005 | **1.011 / 1.007 / 1.002 / 1.000** |
| run1/med TP1 / TP2 / TP4 / TP8 | 1.013 / 1.013 / 1.011 / 1.009 | 1.012 / 1.011 / 1.010 / 1.010 | **1.008 / 1.006 / 1.004 / 1.003** |
| median position profile TP8, runs 1→25 | 1009 → 988 permille | 1010 1008 1005 1003 1001 1000 … | 1003 1001 1001 then 1000 flat |
| median position profile TP1, runs 1→25 | 1013 → 983 | 1012 → 994 | 1008 → 995 (a slow, linear 1.3 % decline; TP2 1006 → 997) |
| argmin at run 25, TP1 / TP2 / TP4 / TP8 | 33 / 31 / 27 / 22 % | 17 / 13 / 9 / 6 % | 17 / 12 / 7 / 5 % (baseline 4 %) |
| per-row relative MAD (median over rows) TP1…TP8 | 0.98 … 0.77 % | 0.59 … 0.58 % | 0.55 … 0.54 % |
| medians vs job 21519 (p50, all ops) | — | — | 1.000–1.002 (vs job 21483: 0.976–0.984, the settled clock) |
| run-20 `attn_pre_proj` max; legacy runs-21/22 stall | present | absent | absent (legacy argmax = run 1 in 77–85 % of rows, the ordinary first-run effect) |

**Conclusion.** Eight settle forwards remove the runs-1–4 transient at TP4 and TP8 (profiles flat to ±0.1 % from run 1) and
halve it at TP1/TP2. What remains at TP1 and TP2 is a slow, linear 0.9–1.3 % decline over the whole 25-run block that does not
depend on the settle count (identical with 3 and 8) and is absent in single-GPU probe runs; it is therefore attributed
(SPECULATIVE) to clock/thermal settling under full-node load, on a timescale longer than a block. Medians are unchanged from
job 21519, so the 8-settle setting costs nothing in the statistics and ≈1–5 ms per block in wall time; it is now the profiler
default (commit after this run).
