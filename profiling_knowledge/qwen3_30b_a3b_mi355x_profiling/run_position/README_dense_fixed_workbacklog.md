# linear_op dense collection with both run-position fixes (GEMM work backlog + settle, batch-flush flag)

**What this directory is.** A re-collection of the fixed dense linear_op grid for Qwen3-30B-A3B on MI355X (3,327 token counts ×
TP {1,2,4,8}, 3 warm-up + 25 timed forwards per shape, two timing columns, fused RoPE, GC disabled) with **two changes relative
to the 2026-09-22 collection in `data/profiling_dense_fixed/` (Slurm job 21483)**:

1. **Work backlog instead of the `_sleep` spin, plus 3 settle forwards (Effect 1 fix, code change).** The GPU-bound pass holds
   the device behind the host with a backlog enqueued before every block of 25 timed forwards. That backlog is now a chain of
   4096² bf16 GEMMs of the requested length (`linear_op_wrapper.BACKLOG_KIND = "gemm_alloc"`, env `FRONTIER_GPU_BACKLOG_KIND`;
   `"sleep"` restores the old spin, `"gemm"` and `"stream"` are the other tested kinds), followed by `SETTLE_STEPS = 3` untimed
   forwards (env `FRONTIER_LINEAR_SETTLE_STEPS`) during which the CudaTimer scopes are paused (`TimerStatsStore.pause()`: no
   event records, no samples) so the caches refill after the GEMMs. New CSV columns `settle_steps`, `backlog_kind`. The GEMM
   count is calibrated once per worker (20 warm-up calls, then 50 timed) and re-fitted from every delivered backlog, like the
   old spin's cycles-per-ms.
2. **`DEBUG_CLR_MAX_BATCH_SIZE=1000000` in the container (Effect 2 fix, environment only)**, as in
   `data/profiling_dense_fixed_batchflush/` (job 21500; README there).

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
- Code: worktree commit `4b8fce3` plus the uncommitted changes to `frontier/profiling/linear_op/linear_op_wrapper.py`
  (md5 `713a8edc…`), `common/cuda_timer.py` (`50957b54…`), `common/timer_stats_store.py` (`1f508e70…`); cluster copy verified
  identical before submission.
- Image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`; sbatch `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` with
  `STAGE=linear_op COLLECT_DIR=data/profiling_dense_fixed_workbacklog LINEAR_LOG_SUFFIX=_dense_workbacklog FRONTIER_COMMIT=4b8fce3+workbacklog`
  `LINEAR_DOCKER_ENV="-e FRONTIER_LINEAR_ACTIVE_STEPS=25 -e DEBUG_CLR_MAX_BATCH_SIZE=1000000 -e FRONTIER_GPU_BACKLOG_KIND=gemm_alloc -e FRONTIER_LINEAR_SETTLE_STEPS=3"`.
- Validation on the same node just before: Slurm job 21518 (`q3-posprobe8`, `scripts/slurm/qwen3_posprobe8.sbatch`):
  gemm_alloc + settle 3 at TP1 and TP8 with device-side clock traces and a 200-step run (`data/profiling_posprobe/R50…R54*.jsonl`);
  earlier rounds: jobs 21513 (`posprobe5`), 21515 (`posprobe6`), 21517 (`posprobe7`).
- Logs: `data/profiling/sweep_work/logs/q3-dense-workbacklog-<jobid>.out`, `linear_op_dense_workbacklog_{cuda_event,record_function}.log`.
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
| collection job | Slurm **21519** (`q3-dense-workbacklog`), node `amd-mi355x-1`, 2026-09-22 17:35:53–17:51:07 UTC (15 min; ≈4 min cuda_event pass, ≈11 min record_function pass) |
| validation job | Slurm **21518** (`q3-posprobe8`), same node, 2026-09-22 ≈17:34 UTC, 31 s (earlier rounds 21513, 21515, 21517) |
| container env | `FRONTIER_LINEAR_ACTIVE_STEPS=25 DEBUG_CLR_MAX_BATCH_SIZE=1000000 FRONTIER_GPU_BACKLOG_KIND=gemm_alloc FRONTIER_LINEAR_SETTLE_STEPS=3` |
| code | commit `4b8fce3` + uncommitted wrapper/timer changes (md5 `713a8edc…`, `50957b54…`, `1f508e70…`), identical on the cluster |
| output | `compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` 45,694,061 B, md5 `7016bd6260aacf40824122d09f627a44`, 13,308 rows × 150 columns; `linear_op_kernel_only.csv` 5,873,472 B |
| GEMM calibration | `ms_per_call` 0.128–0.130 ms on every worker (log `linear_op_dense_workbacklog_cuda_event.log`) |

**Validation (job 21518, single process, GPU 0, 7 token counts each).** `attn_pre_proj` early/late (runs 1–5 over runs
20–25): TP1 0.999 0.985 1.006 1.010 0.997 1.001 0.997, TP8 0.998 0.996 0.995 1.001 1.003 0.995 1.024 (spin controls
1.06–1.02 at TP1, 1.03–1.00 at TP8); position profiles 997–1005 permille at every position; shader clock 2330–2340 MHz from
the first timed run; medians at the settled values (TP1/6144 tokens 229 µs vs 240–268 with the spin; TP8/64 tokens 22.6 vs
22.7); 200-step run flat.

**Verification on this CSV** (`run_position_hist.py`, `run_position_profile.py`; comparison with job 21483 by `num_tokens` × TP):

| check | job 21483 (spin) | this run (gemm_alloc + settle 3 + flag) |
|---|---|---|
| `time_stats.attn_pre_proj` argmin at run 25, TP1/2/4/8 | 33.2 / 30.5 / 27.2 / 22.4 % | 16.9 / 13.1 / 9.2 / 6.3 % (baseline 4 %) |
| per-position median profile `attn_pre_proj` TP1, runs 1→25 | 1013 → 983 permille (3.0 % ramp) | 1012 → 994 (1.8 %: ~1 % in runs 1–4, then a slow 0.7 % decline) |
| same, TP8 | 1009 → 988 (2.1 %) | 1010 1008 1005 1003 1001 then 999–1000 flat (a 1 % transient in runs 1–4) |
| early/late `attn_pre_proj` (median over rows), TP1/2/4/8 | 1.028 / 1.026 / 1.020 / 1.016 | 1.014 / 1.010 / 1.007 / 1.005 |
| argmax position | runs 3–5 (7–13 %); run 20 at TP>1 (42–60 %) | run 1 (8–14 %), run 2 (7–10 %); run 20 at baseline |
| `time_stats_hostbound.attn_pre_proj` argmax runs 21–22, TP>1 | 62–76 % | baseline; run 1 = 84–86 % (ordinary first-run effect) |
| medians, this run / 21483 (p50) | — | attn_pre_proj 0.976, attn_post_proj 0.974, attn_rope 0.985, norms 0.98, emb 0.995, forward span 0.984 — i.e. the settled clock, as predicted |
| per-row relative MAD of `attn_pre_proj` samples (median over rows) | 0.98 / 0.95 / 0.85 / 0.77 % | 0.59 / 0.59 / 0.56 / 0.58 % (the ramp was most of the spread) |
| backlog delivered / requested | 0.97–1.03 | p1 0.98, p99 1.02; 32 rows below 0.95, none below 0.90; coverage ≥ 3.21× the legacy loop wall on every row |
| `sanity_check.py` | 875 rows trip the 1.05× GPU-bound/legacy gate | **PASS** (0 rows) |

**Conclusion.** Effect 2 is absent. Effect 1 is reduced to roughly half of its size, not eliminated: a ≈1 % transient over
runs 1–4 at every TP (the 3 settle forwards do not fully absorb the cache refill / clock settling under the 8-GPU load of the
dense run, whereas the single-GPU validation was flat) and a residual slow 0.7 % decline over the block at TP1 only. Medians
now sit at the settled clock and the per-row spread is 40 % smaller. Next cheap step if the residual matters: raise
`FRONTIER_LINEAR_SETTLE_STEPS` to 8 (adds ≈0.2–3 ms per block, ≈230 packets) and re-check the runs-1–4 transient; the slow TP1
decline would need a clock/thermal probe under full-node load to attribute.
