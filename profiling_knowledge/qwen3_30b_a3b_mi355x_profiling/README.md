# Qwen3-30B-A3B on MI355X — profiling knowledge base

Working notes for profiling Qwen3-30B-A3B's attention-layer operators on the 8× MI355X nodes of the Memphis Slurm
cluster with Frontier's `linear_op` and attention profilers. Files are numbered in the order the work happened; each
document is self-contained and states its own date, jobs and data paths. This README is the map.

| file | what it is |
|---|---|
| `00_requirements_and_source_map.md` | plan-loop stages 0–2: requirements, sources, what the profilers measure |
| `01_plan.md` | the approved plan (five operators; AITER attention + attention-layer GEMMs) |
| `02_run_record.md` | run record of the 2026-09-15 sweeps that produced the dataset |
| `03_linear_op_spike_investigation.md` | first write-up of the run-11 `attn_pre_proj` spike (pre-re-run reasoning; superseded by 05) |
| `04_linear_op_spike_specialist_brief.md` | the self-contained brief handed to the root-cause investigation (facts, candidates, proposed experiments) |
| `05_linear_op_spike_root_cause.md` | **the answer**: root cause with CONFIRMED / RULED OUT / SPECULATIVE labels, all experiments, evidence paths |
| `06_post_proj_rope_dip_specialist_brief.md`, `07_…root_cause.md`, `08_…preregistration.md` | a separate, later investigation of the `attn_post_proj` / `attn_rope` mid-range dip at TP>1 (own README material; not covered here) |
| `12_dip_investigation_summary.md` | plain-language summary of the dip investigation and the fixes |
| `13_run_position_anomalies_root_cause.md`, `posprobe.py`, `run_position/`, `../scripts/slurm/qwen3_posprobe*.sbatch` | run-position anomalies in the 2026-09-22 dense collections (within-block drift from the `_sleep` backlog spin; ROCclr's 1000-command batch-flush barrier) with the probe tooling |
| `14_regressor_dataset_handoff.md` | handoff for the new per-(op, TP) regressor: where the settled-clock dataset is (shared store, job 21539), what a row is, label and feature columns, caveats |
| `sanity_check.py`, `viz/` | dataset sanity checks and the run-group/statistics notebook |
| `trace_ops.py`, `mark_host_bound.py`, `test_measurement_validity.py`, `../scripts/slurm/qwen3_validation_trace.sbatch` | tooling of the 06–08 investigation |
| `../scripts/slurm/qwen3_mi355x_profiling.sbatch` | the one sbatch used for every linear_op / attention job here (`STAGE=` selects) |

The dataset itself lives in `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/` (see its `README.md`; each run folder has a
`RUN.md`).

## The run-11 `attn_pre_proj` spike, in one screen

**Symptom.** In the dense linear_op sweep (3,327 token counts × TP {1,2,4,8}, 3 warm-up + 50 timed forwards per cell),
exactly one timed run — index 11 — of the `attn_pre_proj` scope took 150–290 ms instead of ~0.13 ms, for exactly the
eight token counts 1968–1975, at every TP, on two nodes 17 h apart (jobs 21313, 21334). Nothing else in the file is
anywhere near that. Medians were never affected.

**Cause (confirmed, 05 §3).** A CPython generation-2 (full) garbage collection on the worker's main thread, between the
scope's start CUDA event and the QKV GEMM launch. The CUDA-event gap therefore measures a host stall while the GPU
idles. Why that exact position:

- Each profiler task builds a fresh model, so the GC's gen-0 counter (a net count) restarts near 0 every task, grows
  ≈255 during model build and ≈32 per forward, and crosses the threshold of 700 at forward 14 (= timed run 11) and
  every 22 forwards after. Collections of any generation can only fire there.
- The full collection fires at the first such point after CPython's `long_lived_pending ≥ long_lived_total/4` rule
  is met, which in this process happens once per 416-task pass, at per-worker task 169. Tokens 1968–1975 are the
  eight tasks at that index (one per GPU worker, round-robin over a descending token list). Every TP pass starts
  fresh workers that replay the same sequence, so all four passes hit the same task.
- Cost 146–290 ms = traversing the worker's ~480k tracked objects (torch + vLLM import graph). Not contention: two
  workers instead of eight give the same durations.

**What was ruled out.** PyTorch caching allocator (counters flat), hipBLASLt lazy work, HIP/ROCr housekeeping,
8-process contention, shape, environment.

**How it was proven.** An opt-in diagnostic module, `frontier/profiling/linear_op/spike_diag.py` (inert unless
`FRONTIER_SPIKE_DIAG=<dir>` is set), records every GC collection and the host time of every forward per worker. Twelve
jobs on `amd-mi355x-8` (ids 21342–21353, 2026-09-16): the instrumented full grid, `gc.collect()` / `gc.freeze()` /
`gc.disable()` interventions, the token-list shift, a 2-worker run, an inert-module baseline, and a 200-repetition
run. 184 gen-2 collections recorded; the 176 that landed in a timed scope match the spike to within 1 ms;
`gc.disable()` alone removes the band; every positional prediction held. Full tables in 05 §2–3.

**Side findings.** (1) Runs 11 and 33 are systematically ~1.3× / ~1.05× the row median in *every* row, from the
cheap gen-0 collections landing there; harmless for medians. (2) A separate class of sporadic ≤2 ms host-jitter
outliers exists at random positions. (3) With 200 repetitions per task, 7 of 9 full collections per worker fall
between timed scopes and are invisible in the CSV although the GPU idles ~200 ms each time.

**Status.** Root cause confirmed and committed (`c7d49c3`). The profiler's default behaviour is unchanged; no fix
has been applied or re-run. The intended fix is `gc.disable()` before the warm-up loop and `gc.enable()` after the
final synchronize in `LinearOpWrapper.profile` (cyclic garbage is 133 objects per pass), followed by a full-grid
re-run against the acceptance criteria in 05 §7.

## Where everything is

Repo (branch `smatar/qwen3-30b-mi355-profiling`, worktree `/home/dn/Frontier-qwen3-profiling`):

- documents: this directory; code: `frontier/profiling/linear_op/{main.py,linear_op_wrapper.py,spike_diag.py}`;
  sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`
- original data with the spike: `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/2026-09-15_1308_linear_op_dense3327/`
  and `runs/2026-09-16_0629_linear_op_dense3327_rerun/`; failed 26-token repro: `runs/2026-09-15_1359_linear_op_spike_repro/`

Cluster only (not in git; `/opt/shared/frontier-qwen3-profiling/Frontier` is an rsync copy of the worktree, owned by `dn`):

- experiment CSVs: `data/profiling_spike_<e>/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` for
  `e ∈ {e1,e1b,e2,e2b,e3,e3b,e4b,e5,e6,e7,e8}` (~29 MB each)
- per-worker GC/host-time diagnostics: `data/profiling/sweep_work/logs/spike_diag/<e>/worker_gpu<k>_pid<pid>.jsonl`
- Slurm logs: `data/profiling/sweep_work/logs/spike-<e>-<jobid>.out`, profiler stdout `linear_op_<e>.log`

## How to reproduce or extend

```
# from the VM; always sudo -u dn on the cluster
rsync -c --rsync-path="sudo -u dn rsync" <files> cluster:/opt/shared/frontier-qwen3-profiling/Frontier/<path>/
ssh cluster "sudo -u dn bash -c 'cd /opt/shared/frontier-qwen3-profiling/Frontier && \
  STAGE=linear_op COLLECT_DIR=data/profiling_<tag> LINEAR_LOG_SUFFIX=_<tag> \
  LINEAR_DOCKER_ENV=\"-e FRONTIER_SPIKE_DIAG=/workspace/frontier/data/profiling/sweep_work/logs/spike_diag/<tag>\" \
  sbatch -w amd-mi355x-8 -t 00:30:00 -J spike-<tag> --export=ALL profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch'"
```

Knobs (env, all optional): `LINEAR_TOKENS_PY` (Python expression for the token list), `LINEAR_NUM_GPUS`,
`LINEAR_DOCKER_ENV` (extra `-e VAR=…` for the container), `LINEAR_LOG_SUFFIX`; inside the container
`FRONTIER_SPIKE_DIAG=<dir>`, `FRONTIER_SPIKE_GC=disable|freeze`, `FRONTIER_SPIKE_INIT_COLLECT=1`,
`FRONTIER_LINEAR_ACTIVE_STEPS=<n>` (default 50). A full grid takes ~2.5 min on one node; the JSONL is one line per task
with `gc_events`, `forwards_ms` (host window of each forward) and all per-op sample lists, so a 20-line script
correlating `gc_events[].start_ms` with `forwards_ms` reproduces every table in 05.
