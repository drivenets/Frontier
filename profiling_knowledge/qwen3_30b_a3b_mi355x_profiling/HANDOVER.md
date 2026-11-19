# Handover: Qwen3-30B-A3B operator profiling on MI355X

Written 2026-11-19 for the engineer who continues this work. The owner, Shaked Matar, may not be available.

- Repo: github.com/drivenets/Frontier, branch `smatar/qwen3-30b-mi355-profiling`.
- Worktree: `/home/dn/Frontier-qwen3-profiling` on the VM `smatar-dev`. The VM has no GPUs.
- GPU work runs on the MI355X Slurm cluster. Reach it with `ssh cluster` (login node `amd-mi355x-1`).
- Run every Slurm command on the cluster as `sudo -u dn ...`.
- `/opt/shared` is the only path the cluster nodes share. The VM does not mount it.

Labels used below:

- **measured**: a number from a run. The cited document reports it.
- **reading**: an interpretation that the cited document makes.
- **owner's note**: from the owner's notes. I did not find it in the documents.
- **checked 2026-11-19**: I checked it while I wrote this file.

All file names without a path are in this folder (`profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/`).

## 1. What the project is, and why

- The Frontier simulator predicts LLM serving time. It uses one cost model (a regressor) per operator.
- Each regressor trains on profiling CSV files.
- This project collects those files for Qwen3-30B-A3B (`qwen3-a3b-30b-moe`) on AMD MI355X.
- The scope is the attention layer: `attn_pre_proj`, `attn_post_proj`, `attn_rope`, two norms, `emb`, and AITER attention.
- MoE experts are out of scope (`01_plan.md`).
- The inherited MI355X data used TORCH_SDPA attention, 20 repetitions and aggregates only (`story/STORY.md` section 1).
- The new collection keeps all 50 timed runs of every shape, in order. This made the later bug hunts possible.
- The Frontier trainer scores on its own training data. Its reported error is a training error (`docs/profiling_project_summary.md` section 3).
- The new data showed patterns that a GEMM cannot produce. Each pattern was a measurement bug (section 2).

## 2. Status

### Done

| What | Evidence |
|---|---|
| Fresh collection: AITER attention, `head_dim` recorded, 3 warm-up + 50 timed runs, all samples kept | `02_run_record.md`; commit `deef568` |
| Spike of 150-290 ms (measured) in one run: CONFIRMED as a CPython gen-2 garbage collection | `05_linear_op_spike_root_cause.md`; fix `74eee7c` |
| V-shaped dip in medians: the CUDA-event pair measured the host launch span, not GPU time | `07_`, `08_`, `09_`, `12_dip_investigation_summary.md`; fix `d4469be` |
| Clock probe on every row | `09_` sections 3a-3b; commit `9dac93b` |
| Measurement contract: label = kernel time at a stated clock, plus a separate overhead term | `10_measurement_contract.md`; commit `ec63f31` |
| `attn_rope` timed a wrong torch fallback. The profiler now uses the fused kernel | `11_attn_rope_task.md`; commit `4b8fce3` |
| Two run-position effects: spin-backlog clock drift and a ROCm 1000-command barrier | `13_run_position_anomalies_root_cause.md`; commits `3a8eb55`, `88da6a4` |
| Fixed datasets staged on shared storage with README and SHA256SUMS | `story/STORY.md` section 13; `14_regressor_dataset_handoff.md` |
| Regressor bench with sealed splits. Mean test MAPE over 15 regressors (measured): `rf` 1.25 %, `xgb_exact` 1.27 %, `interp` 1.37 %; noise floor 1.19 % | `regressor_bench/RESULTS_baseline_v2.md` (random split, test tier) |
| P95 error 2.5-4.0 % for the tree models (measured from the result file, not in a committed table) | `docs/profiling_project_summary.md` section 9 |
| Operator-boundary study: 6 of 9 operators fit SGLang on AMD. RoPE and KV write are one fused kernel. `emb` is unused | `operator_boundary_investigation/REPORT.md` (source reading only, no trace) |
| Profiler fixes upstreamed to github.com/drivenets/dnai-Frontier, branch `smatar/profiler-measurement-fixes` | checked 2026-11-19 with `git ls-remote`: head `1090a13` |
| Dataset B collector: a TP1 eager pilot passes end to end. The work then moved to HANDS (section 5) | commit `237dd3e`; `dataset_b/REVIEW_SESSION_SUMMARY_STE.md` |

### Open

See section 6 for the full list. The top three:

1. Two files compete as "the training file". Pick one (section 4).
2. Decisions D1-D5 of `operator_boundary_investigation/REPORT.md` section 9 are open.
3. The GC-fix acceptance criterion 3 failed as written. Nobody re-stated it or signed off (section 6, item 7).

### Not tested

- No Dataset A to Dataset B comparison exists.
- The regressor hold-out tier is sealed. No model has a hold-out score.
- Kernels below about 10 µs are not cross-checked by a second instrument.
- No locked-clock dense dataset exists.
- Whether the GC stall matters in real serving is SPECULATIVE (`05_` section 4).
- The kernel count inside each fused unit is not verified by a trace (`REPORT.md` section 10).
- The attention datasets never got the per-run checks that `linear_op` got (`story/STORY.md` section 15).
- Attention was never made GPU-bound. Its small-shape labels are host spans (reading, `REPORT.md` section 1).
- For this handover I ran no tests and no GPU jobs.

## 3. Reading order

1. This file.
2. `../../docs/profiling_project_summary.md`: the full flow. It labels each claim Confirmed, Not verified or Contradicted.
3. `12_dip_investigation_summary.md`: the start of the dip story, in plain language.
4. `story/STORY.md`: every chapter with job ids and commits.
5. `story/review/STORY_CORRECTED.md` and `story/review/CORRECTIONS_TO_EXISTING.md`: corrections from a re-check on 2026-10-11.
6. `05_linear_op_spike_root_cause.md`: the GC stall, with CONFIRMED / RULED OUT / SPECULATIVE labels.
7. `13_run_position_anomalies_root_cause.md`: the two run-position effects and their fixes.
8. `10_measurement_contract.md`: what the label means.
9. `14_regressor_dataset_handoff.md`: rows, label, features and cautions for the regressor.
10. `regressor_bench/README.md`, then `regressor_bench/RESULTS_baseline_v2.md`.
11. `HOWTO_run_linear_op_collection.md`: the runbook.
12. `operator_boundary_investigation/REPORT.md`: operator definitions and decisions D1-D5.
13. History, if needed: `README.md` (folder map), `00_`, `01_`, `02_`, `03_`, `04_`, `06_` to `09_`, `11_`.

Note: the "Status" paragraph in `README.md` says no GC fix was applied. That text is out of date. The fix is `74eee7c`.

## 4. Where the data is

No measurement data is in this commit. Data lives on the cluster or locally on the VM.

### 4.1 Shared dataset store (cluster)

Root: `/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/`.
Each run folder has `README.md` (its provenance) and `SHA256SUMS`.

Two files compete as the training file:

| Folder under `linear_op/` | Job | What the documents say | sha256 of `linear_op.csv` (checked 2026-11-19) |
|---|---|---|---|
| `2026-09-22_0849_dense_200fwd_all_fixes/` | 21486 | Named canonical for training in `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/README.md` and in `12_` section 5 | `73fb3e8f6295f498…` |
| `2026-09-23_0949_dense_workbacklog_settle8/` | 21539 | The file `14_` points to. `regressor_bench` trains on it (md5 `3635e4d305ae5cf84884e275d3ff37ef`) | `208a29b1bdac0f56…` |

- The 200-forward medians are 2-3 % higher (measured, `14_` section 5). Its clock was still ramping (reading).
- The 200-forward file predates the settle-forward fix. Its per-run samples carry both effects of `13_` (`HOWTO_` section 7).
- `story/STORY.md` section 15 and the summary section 13 list this conflict as open.
- Owner's note: if you keep the 200-forward file, re-collect when regressor residuals show block-position structure.

Other content of the store (checked 2026-11-19):

- `linear_op/`: older runs, moved there on 2026-09-23 and 2026-10-11 to 10-14.
- `attention/`: the 16k, 32k and 64k AITER runs.
- `analysis/`: `story_evolution_cache`, `story_inputs`, `story_presentation` (inputs of the story notebooks).
- `linear_op/2026-11-10_*` and `moe/`: their READMEs say they come from `/home/dn/dnai-Frontier`, branch `feature/profiling-aiter`.
- The documents in this branch do not describe the 2026-11-10 runs.

### 4.2 Cluster scratch trees

- The cluster checkout `/opt/shared/frontier-qwen3-profiling/Frontier` is an rsync copy of this worktree. It is not a git clone.
- Experiment trees: `/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling_*`. `story/STORY.md` section 16 maps them to jobs.
- Spike diagnostics (GC experiments): `/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/spike_diag/<e>/`.
- Slurm and profiler logs: `/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/`.

### 4.3 Data still in git

- `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/` holds CSVs that predate the shared-storage rule.
- The root `linear_op.csv` there is the pre-fix 2026-09-15 file. Do not train on it (`14_` section 1).
- The root `attention_combined.csv` is what the simulator reads.

### 4.4 Local only on the VM (left out of git)

| Path on `smatar-dev` | Size | What it is | How to get it again |
|---|---|---|---|
| `/home/dn/Frontier-qwen3-profiling/data/profiling_gcfix/` | 28 MB | GC-fix check, job 21361, with `RUN.md` | CSV equals `linear_op/2026-09-16_0853_gcfix_verification/` in the store (sha256, checked 2026-11-19) |
| `/home/dn/Frontier-qwen3-profiling/logs/qwen3_mi355x/` | 14 MB | Slurm and profiler logs of the 2026-09-15 pilots and first runs (jobs 21280-21311) | Local only |
| `/home/dn/Frontier-qwen3-profiling/data/local_datasets/` | 87 MB | Git-ignored mirrors of the settle-8 file and the job 21519 settle-3 file. `regressor_bench` reads them | Copy from the store |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench/results/` | 97 MB | Bench outputs (git-ignored) | `regressor_bench/README.md` quick start |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench/EVAL_baseline_rf_xgb.html` | 3.1 MB | Rendered evaluation notebook (git-ignored) | `build_eval_notebook.py --execute` |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench/DATA_median_per_op.ipynb` | 680 KB | The working copy keeps its plot outputs. Git holds it with outputs cleared | Run the notebook |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench_nuzan_sidecar/data/` | 508 KB | Pre-fix CSV taken from commit `3d1cabd` (git-ignored) | `SIDECAR_REPORT.md` gives the `git show` command |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench_nuzan_sidecar/results/` | 184 KB | Side-comparison outputs (git-ignored) | `run_sidecar.py` |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/story/evolution/cache/` | 18 MB | Per-step CSV caches D0-D5 (git-ignored by the root `cache*` rule) | Store folder `analysis/story_evolution_cache` (contents not compared) |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/story/evolution/evolution.ipynb` | 14 MB | Evolution notebook with outputs | `story/evolution/build_evolution_notebook.py` |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/story/evolution/figs/` | 11 MB | Figures of that notebook. No document embeds them | Same builder |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/story/evolution/dataset_table.csv` and `section2_clipped_counts.csv` | 12 KB | Builder output tables | Same builder |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/story/presentation_11_10_26/numbers.json` and `numbers_v2.json` | 68 KB | Measured ratios. `numbers.md` and `numbers_v2.md` (in git) hold the same tables | `make_presentation_figures.py`, `make_presentation_figures_v2.py` |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/story/figures/attn_pre_proj_median_no_band.png` | 72 KB | One figure that no document uses | Local only |
| `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/evidence/pilot_runs/` and `rocprofv3-format-probe*` | 1.1 GB | Raw Dataset B evidence (git-ignored) | HANDS `dataset-b/migration/` freeze manifest lists them |

## 5. Related places

- **Dataset B** moved to `/home/dn/HANDS/dataset-b` (repo github.com/drivenets/HANDS). It was frozen from commit `25e163c`.
- Start at `/home/dn/HANDS/dataset-b/README.md`. On 2026-11-19 it says the project is parked.
- The `dataset_b/` folder here is a frozen copy. Do not continue Dataset B here.
- **Profiler fixes upstream**: github.com/drivenets/dnai-Frontier, branch `smatar/profiler-measurement-fixes` (pushed).
- **Confluence (DAD space)**:
  - Plan page: https://drivenets.atlassian.net/wiki/spaces/DAD/pages/7126188071
  - Bug-hunt story page: https://drivenets.atlassian.net/wiki/spaces/DAD/pages/7166984214 (published 2026-09-24).
  - On 2026-10-11 the corrected story in `story/review/` was not published (`story/review/FINDINGS_LOG.md`). Check the live page first.
- **Jira**: GPU-977 (Done) tracks the `linear_op` story.
- **Process records** are local files, because `.claude` is git-ignored:
  - `/home/dn/amd-playground/.claude/debug-reports/debug_linear_op_host_bound_timing_2026-09-16.md`
  - `/home/dn/amd-playground/.claude/debug-reports/implement_linear_op_two_column_timing_2026-09-17.md`
  - `/home/dn/amd-playground/.claude/debug-reports/implement_linear_op_recollection_2026-09-17.md`
  - `/home/dn/amd-playground/.claude/plans/linear-op-recollection-contract-a.md` (and `.iterations.md`)
  - `/home/dn/amd-playground/.claude/codex_verdicts/debug_verdict_linear_op_iter3.md` and `debug_verdict_linear_op_iter4.md`
- **Other documents in this repo**:
  - `../../docs/profiling_walkthrough.md` (Marp slides) and `../../docs/walkthrough.html`: short walkthroughs.
  - `../../docs/profiling_slides_kit/`: figure kit for slides.
  - `../../docs/HANDS_profiler_review.md`: a 2026-10-07 review of the HANDS profiler.
  - `regressor_bench_nuzan_sidecar/SIDECAR_REPORT.md`: RF and XGBoost on the pre-fix file. Not an input to current regressors.
  - `story/CONFLUENCE_STORY_REVIEW_HANDOFF.md`: the brief for the 2026-10-11 story re-check.
- **Local Python environment**: `/home/dn/.virtualenvs/qwen3-profiling`.

## 6. Known open items

1. **Training file.** The dataset README and `14_` name different files (section 4.1). Decide, then fix the other document.
2. **Settle-forward fix.** The 200-forward file predates it (`HOWTO_` section 7). See the owner's note in section 4.1.
3. **Small kernels.** Kernels below about 10 µs have a ±3-5 µs instrument floor (reading, `14_` section 5). No second instrument cross-checked them (`12_` section 5).
4. **Locked clock.** No locked-clock dense dataset exists (`story/STORY.md` section 15).
   - `profiling_knowledge/scripts/slurm/qwen3_fixed_clock_validation.sbatch` covers only the 8-token validation grid.
   - Its trace step sets `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1`. That predates the RoPE fix. Change it before reuse.
   - `HOWTO_` section 2.2 says a general collection script is a planned task.
5. **CSVs in git.** About 368 MB of CSVs sit under `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/` (checked 2026-11-19 with `du`).
   - Owner's note says about 300 MB. `story/STORY.md` section 13 says these files predate the shared-storage rule.
   - Removing them needs a decision. Git history keeps them anyway.
6. **Scratch duplicates.** Owner's note: three cluster scratch trees are deletable duplicates. Checked 2026-11-19 by sha256:
   - `Frontier/data/profiling_dense200/` CSV equals `linear_op/2026-09-22_0849_dense_200fwd_all_fixes/`.
   - `Frontier/data/profiling_dense_fixed/` CSV equals `linear_op/2026-09-22_0647_dense_two_column_rope_fix/`.
   - `Frontier/data/profiling_rope_fix2/` CSVs equal `linear_op/2026-09-17_1250_linear_op_rope_fix_validation_grid/`.
   - `profiling_rope_fix2/` also holds `profiler_traces/` (about 120 MB). The staged folder does not have them.
   - Nothing was deleted. Delete only after a decision.
7. **GC-fix acceptance.** Jobs 21361 and 21362 ran the full grid against the five criteria of `05_` section 7.
   - Criteria 1, 2, 4 and 5 pass. Criterion 3 fails as written (measured, `data/profiling_gcfix/RUN.md`).
   - The same band also fails between two unmodified baseline runs. `RUN.md` reads this as noise in tiny kernels.
   - `RUN.md` was left uncommitted for that reason. Re-state criterion 3 or record a sign-off.
   - Later dense runs with all fixes: worst sample 0.74 ms over 200 samples per row (measured, `12_` section 5).
8. **Residual drift.** At TP 1 and 2 the samples decline 0.9-1.3 % over a block (measured). Medians move less than 0.5 %. Cause not confirmed.
9. **Backlog kinds.** Why `gemm` and `gemm_alloc` differ in job 21515 but not in 21517 is not known (`story/STORY.md` section 15).
10. **Instrument choice.** Event-pair kernel time or kineto kernel-only time: not decided (`story/STORY.md` section 15).
11. **Run pointers.** Jobs 21519 and 21539 have no `runs/.../RUN.md` pointer in git (`story/STORY.md` section 15; checked 2026-11-19).
12. **Barrier flag.** `DEBUG_CLR_MAX_BATCH_SIZE=1000000` is not a default in the sbatch. Pass it in `LINEAR_DOCKER_ENV` (summary section 5).
13. **Operator decisions.** D1-D5 of `REPORT.md` section 9 are open. They block a Dataset B sweep.
14. **Frontier accuracy defects.** The trainer reports a training error. The validation tool uses the nominal output length. Neither has a code fix (summary section 5).
15. **Hold-out.** `regressor_bench/README.md` says: unseal the hold-out only when the user approves.
16. **Attention set.** Block 16 may not belong in the canonical attention trio. Cold-start timing was never measured (`02_run_record.md`).

## 7. How to run things

- **Runbook**: `HOWTO_run_linear_op_collection.md`. It covers sync, submit, knobs, checks, staging and tests.
- **Main sbatch**: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`. `STAGE=` selects the profiler.
- **Locked clock**: `profiling_knowledge/scripts/slurm/qwen3_fixed_clock_validation.sbatch`.
  - It locks all eight GPUs with `rocm-smi --setperfdeterminism <MHz>`. An `EXIT` trap resets them.
  - It needs the whole node (`--gres=gpu:8`). Never lock clocks outside a job (`HOWTO_` section 2.2).
  - `--setperfdeterminism` works for this account (`12_` section 3).
  - One probe once left a GPU locked for two minutes outside a job (`09_` section 3b).
- **Checks**: `sanity_check.py <dir>` and `test_measurement_validity.py <csv> --expect green`. Both take the path as a positional argument.
- **Run-position probes**: `posprobe.py` and `../scripts/slurm/qwen3_posprobe*.sbatch`.
- **Regressor bench**: the quick start in `regressor_bench/README.md`.
- **Figures**:
  - `story/make_story_figures.py` (story figures).
  - `story/evolution/build_evolution_notebook.py` (evolution notebook).
  - `story/presentation_11_10_26/make_presentation_figures.py` and `_v2.py` (presentation figures).
  - `story/review/make_corrected_figures.py` (corrected figures).
- **Cluster rules** (`02_run_record.md`, `HOWTO_` section 7):
  - `/opt/shared` uses root_squash. Create output folders and `chmod -R a+rwX` them before a job.
  - Docker images differ per node. Pin the node with `-w` when needed.
  - Sync code by explicit path with rsync. Never use `--delete`.

## 8. Git state

- Before this handover the branch was 3 commits ahead of origin: `237dd3e`, `fc3638c`, `25e163c`.
- Do not rebase, amend or rewrite them. HANDS cites `25e163c` as its freeze source.
- The handover commit adds documents, scripts, figure manifests and figures.
- It adds no CSV, no measurement JSON, no logs and no traces. Section 4.4 lists what stays local.
