# Profiling project summary: from the Frontier pipeline to Dataset B

This file explains the full flow of the profiling project. It covers the start, the findings, and the fixes.
Managers can read the Summary, the Timeline, and the **Status** part of each section.
Engineers can use the **Evidence** part of each section and the Evidence index.

Every claim has one status:

- **Confirmed**: the code or the history supports the claim. A citation follows.
- **Not verified in the code**: the repository has no evidence for the claim.
- **Contradicted**: the code shows a different fact. The file states the fact from the code.

Path rule: the "Q3 directory" is `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/`. A bare file name such as `07_post_proj_rope_dip_root_cause.md` is in the Q3 directory.
`REPORT.md` is `operator_boundary_investigation/REPORT.md`. `linear_op_wrapper.py` is `frontier/profiling/linear_op/linear_op_wrapper.py`.
All other paths start at the repository root. A commit hash is on the branch `smatar/qwen3-30b-mi355-profiling`.

## 1. Summary

1. The goal is a cost model of Qwen3-30B-A3B on AMD MI355X that the Frontier simulator can use.
2. The project started with a task: collect data with the Frontier pipeline and examine the regressor. The team did not understand why Frontier uses a random forest (RF) and not another predictor.
3. The examination found that the trainer reports the error on its own training data, so the reported results cannot be trusted. The collected data also did not make physical sense. Three root causes were found: a garbage collection stall, host time instead of GPU time, and a wrong RoPE operator.
4. Fixes for all three are in the code. Two run-position effects were also found and fixed.
5. Dataset A (operators measured in isolation) gives a regressor with a test MAPE of 1.25 to 1.28 percent and a P95 error of 2.5 to 4.0 percent for the tree models.
6. Dataset B (operators measured inside a real SGLang run) has a validated collector. A TP1 eager pilot passes the checks. No Dataset A to Dataset B comparison exists.
7. A source study shows that Frontier inherited its operator definitions from Vidur and vLLM on NVIDIA. Six of nine operators fit SGLang on AMD. Two operators are one fused kernel in SGLang. One operator is unused.
8. Five decisions (D1 to D5) are open. The Dataset B sweep waits for them.

## 2. Timeline

Dates are UTC dates of the commit or of the document. Verify each date with `git log`.

| Step | Question | Finding | Fix | Evidence (file or commit) |
|---|---|---|---|---|
| 2026-06-08 | What does Frontier measure and for which engine? | The public Frontier code inherits the Vidur operator split. It targets vLLM on NVIDIA. | None. This is the base. | `6897819`; `operator_boundary_investigation/evidence/E4_operator_partition_provenance.md` |
| 2026-08-09 | Can the Frontier pipeline collect data on MI355X? | It needed MI355X support and config fixes. | Add MI355X support. | `3d1cabd` |
| 2026-08-11 | What data exists for GPT-OSS and Qwen3? | The attention data uses the portable `TORCH_SDPA` backend, not the SGLang kernels. | Add the AITER dense backend. | `c6fada7`; `profiling_knowledge/AITER_KERNELS.md`; `645c98a` |
| 2026-08-11 to 2026-08-26 | Does the simulator match real serving? | The validation tool and its reports exist. One workload mismatch is open. | Validation tool only. | `99c1e6e`, `2177809`; `profiling_knowledge/REAL_BENCHMARK_DATA_QUALITY.md` |
| 2026-09-10 to 2026-09-14 | What must a new Qwen3 collection record? | The inherited data has 20 repetitions, aggregates only, and no `head_dim`. | Record all 50 timed runs in order. Require `head_dim`. | `00_requirements_and_source_map.md`; `deef568` |
| 2026-09-15 | Does the new AITER attention run? | The `v0.5.11` image faults on `head_dim` 128. | Use image `v0.5.17` for attention. | `02_run_record.md`; `b65842c`, `0fcac5f` |
| 2026-09-15 to 2026-09-16 | Why does one run take 150 to 290 ms? | A CPython full garbage collection stops the worker thread. | Disable garbage collection around the timed loop. | `05_linear_op_spike_root_cause.md`; `c7d49c3`, `74eee7c` |
| 2026-09-16 to 2026-09-17 | Why do the medians have a V-shaped dip? | The event timer measures the host launch span when the GPU is idle. | Time every shape twice. The GPU-bound pass is the primary column. | `07_post_proj_rope_dip_root_cause.md`, `09_measurement_validation_results.md`; `d4469be` |
| 2026-09-17 | What does the label mean? | It must be kernel time at a stated clock plus a separate overhead term. | Write the measurement contract. | `10_measurement_contract.md`; `ec63f31` |
| 2026-09-17 to 2026-09-22 | Does `attn_rope` time RoPE? | No. A forced torch fallback timed a wrong computation. | Use the fused kernel. Record `attn_rope_impl`. | `11_attn_rope_task.md`; `4b8fce3` |
| 2026-09-22 to 2026-09-23 | Why do run positions differ? | A spin backlog slows the clock. The ROCm runtime adds a barrier at 1000 commands. | Use a GEMM backlog and 8 settle forwards. Set `DEBUG_CLR_MAX_BATCH_SIZE=1000000`. | `13_run_position_anomalies_root_cause.md`; `3a8eb55`, `88da6a4` |
| 2026-09-23 | Can a regressor learn the data? | Yes. See section 9. | Hand off the settle-8 file. | `14_regressor_dataset_handoff.md`; `bd3bdf1`; `88463bf`, `58d7789`, `284130b` |
| 2026-10-04 to 2026-10-05 | Are the nine operator definitions right for SGLang on AMD? | Six fit. Two are one fused kernel. One is unused. | Report with five open decisions. | `operator_boundary_investigation/REPORT.md`; `68e1d7e` |
| 2026-10-07 | Can Dataset B be collected? | A TP1 eager pilot passes end to end. 2150 kernels per run have no operator label. | Fix six collector bugs. Add a canary check. | `237dd3e`, `fc3638c`; `dataset_b/REVIEW_INSTRUCTIONS.md` |

## 3. Starting point: collect data and examine the regressor

### Problem
The project started with a task: collect profiling data with the Frontier pipeline on MI355X, and examine the regressor.
Frontier reads profiling CSV files and trains one regressor for each operator. The team did not know why the pipeline uses a random forest (RF) and not another predictor.
To answer this, the team read the trainer. The trainer showed a problem with the reported results. This made the team look at the data itself.

### Evidence
- The trainer calls `grid_search.fit(X, y)` and then `grid_search.score(X, y)` on the same data. It logs the result as the error. See `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` lines 2840 and 2841.
- A code comment states that the trainer creates no train/test split (line 2836). The reported error is therefore a training error. It does not measure the error on new inputs.
- The RF is the default estimator. The linear model is the alternative. No document records why RF was chosen (`operator_boundary_investigation/evidence/E2_frontier_predictor_training_and_selection.md`).
- The Frontier code base starts at commit `6897819`. The AMD support commit is `3d1cabd`.
- The baseline data was collected with the `TORCH_SDPA` backend (`profiling_knowledge/MI355X_FOUR_MODEL_PROFILING.md`).

### Fix
- The team built a regressor bench with sealed train, test, and hold-out splits to compare predictors (section 9).
- The team then examined the data (section 4).

### Status
- Claim: the project started with the task to collect data and examine the regressor. **Confirmed** by the Qwen3 plan (`00_requirements_and_source_map.md`: collect data, then train regressors in the next phase) and by `88463bf`. The motivation "why RF" is **Not verified in the code**. No document states it.
- Claim: the reported accuracy cannot be trusted. **Confirmed** (`sklearn_execution_time_predictor.py` lines 2840 and 2841).

## 4. Data problems

### Problem
The dense `linear_op` data showed three patterns that a GEMM cannot produce:

- One timed run in 32 rows took 150 to 290 ms. The other 49 runs took approximately 0.13 ms.
- The medians of `attn_post_proj` and `attn_rope` fell by 17 to 37 percent between 4,000 and 6,000 tokens at TP 2, 4, and 8. They then rose.
- The curves were flat for thousands of tokens below the dip.

### Evidence
- `story/STORY.md` sections 3 to 5 and 7 give the plots and the job numbers. Job 21313 is the first dense run. Job 21334 is the replicate.
- The replicate on another node reproduced the same 32 rows. This rules out a one-time stall (`03_linear_op_spike_investigation.md`, `story/STORY.md` section 4).
- The figures are in `story/figures/`. The notebooks are `viz/linear_ops_run_groups_and_stats.ipynb` and `viz/linear_ops_old_vs_fixed_comparison.ipynb`.
- The ratio of old data to fixed data (median over the grid) is 1.3 to 2.9 for the GEMM operators and 4.3 to 12 for `attn_rope` (`12_dip_investigation_summary.md` section 5).

### Fix
Section 5 lists the root causes and the fixes.

### Status
- Claim: the collected data did not make sense when plotted. **Confirmed** (`story/STORY.md` chapters 1 to 6).

## 5. Bugs in collection and in reported accuracy

### Problem
Five collection bugs and four accuracy-reporting defects are in the record.

Collection bugs:

1. A CPython generation-2 garbage collection stops the worker thread inside a timed scope.
2. A CUDA-event pair measures the host launch span when the GPU is idle. The CSV number is then not GPU time.
3. The warm-up does not raise the GPU clock. The legacy pass starts at 648 to 948 MHz on 32 of 36 rows.
4. The `torch.cuda._sleep` backlog slows the next forwards by 2 to 8 percent in each block of 25.
5. The ROCm runtime adds a barrier of approximately 5.5 µs after every 1000 commands.

Accuracy-reporting defects:

1. The Frontier trainer fits on all data and then scores on the same data (`fit(X, y)` and `score(X, y)`). It reports a training error as the MAPE and has no hold-out split.
2. The validation tool simulates the nominal output length (1024 tokens). Real GPT-OSS runs on vLLM produce 57 to 71 tokens. The real-versus-simulated error is then not comparable.
3. The project acceptance tests T1 and T3 were wrong. No correct instrument could pass them. The team replaced them and kept the record.
4. `sanity_check.py` gave a pass twice when a positional path hid an option. It now gives a usage error.

### Evidence
- Bug 1: `05_linear_op_spike_root_cause.md` (twelve experiments E1 to E8). Code: `frontier/profiling/linear_op/linear_op_wrapper.py`, function `LinearOpWrapper.profile` (`gc.disable()` before the warm-up and `gc.enable()` in a `finally` block).
- Bug 2: `07_post_proj_rope_dip_root_cause.md`; the pre-registered test in `08_measurement_validation_preregistration.md` and `09_measurement_validation_results.md`. Code: `time_stats_hostbound` and `compute_legacy_host_bound` in `linear_op_wrapper.py`.
- Bug 3: `09_measurement_validation_results.md` sections 3a and 3b (jobs 21395 to 21400).
- Bug 4: `13_run_position_anomalies_root_cause.md` section 8 (jobs 21513 to 21519). Code: `BACKLOG_KIND` and `SETTLE_STEPS` in `linear_op_wrapper.py`.
- Bug 5: `13_run_position_anomalies_root_cause.md` and its addendum (jobs 21499 and 21500). Commit message of `3a8eb55`.
- Defect 1: `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` lines 2836 to 2841 (`fit(X, y)`, then `score(X, y)` on the same data; no train/test split). Also `operator_boundary_investigation/evidence/E2_frontier_predictor_training_and_selection.md`.
- Defect 2: `profiling_knowledge/REAL_BENCHMARK_DATA_QUALITY.md`. Code: `tools/validation/frontier_cli_translator.py` lines 162 and 173 pass `real.random_output_len`. The document says "Not yet fixed". The code still shows the nominal value.
- Defect 3: `12_dip_investigation_summary.md` section 3.
- Defect 4: `story/STORY.md` section 10. Code: `sanity_check.py` line 18.

### Fix
- Bug 1: disable garbage collection around the timed loop. Verified on the full grid against five criteria (`data/profiling_gcfix/RUN.md`, jobs 21361 and 21362). The largest sample in the new data is 2.09 ms.
- Bug 2: time every shape twice. The GPU-bound pass has a calibrated device backlog and is the primary column `time_stats`. The legacy pass is the column `time_stats_hostbound`. Commit `d4469be`.
- Bug 3: the per-row clock probe records `sclk_mhz_*` columns. Commit `9dac93b`.
- Bug 4: use a chain of allocating GEMMs as the backlog (`BACKLOG_KIND = "gemm_alloc"`). Run 8 untimed settle forwards (`SETTLE_STEPS = 8`). Commits `3a8eb55` and `88da6a4`.
- Bug 5: set `DEBUG_CLR_MAX_BATCH_SIZE=1000000` in the container.
- Defects 1 and 2: no fix in the code. The regressor bench uses a sealed train, test, and hold-out split (section 9).
- Defect 3 and 4: replaced tests and a usage error.

### Status
- Claim: bugs exist in the data collection. **Confirmed** (citations above).
- Claim: bugs exist in the reported accuracy. **Confirmed**, but as four separate defects. The repository has no single bug called "reported accuracy".
- Claim: the team fixed the major problems. **Confirmed** for bugs 1 to 4.
- **Contradicted in part** for bug 5. The flag is not a default in the code. `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` does not set it. The caller must pass it with `LINEAR_DOCKER_ENV`. `dataset_b/run_b.py` line 38 requires it for Dataset B. `story/STORY.md` section 14 says the flag is set in `LINEAR_DOCKER_ENV`. The sbatch default for that variable is empty.
- Open: the TP 1 and TP 2 decline of 0.9 to 1.3 percent over a block. It moves the median by less than 0.5 percent (`14_regressor_dataset_handoff.md`).

## 6. Silent kernel-selection failure

### Problem
Frontier can run, exit with code 0, and measure a different operator than the one the serving engine runs. Three cases are in the record.

1. **RoPE.** The collection script forced a torch fallback. The fallback rotated only the first 64 columns of head 0. It left 88 to 98 percent of the elements unchanged. The process ended normally.
2. **GPT-OSS attention.** Frontier binds GPT-OSS to the dense GQA attention family. It does not model alternating sliding-window attention or attention sinks. The profiler measures full attention for every layer.
3. **Failed jobs.** A pilot job ended with exit code 0 after a `mkdir` error. The attention sweep is non-fatal per cell by design.

### Evidence
- Case 1: `07_post_proj_rope_dip_root_cause.md` section 3.5; `11_attn_rope_task.md`. Code: `frontier/profiling/common/layers/rotary_embedding.py`, functions `_should_prefer_torch_rope_fallback`, `rope_impl_name`. The environment variable `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK` still works.
- Case 2: `profiling_knowledge/MI355X_FOUR_MODEL_PROFILING.md` section 2 ("Two gpt-oss architecture features are not modeled"). Code: the docstring of `frontier/profiling/attention/backends/aiter_attention_wrapper.py` ("Sliding-window and attention-sink cost is not exercised"). The Qwen3 config has `sliding_window: null` and `use_sliding_window: false` (`data/config/models/qwen3-a3b-30b-moe.json`).
- Case 3: `02_run_record.md` (job 21280). Code: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` ("the sweep is deliberately non-fatal per cell; make Slurm report the failure").

### Fix
- Case 1: `get_rope` prefers Frontier's own `RotaryEmbedding` with `vllm._custom_ops.rotary_embedding`. Each row records `attn_rope_impl`. `sanity_check.py` rejects every value except `vllm_kernel` (and `none`). Commit `4b8fce3`.
- Case 2: no fix. The wrapper docstring states the limit.
- Case 3: the sbatch checks that a CSV file exists and exits with code 1 if not.

### Status
- Claim: kernel selection failed silently, with exit code 0. **Confirmed** for cases 1 and 3.
- Claim: for GPT-OSS, Frontier profiles an attention kernel that GPT-OSS does not use. **Contradicted in part.** The AITER wrapper calls the same two entry points as SGLang: `mha_batch_prefill_func` and `paged_attention_ragged` (wrapper docstring; `operator_boundary_investigation/REPORT.md` rows 8 and 9). The mismatch is the attention variant (sliding window and sinks). Before the AITER wrapper, the `TORCH_SDPA` backend was in use. It is not the SGLang kernel (`profiling_knowledge/AITER_KERNELS.md`).

## 7. Model choice (Qwen3-30B-A3B) and why

### Problem
The team must choose one model for the profiling work. GPT-OSS needs sliding-window attention and sinks. Frontier cannot represent them.

### Evidence
- The request file (`/home/dn/qwen_profiling_plan_01.md`, outside the repository) selects Qwen3-30B-A3B. `00_requirements_and_source_map.md` records this ("request pre-selects 30B-A3B").
- `profiling_knowledge/AITER_KERNELS.md`: GPT-OSS and Qwen3-30B-A3B both use plain GQA attention.
- The Qwen3 config has no sliding window (section 6).
- For attention, Frontier and SGLang use the same two AITER entry points (`REPORT.md` rows 8 and 9).
- For RoPE and the KV cache write, they differ (section 11).

### Fix
Use Qwen3-30B-A3B (`qwen3-a3b-30b-moe`). Scope: the attention-layer operators. Mixture-of-experts operators are deferred.

### Status
- Claim: the team chose Qwen3-30B for now. **Confirmed** (`00_requirements_and_source_map.md`).
- Claim: the stated reason is that the kernels are similar enough. **Not verified in the code.** No document records this reason. The repository supports it for attention only. It contradicts it for RoPE and the KV cache write.

## 8. Measurement method (host time versus GPU time)

### Problem
The first profiler timed each operator with a CUDA-event pair around a Python scope. The pair measures the time from the moment the device reaches the start event to the moment it reaches the end event. If the device is idle, the pair measures the host launch span.

### Evidence
- Kernel traces and event medians disagree. At 8 to 64 tokens, the event value is 3.2 to 6.9 times the traced `o_proj` kernel (`09_measurement_validation_results.md`).
- With a GPU backlog, the event median equals the traced kernel plus 0.3 to 3.9 µs on all 32 shapes (`07_post_proj_rope_dip_root_cause.md`).
- The team wrote the predictions of both hypotheses before the run (`08_measurement_validation_preregistration.md`). The result: 262 of 369 numbers fall inside their bands, and 107 fall outside (`09_measurement_validation_results.md`).
- The profiler options are `cuda_event` and `record_function`. The sbatch runs both by default (`LINEAR_PROFILE_METHODS`). `record_function` writes `linear_op_kernel_only.csv`.
- The CSV columns are `time_stats.<operator>.median` (primary), `time_stats_hostbound.<operator>.*`, and the clock probe columns.

### Fix
The measurement contract (`10_measurement_contract.md`, commit `ec63f31`, agreed 2026-09-17) defines the label as follows:

- The label is kernel-only device time at a stated clock.
- A separate term holds the launch and gap overhead.
- Use the event median. Do not use the mean, the minimum, or the maximum (`14_regressor_dataset_handoff.md`).
- Each event scope adds approximately 2.5 to 3 µs.

### Status
- Claim: the team checked whether the profiler measures host time or GPU time. **Confirmed.**
- Claim: the original data was host time. **Confirmed.** The old data inflates the medians by 1.2 to 3 times over most of the TP greater than 1 grid (`story/STORY.md` section 8). In the new file, the legacy columns `time_stats_hostbound.*` are host launch spans on approximately 60 to 80 percent of the rows (`14_regressor_dataset_handoff.md`).
- Open: kernels below 10 µs have an instrument floor of ±3 to 5 µs. The locked-clock dataset is not collected.

## 9. Dataset A and the regressor result

### Problem
Dataset A is the standalone profile of each operator, one operator at a time. The simulator needs a regressor for each (operator, TP) pair. There are 15 regressors: 3 attention operators at 4 TP values plus 3 replicated operators at TP 1.

### Evidence
- The training file is `2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv`. Job 21539. MD5 `3635e4d305ae5cf84884e275d3ff37ef`. It has 13,308 rows (3,327 token counts at 4 TP values).
- The protocol is in `regressor_bench/README.md`. The split is by token value: 75 percent train, 15 percent test, 10 percent hold-out. The hold-out is sealed.
- Result of the baseline v2 (`regressor_bench/RESULTS_baseline_v2.md`, random split, test tier, 499 token values):

| Model | Mean MAPE over 15 regressors | Mean noise floor |
|---|---|---|
| `interp` | 1.37 percent | 1.19 percent |
| `rf` | 1.25 percent | 1.19 percent |
| `xgb_exact` | 1.27 percent | 1.19 percent |

- P95 of the absolute percentage error, computed from `regressor_bench/results/baseline_v2/test_predictions.csv`: `rf` 2.60 to 4.03 percent per regressor; `xgb_exact` 2.47 to 3.83 percent. The notebook text in `regressor_bench/build_eval_notebook.py` line 181 gives the tree range as 2.5 to 4.0 percent.
- The block split (gaps of 10 percent relative width) is harder. In `regressor_bench/RESULTS_full_v1.md`, the mean P95 of `rf_frontier` is 9.04 percent.
- The tile-selection steps of 15 to 30 percent at TP 2, 4, and 8 are real (`regressor_bench/RESULTS_baseline_v2.md`, section "Steps detected").

### Fix
No fix is needed in the data. The regressor family is open. The hold-out stays sealed until the project owner approves its use (`regressor_bench/README.md`, last command).

### Status
- Claim: a regressor reaches approximately 4 percent error at P95. **Confirmed with limits.** The value 4 percent is the upper end of a 2.5 to 4.0 percent range. The range is for the random split on the test tier. The mean MAPE is 1.25 to 1.28 percent. The block split gives a higher P95.
- Claim: Dataset A is the isolated operator data. **Confirmed.** Dataset A in this file means the `linear_op` data for the attention-layer operators. The attention CSV files are also called Dataset A in the Dataset B documents.
- The P95 values were computed in this work from the result file. They are not in a committed summary table.

## 10. Dataset B and the reason for it

### Problem
Dataset A never runs inside a real server. The team does not know if an isolated measurement predicts the cost of the same operator in a real run. The team also does not know which statistic (mean, median, or other) to use.

### Evidence
- The goal is stated in `dataset_b/HANDOFF.md` section 1 and `dataset_b/REVIEW_INSTRUCTIONS.md` section 1. Dataset B measures the same operators with GPU kernel timestamps while SGLang serves real requests.
- The code is in `dataset_b/`: `marker_runtime.py`, `sglang_b_markers.patch`, `drive_b.py`, `b_traces.py`, `b_extract.py`, `b_check.py`, `check_b.py`, `check_dataset_a_scope.py`, `run_b.py`, and `scope_spec.json`.
- Commit `237dd3e` documents jobs 21791 to 21798 for the TP1 eager pilot. It fixes six collector bugs. In job 21798, 3659 of 3659 in-scope kernels resolve. 2150 out-of-scope kernels are ignored by the canary check.
- The launch plan `dataset_b/launch_plan_tp1_eager_pilot.json` uses `--page-size`, `--disable-radix-cache`, and `--disable-cuda-graph`. It has no memory-limit option.
- The unit tests report 135 passed and 1 skipped (`dataset_b/IMPLEMENTATION_STATUS.md`). This work did not run them.

### Fix
Not complete. The collector works for TP1 eager. The following items are open:

- 2150 kernels per run have no operator label (MoE, sampling, scheduler support).
- Graph replay attribution fails by design.
- The TP4 pilot, the mixed batches, and the full sweep are not run.

### Status
- Claim: Dataset B measures the same operators inside a full model run to test whether Dataset A is related to the real run. **Confirmed** (`dataset_b/HANDOFF.md`).
- Claim: the run has memory constraints. **Not verified in the code.** The documents describe a real KV cache and real queue state. They do not describe a memory limit.
- Result: no Dataset A to Dataset B comparison exists. The documents state that this work is out of scope for the current phase.

## 11. Operator-definition mismatch (NVIDIA and vLLM versus AMD and SGLang)

### Problem
Frontier defines nine operators. The definitions come from Vidur and Sarathi-Serve, for vLLM on NVIDIA. The project runs SGLang on AMD. SGLang fuses two of the operators. These two operators do not run in isolation in a real Qwen3 run.

### Evidence
Source: `operator_boundary_investigation/REPORT.md` section 1 and the verdict matrix (section 4). The target is SGLang `v0.5.18` on the image `dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04`. The investigation read source and did not run a GPU job.

| Dataset A operator | Verdict in the report |
|---|---|
| `attn_pre_proj` | Keep. Add a provider tag. |
| `attn_post_proj` | Keep. Add a provider tag. |
| `input_layernorm` | Keep. Replace the harness (`x` and `residual` are the same tensor). |
| `post_attention_layernorm` | Keep. |
| `attn_prefill` | Keep. Add the fill kernel of the image. |
| `attn_decode` | Keep. Replace the harness (all sequences share one block table). |
| `attn_rope` and `attn_kv_cache_save` | Merge into one fused unit `attn_rope_kv_write`. On HIP, SGLang always runs `fused_qk_rope_reshape_and_cache`. |
| `emb` | Unused by the simulator (`predictor_target=False`). |

Other findings in the report:

- `SGLANG_USE_AITER` is off by default and the image does not set it. It decides the kernel provider for five of the nine operators.
- The simulator can double count an all-reduce if `--enable-aiter-allreduce-fusion` is on.
- The operators `attn_prefill_mixed` and `attn_decode_in_mixed` model a two-kernel mixed batch. SGLang does not produce it by default.
- Attention was never made GPU-bound in Dataset A, so its small-shape labels are host spans (`REPORT.md` section 1).
- Provenance: `operator_boundary_investigation/evidence/E4_operator_partition_provenance.md` and `operator_boundary_investigation/evidence/E7_vidur_sarathi_primary_sources.md`.

### Fix
The report proposes a partition of units U1 to U9 and P1 (`REPORT.md` section 7). No code change follows yet. The consumer code is not changed.

### Status
- Claim: Frontier's operator definitions are for NVIDIA with vLLM. **Confirmed** (`REPORT.md` sections 1 and 3, evidence E4 and E7).
- Claim: SGLang uses fused kernels, so two isolated operators do not run that way. **Confirmed** for RoPE with the KV cache write. The fused QK-norm with RoPE exists only for CUDA and is off by default. The fused all-reduce with norm is off by default.
- The report states that no kernel trace of the image exists. The kernel counts inside each unit are **Not verified in the code** (`REPORT.md` section 10).

## 12. Current decision and next step

### Problem
The team must decide which operators to measure in Dataset B and what to do with the other operators.

### Evidence
- The report recommends: keep six operators, merge two operators, and mark `emb` unsupported (`REPORT.md` section 4).
- The report lists five decisions that need the project owner (`REPORT.md` section 9):
  - D1: the SGLang flags (`SGLANG_USE_AITER`, the attention backend, all-reduce fusion, mixed chunk).
  - D2: accept the merge of RoPE and the KV cache write.
  - D3: include the once-per-forward work (embedding, final norm, `lm_head`, sampler).
  - D4: recollect Dataset A under the serving provider, or keep it as a control.
  - D5: page size 1 or 16.
- Commit `237dd3e` says that D1 to D5 are still open on 2026-10-07.
- The collector uses `dataset_b/scope_spec.json`. Its status is `provisional` and its version is `qwen3-provisional-0.1`. It has ten rules. The rules include `embedding`, the fused `rope_kv` (mapped to two Dataset A operators), and `attention_mixed`.
- The canary `check_dataset_a_scope.py` checks only the kernels that a rule maps to a Dataset A operator.

### Fix
No decision is recorded. The Dataset B collector keeps the scope provisional and records the Dataset A mapping in every row.

### Status
- Claim: the current decision is to use only the operators on which SGLang and Frontier agree. **Contradicted in part.** The report recommends this for six operators. It is not a recorded decision. The provisional scope also includes the fused RoPE unit, the embedding, and the mixed attention.
- Claim: the next step is to decide what to do with the remainder. **Confirmed** (D2 to D5, `REPORT.md` section 9).
- Next steps in order of priority (`REPORT.md` section 9):
  1. Fix the serving configuration and record it in the dataset identity.
  2. Measure the fused RoPE and KV cache write unit.
  3. Replace the three harness implementations.
  4. Add the conditional variants in the consumer code.
  5. Decide on the once-per-forward work.
  6. Recollect the linear operators under the chosen provider.

## 13. Open questions and risks

Open questions:

1. D1 to D5 are not answered (section 12). The Dataset B sweep must wait.
2. The dataset README names the 200-forward file (job 21486) as canonical. The handoff names the settle-8 file (job 21539). One of them must change (`story/STORY.md` section 15).
3. The team does not know why the `gemm` and `gemm_alloc` backlogs differ in job 21515 and not in job 21517.
4. The TP 1 and TP 2 decline over a block (0.9 to 1.3 percent) has no confirmed cause.
5. The hold-out tier of the regressor bench is sealed. The final regressor error is therefore not measured on the hold-out.
6. Dataset A attention was not made GPU-bound. Its small-shape labels are host spans (`REPORT.md` section 1).

Risks:

- The three Dataset A harness defects (aliased norm input, shared block table, one-slot KV writes) bias specific labels.
- The image uses three different kernel stacks across Dataset A and the serving image. Tile-switch token counts can move.
- The validation tool still simulates the nominal output length (section 5, defect 2).
- Dataset B uses `DEBUG_CLR_MAX_BATCH_SIZE=1000000`. This can differ from production.
- The 2150 unlabeled kernels per run prevent a full-dataset pass of `b_check.py`.
- Graph replay attribution is not solved. Production decode uses graphs by default.

## 14. Evidence index

Status codes: C = Confirmed, P = Partly confirmed (the file states the limit), N = Not verified in the code, X = Contradicted.

| Claim | Status | Files, commits, and result files |
|---|---|---|
| The project started with data collection and a regressor examination (why RF) | C for the task, N for the "why RF" motivation | `sklearn_execution_time_predictor.py`; `00_requirements_and_source_map.md`; `88463bf` |
| Plotted data did not make sense | C | `story/STORY.md`; `12_dip_investigation_summary.md`; `story/figures/` |
| Garbage collection stall | C | `05_linear_op_spike_root_cause.md`; `c7d49c3`, `74eee7c`; `frontier/profiling/linear_op/linear_op_wrapper.py`; `data/profiling_gcfix/RUN.md` |
| Event timer measured host time | C | `07_post_proj_rope_dip_root_cause.md`; `08_measurement_validation_preregistration.md`; `09_measurement_validation_results.md`; `d4469be` |
| Clock does not warm up | C | `09_measurement_validation_results.md` sections 3a and 3b; `9dac93b` |
| Spin backlog drift | C | `13_run_position_anomalies_root_cause.md`; `3a8eb55`, `88da6a4`; `run_position/README_dense_fixed_workbacklog_settle8.md` |
| Runtime barrier at 1000 commands | C | `13_run_position_anomalies_root_cause.md`; `3a8eb55` |
| Barrier flag is a default in the code | X | `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`; `dataset_b/run_b.py` line 38 |
| Trainer reports accuracy on its training data (`fit(X, y)` then `score(X, y)`) | C | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` lines 2836 to 2841; `operator_boundary_investigation/evidence/E2_frontier_predictor_training_and_selection.md` |
| Validation tool uses the nominal output length | C (still open) | `profiling_knowledge/REAL_BENCHMARK_DATA_QUALITY.md`; `tools/validation/frontier_cli_translator.py` |
| RoPE timed a wrong computation | C | `11_attn_rope_task.md`; `frontier/profiling/common/layers/rotary_embedding.py`; `4b8fce3`; `sanity_check.py` |
| Jobs end with exit code 0 after a failure | C | `02_run_record.md` (job 21280); `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` |
| GPT-OSS attention uses a different kernel | P (entry points are the same; the variant differs) | `frontier/profiling/attention/backends/aiter_attention_wrapper.py`; `profiling_knowledge/MI355X_FOUR_MODEL_PROFILING.md`; `profiling_knowledge/AITER_KERNELS.md` |
| Qwen3 chosen because the kernels are similar | N (the request selected the model) | `00_requirements_and_source_map.md`; `data/config/models/qwen3-a3b-30b-moe.json` |
| Measurement contract | C | `10_measurement_contract.md`; `ec63f31` |
| Regressor error at P95 is approximately 4 percent | P (2.5 to 4.0 percent, random split, test tier) | `regressor_bench/RESULTS_baseline_v2.md`; `regressor_bench/results/baseline_v2/test_predictions.csv`; `regressor_bench/build_eval_notebook.py`; `regressor_bench/RESULTS_full_v1.md` |
| Dataset B tests the relation of Dataset A to a real run | C | `dataset_b/HANDOFF.md`; `dataset_b/REVIEW_INSTRUCTIONS.md`; `237dd3e`, `fc3638c` |
| Dataset B has memory constraints | N | `dataset_b/launch_plan_tp1_eager_pilot.json` |
| Operator definitions come from NVIDIA and vLLM | C | `operator_boundary_investigation/REPORT.md`; `operator_boundary_investigation/evidence/E4_operator_partition_provenance.md`; `operator_boundary_investigation/evidence/E7_vidur_sarathi_primary_sources.md`; `68e1d7e` |
| SGLang fuses RoPE with the KV cache write | C (source reading, no trace) | `operator_boundary_investigation/REPORT.md` section 5.2; `operator_boundary_investigation/evidence/E5_image_and_sglang_source_verification.md` |
| Decision: keep only the shared operators | X (recommendation, not a decision) | `operator_boundary_investigation/REPORT.md` sections 4 and 9; `dataset_b/scope_spec.json`; `dataset_b/check_dataset_a_scope.py` |
| Next step: decide the remainder | C | `operator_boundary_investigation/REPORT.md` section 9; `237dd3e` |

## 15. How to reproduce

I did not run any command in this section. I took each command from a file in the repository and checked that the file exists. Some commands need the Slurm cluster or the shared dataset store. Run each command from the repository root unless the file says otherwise.

Data collection on the cluster (from `HOWTO_run_linear_op_collection.md` section 2):

```bash
ssh cluster "sudo -u dn bash -s" <<'EOF'
cd /opt/shared/frontier-qwen3-profiling/Frontier
TAG=dense_$(date +%Y%m%d_%H%M)
mkdir -p data/profiling_$TAG && chmod -R a+rwX data/profiling_$TAG
export STAGE=linear_op COLLECT_DIR=data/profiling_$TAG LINEAR_LOG_SUFFIX=_$TAG
export LINEAR_PROFILE_METHODS=cuda_event
export LINEAR_DOCKER_ENV="-e FRONTIER_LINEAR_ACTIVE_STEPS=25 -e DEBUG_CLR_MAX_BATCH_SIZE=1000000"
sbatch --parsable -t 03:00:00 -J q3-$TAG --export=ALL profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch
EOF
```

The HOWTO file uses `FRONTIER_LINEAR_ACTIVE_STEPS=200` in this command. It does not contain `DEBUG_CLR_MAX_BATCH_SIZE`. I added the flag from `3a8eb55` and `dataset_b/run_b.py`.
Do not set `COLLECT_DIR=data/profiling`. That path is the tracked dataset tree.

Dataset check (from `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py` usage text):

```bash
python3 profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py <dataset-dir>
```

Regressor bench (from `regressor_bench/README.md`):

```bash
source /home/dn/.virtualenvs/qwen3-profiling/bin/activate
B=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench
python -m pytest $B/tests -q
python $B/baseline_v2.py
python $B/build_eval_notebook.py --execute
python $B/build_per_sample_notebook.py --execute
```

The bench needs a local mirror of the settle-8 file with MD5 `3635e4d305ae5cf84884e275d3ff37ef`. The README gives the path. Do not run `bench.py holdout --unseal` until the project owner approves.

P95 check (a command that I ran in this work; it reads `test_predictions.csv`):

```bash
python3 - <<'EOF'
import pandas as pd
p = "profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench/results/baseline_v2/test_predictions.csv"
d = pd.read_csv(p)
d["ape"] = (d["pred"] - d["y"]).abs() / d["y"] * 100
print(d.groupby(["model", "op", "tp"]).ape.quantile(0.95).unstack(0).round(2))
EOF
```

Dataset B local checks (from `dataset_b/IMPLEMENTATION_STATUS.md`; I did not run them):

```bash
B=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b
PY=/home/dn/.virtualenvs/qwen3-profiling/bin/python
$PY -m pytest "$B/tests" -q
$PY "$B/check_b.py" "$B/evidence/synthetic_validation/bundle.json"
```

Figures (from `story/STORY.md`, last section; I did not run it):

```bash
/usr/bin/python3 profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/story/make_story_figures.py
```

## 16. Glossary

| Term | Meaning in this file |
|---|---|
| operator | A named part of the model forward pass that the profiler times as one unit. Examples: `attn_pre_proj`, `attn_rope`. |
| kernel | One GPU program launch. An operator can contain one or more kernels. |
| Dataset A | The operators timed one at a time, outside a server, as Frontier does. |
| Dataset B | The same operators timed inside a real SGLang serving run. |
| label | The value that a regressor predicts: `time_stats.<operator>.median` in ms. |
| TP | Tensor-parallel size. The number of GPUs that share one layer. |
| host time | The time that the CPU spends to launch work. |
| GPU time | The time that the GPU spends to run the kernels. |
| GPU-bound pass | A timed pass with a device backlog, so that the event pair measures GPU time. |
| legacy pass | The first timed pass without a backlog. It can measure host time. |
| backlog | GPU work that the host enqueues before the timed loop. |
| settle forward | An untimed forward after the backlog. |
| fused kernel | One kernel that does the work of two or more operators. |
| MAPE | Mean absolute percentage error. |
| P95 | The 95th percentile of the absolute percentage error. |
| test tier | The 15 percent of token values that the regressor does not see in training. |
| hold-out | The sealed 10 percent of token values. No one scores it until approved. |

## 17. Terms where STE compliance is not certain

The author has no official ASD-STE100 dictionary. The final report in the chat lists these terms:
"MAPE", "P95", "backlog", "settle", "fused", "hold-out", "canary", "provider", "tile", "unused", "rope", "garbage collection", "pre-registered", "replicate", "harness", "tier", "pilot".
