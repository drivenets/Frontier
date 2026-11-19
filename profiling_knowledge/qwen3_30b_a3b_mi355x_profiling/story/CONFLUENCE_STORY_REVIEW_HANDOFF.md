# Handoff: independently verify and rewrite the Qwen3 / MI355X profiling story

Prepared 2026-10-11 for the agent working on the replacement Confluence page.

## Read this first

**These are some findings from an earlier review, not an authoritative verdict or a complete list of mistakes. Double-check and independently verify them. Do not follow these suggestions blindly, copy the numbers without recomputing them, or treat this document as evidence for its own claims. The reviewer may have made mistakes or missed context.**

Your task is to investigate the implementation, computation scripts, raw results, and document history, then write a complete, corrected, updated story in English. Confirm, revise, or reject the findings below based on evidence. Look for additional problems beyond this list. Preserve conclusions that withstand scrutiny; do not assume every earlier conclusion is wrong.

**You must print the investigation you actually performed before presenting the new story.** Provide an explicit, reproducible account of sources inspected, calculations and commands executed, results, discrepancies, and unresolved questions. A statement such as “I verified everything” is insufficient. This means an evidence and methods report, not private internal reasoning.

Aim for a story with no known factual or numerical errors and no unsupported conclusions. Do not promise that a review proves the absence of all mistakes. If evidence is missing, narrow the claim, label the uncertainty, and specify the additional check needed. Still complete the new story using the evidence that is available; do not stop at a plan or a list of recommendations.

## User intent and constraints

- Replace the old narrative with a new story incorporating the new work and corrections, rather than merely patching Chapter 7.
- The user will answer Neria in English; the earlier request for a Hebrew response was superseded.
- Do not launch new GPU experiments. Recompute from existing data. Preserve raw data and existing source artifacts during the audit; write separate report/story outputs.
- The user is considering deleting or archiving the old Confluence page. That choice has not been settled. This handoff does not itself authorize deleting or archiving it. Follow the user's actual publication instructions in your session; prepare a complete reviewable replacement regardless of publishing access.
- The earlier review did not access a verified copy of the live Confluence page. Do not assume it exactly matches any local file. Retrieve its contents and relevant page history if your access permits; otherwise state this limitation.

Old page, described by the user as published September 24:

https://drivenets.atlassian.net/wiki/spaces/DAD/pages/7166984214/Qwen3-30B-A3B+on+MI355X+the+linear_op+profiling+story

## Establish the source versions first

Repository: `/home/dn/Frontier-qwen3-profiling`.

Branch during review: `smatar/qwen3-30b-mi355-profiling`; HEAD was `25e163c`. The initial Chapter 7 review targeted `4d18e8d`. Recheck the current state rather than assuming these are still current.

Paths below are repository-relative. Let `K` denote `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling` only as a reading abbreviation, not a literal path.

| Material | Starting points |
|---|---|
| Original narrative and publishing pipeline | `K/story/STORY.md`, `STORY.ipynb`, `STORY.confluence.html`, `make_story_figures.py`, `build_story_notebook.py`, `build_confluence_html.py`, `captions.json` |
| Updated local narrative | `K/story/evolution/evolution.ipynb`, `build_evolution_notebook.py`, `dataset_table.csv`, `INCONSISTENCIES.txt`, `cache/` |
| October 11 figures and calculations | `K/story/presentation_11_10_26/README.md`, `make_presentation_figures.py`, `numbers.md`, `numbers.json`, `manifest.json` |
| Original investigations | Numbered documents in `K`, especially `05_`, `08_`, `09_`, `10_`, `13_`, `14_`; follow their evidence references |
| Timing implementation | `frontier/profiling/linear_op/linear_op_wrapper.py`, `linear_op_impl.py`; follow calls into timer/statistics implementation |
| Validation | `K/test_measurement_validity.py`, `tests/unit/test_linear_op_timing_columns.py`, historical versions and run outputs |
| Regression results | `K/regressor_bench/RESULTS_baseline_v2.md`, `baseline_v2.py`, `results/baseline_v2/`, split definitions |
| Operator boundaries | `K/operator_boundary_investigation/REPORT.md` and its source evidence |
| Serving-side pilot | `K/dataset_b/REVIEW_INSTRUCTIONS.md`, `TP1_EAGER_REVIEW_STE100.md`, current implementation and `evidence/pilot_runs/` |

The previous review found `STORY.md` unchanged since the older publication period, while the evolution and October 11 presentation directories were untracked. The newer narrative figures still ended at the September 23 dataset; later work lived in separate reports. Verify all of this. Record hashes for untracked inputs because a Git commit alone does not identify them. Do not equate a recent figure-generation date with a recent experiment date.

Line numbers cited below refer to the files as reviewed and may move. Locate the actual statements and record current file/line or commit references in your report.

## Required investigation and deliverables

1. Identify the old page/version, updated narrative, relevant code revisions, datasets, jobs, and timestamps. Separate experiment chronology from a conceptual ordering of fixes.
2. Define every timing quantity: boundaries, included/excluded work, units, repetitions, warm-up and settle treatment, aggregation, and ratio direction. Distinguish H0/H1 hypotheses from informal h0/h1 names for measurement curves.
3. Inspect the historical code used for each run as well as current code. Later fixes must not be attributed retroactively to earlier experiments.
4. Recompute headline values from raw samples/traces wherever available, not just `numbers.md`, cached summaries, figures, or prose. Verify joins, row counts, missing/duplicate shapes, units, sample exclusions, and kernel attribution.
5. Compare conditions: shapes, operator implementations, worker/GPU layout, repetitions, synchronization, queue state, clocks, warm-up, settling, profiling overhead, and runtime/library versions. Identify confounded comparisons.
6. Report distributions: counts, median, quantiles, full range, signed/absolute discrepancies, and repeatability where available. Distinguish variability across shapes from repeated measurements of the same shape. Do not invent confidence intervals from one collection.
7. Audit preregistration and validation: what was specified, when that can be established, what failed, what changed after results, and which acceptance criteria apply to which dataset. A declared timestamp is not independent proof of preregistration.
8. Audit the figures, captions, notebook output, generated HTML, and narrative against the same verified quantities. Check hard-coded trace points, dataset mappings, clipping, axis labels, and whether cached data match raw inputs.
9. Review later regression, boundary, and Dataset B work. Separate source inspection, synthetic test success, live-run structural validation, timing validation, and serving-equivalence evidence.
10. Produce an investigation report and then a complete replacement story, with an English reply to Neria. Correct or omit unsupported claims instead of merely attaching a disclaimer to them.

The investigation report must include this claim ledger (add findings beyond the rows suggested here):

| Finding ID | Earlier claim / review suggestion | Evidence and calculation actually checked | Verdict: confirmed / revised / rejected / unresolved | Corrected statement and remaining limitation |
|---|---|---|---|---|

Print the report's substantive findings and key numerical results in your response; do not only link a file. Also save the detailed report and the complete story as separate Markdown artifacts, with paths or page links in your response. Include reproducible calculation snippets or exact commands and concise output in the report. Record checks you could not perform and why. Do not report a command as executed if you only inspected its code.

## Findings to independently recheck

### F01 — Monotonicity is not a correctness criterion

`K/story/evolution/build_evolution_notebook.py:256` says: “More tokens cannot mean less time: a bug in the measurement.” `STORY.md:264–268` says monotonicity was never a criterion. The final dataset itself reportedly contains downward steps:

- `attn_pre_proj`, TP1, 256 to 257 tokens: approximately **−15.83%**.
- `attn_post_proj`, TP8, 12288 to 12304 tokens: approximately **−42.00%**.

Recompute these. A decrease alone proves neither a measurement bug nor a kernel-selection explanation. Require trace/intervention evidence before assigning a cause. Suggested wording, subject to verification: “The curves motivated investigation; the diagnosis rests on controlled changes and trace evidence, not monotonicity alone.”

### F02 — Event durations, kernel durations, and ground truth

Inspect `linear_op_wrapper.py:291` and `:424`, the timer implementation, and `14_regressor_dataset_handoff.md:41`. The primary label is reported as a median event interval around an operator under a backlogged, settled protocol, in milliseconds. It is not automatically a sum of kernel durations or a production latency.

Establish exactly what rocprofv3 timestamps represent in these traces and how kernel durations are grouped into operators. Determine its independence at the instrumentation level and whether it is a valid reference for the same quantity. Profiling can change execution conditions; compare profiled/unprofiled evidence rather than assuming zero perturbation.

The two-column fix retains the legacy pass and adds a pass with changed queue/clock conditions. It is not simply subtraction of CPU overhead. An event/trace ratio near 1 supports agreement for tested cases; it does not establish equivalence of boundaries or universal correctness.

### F03 — Final validation is not uniformly GREEN

`14_regressor_dataset_handoff.md:109` discloses final-dataset T2 TP4 below its threshold. Prior recomputation:

`median(post_proj, TP4, tokens=4096) / median(post_proj, TP4, tokens=64) = 1.9724137433`, versus **2.0**.

Recheck the validator and actual acceptance thresholds. Do not call the failure a “flake” without evidence of repeatability under an identical protocol. Separate original preregistration from revised acceptance tests. “262/369 predictions inside bands” is a scorecard count, not an instrument accuracy percentage.

### F04 — Run index, embedding invocations, and variability

The notebook generator at `:300–314` labels plots as run indices 1..25. The implementation at `linear_op_impl.py:1088–1097` calls embedding twice per forward. The final CSV reportedly has 50 timed embedding samples plus six warm-up samples, versus 25 timed plus three warm-up samples for most operators.

After warm-up exclusion and normalization by each row median, the earlier reviewer found median embedding position profiles of roughly **1.0166 for first invocations** and **0.9699 for second invocations**. Each invocation sequence was much flatter than their alternation suggests. Verify sample order before interpreting this as progressive drift.

Median within-row relative MAD, computed across timed samples and summarized over shapes, was approximately:

| Operator | Reported range across TP, or TP1 value for embedding |
|---|---:|
| Pre-projection | 0.54–0.55% |
| Post-projection | 0.68–0.92% |
| Embedding, both invocation positions pooled | 2.47% |

Check the exact formula and populations. The 0.55% statement is not established for every operator. Endpoint clock probes do not establish a constant clock throughout an interval. Do not promote a speculative thermal/clock explanation into a measured cause.

### F05 — “Noise floor” is not established by the current regression calculation

Inspect `regressor_bench/RESULTS_baseline_v2.md:13–15` and `baseline_v2.py:148–152,198`. Jobs 21519 and 21539 reportedly use three versus eight settle forwards. Their mean test-shape discrepancy was **1.188% MAPE**.

The code divides pairwise MAPE by √2 and subtracts that result from model MAPE as `learnable_mape_above_floor`. Investigate whether the required statistical assumptions are stated and supported. Protocol changes can introduce systematic differences; MAPE does not generally decompose additively into noise and learnable error.

Unless stronger evidence exists, call this a **cross-collection discrepancy**, not an established irreducible noise floor. Specify identical-protocol independent recollections as a possible resolving check; do not launch them under this task.

### F06 — Presentation sample availability and experimental controls

- `presentation_11_10_26/README.md:35` says original per-repetition samples were not retained. The prior reviewer found sample columns in the original CSV. Check them; distinguish what a figure uses from what data exist.
- `build_evolution_notebook.py:271,288` calls pre/post-projection “unaffected controls” for the RoPE change. Unchanged operator code does not establish identical measurement conditions. Check repetitions, protocol, grid, and job changes.
- Original/final RoPE ratios mix a computation correction with timing changes. Do not present them as a pure implementation speedup.
- D3 job 21356 reportedly precedes D2 job 21361. Verify dates and do not label the D0–D5 conceptual sequence chronological without qualification.

### F07 — Reproduce the updated ratio tables, including variability

Dataset mapping to recheck:

| ID | Job | Meaning in updated figures |
|---|---|---|
| D0 | inherited | Older dataset, excluded from October 11 presentation |
| D1 | 21313 | Original recollection |
| D2 | 21361 | GC correction |
| D3 | 21356 | Backlog-knob collection, tokens >=1024 |
| D4 | 21483 | Fused RoPE / earlier run-position protocol |
| D5 | 21539 | Final settle-eight collection |

Raw starting paths:

- D1: `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv`.
- D2: `data/profiling_gcfix/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv`.
- D3/D4: resolve the original paths through `story/evolution/dataset_table.csv`, notebook configuration and presentation manifest; earlier review used temporary local source files. Cached summaries are a fallback with weaker provenance, not raw sample replacements.
- D5: `data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv`.

D5 was reported as 13,308 rows, 3,327 token counts per TP, MD5 `3635e4d305ae5cf84884e275d3ff37ef`, `active_steps=25`, `warmup_steps=3`, `settle_steps=8`, `backlog_kind=gemm_alloc`, `measurement_type=CUDA_EVENT`. Verify metadata and actual samples, including operator coverage and blank replicated-op fields at TP>1.

For each shared token count, ratio = D1 operator median / D5 operator median. Then summarize across token counts. The following are prior-review results, **not expected answers to force your calculations to match**:

| TP | Pre-projection median [min–max] | Post-projection median [min–max] |
|---|---:|---:|
| 1 | 1.381 [0.952–4.673] | 1.607 [0.957–3.205] |
| 2 | 1.725 [0.956–5.389] | 2.101 [0.933–3.556] |
| 4 | 2.359 [0.967–5.496] | 2.648 [0.972–4.744] |
| 8 | 2.873 [0.953–5.656] | 2.879 [0.962–5.503] |

Do not confuse the median of per-shape ratios with a ratio of pooled medians. Add quantiles and below-one counts. These are label differences across a nonuniform token grid, not workload-weighted serving speedups. The original/final RoPE median ratios were approximately 4.5–13.0 across TP; their interpretation is confounded by computation changes.

### F08 — Carry forward, but re-audit, the Chapter 7 findings

Starting points: `STORY.md:262–305`, numbered preregistration/results documents, historical scripts and raw runs, `tests/unit/test_linear_op_timing_columns.py:142–155`.

Earlier review reported:

- Job 21389 post-projection event-minus-rocprof gap reaches **10.7005 µs** at TP1/tokens6144, not uniformly +0.3…+4 µs. Its small-token legacy/backlogged ratio reaches **5.5366**. Verify which jobs each sentence and figure actually use.
- Job 21400 pre-projection event/kernel ratios: **0.9206–1.3032**, median **1.0117**, with **9/32 below one**. RoPE: **0.8934–1.4255**, median **1.0067**, with **15/32 below one**. These operator comparisons use a particular trace reference; do not combine them with a different trace condition.
- Some below-reference cells persist across multiple collections: reportedly nine pre-projection and twelve RoPE cells across three event runs. Repeated signed discrepancies do not by themselves identify the cause, but undermine an unsupported “just noise” explanation.
- The closure calculation uses **median forward span × count** instead of the sum of observed spans. For job 21400/tokens8/TP2,4,8, reported median-based closure **0.779/0.908/0.738** becomes approximately **0.982/0.979/0.982** using sums. Audit numerator, denominator, measured backlog subtraction, warm-up exclusion and actual repetition count before judging closure. Backpressure observations and this arithmetic issue are separate questions.
- Locked-clock results do not by themselves justify “the residual was clock.” Endpoint probes do not justify “constant clock throughout.” Look for remaining locked-clock discrepancies and profiling effects.
- Figure f14 reportedly uses job 21400 event points and a limited set of rounded/hard-coded trace values, despite nearby text discussing job 21389. Inspect the generator and disclose the true source selection.

Historical raw evidence was previously found under:

`/tmp/claude-1001/-home-dn-amd-playground/0de4f46f-80c9-420e-86e2-ee0923dbf5d3/scratchpad/`

Relevant names included `analyze_trace.py`, `scorecard.py`, `trace_metrics.json`, `val/trace/`, `val/b/`, `fixedclk/`, and `rf/`. These paths may be ephemeral. Locate equivalent archived evidence if missing. Never convert inability to reproduce into confirmation of either the chapter or this review.

### F09 — Regression success is label prediction, with important tails

Recompute from `regressor_bench/results/baseline_v2/test_predictions.csv`, inspect split construction, and confirm the separately reserved holdout was not consumed. The previous review found 499 test token values for each of 15 operator/TP targets and three methods (22,455 prediction rows):

| Method | Mean absolute percentage error | p95 absolute percentage error | Maximum absolute percentage error |
|---|---:|---:|---:|
| Random forest | 1.249% | 3.068% | 93.857% |
| XGBoost exact | 1.272% | 3.089% | 92.332% |
| Linear interpolation | 1.372% | 3.448% | 62.925% |

RF's worst row was reportedly pre-projection, TP8, one token: label ~0.01092 ms, prediction ~0.02117 ms. Investigate endpoint coverage and boundary cases. Do not present the average as a bound, or a random shape split as proof of extrapolation, kernel-time accuracy, or end-to-end simulator accuracy.

### F10 — Incorporate the operator-boundary investigation carefully

Audit `operator_boundary_investigation/REPORT.md:29–61` against its cited sources and current configuration. Findings to investigate include:

- The examined SGLang HIP path fuses RoPE and KV writes. Summing Dataset A's separate labels is a cost hypothesis, not measured equivalence to the fused kernel.
- Serving configuration changes kernel providers; the approved image alone does not determine them. Qualify claims by the actual configuration and code-path gates.
- Fused all-reduce plus norm can conflict with separately added communication terms.
- Embedding is profiled/trained but reportedly not included in the relevant execution-time composition; final norm/lm_head/sampler work is not modeled there.
- Harness concerns include aliased input/residual tensors, shared attention block tables, and a KV-write implementation different from serving.

That report explicitly distinguished source-level findings from live trace validation. Do not erase the distinction, or extend statements to unexamined serving configurations.

### F11 — Dataset B is a restricted structural pilot, not completed A/B timing validation

Use the latest review response and actual reports; early `IMPLEMENTATION_STATUS.md` and pilot notes may contain superseded blockers.

Read `dataset_b/evidence/pilot_runs/b-pilot-21798-dataset-a-scope-check.json`, `b-pilot-21798-extracted/failure.json`, and `dataset_b/REVIEW_INSTRUCTIONS.md:270–306` plus its later response-to-review section. Prior findings:

- Restricted Dataset A scope check: **3,659 in-scope kernels resolved**, comprising **2,261 measurement** and **1,398 warm-up** kernels; `structural_pass=true` for that restricted check.
- Clock states: **2,793 unknown_resolution**, **866 outside_calibration**. Do not describe these as accepted clock-qualified timing evidence.
- **2,150 out-of-scope kernels unresolved**; full extraction/check reports `structural_pass=false` with unresolved/ambiguous target attribution.
- The actual Dataset A versus Dataset B comparison had not started. Broader execution-mode/TP coverage, output equivalence and instrumentation-perturbation checks remained open.
- Later code fixes and synthetic tests do not retroactively validate an earlier live run. Identify which fixes have fresh live evidence.

Verify the current state before repeating any of these numbers or blockers.

### F12 — Evidence availability and publication provenance

The handoff document claims a companion `linear_op_kernel_only.csv` next to D5, but it was absent at that local path during review. Search for the actual artifact and verify its identity and sampling protocol. If it remains missing, state that the proposed cross-check could not be performed; do not imply the experiment never existed.

Resolve temporary paths, untracked inputs, notebook caches and figure provenance before publication. A published reader should be able to identify the exact supporting datasets and computations. Add a source manifest and durable references within the available repository/page workflow.

## Questions the final story must answer plainly

Answer Neria's four questions in English, using terminology established from the code:

1. Is the third quantity rocprofv3 kernel duration compared with legacy and backlogged event measurements?
2. For which precise quantity, if any, is it justified as ground truth? What makes it a reference or estimate instead?
3. When the backlogged event measurement is smaller, what evidence supports variability, a systematic difference, or another explanation? What cannot be determined?
4. What does agreement near one establish, and what does it leave unverified?

Include concrete values and traceable sources. Avoid calling H0/H1 hypotheses measurement methods without explaining the naming distinction.

## Write the complete replacement story after verification

Use the investigation to choose the narrative, not the reverse. A useful structure is:

1. Problem, model/hardware/software context, and the exact quantity needed by the simulator.
2. What the original profiler measured and the evidence for each diagnosed defect.
3. Interventions, changed measurement contracts, and what each experiment could isolate.
4. Preregistered predictions, failures, revised criteria, and limitations of trace agreement.
5. Final dataset definition, provenance, variability, remaining biases and failed checks.
6. Regression results including tails and scope of generalization.
7. Serving operator boundaries, provider/fusion differences, and Dataset B's actual validation status.
8. Established conclusions and specific unresolved questions with resolving checks.

Replace misleading terminology in text, tables, figures and captions consistently. Distinguish timing-method changes from computation changes and dataset changes. Keep historical results labeled with their historical conditions. Include an explicit correction record for materially changed claims from the old page.

A possible framing to evaluate, not copy unquestioningly:

> We identified and corrected several artifacts in Frontier's linear-operator profiling, recollected a documented dataset, and fitted regressors that reproduce its measured labels. These results support prediction of those labels under the specified protocol. Equivalence to production serving costs remains a separate validation question because operator boundaries, providers, fusion and execution conditions can differ.

Finish by printing: the investigation results, what this handoff got right or wrong, unresolved evidence gaps, links to the report and complete replacement story, and the short English reply to Neria. State exactly whether anything was published and whether the old page was changed. Do not claim a correction was applied to Confluence if you only wrote a draft.
