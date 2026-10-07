# Dataset B remediation plan

> Current implementation task: [02_IMPLEMENTATION_HANDOFF.md](02_IMPLEMENTATION_HANDOFF.md).
> Image choice and the old universal question gate below are superseded. See the corrected inventory and `IMPLEMENTATION_STATUS.md` for current work. This file retains historical context.

Written 2026-10-04. Status: proposed work, not executed.

This plan addresses the findings in [REVIEW_REPORT.md](REVIEW_REPORT.md). The immediate objective is to repair the existing inventory and collection design. The subsequent objective is to implement and validate that design before collecting Dataset B. The previous work contains useful evidence and should be revised rather than discarded.

The deliverable remains kernel measurements of the inventoried operators inside a complete Qwen3-30B-A3B serving run on MI355X. Preserve individual occurrences, actual shapes, clock context, and provenance. A-versus-B analysis, fitting regressors, modifying Dataset A, and expanding into standalone MoE or collective-cost profiling remain outside this work.

## Execution order

| Stage | Main deliverable | Exit condition |
|---|---|---|
| 1. Repair the existing documents | Corrected inventory, coverage, and decision list | Every review finding has a specified correction; inventory agrees with data and consumer code. |
| 2. Define the measurement contract | Pinned configuration, scope mapping, and `B_SCHEMA.md` | Every required field has a source; unresolved scope choices are explicit; no ambiguous timing definitions. |
| 3. Implement the minimum collector | Marker patch, driver, extractor, clock sampler, and checker | Local validation fixtures pass, including cases that must fail. |
| 4. Validate a small pilot | TP1 evidence with both tracers and a TP4 validation | Attribution, coverage, correctness, telemetry, and instrumentation limits are demonstrated. |
| 5. Collect and package B | Validated data, raw evidence, and run records | Frozen coverage is met and extraction can be reproduced from retained inputs. |

Stages 1 and the independent schema work in Stage 2 can proceed without GPU runs. Resolve image choice, inseparable scope mappings, and any production-policy conflict before the affected instrumentation or live collection. Present the pilot results before starting the full sweep, as required by the original task. Do not treat a request to create this plan as authorization to execute the sweep.

## Stage 1 Repair the inventory and planning record

Revise [00_inventory_and_plan.md](00_inventory_and_plan.md) in place when this plan is implemented. Update [HANDOFF.md](HANDOFF.md) to point to the corrected design and distinguish completed work from proposals. Keep the review as the record of findings; avoid creating another competing collection specification.

1. **Correct mixed attention coverage.** Record the verified 25,032 attention rows: 22,152 standard and 2,880 true-mixed. Include the mixed consumers of `attn_prefill` and `attn_decode`; these are regimes of existing operators, not two automatically distinct scopes. Remove the claim that mixed training data are absent.
2. **Separate consumer contracts.** For each consumer, list its source file, filters, derived features, label, and applicable regimes. Include both cache-write feature contracts: `num_tokens` for one path and `(total_tokens, kv_cache_size, batch_size)` for the shared path. Preserve the new linear regressor's six scopes and TP coverage.
3. **Make coverage explicit.** State that the proposed primary release uses TP1 and TP4, page size 1, standard and chunked prefill, standard and mixed attention, and eager/graph decode where supported. List excluded A cells, including TP2/8 and block size 16, with reasons. Distinguish promised coverage from exploratory cases.
4. **Correct counts and estimates.** D1 has 155 listed lengths. D2 implies 186 prefill forwards per TP over three repeats under exact chunking. D3's largest cohort needs at least 32 prefill forwards under a 16,384-token cap. Replace hand-written counts with counts generated from a configuration manifest; verify generated-token/forward semantics in the pilot. Include all collected ranks in storage estimates.
5. **Qualify unsupported claims.** Describe A-attention host contamination as a concern supported by the legacy loop and observed curves, not an established decomposition of every row. Rewrite predictions using compatible whole scopes; remove inconsistent event-overhead subtraction and any implied causal conclusion from changed kernel libraries.
6. **Restore an evidence trail.** Retain a small repeatable inventory audit and its output: checksums, row counts by regime/TP/page size, non-null targets, consumer feature coverage, and the transition values cited in the plan. Reuse the verified local linear mirror when the shared store is unavailable.

Completion check: the corrected inventory reproduces the review's counts and explains every supported regime. The handoff no longer incorrectly requires answers to all seven questions before any useful work can proceed.

## Stage 2 Define the configuration and measurement contract

### Resolve substantive choices and use defaults for routine ones

These are recommendations for the corrected plan, not decisions already approved by the user.

| Topic | Proposed treatment |
|---|---|
| Serving image | Retain the previously recommended DriveNets image as the candidate. Reconfirm its suitability, record its immutable digest, and preserve the relevant source inspection output. Resolve the original image-choice question before dependent live work. |
| Fused RoPE and cache write | Recommend keeping production fusion and recording one physical scope mapped to both A operators. Resolve the boundary exception explicitly. An unfused control is optional later work. |
| Mixed attention mapping | Inspect the actual backend. If it combines work that A measures separately, document and resolve that mismatch. Do not silently label combined work as separate prefill/decode costs. |
| All-reduce fusion and TP embedding | Establish the effective serving recipe. Keep actual production behavior as the proposed reference; label composites and sharded variants. Do not disable fusion merely to make attribution easier without recording the changed scope/configuration. |
| A's runtime hygiene | Retain `DEBUG_CLR_MAX_BATCH_SIZE=1000000` as the proposed main collection setting to honor the original hygiene requirement. Label this as a production deviation if applicable. A small default-setting control is optional; changing the main requirement needs an explicit decision. |
| TP selection | Use TP1 and TP4, collecting every rank. No separate question is needed merely to choose TP4 within the requested TP>1 coverage. |
| Decode partition count | Preserve the server's behavior and record the effective value; do not alter it to imitate A. |
| Legacy A-attention | Preserve its identity and limitations as provenance. No approval for an A-versus-B comparison is needed because that analysis is deferred. |

Bundle genuinely unresolved choices into one concrete configuration proposal after inspection. Do not re-request decisions already answered in the session. Missing graph support or an unrecoverable shape key must be reported as a limitation requiring resolution, rather than hidden through guessed labels.

### Specify scope boundaries and record identity

Create `B_SCHEMA.md` before collecting B. It should define logical records and joins, without requiring a separate file for every record type:

- **Run metadata:** image digest; model/tokenizer revision; patch and source hashes; tool versions; effective flags and relevant environment; node; rank-to-process/GPU mapping; input seed; trace windows.
- **Forward and request membership:** rank-local forward identity and validated cross-rank correspondence; actual phase/batch composition; request IDs; per-sequence prefix, query, and attended lengths; real and padded execution sizes; graph capture/replay identity when applicable.
- **Kernel occurrences:** stable occurrence key using run, process instance, device, and native dispatch identity or a documented equivalent; native correlation/queue/stream IDs; scope invocation and layer; raw timestamps and duration; attribution method; predecessor and clock-sample references.
- **Scope summaries:** kernel count, kernel-duration sum, elapsed span, variant, and compatibility with A. Summaries are derived from retained occurrences.
- **Telemetry:** device identity, sensor values, sampling timestamps, clock domain, read latency, update resolution, join age, and validity status.

Keep physical scopes separate from the list of A operators to which they correspond. Never duplicate a fused kernel duration into multiple operator costs. Mark only normalization calls when communication is separable; an enclosing communicator function is not automatically a norm boundary. Label the first-layer norm and TP embedding differences explicitly.

Use GPU kernel timestamps as the timing instrument; no added HIP/CUDA event pairs. Define:

- `kernel_dur_us = (end_ns - begin_ns) / 1000`.
- `kernel_sum_us = sum(kernel_dur_us)` over assigned occurrences.
- `scope_span_us = (max(end_ns) - min(begin_ns)) / 1000`.
- Same-queue predecessor interval using the complete raw trace before target filtering. Define missing predecessors and trace-boundary censoring explicitly. Do not interpret this interval as device-wide idle or proven host overhead.

Account for concurrent streams explicitly; an interval union can describe busy time when useful. Preserve raw timestamps even when a derived gap requires investigation. Do not force a single-stream closure formula onto overlapping work.

Completion check: every field has a specified source and missing-value policy; every scope has applicability rules by phase, layer, TP, and fusion. No required row can acquire a GPU, shape, scope, or clock by an undocumented guess.

## Stage 3 Implement the minimum collection path

Keep implementation under `dataset_b/`, with the external SGLang modification delivered as a patch against the pinned source. Use the existing proposed script names. Reuse existing profiling utilities where they fit; avoid a general-purpose instrumentation framework.

| Component | Required behavior |
|---|---|
| `sglang_b_markers.patch` | Enable explicitly; mark exact scope calls and forwards; use the appropriate annotation backend for each tracer; precompute static labels. Capture request/shape metadata with minimal buffering work, with no periodic disk flush in the measured path. |
| `drive_b.py` | Execute a saved request/configuration manifest; preserve exact input IDs and seeds; control submissions; record actual completions and errors; distinguish warm-up from collection. Requested concurrency is a target, not a batch-size label. |
| Clock sampler | Sample each participating GPU outside the model path; retain native timestamp information; calibrate the join and report freshness/uncertainty. Do not add clock-probe kernels or force the clock in the main run. |
| `extract_b.py` | Normalize raw tracer records, preserve IDs, map occurrences uniquely, compute predecessor context before filtering, and produce per-kernel data plus derived scope summaries. Retain extraction version and exclusion accounting. |
| `check_b.py` | Reconcile raw records, scope invocations, request logs, all ranks, and promised coverage. Report structural failures separately from measurement-quality limitations; fail on empty inputs and silent losses. |

Develop graph attribution before scaling the driver. Inspect the pinned tracer's actual outputs and supported capture information. Prefer demonstrated capture/graph-node correspondence. Kernel names alone, or an eager sequence assumed equivalent to graph replay, cannot establish identity across repeated layers. If the tools lack sufficient identity, investigate a supported capture-mapping mechanism and prove uniqueness; do not upgrade the stack or remove graph coverage silently.

Use small fixtures to exercise the failures that matter: repeated kernel names across layers, cross-stream execution, mixed attention, one fused occurrence mapped to two A scopes, a missing rank, a duplicate dispatch, a missing marker, an unknown target attribution, stale/wrong-device telemetry, and an empty or truncated trace. Known excluded model work must be accounted for without being exported as new operator targets.

Completion check: valid fixtures pass and damaged fixtures fail for the intended reason. Kernel counts may vary with legitimate shape/backend changes; unexpected target losses must still fail. Passing a name template is not itself evidence of completeness.

## Stage 4 Validate the collector with a small pilot

Use one pilot in successive steps, retaining each useful trace as a validation fixture. Avoid launching separate broad diagnostic campaigns.

1. **Establish eager TP1 attribution.** Start with short, medium, and long prompts. Confirm exact operation boundaries, sequence-length conventions, output equivalence with markers disabled, and no missing/duplicated target occurrences.
2. **Establish graph replay attribution.** Validate batch size 1 and a batched case. Exercise one non-capture-size batch to validate padded-versus-real metadata. Verify scope/layer identity across replay using evidence beyond kernel names.
3. **Exercise chunked and mixed work.** Use a prompt requiring several chunks and deliberately introduce prefill requests while other requests decode. Check the actual scheduler/attention regime. If the chosen production path never forms the required mixed regime, record that evidence and resolve the coverage mismatch instead of forcing a misleading label.
4. **Cross-check with Kineto.** Use the same small representative TP1 cases under separate tracer passes, covering eager and graph decode and short/long scopes. Compare matched shapes, variants, kernel identities, and clock conditions. A full 155-length duplicate sweep is not required initially.
5. **Check instrumentation effects and clocks.** On a small matched subset, compare trace-only and trace-plus-marker execution and inspect buffering effects. Quantify clock alignment, sensor cadence, stale joins, and trace completeness. Do not infer absence of perturbation solely from agreement between two tracers.
6. **Validate TP4 before scaling.** Run a short subset that confirms all four ranks, rank-specific telemetry, communication boundaries, and graph identity where used.

The existing approximately ten-prompt TP1 pilot is an appropriate starting budget. Add only the cohorts needed to cover chunking, mixing, padding, and ranks. Keep each occurrence in the measured windows. Use independent repeated blocks to distinguish reproducible differences from one-run variation; layers within one forward are not independent run replicates.

Write the pilot results into `RUN.md`: attribution examples, checker output, achieved coverage, tracer differences in absolute and relative units, output checks, telemetry quality, and measured storage/runtime. Freeze numeric freshness and perturbation acceptance limits after an initial calibration subset and before evaluating the remaining pilot cases; justify them against sensor resolution and the intended measurements. Do not import a universal 2% tracer threshold or relax a threshold merely because validation failed.

Pilot exit conditions are exact structural accounting, unique target attribution, complete rank/request mapping, supported scope coverage, demonstrated clock joins, and documented measurement-quality limits. Unexplained scope or kernel-identity disagreement blocks the affected configuration. A valid partial pilot must remain labeled partial; it cannot certify the full dataset.

## Stage 5 Generate the sweep and package the dataset

After presenting the pilot evidence, freeze a machine-readable coverage manifest. Retain the useful original D1 transition neighborhoods, D2 chunk sizes, and D3 batch/context axis, but generate counts and classify observed execution shapes. Add mixed workloads from the corrected inventory. Use a saved randomized or interleaved workload order to reduce confounding between shape and clock drift.

The driver must reconcile each requested cell with actual forwards: measured, legitimately inapplicable, failed, or still missing. Do not equate C concurrent clients with batch size C. Derive forward counts from actual scheduler and completion records, with independent request accounting, rather than requiring approximate arithmetic to match execution exactly.

Collect one configuration at a time, validate it, and resume using immutable run IDs and manifest checksums. Preserve failed attempts and reasons; recollect only missing or invalid cells. Recalculate runtime and storage from pilot measurements before allocation. Prefer compressed Parquet plus raw traces and small CSV inspection samples unless a consumer requires full CSV exports.

Release with the reviewable patch, input/configuration manifest, `B_SCHEMA.md`, per-run `RUN.md`, telemetry, checksums, checker results, extraction version, and a short confounder note. Verify that a fresh extraction reproduces occurrence identities, values, and counts from retained raw inputs. Larger files follow the existing shared-store convention; repository records point to immutable locations.

Completion means the agreed coverage is met with auditable measurements. It does not mean B agrees with A or proves a particular explanation of their differences.

## Work deferred until it has a clear need

Defer additional TP2/TP8 sweeps, locked-clock runs, unfused control sweeps, broad image comparisons, natural-text sensitivity sweeps, and full cross-tracer duplication. These may answer later questions, but do not replace the required correctness work. The first implementation milestone is the corrected inventory and schema, followed by demonstrated attribution on a small trace.

## Review finding closure

| Review finding | Addressed by |
|---|---|
| Mixed attention and consumer keys | Stages 1–2; observed mixed coverage in Stage 4 |
| Graph attribution | Schema in Stage 2; implementation in Stage 3; replay evidence in Stage 4 |
| Missing TP/device identity | Stages 2–3; all-rank validation in Stage 4 |
| Incorrect scope boundaries | Configuration/mapping in Stage 2; exact markers and pilot traces |
| Contradictory checker | Stage 3 fixtures and Stage 4 independent reconciliation |
| Kernel sum versus elapsed span | Stage 2 definitions and Stage 3 overlap fixtures |
| Instrument and telemetry limitations | Stages 2–4 |
| Incorrect workload/count assumptions | Stage 1 corrections and Stage 5 generated/observed accounting |
| Overstated coverage and scientific claims | Stage 1 revised claims and Stage 5 coverage/confounder records |
