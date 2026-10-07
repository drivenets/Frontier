# Dataset B planning and evidence review

Reviewed 2026-10-04 against checkout `284130b` and the untracked Dataset B planning documents.

The previous agent produced a useful investigation and a detailed proposal, but **the proposal needs revision before implementation or a collection pilot**. No Dataset B collection software or measurements are present. The strongest work is the reconstruction of Dataset A and the identification of production scope mismatches. The largest verified omission is mixed attention: the plan says the canonical data lack mixed-model training data, but the file contains 2,880 true-mixed rows with complete decode-in-mixed features and timing targets. Graph attribution, multi-GPU identity, and validation also remain unresolved.

This report covers the problem, the necessary collection plan, the work completed, its strengths and weaknesses, and prioritized fixes. It reviews the work; it does not authorize or execute collection or change the existing plan.

## A Problem and motivation

Frontier predicts inference execution time using measurements of individual operators. Dataset A runs those operators largely in isolation with synthetic tensors. The question motivating Dataset B is whether those measurements represent the operator costs observed while SGLang serves requests through the complete Qwen3-30B-A3B model on MI355X.

The distinction matters. In a complete model, an operator follows other kernels, reads actual weights and KV-cache contents, encounters changing GPU clocks and cache state, and may execute through graph replay or concurrently with other work. Earlier investigations also showed that some apparent operator costs were properties of the measurement harness: host launch delays, event overhead, garbage collection, queue behavior, and clock ramping. A model can fit those numbers accurately while predicting the wrong physical quantity. The agreed [measurement contract](../10_measurement_contract.md) separates kernel device time from launch and gap overhead.

B should therefore be an occurrence-level reference dataset: each retained kernel execution has device timestamps, an unambiguous operator/forward/layer/GPU association, actual shape information, and clock and queue context. Scope summaries are derived views; the original occurrences remain available. B is intended to support later comparison with A. Regression fitting, A-versus-B conclusions, and end-to-end serving benchmarks are outside the collection task described in the [handoff](HANDOFF.md:9).

There are two limitations to state at the outset:

- **Production relevance and causal isolation are different claims.** A current serving image is a sensible reference, but changing kernel libraries, fusion, runtime flags, and tensor layouts alongside execution context means that a later A–B difference cannot be attributed to context alone. Collection must preserve these distinctions; a controlled bridge experiment can be considered in the later analysis phase.
- **This is not a complete model-cost dataset.** The inventory covers six linear/normalization/embedding scopes and three attention scopes. It intentionally excludes usable MoE measurements, collectives as standalone targets, the output head, and sampling. Those operations still affect the environment of the retained scopes.

## B What a dataset creation plan should contain

A good plan should specify what each record means, how its identity is established, what coverage is promised, and which evidence makes the collection acceptable.

| Required part | What the plan must establish |
|---|---|
| Scope and consumers | Exact A files and checksums; operator boundaries; consuming trainer/predictor paths; phase and batch regimes; features and filters per consumer. Distinguish operator names from the multiple models trained from them. |
| Production correspondence | A-to-SGLang mapping for each scope, including fusions, layer-0 exceptions, TP behavior, graph behavior, and cases that cannot be separated without changing execution. |
| Measurement definition | Device duration for each kernel; kernel-duration sum, elapsed scope span, and overlap-aware quantities separately; precise gap semantics and units. No implicit conversion of span into kernel-only cost. |
| Attribution and schema | Stable run, process, TP-rank, GPU, dispatch, forward, scope, and graph identities; per-sequence membership and lengths; real versus padded shapes; explicit missing/ambiguous states. |
| Instrument validation | Demonstrated eager and graph attribution on the pinned image; bounded marker and tracer perturbation; clock timebase alignment; capture-window and trace-loss checks; a second tracer on representative cases. |
| Workload and coverage | A generated request/configuration manifest covering small shapes, transition boundaries, long and chunked prefill, decode contexts and batch sizes, mixed attention, TP1 and a shared TP>1, graphs on/off. Actual observed shapes must be reconciled with requested shapes. |
| Reproducibility | Image digest, source and patch hashes, model/tokenizer revision, relevant environment and effective flags, seeds/input IDs, node/GPU identities, warm-up procedure, raw traces, telemetry, and extraction version. |
| Acceptance and delivery | Nonempty results; all expected scope invocations and ranks accounted for; no silently lost or duplicated target dispatches; explicit exclusions; validated timing/clock joins; schema, run record, checker output, and confounder note. |

The work should progress through four stages:

1. Correct the inventory and freeze the measurement/schema contract, resolving actual scope and production-configuration choices.
2. Build the smallest collector and test the uncertain mechanisms: eager attribution, graph replay, scheduler-to-forward mapping, clock joins, and multi-rank collection.
3. Run the requested TP1 pilot with both tracers and a small TP>1 validation before the full TP sweep. Include a chunked case and a mixed case or document an approved limitation.
4. Generate and run the full coverage manifest, validate every configuration, and publish immutable raw and extracted artifacts with the run records.

The pilot should answer whether the instrument works, not merely whether a server runs and emits a CSV. Acceptance should be based on observed forwards and dispatches reconciled against independent request/scheduler records, rather than counts derived from the same template used to label the data.

## C What the previous agent completed

The local Dataset B directory initially contained only [HANDOFF.md](HANDOFF.md) and [00_inventory_and_plan.md](00_inventory_and_plan.md). The handoff explicitly records planning-only status: no marker patch, extractor, driver, checker, pilot, sweep, B schema, or B run record exists. Local inspection agrees with that status.

According to the handoff, the agent read the A investigation record and profiler/consumer code, delegated inventory work across three agents, inspected five container images without running the model, checked historical cluster availability and model paths, computed shape transitions from A, and wrote the inventory and collection proposal. The three subagent reports and raw image inspection dumps are not retained in this directory; the handoff says the image dumps were lost. I therefore treat the image-internal findings as reported evidence requiring reproduction against a pinned image, not as independently verified facts from this review.

I independently checked these local artifacts:

| Check | Result |
|---|---|
| Local settle-8 A-linear mirror | MD5 `3635e4d305ae5cf84884e275d3ff37ef`, matching the handoff; 13,308 rows and 150 columns. |
| Linear scope coverage | 3,327 rows at each TP 1/2/4/8 for pre-projection, post-projection, and RoPE; 3,327 TP1 rows each for the two norms and embedding, with no non-null TP>1 labels for those three. |
| Claimed post-projection transition at 8192→8208 tokens | +30.07% at TP1 and +80.67% at TP4, confirming the rounded claims. |
| Canonical attention file | 25,032 rows: 22,152 standard and **2,880 true-mixed**. All mixed rows have non-null prefill, decode, and cache-write median targets. |
| Decode-in-mixed feature completeness | All 2,880 mixed rows contain every feature used by the cited predictor branch, plus the decode target. |
| Consumer behavior | The new regressor explicitly consumes six linear scopes. Existing attention consumer code includes standard and mixed training paths; cache-write features differ between consumers. |

The shared `/opt/shared` dataset path was not mounted here; the checksum-matching local mirror was used for linear data. No cluster state was refreshed and no GPU experiment was run. This is a document, code, data, and arithmetic review, not runtime validation of the proposed instrument.

## D Review of the work

### What was good

**The agent did substantial useful archaeology.** It selected the settled linear dataset instead of the obsolete root CSV, distinguished linear and attention measurement regimes, traced QK normalization into the pre-projection boundary, noticed the duplicate embedding invocation in A, and accounted for TP1-only replicated-op labels. The checksum, coverage, and transition checks above support that work.

**The production mapping asks the right questions.** Fused RoPE/cache writes, layer-0 norm behavior, all-reduce fusion, embedding sharding, and server-wide attention partition counts can invalidate apparently direct comparisons. Surfacing them before instrumentation is valuable. Reporting a scope mismatch and pausing the relevant execution was appropriate under the original task's explicit instructions.

**The proposed outputs preserve useful evidence.** Per-kernel records, raw traces, clock/gap context, a second tracer, reviewable patches, and a staged pilot are justified. None of these should be discarded merely to simplify the project. Choosing shape neighborhoods from measured discontinuities is also stronger than an arbitrary powers-of-two grid.

### What is overkill or premature

The five-image investigation was defensible, but further image exploration has diminishing value. Pin the selected production image and preserve its evidence. The immediate unknown is whether the collector can reliably identify executions on that image.

The seven-question universal gate is too broad. Image choice, inseparable scopes, and deviations from actual production fusion policy require substantive decisions. TP4 selection, recording the unmodified partition count, and labeling legacy A-attention limitations can be proposed defaults under the existing task. Q7 should not request approval for a comparison that is explicitly deferred. The original stop-and-ask constraints still apply to genuine boundary/image/key problems; they do not imply that all schema design must wait.

The full 155-length cross-tracer pass, duplicate full CSV and Parquet exports, and optional extra TP, clock-lock, or unfused sweeps should be staged after the pilot. A stratified tracer cross-check is a reasonable starting point. Keep every occurrence in selected collection windows; reduce unnecessary configurations before reducing record fidelity. Store repeated image/version metadata once in a manifest, referenced by row IDs, and provide CSV samples if full duplication has no identified consumer.

The plan's exact marker-cost and wall-time estimates are hypotheses until measured. Avoid designing the whole sweep around them.

### Findings that require correction

**1. The inventory incorrectly excludes mixed attention coverage.**

[Plan §1.1](00_inventory_and_plan.md:32) says `attn_prefill_mixed` and `attn_decode_in_mixed` have no canonical data. The canonical CSV contains 2,880 `mode=true_mixed` rows. Its [run record](../../../data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/2026-09-15_1142_attention_16k/RUN.md) explicitly records `--enable_true_mixed`. The [predictor](../../../frontier/execution_time_predictor/sklearn_execution_time_predictor.py:3273) contains both mixed training paths, including a decode branch whose required features are all populated in those rows.

These are additional regimes and consumers of the existing scopes, not automatically two new operator boundaries. Correct the coverage table, record full batch composition, and inspect how the chosen SGLang backend handles mixed work. A single `attn_prefill XOR attn_decode` rule cannot express A's mixed invocation. If production combines work differently, that is another mapping mismatch to resolve, not grounds for claiming the A data are absent.

The cache-write feature inventory also conflates consumers: [profiling_mapping.py](../../../frontier/attention/profiling_mapping.py:251) specifies `num_tokens` for one predictor path and `(total_tokens, kv_cache_size, batch_size)` for the shared path. List both explicitly.

**2. Graph replay attribution is an unproven mechanism.**

[Plan §4.B](00_inventory_and_plan.md:109) proposes assigning graph kernels from eager kernel-name templates. Repeated names across 48 layers, graph-specific execution, and the plan's own alternate-stream example make name/order matching insufficient to establish unique identity. Moreover, the graph-enabled run does not specify how its required eager decode templates are obtained.

Use capture-derived identities or graph-node/dispatch information where the pinned tools support it, and prove replay correspondence on a small trace. Require unique assignments and retain the mapping evidence; “a template matched” is not sufficient. Current [AMD documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hip-graph-attribution) describes graph attribution fields and conditions, and notes that direct CSV omits those fields. That documentation is newer than the plan's SDK 1.1.0 and does **not** establish availability in the proposed image. Current [SGLang documentation](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx) also describes capture profiling; its availability must likewise be checked in the actual image.

**3. TP>1 records lack essential identities.**

The proposed rows include `tp` and `queue_id`, but omit TP rank, stable GPU identity, process identity, and retained dispatch/correlation identity. Scope rows keyed only by `(forward, layer, scope)` would collide across ranks or leave rank selection implicit. Clocks could be joined to the wrong GPU.

Specify whether every rank is collected; collecting all ranks is the appropriate default for the proposed reference. Key records by run/process/rank/device/dispatch and scope invocation. Preserve native IDs and a rank-to-device map, verify all worker processes are traced, and make row-count and telemetry checks rank-aware. If only one rank is retained, explicitly narrow the claim and justify that choice.

**4. Proposed marker placement can include excluded work.**

The serving-path table places all-reduces in `prepare_attn`/`prepare_mlp`, but §4.A proposes marking those enclosing functions as normalization scopes. Even with fusion off, the scope can therefore include communication that A's norm boundary excludes. Likewise TP>1 embedding includes masking/sharding/communication absent from A's TP1 gather.

Place markers at the exact operation calls, or explicitly represent a composite scope and its compatibility status. Record the layer-0 norm-only exception. For fused RoPE/cache write, retain one physical occurrence with a mapping to two A scopes; do not duplicate its duration into two operator rows. Comparing it with a sum of A labels is a later hypothesis, not a measured split.

**5. The checker contains contradictory acceptance rules.**

[Plan §4.F](00_inventory_and_plan.md:132) rejects every unclaimed kernel within a forward and every kernel outside a forward during the window. Yet the inventory excludes MoE, collectives, output-head and sampling targets. It also requires identical kernel names/counts across a grouping that excludes shape, even though the grid targets shape-dependent kernel transitions. Its every-scope/every-forward rule conflicts with fusion and phase-specific applicability.

Replace these rules with explicit accounting: target dispatches assigned exactly once; known out-of-scope work counted separately; ambiguous or missing target assignments rejected; allowed variants conditioned on actual shape, phase, graph, layer class, and rank. Do not export excluded work as extra B operator targets. Compare raw and retained counts with reason-coded exclusions, and derive expected scope invocations from observed execution metadata.

**6. Kernel-only cost, elapsed span, and gaps need separate definitions.**

The plan defines `scope_dur_us = last_end - first_begin`. That includes intra-scope idle time and is not kernel-only time. Across concurrent streams, kernel-duration sums and elapsed span also differ. For example, two 10 µs kernels running concurrently have a 20 µs duration sum and a 10 µs span.

Name and retain `kernel_sum_us` and `scope_span_us` separately; where needed derive the union of kernel intervals to describe busy time. Define scope boundaries using timestamp extrema, not arbitrary row order. A same-queue gap describes that queue's interval, not device-wide idle or proven CPU launch overhead. Another queue, a copy, or a dependency can explain it. Compute predecessor context before filtering out non-target kernels; distinguish the first observed predecessor from a missing trace boundary. Do not impose an invalid span-equals-sum-plus-gaps identity across queues.

**7. Instrument perturbation and clock resolution are not yet controlled.**

The plan allows roughly 16 marker boundaries per layer at an estimated 1–3 µs each. Its own estimates imply roughly 0.77–2.30 ms of added host work across 48 layers per eager forward. This is not a measured overhead, but it is large enough to require a control when queue gaps are part of the result. A scheduler JSONL flush every 500 steps can also alter the next step even though it occurs outside model code.

Use the marker backend needed for each tracing pass, precompute static labels, buffer metadata outside measured windows where feasible, and measure marker/tracer effects on a small representative subset. Check marked/unmarked output agreement as well as execution changes. Record the exact tracing window and clean trace drain; detect dropped records.

Clock handling needs an actual sampler design: sensor/API, GPU mapping, update resolution, timestamp clock, read latency, alignment calibration, and missing/stale policy. A nearest sample up to 20 ms old is context for a microsecond kernel, not proof of its exact instantaneous clock. Preserve sample age and uncertainty. Pilot the join rather than turning an arbitrary sampling threshold into a guarantee of validity.

**8. Requested concurrency is being mistaken for actual execution shape.**

D3 assumes uniform batches of size C and prefill `num_tokens=C×L`. At C=64 and L=8192, that is 524,288 input tokens, but the proposed per-forward prefill budget is 16,384: at least 32 prefill forwards are required in aggregate if that cap is respected, not the estimated 1–4. Independently arriving requests may be admitted at different steps, have different decode contexts, and trigger graph padding despite capture sizes matching the intended concurrency.

Record actual per-forward sequence membership, real/padded sizes, and per-sequence lengths. Do not replace a nonuniform attention shape with a single mean KV length and claim equivalence to A's uniform batch. Validate coverage from observations; use synchronized cohorts where useful, without assuming the scheduler preserves them.

Additional arithmetic needs repair: D1 enumerates 155 lengths; D2's listed prompt multipliers yield **186 prefill forwards per TP across three repeats**, rather than approximately 70, assuming exact chunking. Under the usual convention where prefill produces the first generated token, D1 has 16 model forwards per request for G=16, not 17; verify that convention on this image. All-rank TP4 output estimates also need the rank factor. Generate counts from a manifest and replace estimates after the pilot.

**9. Coverage and scientific claims need qualification.**

The proposed grid is a useful sample, not complete replication of A: it omits TP2/8, block-16 attention, most batch sizes and chunk sizes, and currently mixed attention. Some exclusions are justified by the task and production path; put them in an explicit coverage matrix rather than claiming complete coverage of A's key space.

Random token IDs are reproducible synthetic inputs through a real model, but their MoE routing and preceding cache/memory activity need not represent natural prompts. Retain seeds and token IDs and label that workload assumption; a small natural-text sensitivity subset is useful if production representativeness is a deliverable.

Finally, flat small-shape A-attention curves and its legacy loop justify concern about host contamination, but do not by themselves prove the precise device/host decomposition of every row. The plan sometimes states this more strongly than the handoff. Its prediction also conflates a single QKV GEMM with A's GEMM-plus-norm scope and uses inconsistent event-overhead subtractions. Rewrite the pre-pilot prediction by compatible scope and regime, with unverified values labeled hypotheses. Cross-tracer agreement alone does not eliminate shared instrumentation bias.

## E What needs to be fixed and in what order

| Priority | Required change | Evidence that it is complete |
|---|---|---|
| Before implementation | Correct mixed coverage and consumer-specific keys; define supported/excluded A cells. | Inventory reconciles with CSV counts and exact consumer paths; no false claim that mixed data are absent. |
| Before implementation | Resolve physical scope mappings and production baseline, including fused operations and TP norms/embedding. | Pinned configuration plus explicit A-to-B scope and variant mapping; no duplicate attribution of fused work. |
| Before pilot collection | Specify rank/device/dispatch identities, graph attribution, request-to-forward joins, timing quantities, and telemetry contract. | Schema draft and small trace fixtures demonstrate unique joins; unresolved mappings fail explicitly. |
| Before accepting the pilot | Replace contradictory checker rules; add trace-loss, output-equivalence, and perturbation checks. | Positive and deliberately damaged fixtures exercise missing rank, duplicate dispatch, lost marker, mixed mode, and ambiguous graph attribution. Pilot results show what passed and what remains uncertain. |
| Before the full sweep | Generate the workload/count manifest and validate actual batching, chunking, padding, warm-up, and storage cost. | Pilot-derived resource estimate and an observed coverage report, including a small TP>1 validation. |
| Before release | Preserve image/source evidence, raw traces, telemetry, patch, hashes, run records, schema, and confounder note. | A fresh extraction reproduces counts and validation results from immutable inputs. |

Keep the original inventory work as the starting point. The next deliverable should be a corrected plan and schema followed by a narrowly targeted instrumentation pilot. A larger sweep will become useful once attribution and measurement semantics are demonstrated; it cannot compensate for missing regimes or ambiguous labels.

The present review adds only this report. It leaves Dataset A, the existing Dataset B plan, simulator code, and cluster state unchanged.
