# Investigate Frontier operator boundaries for SGLang on AMD

You are conducting an independent research and code-review task alongside a separate Dataset B collection workstream. Your task is to determine whether the operator definitions measured in Dataset A are appropriate for modeling Qwen3-30B-A3B inference in SGLang on AMD MI355X, and recommend better definitions where evidence warrants them.

## Problem and user intent

We inherited Frontier because our engineers considered it the best starting point. The user's understanding is that its original operator partition was designed around vLLM on NVIDIA, and was not systematically reconsidered when moving to SGLang on AMD. Verify the relevant history rather than assuming every current boundary is inherited unchanged.

Dataset A already measures a set of Frontier-defined operator scopes. Dataset B is being planned to measure work inside complete SGLang model executions. Discussion of B exposed an architectural question: if the actual serving backend fuses two A operators into one kernel, are the two independent costs still useful prediction targets? Conversely, if an A operator encompasses several kernels, should it remain a single useful modeling unit?

Do not assume either that Frontier's current boundaries must be preserved or that every GPU kernel should become an operator. Reconstruct the reasoning behind the original partition, identify which principles still apply, and use those principles to propose a partition suited to SGLang/AMD. A recommendation to keep an existing operator is as valid as a recommendation to change it, when supported by evidence.

Answer this central question:

**For each operator measured in A, what is the most defensible unit of measurement and prediction for our target serving stack, and how would Frontier consume that cost without omissions or double counting?**

This investigation is allowed to challenge the earlier Dataset B assumption of identical A/B scope boundaries. Earlier handoff instructions to stop when encountering a mismatch governed collection; discovering and evaluating mismatches is the purpose of this separate task. Recommend changes without implementing them.

## Workspace and target stack

Repository: `/home/dn/Frontier-qwen3-profiling`. Read its `AGENTS.md` and any applicable nested instructions.

Primary target:

- Qwen3-30B-A3B, BF16, AMD MI355X, SGLang with the ROCm/AITER path.
- User-approved image: `dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04`. The tag is approved; its immutable digest and actual contents still need verification. Historical notes report SGLang 0.5.18; verify it.
- Assess TP1 and TP4 first. Explain relevant consequences for TP2/TP8 when the existing A coverage or replicated KV heads changes the conclusion.
- Cover prefill, chunked prefill, decode, mixed prefill/decode batches, and eager versus graph execution where applicable.
- Determine effective fusion/attention/runtime flags from source and available launch evidence. There is no verified exact Qwen3 serving recipe in the current handoff. Do not present defaults or a recipe for another model as confirmed production configuration.

Available neighboring checkouts may include `/home/dn/Frontier-upstream` and `/home/dn/Frontier-drivenets`. Use them read-only if useful for provenance; verify their commits and relationship before treating either as original upstream.

## Starting evidence

Read these local documents first, treating their claims as evidence to verify rather than authority over this investigation:

1. `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/REVIEW_REPORT.md`
2. `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/HANDOFF.md`
3. `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/00_inventory_and_plan.md`
4. `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/14_regressor_dataset_handoff.md`
5. `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/10_measurement_contract.md`
6. `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/README.md` and the relevant collection `RUN.md` files.

Use the numbered investigation documents and `story/STORY.md` when needed to resolve timing artifacts or implementation changes. Do not spend the entire task rereading unrelated investigations.

Dataset A inputs:

- A-linear local mirror: `data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv`; expected MD5 `3635e4d305ae5cf84884e275d3ff37ef`; 13,308 rows. Shared-store counterpart: `/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv`.
- A-attention: `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/attention_combined.csv`; 25,032 rows.
- Do not use the obsolete root linear CSV as the corrected A-linear reference. Do not treat the legacy MoE CSV as usable current data.

Known corrections and limitations:

- A-attention has **2,880 true-mixed rows**, including prefill/decode targets; the initial B inventory incorrectly says mixed-consumer data are absent. The other 22,152 rows are standard rows.
- A-linear and A-attention came from different images and measurement regimes. A-linear uses vLLM kernels in an isolated harness; A-attention uses the AITER wrapper. A is not a trace of a single vLLM or SGLang server.
- The previous agent reported a fused RoPE/cache-write path in the proposed SGLang image. Its raw image source dumps were lost. Reproduce the code evidence if accessible; otherwise mark this claim unverified.
- Later vLLM releases can also fuse operations. Avoid the blanket claim that vLLM is unfused and SGLang is fused. Compare the relevant versions and effective paths.
- The existing remediation plan predates the latest scope discussion. Its recommendation to preserve production fusion is not a decision this investigation must accept.

## Work to perform

### 1 Reconstruct the original partition and its purpose

Use profiler code, predictor/trainer code, simulator composition, repository history, and original documentation or papers where available. Establish which boundaries came from original Frontier or inherited code, which were introduced locally, and which were introduced specifically for this model/backend.

Investigate why work was grouped: shape dependence, reuse of regressors, separate communication modeling, scheduler phases, tensor parallelism, fusion in the original engine, convenient instrumentation boundaries, or other reasons. Distinguish documented intent from inference. If no design explanation exists, state that and explain the inferred rationale with code evidence; do not invent an author's motivation.

Trace the complete contract: profiler scope → CSV label/features → consumer selection → prediction lookup → execution-time composition. Check where costs are added, multiplied by layers, replicated or sharded, calibrated, or combined with communication. Identify whether an apparent partition is essential to simulator behavior or simply the current data schema.

Useful entry points include:

- `frontier/profiling/linear_op/{main.py,linear_op_impl.py,linear_op_wrapper.py}`
- `frontier/profiling/attention/{main.py,attention_wrapper.py,sequence_metadata.py}` and `backends/aiter_attention_wrapper.py`
- `frontier/attention/{families.py,ops.py,profiling_mapping.py}`
- `frontier/training/{linear_op_trainer.py,attention_trainer.py}`
- `frontier/execution_time_predictor/{sklearn_execution_time_predictor.py,shared_prediction_model_manager.py}`
- `frontier/entities/execution_time.py` and relevant communication/scheduling consumers reached from those paths
- `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench/dataset.py`

Treat this list as entry points, not a complete call graph. Prioritize the target co-located execution path; inspect other architectures only where a proposed partition would affect their supported interfaces.

### 2 Assess every operator in A

Cover all nine named scopes explicitly:

1. `attn_pre_proj`
2. `attn_rope`
3. `attn_post_proj`
4. `input_layernorm`
5. `post_attention_layernorm`
6. `emb`
7. `attn_kv_cache_save`
8. `attn_prefill`
9. `attn_decode`

For each, establish:

- Exact work inside A's timer, including reshapes/copies, normalization, residual addition, allocation or other incidental work, and exclusions.
- Actual consumers, features, phase/TP applicability, and how the predicted cost contributes to simulation. Include mixed consumers of the existing attention scopes.
- Corresponding SGLang call path in the pinned image, its backend dispatch conditions, and physical kernels or kernel families where identifiable.
- Whether it is a distinct runtime unit, a compound sequence, part of a fused unit, absent in a regime, or overlapped with work assigned elsewhere.
- Whether A's standalone computation represents the intended serving computation. Separate errors/artifacts from legitimate implementation differences and from timing-instrument limitations.
- Whether the current definition is useful as a modeling unit. Consider physical measurability, independent feature dependence, fusion, overlap, stability across regimes, and simulator composition—not just source function names or kernel count.
- A specific recommendation: keep; change boundary; merge; split; introduce a conditional variant; replace an incorrect harness implementation; or mark unsupported pending evidence. Name exact included/excluded work and applicability conditions.
- What that recommendation means for existing A labels: reusable as-is within stated limits, usable only as a historical/control reference, not directly comparable, or requiring recollection. Do not assert that summing independent measurements reproduces a fused cost, or that a measured fused duration can be divided into its former components.
- Precise consumer/schema changes that would be needed, and a minimal validation experiment if static evidence cannot settle the issue.

Do not force different prefill/decode or graph/eager paths into one global partition if the code supports different physical groupings. Conversely, do not create separate operators simply because a compiler emits several kernels for one stable cost unit.

### 3 Investigate cross-boundary problems

Explicitly examine:

- QKV projection plus Q/K normalization: is A's combined scope still a coherent prediction target, and can normalization fuse with subsequent work?
- RoPE plus KV-cache write: determine fusion gates and whether two independent simulator terms remain valid. Explain when separate profiling is useful as a control and when it cannot represent deployed execution.
- Norm/residual/all-reduce combinations, including layer 0: establish where communication is counted and how to avoid adding it twice when a composite cost is used.
- TP embedding: compare A's TP1 gather with sharded/masked/reduced serving behavior; distinguish once-per-forward work from once-per-layer work.
- Standard, chunked, batched, and mixed attention: determine whether serving separates prefill and decode kernel calls, and whether the required shape keys survive the proposed grouping.
- Graph capture/replay, compilation, shape-dependent dispatch, and multiple streams: do boundaries or overlap change, and does Frontier's additive timing composition remain appropriate?

Inspect adjacent MoE, sampling, final norm, output-head, or communication work only as needed to identify boundary contamination, missing costs, and overlap. Flag broader coverage gaps, but do not turn this into a complete redesign or collection project for all model operators.

### 4 Recommend a coherent target partition

Produce a minimal recommended SGLang/AMD cost model, not just nine isolated critiques. Show how A scopes map many-to-one, one-to-many, conditionally, or not at all to proposed units. Account for every in-scope physical operation exactly once per applicable invocation. Keep runtime dispatch gaps separate from kernel execution and explain how overlap affects summation.

For every proposed merge, explain how existing individual terms are suppressed or replaced in the consumer. For every split, explain why separate prediction adds value and how timings can actually be attributed. If communication is included in a composite, identify the communication term that must cease to be added independently.

State separately whether preserving A boundaries helps historical comparability and whether it helps accurate prediction of the target serving stack. Recommend how Dataset B should handle that distinction, without implementing a scope change in the parallel collection workstream.

## Evidence standard

Every material per-op verdict must cite both the relevant Frontier definition/consumer and the SGLang implementation or an explicit evidence gap. Use repository commit plus file, symbol, and line references. For container source, preserve image digest, source path, version/hash, and the minimal relevant excerpt. A link to a broad directory is not sufficient evidence.

For historical rationale, use commit/blame/history and original primary documentation when available. For web sources, prefer pinned official source and versioned documentation. Current upstream code is not proof of what an older container runs.

Classify findings as observed in a runtime artifact, established by source under stated conditions, inferred rationale, or unresolved. Source inspection can prove a code path exists; without its effective configuration it does not prove that path executed in a particular run. Numerical benefit, overhead, and accuracy claims require measurements; otherwise give a testable hypothesis.

If the cluster or image is inaccessible, continue with local evidence and finish a useful report. Identify exactly which conclusions remain conditional and the commands/source fragments needed to settle them. Do not substitute an unverified modern implementation for the target image.

## Deliverables and working boundaries

Write your report to:

`/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/operator_boundary_investigation/REPORT.md`

Include:

1. An executive conclusion: whether the inherited partition fits our target and the most consequential changes, with uncertainty stated.
2. The reconstructed design rationale and which parts are documented versus inferred.
3. A nine-row verdict matrix with current scope, target behavior, recommendation, applicability, evidence references, and impact on A/B comparability.
4. Per-operator explanations with code evidence and consumer consequences.
5. The proposed coherent partition and explicit accounting rules, including fusion/phase variants and communication treatment.
6. A prioritized migration outline identifying affected files/interfaces, data that would need recollection, minimum validation, and decisions that genuinely require user input.
7. A short handoff to the Dataset B agent: what collection design can proceed independently, which boundaries should remain provisional, and which traces/metadata should be preserved to support the decision.

Keep minimal supporting evidence under this same investigation directory. This directory is your write scope so another agent can work on Dataset B concurrently. Do not modify Dataset A, Dataset B plans or collectors, shared simulator code, configurations, or other agents' artifacts. Do not submit GPU jobs, start model servers, install packages, or change clocks. Non-mutating source/history/data inspection is in scope; propose any required live experiment rather than executing it.

Finish with a report even if some runtime questions remain unresolved. The investigation is complete when all nine scopes have evidence-backed or explicitly conditional verdicts, the recommended partition composes coherently, and the next implementation decisions are concrete. Neither agreement with the old partition nor discovery of fusions is a predetermined desired outcome.
