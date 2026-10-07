# Dataset B collection implementation handoff

> Implementation follow-up: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) records the local collector, tests, and precise live-access blockers. The task below remains the governing scope; no real B pilot has been collected.

Prepared 2026-10-04 for a new implementation session. Read this file as the current task, then inspect the referenced artifacts. Implement the collection tooling and validate it; do not stop after producing another plan.

## Objective and first milestone

Build Dataset B: reproducible, occurrence-level GPU kernel measurements inside complete Qwen3-30B-A3B serving runs in SGLang on AMD MI355X. B must retain enough provenance, actual shape information, scope attribution, clock context, and queue context to support later comparison with Frontier's standalone Dataset A.

Your first delivery is working collection software, a reviewable SGLang instrumentation patch, a documented schema, meaningful validation, and a small real pilot where the environment and scope decisions permit it. If live work is blocked, finish the independent local implementation and checks and report the precise remaining blocker. Do not represent synthetic fixtures or source inspection as real GPU validation.

The full sweep follows presentation of the pilot results and resolution of its material failures and operator-scope decisions. Do not launch a broad collection on provisional boundaries. A-versus-B performance conclusions and regressor fitting are later tasks.

## Workspace and current state

Repository: `/home/dn/Frontier-qwen3-profiling`

Knowledge directory: `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/`

Collection work directory: `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/`

At handoff, HEAD is `284130bb8d632d4a4c3c3efdcea8a8f26178110b`. Recheck the current state before editing. `dataset_b/` and `operator_boundary_investigation/` are untracked. Several unrelated paths are also untracked, including `data/profiling_gcfix/`, `logs/qwen3_mi355x/`, the regressor sidecar, an image, and a fixed-clock script. Preserve them. Do not stage unrelated work or assume untracked files are disposable. If using an isolated checkout, remember these task documents will not appear automatically because they are untracked.

What exists:

- The original inventory and collection proposal, `00_inventory_and_plan.md`.
- The earlier agent's historical `HANDOFF.md`.
- An independent `REVIEW_REPORT.md` with verified problems.
- `01_REMEDIATION_PLAN.md`, a proposed implementation sequence.
- A separate operator-boundary investigation prompt.

What does not exist at handoff: a Dataset B marker patch, request driver, trace extractor, clock sampler, checker, B schema, pilot, or collected B data. The previous agent reported source/container inspections, but no model-serving or collection jobs. Its raw container source dumps were lost. The review verified local data/code and arithmetic; it did not repeat image inspection or run GPUs.

## User decisions and latest direction

The following session decisions supersede stale assumptions in the earlier planning files.

**The serving image is approved:**

`dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04`

Do not ask the user to choose this image again. Resolve its digest, inspect its contents, and preserve the evidence. Historical notes report SGLang 0.5.18 and ROCm/AITER tooling; these still require verification against the actual image. Do not silently substitute a different image or library stack if something is unavailable.

**Operator boundaries are under active investigation.** The user initially clarified that we should stay close to Frontier's measured operations and avoid gratuitously breaking a compound operator into individual kernels. They then raised a deeper issue: Frontier's partition was inherited from a vLLM/NVIDIA-oriented design and may not be right for SGLang/AMD. A separate investigation must reconstruct that rationale and assess every A scope with code evidence.

Consequently:

- The inventory of A is the starting comparison map, not proof that its partition is optimal for SGLang/AMD.
- Preserve multi-kernel scopes where they remain appropriate. Raw per-kernel records do not imply one prediction target per kernel.
- Do not assume preserving every production fusion is already approved as the final modeling policy.
- Do not force unfused execution merely to restore inherited labels before evaluating whether those labels make sense for the target stack.
- Record physically inseparable work honestly. Never split a measured fused duration by guesswork or duplicate its duration into multiple targets.
- Keep scope definitions and A-to-B correspondence explicit and versioned so final decisions do not require rewriting the driver or trace storage.

The old blanket instruction that nothing can proceed until all seven questions are answered is stale. Start independent implementation now. Only an actual unresolved execution/scope decision should block its dependent work, and it should be presented with concrete code evidence and alternatives.

## Parallel operator investigation

The separate task is defined at:

`profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/operator_boundary_investigation/AGENT_PROMPT.md`

Its expected result is `operator_boundary_investigation/REPORT.md`. That report does not exist at handoff; only the prompt exists. Do not assume the other session has started or finished. Check for new findings as implementation progresses. Do not duplicate its historical investigation or modify its files.

You own Dataset B tooling and collection records. The other task owns the assessment of operator definitions. While its results are pending, implement raw record preservation, identities, request/forward metadata, telemetry, extraction primitives, and validation. Investigate concrete serving call sites as needed for the collector and retain your evidence for the other task.

Unambiguous scopes and trace-mechanism tests can be piloted under a clearly labeled provisional mapping. Do not claim complete B coverage or start the full sweep until the final mapping is resolved. If the report recommends material boundary changes, summarize the proposed collection consequences and resolve any remaining original stop-and-ask issues with the user; recommendations are not measurements or automatically approved simulator changes.

## Read first

Read repository `AGENTS.md` and any applicable nested instructions. Then read:

1. `dataset_b/REVIEW_REPORT.md` — the substantive review and verified findings.
2. `dataset_b/01_REMEDIATION_PLAN.md` — useful implementation sequence, subject to the updated decisions in this handoff.
3. `dataset_b/00_inventory_and_plan.md` — detailed historical inventory and draft source mapping; contains errors identified by the review.
4. `dataset_b/HANDOFF.md` — the original task and historical inspection provenance.
5. `14_regressor_dataset_handoff.md` and `10_measurement_contract.md` — A's labels, timing semantics, and limitations.
6. Dataset A's README and relevant `RUN.md` files; the operator investigation report if available.

Read other investigation documents on demand. Do not restart the full historical analysis before implementing. In particular, the remediation plan still describes image selection as pending and proposes a fusion preference: both are superseded by the latest context above.

## Dataset A reference and verified corrections

A-linear local mirror:

`data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv`

Shared-store counterpart:

`/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv`

Verified MD5: `3635e4d305ae5cf84884e275d3ff37ef`; 13,308 rows × 150 columns. There are 3,327 token shapes per TP in {1,2,4,8}. `attn_pre_proj`, `attn_post_proj`, and `attn_rope` have labels at every TP; `input_layernorm`, `post_attention_layernorm`, and `emb` have labels at TP1 only. Do not use the obsolete root linear CSV in place of this corrected file.

A-attention:

`data/profiling/compute/mi355x/qwen3-a3b-30b-moe/attention_combined.csv`

Verified: 25,032 rows, comprising 22,152 standard and **2,880 true-mixed** rows. The mixed rows have prefill, decode, and cache-write targets; all 2,880 have the complete feature set used by the inspected decode-in-mixed consumer. The old B plan incorrectly says these data are absent.

A starts with nine named scopes: `attn_pre_proj`, `attn_rope`, `attn_post_proj`, `input_layernorm`, `post_attention_layernorm`, `emb`, `attn_kv_cache_save`, `attn_prefill`, and `attn_decode`. Mixed consumers reuse attention scope labels; they are not automatically extra physical operations.

Other corrections to carry into implementation:

- A's linear and attention data use different images and measurement regimes. Legacy A-attention host contamination is a concern, not a measured decomposition of every row.
- Cache-write features differ by consumer: one path uses `num_tokens`; the shared path uses `(total_tokens, kv_cache_size, batch_size)`.
- The proposed D1 list contains 155 prompt lengths. D2 implies 186 prefill forwards per TP across its three repeats under exact chunking. The largest D3 cohort contains 524,288 prompt tokens, requiring at least 32 prefill forwards at a 16,384-token cap. Verify actual forward semantics and derive observed counts from execution records.
- The reported TP1 and TP4 post-projection jumps at 8192→8208 tokens were independently verified as approximately +30% and +81%.
- No usable current MoE dataset was identified. Do not silently expand the collection into all model operators.

## Environment and evidence recovery

Local analysis environment: `/home/dn/.virtualenvs/qwen3-profiling`.

Historical cluster access is through `ssh cluster`, with Slurm partition `XAI` and MI355X nodes. Historical model path: `/mnt/data/data/hai/hai2/models/Qwen3-30B-A3B`. Historical shared checkout: `/opt/shared/frontier-qwen3-profiling/Frontier`. These are leads, not current availability guarantees. Inspect existing access and launch conventions without exposing credentials. Use Slurm for GPU work, not a login-node model server.

Relevant launch references are under `profiling_knowledge/scripts/slurm/`, including `qwen3_mi355x_profiling.sbatch` and `qwen3_validation_trace.sbatch`. Reuse proven environment/container handling where applicable; do not copy the old event-timing measurement loop into B. The shared store was not mounted in the review session, so use the local A mirror when needed.

Recover only the image evidence needed to implement: effective versions and profiler capabilities; the Qwen3 model path; norm/communication, RoPE/cache, attention backend and embedding paths; graph capture/replay; scheduler/forward metadata; profiler start/stop behavior. Preserve source excerpts or hashes plus digest and inspection commands. Do not repeat a five-image comparison.

## Implementation requirements

Keep code modular and local to `dataset_b/`, reusing existing utilities when appropriate. Suggested components: `sglang_b_markers.patch`, `drive_b.py`, `extract_b.py`, `check_b.py`, a telemetry helper, a run wrapper, `B_SCHEMA.md`, and focused tests/fixtures. Exact internal organization is your choice; avoid a new general-purpose profiling framework.

### Measurement and instrumentation

Use GPU kernel begin/end timestamps. Primary tracer: rocprofv3. Cross-check: PyTorch profiler/Kineto on representative matched configurations. Do not add HIP/CUDA event pairs as B's instrument, synthetic backlog kernels, settling spins, or clock locks to the main run.

Enable markers explicitly, keep hot-path work minimal, and use only the annotation backend required for the selected tracer pass. Precompute static labels. Request/shape bookkeeping must avoid synchronization, device-to-host reads, and periodic disk flushes in measured execution. Buffer metadata and quantify any residual perturbation. Verify marked/unmarked numerical behavior as well as trace behavior.

Place scopes around exact calls, not broad wrappers that include excluded work. A marker around `prepare_mlp`, for example, may include all-reduce as well as norm. Layer-0 norm, TP embedding, and fused operations need explicit variants. Preserve physical observations even when their final A mapping is provisional.

Keep raw traces and every retained target kernel occurrence; do not aggregate away repeats. Record a derived scope table separately. Preserve the raw full-model trace for attribution and predecessor accounting without declaring its excluded kernels new B targets.

### Identity and schema

Identify runs, process instances, TP ranks, GPUs, native dispatches, correlations, queues/streams, forwards, layers, and scope invocations. Collect all participating ranks. Establish rank-to-GPU mapping and cross-rank forward correspondence explicitly. Never join clocks using TP degree alone or merge scope records across ranks accidentally.

Record request membership and actual per-sequence query/prefix/attended lengths. Preserve real versus padded tokens/batch size and mixed batch composition. Requested HTTP concurrency is not an observed batch size, and a mean context length is not a replacement for the actual vector. Distinguish model-global and per-shard dimensions and provide the exact aliases/derivations needed for A's keys.

Version the physical scope specification and the correspondence to A. Store immutable software/configuration metadata in a referenced run manifest rather than copying long strings into every row. Document units, nullability, capture-window boundaries, and validity states.

Define at least `kernel_dur_us`, `kernel_sum_us`, and `scope_span_us` separately. Span is the last end minus the first begin and includes gaps; sums can exceed span with overlapping kernels. Use timestamp extrema, not arbitrary trace order. Compute queue predecessor context before filtering excluded kernels. A same-queue interval is not necessarily device idle or host overhead. Handle a missing predecessor at the window boundary explicitly and preserve raw timestamps for diagnosis.

### Attribution and graphs

For eager execution, validate the correlation chain from device dispatch to launch and host marker, including process/thread identity and nested ranges. Device execution need not occur inside the host range's wall-clock interval.

For replay, prove attribution using capture-derived correspondence or supported graph/node identity in the pinned toolchain. Repeated kernel names and an assumed eager sequence are insufficient to label 48 similar layers. Determine which output format retains the required fields. New documentation is not proof that an old SDK supports a feature. Preserve enough evidence to reproduce each assignment and reject ambiguity.

If graph attribution is not yet solved, continue and validate the eager path while resolving it. Do not quietly drop graphs from the required coverage, upgrade the image, or label a partial result complete.

### Telemetry and configuration

Sample clocks continuously outside the model's execution path and join by GPU and calibrated timebase. Retain raw samples, sensor update resolution, read latency, age, alignment uncertainty, and gaps. A nearby sample is not an instantaneous per-kernel clock measurement. Test stale and wrong-device joins. Numeric validity limits must be justified by calibration, then frozen before acceptance checks.

Use TP1 and TP4 as implementation defaults; BF16, page size 1, radix cache disabled, and automatic clocks. Confirm effective backend, chunking, graph, and fusion flags against the image. Record the server-wide decode partition count rather than rewriting it to imitate A.

The original task requires A's relevant runtime hygiene. Preserve `DEBUG_CLR_MAX_BATCH_SIZE=1000000` as the planned main setting and label its potential difference from ordinary production. Do not claim a fully production-default baseline. Optional default-setting or clock-lock controls are not prerequisites for beginning implementation.

### Checker and meaningful tests

Replace the original contradictory checks. Expected target invocations must be derived from observed forward metadata and scope applicability; kernel identities/counts may legitimately change with shape and execution variant. Known MoE, sampling, output-head, and communication work outside target scopes is not itself a failure.

Require nonempty data; unique target attribution; explicit exclusions and raw-to-extracted accounting; all ranks and requests reconciled; valid timestamp/shape identities; clock coverage; and trace completeness. Detect duplicates, lost records, missing ranks, unresolved target assignments, incomplete windows, and metadata loss. If a tracer lacks drop diagnostics, state what other checks establish and what they cannot guarantee.

Test repeated kernel names, overlapping streams, mixed regimes, padding, fused scopes without double counting, legitimate dispatch changes, missing/duplicate records, stale/wrong-device clocks, and empty/truncated inputs. Use independent observations for expected counts; do not let an attribution template certify its own completeness. Apply repository development gates to the actual changes, reporting exactly what was run and any unmet requirement. Simulator-wide changes are outside this implementation task.

## Sequence of work

1. Inspect the worktree and read the current task documents. Check whether the operator investigation has produced findings. Briefly state the implementation approach and any genuine blockers, then proceed.
2. Correct the existing inventory and decision/status sections using the review and latest user decisions. Draft `B_SCHEMA.md` and a small explicit scope/coverage specification. Record unresolved fusion mappings instead of choosing silently. Present the corrected pre-cluster plan as required by the original task; independent local coding can continue.
3. Implement normalization, identity, attribution primitives, telemetry joins, checker fixtures, and the manifest-driven request path. Recover pinned image source and develop the minimal marker patch and launch wrapper. Keep source modifications reviewable and confined to the intended image/run.
4. Validate launch arguments, model access, tracing capabilities, and output paths before GPU allocation. Use the smallest run that answers each uncertain mechanism. A missing remote capability does not block local parser/checker/driver work.
5. Run a small TP1 pilot under rocprofv3, then matched Kineto cases, once the affected mappings are resolved or explicitly designated exploratory. Include short and long prefill, decode eager/replay, chunking, a mixed case if the target path supports it, and padding behavior. Add a small TP4 validation covering every rank. Preserve actual outputs as regression fixtures where size permits.
6. Assess attribution, numerical equivalence, telemetry, marker/tracer perturbation, scope coverage, and trace loss. Separate structural failures from documented instrument limits. Do not enforce the draft's universal 2% tracer-agreement threshold without supporting calibration.
7. Deliver the pilot records, checker results, unresolved limitations, and a measured estimate for the full sweep. Incorporate the operator investigation's resolved boundaries before freezing full coverage. Present results before starting the full sweep; if a required scope decision remains unresolved, ask about that concrete decision while completing independent work.

Implement and verify within this sequence rather than waiting for answers to obsolete questions. Do not treat a pending operator research report as a reason to leave the entire collector unwritten.

## Output and operational requirements

Keep the collection code, schema, patch, tests, and concise records under `dataset_b/`. Update the existing plan and historical handoff pointers so the next reader can identify the current status. Preserve `REVIEW_REPORT.md` as the review record. Do not modify the other investigation's files, existing A datasets, profiler/regressor semantics, or unrelated user work.

Historical proposed dataset destination:

`/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/insitu_b/<run_id>/`

Verify access and use a new unique run directory. Never overwrite old collections. Preserve input manifest, output tokens/completion metadata needed for validation, image/source/patch hashes, effective configuration, rank/device map, raw traces, telemetry, extracted Parquet, checker reports, and `RUN.md`. Full duplicate CSV exports are optional; inspection samples suffice initially. Keep secrets out of environment manifests.

Use existing cluster allocation conventions and bounded jobs. Clean up only servers/jobs started by this work. Do not alter shared clock settings or add optional large experiments. Preserve failures and partial artifacts with honest status; do not quietly drop bad cells to make a checker green. No commit/push or broad staging is implied by this handoff.

## Required handoff from your session

Finish with concrete evidence:

- What was implemented and where; the patch and runnable collection/extraction/check commands.
- What tests and real runs passed, including whether graphs and every TP rank were demonstrated.
- The chosen scope-definition version, A mapping, and any dependency on unresolved operator research.
- Exact paths to pilot data, raw evidence, schema, run records, and checker output.
- Remaining blockers with the smallest next action that resolves each.

If blocked from live collection, deliver the completed local work and state that no real B measurements were produced. If the pilot succeeds, report readiness for the full sweep separately from dataset completion. A useful collection result must be traceable to actual kernel executions; it does not need to agree numerically with A.
