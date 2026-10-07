# Dataset B — implementation and collection agent prompt

Implement and collect Dataset B: occurrence-level GPU kernel measurements inside complete Qwen3-30B-A3B serving runs on MI355X, using the approved SGLang image.

This task includes completing the instrumentation, running real GPU pilots, validating the measurement pipeline, and collecting the agreed sweep. The previous inspection-only restriction is superseded. Do not stop after another environment report or after running synthetic tests.

## Workspace and required reading

Local repository:

```text
/home/dn/Frontier-qwen3-profiling
```

Task directory, relative to the repository:

```text
profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b
```

Remote repository:

```text
/opt/shared/frontier-qwen3-profiling/Frontier
```

Read `AGENTS.md` and:

- [02_IMPLEMENTATION_HANDOFF.md](02_IMPLEMENTATION_HANDOFF.md)
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- [B_SCHEMA.md](B_SCHEMA.md)
- [CLUSTER_ACCESS.md](CLUSTER_ACCESS.md)
- [Inspection RUN.md](evidence/cluster-inspection-21725-amd-mi355x-3/RUN.md)
- [Model-path follow-up](evidence/cluster-inspection-21725-amd-mi355x-3/model_path_followup.txt)
- [Operator investigation REPORT.md](../operator_boundary_investigation/REPORT.md) and relevant E5/E6 evidence

Keep implementation and collection records under `dataset_b/`. Do not modify Dataset A, simulator/regressor semantics, the other investigation’s files, or unrelated work. No commit/push is requested.

## Verified starting point

Cluster: primary Memphis, partition `XAI`.

Access: `ssh cluster`, then `sudo -n -u dn` for remote commands.

Use `gpu-cluster:cluster-admin` and the execution mechanism available to you that permits SSH.

Preferred node: `amd-mi355x-3`. Recheck current availability before allocating; its previously idle state is not a reservation. Do not disturb other users’ jobs.

The exact approved image was pulled and verified on node 3 by job 21725:

```text
10.1.1.0:5050/dnis/sglang-rocm@sha256:d8e0eb9c3211b593fbd574ada06e03d48d956c159bb9b6c123f4d86c12d8e6dc
```

Expected Docker image/config ID:

```text
sha256:dece67677b2b6cfae2aa12a546fab317a1499745d4ec866c45b9bd4ea0fc0bde
```

Observed versions:

- SGLang 0.5.18
- torch 2.11.0+gitd0c8b1f — use this runtime observation over older planning claims
- ROCm 7.2.3
- rocprofv3 / rocprofiler-sdk 1.1.0
- hipBLASLt 1.2.2
- ROCTX library present

Query the AITER distribution as `importlib.metadata.version("amd-aiter")`, not `"aiter"`.

The historical model path is absent on node 3. Candidate:

```text
/mnt/data/data/hai/qwen-blockwise-fp8-ab-20260810/models/Qwen3-30B-A3B
```

Only its config was checked. Before serving, verify tokenizer completeness, weight index, all referenced shards/symlinks, actual tensor dtypes and quantization metadata. Confirm this is the required BF16 model, not the neighboring FP8 checkpoint. Record model/tokenizer provenance and hashes. A BF16 config field alone is insufficient.

Image source files are already available in:

```text
dataset_b/evidence/cluster-inspection-21725-amd-mi355x-3/
dataset_b/evidence/approved_image_source/
```

Do not repeat broad image exploration. Verify the pinned identity on the selected node and recover only missing source/capability evidence.

## Important: the existing collector is not ready for live collection

- `sglang_b_markers.patch` currently adds only marker support; no serving call sites are instrumented.
- Native ROCTX adaptation remains unfinished.
- Graph replay attribution remains unfinished.
- The launch plan is a placeholder.
- b0.1 deliberately always reports `dataset_accepted=false`.
- The existing tests use synthetic records.
- Parquet export needs a working PyArrow environment and a real roundtrip check.

Complete these components. Do not make them appear ready by changing verification booleans, weakening checks, or replacing `dataset_accepted=false` with an unconditional true.

## 1. Freeze the explicit collection recipe

Use this recipe for the initial pilot:

- Qwen3-30B-A3B, BF16 weights/compute and BF16-compatible KV; no quantization override.
- TP1 first; TP4 validation and sweep collect every rank.
- Attention backend AITER; `SGLANG_USE_AITER=1`.
- Page size 1; radix cache disabled.
- Automatic clocks.
- `DEBUG_CLR_MAX_BATCH_SIZE=1000000`.
- `HSA_ENABLE_COREDUMP=0`.
- Explicit allocated GPU visibility.
- Baseline chunked-prefill size and prefill-token budget 16384; chunking cases override as specified below.
- Context capacity sufficient for the longest prompt plus generation; 16640 is the planned baseline.
- Maximum running requests 64.
- Separate eager and graph-enabled configurations.
- Leave all-reduce fusion and unified-attention/prefill opt-ins disabled for this initial recipe.
- Preserve the ordinary scheduler/overlap behavior unless a concrete instrumentation limitation requires an explicitly labeled diagnostic control.
- Preserve and record the server-wide decode partition count.

This is an explicit collection configuration, not a claim of universal production defaults. Record image-inherited relevant environment as well as explicit overrides.

Check the actual pinned CLI/source and record effective settings after initialization. Do not silently accept backend, dtype, graph, fusion or chunking fallback.

## 2. Implement exact source-matched instrumentation

Produce a reviewable patch against the approved image’s actual sources and retain original/patched hashes. Mount the patched files or package tree into that image without changing its dependencies.

Instrument physical units:

- QKV projection plus Q/K normalization as one compound scope.
- Output projection GEMM, excluding communication.
- Exact input-normalization call; distinguish layer-0 norm-only.
- Exact post-attention normalization call.
- Embedding once per forward, with TP-specific variant.
- Fused RoPE plus KV write as ONE physical scope, mapping to both historical A labels.
- Actual attention-core call, preserving backend-required fill/work kernels inside the chosen boundary and distinguishing execution regimes.

Do not mark enclosing `prepare_attn`/`prepare_mlp` communicator wrappers as pure norms. Do not split a fused duration or count it twice. Do not force unfused execution to restore A’s labels.

MoE, collectives outside the selected physical scopes, final norm, output head and sampling remain excluded prediction targets. Preserve their raw trace records for accounting and predecessor context.

Use one annotation backend per pass: ROCTX for rocprofv3, `record_function` for Kineto. Precompute static labels. Add no event-timing pairs, synchronization, device-to-host shape reads, settling spins, synthetic backlog or periodic worker disk flushes.

Collect CPU-side metadata for:

- run/process-instance/native PID/TP rank/physical GPU identity
- rank-local and cross-rank forward correspondence
- ordered request membership and actual query/prefix/attended-length vectors
- real and padded token/batch sizes
- phase/backend, layer, variant, graph capture/replay identity

Buffer metadata and flush only after a verified drain. Detect overflow and incomplete shutdown.

## 3. Complete native trace extraction and telemetry

Primary instrument: GPU kernel begin/end timestamps from rocprofv3.

Start from the installed tool’s supported flags:

```text
--kernel-trace --hip-runtime-trace --marker-trace
```

Select output formats based on actual identity fields. The installed version lists `csv`, `json`, `pftrace`, `otf2` and `rocpd`; inspect real output before choosing. Preserve original native traces.

For eager execution, demonstrate:

```text
dispatch → correlation → host launch → process/thread marker → scope/forward/layer
```

For graph replay, demonstrate capture-derived or native graph/node correspondence on this exact toolchain. Repeated names, eager-order templates, and the existence of `--kernel-rename` are not sufficient proof. Preserve the assignment evidence.

If graph attribution is initially unresolved, keep implementing and collecting clearly labeled eager pilot data while resolving it. Do not claim the graph requirement is fulfilled or silently remove it from final coverage.

Sample clocks continuously in a separate process, per physical GPU. Preserve raw samples, read intervals, sensor update resolution, gaps and errors. Establish the tracer-to-sampler timebase relationship experimentally, validate held-out anchors, and freeze justified freshness/uncertainty limits before acceptance. Do not copy synthetic-fixture limits.

Retain every kernel occurrence. Compute queue predecessors before target filtering. Keep kernel duration, scope kernel sum and scope span separate; overlapping streams are legal.

Use independently observed scheduler/request records for completeness, not the same attribution template that labels kernels. Extend the checker to support the demonstrated native/graph paths while retaining rejection of ambiguity, dropped records, missing ranks, metadata loss and incomplete windows.

## 4. Run the real pilot

Use Slurm, node pinning and bounded jobs. Begin with a small TP1 eager run that answers whether the instrument works, then expand within the pilot.

### Pilot matrix

| Case | Configuration and required observation |
|---|---|
| A. TP1 eager | Prompt lengths `{1,8,64,512,1024,1025,4096,4352,4360,8192}`; `max_new_tokens=8`, concurrency 1. |
| B. TP1 graph decode | Observed batches near 1 and 8, plus a non-capture-size cohort such as 3 to exercise padding. Use explicit capture buckets and record actual replay/padding. |
| C. Chunking | Prompt length 8192, chunk size 1024, `max_new_tokens=4`. Verify the actual sequence of prefix/query lengths. |
| D. Mixed execution | Introduce a prefill request while another request is decoding. Confirm actual scheduler/backend behavior; HTTP overlap alone is not mixed-batch evidence. Inspect whether `--enable-mixed-chunk` is required and, if used, collect it as a separately identified configuration. SGLang need not partition mixed attention the way A’s synthetic profiler did. |
| E. TP4 | Short and long prefill plus eager and graph decode, collecting all four ranks and corresponding telemetry. |
| F. Kineto | Separate matched TP1 passes covering short/long prefill and representative eager/graph decode. |
| G. Controls | Matched unmarked/untraced, trace-only and trace-plus-marker cases sufficient to check output equivalence and quantify marker/tracer perturbation. |

Use fixed exact token-ID manifests, greedy generation and `ignore_eos`. Verify vocabulary/special-token handling and preserve output token IDs and response metadata. Warm up JIT/allocators before the measured window.

Check actual first-token/forward semantics rather than assuming the current G−1 decode rule. Verify profiler start/stop and worker drain behavior against the pinned implementation.

Write pilot `RUN.md`, native-trace fixtures, extraction outputs, checker reports, numerical-equivalence results, telemetry calibration and perturbation findings.

Present the pilot results and measured runtime/storage estimate before starting the sweep. Resolve material scope/configuration failures. Do not impose an unsupported universal 2% tracer-agreement threshold.

## 5. Collect the sweep after the pilot gates pass

Use [workload_grid.json](workload_grid.json) as the saved historical grid, then create a frozen executable manifest incorporating the validated mixed and padding coverage.

### D1

- All 155 saved prompt lengths.
- TP1 and TP4.
- Eager and graph-enabled configurations.
- Concurrency 1, `max_new_tokens=16`.
- Three independent repetitions.

### D2

- Chunk sizes 1024, 4096, 8192.
- Corresponding prompt lists in `workload_grid.json`.
- TP1 and TP4.
- `max_new_tokens=4`.
- Three independent repetitions.
- At minimum the planned graph-enabled configuration.
- Expected arithmetic is 186 prefill forwards per TP under exact chunking; actual observations govern.

### D3

- Concurrency targets `{8,32,64}`.
- Prompt lengths `{128,512,1024,2048,4096,8192}`.
- `max_new_tokens=32`.
- Two independent repetitions.
- TP1 and TP4, eager and graph-enabled.
- Concurrency is not a batch label. Reconcile observed membership, contexts and padding.
- The largest cohort contains 524288 prompt tokens and requires at least 32 prefill forwards at a 16384-token budget.

### Mixed/padding extension

- Freeze a small pilot-derived manifest that actually produces the required observed regimes.
- Do not assert mixed coverage from request submission patterns.

Randomize/interleave workload order using a retained seed. Collect every occurrence and rank. Split collection into bounded, independently checkable runs so the in-memory prototype does not exhaust host memory; implement streaming/chunked extraction where needed.

The full grid is not automatically a full cross-tracer duplicate. Use the validated representative Kineto cross-checks unless measurements demonstrate a need for more.

If a required mapping remains unresolved, present that concrete decision with source/runtime evidence while finishing independent cells. Do not quietly narrow coverage or label a partial collection complete.

## 6. Operations and outputs

Check queue/node availability before every allocation. Use short identifiable job names, bounded wall times and only the allocated GPUs. Clean up only jobs/servers started by this task.

Shared dataset destination:

```text
/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/insitu_b/<unique_run_id>/
```

Never overwrite a run. Preserve failures.

Account for NFS `root_squash`: distinguish pre-created writable parent/mount directories from leaf directories that the Python tools intentionally create with `exist_ok=False`. Fix the runner/container UID or directory layout as needed, without weakening immutable-run semantics.

Preserve per run:

- request/configuration manifests and exact commands
- image, model/tokenizer, original-source and patch identities
- effective configuration and rank/device map
- scheduler/forward/scope metadata and completion/output tokens
- full native traces and raw telemetry
- occurrence-level Parquet, derived scope/forward tables
- extraction version/source hashes and input hashes
- checker/calibration/control reports
- `RUN.md` and explicit coverage/limitations

Sync scripts with rsync checksums as `dn`, without `--delete`. Copy small evidence and reports locally; keep large datasets in the shared store with exact pointers and checksums.

Run meaningful local tests and the applicable repository gates; distinguish existing failures from collector regressions. Do not claim synthetic tests establish GPU correctness.

## Final delivery

Deliver:

1. Source-matched instrumentation patch and runnable launch/driver/extraction/check commands.
2. Real pilot results, including graph attribution and every TP4 rank, or precise remaining failures.
3. Frozen physical-scope/A correspondence and actual coverage matrix.
4. Exact raw-data, Parquet, telemetry, manifest and `RUN.md` paths.
5. Reproducible checker results and documented instrument limits.
6. A short confounder note; no A-versus-B fitting or causal performance conclusions.
7. An honest completion statement: full validated B coverage, or explicitly partial with the smallest remaining action.

A plan, source inspection, model-server startup, or green synthetic checker alone does not complete this task.
