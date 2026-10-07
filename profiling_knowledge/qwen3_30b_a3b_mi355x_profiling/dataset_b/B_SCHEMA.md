# Dataset B schema b0.1

Status: implemented **offline/eager contract**, provisional physical mapping `qwen3-provisional-0.1`. This is not evidence that the pinned image emits these records. No live measurements exist in this session. Every file under `evidence/synthetic_validation/` is synthetic. Native roctx adaptation and SGLang call sites await image access; replay is rejected rather than assigned by names.

## Storage and identity

The pilot interchange is a strict JSON bundle with keys `manifest`, `kernels`, `launches`, `ranges`, `scopes`, `forwards`, `scheduler`, `requests`, and `telemetry`. Native traces remain immutable separate files; native payloads and source references are retained in normalized rows. `normalize_b.py` records raw-file SHA256s. `extract_b.py` records input-bundle and scope-spec SHA256s and refuses an existing output directory. It writes all normalized kernel occurrences (including explicit exclusions), derived physical scopes, derived forward features, and a checker report. `--parquet` also writes Zstandard Parquet using PyArrow, or fails explicitly if unavailable. JSON remains the dependency-free interchange, not a substitute claimed as Parquet. Native payloads are serialized into the Parquet `native` string column.

All timestamps are integer nanoseconds; durations are floating microseconds. Native clock domains are run-specific, never assumed to be Unix time. CPU sampler timestamps use `time.monotonic_ns`. Driver records both monotonic and Unix timestamps for provenance, not for an uncalibrated device join. IDs are strings; rank and layer are integers. `process_id` is an opaque **process-instance** ID, unique within a run, not merely an OS PID. Generate it at worker startup; preserve native PID and startup evidence in the process manifest. A restart must get a new identity. b0.1 supports one worker process per TP rank within a capture; restarted windows require separate runs.

| Record | Unique key (all within run_id) | Required contents and source |
|---|---|---|
| Manifest | `run_id` | `schema_version=b0.1`, `data_kind=live\|synthetic_fixture`, `tp`, `scope_spec_version`, model dimensions, process map, windows, trace diagnostics, calibration and policy; observed evidence, never inferred from TP |
| Process | `process_id` | native PID, TP `rank`, stable physical `gpu_id` (UUID or PCI identity plus node), device/visible-index mapping evidence; b0.1 requires a distinct GPU per rank |
| Raw kernel | `(process_id,gpu_id,dispatch_id)` | native correlation, queue, stream, name, begin/end, `execution=eager\|graph_unresolved`, `raw_ref`, native payload; use native dispatch ID if present |
| Host API | `(process_id,correlation_id)` | native PID, thread ID, name, begin/end, source reference; multiple ambiguous records fail instead of picking one |
| Paired range | `(process_id,marker_id)` | thread, begin/end, source reference, static label; optional `scope_id`, enclosing `kind=forward`/`forward_id`, or explicit exclusion reason |
| Scope invocation | `(process_id,scope_id)` | forward ID, physical scope, layer (null only for model-level embedding), variant; metadata from exact call sites |
| Forward | `(process_id,forward_id)` | explicit cross-rank `global_forward_id`, phase, execution (`eager` or `replay`), ordered sequence vector, padded token/batch counts; observed worker metadata |
| Scheduler | `global_forward_id`; independent unique `ordinal` | ordered sequence vector, phase, source reference from scheduler bookkeeping, collected independently of marker/dispatch labeling |
| Request | `request_id` | exact input and output token IDs, prompt tokens, requested/generated tokens, complete/failed status; driver response is preserved separately |
| Telemetry | `sample_id` | GPU, read begin/end, timebase, sclk/mclk MHz, native text/errors, sensor update resolution (nullable pending calibration) |

A Kineto trace lacking native dispatch IDs uses `event:<original traceEvents index>` scoped to its immutable trace/process. This is an explicitly documented surrogate, not a invented hardware dispatch ID. b0.1 assumes one complete Kineto file per process; splitting trace files requires a trace-file identifier in this surrogate before combining.

The run manifest stores image tag, immutable image ID/repo digest, model/tokenizer revision, source/patch hashes, tool versions, node/job, effective server flags and backend, allowed environment, seed, rank/device evidence and capture controls. Long configuration strings belong here, not on every occurrence. Required evidence names checked for live readiness are `image_digest`, `source_hashes`, `patch_sha256`, `effective_config`, `model_revision`, `tokenizer_revision`, `numerical_equivalence_ref`, `perturbation_ref`, `tracer_crosscheck_ref`, and `rank_device_evidence_ref`. Merely filling these names cannot certify full coverage: **b0.1 always reports `dataset_accepted=false`**.

## Shape semantics and A aliases

Each sequence has `request_id`, `kind=prefill|decode`, `query_len`, `prefix_len`, and `attended_len=prefix_len+query_len`. Decode query length is exactly one; speculative decoding is outside b0.1. Sequence order and all lengths are retained. Requests can occur across multiple chunked forwards; cross-rank membership must equal the scheduler record. Forward `phase` is derived from this vector, including true mixed batches. HTTP concurrency is never used as batch size.

`real_tokens=sum(query_len)`, `real_batch_size=len(sequences)`. `padded_tokens` and `padded_batch_size` are actual execution sizes read from CPU-side scheduler/graph metadata; both must be at least real sizes. `num_tokens=padded_tokens` gives the linear GEMM execution M; `total_tokens=real_tokens` follows the raw attention input convention. Padding and backend-specific changes can make this insufficient for individual physical operators: do not claim exact A compatibility without verifying their execution sizes. Additional per-scope shape overrides are a future source-integration requirement if an operator trims or repads the forward.

Standard-A `batch_size`, `kv_cache_size`, `prefill_chunk_size` and its square are populated only for uniform, unpadded, non-mixed forwards. Otherwise they are null; vectors remain authoritative. Prefill chunk is query length, prefix is KV cache length. Decode chunk is zero, prefix is attended length minus one. Uniform multi-sequence prefill still requires the appropriate mixed-prefill consumer; `standard_a_compatible` means a shape can be expressed by scalar keys, not that every A consumer or physical scope is equivalent.

The model stores global `n_q_head=32`, `n_kv_head=4`, `n_embd=2048`, `n_expanded_embd=768`, `vocab_size=151936`, `head_dim=128`, `num_layers=48`, QK norm/gated-MLP/output-gate flags and page size for the real run. Synthetic fixtures deliberately use two layers for size. `n_head` aliases global query heads. Per-shard `local_q_heads=n_q_head/TP`, `local_kv_heads=max(1,n_kv_head/TP)`, and `qkv_projection_width=(local_q_heads+2*local_kv_heads)*head_dim` preserve KV replication at TP8. b0.1 live pilots default to TP1/TP4. Record server-wide decode partition count and effective chunking/layout/fusion settings in `effective_config`; do not emulate A's partition policy.

| Consumer | Alias / derivation |
|---|---|
| Linear/regressor six scopes | `num_tokens`; partition key `num_tensor_parallel_workers`; replicated A norms/embedding have TP1 labels only |
| Execution predictor cache write | `num_tokens` (physical padding relevance must be checked) |
| Shared cache-write predictor | `(total_tokens, kv_cache_size, batch_size)`; scalar compatibility nullable as above |
| Standard prefill | `(kv_cache_size, prefill_chunk_size_squared)` |
| Standard decode | `(batch_size, kv_cache_size)` |
| Mixed prefill | `prefill_mixed_*`: batch size, total query tokens/square, query-length mean/min/max, population variance, CV=std/mean, range, batch×variance, batch×CV |
| Mixed prefill context | exact floating mean `prefill_context_mean_exact`; integer truncation `prefill_mixed_kv_cache_size_raw`; consumer alias `prefill_mixed_kv_cache_size` rounds that integer up to **explicit** `manifest.consumer_aliases.kv_cache_prediction_granularity`, otherwise null |
| Decode-in-mixed | `decode_batch_size`, `num_prefill_seqs`, `total_prefill_tokens`, `total_batch_size`, `batch_composition_ratio=num_prefill_seqs/total_batch_size`, `total_tokens` |
| Decode-in-mixed context | CSV `decode_avg_kv_cache_size=int(mean(prefix_lengths))`; exact mean also retained; `decode_prediction_kv_cache_size` applies explicit granularity for runtime-consumer use, otherwise null |

These derivations follow `frontier/attention/profiling_mapping.py`, `TrueMixedBatchInput` and the inspected predictor derived-feature branch (including integer context means and configurable rounding). They do not erase nonuniformity or turn a compound physical scope into separate measured A labels.

## Attribution, timing and exclusion

Static scope labels: `B|scope|<physical_scope>|<layer integer or model>|<variant>`. Forward labels: `B|fwd|<rank-local forward_id>`. Static scope labels are prepared at initialization. Native marker occurrence IDs differ even when labels repeat. Bind each scope marker to the unique enclosing forward on the same process/thread, then to independent invocation metadata. Every physical duration belongs to one invocation only. `scope_spec.json` records A correspondence, including one `rope_kv` observation corresponding provisionally to two A names.

Eager attribution is dispatch → native correlation → host API → process/thread nested host ranges → unique scope and matching enclosing forward. The **entire host API** must lie within the range. GPU execution can occur much later. Multiple enclosing target scopes, target plus exclusion, mismatched process IDs, missing forward parents, crossing or equal ranges, and unresolved correlations fail. b0.1 supports complete Kineto `X` events and normalized roctx push/pop events; native roctx CSV spelling/pairing is not guessed. Native graph launches are flagged and rejected. Capture/node replay correspondence remains unimplemented; repeated names/order are never replay evidence.

Every raw kernel is exported with attribution status. Allowed exclusions are explicit source-evidenced ranges/reasons (`moe`, communication, output head, sampling, final norm, scheduler support, warmup, graph capture). Do not classify an unknown kernel by name or silently exclude unclaimed work. Exclusion annotations are normalization evidence, not a requirement to add per-kernel hot-path markers. Unknown/ambiguous records fail acceptance and raw inputs remain intact. The current provisional mixed/TP scope recipe is a test specification, not verified source evidence.

For retained and excluded kernels, `kernel_dur_us=(end_ns-begin_ns)/1000`. A scope's `kernel_sum_us` is the sum over occurrences; `scope_span_us=(max(end_ns)-min(begin_ns))/1000`; `kernel_union_us` is the union length of device intervals across queues. Overlapping kernels can make sum exceed span. No equality between sum/span/gaps is assumed. Timestamp order, not trace file order, determines extrema.

`predecessor_id` is the immediately preceding observed **kernel** by begin time in `(process,GPU,queue)` across the full normalized capture, before exclusions. `queue_interval_us=(begin-current - end-previous)/1000` may be negative and is preserved for diagnosis. First observed kernels have null predecessor and interval with `predecessor_state=window_boundary`. This is not device idle or proven CPU overhead; memory copies and other queues may be active. Scope summaries never sum these intervals into a false closure claim.

## Telemetry and windows

`telemetry_b.py` is a separate CPU process reading explicit per-GPU sysfs sclk/mclk paths. Its DPM-state value is sampled context, not guaranteed instantaneous hardware frequency. Resolve paths to physical PCI/UUID identities before collection; never map by TP alone. Preserve raw text, errors, read duration, sensor resolution and gaps. Unknown sensor resolution stays null and fails quality checks. No probe kernels or clock settings are added.

Clock calibration records: `gpu_id`, `scale`, integer trace/host anchor, observed uncertainty, valid trace begin/end, evidence reference and method. Conversion uses anchor-relative arithmetic to avoid large-origin float loss. `calibrate_b.py` uses at least three independent host-bracketed native anchors; it does not prove the native host/device domains agree or guarantee drift between anchors. Verify that on the pinned tracer and retain held-out calibration evidence. Extrapolation outside the calibration interval fails.

Clock policy contains `frozen`, calibration `evidence_ref`, `max_age_ns`, `max_gap_ns`, `max_uncertainty_ns`. Freeze numeric values from a calibration subset before acceptance. The synthetic fixture limits have no live validity. A join requires before/after samples on the same GPU, records both IDs and the nearest ID, absolute sample age, read-bracket plus alignment uncertainty, sensor resolution and bracket gap. Freshness checks conservatively add age + uncertainty + update resolution. States include `valid_context`, missing device, uncalibrated, outside calibration, unbracketed, stale, sampler gap, uncertain alignment, unknown/invalid resolution and sensor error. Required target rows must have valid context; excluded rows retain diagnostic states.

Per-process capture windows contain begin/end in native trace time plus `complete`, `drained`, `metadata_complete` evidence flags. Retain failed/partial windows rather than toggling these flags to pass. Counts in `trace_diagnostics.raw_kernel_count` must be established from raw tracer evidence, independently of attribution. `dropped_records=null` means the tracer lacks a verified diagnostic; independent scope/request/count reconciliation cannot prove no interior dispatch loss. Missing rank diagnostics, truncation, overflow, or reported drops fail. Startup/capture raw traces may be stored separately; measured-window selection and boundary provenance must be explicit.

## Acceptance limits

`check_b.py` derives expected physical scope invocations from scheduler-observed phases, layers and the versioned applicability rules. It does not require invariant kernel names/counts. It reconciles every global forward across every rank, requires nonempty attributed invocations, and reconciles prompt chunks and decode steps against independent request outputs. The current first-token convention is one generated token at prefill completion and G−1 decode forwards; it must be verified on the pinned SGLang image before live use.

A green `structural_pass` validates the supplied normalized evidence contract, not numerical equivalence, low perturbation, replay coverage, completeness of the full dataset, or trustworthiness of a manually asserted manifest. Exact source instrumentation, native adapter verification, graph identities, calibrated limits, output comparisons and matched tracer controls remain necessary. No full-sweep approval or dataset acceptance is implied by any b0.1 checker result.
