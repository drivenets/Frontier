<!-- Evidence E2 — Frontier sklearn predictor training/selection trace for the nine labels (subagent report, worktree 284130b). Generated 2026-10-04 by the operator-boundary investigation session. Treat as evidence to verify, not authority. -->

# Frontier sklearn predictor: training/selection trace for 9 operator labels (Qwen3-30B-A3B, co-location = ClusterType.MONOLITHIC)

Worktree: /home/dn/Frontier-qwen3-profiling @ 284130b. Read-only; no files modified. Abbreviations: SEP = frontier/execution_time_predictor/sklearn_execution_time_predictor.py, SPM = frontier/execution_time_predictor/shared_prediction_model_manager.py, PM = frontier/attention/profiling_mapping.py, AT = frontier/training/attention_trainer.py, LT = frontier/training/linear_op_trainer.py.

## 0. Context that applies to every label

- **Model-config mapping for Qwen3 JSON** (frontier/config/model_config.py:616-635): `num_q_heads=num_attention_heads(32)`, `num_kv_heads=num_key_value_heads(4)`, `embedding_dim=hidden_size(2048)`, `use_qk_norm` explicit `true` (model_config.py:223-226). Qwen3 binds to `DENSE_ATTENTION_FAMILY` variant `gqa` (frontier/attention/model_binding.py:166-180: `1 < num_kv_heads < num_q_heads` -> "gqa"). Architecture profile is GENERIC (frontier/model_architectures.py:144-161): `sharded_ops=("attn_pre_proj","attn_rope","attn_post_proj")`, no `replicated_ops`, no `predictor_attention_extra_ops`.
- **Estimator**: RF default: `RandomForestRegressor(random_state=0)` grid `n_estimators∈{250,500,750}`, `max_depth∈{8,16,32}` (random_forrest_execution_time_predictor.py:64-72; config.py:2964-2971). Linear alternative: `make_pipeline(PolynomialFeatures(), LinearRegression())`, degree 1..5 grid (linear_regression_execution_time_predictor.py:67-76; config.py:2940-2955). Training is GridSearchCV with MAPE scorer, cv = `k_fold_cv_splits` (default 10), **no train/test split** (SEP:2823-2843: "we don't create a train/test split ... we only want to predict execution time within the same domain"). NaN rows in features/target are dropped before fit (SEP:2795-2812). Models are cached by name+hash (`_store_model_in_cache`, SEP:2755). Same logic in SPM `_train_single_model` (SPM:2057-2210), which also stamps `_frontier_feature_names` (SPM:2201).
- **Two measurement families**: eager (`MeasurementType.CUDA_EVENT`) and kernel_only. For MONOLITHIC, kernel_only is enabled only if `decode_cuda_graph_mode != "none"` (SEP:896-908, 928-964). Per-batch selection in MONOLITHIC (SEP:966-995): any `num_prefill_tokens>0` -> CUDA_EVENT; decode-only with cuda-graph runtime_mode != NONE -> KERNEL_ONLY; else CUDA_EVENT. Each family has its own `_models`/`_predictions` (SEP:360-420). Kernel-only CSVs are separate files (`linear_op_kernel_only_input_file`, `atten_kernel_only_input_file`, SPM:349-351; SEP:775-854).
- **Metadata validation on load**: every CSV must have single-valued `profiling_precision`, `model_arch`, `quant_signature` (SEP:2102-2192); quant mismatch is logged, not fatal (SEP:2192).
- **Per-layer**: all nine predictions are **per-layer, per-batch** quantities (ms); none is multiplied by num_layers inside this predictor layer. `predict_attention_layer_time` (SEP:6552-6761) just sums them into `AttentionTime`. The ×num_layers happens elsewhere (trace_mapping.py:44-47 multiplies `op_times * execution_time.num_layers` for E2E trace only).

## 1. Summary table

| label | CSV / column / row filter | training features (feature engineering) | model name(s) cached | prediction-side feature source |
|---|---|---|---|---|
| attn_pre_proj | linear_op.csv, `time_stats.attn_pre_proj.median`; rows: n_head==32 & n_kv_head==4 & n_embd==2048 & n_expanded_embd==mlp_hidden_dim & use_gated_mlp & vocab_size & `num_tensor_parallel_workers==attn TP (policy)` & `use_qk_norm==True` (SEP:1112-1131) | `["num_tokens"]` raw, no transform (SEP:3015) | `attn_pre_proj` | `batch.get_effective_total_tokens_rounded(cluster)` -> lookup `(tokens,)` (SEP:4394-4411) |
| attn_rope | same file/filter, `time_stats.attn_rope.median` | `["num_tokens"]` | `attn_rope` | same effective tokens (SEP:5487-5497) |
| attn_post_proj | same file/filter, `time_stats.attn_post_proj.median` | `["num_tokens"]` | `attn_post_proj` | same (SEP:4413-4430) |
| input_layernorm | linear_op.csv, `time_stats.input_layernorm.median`; same model filter but **TP key forced to 1** (replicated) (SEP:2637-2651) | `["num_tokens"]` | `input_layernorm` | same effective tokens (SEP:4477-4483) |
| post_attention_layernorm | linear_op.csv, `time_stats.post_attention_layernorm.median`; TP key 1 | `["num_tokens"]` | `post_attention_layernorm` | same (SEP:4485-4495); gated on `model_config.post_attn_norm` |
| emb | linear_op.csv, `time_stats.emb.median`; TP key 1 | `["num_tokens"]` | `emb` (trained; **prediction cache built only when target-embedded MTP enabled**) | only via MTP structural path `_get_named_linear_op_execution_time(op_name="emb", num_tokens=total_forward_tokens)` (SEP:4833) |
| attn_kv_cache_save | attention.csv, `time_stats.attn_kv_cache_save.median` (NaN/missing -> 0!) ; rows: n_embd==2048 & n_q_head==32 & n_kv_head==4 & block_size==cfg & `num_tensor_parallel_workers==effective attn TP` — **all rows incl. decode, prefill, mixed, true-mixed** (SEP:1200-1229, 3019-3056) | shared contract `["total_tokens","kv_cache_size","batch_size"]` (PM:296-300); legacy/standalone contract `["num_tokens"]` where num_tokens=max(prefill_chunk_size,batch_size) (PM:257; AT:261) | `attn_kv_cache_save` (one name, two possible feature shapes) | 3-feature: `total_tokens=batch.total_num_tokens`, `batch_size=len(batch.requests)`, `kv_cache_size`= decode avg rounded (0 if no decode) (SEP:5521-5531); 1-feature: lookup `(batch.total_num_tokens,)` unrounded (SEP:5543-5544) |
| attn_prefill | attention.csv, `time_stats.attn_prefill.median`; same structural filter, then `~is_true_mixed_batch & ~is_decode & prefill_chunk_size>0` (SEP:3189-3193, 3226) | `["kv_cache_size","prefill_chunk_size_squared"]` (PM:301-304); squared = `prefill_chunk_size**2` (SEP:2332) | `attn_prefill`; plus `attn_prefill_mixed` (12 features, target same column) | per-request `(kv_cache_size=ceil64(num_processed_tokens), chunk=batch.num_tokens[i])` for requests with `not _is_prefill_complete`; aggregate `sum(kv)`, `round(sqrt(sum chunk²))²` (SEP:3969-3995, 5648-5657). If >1 prefill request and `attn_prefill_mixed` exists -> 12-feature on-demand (SEP:5636-5641) |
| attn_decode | attention.csv, `time_stats.attn_decode.median`; same structural filter, then `~is_true_mixed_batch & is_decode` (SEP:3193, 3239-3262) | `["batch_size","kv_cache_size"]` (PM:305) | `attn_decode`; plus `attn_decode_in_mixed` (7 features, MONOLITHIC only) | `(decode_batch_size, decode_avg_kv_cache_size)` from completed-prefill requests, avg kv rounded up to granularity 64 (SEP:3911-3948). If `batch.num_prefill_tokens>0` -> `attn_decode_in_mixed` on-demand (SEP:5559-5578) |

## 2. Linear-op labels (attn_pre_proj, attn_rope, attn_post_proj, input_layernorm, post_attention_layernorm, emb)

### (a) CSV and filter
`_load_compute_df` (SEP:1081-1165) reads `self._compute_input_file` (linear_op.csv or kernel-only variant), `drop_duplicates()`, then:
```
df = df[(df["n_head"] == self._model_config.num_q_heads) & (df["n_kv_head"] == self._model_config.num_kv_heads)
      & (df["n_embd"] == self._model_config.embedding_dim) & (df["n_expanded_embd"] == self._model_config.mlp_hidden_dim)
      & (df["use_gated_mlp"] == ...) & (df["vocab_size"] == ...) & (df["num_tensor_parallel_workers"] == tensor_parallel_size)]
```
(SEP:1112-1122). Then `use_qk_norm` filter is mandatory for Qwen3 (`use_qk_norm=True`): raises if column missing, else `df[df["use_qk_norm"].astype(bool) == expected_use_qk_norm]` (SEP:1124-1131). `attn_output_gate` filter applies only if column present (SEP:1133-1150). Note: `n_expanded_embd` must equal `mlp_hidden_dim` — for Qwen3-MoE this is whatever `BaseModelConfig` derives (not verified which of `intermediate_size=6144` / `moe_intermediate_size=768` it is; flag).

**TP key per op** — `_get_linear_op_tp_key` (SEP:2636-2687): replicated set = `get_family_profiling_name_set(MEMORY_FAMILY)` = {input_layernorm, post_attention_layernorm, add, emb} (operators/families.py:248-305) plus profile `replicated_ops` (empty for GENERIC) -> **TP=1** for norms/emb. `attn_*` ops -> `resolve_effective_attention_tp_size(op_name, requested_tp, num_kv_heads=4, include_linear_ops=True)` (SEP:2676-2683). Each op loads its own TP slice via `_get_compute_df_for_tp(tp_key)` (SEP:2954-2961, 2977). Same mapping in SPM:611-665 and LT:296-340, AT:407-429.

**TP policy** (attention_tp_policy.py:66-102): there is **no TP>1 -> TP1 fallback anymore**. TP=1 returns 1; supported TP (`num_kv_heads % tp == 0` or `tp % num_kv_heads == 0`) is returned as-is; unsupported TP raises ValueError for non-linear ops (kv_cache_save/prefill/decode) always, and for linear attn ops only when `include_linear_ops=True` (which the linear-op path sets). Docstring: "Unsupported TP settings are surfaced as explicit errors instead of silently falling back to TP=1." For Qwen3 (4 kv heads) TP∈{1,2,4,8,...} are all valid.

### (b) Features / model
`_train_compute_models` loop (SEP:2975-3017): `target_col = f"time_stats.{model_name}.median"`, `feature_cols=["num_tokens"]`, no engineering. Missing column -> warning + skip (SEP:3005-3010) except for profile-required ops; all-NaN -> raise (SEP:3011-3014). Model names = `_get_compute_model_names()` (SEP:2598-2621): `["emb","attn_pre_proj","attn_post_proj","input_layernorm","post_attention_layernorm","attn_rope"]` + dense FFN ops only if `_requires_dense_mlp_compute_models()` (false for Qwen3: all 48 layers MoE, `mlp_only_layers=[]`) + `add` only if not fused-add-norm (RMSNorm -> fused -> no `add`, model_config.py:385-399).

Shared-manager path trains the same ops with the same `["num_tokens"]` from `_load_linear_op_df` (SPM:1250-1317; filter there is **only** `num_tensor_parallel_workers==tp` plus `use_qk_norm` — SPM:2273-2300 — no n_head/n_embd filter; the structural filter is in the sklearn-predictor path only). Standalone LT (LT:238-246, 266-273) and AT (AT:51-57, 511-514) also use `["num_tokens"]`.

### (c) Distinct models
One model per label: `attn_pre_proj`, `attn_rope`, `attn_post_proj`, `input_layernorm`, `post_attention_layernorm`, `emb`. No mixed variants.

### (d) Prediction-side lookup
`_predict_for_compute_models` (SEP:3480-3618) precomputes a lookup table over `num_tokens = 1.._max_tokens` (SEP:3577-3594) for every single-feature model: `predictions[name][(tokens,)]`. Getters:
- `_get_attention_layer_pre_proj_execution_time` (SEP:4394-4411) / `_post_proj` (4413-4430): `effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)`; `raw_time = self._predictions["attn_pre_proj"][(effective_tokens,)]`; then ×`prefill_phase_attn_pre_proj_calibration_scale` if `batch.num_prefill_tokens>0` and configured, else ×`attn_pre_proj_calibration_scale` (default 1.0; config.py:2276+).
- `_get_attention_rope_execution_time` (SEP:5487-5497): same effective tokens, **no calibration scale**.
- `_get_attn_norm_layer_act_execution_time` (SEP:4477-4483) -> `input_layernorm`; `_get_mlp_norm_layer_act_execution_time` (SEP:4485-4495) -> `post_attention_layernorm` (raises if `not model_config.post_attn_norm`; RMSNorm -> True, model_config.py:629). No scale.
- `get_effective_total_tokens_rounded` (frontier/entities/batch.py:867-875) **no longer rounds to multiple of 8**; it returns `get_effective_total_tokens_for_compute(cluster)` (batch.py:729-797): for MONOLITHIC with `decode_cuda_graph_metadata` -> `padded_total_tokens` if runtime_mode FULL/PIECEWISE (batch.py:309-312); with target-embedded MTP spec metadata -> total + planned drafts; otherwise plain `self._total_num_tokens` (= sum of `batch.num_tokens`, prefill+decode tokens together, batch.py:709-710). So a mixed batch is NOT split for linear ops; one lookup at total tokens. No layer-0 special-casing; TP enters only via training-row filter.

### (e) emb
operators/families.py:294-304: `OperatorSpec(name="emb", role=EMBEDDING, ..., predictor_target=False, e2e_trace_target=False, execution_time_attr=None, tp_mode=REPLICATED)`. However `_get_compute_model_names` hardcodes `"emb"` (SEP:2600) so it **is trained** (if column present) at TP=1, and LT requires `time_stats.emb.median` (LT:201). On the prediction side `_predict_for_compute_models` adds `emb` to the lookup-table build **only when `_requires_target_embedded_mtp_compute_models()`** (SEP:3562-3568); otherwise no `predictions["emb"]` is built and nothing reads it. The sole consumer is the MTP structural step (SEP:4833: `predictor._get_named_linear_op_execution_time(op_name="emb", num_tokens=total_forward_tokens)`). `predict_attention_layer_time` does not include emb. Conclusion: for Qwen3-30B-A3B without target-embedded MTP, **emb is trained but never predicted/used**.

## 3. attn_kv_cache_save

### (a) CSV/filter
`_load_attention_df` (SEP:1182-1229): `pd.read_csv` + `drop_duplicates`; `enforce_mixed_attention_input_contract` (attention_dataset_contract.py:35-75: if `attention_mixed.csv`/`attention_true_mixed.csv`/`attention_combined.csv` exist beside the input, the input must contain one of `is_mixed_batch`/`is_true_mixed_batch`/`total_tokens` or raise). Then:
```
for column in [cache_write_median_column]:
    if column not in df.columns: df[column] = 0
    else: df.fillna({column: 0}, inplace=True)
```
(SEP:1204-1208) — **missing/NaN kv_cache_save targets become 0 and are trained on**. `effective_tp = resolve_effective_attention_tp_size(op_name="attn_prefill", requested_tp=attn_tp, num_kv_heads, include_linear_ops=False)` (SEP:1210-1221). Filter: `n_embd==2048 & n_q_head==32 & n_kv_head==4 & block_size==self._block_size & num_tensor_parallel_workers==effective_tp` (SEP:1223-1229). No `is_prefill`/`batch_size` filter; training df for kv_cache_save = **entire filtered attention df** (SEP:3049-3054), i.e. prefill, decode, mixed-prefill and true-mixed rows together. (AT standalone instead uses `standard_df` = non-mixed rows, AT:656-659.) Identical filter in SPM:2451-2457 and AT:230-234 (AT uses raw `self.tensor_parallel_size`, no policy call).

### (b)/(c) Two feature contracts, same model name
- PM:290-320 `get_enabled_shared_predictor_feature_columns`: `CACHE_WRITE: ("total_tokens","kv_cache_size","batch_size")`. Used by SEP `_train_compute_models` (SEP:3031-3056, raises "Re-run attention profiling with mixed-batch metadata" if columns absent) and SPM:1444-1468.
- PM:251-275 `get_enabled_predictor_feature_columns`: `CACHE_WRITE: ("num_tokens",)`. Used only by the standalone AT (`DENSE_LAYER_FEATURE_COLUMNS`, AT:65, 519) where `num_tokens = max(prefill_chunk_size, batch_size)` (AT:261).
- Note: `_train_attention_layer_models` (SEP:3185+) does **not** train kv_cache_save; it is trained in `_train_compute_models` (SEP:3019-3056). Model name is always `attn_kv_cache_save`.

### (d) Prediction
`_predict_for_compute_models` includes the cache-write op in attention-cluster names (SEP:3493-3500). If `n_features_in_==1` -> lookup table over num_tokens (SEP:3589-3594); else on-demand dict with `_feature_names` (SEP:3596-3616). `_get_attention_kv_cache_save_execution_time` (SEP:5499-5550):
```
total_tokens = batch.total_num_tokens; batch_size = len(batch.requests); kv_cache_size = 0
if batch.num_decode_tokens > 0: _, kv_cache_size = self._get_batch_decode_attention_params(batch)
```
(SEP:5521-5531) -> `_get_on_demand_prediction("attn_kv_cache_save", features)`. Legacy 1-feature: `prediction_info[(batch.total_num_tokens,)]` with comment "don't use round up to the nearest multiple of 8 here" (SEP:5541-5544). Both paths then × `prefill_phase_attn_kv_cache_save_calibration_scale` (if prefill tokens and configured) else × `attn_kv_cache_save_calibration_scale`. Spec-decode piecewise may force CUDA_EVENT family for this op (SEP:6629-6636). **Semantic mismatch to flag**: training `kv_cache_size` column is the profiler's row-level kv context (for prefill rows, the prefix length), but prediction uses the decode-side average KV and 0 for pure prefill batches; `batch_size` at training is the profiler's `batch_size` column vs `len(batch.requests)` at prediction.

## 4. attn_prefill (+ attn_prefill_mixed)

### (a) Training rows
`_train_attention_layer_models` (SEP:3185-3469), eager family only (`need_prefill` requires CUDA_EVENT, SEP:3721-3724; kernel-only branch trains no prefill model, SEP:3426-3469). Derived features (SEP:2309-2537): `num_tokens=max(prefill_chunk_size,batch_size)`; `is_decode = ~coerce_truthy_bool(is_prefill)` if column exists else `prefill_chunk_size==0`; `prefill_chunk_size_squared`; bool-normalised `is_mixed_batch`, `is_true_mixed_batch` (default False). Split:
```
true_mixed_df = attention_df[attention_df["is_true_mixed_batch"]]; standard_df = attention_df[~...]
prefill_df = standard_df[~standard_df["is_decode"]]; decode_df = standard_df[standard_df["is_decode"]]
standard_prefill_df = prefill_df[prefill_df["prefill_chunk_size"] > 0]
```
(SEP:3189-3193, 3226). No `batch_size==1` filter at training (the comment at SEP:3785 "PREFILL training data uses batch_size=1" is an assumption about the data, not a filter) — mixed-prefill rows with `batch_size>1` and `prefill_chunk_size>0` are included in `attn_prefill`. SPM has the same split (SPM:1472-1480).

### (b) Features
`attn_prefill`: `["kv_cache_size","prefill_chunk_size_squared"]`, target `time_stats.attn_prefill.median` (PM:301-304; SEP:3232-3237). 
`attn_prefill_mixed`: 12 features `avg_seq_len, batch_cv_interaction, batch_size, batch_variance_interaction, kv_cache_size, max_seq_len, min_seq_len, seq_len_cv, seq_len_range, seq_len_variance, total_tokens, total_tokens_squared` (SEP:3270-3283), same target. Training rows = `prefill_df[is_mixed_batch | batch_size>1]` ∪ true-mixed rows with `time_stats.attn_prefill.median` notna, whose features are remapped from `prefill_mixed_*` columns computed per row from `prefill_seq_lens`/`prefill_kv_cache_sizes` lists (SEP:2399-2480, 3285-3372). Derived: `total_tokens_squared`, `seq_len_range=max-min`, `batch_variance_interaction=batch_size*seq_len_variance`, `batch_cv_interaction=batch_size*seq_len_cv` (SEP:2483-2510). SPM trains attn_prefill_mixed from `prefill_df[is_mixed_batch | batch_size>1]` only (SPM:1577-1605).

### (c) Prediction cache / selection
`_predict_for_attention_layer_models` (SEP:3771-3797): grid `kv_cache_size ∈ range(0, max_tokens+1, 64)` × `prefill_chunk_size ∈ 1..4096` squared, where `max_tokens = max(prediction_max_tokens_per_request(4096), max_model_len/max_position_embeddings)` (SEP:3873-3884; for Qwen3 262144 -> very large grid). `attn_prefill_mixed` stored for on-demand (SEP:3801-3836).

### (d) Batch -> features
`_get_batch_prefill_attention_params` (SEP:3969-3995): for each `(request, num_tokens_to_process) in zip(batch.requests, batch.num_tokens)` with `not request._is_prefill_complete`: `prefill_chunk_size = num_tokens_to_process` (this step's tokens, unpadded), `kv_cache_size = ceil(request.num_processed_tokens / 64) * 64`. Decode requests excluded -> this is the prefill-part of a mixed batch. `_get_attention_prefill_execution_time` (SEP:5611-5658): empty -> 0.0 if `num_prefill_tokens==0` else raise. **Selection rule**: `if len(prefill_params) > 1 and "attn_prefill_mixed" in self._predictions` -> `_get_batch_prefill_mixed_features` (SEP:3997-4087: seq_lens = per-prefill-request `num_tokens`, kv = `num_processed_tokens` avg rounded to 64; stats as above) -> on-demand, **no batching overhead and no calibration applied**. Otherwise single-model lookup: `agg_kv = sum(kv_cache_sizes)`, `agg_chunk = sqrt(sum(chunk²))`, key `(agg_kv, round(agg_chunk)**2)`, × `(1 + attention_prefill_batching_overhead_fraction(0.1) * int(len>1))` (SEP:5648-5657). No attn_prefill calibration scale in this method. `_supports_operation("attn_prefill")` is PREFILL/MONOLITHIC only (SEP:4331-4332).

## 5. attn_decode (+ attn_decode_in_mixed)

### (a)/(b) Training
`attn_decode`: rows `decode_df` (standard, is_decode), features `["batch_size","kv_cache_size"]`, target `time_stats.attn_decode.median` (SEP:3239-3262 eager; SEP:3431-3440 kernel-only where it is mandatory). Skipped with an info log if `decode_df` empty or columns missing (eager). 
`attn_decode_in_mixed`: rows `true_mixed_df`, 7 features `decode_batch_size, decode_avg_kv_cache_size, num_prefill_seqs, total_prefill_tokens, total_batch_size, batch_composition_ratio, total_tokens` (SEP:3391-3406; also trained in kernel-only branch SEP:3442-3469). Derived: `total_batch_size=num_prefill_seqs+num_decode_seqs`, `batch_composition_ratio=num_prefill_seqs/total_batch_size`, `decode_batch_size=num_decode_seqs` if absent (SEP:2512-2537). Requires `decode_avg_kv_cache_size` and `total_prefill_tokens` to be present in the CSV (not derived).

### (c) Prediction cache
Grid `batch_size ∈ 1..prediction_max_batch_size(128)` × `kv_cache_size ∈ range(0, max_tokens+1, 64)` (SEP:3747-3769). `attn_decode_in_mixed` on-demand, registered **only for MONOLITHIC** (SEP:3838-3869).

### (d) Batch -> features
`_get_batch_decode_attention_params` (SEP:3911-3948): for requests with `_is_prefill_complete`, kv = `_get_decode_attention_context_tokens(request)` = `num_processed_tokens` (+ `num_emitted_decode_tokens` when `num_processed_decode_tokens==0`, SEP:3950-3967); `decode_batch_size = batch.get_effective_decode_batch_size_for_attention()` (cuda-graph padded batch size in FULL mode, batch.py:314-317, 877-891), `decode_avg_kv_cache_size = ceil64(mean(kv))`. `_get_attention_decode_execution_time` (SEP:5552-5609): 0.0 if no decode requests. **Mixed rule**: `if batch.num_prefill_tokens > 0`: cluster must be MONOLITHIC (else raise), `attn_decode_in_mixed` must exist (else raise with "Please provide merged attention profiling data via atten_input_file (Option A)"), features from `_get_batch_decode_mixed_features` (SEP:4089-4137: decode kv = `num_processed_tokens` of completed requests, `total_prefill_tokens=sum(prefill num_tokens)`, `total_tokens=batch.total_num_tokens`) -> on-demand × optional `attn_decode_in_mixed_calibration_scale`. Otherwise lookup `(decode_batch_size, decode_avg_kv_cache_size)` × `(1 + attention_decode_batching_overhead_fraction(0.1) * int(bs>1))` × (`late_decode_attn_decode_calibration_scale` or `attn_decode_calibration_scale`).

## 6. How the predictor decides standard vs mixed
- Prefill: count of prefill requests `>1` AND `attn_prefill_mixed` in predictions (SEP:5636-5641). A pure-prefill batch of 2 requests therefore uses the mixed model even though no decode is present.
- Decode: `batch.num_prefill_tokens > 0` (true prefill+decode) -> `attn_decode_in_mixed`, mandatory in MONOLITHIC (SEP:5559-5578).
- Measurement family: `_select_measurement_type_for_batch` (SEP:966-995) picks eager whenever prefill tokens exist, so true-mixed batches always use eager `_predictions`.

## 7. Operators composed (naming only)
`predict_attention_layer_time` (SEP:6552-6761) calls `_get_attention_prefill_execution_time`, `_get_attention_decode_execution_time`, `_get_attention_layer_pre_proj_execution_time`, `_get_attention_layer_post_proj_execution_time`, `_get_attention_rope_execution_time`, `_get_attention_kv_cache_save_execution_time`, `_get_attn_norm_layer_act_execution_time` and returns `AttentionTime(...)` (SEP:6747-6761). `post_attention_layernorm` is returned via `_get_mlp_norm_layer_act_execution_time`, consumed in the MoE subclass (sklearn_moe_execution_time_predictor.py:2131, 2401-2407). The MoE subclass overrides none of the nine getters (grep of sklearn_moe_execution_time_predictor.py shows only call sites, lines 113, 525, 561, 1938, 2131, 2405); its `_train_models` just adds MoE models (moe:987-998), and `_predict_for_compute_models` adds prefill-hot gating models (moe:1000-1012).

## 8. Flags / could not determine
1. `n_expanded_embd == model_config.mlp_hidden_dim` filter (SEP:1116): which Qwen3-MoE value `mlp_hidden_dim` resolves to (6144 vs 768) was not traced; a mismatch would zero out all linear-op rows.
2. kv_cache_save: NaN/missing targets are filled with 0 before training (SEP:1204-1208) — rows without kv_cache_save timing contribute zero-time samples.
3. kv_cache_save feature semantics differ between training (`kv_cache_size`, `batch_size` columns as profiled) and prediction (decode-only avg KV, 0 for pure prefill; `len(batch.requests)`).
4. `attn_prefill` training includes mixed-prefill rows with `batch_size>1` (only `is_true_mixed_batch` and `prefill_chunk_size>0` are excluded); the `batch_size=1` assumption is a comment, not a filter.
5. Prefill/decode grids use `max(4096, max_position_embeddings)`; for Qwen3's 262144 this is a 4097×4096 (prefill) and 128×4097 (decode) lookup — not a correctness issue but worth knowing for cache behaviour.
6. `attn_rope`, `input_layernorm`, `post_attention_layernorm` have no calibration scale; `attn_pre_proj`/`attn_post_proj`/`attn_kv_cache_save`/`attn_decode` do (defaults 1.0 in config.py:2276+; not verified individually).
7. The SPM path (`_load_linear_op_df`, SPM:2221-2316) filters linear_op.csv only by TP and `use_qk_norm`, not by n_head/n_embd/vocab — if the shared manager is used with a multi-model linear_op.csv, rows from other models would leak in. Whether the simulation uses SPM or the per-predictor path depends on `model_manager` being passed (SEP:375-388); not determined from this read.
