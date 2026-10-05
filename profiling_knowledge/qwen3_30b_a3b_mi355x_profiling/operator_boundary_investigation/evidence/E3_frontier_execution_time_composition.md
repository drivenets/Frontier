<!-- Evidence E3 — Frontier ExecutionTime composition trace (subagent report, worktree 284130b). Generated 2026-10-04 by the operator-boundary investigation session. Treat as evidence to verify, not authority. -->

# Frontier ExecutionTime composition — code archaeology (worktree `/home/dn/Frontier-qwen3-profiling`, HEAD 284130b, read-only)

All paths below are absolute under `/home/dn/Frontier-qwen3-profiling/`. Abbreviations: `ET.py` = `frontier/entities/execution_time.py`; `TC.py` = `frontier/entities/time_components.py`; `SK.py` = `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`; `MOE.py` = `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`; `DIS.py` = `frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`; `FAM.py` = `frontier/operators/families.py`.

**Which predictor applies to Qwen3-30B-A3B co-location.** `frontier/execution_time_predictor/random_forrest_execution_time_predictor.py:22-35`: disaggregated mode → `SklearnDisaggregationExecutionTimePredictor`; else `model_config.is_moe` → `SklearnMoEExecutionTimePredictor`; else dense. Qwen3-30B-A3B (`data/config/models/qwen3-a3b-30b-moe.json`: `model_type: qwen3_moe`, `num_hidden_layers: 48`, `mlp_only_layers: []`, `num_experts: 128`, `num_experts_per_tok: 8`) is MoE, so the live path is `MOE.py::predict_stage_execution_time` → `MOE.py::_get_execution_time_internal` → `ExecutionTime(...)`. The dense predictor's `predict_stage_execution_time` (`SK.py:7080-7468`) composes identically except MLP vs MoE terms; I cover both.

---

## 1. Exact formula for one batch's model execution time

### 1a. Where the sum lives (`ET.py`)

`ET.py:1370-1393` `model_time`:
```
single_layer_block_time = self._get_block_execution_time()
total_computation_time = single_layer_block_time * self._num_layers_per_pipeline_stage
pipeline_stage_execution_time = (total_computation_time + self.pipeline_parallel_communication_time
    + self._decode_draft_proposer_time + self._mtp_terminal_overshoot_time)
return pipeline_stage_execution_time * 1e-3   # seconds
```
`ET.py:1400-1407` `total_time = self.model_time + self._get_cpu_overhead() * 1e-3`.

`ET.py:896-907` `_get_block_execution_time` (single layer, ms):
```
self._get_attention_layer_execution_time()
+ (self._get_moe_execution_time() if self._is_moe else self._get_mlp_layer_execution_time())
+ self._residual_time.add_time
```
`ET.py:884-893` `_get_attention_layer_execution_time = self._attention_time.total_time() + self._get_attn_tp_allreduce_time()`.

`TC.py:427-453` `AttentionTime.total_time()` (legacy sum; `operator_times` is `None` on the live path, see below):
```
attention_prefill + attention_decode + attention_layer_pre_proj + attention_layer_post_proj
+ attention_rope + attention_kv_cache_save + [6 attn_mla_* = 0 for dense attention] + attn_norm_time
+ attn_inter_norm_time + attn_wq_proj_time   # Step2Mini only, 0 otherwise
```

MoE FFN block, `ET.py:849-864` `_get_moe_execution_time`:
```
self._moe_or_mlp_time.total_time() + self._get_expert_parallel_communication_time()
+ self._get_moe_tp_allreduce_time() + self._get_tensor_parallel_allgather_time()
+ self._get_share_expert_tensor_parallel_allreduce_time() + self._get_dp_input_allreduce_time() + self._get_dp_output_allreduce_time()
```
`TC.py:528-548` `MoETime.total_time()` = `moe_grouped_gemm + moe_gating_linear + moe_gating_routing_topk + moe_shuffling + mlp_norm_time + share_expert_up/down/act` (share_expert = 0 for Qwen3; `operator_times` present on the MoE path but contains the same values → `legacy − covered + op_total` is numerically identical, `TC.py:544-548`).

Dense MLP block, `ET.py:837-847`: `self._moe_or_mlp_time.total_time() + self._get_moe_tp_allreduce_time()`; `TC.py:470-484` `MLPTime.total_time()` = `up_proj + down_proj + act + mlp_norm_time`.

Residual, `TC.py:669-689` `ResidualTime.add_time = add_attn_residual_time + add_ffn_residual_time`.

### 1b. Resulting closed form (co-location, MoE, dense attention, PP=1, TP=t)

With L = `num_layers_per_pipeline_stage` = `model_config.num_layers // num_pipeline_stages` (`frontier/execution_time_predictor/base_execution_time_predictor.py:37-39`; = 48 for Qwen3-30B-A3B at PP=1), and all per-op terms predicted ONCE for the batch (not per layer):

```
model_time_ms =
  L * [ input_layernorm + attn_pre_proj + attn_rope + attn_prefill + attn_decode + attn_kv_cache_save + attn_post_proj
        + attn_TP_allreduce
        + post_attention_layernorm + moe_gating_linear + moe_gating_routing_topk + moe_shuffling + moe_grouped_gemm
        + moe_TP_allreduce + EP_comm(+allgather/share-expert/DP allreduce terms, 0 for Qwen3 generic profile)
        + add_attn_residual + add_ffn_residual ]
  + pipeline_parallel_send_recv            (once per stage; 0 on last stage)
  + decode_draft_proposer_time + mtp_terminal_overshoot_time   (once; 0 without spec-decode)

total_time_ms = model_time_ms + schedule + sampler_e2e + prepare_inputs_e2e + process_model_outputs + ray_comm
                + pp_producer_send_path_runtime + pp_receiver_head_runtime + pp_prefill_consumer_active_runtime
                + pp_stage_boundary_residual_runtime      (OverheadTime.simulated_total_time, TC.py:617-630; once per stage)
```

### 1c. How the specific ops enter

| Op | Enters via | Multiplied by L? | Source of value (MoE live path) |
|---|---|---|---|
| `emb` | **Nowhere.** No `ExecutionTime` field exists; `FAM.py:296-306` registers `emb` with `predictor_target=False, e2e_trace_target=False, execution_time_attr=None`. It is trained as a model name (`SK.py:2599-2605`) but only consumed by the MTP structural-step path (`SK.py:4840-4870`, `mtp_fusion_proj`/`lm_head_linear`/norm/emb-allreduce), which is MTP-only. | n/a | — |
| final norm / lm_head / sampler GPU work | No per-forward term. `sampler_e2e_time` is a CPU-overhead lookup (`SK.py:6124-6125` → `_get_cpu_overhead_prediction_or_default`), not an lm_head GEMM. `lm_head_linear` is predicted only inside the MTP contract (`SK.py:4855-4858`). | no | cpu_overhead CSV, or 0 if missing (`SK.py:6050-6090`) |
| `attn_rope` | `AttentionTime.attention_rope_execution_time` | yes | `SK.py:5487-5497` direct lookup `_predictions["attn_rope"][(effective_tokens,)]`, no calibration |
| `attn_kv_cache_save` | `AttentionTime.attention_kv_cache_save_execution_time` | yes | `SK.py:5499-5550`; keyed on `batch.total_num_tokens` (comment at 5537-5538: "don't use round up ... exact number of tokens"), × `attn_kv_cache_save_calibration_scale` (or prefill-phase variant) |
| `input_layernorm` | `AttentionTime.attn_norm_time` | yes | `SK.py:4477-4483` `_predictions["input_layernorm"][(effective_tokens,)]` |
| `post_attention_layernorm` | `MoETime.mlp_norm_time` / `MLPTime.mlp_norm_time` | yes | `SK.py:4485-4495` `_predictions["post_attention_layernorm"][(effective_tokens,)]`; raises if `not model_config.post_attn_norm` |
| `add` | `ResidualTime.add_attn_residual_time + add_ffn_residual_time` | yes | `SK.py:4497-4505`: **returns 0.0 if `model_config.uses_fused_add_norm`**; `qwen3_moe` is in `FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST` (`frontier/config/model_config.py:191-200`), so for Qwen3-30B-A3B `add_time = 0`. Legacy `add_time` is split 50/50 into attn/ffn residual (`ET.py:273-276`) unless the architecture profile is `FFN_RESIDUAL_ONLY` (Step3 only, `frontier/model_architectures.py:218`; `MOE.py:2053-2060`). |
| `attn_pre_proj`/`attn_post_proj` | `AttentionTime.attention_layer_pre/post_proj_execution_time` | yes | `SK.py:4394-4430`, × `attn_pre_proj_calibration_scale` / `attn_post_proj_calibration_scale` (prefill-phase override if set) |
| `attn_prefill` | `AttentionTime.attention_prefill_execution_time` | yes | `SK.py:5611-5658`, keyed `(sum kv_cache, round(sqrt(sum chunk²))²)`, × `(1 + prefill_batching_overhead_fraction·[n_prefill>1])` |
| `attn_decode` | `AttentionTime.attention_decode_execution_time` | yes | `SK.py:5552-5609`, keyed `(decode_batch_size, avg_kv)`, × `(1 + decode_batching_overhead_fraction·[bs>1])` × `attn_decode_calibration_scale`; mixed prefill+decode batches (MONOLITHIC only) use on-demand `attn_decode_in_mixed` |

Note the per-layer components are deliberately kept single-layer and the layer count is applied only via `num_layers_per_pipeline_stage` (`SK.py:7418-7421` comment "Keep component fields at single-layer granularity and let ExecutionTime apply layer aggregation"; `MOE.py:2583-2588` docstring). Every `ET.py` public per-op property (e.g. `attention_rope_execution_time`, `ET.py:1058-1063`) is `single_layer × L` via `_scaled_time_attr_value` (`ET.py:579-583`), except `pipeline_parallel_communication_time` (`ET.py:1151-1157`, not scaled), overhead properties (`ET.py:1180-1240`, not scaled), and `decode_draft_proposer_time`/`mtp_terminal_overshoot_time`.

Token feature: `batch.get_effective_total_tokens_rounded(cluster_type)` → `get_effective_total_tokens_for_compute` (`frontier/entities/batch.py:867-875`, "no longer applies additional fixed multiple-of-8 rounding"; `batch.py:729-751` → for MONOLITHIC uses `decode_cuda_graph_metadata.padded_total_tokens` when `runtime_mode ∈ {FULL, PIECEWISE}` else original tokens, `batch.py:309-312`).

---

## 2. TP all-reduce: where, how many, and independence from norm/proj

**Count: exactly 2 all-reduce terms per layer** — one after attention (`_get_attn_tp_allreduce_time`, added in `_get_attention_layer_execution_time`, `ET.py:884-893`) and one after the FFN/MoE block (`_get_moe_tp_allreduce_time`, added in `_get_moe_execution_time` / `_get_mlp_layer_execution_time`, `ET.py:837-864`). Both are multiplied by L through `_get_block_execution_time`. There is no position-in-layer modelling beyond this additive sum (no overlap, no "after which kernel" ordering — it is a flat sum).

**Values (MoE path), `MOE.py:1957-1984`:**
```
if attn_tensor_parallel_size == 1: attn_tp_allreduce_time = 0
else: attn_tp_allreduce_time = self._predict_comm_operator(get_comm_operator("attn_tensor_parallel_allreduce"), batch)
...
if include_moe and moe_tensor_parallel_size > 1: moe_tp_allreduce_time = _predict_comm_operator(get_comm_operator("moe_tensor_parallel_allreduce"), batch)
elif attn_tensor_parallel_size > 1: moe_tp_allreduce_time = attn_tp_allreduce_time   # reuse, key "mlp_tensor_parallel_allreduce"
```
Dense path `SK.py:7145-7154`: a single `attn_tensor_parallel_allreduce` prediction is written to BOTH `communication_operator_times["attn_tensor_parallel_allreduce"]` and `["mlp_tensor_parallel_allreduce"]` and passed as `attn_tensor_parallel_allreduce_time = moe_tensor_parallel_allreduce_time = tp_comm_time` (`SK.py:7352-7355`, `7379-7380`).

**Payload and source:** `_predict_comm_operator` (`SK.py:5403-5471`) → `predict_allreduce_time` (`SK.py:6875-6925`) → `self._cc_backend.predict_allreduce(data_size_bytes, num_devices, cluster_type, comm_domain)`. Payload = `embedding_dim * 2 * effective_tokens` bytes (`FAM.py:309-321`, `_hidden_state_bytes`), i.e. one bf16 hidden-state tensor, with quantization size adjustment. `num_devices` = attn TP size (`FAM.py:371-372`) or moe TP size. The legacy `_get_tensor_parallel_communication_time` (`SK.py:5297-5353`) does the same via CC backend directly (`comm_domain="ATTN_TP"`). Both fail fast without a CC backend unless `--enable_dummy_mode` (`SK.py:5345-5353`). Comment `SK.py:3060`: "Communication models (all_reduce, send_recv) are no longer trained here."

**Independence / double-counting answer:** The all-reduce time comes from the CC backend (collective simulator / analytical), with NO input from the `linear_op` profiling CSV; `input_layernorm`, `post_attention_layernorm`, `attn_post_proj`, `mlp_down_proj`/`moe_grouped_gemm` come from the sklearn per-op predictions. They are summed additively (`ET.py:884-907`). Therefore: if a profiled norm (or post-proj) measurement already included a fused all-reduce+norm kernel's communication time, the collective's cost **would be counted twice** — once inside the profiled op and once as `attn/moe_tensor_parallel_allreduce_time`. Nothing in `ET.py`, `TC.py`, `SK.py` or `MOE.py` subtracts or de-duplicates between the profiled norm/projection predictions and the CC-backend all-reduce. The only subtraction on the all-reduce is launch-overhead stripping for CUDA-graph decode (`SK.py:4233-4291`, below).

Legacy single-field fallback: when neither split field is provided, `tensor_parallel_communication_time` is used for both positions (`ET.py:866-882`; `TC.py:573-590`).

---

## 3. Layer-0 / prefill-vs-decode / eager-vs-graph distinctions

**Layer 0 vs other layers:** No per-layer distinction in the composition. Dense: `SK.py:7105-7107` "Dense execution-time prediction is layer-homogeneous and ignores this value [layer_id]". MoE: `MOE.py:1939-1943` always calls `predict_attention_layer_time(batch, layer_id=0, ...)`; `layer_id` is used only (a) for `is_moe_layer(layer_id)` (mixed dense/MoE models, `MOE.py:2622-2624`; all Qwen3-30B-A3B layers are MoE) and (b) as a routing seed for per-expert token allocation (`MOE.py:1082-1127`, `_get_moe_tokens_input(..., layer_id)`). The stage scheduler passes `layer_id = 0` for co-location (`frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py:252`) and `num_layers = predictor._num_layers_per_pipeline_stage` (`:243`). No layer-0-specific cost (e.g. embedding, first-layer warm-cache) exists.

**Prefill vs decode:** Not a different composition — the same sum, with per-op values conditioned on the batch:
- `attn_prefill` = 0 if `num_prefill_tokens == 0`; `attn_decode` = 0 if no decode requests (`SK.py:6616-6625`, `5556-5558`, `5612-5620`).
- Mixed batches route decode attention to `attn_decode_in_mixed` (`SK.py:5560-5581`; MONOLITHIC only).
- Calibration scales have prefill-phase / decode-phase / late-decode / request-length variants gated on `num_prefill_tokens` (`SK.py:476-500`, `4401-4410`, `4457-4463`; `MOE.py:1414-1438`).
- Spec-decode verify tokens are routed to the prefill predictor (`SK.py:6627-6648`).

**Eager vs graph (measurement family):** `SK.py:966-995` `_select_measurement_type_for_batch`: for MONOLITHIC, any batch with `num_prefill_tokens > 0` → `CUDA_EVENT` ("eager" family); decode-only batch with `decode_cuda_graph_runtime_mode != "NONE"` → `KERNEL_ONLY`; else `CUDA_EVENT`. `_activate_measurement_type` (`SK.py:997-1004`) swaps `_predictions`/`_models` to the eager or kernel-only trained family (separate CSVs `linear_op_kernel_only.csv`, `attention_kernel_only.csv`, `moe_kernel_only.csv`, `frontier/profiling/README.md:575-586`). The composition formula is unchanged; only the per-op values differ. Additionally the all-reduce gets `estimate_intra_server_allreduce_launch_overhead_ms` subtracted when `KERNEL_ONLY` and decode-only in FULL (or pure-decode PIECEWISE) graph mode (`SK.py:4233-4291`; backend impl `frontier/cc_backend/backends/collective_sim_cc_backend.py:681-711`, only for `intra_server_model == "nvlink_analytic"` with `nvlink_allreduce_launch_overhead_us`).

---

## 4. Calibration / scaling / overhead terms

**Multiplicative calibration scales** (config fields `*_calibration_scale`, default 1.0, `SK.py:431-441`; validated > 0, `SK.py:425-429`):
- `attn_pre_proj`, `attn_post_proj`, `attn_decode`, `attn_kv_cache_save`, `mlp_up_proj`, `mlp_down_proj`, `moe_shuffling`, `moe_grouped_gemm`, `expert_parallel_communication` (`SK.py:285-316`), plus optional phase variants: `prefill_phase_*`, `decode_phase_*`, `late_decode_*`, `attn_decode_in_mixed_calibration_scale`, and decode request-length shape scales (`SK.py:502-532`, `597-720`). Applied at the per-op getter (`SK.py:4394-4466`, `5531-5549`, `5597-5608`; `MOE.py:1414-1438`, `1900-1910`, `2015-2021`).
- NOT calibrated: `attn_rope`, `input_layernorm`, `post_attention_layernorm`, `add`, `mlp_act`, `moe_gating_*`, `attn_prefill` (prefill gets only the batching-overhead fraction), TP all-reduce, PP send/recv.

**Batching-overhead fractions:** `attention_prefill/decode_batching_overhead_fraction` from predictor config, applied only when `num_q_heads > num_kv_heads` (GQA — true for Qwen3: 32 vs 4) and only when more than one request (`SK.py:275-283`, `5593-5597`, `5652-5658`).

**Quantization adjust:** dense path applies `quant_manager.adjust_compute_time(op, t, cluster_type)` to every per-op value (`SK.py:7295-7340`) and `adjust_tensor_size` to comm payloads (`FAM.py:312-317`).

**CPU/framework overhead (`OverheadTime`, `TC.py:598-640`):** `schedule_time`, `sampler_e2e_time`, `prepare_inputs_e2e_time`, `process_model_outputs_time`, `ray_comm_time`, four `pp_*_runtime_time` terms → `simulated_total_time` added once per stage in `total_time` (`ET.py:909-911`, `1400-1407`). `pp_stage_boundary_handoff_time` is diagnostic-only (exported, not simulated, `TC.py:632-636`). Values: `_get_cpu_overhead_prediction_or_default` (`SK.py:6050-6090`) — from the `cpu_overhead` CSV family keyed on `(batch_size, num_prefill_tokens, num_decode_tokens)`; returns 0.0 with a one-time warning if missing or if `config.skip_cpu_overhead_modeling`. No per-kernel `kernel_launch_overhead` term exists anywhere in `ET.py`/`TC.py`; the only "launch overhead" concept is the all-reduce strip in §3.

**Scheduler consumption:** `replica_stage_schduler.py:311-312` `BatchStage(execution_time.total_time, execution_time.model_time, ...)`; `BatchStage.execution_time` drives stage end time (`frontier/entities/batch_stage.py:161-162` asserts `time == scheduled_at + execution_time`). `base_cluster_scheduler.py:2696`, `3743-3747` recover `cpu_overhead = total_time − model_time`. `metrics_store.py:3905-3934` requires the per-component ledger to sum exactly to `total_time_ms`.

---

## 5. Schema-only vs essential fields

**Essential (enter `total_time` for the Qwen3 co-location path):** `attention_prefill`, `attention_decode`, `attention_layer_pre_proj`, `attention_layer_post_proj`, `attention_rope`, `attention_kv_cache_save`, `attn_norm_time`, `mlp_norm_time`, `moe_gating_linear`, `moe_gating_routing_topk`, `moe_shuffling`, `moe_grouped_gemm`, `attn_tensor_parallel_allreduce`, `moe_tensor_parallel_allreduce`, `expert_parallel_communication` (EP>1), `add_attn_residual`/`add_ffn_residual` (0 for Qwen3 due to fused add+norm), `pipeline_parallel_communication` (PP>1), the nine `OverheadTime` simulated fields, `decode_draft_proposer_time`, `mtp_terminal_overshoot_time`.

**Carried but zero / unused for this model:** six `attn_mla_*` (MLA only), `attn_inter_norm`/`attn_wq_proj` (Step2Mini), `share_expert_*` ×3 and `share_expert_tensor_parallel_allreduce` (Step2Mini/Step3), `tensor_parallel_allgather` (Step3 `moe_tensor_parallel_allgather_op`), `dp_input/output_allreduce` (MoE DP prefill), `mlp_layer_up/down/act` (dense FFN; 0 on MoE path, `MOE.py:1994-1996`), `expert_parallel_allgather_time` hard-coded 0.0 (`ET.py:323`), `pp_stage_boundary_handoff_time` (diagnostic only).

**Deprecated/back-compat:** `add_time`, `moe_gating_time`, `tensor_parallel_communication_time` legacy single field (`ET.py:50-54`, `273-276`, `266-270`, `302-306`); `_op_times` canonical map and `*OperatorTimes` dataclasses (`TC.py:190-390`) — a parallel "op_id → attr" representation that, when present, replaces the legacy attr value (`TC.py:448-453`) but is numerically identical on both live paths.

**Is the partition essential to simulator behaviour?** No. The simulator consumes only two scalars per stage: `total_time` and `model_time` (`replica_stage_schduler.py:311-312`; `base_cluster_scheduler.py:3744-3802` additionally reads `pipeline_time`, `decode_draft_proposer_time`, `mtp_terminal_overshoot_time` separately). `_get_block_execution_time` is a flat sum, so any re-partition of the per-layer compute (e.g. merging rope into pre_proj, or norm into add) leaves `model_time` unchanged provided the summed value is the same. The partition matters only for (a) per-op calibration scales and quantization adjustments that are keyed by op name (§4), (b) which op is zeroed by flags (`uses_fused_add_norm` → `add`; `post_attn_norm` → mlp_norm), (c) prefill/decode conditioning of attention ops, and (d) the metrics component ledger / OP-TRACE logs (`metrics_store.py:3705-3830`; `MOE.py:2660-2770`). The one structural coupling is the two all-reduce slots: they are separate additive terms, so a fused all-reduce+norm measurement cannot be expressed without either zeroing the all-reduce or double counting (§2).

---

## 6. Documented rationale for the operator partition

- `frontier/profiling/linear_op/README.md:16-31` (verbatim):
  > "The original "MLP" naming was too narrow ... The new "linear_op" naming better reflects: 1. **Broader Scope**: The module profiles not just MLP layers, but all operations with linear complexity: - MLP layers: `mlp_up_proj`, `mlp_down_proj`, `mlp_act` - Normalization: `input_layernorm`, `post_attention_layernorm` - Attention projections: `attn_pre_proj`, `attn_post_proj`, `attn_rope` - Residual connections: `add`"
  and lines 50-55: "MoE models **still need** common linear operations (LayerNorm, attention projections, residual add)".
- `frontier/config/model_config.py:385-389` (docstring):
  > "Whether the model uses fused add+norm kernel (RMSNorm). When True, input_layernorm/post_attention_layernorm time already includes residual addition, so 'add' should NOT be profiled or predicted separately."
- `frontier/profiling/linear_op/PROFILING_QA.md:83-94`: "**Q**: Does `input_layernorm` fuse residual addition? **A**: **Yes, (A) Inside the function itself** ... Calls `fused_add_rms_norm` when residual is provided - Both operations performed in single CUDA kernel".
- `SK.py:4388-4390`: `# add is only a separate operation for non-fused LayerNorm models ... return not self._model_config.uses_fused_add_norm`.
- `frontier/profiling/README.md:40-48` Key Design Principles: "1. **Single-GPU Profiling with Weight Sharding**: Profiling executes on a single GPU with TP-style weight sharding, measuring per-shard compute cost without requiring `torch.distributed`." (this is the stated reason communication is not inside profiled ops) and "3. **Architecture-Based Organization**: Profiling is organized by model architecture component (attention, linear_op, moe) rather than deployment topology."
- `docs/profiling/README.md:49-55` Operator Coverage table: `linear_op` → "Dense linear operators, projections, LayerNorm, residual add, and replicated ops."; `attention` → "Attention prefill/decode timing."; `moe` → "MoE gating, routing, shuffling, and grouped GEMM paths."
- `FAM.py:246-306` encodes the partition declaratively: norms/adds/emb are `MEMORY_FAMILY` with `tp_mode=TensorParallelMode.REPLICATED`; `frontier/model_architectures.py:150-158` generic profile `sharded_ops=("attn_pre_proj","attn_rope","attn_post_proj")`.
- `TC.py:513-516` (MoETime docstring): "expert_parallel_communication_time is NOT included here ... This maintains clear separation between compute and communication times."

No document explains why `attn_rope` and `attn_kv_cache_save` are separate ops rather than folded into projection/attention kernels, nor why exactly two all-reduces per layer are assumed, beyond the structural comments above.

---

## Could not determine / caveats
- The CC backend's internal all-reduce cost model (vidur/analytical/collective_sim) was not traced beyond the call boundary (`predict_allreduce`, launch-overhead estimator); whether it includes kernel launch cost for the eager family is backend-config dependent.
- `_get_batch_decode_attention_params` / `_get_batch_prefill_attention_params` (feature construction) were not read; irrelevant to the composition formula.
- Dense-path double-write of one all-reduce prediction into both comm slots (`SK.py:7145-7154`) vs MoE-path separate `MOE_TP`-domain prediction is a behavioural difference between predictors when `attn_tp ≠ moe_tp`; for Qwen3 co-location the MoE path applies.
- `DIS.py` was only surveyed (`DIS.py:1151-1200` docstring: "PREFILL: Full model (attention + MLP/MoE) - DECODE_ATTN: Attention only - DECODE_FFN: MLP/MoE only ... ExecutionTime applies num_layers_per_pipeline_stage aggregation"); it builds `ExecutionTime` with cluster-filtered zero components and is not on the co-location path.
