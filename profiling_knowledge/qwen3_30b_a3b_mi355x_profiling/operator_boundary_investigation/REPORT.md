# Frontier operator boundaries for Qwen3-30B-A3B on SGLang / AMD MI355X — investigation report

Written 2026-10-04 against worktree `/home/dn/Frontier-qwen3-profiling` @ `284130b` (branch `smatar/qwen3-30b-mi355-profiling`).
Scope: the nine Dataset-A operator labels, their Frontier consumers, and their counterparts in the approved serving image.
Nothing outside this directory was modified; no GPU job, server, pull, or install was run. Supporting evidence lives in
[`evidence/`](evidence/) (E1–E7); file:line references below point at the worktree for Frontier code and at SGLang tag
`v0.5.18` (commit `71de97b2`) for serving code, which E5 shows is byte-identical to the approved image's installed wheel for
every SGLang file cited here except `aiter_backend.py` (cited with **image** line numbers, marked "img").

Evidence classes used throughout: **[observed]** = read from a runtime artifact (registry, image layer, cluster file);
**[source]** = established by reading pinned source under stated configuration; **[inferred]** = rationale reconstructed from
code, no author statement; **[unresolved]** = cannot be settled without a live run or missing data.

---

## 1. Executive conclusion

**The inherited partition is Vidur's token-level / sequence-level / communication split, instrumented with Sarathi-Serve's
vLLM-era scope names, and it was never re-derived for SGLang on ROCm.** That history is verified (E4, E7): all nine labels, the
two-CSV schema, the `reduce_results=False` exclusion of collectives, the single-GPU sharded profiling, the attention
prefill/decode split, the QK-norm-inside-`attn_pre_proj` rule ("matching vLLM's attn_pre_proj scope"), and even the double
embedding call are in Vidur or in Frontier upstream's first public commit (2026-06-08). The only local changes are kernel
providers (AITER attention wrapper, vLLM fused RoPE op) and measurement hygiene; no local commit touched a boundary.

**Most of the partition still fits.** Six of the nine units remain the right prediction targets for SGLang/ROCm because their
boundaries coincide with SGLang's own call structure and their feature dependence is unchanged: `attn_pre_proj` (QKV GEMM +
QK-norm), `attn_post_proj`, `input_layernorm`, `post_attention_layernorm`, `attn_prefill`, `attn_decode`.

**Three findings are consequential and change what should be measured or how it is consumed:**

1. **`attn_rope` and `attn_kv_cache_save` are one kernel on this stack, in every regime.** On HIP SGLang passes a
   `fused_set_kv_buffer_arg` to the rotary embedding unconditionally (`srt/models/utils.py:284-305`, gate `_is_hip and not
   prefill-CP and no dcp mask`; no flag exists), which runs the Triton kernel `fused_qk_rope_reshape_and_cache` (RoPE on q and k
   plus the K and V writes into the paged pool; `rotary_embedding/base.py:388-418`, `kernels/ops/kvcache/rope_cache.py:546-591`)
   and then the attention backend is told `save_kv_cache=False` (`qwen3_moe.py:683-694`). The previous agent's lost finding is
   therefore **re-established by source** [source]. Two independent simulator terms do not describe this; they must become one
   unit, and the sum of A's two labels is a hypothesis about its cost, not its value.
2. **The serving configuration, not the image, decides the kernel provider for five of the nine units.** `SGLANG_USE_AITER`
   defaults to False (`srt/environ.py:746`) and the approved image does not set it [observed, E5 §1]; DriveNets' own MI355X
   recipes are split between `SGLANG_USE_AITER=1 --attention-backend aiter` and `--attention-backend triton` with no AITER
   env [observed, E5 §6], and none is a Qwen3-30B-A3B recipe. With AITER off, the image has no vLLM (E5 §5), so every RMSNorm
   (both hidden-state norms **and** q/k norms) runs torch-native multi-kernel code and GEMMs run `F.linear`; with AITER on they
   run aiter `rmsnorm2d_fwd[_with_add]` and `tgemm.mm`. A's labels were measured with a third provider set (vLLM 0.9.2 kernels,
   hipBLASLt 1.0.0). **This is the single largest open decision and it needs the user**, because every boundary verdict for the
   linear/norm units is "keep the boundary, recollect under the chosen provider".
3. **Frontier's composition cannot absorb a fused all-reduce+norm without double counting, and it currently models no
   once-per-forward work.** Two CC-backend all-reduce terms are added per layer independently of the norm predictions
   (`frontier/entities/execution_time.py:884-907`, E3 §2); if `--enable-aiter-allreduce-fusion` is used (default off, requires
   AITER), the fused kernel's communication would be counted twice. `emb` is trained but never enters `ExecutionTime`
   (`frontier/operators/families.py:294-304`, E3 §1c), and final norm / lm_head / sampler GPU work are unmodeled.

Lesser but real: A's harness has three implementation defects that are not boundary issues but make specific labels
unrepresentative — `input_layernorm` is timed with `x` and `residual` aliased to the same tensor (`linear_op_impl.py:1093,
919-922`), standard attention rows give every sequence the same block table (`attention_wrapper.py:213`, inherited from
upstream Frontier, not Vidur) so batched decode reads one shared KV region and decode cache-writes hit one slot, and
`attn_kv_cache_save` is two `index_put_` scatters that match neither SGLang's fused path nor its unfused `store_cache` kernel.
A-attention was also never GPU-bound-fixed (E1 §B.0), so its small-shape labels are host spans.

Uncertainty: no kernel trace of the approved image was taken in this task. Kernel *counts* inside each unit (e.g. the extra
output-fill kernel DriveNets patched into the aiter prefill path, E5 §2.1) and the exact hipBLASLt/tgemm tile behaviour are
[source]-level claims to be confirmed by the one-forward trace proposed in §10.3.

---

## 2. The target stack, verified

| item | finding | class |
|---|---|---|
| Image tag / digest | `dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04` = `sha256:d8e0eb9c3211b593fbd574ada06e03d48d956c159bb9b6c123f4d86c12d8e6dc` (registry `10.1.1.0:5050`, created 2026-09-23T11:15:32Z). Not present on any node reachable with this account; the registry copy was inspected layer-by-layer without pulling. | observed |
| SGLang version | wheel `sglang-0.5.18` built from `SGLANG_REF=v0.5.18` = upstream commit `71de97b2`, plus DriveNets patches: 24 modified + 19 new files, almost all "Saguaro" speculative decoding; **one operator-relevant patch in `srt/layers/attention/aiter_backend.py`** (opt-in unified prefill behind `SGLANG_AITER_UNIFIED_PREFILL`; a zero-filled caller-owned output buffer before the BF16 multi-token prefill kernel; page-table transform for page size 1 in graph metadata). Historical note "0.5.18" is confirmed. | observed |
| aiter | `amd_aiter 0.1.1.dev1+gb4d9154d1.d20260923` (commit `b4d9154d1`), prebaked JIT modules. A-linear used aiter `a6bb49937` (+ vLLM 0.9.2 kernels, hipBLASLt 1.0.0, ROCm 7.0); A-attention used aiter `d9e5ef7ce` (ROCm 7.2.0, hipBLASLt 1.2.1); the image has ROCm 7.2.3, hipBLASLt 1.2.2, torch 2.12. Three different kernel stacks. | observed |
| Default attention backend on HIP (MHA) | `aiter` (`srt/server_args.py:5917-5918`) | source |
| `SGLANG_USE_AITER` | default False (`environ.py:746`); gates GEMM (`unquant.py:65,235-236,265`), RMSNorm (`layernorm.py:53,110-125,456-469`), rope module flag (`rotary_embedding/base.py:36`, unused for the fused path), memory pool, communicator fusion. **Not set by the image ENV**; vLLM absent → without it norms are torch-native. | observed + source |
| Fused RoPE + KV write | unconditional on HIP for non-MRoPE models (`models/utils.py:284-305`, `qwen3_moe.py:515-517, 641-650`) | source |
| Fused QK-norm + RoPE | CUDA-only and default-off (`qwen3_moe.py:522-532` requires `_is_cuda`; `server_args.py:1976-1980` default False) | source |
| All-reduce + norm fusion | `--enable-aiter-allreduce-fusion` default False (`server_args.py:2051-2055`); requires AITER; auto-enable lines are commented out (`:5480, :5577`) | source |
| Mixed prefill/decode batches | only with `--enable-mixed-chunk` (default False, `server_args.py:981-985`); then decode rows become 1-token extends inside the **prefill kernel** (`schedule_batch.py:2749-2775`, `base_attn_backend.py:227-250`, E6 §1) | source |
| Graphs | decode-only by default on HIP; prefill graph backend `tc_piecewise` is auto-disabled on non-CUDA → EXTEND/MIXED eager (`cuda_graph_config.py:109-128`, `server_args.py:4548-4557,4604-4607,4662-4664`); decode batches padded to capture buckets `[1,2,4,8,12]+range(16,257,8)+…`; padded rows run attention over 1 KV token (E6 §2) | source |
| Chunked prefill default | 16384 tokens for the ≥160 GB tier (`server_args.py:4877-4884`); the tier helper was not inspected | source / unresolved |
| Production recipe for this model | **none exists on the cluster.** Closest DriveNets recipes (`/opt/shared/bp-inferencex/benchmarks/single_node/`) are Qwen3.5 scripts that disagree: `fixed_seq_len/qwen3.5_bf16_mi355x.sh` = `--attention-backend triton --enable-aiter-allreduce-fusion --disable-radix-cache`, no AITER env; `agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh` = `SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 --attention-backend aiter --page-size 16 --chunked-prefill-size 32768 --speculative-algorithm EAGLE`. | observed |

Two configuration **forks** therefore have to be fixed by the user before any unit's kernel identity is final:
**F1** `SGLANG_USE_AITER` on/off (changes GEMM and all norm kernels), **F2** attention backend `aiter` (A's kernel family) vs
`triton` (SGLang's `extend_attention_fwd` / split-KV `decode_attention_fwd`, `triton_backend.py:143-161`), plus the secondary
knobs `--enable-aiter-allreduce-fusion`, `--enable-mixed-chunk`, `--page-size` (1 is SGLang's default and A's block-1 rows;
the qwen3.5 recipe uses 16), `--chunked-prefill-size`, graph capture sizes. This report's verdicts are written for the
**default-flag path** (aiter attention, page 1, no mixed chunk, no AR fusion, decode graphs on) and state how each fork changes them.

---

## 3. Reconstructed design rationale

### 3.1 Documented intent (quoted; E7, E4)

- **Vidur paper §4.3** (the only design document): operators fall in three buckets — *token-level* ("linear, and activation
  functions … runtime only depends on the total number of tokens being processed (prefill plus decode)"), *sequence-level*
  ("attention … depends … also [on] the context length of each request"; "separately profile the attention kernels for prefill
  and decode"; decode modeled on "total KV-Cache reads" because PagedAttention v2 / FlashDecoding handle skew), and
  *communication* ("profile these kernels … in a model-agnostic manner for different topologies"). Token-level ops are profiled
  "on a single GPU" over "all the different tensor parallel sharding configurations" with "standard PyTorch kernels"; RF
  regression is chosen because of "tile and wave quantization". Target engine/hardware: Sarathi-Serve (vLLM lineage) on
  A100/H100 (`vidur/docs/profiling.md`; paper §5).
- **Scope names** are Sarathi-Serve's `OperationMetrics` enum (`sarathi/metrics/constants.py:12-30`), which also names
  `ATTN_POST_PROJ_ALL_REDUCE`, `ATTN_PRE_PROJ_ALL_GATHER`, `EMBED_ALL_REDUCE`: communication was a separate operator by design,
  which is why `reduce_results=False` on `o_proj`, `mlp_down_proj` and `emb` (`vidur/profiling/mlp/mlp_impl.py:50,90,110`).
- **Frontier upstream** adds only: "Linear operations are characterized by having linear complexity with respect to sequence
  length" (`frontier/profiling/linear_op/README.md:14-31`); "When QK-norm is enabled, we use an external timer to wrap QKV
  projection + QK-norm together (matching vLLM's attn_pre_proj scope)" (`linear_op_impl.py:430-431`, also `:458-459, :502,
  :518`); "When True [fused add+norm], input_layernorm/post_attention_layernorm time already includes residual addition, so
  'add' should NOT be profiled or predicted separately" (`frontier/config/model_config.py:385-389`); "Single-GPU Profiling with
  Weight Sharding … without requiring torch.distributed" (`frontier/profiling/README.md:40-48`); the operator-family registry
  (`frontier/operators/families.py`: `MEMORY_FAMILY` = norms, adds, emb, `tp_mode=REPLICATED`; `emb` `predictor_target=False`).
- **Local (DriveNets) intent**: the AITER wrapper docstring, "Real sglang serving on MI355X does not run SDPA — it dispatches
  AMD's aiter kernels. This backend calls the same two entry points sglang's own aiter_backend.py uses"
  (`frontier/profiling/attention/backends/aiter_attention_wrapper.py:11-20`, commit `645c98a`); the RoPE fix "the same kernel
  family SGLang launches on this image" (`4b8fce3`). Kernel-provider matching, not boundary matching, was the local goal.

### 3.2 Inferred rationale (no author statement found)

- `attn_rope` and `attn_kv_cache_save` are separate because in Sarathi/vLLM they were separate kernels in separate modules
  (rotary op in the model layer; `reshape_and_cache` in the attention wrapper) and because the KV write lives in the attention
  CSV keyed by attention shapes while RoPE lives in the linear CSV keyed by `num_tokens`. The split follows *instrumentation
  convenience* and the *engine of the day*, not a measured need for two regressors.
- The two-all-reduce-per-layer structure (`vidur/entities/execution_time.py:59-85`, kept in `frontier/entities/execution_time.py`)
  mirrors Megatron-style TP in vLLM/Sarathi: one after `o_proj`, one after the FFN; collectives come from a separate profile.
- The shared block table for all sequences (upstream Frontier replaced Vidur's random tables) is most plausibly a
  reproducibility/bounds-safety change; it has no documented rationale and it changes memory traffic.

### 3.3 Provenance summary (E4)

| boundary / choice | origin |
|---|---|
| nine labels, two CSVs, `reduce_results=False`, single-GPU sharded profiling, prefill/decode split, RF models | Vidur + Sarathi-Serve (vLLM/NVIDIA era) |
| QK-norm inside `attn_pre_proj`; `add` suppressed for fused-add-norm models; mixed-batch models (`attn_prefill_mixed`, `attn_decode_in_mixed`); operator-family registry; `emb` not a predictor target | Frontier upstream, present in or before the first public commit |
| shared block table in attention profiling; double `emb` call | upstream Frontier (shared table) / Vidur (double call) |
| AITER attention wrapper; vLLM fused RoPE op in `attn_rope`; GPU-bound timing, clock probe, settle | DriveNets (local), no boundary change |

### 3.4 The contract, end to end (E1–E3)

```
profiler scope ──► CSV label ──► consumer filter ──► features ──► model ──► prediction lookup ──► composition
linear_op: CudaTimer(name)   time_stats.<op>.median   n_head/n_kv_head/n_embd/…/use_qk_norm/TP   num_tokens        RF   per batch: effective total tokens   AttentionTime/MoETime fields
attention: CudaTimer(name)   time_stats.<op>.median   n_embd/n_q_head/n_kv_head/block_size/TP     per-op (see §6)  RF   per batch: prefill part / decode part   AttentionTime fields
```
Composition (`execution_time.py:1370-1393, 896-907, 884-893, 849-864`; `time_components.py:427-453`):

```
model_time = L × [ input_layernorm + attn_pre_proj + attn_rope + attn_prefill + attn_decode + attn_kv_cache_save + attn_post_proj
                   + attn_TP_allreduce(CC backend)
                   + post_attention_layernorm + MoE terms + moe_TP_allreduce(CC backend) + add(=0, fused-add-norm) ]
             + PP send/recv + spec-decode terms ;   total = model_time + CPU-overhead terms
```
with L = 48, every per-op term predicted **once per batch** for a representative layer (no layer-0 case: `MOE.py:1939-1943`
always passes `layer_id=0`; `SK.py:7105-7107` "layer-homogeneous"), `emb` absent, and the two all-reduces independent of the
norm predictions. The simulator consumes only `total_time` and `model_time` (`replica_stage_schduler.py:311-312`), so **the
partition is a data schema plus per-op calibration hooks, not a structural requirement** — any re-partition that preserves
the per-layer sum is behaviour-preserving, except where terms are conditionally zeroed (prefill/decode, fused-add-norm) or
where a composite would overlap the all-reduce slots.

---

## 4. Verdict matrix

Columns: A scope (what A's timer contains) → target behaviour on the default-flag path → recommendation → applicability →
evidence → what it means for A-vs-B comparability.

| # | A scope (timer contents) | Target (SGLang v0.5.18 / ROCm, default flags) | Recommendation | Applies | Evidence | A/B comparability |
|---|---|---|---|---|---|---|
| 1 | `attn_pre_proj`: fused QKV GEMM [T,2048]×[2048,N], N=4096/TP+256·max(1,4/TP); `.split` views; q `.view`→vLLM `rms_norm`; k same (5 kernels: GEMM, copy, norm, copy, norm) | `qkv_proj` GEMM (`tgemm.mm` if AITER else `F.linear`) → `apply_qk_norm`: `_reshape_for_qk_norm` copy + RMSNorm ×2 (aiter `rmsnorm2d_fwd` or torch-native); fused QK-norm+RoPE is CUDA-only; no alt stream on HIP | **Keep boundary** (GEMM + QK-norm, token-level, per TP). Add provider variant tag. Record GEMM and norm kernels separately inside the unit in B so a future split is derivable. | all phases, TP 1/2/4/8, eager + graph | `linear_op_impl.py:503-526`; `qwen3_moe.py:592-594, 626-636`; `models/utils.py:433-453, 486-518`; `unquant.py:235-265`; `layernorm.py:110-125, 456-469, 588-663, 665-703` | same boundary; **different providers** (hipBLASLt 1.0.0/vLLM norm vs 1.2.2/`F.linear` or tgemm/aiter norm). A usable as historical reference only; tile-switch points will move; recollect under the chosen provider. |
| 2 | `attn_rope`: one in-place vLLM `rotary_embedding` over q and k, neox, positions 0..T-1 | `fused_qk_rope_reshape_and_cache` (Triton): RoPE q,k **and** K/V paged write; backend then skips its own write | **Merge** with #7 into `attn_rope_kv_write` (token-level, per TP; writes 2·kv_heads·128·T values). Keep an unfused RoPE-only measurement only as a control. | all phases incl. graph; always on HIP (no flag) | `qwen3_moe.py:637-652, 683-694`; `models/utils.py:284-305, 308-367`; `rotary_embedding/base.py:103-124, 371-434`; `rope_cache.py:546-591` | **not comparable**: different physical unit. A rope ≈ SGLang's *unfused fallback* (`sgl_kernel.rotary_embedding`) which the default path never runs. Σ(A rope + A kv_save) is a hypothesis, not the fused cost. Recollection (B or standalone fused-kernel harness) required. |
| 3 | `attn_post_proj`: one GEMM [T,4096/TP]×[4096/TP,2048], input `randn_like(q)`, no all-reduce | `o_proj` with `reduce_results=False` → GEMM only; AR issued later in `prepare_mlp` | **Keep boundary**; provider variant as #1 | all phases, TP 1/2/4/8 | `linear_op_impl.py:473-482, 546-547`; `tensor_parallel_layers.py:968-990`; `qwen3_moe.py:496-505, 695`; `communicator.py:1155-1181` | same boundary; provider differs → historical reference; recollect under chosen provider. |
| 4 | `input_layernorm`: vLLM `fused_add_rms_norm(x, residual)` with **x is residual** (aliased); 1 kernel | layers 1..47: `rmsnorm2d_fwd_with_add` (AITER) or torch-native (several kernels) on distinct tensors, preceded at TP>1 by the post-MoE all-reduce (separate RCCL kernel; or fused AR+add+norm if `--enable-aiter-allreduce-fusion`); layer 0: norm only, no residual | **Keep boundary** (add+norm, token-level, replicated) **but replace the harness implementation** (distinct x/residual). Treat layer 0 as a documented 1/48 approximation or add a `norm_only` variant. Add a conditional composite `ar_add_norm` for the AR-fusion configuration that **replaces** both the norm term and the preceding all-reduce term. | all phases; TP-independent op; variants: L0, AR-fusion | `linear_op_impl.py:784-793, 919-922, 1093`; `layernorm.py:48-58` (Frontier); `communicator.py:619-620, 669-670, 722-726, 576-586, 614-617`; `layernorm.py:217-245, 646-659` (SGLang) | same op; A's aliasing understates memory traffic (reads one tensor, not two); provider differs → recollect. TP1-only labels are legitimately TP-agnostic. |
| 5 | `post_attention_layernorm`: `fused_add_rms_norm(o_proj_out, residual)`; 1 kernel | TP1: `post_attention_layernorm(hidden, residual)`; TP>1: `attention_tensor_model_parallel_all_reduce` then norm (or fused AR+norm when enabled) | **Keep boundary**; same AR-fusion composite rule as #4 | all phases; TP-independent | `linear_op_impl.py:802-815, 937-941`; `communicator.py:997-1025, 1075-1077, 1155-1181` | same op; provider differs → recollect; A labels fine as reference. |
| 6 | `emb`: `F.embedding` TP1 gather of random ids; **timed twice per forward** (label = per-call median) | once per forward, first PP rank only; TP1 gather; TP>1 mask + gather + `masked_fill_` + all-reduce (or fused Triton mask-gather + AR) | **Mark unsupported pending a decision**: Frontier never consumes it (`predictor_target=False`, no `ExecutionTime` field). If once-per-forward work is to be modeled, introduce a `per_forward_fixed` unit (embedding [+ AR at TP>1], final norm, lm_head, sampler) — not ×48. Drop from B's mandatory scope or keep as one cheap occurrence. | once per forward; TP changes the op | `linear_op_impl.py:458-483, 1089-1092`; `operators/families.py:294-304`; `SK.py:3562-3568, 4833`; `qwen2_moe.py:886-895, 939-948`; `vocab_parallel_embedding.py:250-260, 538-579` | A TP1 gather is a valid reference for the TP1 op; irrelevant to current predictions; TP>1 not covered by A. |
| 7 | `attn_kv_cache_save`: two `index_put_` scatters into `(2, blocks, block, kv_heads, 128)`; standard decode rows write **one shared slot**; prefill batch 1 | fused into #2 on every path; the unfused `set_kv_buffer` → `store_cache` JIT kernel only when the fusion gate is off (prefill-CP / dcp mask), never on this stack | **Merge** into `attn_rope_kv_write` (see #2); remove the independent term and both of its feature contracts for this stack | — | `aiter_attention_wrapper.py:394-401, 267-271, 285-287`; `attention_wrapper.py:213`; `profiling_mapping.py:251-275, 290-320`; `SK.py:1204-1208, 5499-5550`; `memory_pool.py:141-167, 2327-2403` | **not comparable** (different op, degenerate slot pattern, host-bound loop). A labels: historical only. |
| 8 | `attn_prefill`: one `mha_batch_prefill_func` (CK), batch 1, `chunk` queries over `kv+chunk` paged keys, causal; true-mixed rows: separate prefill kernel over 1–2 prefill seqs | `forward_extend` → same entry point over the **whole extend batch** (variable prefix/extend per sequence), page 1, nhd; **plus** on this image a zero-fill of the output buffer for multi-token queries; chunked prefill = same kernel with longer `kv_indptr`; MIXED (if enabled) = decode rows as 1-token extends **inside this kernel**; always eager on HIP | **Keep as the prefill unit**, add the image-specific fill kernel to the unit, and make the mixed behaviour **conditional**: with `--enable-mixed-chunk` the decode rows belong to this unit and `attn_decode_in_mixed` must be 0. Feature contract per sequence `(q_len, kv_len)`; Frontier's multi-sequence aggregation (Σkv, √Σchunk², +10 %) is a heuristic to validate, not a kernel fact. | EXTEND/MIXED forwards, eager | `aiter_attention_wrapper.py:403-421`; `aiter_backend.py img:2521-2574, 1939-1953`; `forward_batch_info.py:127-135`; `schedule_batch.py:2749-2775`; `SK.py:3969-3995, 5611-5658` | same kernel entry point (different aiter commit); A valid as reference for ≥~1k-token shapes; small shapes host-bound; block-16 rows have no counterpart at page 1; true-mixed rows have **no SGLang counterpart** (SGLang mixed is one kernel). |
| 9 | `attn_decode`: one `paged_attention_ragged`, uniform kv per seq, **shared pages**, `max_num_partitions = ceil((kv+1)/256)` per shape | `forward_decode` → same entry point, `max_num_partitions = ceil(context_len/256)` **server-wide**, real distinct pages, padded graph batch (padded rows attend 1 token), `o` allocated (no kernel), under HIP-graph replay by default | **Keep as the decode unit**, **replace the harness** (distinct pages per sequence, server-wide partition count or record it), key on padded batch size (Frontier already does in FULL graph mode). Note Frontier needs an `attention_kernel_only.csv` family for graph-mode decode that does not exist for this model. | DECODE forwards, graph or eager | `aiter_attention_wrapper.py:423-446, 316-318`; `aiter_backend.py img:280-293, 2705-2713, 2772-2792`; `decode_cuda_graph_runner.py:1312-1315`; `batch.py:877-891`; `SK.py:966-995, 3911-3948, 5552-5609` | same kernel; A's shared pages plausibly understate bandwidth-bound large-batch cost and its small shapes are host spans → recollect; partition-count difference recorded. |

---

## 5. Per-operator detail

Each subsection: what A times → consumers → SGLang path → unit assessment → recommendation and consumer/schema change →
validation if static evidence is insufficient. Numbers from A are cited only where they bear on a decision.

### 5.1 `attn_pre_proj`

**A** (`linear_op_impl.py:503-526`, E1 §A.1): `ColumnParallelLinear` GEMM with one fused weight `[N,2048]`, N = 5120/2560/1280/768 at
TP 1/2/4/8 (TP8 carries the replicated KV head, `:39-52`), bias None; `qkv.split` views; `q.view(T,-1,128)` → `QKNorm` → vLLM
`rms_norm` (new output), same for k. Trace-confirmed five kernels (two copies). Excluded: all-gather (`gather_output=False`),
RoPE, the `randn_like` attention stand-in. GEMM dispatch on HIP goes through vLLM 0.9.2's `dispatch_unquantized_gemm`
(`tensor_parallel_layers.py:204-226`) — skinny-GEMM special cases for small M are plausible but not verified [unresolved].

**Consumers** (E2 §2): one RF per TP on `num_tokens`; filter requires `use_qk_norm == True`; prediction key =
`batch.get_effective_total_tokens_rounded` (prefill+decode tokens summed; padded total in graph modes); ×
`attn_pre_proj_calibration_scale`. The new regressor bench consumes the same label per TP.

**SGLang** (`qwen3_moe.py:592-594, 626-636`): `qkv_proj` (`QKVParallelLinear`, same N formula: `:463-478`) → `UnquantizedLinearMethod.apply`
→ `tgemm.mm` (AITER on) or `F.linear` (`unquant.py:235-265`) → `apply_qk_norm` (`models/utils.py:456-521`): the fused in-place
QK-norm kernel is gated `_is_cuda` (`:487`), the alt-stream branch only runs under capture and `alt_stream` is `None` on HIP
(`qwen3_moe.py:920`), so the executed path is `_reshape_for_qk_norm` (`x.reshape(-1, head_dim)`, a copy for the strided split view,
docstring `:441-445`) + `RMSNorm` ×2. With AITER: `forward_aiter` → `rmsnorm2d_fwd` (`layernorm.py:660`); without: `forward_hip` →
no vLLM → `forward_native` (`:672-674, 729-770`: fp32 cast, `pow`, `mean`, `rsqrt`, multiply, cast — ≥6 small kernels per norm).
`q_norm/k_norm` weight dtype must equal activation dtype for the aiter path (`:601-604`), else native [unresolved for loaded weights].

**Assessment**: the boundary is coherent — GEMM and QK-norm are both token-level, always adjacent, never fused with anything else
on this version, and the simulator's single `num_tokens` lookup fits. Keeping QK-norm inside also keeps the TP-dependence
honest (norm width scales with heads/TP). Splitting would only pay off if a future ROCm build fuses QK-norm with RoPE
(`fused_qk_norm_rope` exists for CUDA at `qwen3_moe.py:600-625`); B should therefore keep kernel-level rows inside the unit so
`GEMM` and `norms` can be re-grouped later without recollection.

**Recommendation**: keep; add `gemm_provider ∈ {torch_hipblaslt, aiter_tgemm}` and `norm_provider ∈ {aiter, native}` as
collection metadata, not as separate operators. A's labels: historical/control reference; recollect under the chosen F1
setting (hipBLASLt 1.0.0 → 1.2.2 alone moves tile-switch token counts — the +81 % step at TP4 8192→8208 is a library fact, not a
hardware fact). Consumer change: none structurally; the kernel-only family (`linear_op_kernel_only.csv`) is what graph-mode decode
batches read (`SK.py:966-995`) and does not exist for this model yet.

### 5.2 `attn_rope` (and 5.7 `attn_kv_cache_save`)

**A rope** (`linear_op_impl.py:543-544`; `common/layers/rotary_embedding.py:251-302`): one `vllm._custom_ops.rotary_embedding(positions,
q, k, 128, cos_sin_cache, is_neox=True)` in place over q `[T,4096/TP]` and k `[T,128·max(1,4/TP)]`, positions `0..T-1`
(`utils/__init__.py:239`). Consumer: RF per TP on `num_tokens`, no calibration (`SK.py:5487-5497`).

**A kv save** (`aiter_attention_wrapper.py:394-401`): `key_cache[blk, off] = key; value_cache[…] = value` — two `index_put_` into a
`(2, blocks, block_size, kv_heads, 128)` randn cache; write plan `processed..processed+chunk` for prefill, `context_len-1` for
decode; because standard rows share one block table, **all B decode sequences write the same slot** (E1 §B.7). Consumers: the
predictor trains it on **all** attention rows with NaN→0 (`SK.py:1204-1208, 3019-3056`) using either the shared 3-feature contract
`(total_tokens, kv_cache_size, batch_size)` (`profiling_mapping.py:290-320`) or the legacy `num_tokens` (`:251-275`), with a
training/prediction semantic mismatch on `kv_cache_size` (prefix at training vs decode-average at prediction, E2 §3d).

**SGLang** (default path, every regime): `apply_qk_norm_rope` builds `create_fused_set_kv_buffer_arg(value=v, layer, forward_batch)`
when `enable_fused_set_kv_buffer(forward_batch) and self.compatible_with_fused_kv_buffer` (`qwen3_moe.py:641-650`); the gate is
`(_is_hip and not is_prefill_context_parallel_enabled() and dcp_kv_mask is None)` (`models/utils.py:302-304`) — **no server flag
or env var disables it**. The ROCm arg (`:328-367`) views the NHD pool as `(-1, page_size, kv_heads, 128)` and passes `v`,
`slot_mapping=out_cache_loc`, scales. `RotaryEmbedding.forward` dispatches `forward_hip → forward_cuda` (`kernels/fused_op.py:147-149`);
on HIP `use_fallback_kernel=True` (`rotary_embedding/base.py:103-124`), and `fused_set_kv_buffer_arg is not None and _is_hip` selects
`fused_qk_rope_reshape_and_cache(q, k, pos, cos_sin, is_neox, flash_layout=True, q_out=q, k_out=k, **arg)` (`:388-418`) whose
docstring is "Perform RoPE on q and k … and copy k and v in to key_cache and value_cache inplace" (`rope_cache.py:567-591`).
`forward_core` then computes `save_kv_cache = must_save_kv or not (enable_fused_set_kv_buffer and compatible)` = False
(`qwen3_moe.py:683-687`), so `forward_extend`/`forward_decode` skip their `set_kv_buffer` branches (`aiter_backend.py img:2054-2130,
2594-2665`). Under decode graph replay the same kernel is captured. Only if the gate were off would the unfused path run:
`token_to_kv_pool.set_kv_buffer` → `_set_kv_buffer_impl` → JIT `store_cache` kernel (`memory_pool.py:141-167`), not torch scatters.

**Assessment**: on this stack there is one physical, token-level unit: *RoPE(q,k) + KV write*. Its cost depends on `T`, TP (q/k head
counts), KV dtype and layout — not on batch composition, context length or phase — which is exactly the `num_tokens`-per-TP
feature the linear CSV already uses. Two independent regressors (one on `num_tokens`, one on attention-shape features) cannot
represent it; the attention-CSV placement of the KV write is purely historical. The unfused measurements remain useful as a
*control*: A's rope kernel is the same family as SGLang's unfused fallback (`sgl_kernel.rotary_embedding`), so a RoPE-only
baseline exists if ever needed, but it does not represent deployed execution.

**Recommendation**: merge into `attn_rope_kv_write` (features `num_tokens`, TP; applicability all phases/graph modes; variant
`fusion=on` default, `fusion=off` control-only). Consumer changes: `AttentionTime.attention_rope_execution_time` carries the fused
prediction; `attention_kv_cache_save_execution_time` → 0 and its getter/feature contracts (`SK.py:5499-5550`,
`profiling_mapping.py` `CACHE_WRITE`) become inapplicable for this stack; `frontier/attention/families.py:54-60` role `CACHE_WRITE`
must be marked absorbed; `linear_op` schema gains `time_stats.attn_rope_kv_write.*` (or B supplies it). Data: A's `attn_rope` and
`attn_kv_cache_save` are **not comparable** and the fused unit **requires recollection** (in-situ B, or a standalone harness that
calls `fused_qk_rope_reshape_and_cache` with a real paged pool and distinct slots). Do not divide a measured fused duration back
into "rope" and "save" parts.

**Validation**: trace one eager forward (§10.3) to confirm exactly one Triton kernel between the k-norm and the attention kernel
per layer; measure fused vs A-sum at matched `T` (hypothesis: fused < sum, one pass over q,k,v).

### 5.3 `attn_post_proj`

**A**: one GEMM `[T,4096/TP]×[4096/TP,2048]` on `randn_like(q)`, `reduce_results=False`, no bias (`linear_op_impl.py:473-482, 546-547`;
`tensor_parallel_layers.py:968-990`). Consumer: RF per TP on `num_tokens` × `attn_post_proj_calibration_scale`.

**SGLang**: `o_proj` `RowParallelLinear(..., reduce_results=False)` (`qwen3_moe.py:496-505`); `RowParallelLinear` only reduces when
`reduce_results and tp_size > 1` (`linear.py:1641-1644`); the all-reduce happens in `prepare_mlp` (`communicator.py:1155-1181`).
Provider per F1 as in 5.1.

**Assessment/recommendation**: keep. A's labels: historical reference; recollect under F1. Consumer: none. This is the unit most
sensitive to hipBLASLt heuristics (TP4/TP8 steps), so B should record the GEMM kernel name per occurrence to attribute steps to
library tile choice rather than to the model.

### 5.4 `input_layernorm` and 5.5 `post_attention_layernorm`

**A**: inner `RMSNorm._norm_timer` around vLLM `fused_add_rms_norm` (`layernorm.py:48-58`); `input_layernorm` is called with
`residual = hidden_states` **the same tensor object** (`linear_op_impl.py:1093, 919-922`) — an in-place op on aliased inputs that
reads one tensor instead of two (lower traffic, cache-friendly) and produces a numerically meaningless residual;
`post_attention_layernorm` uses distinct tensors (`:937-941`). Both run at every TP but TP>1 rows are dropped by the CSV writer
(`main.py:588-610, 750-752`; `utils/replicated_ops.py`). `add` has no timer (`:765-767`). Consumers: RF on `num_tokens` with TP key
forced to 1 (`SK.py:2636-2687`), applied to every TP; no calibration; `add` predicted 0 for `qwen3_moe` (`SK.py:4497-4505`;
`model_config.py:191-200`).

**SGLang**: `prepare_attn`: layer 0 `residual = hidden; hidden = input_layernorm(hidden)` (norm only, `communicator.py:619-620,
669-670`; `Qwen2MoeModel.forward` passes `residual=None`, `qwen2_moe.py:939-948`); layers 1..47 `input_layernorm(hidden, residual)`
(`:722-726`). `prepare_mlp` TP1 `_simple` = `post_attention_layernorm(hidden, residual)` (`:1075-1077`); TP>1
`attention_tensor_model_parallel_all_reduce(hidden)` then the norm (`:1155-1181`), or `forward_with_allreduce_fusion` when
`apply_aiter_all_reduce_fusion` (`:182-194`: AITER + `enable_aiter_allreduce_fusion` + size limits). The post-MoE all-reduce is
inside the MoE block (`qwen3_moe.py:331-343`) unless deferred into the next layer's `prepare_attn` as a fused AR+add+norm
(`:852-868`, `:576-586`). Kernels: AITER `rmsnorm2d_fwd_with_add` with two `empty_like` allocations (`layernorm.py:646-659`), or
torch-native (`:729-770`); fp32-weight mismatch falls back to native (`:601-604`).

**Assessment**: the "fused add + norm" unit is right for layers 1..47 at TP1 and for TP>1 without AR fusion. Layer 0 is a
different (cheaper) kernel; with 48 layers the bias of ignoring it is ≤ 1/48 of the norm term — acceptable to document rather
than model, but B should label it. With AR fusion on, the unit becomes `ar + add + norm`, and Frontier's independent CC-backend
all-reduce slot for that position must be suppressed, otherwise the collective is counted twice (E3 §2: "Nothing … subtracts or
de-duplicates between the profiled norm/projection predictions and the CC-backend all-reduce").

**Recommendation**: keep both units; **replace the `input_layernorm` harness** so `x` and `residual` are distinct (this changes the
measured value, so A's `input_layernorm` labels are a biased reference); introduce `variant=norm_only` for layer 0 (optional) and a
conditional composite `ar_add_norm` (TP>1, AR fusion on) that replaces the norm term **and** the matching `attn_tp_allreduce_time` /
`moe_tp_allreduce_time`. Consumer change for the composite: zero the CC-backend term for that slot (`MOE.py:1957-1984`) when the
composite is active. Recollect under F1; provider decides kernel count (1 vs ≥6).

### 5.6 `emb`

**A**: `F.embedding` of `T` random ids from `[151936/TP, 2048]`, TP1 rows only (TP>1 dropped), called twice per forward
(`linear_op_impl.py:1089, 1092`; inherited from Vidur, E7 §2); label = per-call median over 2× samples, so not double-counted but the
second call re-reads rows just touched by the first (cache-warm bias, [inferred]).

**Consumers**: trained (`SK.py:2598-2605`) but the prediction table is built only for target-embedded MTP (`SK.py:3562-3568`) and the
only reader is the MTP structural path (`:4833`); `ExecutionTime` has no field (`operators/families.py:294-304`, E3 §1c). The new
regressor bench fits it (TP1).

**SGLang**: once per forward on the first PP rank when `input_embeds is None` (`qwen2_moe.py:939-948`); TP1 plain gather; TP>1
`get_masked_input_and_mask` + gather + `masked_fill_` + `tensor_model_parallel_all_reduce` (`vocab_parallel_embedding.py:149-163,
538-579`), optionally a fused Triton mask-gather (`:508-525`) — still followed by the all-reduce. Not replicated for Qwen
(`SGLANG_ENABLE_EMBED_REPLICATION` is DeepSeek-only).

**Assessment/recommendation**: as a modeling unit `emb` is a once-per-forward term, not a per-layer one, and today it is dead. Either
drop it from the required set or create a `per_forward_fixed` unit covering embedding (+ its TP>1 all-reduce), final norm, lm_head
GEMM and sampler — all currently unmodeled GPU work that matters most for small decode batches. That is a coverage gap to flag, not a
boundary fix. A TP1 labels are an adequate reference for the TP1 gather; TP>1 is uncovered. No recollection priority.

### 5.8 `attn_prefill`

**A**: `mha_batch_prefill_func(q[:n_prefill], K, V, cu_seqlens_q, kv_indptr, kv_pages, max_q, max_kv, scale, causal=True, out=…,
kv_last_page_lens)` at batch 1, queries = `prefill_chunk_size`, keys = `kv_cache_size + chunk` (`aiter_attention_wrapper.py:403-421`;
`attention_input.py:15-16`); true-mixed rows run this kernel over 1–2 prefill sequences and `paged_attention_ragged` over the decode
sequences as two separately timed kernels in one forward (E1 §B "True-mixed"). Legacy 3+50 loop, never GPU-bound-fixed (E1 §B.0).
Consumers: `attn_prefill` RF on `(kv_cache_size, prefill_chunk_size²)` over standard prefill rows (which include `batch_size>1`
mixed-prefill rows if present, E2 §4a); `attn_prefill_mixed` (12 features) chosen whenever more than one request is prefilling
(`SK.py:5636-5641`); single-model aggregation Σkv, `round(√Σchunk²)²`, ×(1+0.1) for multiple prefills (`:5648-5657`).

**SGLang** (img): `forward_extend` legacy NHD path → `_legacy_prefill_explicit_output_buffer` (`img:1939-1953`: `out.zero_()` on the
pre-sliced `forward_batch._attn_output`, set by `radix_attention.py:364/597`, or `q.new_zeros` — **an extra fill kernel for
`max_q_len > 1`**, DriveNets patch) → `mha_batch_prefill_func(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, max_q_len,
max_kv_len, causal=True, …)` over the whole extend batch (`img:2546-2566`). `qo_indptr = cumsum(seq_lens − prefix_lens)`,
`kv_indptr = cumsum(seq_lens)` (E6 §1) — one call for first chunks, later chunks, radix hits and MIXED batches alike. Prefill
graphs are disabled on HIP by default (E6 §2), so this unit is always eager. F2 = `triton` would replace it with
`extend_attention_fwd` (two-stage Triton kernel).

**Assessment**: the sequence-level prefill unit stands. Three refinements: (a) the unit on this image is *fill + prefill kernel*, and
the fill scales with `T·heads·128` — token-level inside a sequence-level unit; keep it inside (it is launched unconditionally for
multi-token extends) and record it; (b) the feature key is a *list* of `(q_len, kv_len)` per sequence; A's batch-1 grid is a valid
sample of single-sequence cost, but the multi-sequence aggregation in Frontier is a heuristic that B can test directly; (c) mixed:
Frontier's `attn_prefill_mixed`/`attn_decode_in_mixed` encode A's two-kernel true-mixed structure, which SGLang never produces —
by default decode and prefill are separate forwards; with `--enable-mixed-chunk` the decode rows ride inside this kernel as
1-token extends and `attn_decode_in_mixed` must contribute 0. The block-16 rows of A have no counterpart at page size 1.

**Recommendation**: keep; declare variants `extend_single`, `extend_multi`, `extend_mixed` (conditional on the flag) and record the
fill kernel; A standard prefill rows: reference for ≥~1k-token shapes (same entry point, different aiter commit, host-bound floor
below); A true-mixed rows: historical only. Consumer: gate the mixed models on the serving flag; if mixed-chunk is on, route
decode-in-mixed to the prefill unit's feature list.

### 5.9 `attn_decode`

**A**: `paged_attention_ragged(out, workspace, q, K, V, scale, kv_indptr, kv_pages, kv_last_page_lens, block_size,
max_num_partitions=ceil((kv+1)/256), None, "auto", "NHD", 0.0, k_scale, v_scale, None, 256)` with `batch_size` sequences at
uniform `kv_cache_size`, **all reading the same pages** (`aiter_attention_wrapper.py:423-446, 316-318`; `attention_wrapper.py:213`).
Consumers: RF on `(batch_size, kv_cache_size)`; prediction key `(padded decode batch size in FULL graph mode, ceil64(mean kv))`
×(1+0.1 if bs>1) × calibration (`SK.py:3911-3948, 5552-5609`); decode-only graph batches switch the whole predictor to the
`KERNEL_ONLY` family (`SK.py:966-995`), which needs `attention_kernel_only.csv` — absent for this model.

**SGLang** (img): `forward_decode` → `o = torch.empty_like(q)` (`img:2713`) → `paged_attention_ragged(o, workspace, q, k_cache.view(-1,1,H,128),
v_cache…, scale, kv_indptr, kv_indices, kv_last_page_len, 1, self.max_num_partitions, None, kv_dtype, "NHD", logit_cap, k_scale,
v_scale, None, 256)` (`img:2772-2792`) with `max_num_partitions = ceil(context_len/256)` fixed at init (`img:280-282`) and the
workspace sized for `max_bs` (`img:287-293`). Under graph replay the batch is padded to the capture bucket and padded rows attend one
token (E6 §2). F2 = `triton` would use the split-KV `decode_attention_fwd` with `num_kv_splits=16` on HIP.

**Assessment**: unit and features are right; the harness is not. Shared pages mean the kernel's KV reads for `B` sequences hit the
same addresses (L2/MALL-friendly) — for a bandwidth-bound kernel this plausibly understates cost at large `B·kv` [inferred; testable];
the per-shape partition count changes the reduction-stage work relative to the server's fixed value; small-shape labels are host
spans (A's bs-1 curve is flat at 32–38 µs from kv 128 to 16k, E1/plan §1.2).

**Recommendation**: keep; **replace the harness** (distinct page tables per sequence; use the server-wide partition formula or record
it as a column); key on the padded batch size; collect or explicitly disable the kernel-only family for graph-mode decode. A labels:
recollect.

---

## 6. Cross-boundary problems

**QKV projection + Q/K normalization.** Coherent on this version: the QK-norm kernels are always launched right after the GEMM, on
the same stream, and the only fusion that would move them (`fused_qk_norm_rope`) is CUDA-only and default-off. Keep together;
keep kernel-level rows in B so the grouping can be revisited when ROCm gets that kernel.

**RoPE + KV-cache write.** Fused unconditionally on HIP (gate has no flag). Separate profiling of RoPE is a control for the unfused
fallback kernel only; separate profiling of the KV write has no deployed counterpart at all (SGLang's unfused write is a JIT
`store_cache` kernel, A's is torch scatters). One unit, token-level, replaces both terms; B must record one occurrence mapped to two A
labels, never duplicate it.

**Norm / residual / all-reduce, including layer 0.** Communication is counted **only** by the CC backend (two slots per layer,
`execution_time.py:884-907`); norms exclude communication in both A and SGLang's default path, so no double counting today. Layer 0
runs a norm without residual add (1/48). With `--enable-aiter-allreduce-fusion` the post-attention norm fuses with the post-o_proj
all-reduce and the post-MoE all-reduce is deferred into the next layer's input norm: then a composite `ar_add_norm` must replace
*both* the norm term and the corresponding all-reduce slot (attention slot for `post_attention_layernorm`, MoE slot for
`input_layernorm` of layer ≥ 1), and the CC-backend prediction for that slot must be zeroed. Where B marks a scope: mark the norm
*call*, not the enclosing `prepare_attn`/`prepare_mlp`, unless the composite is intended and labeled.

**TP embedding.** Once per forward, not per layer; A measures the TP1 gather only; SGLang at TP>1 adds mask, fill and an all-reduce.
Frontier models none of it. If modeled, it belongs in a `per_forward_fixed` unit with the embedding all-reduce as its own term (or
inside the unit with the CC backend not adding it).

**Standard, chunked, batched, mixed attention.** Default SGLang keeps prefill (EXTEND) and decode (DECODE) in separate forwards;
chunking reuses the prefill kernel with a prefix; a batch of extends is one kernel call; MIXED exists only behind a flag and then is
one prefill-kernel call containing 1-token rows. A's keys survive per sequence (`prefix_len`, `extend_len`, `seq_len`) and SGLang
exposes them (`extend_prefix_lens`, `extend_seq_lens`, `seq_lens`); the server-wide `max_num_partitions`, the padded batch size, and
the real page tables are the additional keys B must preserve.

**Graph capture/replay, compilation, dispatch, streams.** Decode-only graphs on HIP by default; same kernels inside the graph
(fused rope+kv included), no host gaps between them; padding changes the effective batch size. `torch.compile` is off by default; the
attention `alt_stream` is `None` on HIP and `SGLANG_ROCM_USE_MULTI_STREAM` is off, so within the attention block kernels are serial
on one stream and additive summation of kernel durations is appropriate. Frontier's contract option (a) — kernel time plus a separate
regime-dependent overhead term — remains the right shape: overhead ≈ 0 inside graph replay and saturated prefill, launch-paced in
eager small-batch decode. Overlap with MoE-internal streams (if any) is outside this scope but should be checked in B's traces before
treating per-layer sums as wall time.

---

## 7. Proposed coherent partition (target cost model)

Per layer ℓ ∈ 0..47, per forward; `T` = tokens in the forward (padded in graph mode), `S` = per-sequence `(q_len, kv_len)` list,
TP = attention TP. Each physical kernel belongs to exactly one unit; dispatch gaps are a separate regime term.

| unit | contents (kernels) | features | variants / conditions | replaces A | communication |
|---|---|---|---|---|---|
| U1 `norm_in` | fused add+RMSNorm (ℓ≥1) | T | `norm_only` for ℓ=0; `ar_add_norm` when TP>1 **and** AR-fusion on (then it also contains the deferred post-MoE all-reduce) | `input_layernorm` | if `ar_add_norm`: zero the MoE all-reduce slot of layer ℓ−1 |
| U2 `qkv_proj_qknorm` | GEMM + 2 reshape copies + 2 head-norms | T, TP | provider tag (F1) | `attn_pre_proj` | none |
| U3 `attn_rope_kv_write` | `fused_qk_rope_reshape_and_cache` | T, TP | `fusion=off` control only (would split into rope + `store_cache`) | `attn_rope` + `attn_kv_cache_save` | none |
| U4 `attn_core` | prefill: [output fill] + `mha_batch_prefill_func`; decode: `paged_attention_ragged` | prefill: S (per-seq q_len, kv_len); decode: (padded bs, kv per seq), partition count | `extend_single` / `extend_multi` / `extend_mixed` (flag) / `decode_eager` / `decode_graph`; backend F2 changes kernels, not the unit | `attn_prefill`, `attn_decode`; true-mixed rows → `extend_mixed` only | none |
| U5 `o_proj` | GEMM | T, TP | provider tag | `attn_post_proj` | none |
| U6 `ar_post_attn` | RCCL/custom all-reduce of `[T,2048]` | T, TP | TP>1 only; absent when fused into U7 | (CC backend term) | this is the communication term |
| U7 `norm_post_attn` | fused add+RMSNorm | T | `ar_add_norm` when TP>1 and AR-fusion on | `post_attention_layernorm` | if `ar_add_norm`: zero U6 |
| U8 MoE block | out of scope | — | — | (MoE CSV) | post-MoE all-reduce U9 inside the block or deferred into U1 of ℓ+1 |
| U9 `ar_post_moe` | all-reduce | T, TP | TP>1; absent when deferred/fused | (CC backend term) | communication term |
| P1 `per_forward_fixed` (once) | embedding gather [+ mask/fill + AR at TP>1], final norm, lm_head GEMM, sampler kernels | T, TP, vocab | currently unmodeled | `emb` (partially) | embedding AR inside P1 or as its own term, not both |

Accounting rules:
1. `layer_time(ℓ) = U1 + U2 + U3 + U4 + U5 + U6 + U7 + U8 + U9` with U6/U9 = 0 at TP1 and set to 0 whenever the corresponding norm
   unit is the `ar_add_norm` composite. `model_time = Σ_ℓ layer_time(ℓ) + P1 + gaps(regime)`. With layer-homogeneous prediction,
   `Σ_ℓ ≈ 47·layer(ℓ≥1) + layer(ℓ=0)`; using 48·layer(ℓ≥1) is an explicit approximation.
2. A kernel measured inside a composite is never also predicted as a standalone term (fused rope/kv, fused AR+norm, output fill).
3. Gap/launch overhead is a regime term (graph replay ≈ 0; eager = engine launch pace), kept out of every unit's kernel time
   (measurement contract option (a)).
4. Overlap: on the default HIP path all attention-block kernels are serial on one stream; if B observes concurrent queues (MoE, copy
   engines), use interval union for busy time and do not sum across queues.
5. Provider and image are metadata on every unit; a unit is comparable across datasets only at equal provider/aiter/hipBLASLt.

Mapping of A scopes: 1:1 (U2, U5, U1, U7, U4-prefill, U4-decode), 2:1 (`attn_rope`+`attn_kv_cache_save` → U3), conditional
(true-mixed rows → only the `extend_mixed` variant, which the default configuration never runs), not at all (`emb` → P1, which the
simulator does not consume today).

---

## 8. Comparability versus prediction accuracy

- **Preserving A boundaries helps historical comparability** for U2, U5, U4 (identical boundaries; different providers/aiter commits)
  and for U1/U7 (same op; A's `input_layernorm` aliasing and the provider differ). It does **not** help for U3: the deployed unit is
  physically different from either A scope, and comparing it to Σ(A rope + A kv_save) is a hypothesis about kernel fusion gain, not a
  like-for-like check.
- **Preserving A boundaries does not by itself improve prediction of the target stack** anywhere; prediction quality is dominated by
  (i) the provider fork F1/F2, (ii) the harness defects (aliased norm input, shared pages, one-slot KV writes, host-bound attention
  loop), and (iii) the unmodeled once-per-forward work and the missing kernel-only family. Those need recollection or new data, not a
  re-partition — except U3, which needs both.
- **Dataset B** should record *physical* units (U-table) at occurrence level and carry a `maps_to_A` list per scope row
  (`attn_rope_kv_write → [attn_rope, attn_kv_cache_save]`); A-compatible summaries are derived views, never the primary rows.

---

## 9. Prioritized migration outline

| priority | change | files / interfaces | data impact | minimum validation |
|---|---|---|---|---|
| 1 | **Fix the effective serving configuration** (F1 AITER on/off, F2 attention backend, AR fusion, mixed chunk, page size, chunked-prefill size, graph buckets) and record it as part of the dataset identity | collection `RUN.md`/manifest; `data/profiling/compute/<device>/<model>/README.md` | none until decided; every linear/norm recollection depends on it | one eager prefill + one decode trace on the image under the chosen flags (§10.3) |
| 2 | **Merge `attn_rope` + `attn_kv_cache_save` → `attn_rope_kv_write`** | profiler: new scope in `frontier/profiling/linear_op/linear_op_impl.py` (needs a real paged pool + slot mapping, i.e. an in-situ or pool-backed harness) and/or B-sourced CSV; consumers: `frontier/attention/families.py` (CACHE_WRITE spec), `frontier/attention/profiling_mapping.py` (CACHE_WRITE contracts), `SK.py` getters `:5487-5550` and `_get_compute_model_names`, `frontier/entities/time_components.py` AttentionTime fields, metrics ledger names | recollection required; A rope/kv labels become historical | fused-vs-sum measurement at matched T; per-layer kernel count = 1 for this unit |
| 3 | **Replace three harness implementations**: distinct `x`/`residual` for `input_layernorm`; distinct per-sequence page tables and server-wide/recorded partition count in the attention profiler; (optionally) remove the double `emb` call | `linear_op_impl.py:1093`; `attention_wrapper.py:200-215`; `aiter_attention_wrapper.py:316-318`; `linear_op_impl.py:1089` | recollection of `input_layernorm`, `attn_decode`, (`attn_kv_cache_save` obsolete) | shared-vs-distinct pages at bs 64 / kv 4096; aliased-vs-distinct norm at T=4096 |
| 4 | **Conditional variants in the consumer**: layer-0 norm (document or model), `ar_add_norm` composite with all-reduce slot suppression, `extend_mixed` gating of `attn_decode_in_mixed`, kernel-only family for graph decode | `execution_time.py:884-907`; `MOE.py:1957-1984`; `SK.py:5559-5578, 966-995`; config flags | new CSV columns `variant`, `provider`, `fusion` | AR-fusion on/off at TP4; mixed-chunk on/off |
| 5 | **Decide on once-per-forward work** (`emb` + final norm + lm_head + sampler) | new `ExecutionTime` term or explicit exclusion note | new collection if adopted | small-batch decode wall vs Σ units |
| 6 | Re-profile A-linear under the chosen provider (hipBLASLt 1.2.2 / tgemm) and rebuild the regressor bench on it | existing linear_op profiler in the serving image (needs vLLM-free GEMM/norm paths or SGLang's own layers) | full linear recollection | existing sanity/validity scripts + tile-switch map |

Decisions that genuinely require the user: (D1) F1/F2 and the secondary flags — there is no production recipe to defer to;
(D2) accept the U3 merge and the removal of the independent cache-write term for this stack; (D3) whether once-per-forward GPU
work is in scope; (D4) whether A-linear is recollected under the serving provider or kept as a control only; (D5) page size 1
(SGLang default, A block-1 rows) vs 16 (used by a DriveNets Qwen3.5 recipe) — it changes U3's write pattern and U4's page tables.

---

## 10. What static evidence cannot settle, and the experiments that would

10.1 [unresolved] Kernel count and names inside each unit on the approved image under the chosen flags (e.g. the DriveNets fill kernel
before prefill, native-norm kernel counts, tgemm's library choice per shape, whether `q_norm` weights are bf16 so the aiter path is
taken). 10.2 [unresolved] Whether hipBLASLt tile transitions in A-linear persist with hipBLASLt 1.2.2 / `F.linear` / `tgemm`.
10.3 **Proposed experiment (not run):** on one node, approved image by digest, TP1, the chosen flags, `rocprofv3 --kernel-trace
--marker-trace` around one eager prefill forward of ~1024 tokens and one decode forward at bs 1 and bs 64 (eager and graph); expected
per-layer sequence on the default path with AITER on: `rmsnorm2d_fwd_with_add` (or `rmsnorm2d_fwd` at ℓ=0) → GEMM → copy → `rmsnorm2d_fwd`
→ copy → `rmsnorm2d_fwd` → `_fused_qk_rope_reshape_and_cache_kernel` → [fill] → `fmha_batch_prefill*` | `paged_attention_ragged*` (+ reduce)
→ GEMM → [all-reduce at TP>1] → `rmsnorm2d_fwd_with_add` → MoE kernels → [all-reduce]. Any deviation changes the unit table above.
10.4 **Proposed experiment:** same harness, fused `attn_rope_kv_write` vs A's `attn_rope` + `attn_kv_cache_save` at T ∈ {1, 64, 1024,
8192}; hypothesis fused < sum. 10.5 **Proposed experiment:** `paged_attention_ragged` with shared vs distinct pages at bs 64, kv 4096,
page 1; hypothesis shared < distinct. 10.6 [unresolved] Qwen3-30B-A3B production flags (D1).

---

## 11. Handoff to the Dataset B agent

**Proceed independently** (does not depend on the open decisions): collector mechanics (marker placement at exact calls, rank/device/
dispatch identity, clock sidecar, trace extraction, checker fixtures), eager TP1 attribution, and the scopes U2 (`qkv_proj` +
`apply_qk_norm`), U5 (`o_proj`), U4 (`self.attn(...)`), U1/U7 (the norm *calls* inside `prepare_attn`/`prepare_mlp`, not the
enclosing functions), and U3 (`self.rotary_emb(...)`) as **one** physical scope with `maps_to_A = [attn_rope, attn_kv_cache_save]`.

**Keep provisional** until the user decides: the F1/F2 flag set (collect with the flags recorded; if two candidate recipes remain,
pilot both at TP1 for one short prompt set before any sweep), page size, `--enable-mixed-chunk`, AR fusion (if on, the norm scopes
become composites and the all-reduce must be inside them, labeled), and whether `emb`/final norm/lm_head are collected.

**Preserve in traces/metadata to support the decisions:** per-kernel rows inside U2 (GEMM vs norm kernels) and U4 (fill vs attention),
the kernel names (library/tile attribution), `max_num_partitions`, padded vs real batch size, per-sequence `(prefix_len, extend_len,
seq_len)` lists, layer index (ℓ=0 norm variant), the exact env (`SGLANG_USE_AITER`, `SGLANG_USE_AITER_UNIFIED_ATTN`,
`SGLANG_AITER_UNIFIED_PREFILL`, `SGLANG_AITER_KV_CACHE_LAYOUT`) and server flags, the image digest `sha256:d8e0eb9c…`, the
sglang wheel sha256 `3706017c…`, aiter `b4d9154d1`, and the patch set listed in E5 §2.

**Do not** assume A-attention's true-mixed structure exists in SGLang, divide a fused duration into A parts, or mark the enclosing
communicator functions as norm scopes.

---

## 12. Evidence index

| file | content |
|---|---|
| `evidence/E1_frontier_profiler_scopes.md` | exact timer contents of all nine scopes, harness mechanics, TP>1 row handling, regressor-bench consumption |
| `evidence/E2_frontier_predictor_training_and_selection.md` | CSV filters, features, model names, prediction-side feature construction, mixed/standard selection |
| `evidence/E3_frontier_execution_time_composition.md` | ExecutionTime formula, all-reduce slots, calibration/overhead terms, what is schema vs essential |
| `evidence/E4_operator_partition_provenance.md` | git history of upstream and fork; documented vs inferred rationale; Vidur lineage |
| `evidence/E5_image_and_sglang_source_verification.md` | registry digest, config/ENV/labels, layer map, wheel diff vs `v0.5.18`, aiter version, flag gates, cluster recipes, vLLM absence |
| `evidence/E6_sglang_v0518_regimes_graph_comm_embedding.md` | forward modes, mixed-chunk, graphs/padding, communicator and AR fusion, embedding TP path, partition count, GEMM/norm dispatch |
| `evidence/E7_vidur_sarathi_primary_sources.md` | verbatim Vidur paper §4.3/4.4 and Vidur/Sarathi code quotations |

Investigation limits: no GPU run; the approved image was inspected through its registry manifest, config blob and three
individually downloaded layers (SGLang wheel, aiter wheel, `/opt/aiter`), not by executing it; the sibling `20261001-7` image was
used only to establish that vLLM is absent from the lineage; Vidur paper text came from the arXiv HTML rendering (v2).
