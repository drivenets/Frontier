# Dataset B — in-situ kernel timings of A's operators inside a real Qwen3-30B-A3B / SGLang / MI355X run

Status: **Implementation in progress, 2026-10-04; no B measurements yet.** Current task: [02_IMPLEMENTATION_HANDOFF.md](02_IMPLEMENTATION_HANDOFF.md). Implementation/results: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). Historical image findings below require recovery against the approved image. Scope policy is provisional pending the independent operator investigation. Local tooling does not wait for that report.

Companion documents (A): `../00_…` to `../14_…`, `../story/STORY.md`, `../HOWTO_run_linear_op_collection.md`,
`data/profiling/compute/mi355x/qwen3-a3b-30b-moe/README.md` and `runs/*/RUN.md`.

## 1. What A is (Step 0)

A = Frontier's standalone per-op profiles for `qwen3-a3b-30b-moe` on MI355X. Two profilers, two datasets, two images:

| dataset | file(s) that are "A" | profiler | image | timing method | regime |
|---|---|---|---|---|---|
| A-linear | `/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv` (job 21539, md5 `3635e4d3…`, 13,308 rows = 3,327 `num_tokens` × TP {1,2,4,8}); the handoff file (`14_`). Older: 200-fwd file (job 21486), root `linear_op.csv` (job 21313, pre-fix, do not use) | `frontier.profiling.linear_op.main --is_moe`, one decoder block per forward, dummy weights, BF16, no collectives | `lmsysorg/sglang:v0.5.11-rocm700-mi35x` (ROCm 7.0.0, torch 2.9.0a0, vLLM 0.9.2rc2 kernels, hipBLASLt 1.0.0, aiter a6bb49937) | HIP-event pair per scope, device held busy (GEMM work backlog, 8 settle, 25 timed); `time_stats.<op>.median` in ms | kernel time + ≈2.5–3 µs record cost per scope boundary, automatic clock ≈2.3–2.4 GHz (`sclk_mhz_*` per row), `DEBUG_CLR_MAX_BATCH_SIZE=1000000`, GC disabled |
| A-attention | `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/attention_combined.csv` (root; 25,032 rows = block {1,16} × TP {1,2,4,8}; run `2026-09-15_1142_attention_16k`, jobs 21310/21311); 32k/64k siblings on the datasets store | `frontier.profiling.attention.main --attention_backend AITER`, one attention call per forward | `lmsysorg/sglang:v0.5.17-rocm720-mi35x` (ROCm 7.2.0, torch 2.9.1, aiter d9e5ef7ce, hipBLASLt 1.2.1) | HIP-event pair per scope, **legacy loop (3 warm-up + 50 timed, no backlog, no clock probe)** | **never received the GPU-bound / clock fixes** (`STORY.md` §15: "validated but never had the linear_op-style per-run scrutiny"). Small-shape host contamination is a concern, not a measured per-row decomposition |
| A-moe | only `runs/legacy_pre-2026-09_torch-sdpa/moe.csv` (vLLM Triton `fused_moe`, 20 reps, pre-workstream) | `frontier.profiling.moe` | pre-2026-09 | legacy | marked **old — do not use**; MoE was deferred (no aiter MoE path in the profiler) |

### 1.1 Inventory table — nine A scopes forming the initial comparison map

"Trainer" = the old sklearn predictors (`sklearn_execution_time_predictor.py`, `shared_prediction_model_manager.py`,
`frontier/training/{linear_op,attention}_trainer.py`) and the new per-(op, TP) regressor (`../regressor_bench/dataset.py`).
Key columns = the exact-match filter columns + the regression feature(s) the trainer uses; B must derive the same keys from the real forward.

| # | scope (`time_stats.<op>`) | dataset | code A wraps (inside the timer) | kernels in A's scope (trace-confirmed, `13_` §4.2, `09_` §3c) | recorded on | A key columns (filter → feature) | consumer | image |
|---|---|---|---|---|---|---|---|---|
| 1 | `attn_pre_proj` | A-linear | `qkv_proj` GEMM (`ColumnParallelLinear` → vLLM `dispatch_unquantized_gemm`, M=`num_tokens`, K=2048, N=(32·128+2·kv·128)/TP with kv replicated at TP8 → N=5120/2560/1280/768), `qkv.split`, `q.view(T,-1,128)` → vLLM `rms_norm` (q_norm), same for k_norm, `.view` back (`linear_op_impl.py:503-526`) | Tensile/hipBLASLt GEMM (or `wvSplitK` skinny at M≤4) + copy + `vllm::rms_norm_kernel` + copy + `rms_norm_kernel` = 5 kernels (TP>1: +copies/blits from the sharded split) | TP 1,2,4,8 | `n_head`=32, `n_kv_head`=4, `n_embd`=2048, `n_expanded_embd`=768, `use_gated_mlp`, `vocab_size`, `use_qk_norm`=True, `attn_output_gate`=False, `num_tensor_parallel_workers` → feature **`num_tokens`** | old predictor + trainers; new regressor (per TP) | v0.5.11 |
| 2 | `attn_rope` | A-linear | Frontier `RotaryEmbedding.forward` → **one** in-place `vllm._custom_ops.rotary_embedding(positions, q, k, 128, cos_sin_cache, is_neox=True)` over q and k together (`attn_rope_impl == vllm_kernel` on every row) | `vllm::rotary_embedding_kernel<BFloat16,true>` ×1 | TP 1,2,4,8 | same as #1 → `num_tokens` | same | v0.5.11 |
| 3 | `attn_post_proj` | A-linear | `o_proj` GEMM only (`RowParallelLinear`, M=`num_tokens`, K=4096/TP, N=2048, `reduce_results=False`, **no all-reduce**); input is `randn_like(q)` produced outside the scope | hipBLASLt GEMM ×1 | TP 1,2,4,8 | same as #1 → `num_tokens` | same | v0.5.11 |
| 4 | `input_layernorm` | A-linear | vLLM `fused_add_rms_norm(x, residual, w, eps)` with **x and residual the same tensor** (`linear_op_impl.py:919-922`, `layernorm.py:56-58`); no separate `add` scope exists for this model | `vllm::fused_add_rms_norm_kernel` ×1 | **TP 1 only** (replicated op; TP>1 rows NaN by design) | same as #1 with TP key forced to 1 → `num_tokens` | same (TP-1 regressor) | v0.5.11 |
| 5 | `post_attention_layernorm` | A-linear | vLLM `fused_add_rms_norm(o_proj_out, residual)` | ×1 | TP 1 only | same → `num_tokens` | same | v0.5.11 |
| 6 | `emb` | A-linear | `F.embedding(input_ids, weight[151936, 2048])` (TP1 path: no mask, no all-reduce); **recorded twice per forward** (profiler artefact: `embed_tokens` called at `:1089` and `:1092`; `count` = 2×active) | gather ×1 (per call) | TP 1 only | same → `num_tokens` | old predictor trains it (`predictor_target=False` in the family spec); new regressor: yes (TP-1) | v0.5.11 |
| 7 | `attn_kv_cache_save` | A-attention | two PyTorch advanced-index scatters `key_cache[blk, off] = key; value_cache[…] = value` (`aiter_attention_wrapper.py:394-401`) — **not** an aiter kernel; **all sequences of a standard batch write the same slot** (shared block table) | `index_put_` ×2 | all rows (prefill, decode, true-mixed) | filter `n_embd`, `n_q_head`=32, `n_kv_head`=4, `block_size`∈{1,16}, `num_tensor_parallel_workers` → features **`num_tokens`** in the execution predictor; **`total_tokens`, `kv_cache_size`, `batch_size`** in the shared predictor | old predictor (`attn_kv_cache_save` model, all rows) | v0.5.17 |
| 8 | `attn_prefill` | A-attention | **one** `aiter.ops.mha.mha_batch_prefill_func(q, key_cache, value_cache, cu_seqlens_q, kv_indptr, kv_page_indices, max_q, max_kv, softmax_scale=1/√128, causal=True, out=…, kv_last_page_lens=…)`; paged KV, NHD, page = `block_size`; query = `prefill_chunk_size` new tokens attending `kv_cache_size + chunk` keys (bottom-right causal) | CK `fmha_batch_prefill` kernel(s) | standard prefill rows (`is_prefill`, `batch_size`==1) and true-mixed prefill targets | filter as #7 → features **`kv_cache_size`, `prefill_chunk_size²`** (row keys `prefill_chunk_size`, `kv_cache_size`, `batch_size`=1) | old predictor (`attn_prefill`); `attn_prefill_mixed` and `attn_decode_in_mixed` also consume the 2,880 canonical true-mixed rows; these reuse existing physical scope labels | v0.5.17 |
| 9 | `attn_decode` | A-attention | **one** `aiter.ops.attention.paged_attention_ragged(out, workspace, q, key_cache, value_cache, scale, kv_indptr, kv_page_indices, kv_last_page_lens, block_size, max_num_partitions=ceil(seq/256), None, "auto", "NHD", 0.0, k_scale, v_scale, None, 256)`; `batch_size` sequences × 1 new token each, **uniform** context `kv_cache_size` (+1 attended) | aiter paged-attention kernel(s) (+reduce) | standard decode rows (`~is_prefill`, `prefill_chunk_size`==0) and true-mixed decode targets | filter as #7 → features **`batch_size`, `kv_cache_size`** | old predictor (`attn_decode`) | v0.5.17 |

Present in A but **not** consumed by a trainer, therefore **out of scope for B**: `forward_gpu_span` (closure check), `attn_input_reshape` / `attn_output_reshape`
(views, zero kernels), the `mlp_*` scopes (run but dropped for `--is_moe`), `time_stats_hostbound.*` and every instrument column, the legacy `moe.csv`
scopes (`moe_gating_linear`, `moe_gating_routing_topk`, `moe_shuffling`, `moe_grouped_gemm` — old, vLLM Triton, "do not use"). The MoE block, lm_head, final norm,
sampling and every collective are out of scope: B preserves their occurrences in full raw traces and reason-coded exclusion accounting, including predecessor context; they are not additional prediction targets.

### 1.2 A's key space and discontinuities (what B's grid must straddle)

Computed from the settle-8 file (neighbouring-grid-point jumps > 8 % with the smaller value > 8 µs) and the root attention file:

- **`num_tokens` range** 1…16384 (3,327 values). Flat (fixed-cost) region ends at ≈108 (pre_proj TP1/2), ≈167 (TP4), ≈322 (TP8) tokens; `attn_post_proj` leaves the floor at 4–32 tokens; `attn_rope` at ≈470 (TP1) … ≈3,100 (TP8).
- **hipBLASLt tile switches** (both GEMMs; the pairs (last-before, first-after)): TP1 `attn_pre_proj` 159/160, 191/192, 223/224, 255/256, 1536/1537, 3072/3080, 4800/4808, 6400/6408, 10752/10768, 13056/13072; TP1 `attn_post_proj` 16/17, 63/64, 128/129, 256/257, 512/513, 1024/1025, 2048/2056, 3072/3080, 4256/4264, 4864/4872, 5376/5384, **8192/8208 (+30 %)**, 9216/9232, 10752/10768; TP2 `attn_post_proj` 4320/4328, 4480/4488, 4800/4808, 4864/4872, 5120/5128, 5376/5384, 8192/8208, 8960/8976, 9984/10000; TP4 4352/4360, 4864/4872, 5184/5192, 5376/5384, **8192/8208 (+81 %)**, 9216/9232, 10304/10320, 10752/10768, 11776/11792; TP8 2048/2056, 3072/3080, 3200/3208, 4256/4264, 4352/4360, 4448/4456, 4480/4488, 9216/9232, 10752/10768, 11136/11152, 11424/11440, **12288/12304 (−42 %)**. Norm steps: 2048/2056, 6144/6152, 8192/8208, 12288/12304 (input_layernorm); 2048/2056, 8192/8208 (post_attention_layernorm).
- **Attention prefill (kv=0)**: +15–21 % steps every 128 tokens from 1024 up (1024→1088, 1152→1216, …, 3072→3136): the prefill kernel's 128-token tile. A's kv=0 lengths: 32…16128 (step 32/64/256); chunk sizes 64…16384 (178 values, step 16/32/64/128); chunked rows at kv = j·chunk.
- **Attention decode**: `kv_cache_size` ∈ {128, 512, 1024, 2048, 4096, 8192, 16384}, `batch_size` ∈ {1,2,4,8,16,24,32,48,64,96,128,…,512}. Observed A medians at bs 1 are **flat 32–38 µs from kv 128 to 16384** and `attn_kv_cache_save` is flat ≈15 µs: a possible legacy host-span floor; kernel/host decomposition is unmeasured.
- `num_tokens = 4000` faults A's linear_op profiler (documented GPU fault). B includes 4000 as a probe.

## 2. The serving image and its kernel libraries

Images present on the cluster and inspected (login node, no GPU):

| image | sglang | torch / ROCm | aiter | hipBLASLt | rccl | tracers | nodes |
|---|---|---|---|---|---|---|---|
| `dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04` (DriveNets build, also at `10.1.1.0:5050/dnis/…`, 28 h old) | **0.5.18** (pip, `/usr/local/lib/python3.12/dist-packages/sglang`) | 2.12.0+git6bbd260 / **7.2.3** | `0.1.1.dev1+gb4d9154d1.d20260923` | **1.2.2** (70203) | 2.27.7 | rocprofv3 + rocprofiler-sdk 1.1.0 + `libroctx64` | 1 only (pullable from the local registry) |
| `rocm/sgl-dev:v0.5.20-rocm724-mi35x-20260919` | 0.5.20.dev | 2.11 / 7.2.4 | 0.1.21.dev48+g4ad998328 | 1.2.2 (70204) | 2.27.7 | rocprofiler 2.0 (no sdk/roctx pkg listed) | 1, 5 |
| `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260907` | 0.5.19.dev | 2.9.1 / 7.2.0 | 0.1.21.dev48+g4ad998328 | 1.2.1 | 2.27.7 | rocprofiler-sdk 1.1.0 | 1 |
| `lmsysorg/sglang:v0.5.17-rocm720-mi35x` (**A-attention**) | 0.5.17 | 2.9.1 / 7.2.0 | 0.1.20.dev12+gd9e5ef7ce | 1.2.1 | 2.27.7 | rocprofiler 2.0 | 1,2,4,5,8,9 |
| `lmsysorg/sglang:v0.5.11-rocm700-mi35x` (**A-linear**) | 0.5.11 | 2.9.0a0 / 7.0.0 | 0.1.12.post2.dev150+ga6bb49937 | 1.0.0 | 2.26.6 | rocprofv3 (used in `07_`/`13_`) | 1,4,5,6,7,8,9 |

**Approved image:** `dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04`.
Do not select or substitute another image. Pin its digest and re-verify versions/call sites; the previous raw inspection dumps were lost.
The other image rows are historical context, not new candidate choices.

How production launches SGLang on MI355X here (`~/amd-playground/InferenceMAX/sglang_scripts/*.sh`, `benchmark-pipeline/modules/benchmark_runner.py`,
`/opt/shared/bp-inferencex/benchmarks/single_node/agentic/qwen3.5_fp8_mi355x.sh`, `profiling_knowledge/AITER_KERNELS.md`): `SGLANG_USE_AITER=1`, attention backend
`aiter` (the HIP default), `--disable-radix-cache` for benchmarks, large `--chunked-prefill-size`, `--cuda-graph-max-bs`/`--max-running-requests` = concurrency,
`--mem-fraction-static 0.8–0.9`, `--enable-aiter-allreduce-fusion` (qwen3.5 script), `--moe-runner-backend aiter` (older captures). No script for Qwen3-30B-A3B itself
was found; the closest DriveNets recipe is the qwen3.5 one.

## 3. Scope → SGLang mapping (0.5.18 in the approved image (historical source inspection; not yet recovered), ROCm, `SGLANG_USE_AITER=1`, page size 1, BF16 KV, `nhd` layout)

Code path confirmed by reading the image's `srt/models/qwen3_moe.py`, `qwen2_moe.py`, `layers/communicator.py`, `layers/layernorm.py`,
`layers/rotary_embedding/base.py`, `layers/attention/aiter_backend.py`, `mem_cache/memory_pool.py`, `layers/quantization/unquant.py`, `models/utils.py`,
`model_executor/model_runner.py`, `layers/vocab_parallel_embedding.py`. Per layer, `Qwen3MoeDecoderLayer.forward` runs:
`layer_communicator.prepare_attn` (norm) → `self_attn` (`qkv_proj` → `apply_qk_norm` → `rotary_emb(…, fused_set_kv_buffer_arg)` → `attn` → `o_proj`) →
`prepare_mlp` ([all-reduce] + norm) → MoE → `postprocess_layer` ([all-reduce]).

| A scope | SGLang code at the same boundary | kernels B will see | boundary verdict |
|---|---|---|---|
| `attn_pre_proj` | `qkv, _ = self.qkv_proj(hidden)` then `apply_qk_norm(q, k, q_norm, k_norm, 128)` inside `apply_qk_norm_rope` (the fused-QK-norm-RoPE kernel is **CUDA-only**: `use_fused_qk_norm_rope` requires `_is_cuda`) | GEMM = **`aiter.tuned_gemm.tgemm.mm`** when `SGLANG_USE_AITER=1` (`unquant.py:235`; hipBLASLt or aiter asm/CK by tuned table) else `F.linear`→hipBLASLt; then `reshape(-1,128)` copies + **aiter `rmsnorm2d_fwd`** ×2 (A: vLLM `rms_norm`). During graph capture k_norm runs on `alt_stream` (`apply_qk_norm`, capture mode only) | **same boundary; different kernel providers** (GEMM library/tuning, norm kernel). Marker: around `qkv_proj` + `apply_qk_norm`. |
| `attn_rope` | `self.rotary_emb(positions, q, k, fused_set_kv_buffer_arg=…)` → `RotaryEmbedding.forward_cuda`, `use_fallback_kernel=True` on HIP, and **`enable_fused_set_kv_buffer()` is unconditionally True on HIP** (`models/utils.py:284-303`) → **`fused_qk_rope_reshape_and_cache`** (Triton, `sglang/kernels/ops/kvcache/rope_cache.py:546`): RoPE on q,k **and** the K/V write into the paged pool in **one kernel**; then `save_kv_cache=False` in `forward_core`, so the attention backend writes nothing | 1 Triton kernel = A's #2 **+** A's #7 | **MISMATCH (stop-and-ask, §6 Q1)**: A's `attn_rope` (vLLM fused rope kernel, q+k) and `attn_kv_cache_save` (2 scatters) do not exist as separate kernels on this path. |
| `attn_kv_cache_save` | see above (fused). The unfused path (`token_to_kv_pool.set_kv_buffer` → `_set_kv_buffer_impl` → `store_cache` or `k_cache[idx]=k; v_cache[idx]=v`) is only taken when the fusion is off | — | **MISMATCH** (same item) |
| `attn_prefill` | `RadixAttention.forward` → `AiterAttnBackend.forward_extend` → NHD path → **`mha_batch_prefill_func(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices(page table), max_q, max_kv, causal=True, logits_soft_cap=0, window=(-1,-1), out=<explicit buffer>)`** (`aiter_backend.py:2546`), page size 1. Preconditions: not MLA, `SGLANG_USE_AITER_UNIFIED_ATTN` off (default), `SGLANG_AITER_UNIFIED_PREFILL` off (default), `SGLANG_AITER_KV_CACHE_LAYOUT=nhd` (default), bf16 KV | same aiter kernel family as A's block-1 rows (different aiter commit); chunk with prefix → same kernel with `kv_indptr` covering the prefix | **same boundary**, same kernel entry point; A's block-16 rows have no production counterpart |
| `attn_decode` | `forward_decode` → NHD path → **`paged_attention_ragged(o, workspace, q, k_cache.view(-1,1,H,128), v_cache…, scale, kv_indptr, kv_indices, kv_last_page_len, 1, self.max_num_partitions, None, "auto", "NHD", 0.0, k_scale, v_scale, None, 256)`** (`:2772`) | same aiter kernel as A; **`max_num_partitions` = ceil(max_context_len/256) fixed per server** (A: ceil(seq/256) per shape) → workspace/partition loop differs | **same boundary**; parameterisation differs (record `max_num_partitions`) |
| `attn_post_proj` | `output, _ = self.o_proj(attn_output)` with **`reduce_results=False`** (`qwen3_moe.py:497-505`): the TP all-reduce is done later by the communicator | GEMM via `tgemm.mm` / `F.linear` | **same boundary** (no all-reduce inside, as A) |
| `input_layernorm` | `layer_communicator.prepare_attn`: layer 0 → `residual = hidden; hidden = input_layernorm(hidden)` (**norm only, no add**); layers 1..47 → `input_layernorm(hidden, residual)` = **aiter `rmsnorm2d_fwd_with_add`** (`RMSNorm.forward_aiter`); at TP>1 the MoE all-reduce of the previous layer precedes it (or is fused into it with `--enable-aiter-allreduce-fusion`) | layer 0: `rmsnorm2d_fwd`; layers ≥1: `rmsnorm2d_fwd_with_add` (+ RCCL all-reduce kernel at TP>1 as a separate row) | **layer 0 differs from A** (A always times the fused add+norm with x==residual); layers ≥1 same op, aiter kernel vs vLLM. B tags layer 0 rows `variant=norm_only`. |
| `post_attention_layernorm` | `prepare_mlp` → TP1 `_simple`: `post_attention_layernorm(hidden, residual)`; TP>1: `all_reduce` then norm, or one fused AR+norm kernel when aiter all-reduce fusion is enabled | `rmsnorm2d_fwd_with_add` (+ all-reduce at TP>1) | same op at TP1 (A is TP1-only); at TP>1 the all-reduce is a separate kernel row unless fusion is on |
| `emb` | `Qwen2MoeModel.forward`: `hidden = self.embed_tokens(input_ids)` once per forward; TP1 = plain gather; TP>1 = masked gather + all-reduce | gather (+ mask kernels + all-reduce at TP>1) | same op at TP1 (A is TP1-only, but A records it **twice per forward**) |

Other differences that are not boundary mismatches but will show in B: real weights (A: `randn`), real KV contents and real page tables (A: shared block table for all sequences), `positions` = real positions (A: 0..T-1), the MoE block and lm_head between the marked scopes, prefill `out` buffer allocation, graph replay for decode.

## 4. Corrected implementation and pre-cluster plan (2026-10-04)

1. Implement the schema, occurrence identities, CPU-launch correlation, nested markers, queue predecessors, calibrated per-GPU telemetry joins, request manifests, and strict checker locally. Keep raw records immutable and make summaries reproducible. The executable contract is `B_SCHEMA.md` and the versioned `scope_spec.json`.
2. Recover only the approved image's digest, exact serving sources, launch help, and tracing capabilities. Save commands, source hashes, and excerpts. Validate source and flags before any Slurm allocation. No simulator code or A data changes.
3. Instrument exact calls using one annotation backend per pass. Buffer CPU-only metadata until the measured window has drained; never read device tensors for shapes. Norm communicator wrappers are not norm boundaries. Keep all ranks and scheduler request membership.
4. Use provisional, clearly identified scopes for the smallest TP1 eager attribution experiment. No name/order-based graph attribution: replay requires capture/node evidence supported by the pinned tracer; otherwise report unresolved and continue eager work.
5. Then exercise short/long prefill, chunking, mixed execution if supported, decode eager/replay and padding, matched Kineto passes, marked/unmarked outputs, perturbation, telemetry calibration, and all-rank TP4 validation. Present pilot results and measured resource estimates before a full sweep.

Primary tracer: rocprofv3 GPU begin/end timestamps, host HIP API correlation and markers. Retain its richest supported output; direct CSV is sufficient only for eager attribution if required columns exist. Kineto is a separate pass. No event timing, settling kernels, synthetic backlog, or clock locks.

Timing quantities are distinct: `kernel_dur_us=(end-begin)/1000`, `kernel_sum_us=sum(durations)`, `scope_span_us=(max(end)-min(begin))/1000`. Predecessors are computed on all raw kernels per process/GPU/queue before target filtering. A queue interval does not measure device idle or host overhead.

Checks use actual scheduler forwards and scope applicability, independent of dispatch labeling. Excluded MoE, collectives, output-head and sampling are legal; unknown/unresolved dispatches are not. Reject duplicate occurrences, missing ranks/requests/markers, empty/truncated data, ambiguous attribution, stale or wrong-device clocks. Shape-dependent dispatch changes and concurrent queues are legal. Numeric telemetry/tracer limits must be calibrated, not inherited from draft guesses.

### Coverage and workload arithmetic

| Axis | Intended pilot / initial coverage | Deferred or unresolved |
|---|---|---|
| TP | 1 and 4, every rank | 2 and 8 deferred; no complete-A coverage claim |
| dtype / pages | BF16 / page 1 | page 16 lacks the historical serving counterpart |
| attention | standard, chunked, true-mixed where supported | mixed correspondence requires source/runtime evidence |
| decode | eager and replay, real and padded sizes | graph attribution remains required and unproved |
| scopes | nine A scopes as comparison map; physical compounds preserved | fusion/norm/embedding mappings await investigation |
| clocks | automatic, continuous external telemetry | locked clocks and extra controls deferred |

The original D1 list is preserved in `workload_grid.json` and has **155** lengths, three repeats and G=16. If prefill produces the first token and no chunking occurs, this implies 7,440 forwards per graph mode, per TP (14,880 across two modes); confirm from execution. D2 implies **186 prefill forwards per TP** over three repeats under exact chunking. D3's largest cohort has **524,288 prompt tokens**, at least **32 prefill forwards** at a 16,384-token budget. HTTP concurrency never establishes observed batch size. Store query/prefix/attended vectors and padding explicitly.

Freeze the full request manifest only after the pilot. Kernel counts, all-rank storage and runtime are unmeasured; old 4–6 node-hour and per-forward kernel estimates are withdrawn. Start with the ten prompt lengths in §9 plus the smallest chunking/mixing/padding cohorts needed for mechanism validation. Cross-tracer checks use representative matched cases, not an automatic duplicate full sweep.

### Consumer contracts

`frontier/attention/profiling_mapping.py` defines cache-write keys separately: execution predictor `num_tokens`; shared manager `(total_tokens, kv_cache_size, batch_size)`. Standard prefill uses `(kv_cache_size, prefill_chunk_size_squared)`; standard decode uses `(batch_size, kv_cache_size)`. Filters include model dimensions, TP and block size.

`sklearn_execution_time_predictor.py` mixed prefill uses avg/min/max sequence length, population variance/CV/range, batch interactions, total tokens and its square, and KV context; true-mixed columns use `prefill_mixed_*` aliases. Decode-in-mixed uses `(decode_batch_size, decode_avg_kv_cache_size, num_prefill_seqs, total_prefill_tokens, total_batch_size, batch_composition_ratio, total_tokens)`. All 2,880 mixed rows have complete inspected decode features and all three median labels. Canonical attention totals: 25,032 = 22,152 standard + 2,880 mixed. Nonuniform vectors are authoritative; a mean must never be substituted for a uniform standard-A key.

## 5. Environment hygiene carried over from A's fixed collections (and what is not)

Applied (runtime-relevant, per `HOWTO` §2.1, `13_` §7–8, the sbatch, and the settle-8 README):

| item | A's fixed runs | B | why |
|---|---|---|---|
| container flags | `--device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G` | same, `--shm-size 32g` (TP>1 SGLang; production used 32g), `--network host` | rocprofv3 needs ptrace/seccomp; NCCL/RCCL shm |
| `HSA_ENABLE_COREDUMP=0` | set | set | no core dumps on GPU faults |
| `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` explicit | set | set (the TP's GPUs) | deterministic device mapping |
| `DEBUG_CLR_MAX_BATCH_SIZE=1000000` | set (`13_` §7: removes the ROCclr 1000-packet barrier from timed scopes) | **set (approved planned main setting)**; production does not set it | runtime flag; keeps B's gaps comparable to A; flagged as a production deviation |
| whole node, pinned, idle neighbours | `--gres=gpu:8 -w` | same | power/thermal budget, no co-tenants |
| automatic clock, no lock | yes; probe per row | yes; **sidecar sampler** (§4) replaces the probe; D4 optional lock | constraint 4 |
| aiter JIT cache warm before timing | `/tmp/aiter-cache-dn` | warm-up requests before the traced window | JIT stalls out of the data |
| GC | `gc.disable()` around the timed loop (profiler-only) | SGLang's own `/freeze_gc` after warm-up (its log recommends it) — logged | not a profiler fix; a production-recommended runtime setting |
| `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK` unset | yes | N/A (Frontier profiler only) | — |
| work backlog, settle forwards, spun blocks, clock probe, two timing columns | profiler-only | N/A: the serving engine sets the queue state; B records `gap_before` instead | contract §3: overhead is a measured term |

Everything above is written to `env_<run>.json` and `RUN.md` per run.

## 6. Current decisions and dependencies

Approved: the pinned-tag DriveNets image above; TP1/TP4 implementation defaults; BF16, page 1, radix cache disabled, automatic clocks; `DEBUG_CLR_MAX_BATCH_SIZE=1000000` (a deviation from ordinary production); record server-wide decode partition count unchanged.

Pending: final physical scope definitions, fused RoPE/cache correspondence, TP communication/norm/embedding variants, and any mixed-path composite. Consult `../operator_boundary_investigation/REPORT.md` when available. Do not force unfused execution to recreate inherited labels or treat research recommendations as approved simulator changes. Provisional observations are permitted; full coverage cannot be frozen yet.

Mechanism dependencies: actual image/source access, unique replay identities, all-rank request/forward correspondence, and calibrated telemetry. Block only work depending on the missing evidence. The historical seven-question universal gate is superseded.

## 7. Pre-pilot hypotheses

Only compatible whole scopes can be compared later. If A were context-free and kernel providers, shapes, variants and clocks matched, B's kernel sums should track the kernel component of A; A's event/gap term still differs. No fixed event-cost subtraction or tracer-agreement percentage is assumed. Fused work has one physical duration and no measured split. Library/fusion changes preclude causal attribution to context alone. Queue intervals and clock trajectories are observations to collect, not predicted host overhead. No A-versus-B analysis is performed here.

## 8. Seed for the one-page confounder note (differences in how B is collected vs A)

Instrument (kernel timestamps vs event pairs, no record cost, tracer-loaded host), kernel providers (aiter `tgemm`/CK norms/Triton fused rope vs vLLM + hipBLASLt 1.0.0),
image and library versions (§2), fused RoPE+KV write (Q1), layer-0 norm without residual, `emb` once vs twice, real weights/KV/positions/page tables vs `randn` and a shared
block table, the MoE block and lm_head between scopes (cache state), graph replay (no host between kernels; k_norm possibly on a second queue), `max_num_partitions`
server-wide, decode `kv_cache_size` definition (`seq_len − 1`), A-attention's legacy regime with possible host contamination and block-16 rows, A-linear's backlog/settle regime and its
≈3 µs record cost, `DEBUG_CLR_MAX_BATCH_SIZE`, `SGLANG_USE_AITER`, chunked-prefill size, radix cache off, overlap scheduling, `ignore_eos`/greedy sampling, 48 layers
sharing one clock trajectory per forward.

## 9. Pilot (after dependent preflight checks): one node, TP1, ~10 prompts, both tracers
Prompts {1, 8, 64, 512, 1024, 1025, 4096, 4352, 4360, 8192}, G=8, graph on and off, rocprofv3 then kineto; deliver: the table (head + per-scope summary), the
checker output, the marker-attribution completeness, the rocprofv3-vs-host clock alignment used for the sidecar join, and unique attribution evidence.
Only then the full sweep. Outputs under `/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/insitu_b/<run>/` and scratch under
`/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling_b_<tag>/` (pre-created, `chmod a+rwX`, root-squash); nothing under `data/profiling/compute/`.
