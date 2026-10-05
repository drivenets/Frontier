<!-- Evidence E1 — Frontier profiler timed scopes (subagent code-archaeology report, worktree 284130b). Read-only; line numbers from `cat -n` at HEAD. Generated 2026-10-04 by the operator-boundary investigation session. Treat as evidence to verify, not authority. -->

# Timed-scope archaeology: Qwen3-30B-A3B MoE profiling (linear_op + attention)

Worktree `/home/dn/Frontier-qwen3-profiling`, HEAD `284130b`. All paths below are relative to that root; all line numbers are from `cat -n` at HEAD. Nothing was modified.

## 0. Which code actually ran (and where HEAD differs)

**Linear run** (`runs/2026-09-22_0849_linear_op_dense_200fwd_all_fixes/RUN.md`): job 21486, image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`, code **`4b8fce3`** (2026-09-22), `FRONTIER_LINEAR_ACTIVE_STEPS=200`, `LINEAR_PROFILE_METHODS=cuda_event`, "spun blocks of 25". The RUN.md does not print the command line; it comes from the sbatch, `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch:90-92`:
```
python -m frontier.profiling.linear_op.main --disable_ray --yes --device mi355x --models "$MODEL" --is_moe \
  --num_gpus "${LINEAR_NUM_GPUS:-8}" --num_tensor_parallel_workers ${LINEAR_TPS:-1 2 4 8} --precision BF16 --profile_method "$M" \
  --max_tokens 16384 --num_tokens_list $TOKENS --output_dir "${COLLECT_DIR:-data/profiling}" \
```
with `MODEL="${MODEL:-qwen3-a3b-30b-moe}"` (`:26`) and the RoPE torch fallback NOT forced (`:75-77`). Branch selection for the whole report: `--is_moe`, `--profile_method cuda_event`, `--disable_ray` (ProcessPoolExecutor, one process per GPU), no `--disable_replicated`, no `--attn_tp/--ffn_tp` (both default to `1 2 4 8`), no MTP.

`git diff --stat 4b8fce3 HEAD -- frontier/profiling/` touches only `common/cuda_timer.py` (+pause), `common/timer_stats_store.py` (+pause/resume), `linear_op/linear_op_wrapper.py` (+work-backlog kinds, +SETTLE_STEPS). `linear_op_impl.py`, `linear_op/main.py`, `layers/rotary_embedding.py`, `layers/layernorm.py`, `tensor_parallel_layers.py` are identical to 4b8fce3, so the scope contents cited below are exactly what ran. The backlog mechanism differs (see §C.3).

**Attention run** (`runs/2026-09-15_1142_attention_16k/RUN.md`): jobs 21310/21311, image `v0.5.17-rocm720-mi35x`, code **`0fcac5f`** (2026-09-15); command line printed in the RUN.md ("What was run"): `--attention_backend AITER --block_size {16|1} --num_tensor_parallel_workers 1 2 4 8 --max_seq_len 16384 --max_model_len 16384 --batch_size_list 1 ... 512 --decode_kv_cache_size_list 128 ... 16384 --enable_chunked_prefill_grid_search --enable_true_mixed --true_mixed_prefill_batch_sizes 1 2 --true_mixed_prefill_chunk_sizes 1024 4096 8192 ... --precision BF16 --profile_method cuda_event --num_gpus 4`. `git diff --stat 0fcac5f HEAD -- frontier/profiling/attention frontier/profiling/common` shows no change under `frontier/profiling/attention/` (only `cuda_timer.py`, `timer_stats_store.py`, `rotary_embedding.py` changed, none of which affect the attention scopes), so the attention code below is what ran. The sweep script adds `--true_mixed_prefill_kv_cache_size 0` (default, `profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh:135,341`) and `--max_pipeline_parallel_size 1` (`:328`).

**Model constants** (`data/config/models/qwen3-a3b-30b-moe.json`): `hidden_size 2048`, `num_attention_heads 32`, `num_key_value_heads 4`, `head_dim 128`, `vocab_size 151936`, `max_position_embeddings 262144`, `rope_theta 10000000`, `rope_scaling null`, `rms_norm_eps 1e-06`, `attention_bias false`, `model_type qwen3_moe`, `use_qk_norm true`, `moe_intermediate_size 768`, no `shared_expert_intermediate_size`. Derived: `norm=rms_norm` (`frontier/config/model_config.py:826-827`), `post_attn_norm=True` (`:629`), `is_moe=True` (`:611`), `mlp_hidden_dim=768` (`:620`), `use_qkv_bias=False` (`:639-640`, attention_bias present), `use_bias=False` (`:633`), `is_neox_style` left at dataclass default `True` (`:268`; never set in `_create_from_hf_json`), `uses_fused_add_norm=True` (`qwen3_moe` in `FUSED_ADD_NORM_MODEL_TYPE_ALLOWLIST`, `:191-200`; `frontier/profiling/common/model_config.py:216-219`), `use_qk_norm=True` (`:204-207`, also explicit in JSON), `share_expert_dim=None`. Architecture profile = GENERIC with `sharded_ops=("attn_pre_proj","attn_rope","attn_post_proj")` (`frontier/model_architectures.py:152-158`), so the attention module is `CausalSelfAttention` (`linear_op_impl.py:709-710`).

---

## A. Linear-op profiler: the forward and its scopes

### A.0 Forward skeleton (one `self.model(input_ids, positions)` call)
`frontier/profiling/linear_op/linear_op_impl.py:1088-1098`:
```
hidden_states = self.embed_tokens(input_ids)          # 1089  (timed "emb", result discarded)
for _ in range(self.num_repeat_steps):                  # 1091  (=1 for cuda_event, wrapper:254-258)
    hidden_states = self.embed_tokens(input_ids)      # 1092  (timed "emb" again)
    residual = hidden_states                          # 1093  (SAME tensor object)
    hidden_states, residual = self.block(positions, hidden_states, residual)
```
`GPTBlock.forward` → `_forward_with_post_attn_norm` because `post_attn_norm=True` (`:886-887`). Inside (`:906-949`): `input_layernorm` (fused add, `:919-922`) → `self.attn(...)` (`:923-926`) → `_maybe_pad_tensor` (no-op: padded_n_embd=2048, `:897-898`) → `post_attention_layernorm` (fused add, `:937-941`) → `self.mlp(hidden_states)` (`:942`) → return. `share_expert is None` (`:839-846`, no share_expert_dim).

Per forward, timed scopes in order: `emb`, `emb`, `input_layernorm`, `attn_pre_proj`, `attn_rope`, `attn_post_proj`, `post_attention_layernorm`, `mlp_up_proj`, `mlp_act`, `mlp_down_proj` (+ `forward_gpu_span` around the whole call in the GPU-bound pass). The three `mlp_*` scopes DO execute and are timed, then dropped from the CSV (see §C.4).

Inputs (`linear_op_wrapper.py:374-389`): `input_ids = randint(0, padded_vocab//TP, (num_tokens,))`; `positions = [i % 262144 for i in range(num_tokens)]` (`frontier/profiling/utils/__init__.py:239`) = `0..T-1` for every T ≤ 16384. Both built once per shape, before warm-up.

### A.1 `attn_pre_proj`
Construction, qk-norm branch (`use_qk_norm=True`), `linear_op_impl.py:429-456`:
```
q_with_gate_size = self.q_size * (2 if self.attn_output_gate else 1)      # 432, gate=False
qkv_output_size = (q_with_gate_size + 2 * self.kv_size) * world_size       # 433
self.qkv_proj = ColumnParallelLinear(config.embedding_dim, qkv_output_size, bias=config.use_bias or config.use_qkv_bias, gather_output=False, linear_metric_name=None, precision_op_name="attn_pre_proj", ...)  # 434-443
```
with `q_size = (32//TP)*128` (`:410,415`), `kv_size = kv_heads_per_worker*128` (`:416`), `kv_heads_per_worker = 4//TP` for TP≤4, `1` for TP=8 (replication, `_resolve_num_kv_heads_per_worker` `:39-52`). `ColumnParallelLinear` divides `output_size` by `world_size` (`tensor_parallel_layers.py:682`) and allocates ONE weight `[output_size_per_partition, input_size]` (`:723-730`), so the QKV weight is a single fused bf16 tensor of shape **[N, 2048]**, N = 4096/TP + 256·max(1, 4/TP): TP1 **5120**, TP2 **2560**, TP4 **1280**, TP8 **768** (TP8 carries the replicated KV head). K = 2048. Bias is None (`use_qkv_bias=False`). The layer's own `_linear_timer` is `CudaTimer(None)` (disabled) because `linear_metric_name=None` (`:439`, `cuda_timer.py:23,27`).

Timed scope (`linear_op_impl.py:503-526`):
```
with self._attn_pre_proj_timer:                                   # 503  CudaTimer("attn_pre_proj") (:427)
    qkv, _ = self.qkv_proj(hidden_states)                          # 504  GEMM
    q_and_gate, k, v = qkv.split([...], dim=-1)                    # 505-512  views, no kernel
    q = q_and_gate                                                 # 516
    q_by_head = q.view(*q.shape[:-1], -1, self.head_dim)           # 520  view
    q_by_head = self.q_norm(q_by_head)                             # 521  vllm rms_norm kernel
    q = q_by_head.view(q.shape)                                    # 522
    k_by_head = k.view(...); k_by_head = self.k_norm(k_by_head); k = k_by_head.view(k.shape)   # 524-526
```
`QKNorm.forward` → `RMSNorm(head_dim=128, norm_name=None)` (`:97-118`) → `layernorm.py:48-55`: `with self._norm_timer` (disabled, name None) → `vllm_rms_norm(x, weight, eps)` because `residual is None` (`:54-55`). So q_norm/k_norm **are inside** the `attn_pre_proj` timer, as 2 separate `rms_norm` launches on `[T, 32/TP, 128]` and `[T, kv/TP, 128]` (each allocates a new contiguous output). GEMM path: `ColumnParallelLinear.forward` (`:782-804`) → `apply_weights` (`:779-780`) → `_run_unquantized_gemm` (`:204-226`): on HIP (`torch.version.hip` not None, `:211`) it calls vLLM's `dispatch_unquantized_gemm()` (`:56-78`, `vllm.model_executor.layers.utils`) and then `gemm(layer, x, weight, bias)` (`:226`). **Cannot determine from this repo** which ROCm GEMM vLLM 0.9.2 dispatches to for small M (vLLM's `rocm_unquantized_gemm` has skinny-GEMM special cases); flag for anyone attributing small-T behaviour. Excluded from the scope: no all-gather (`gather_output=False`, `:797-802`), no bias, RoPE (separate scope), `torch.randn_like(q)` (`:546`, after the rope scope). `v` is computed and never used.

### A.2 `attn_rope`
Timed scope `linear_op_impl.py:543-544`: `with self._attn_rope_timer: q, k = self.rotary_emb(positions, q, k)`. `rotary_emb = get_rope(head_dim=128, rotary_dim=128, max_position=262144, base=1e7, is_neox_style=True, rope_scaling=None)` (`:485-492`). `get_rope` (`rotary_embedding.py:574-621`): fused op importable → skips vLLM's factory (`:600-611`), `rope_scaling is None` → Frontier's own `RotaryEmbedding` (`:616-621`), cached in `_LOCAL_ROPE_DICT`. Cache: `cos_sin_cache` `[262144, 128]` fp32 at init (`:225-227,240-249`), converted to the query dtype (bf16) on the first forward inside the scope (`:286-291`, warm-up only).

`RotaryEmbedding.forward` (`:251-302`); the branch that ran:
```
positions = positions.flatten()                                    # 293  view
vllm_ops.rotary_embedding(positions, query, key, self.head_size, self.cos_sin_cache, self.is_neox_style)   # 294-301
return query, key                                                  # 302  (in-place)
```
= **one launch** of `vllm._custom_ops.rotary_embedding` rotating **q and k together**, in place, `is_neox=True`, head_size 128, rotary_dim = full head, positions `0..T-1`. Tensors: the q/k returned by `q_by_head.view(q.shape)` / `k_by_head.view(k.shape)` — fresh contiguous outputs of `rms_norm`, shapes `[T, 4096/TP]` and `[T, 128·max(1,4/TP)]`. Fallback gates (`:117-133`): env `FRONTIER_PROFILING_FORCE_TORCH_FALLBACK`, `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK` (truthy values `1/true/yes/on`), or `TimerStatsStore().disabled`; none set by the sbatch (`:75-77` comment confirms "torch fallback is NOT forced any more"). The torch fallback path (`:258-276`, `_apply_rotary_pos_emb` `:136-179`) would be ~10+ kernels. The impl actually used is written per row to column `attn_rope_impl` (`linear_op_wrapper.py:458`, `rope_impl_name` `rotary_embedding.py:187-204`) and should read `vllm_kernel`.

### A.3 `attn_post_proj`
`linear_op_impl.py:473-482`:
```
self.o_proj = RowParallelLinear(config.num_q_heads * self.head_dim, config.embedding_dim, bias=config.use_bias,
    input_is_parallel=True, reduce_results=False, linear_metric_name="attn_post_proj", ...)
```
Weight `[output_size, input_size_per_partition]` = **[2048, 4096/TP]** (`tensor_parallel_layers.py:881,916-923`), bias None. Input: `attn_output = torch.randn_like(q)` (`linear_op_impl.py:546`, `[T, 4096/TP]`, generated **outside** every timer, after the rope scope) → `output, _ = self.o_proj(attn_output)` (`:547`). `RowParallelLinear.forward` (`:971-1000`): `input_is_parallel` → no scatter (`:982-983`); `apply_weights` (`:951-969`) → the timer wraps **only** `_run_unquantized_gemm(self, x, self.weight, None)` (`:968-969`); then `if self.reduce_results and self.world_size > 1:` all-reduce (`:988-990`) is **skipped** because `reduce_results=False` — no communication kernel anywhere in this profiler; bias add at `:994-996` is outside the timer and a no-op (bias None). So `attn_post_proj` = exactly one GEMM `[T,4096/TP]×[4096/TP,2048]`.

### A.4 `input_layernorm` and A.5 `post_attention_layernorm`
Construction `linear_op_impl.py:784-793` / `:802-815`: `RMSNorm(2048, norm_name="input_layernorm"|"post_attention_layernorm", eps=1e-6)`; `_use_inner_*_timer=True` so the block-level `input_layernorm_timer` / `post_attention_layernorm_timer` are `CudaTimer(None)` (`:848-863`) and the **timer is the RMSNorm's inner `_norm_timer`** (`layernorm.py:42,48`). Kernel: residual is not None in both calls → `vllm_fused_add_rms_norm(x, residual, weight, eps)` (`layernorm.py:56-58`) — **one fused kernel including the residual add** (vLLM's op is in-place on both x and residual).
- `input_layernorm` call (`linear_op_impl.py:919-922`): `hidden_states, residual = self.input_layernorm(hidden_states, residual)` where `residual = hidden_states` was assigned at `:1093` → **x and residual are the same tensor object** (the second `embed_tokens` output). Shape `[T, 2048]`.
- `post_attention_layernorm` call (`:937-941`): x = o_proj output `[T,2048]`, residual = the residual tensor returned by the fused input norm — distinct tensors. post_attn_dim = `padded_n_embd` = 2048 (`:799-801`, `profiling_plan.py:157` pads 2048 to a multiple of TP → unchanged).
- `add` scope: `self._profile_add` is forced False at `:765-767` (`if config.uses_fused_add_norm: self._profile_add = False`) → `add_timer = CudaTimer(None)` (`:864`); `with self.add_timer if self._profile_add else nullcontext()` (`:919,938`). **No `add` scope exists / no `time_stats.add` column**, even though `"add"` is in the plan's `enabled_ops` (MEMORY family profiling names `profiling_plan.py:46-57`, `frontier/operators/families.py:273-293` `profiling_key="add"`).
- TP>1: both norms run in every TP's forward (replicated path `profiling_plan.py:160-162,168-175,192-193`: `replicated_enabled = not disable_replicated`; `replicated_ops = [input_layernorm, post_attention_layernorm, add, emb]`). Their TP>1 measurements are discarded by the CSV writer (§C.5).

### A.6 `emb`
`linear_op_impl.py:1017-1025`: `VocabParallelEmbedding(151936, 2048, linear_metric_name="emb", reduce_results=False, world_size=TP, rank=0, pad_vocab_size=...)`; `pad_vocab_size = 151936 % TP != 0` → False for TP 1/2/4/8 (`linear_op_wrapper.py:237-239`). Weight `[151936/TP, 2048]` (`tensor_parallel_layers.py:439-450`). Timed scope `:458-483`:
```
with self._emb_forward_timer:                                      # 459  CudaTimer("emb") (:454)
    if self.tensor_model_parallel_size > 1: input_mask = ...; masked_input = input_.clone() - start; masked_input[input_mask] = 0   # 460-467
    else: masked_input = input_                                     # 468-469
    output_parallel = F.embedding(masked_input, self.weight, ...)   # 471-479
    if self.tensor_model_parallel_size > 1: output_parallel[input_mask, :] = 0.0   # 482-483
```
`reduce_results=False` → no all-reduce (`:484-489`). At **TP1 (the only TP whose emb rows survive)** the scope is a single `F.embedding` gather of `T` ids from `[151936, 2048]`. **Duplicate invocation**: `embed_tokens` is called at `:1089` and again at `:1092` → **2 `emb` samples per forward**; `get_stats` aggregates all samples of a scope (`timer_stats_store.py:50-71`), so `time_stats.emb.median` is the per-call median over 2×ACTIVE_STEPS samples (count=400, warmup_count=6 for the 200-forward run), NOT a per-forward sum; the docstring at `:44-47` acknowledges "a scope timed twice per forward". Ids are `randint(0, 151936/TP)` (`linear_op_wrapper.py:374-381`).

---

## C. Linear-op measurement machinery

### C.1 Timer
`frontier/profiling/common/cuda_timer.py`: `CudaTimer.__enter__` in `CUDA_EVENT` mode records a start event (`:57-59`); `__exit__` records an end event and appends the pair (`:96-101`); names are lower-cased with `vidur_` prefix, stripped again in `record_time` (`:19-22`, `timer_stats_store.py:27`). `profile_method` comes from the singleton `TimerStatsStore(profile_method=...)` (`linear_op_wrapper.py:227`). `get_stats` (`timer_stats_store.py:41-71`) converts pairs with `start.elapsed_time(end)` (ms), drops the first `warmup_count` samples per scope (`mark_warmup_end` `:37-39`), and emits min/max/mean/median/std/count/warmup_count plus the full `samples` JSON list. No synchronisation inside any scope. `record_function` mode (`:54-56,94-95`) was also run by the sbatch into a separate `linear_op_kernel_only.csv` — not the file in question.

### C.2 Two passes, GPU-bound column
`linear_op_wrapper.py:423-446`: pass 1 legacy (`backlog_ms=0`) → `time_stats_hostbound`; pass 2 (CUDA_EVENT only) with `requested_backlog_ms = 4 × legacy loop wall` (`:433`, `BACKLOG_REQUEST_FACTOR` `:58`) → **primary `time_stats`**. `_timed_pass` (`:291-370`): `clear_stats` → sync → 3 warm-up forwards (`WARMUP_STEPS=3`, `:29`) → sync → `mark_warmup_end` → clock probe → `blocks = ceil(ACTIVE_STEPS/25)` = 8 (`:324`, `BACKLOG_BLOCK_STEPS` `:34`) → per block: enqueue backlog (`:330`), [HEAD only: pause timers, 8 settle forwards, resume `:331-336`], then 25 timed forwards (`:337-343`) → sync → clock probe. `gc.disable()` around the loop (`:310-311`). `ACTIVE_STEPS` from env (`:30`) = 200 for this run.

### C.3 Backlog kind: what 4b8fce3 did vs HEAD
At **4b8fce3** (`git show 4b8fce3:…/linear_op_wrapper.py` lines 259-262): `backlog_events.append(_enqueue_gpu_backlog(backlog_ms / blocks))` followed immediately by the timed loop — the backlog was the `torch.cuda._sleep` spin (HEAD `:169-184`), **no settle forwards, no `BACKLOG_KIND`**. HEAD (`:45-46`, commits `3a8eb55`/`88da6a4` dated 2026-09-23) defaults to `BACKLOG_KIND=gemm_alloc` (chain of 4096² bf16 GEMMs, `:107-153`) + `SETTLE_STEPS=8` untimed forwards with `TimerStatsStore.pause()` (`:331-336`; `cuda_timer.py:50-52,91`). So the 2026-09-22 dataset = sleep-spin blocks of 25, no settle; the 2026-09-23 "settle8" dataset (which `regressor_bench/dataset.py` consumes, §E) = GEMM backlog + settle 8. Per-row columns `settle_steps`/`backlog_kind` exist only at HEAD (`:470-471`).

### C.4 `forward_gpu_span`, dropped scopes
`forward_gpu_span` = `CudaTimer("forward_gpu_span")` (`FORWARD_SPAN_SCOPE` `:61`) wrapping each whole `self.model(...)` call in the GPU-bound pass only (`record_forward_span=True`, `:305,316-317,340-341`); it brackets warm-up and timed forwards but not settle forwards. `--is_moe`: `build_profiling_plan` omits FFN ops from `enabled_ops` (`profiling_plan.py:202-204`), but `GPTBlock` still builds `MLP` whenever `_ffn_sharded_enabled` (`linear_op_impl.py:828-834`; true because `ffn_tp` defaults to all TPs, `main.py:400-401`, `profiling_plan.py:152`) with hard-coded timer names `mlp_up_proj`/`mlp_down_proj`/`mlp_act` (`:581,604,608`) and `mlp_hidden_dim=768` (`:564-566,833`; gated → up_proj N=2·768/TP, `:575-583`). These GEMMs run and are timed in every forward; the columns are removed only at write time by `filter_mlp_columns` (`main.py:783-808`, called at `:896-898`; drops any column containing the substring, including `time_stats_hostbound.mlp_*`). **Flag**: the per-forward GPU work therefore includes three MoE-irrelevant dense-MLP kernels (K or N = 768/TP) between the timed attention scopes; whether `forward_gpu_span` closure in the RUN.md counted them cannot be determined from the repo.

### C.5 TP>1 rows for replicated ops (the drop/split code)
`main.py:588-610`: `_should_split = TP>1 and replicated_ops and not disable_replicated`; `split_replicated_result` (`frontier/profiling/utils/replicated_ops.py:10-47`) copies the result into a **sharded row** (TP=N, per-op dicts minus replicated names) and a **replicated row** with `num_tensor_parallel_workers` forced to 1 (`:41`) for all four per-op dict fields (`PER_OP_DICT_FIELDS` `:7`). Then `deduplicate_tp1_rows(all_results, ("num_tokens","model_arch","measurement_type"))` (`main.py:750-752`; `replicated_ops.py:50-78`) keeps the **first** TP=1 row per key. TPs are iterated ascending (`_resolve_tp_ranges` sorts, `main.py:403,568`), so the genuine TP1 run's single unsplit row (which holds attn_* and the replicated ops together) wins and every TP>1 replicated row is discarded. Consequence in the CSV: `input_layernorm`/`post_attention_layernorm`/`emb` have values only at TP1 and NaN in TP>1 rows (`expand_dict_columns` `main.py:769-780` json-normalises the dicts). Note the TP1 replicated values were measured in a forward that also ran the TP1 (largest) attention and MLP GEMMs.

---

## B. Attention profiler (AITER backend)

### B.0 Harness
`frontier/profiling/attention/attention_wrapper.py:30-31`: `WARMUP_STEPS = 3`, `ACTIVE_STEPS = 50` (constants, no env). `profile()` cuda_event branch (`:355-370`): `clear_stats` → 3 warm-up forwards → `synchronize` → `mark_warmup_end` → 50 forwards with no sync → `synchronize` → `get_stats`. Identical loops in `profile_mixed` (`:472-489`) and `profile_true_mixed` (`:552-567`). `begin_forward` is called once per shape before the loop (`:333,531`) and `end_forward` after (`:372,569`), so descriptor/plan construction is outside all scopes. **No GPU-bound fix**: `grep -rn -i "backlog\|spin\|_sleep\|settle\|pause()" frontier/profiling/attention/` returns nothing, and `attention_wrapper.py` never calls `TimerStatsStore.pause`. The attention CSV is single-column, un-backlogged event timing (the same regime as the linear profiler's `time_stats_hostbound`), with "run 0 slowest in ~88 % of prefill rows" per the RUN.md. Timers: `BaseAttentionWrapper.get_timer` → `CudaTimer(operation, layer_id)` per (op, layer_id=None) (`backends/base_attention_wrapper.py:56-73`). Per-worker heads: `num_q_heads = 32/TP`, `num_kv_heads = max(1, 4//TP)` (`common/model_config.py:407-453`; TP8 → 1 replicated KV head), `head_dim=128` (`:455-463`). KV cache `(2, num_blocks, block_size, kv_heads, 128)` bf16 randn (`aiter_attention_wrapper.py:184-203`), sized by `get_max_num_blocks` (`utils/__init__.py:413-445`, 0.9×total memory ÷ 48 layers since `max_pipeline_parallel_size 1`).

Every forward enters all five scopes unconditionally (`:385-449`); in prefill rows the `attn_decode` scope body is skipped by `if self.contains_decode:` (`:424`) but the event pair is still recorded (≈0 ms), and symmetrically `attn_prefill` ≈ 0 in decode rows. `attn_input_reshape` (`:385-388`, `contiguous().reshape` on already-contiguous randn tensors → views) and `attn_output_reshape` (`:448-449`, view) are trivial; they are the `_ALLOW_ZERO_CUDA_OPS` (`attention_wrapper.py:32`).

Inputs: `q/k/v = randn(total_tokens, heads·128)` per shape (`attention_wrapper.py:132-150`). **Block tables for standard rows** (`:200-215`): every sequence gets `block_table=list(range(num_blocks))` — all B sequences share pages `0..num_blocks-1`; `total_len = tokens_per_seq + kv_cache_size`, `processed_len = kv_cache_size` (`:211-212`; `SequenceMetadataProxy` `sequence_proxy.py:20-31` sets `prompt_chunk_len = total_len - processed_len` for prompts). True-mixed rows (`:276-317`) assign disjoint consecutive page ranges per sequence via `next_block_index`.

### B.7 `attn_kv_cache_save`
`backends/aiter_attention_wrapper.py:394-401`:
```
with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
    key_cache[self._write_block_index, self._write_block_offset] = key        # 400
    value_cache[self._write_block_index, self._write_block_offset] = value    # 401
```
Two PyTorch advanced-index scatters (no aiter kernel) of `key/value [N, kv_heads, 128]` into `kv_cache[0]`/`kv_cache[1]` (`:391-392`, contiguous views). Write plan from `begin_forward`: prefill → one slot per new token `processed..processed+chunk` (`:267-271`); decode → one slot at `context_len-1` per sequence (`:285-287`). Because standard decode rows share one block table, **all B decode sequences write the same (block, offset) slot** (duplicate indices); standard prefill rows have batch 1 (`attention_input.py:15-16` rejects prefill batch≠1); true-mixed rows write prefill tokens then decode tokens into disjoint pages. N = `batch·chunk` (prefill), `batch` (decode), `Σchunk + decode_bs` (true-mixed).

### B.8 `attn_prefill`
`:403-421`:
```
mha_batch_prefill_func(query[: self.num_prefill_tokens], key_cache, value_cache, self._prefill_cu_seqlens_q,
    self._prefill_kv_indptr, self._prefill_kv_pages, self._prefill_max_q, self._prefill_max_kv,
    softmax_scale=self.softmax_scale, causal=True, out=output[: self.num_prefill_tokens], kv_last_page_lens=self._prefill_kv_last_page_lens)
```
`softmax_scale = 1/sqrt(128)` (`:146`). Paged layout: `_page_descriptors` (`:205-226`) builds `kv_indptr`, `kv_page_indices` (first `ceil(seq_len/block_size)` entries of the block table), `kv_last_page_lens`. **Chunked prefill representation**: `prefill_seq_lens` = `processed + chunk` = `kv_cache_size + prefill_chunk_size` (`:259-265`), query length = `prefill_chunk_size` (`cu_seqlens_q`, `:296-301`), so the kernel attends `chunk` queries against `kv_cache_size + chunk` paged keys with `causal=True` (bottom-right alignment derived natively, docstring `:37-41`). Grid (`utils/__init__.py:345-358`): chunk sizes from `get_attention_prefill_chunk_sizes_to_profile` (`:273-289`) × `kv_cache_size = partition_index·chunk` (`:348-352`), plus full prefills with `kv_cache_size=0`; `batch_size=1` always (`:354,358`). Row fields `prefill_chunk_size`, `kv_cache_size`, `is_prefill=True` (`attention_wrapper.py:400-403`); `add_chunked_prefill_metadata` adds `chunk_start_token/chunk_end_token/is_chunked_prefill_sample` (`metadata_utils.py:22-29`).

### B.9 `attn_decode`
`:423-446`:
```
paged_attention_ragged(output[self.num_prefill_tokens :], self._workspace, decode_query, key_cache, value_cache, self.softmax_scale,
    self._decode_kv_indptr, self._decode_kv_pages, self._decode_kv_last_page_lens, self.block_size, self._decode_max_num_partitions,
    None, "auto", "NHD", 0.0, self._k_scale, self._v_scale, None, _AITER_PARTITION_SIZE_ROCM)
```
`_AITER_PARTITION_SIZE_ROCM = 256` (`:95`); `max_num_partitions = (max(decode_seq_lens) + 255) // 256` (`:316-318`) with `decode_seq_len = kv_cache_size + 1` (`attention_wrapper.py:211` / `:301`); workspace `batch·q_heads·partitions·128·4 + 2·batch·q_heads·partitions·4` bytes, allocated in `begin_forward` (`:319-321,332-348`) outside the scope. Uniform KV per sequence: all `batch_size` sequences get the same `kv_cache_size` (`attention_wrapper.py:200-215`; grid `utils/__init__.py:381-383`) and, for standard rows, the same pages.

### True-mixed rows
One forward contains both kernels: `mha_batch_prefill_func` on `query[:num_prefill_tokens]` under the `attn_prefill` event pair and `paged_attention_ragged` on `query[num_prefill_tokens:]` under the separate `attn_decode` event pair (`:403-446`), sequential on one stream — **two kernel calls, two event pairs, one forward**. Composition from `get_true_mixed_attention_input_combinations` (`utils/__init__.py:934-938`): `prefill_seq_lens=[chunk]*prefill_bs` (1024/4096/8192 × bs 1,2), `prefill_kv_cache_sizes=[0]*bs` (prefill part is a full prefill, not chunked), `decode_kv_cache_sizes=[kv]*decode_bs`. Row compatibility fields: `is_prefill=True, batch_size=total, prefill_chunk_size=0, kv_cache_size=0, mode="true_mixed", is_true_mixed_batch=True` (`true_mixed_batch_input.py:102-124`) — so `prefill_chunk_size/kv_cache_size` are **zero placeholders** in true-mixed rows; use `prefill_seq_lens`/`decode_kv_cache_sizes`/`total_prefill_tokens`. Written to `attention_true_mixed*.csv` and the combined file (`attention/main.py:1991-2009`). `time_stats` flattening: `pd.json_normalize(...).add_prefix("time_stats.")` (`:1237-1241`, `:1609-1613`).

---

## E. `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench/dataset.py`
- Ops consumed: `SHARDED_OPS=("attn_pre_proj","attn_post_proj","attn_rope")`, `REPLICATED_OPS=("input_layernorm","post_attention_layernorm","emb")` (`:32-34`); TPs `(1,2,4,8)` (`:35`); 15 regressors (`:11`).
- Label: `time_stats.<op>.median` (`:87-88,109`) in ms, plus `y_std/y_min/y_max/n_samples` from `.std/.min/.max/.count` (`:110-113`).
- Filters: **no explicit TP or num_tokens filter**; rows whose label is NaN are dropped (`:116`) — this is how TP1-only ops are handled ("replicated ops at TP>1, by design", `:95-96`). `_check_coverage` (`:127-137`) requires exactly 3327 token values and 3327 rows per (op, tp), raises if a replicated op has rows at TP≠1, and rejects non-positive labels. Duplicate `(num_tokens, TP)` cells raise (`:97-98`). Feature prefixes forbidden: `time_stats_hostbound.`, `host_wall_per_forward_ms`, `host_enqueue_per_forward_ms`, `gpu_backlog_ms`, `sclk_mhz_`, `legacy_host_bound` (`:43-50`).
- **Flag**: the file it loads and md5-verifies is `.../linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv` (`:22-30`, md5 `3635e4d3…`), i.e. the 2026-09-23 collection (GEMM backlog + settle 8, code ≥ `3a8eb55`/`88da6a4`), **not** the 2026-09-22 run whose RUN.md this task named (sha256 `73fb3e8f…`, code 4b8fce3, sleep spin, no settle). There is no `runs/2026-09-23_*` RUN.md in the repo (`ls data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/` ends at `2026-09-22_0849_…`). Scope contents (§A) are identical for both since `linear_op_impl.py` did not change; only the backlog/settle mechanism differs.

---

## F. Things I could not determine / caveats
1. Which ROCm GEMM kernel vLLM 0.9.2's `dispatch_unquantized_gemm` selects per shape (skinny-GEMM paths etc.) — outside this repo (`tensor_parallel_layers.py:214-226` only shows the dispatch call).
2. Internals of `vllm._custom_ops.rotary_embedding`, `vllm_fused_add_rms_norm`, `aiter` kernels (single launch vs. multiple) — assumed per their call sites; the repo's own comments claim one launch for RoPE (`rotary_embedding.py:596-599`).
3. Whether the RUN.md's "whole-forward closure 0.96–0.99" sums the dropped `mlp_*` scopes (they are present in `time_stats` until CSV write).
4. The attention RUN.md describes `v0.5.17` + `0fcac5f` "+ sbatch fixes (later committed in b65842c)"; I verified the profiler Python files are unchanged since 0fcac5f but did not inspect b65842c's sbatch deltas.
5. The `add` op appears in the plan's `enabled_ops` yet has no timer (fused norm) — any consumer expecting `time_stats.add` will find no column; not an error, but worth noting.
