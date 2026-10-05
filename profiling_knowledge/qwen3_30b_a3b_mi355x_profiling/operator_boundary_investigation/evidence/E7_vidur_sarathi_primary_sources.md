# Evidence E7 — Vidur / Sarathi-Serve primary sources for the inherited operator partition

Fetched 2026-10-04 from public pinned sources (quoted verbatim). Vidur repo `microsoft/vidur` main @ `abae7f63` (2026-08-24);
Sarathi-Serve branch `vidur`. Paper: Agrawal et al., "Vidur: A Large-Scale Simulation Framework for LLM Inference", MLSys 2024,
arXiv 2405.05465 (HTML v2 rendering used for text extraction; section 4.3 "Profiler", 4.4 "Runtime Estimator").

## 1. Paper, §4.3 (operator triaging) — documented design intent

> "Operator Triaging. The profiler analyzes different operators to identify their input dependencies. We find that all the operators can be placed on one of the three buckets:"
> "Token-level Operators: The operand dimensions for operations like linear, and activation functions depend on model architecture, however, their runtime only depends on the total number of tokens being processed (prefill plus decode) in the batch."
> "Sequence-level Operators: The attention operation depends not only on the number of tokens in the current batch but also the context length of each request."
> "Communication Operators: The runtime of communication operations like all-reduce and all-gather depend only on the amount of data to be transferred, independently of the model architecture."
> "Profiling Token-level Operators. There are two broad categories of token-level operators - matrix multiplications and simple point-wise apply or reduction operations, like addition, normalization, and activation functions. Based on the model specification, we generate all the different tensor parallel sharding configurations and profile each combination. This approach allows us to obtain traces for different parallelism configurations while profiling on a single GPU. We use standard PyTorch kernels for profiling these operations and measure their performance using CUPTI."
> "Profiling Sequence-level Operators. … First, we separately profile the attention kernels for prefill and decode phases due to their difference in compute characteristics."
> "… the attention decode operation is largely memory-bound … the runtime of this operation is mainly determined by the total data volume that needs to be fetched from the KV-Cache and not the exact split of context lengths between different requests in the batch. … sequence parallel attention kernels such as PagedAttention v2, and FlashDecoding can effectively handle such skews, and thus it is sufficient to model decode based on total KV-Cache reads."
> "Profiling Communication Operators. … Since these operations don't depend on model-specific characteristics, we independently profile these kernels ahead of time in a model-agnostic manner for different topologies."
> §4.4: "simple polynomial regression does not capture the non-linear runtime characteristics of CUDA kernels due to phenomenons like tile and wave quantization. For our scenario, we find that random forest (RF) regression models achieve the right balance between data frugality and fidelity."
> §3/§4: "Vidur uses the key insight that the large majority of LLMs share similar architectures that can be decomposed into a small set of token-level, sequence-level and communication operators."; "Vidur-Bench: … profiling information for popular hardware like A100 and H100 GPUs".

Nothing in the paper motivates the *individual* scope names (`attn_rope`, `attn_kv_cache_save`, …); they come from the Sarathi-Serve instrumentation enum below.

## 2. Vidur / Sarathi code (pinned)

- `vidur/docs/profiling.md`: "Checkout branch `vidur`" of `microsoft/sarathi-serve`; "We use this reference model to profile only the MLP operations of all the models so the attention operations are no-op'ed here."; "For compute profiling (mlp and attention), 1 GPU is enough even for tensor parallel degrees greater than 1."; network profiling is separate (`allreduce.csv`, `send_recv.csv`); "CPU Overhead Profiling … tie the simulator closely to the implementation eg. `vLLM`".
- `vidur/profiling/mlp/mlp_impl.py:36-53`: `qkv_proj = ColumnParallelLinear(..., gather_output=False, linear_metric_name="attn_pre_proj", ...)`; `o_proj = RowParallelLinear(..., input_is_parallel=True, reduce_results=False, linear_metric_name="attn_post_proj", ...)`. `:64,:69-70`: `self._attn_rope_timer = CudaTimer("attn_rope")` / `with self._attn_rope_timer: q, k = self.rotary_emb(positions, q, k)`. `:72`: `attn_output = torch.randn_like(q)`. `:86-93`: `VocabParallelEmbedding(..., linear_metric_name="emb", reduce_results=False, ...)`. `:97-106`:
  ```
  def forward(self, input_ids, positions):
      hidden_states = self.embed_tokens(input_ids)
      residual = hidden_states
      for _ in range(self.num_repeat_steps):
          hidden_states = self.embed_tokens(input_ids)
          hidden_states = self.block(positions, hidden_states, residual)
  ```
  → the double embedding call per forward is Vidur's, inherited by Frontier (`linear_op_impl.py:1089,1092`).
- `vidur/profiling/attention/attention_wrapper.py:25-26`: `WARMUP_STEPS = 2`, `ACTIVE_STEPS = 5`; `:105-113`: per-sequence **random** block tables `np.random.default_rng().integers(low=0, high=self.max_num_blocks - 1, size=num_blocks)`; `:122`: "batch size is always 1 for prefill and can be different for decode". Frontier upstream's first commit (6897819, `attention_wrapper.py:146`) replaced this with `block_table=list(range(num_blocks))` for every sequence (shared pages) — an upstream-Frontier change, present unchanged at the worktree (`attention_wrapper.py:213`).
- Sarathi-Serve `sarathi/metrics/constants.py:12-30` (`OperationMetrics`): `ATTN_PRE_PROJ`, `ATTN_PRE_PROJ_ALL_GATHER`, `ATTN_POST_PROJ`, `ATTN_POST_PROJ_ALL_REDUCE`, `ATTN_KV_CACHE_SAVE`, `ATTN`, `ATTN_PREFILL`, `ATTN_DECODE`, `ATTN_ROPE`, `ATTN_INPUT_RESHAPE`, `ATTN_OUTPUT_RESHAPE`, `EMBED_LINEAR`, `EMBED_ALL_REDUCE`, `INPUT_LAYERNORM`, `POST_ATTENTION_LAYERNORM`, `NORM`, `ADD` — the nine A labels are this enum. The all-gather / all-reduce entries show that communication was named as *separate* operators from the start.
- Sarathi `flashinfer_attention_wrapper.py:74-75`: "We perform both prefill and decode attention in a single call to batched prefill kernel." with separate `prefill_wrapper`/`decode_wrapper` begin_forward calls (`:150-170`) — i.e. the original backend already had the prefill/decode split as two FlashInfer wrappers.
- `vidur/execution_time_predictor/sklearn_execution_time_predictor.py:456-476`: linear ops incl. `attn_rope`, `add` trained on `["num_tokens"]`; `:482-492`: `attn_kv_cache_save` trained on `["num_tokens"]` from the attention CSV, `num_tokens = max(prefill_chunk_size, batch_size)` (`:209-211`); `:565-576`: `attn_prefill` on `["kv_cache_size","prefill_chunk_size_squared"]`, `attn_decode` on `["batch_size","kv_cache_size"]`; `:132-140,147-153`: missing norm/add/kv_cache_save columns default to 0. `vidur/entities/execution_time.py:59-85`: attention-layer time = pre_proj + post_proj + rope + kv_cache_save + decode + prefill + tensor_parallel_communication_time + attn_norm; MLP time adds its own tensor_parallel_communication_time; block = attn + mlp + add — the two-all-reduce-per-layer additive structure Frontier keeps.
