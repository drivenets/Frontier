<!-- Evidence E6 — SGLang v0.5.18 (commit 71de97b2, byte-identical to the approved image's wheel for every file cited here EXCEPT aiter_backend.py, whose image line numbers are ~120 higher after line 1936; see E5) — batch regimes, graph execution, TP communication, embedding, GEMM/norm dispatch. Subagent report generated 2026-10-04. Treat as evidence to verify, not authority. -->

# SGLang v0.5.18 (71de97b) — batch regimes, graph execution, communication, embedding, GEMM/norm dispatch
Target: Qwen3-30B-A3B BF16, ROCm/MI355X (gfx950), aiter attention backend, page_size 1, TP1/TP4. Paths are relative to `python/sglang/`; sources were read from the pinned tree under `scratchpad/sglang_ref/src/`. Modules I had to fetch in addition to the pre-downloaded set (same commit): `srt/managers/scheduler.py`, `srt/managers/schedule_policy.py`, `srt/layers/attention/base_attn_backend.py`, `srt/distributed/communication_op.py`, `srt/model_executor/cuda_graph_config.py`, `srt/model_executor/cuda_graph_buffer_registry.py`, `kernels/fused_op.py`, `srt/layers/moe/utils.py`.

---

## 1. Forward modes and batch composition

**Conclusion.** ForwardMode has EXTEND, DECODE, MIXED, IDLE, TARGET_VERIFY, DRAFT_EXTEND_V2, PREBUILT, SPLIT_PREFILL, DLLM_EXTEND. A MIXED batch is formed only when `--enable-mixed-chunk` (default **False**) is on *and* chunked prefill is enabled: the scheduler calls `running_batch.prepare_for_decode()` then `new_batch.mix_with_running(running_batch)`, appending every running decode request as a **1-token extend row** (`extend_lens += [1]`, `prefix_lens += [seq_len-1]`). The attention backend treats MIXED exactly as EXTEND (`is_extend()` is true for MIXED; the base dispatcher routes MIXED to `forward_extend` on non-NPU) — **one `mha_batch_prefill_func` call over all rows**, with `qo_indptr = cumsum(seq_lens - prefix_lens)` and `kv_indptr = cumsum(seq_lens)`. There is no separate decode kernel call for the decode rows. Chunked prefill splits a prompt by setting `req.extend_range = [len(prefix_indices), len(prefix_indices)+trunc_len)`; for a later chunk the previous chunks were cached into the radix tree so `prefix_indices` (hence `extend_prefix_lens`) equals the number of tokens already prefilled (plus any radix hit). By default (no mixed chunk) prefill chunks and decode steps run as **separate** EXTEND and DECODE forwards.

**Evidence.**

- Mode enum — `srt/model_executor/forward_batch_info.py:98-122`:
  ```
  EXTEND = auto()
  DECODE = auto()
  # Contains both EXTEND and DECODE when doing chunked prefill.
  MIXED = auto()
  ```
  (then `IDLE`, `TARGET_VERIFY`, `DRAFT_EXTEND_V2`, `PREBUILT`, `SPLIT_PREFILL`, `DLLM_EXTEND` at :107-122)
- MIXED counts as extend — `forward_batch_info.py:127-135`:
  ```
  def is_extend(self, include_draft_extend_v2: bool = False):
      return (
          self == ForwardMode.EXTEND
          or self == ForwardMode.MIXED
  ```
  Graph-eligible modes exclude EXTEND/MIXED — `forward_batch_info.py:177-183`:
  ```
  def is_cuda_graph(self):
      return (
          self == ForwardMode.DECODE
          or self == ForwardMode.TARGET_VERIFY
  ```
  (continues `or IDLE or DLLM_EXTEND`).
- Mixed-chunk gate — `srt/managers/scheduler.py:1180-1182`:
  ```
  self.is_mixed_chunk = (
      self.chunked_prefill_size is not None and get_schedule().enable_mixed_chunk
  )
  ```
  Default — `srt/server_args.py:981-985`: `enable_mixed_chunk: A[bool, "Enabling mixing prefill and decode in a batch when using chunked prefill.", ...] = False`.
- Where MIXED is formed — `scheduler.py:3431-3444`:
  ```
  if (
      self.is_mixed_chunk
      and not running_batch.is_empty()
      and not (new_batch.return_logprob or running_batch.return_logprob)
  ```
  ```
      running_batch.filter_batch()
      if not running_batch.is_empty():
          running_batch.prepare_for_decode()
          new_batch.mix_with_running(running_batch)
  ```
  Note `prepare_for_decode` bumps `self.seq_lens = self.seq_lens + 1` (`srt/managers/schedule_batch.py:3078`), so each decode row's seq_len already includes the new token.
- Decode rows become 1-token extends — `schedule_batch.py:2749-2775`:
  ```
  def mix_with_running(self, running_batch: ScheduleBatch):
      self.forward_mode = ForwardMode.MIXED
  ```
  ```
  self.prefix_lens = self.prefix_lens + [
      len(r.origin_input_ids) + len(r.output_ids) + delta
      for r in running_batch.reqs
  ]
  self.extend_lens = self.extend_lens + [1] * running_bs
  ```
  (`delta = 0 if self.enable_overlap else -1`, :2765)
- ForwardBatch wiring — `forward_batch_info.py:732-736`:
  ```
  if batch.forward_mode.is_decode_or_idle():
      extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
  else:
      extend_seq_lens = batch.extend_lens
      extend_prefix_lens = batch.prefix_lens
  ```
- Backend dispatch: MIXED → `forward_extend` (the `forward_mixed` branch is NPU-only) — `srt/layers/attention/base_attn_backend.py:227-250`:
  ```
  if forward_batch.forward_mode.is_idle(): ...
  elif forward_batch.forward_mode.is_decode():
      return self.forward_decode(
  ```
  ```
  elif forward_batch.forward_mode.is_mixed() and is_npu():
      return self.forward_mixed(
  ...
  else:
      return self.forward_extend(
  ```
- aiter metadata for extend (incl. MIXED) is built once for the whole batch from `extend_prefix_lens` — `srt/layers/attention/aiter_backend.py:1280-1281, 1353-1360`:
  ```
  else:
      prefix_lens = forward_batch.extend_prefix_lens
  ```
  ```
  self.indices_updater_prefill.update(
      forward_batch.req_pool_indices,
      forward_batch.seq_lens,
      forward_batch.seq_lens_sum,
      prefix_lens,
  ```
  and `AiterIndicesUpdaterPrefill.update_single_wrapper` — `aiter_backend.py:2727-2728, 2756-2759`:
  ```
  kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)   # paged_kernel_lens = seq_lens
  ```
  ```
  extend_lens = seq_lens - prefix_lens

  qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
  ```
  So a decode row in MIXED has qo length 1 and kv length = full seq_len inside the same prefill kernel.
- Chunk splitting — `srt/managers/schedule_policy.py:1380-1381, 1409-1412`:
  ```
  # Make sure at least one page is available
  trunc_len = chunk_tokens_limit // self.page_size * self.page_size
  ```
  ```
  # Chunked prefill
  req.set_extend_range(
      len(req.prefix_indices), len(req.prefix_indices) + trunc_len
  )
  ```
  `chunk_tokens_limit = self.rem_chunk_tokens` (:1239), seeded from `chunked_prefill_size` via `PrefillAdder(..., chunked_prefill_size, running_bs if self.is_mixed_chunk else 0, ...)` (`scheduler.py:3257-3265`; in mixed mode `rem_chunk_tokens -= num_mixed_decode_tokens`, `schedule_policy.py:537-538`).
- Later chunks — `scheduler.py:3276-3278`:
  ```
  if self.chunked_req is not None:
      self.chunked_req.init_next_round_input()
      self.chunked_req = adder.add_chunked_req(self.chunked_req)
  ```
  `init_next_round_input` re-matches the radix tree and sets `self.prefix_indices = match_result.device_indices` (`schedule_batch.py:1369-1380`); `add_chunked_req` then sets `req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)` with `new_len = min(cand_extend_input_len, _rem_tokens)` (`schedule_policy.py:1029-1031`). `prepare_for_extend` derives `prefix_lens = [len(r.prefix_indices) ...]`, `extend_lens = [r.extend_range.length ...]`, `seq_lens = [r.extend_range.end ...]` (`schedule_batch.py:2380-2383`). Hence for chunk k, `extend_prefix_lens = tokens already prefilled (chunks 1..k-1 + radix hit)`.
- `--chunked-prefill-size` default is memory-tiered; for ≥160 GB devices (MI300/MI355 class): `server_args.py:4877-4884`:
  ```
  # B200, MI300
  # (chunked_prefill_size 16k, max_bs 512)
  if self.chunked_prefill_size is None:
      self.chunked_prefill_size = 16384
  ```
  (I did not verify the `gpu_mem` detection helper itself; flagged below.)

---

## 2. CUDA/HIP graph execution

**Conclusion.** Only the **decode** phase is graph-captured by default; the decode runner captures `ForwardMode.DECODE` (TARGET_VERIFY / DLLM_EXTEND for spec/dLLM). Default decode backend is `full`. The prefill (EXTEND) graph default backend is `breakable` on CUDA but **`tc_piecewise` on non-CUDA, and tc_piecewise is auto-disabled on HIP** ("non-CUDA hardware" rule) unless the user explicitly sets `--cuda-graph-backend-prefill`; the prefill runner then becomes the `EagerRunner`, so on ROCm **EXTEND and MIXED run eagerly by default**. MIXED/EXTEND never run under the decode graph (`is_cuda_graph()` excludes them). Default decode capture sizes: `[1,2,4,8,12] + range(16,257,8) + range(272,512,16) + range(512,max_bs+1,32)` capped by `cuda_graph_config.decode.max_bs` (memory-tiered: 512 for ≥160 GB class GPUs) and by `req_to_token_pool.size`. Batches are **padded up to the next captured bucket**; padded rows get `seq_lens = seq_len_fill_value` (aiter: 1), and the attention metadata (`kv_indptr = cumsum(seq_lens)`) is built over the padded `bs`, so the padded rows run a 1-token-KV attention. `can_run_graph` = mode is DECODE/TARGET_VERIFY/IDLE/DLLM_EXTEND, `bs <= max_bs` (or exact bucket match if `--disable-cuda-graph-padding`), no `replace_embeds`, spec width match, encoder/TBO/ngram checks. ROCm-specific graph lines in `server_args.py` are only: tc_piecewise disabled on HIP; DLLM disables graphs on HIP; `triton_attention_num_kv_splits = 16` on HIP (irrelevant to aiter).

**Evidence.**

- Default per-phase backends — `srt/model_executor/cuda_graph_config.py:109-128`:
  ```
  def default_prefill_backend() -> str:
      """BCG (breakable) is the prefill default on CUDA only; other platforms
      (HIP/NPU/...) keep tc_piecewise until BCG is validated there. ...
      return Backend.BREAKABLE if is_cuda() else Backend.TC_PIECEWISE
  ```
  ```
  decode: PhaseConfig = field(
      default_factory=lambda: PhaseConfig(backend=Backend.FULL)
  )
  prefill: PhaseConfig = field(
      default_factory=lambda: PhaseConfig(backend=default_prefill_backend())
  ```
- HIP auto-disable of tc_piecewise — `srt/server_args.py:4548-4557, 4604-4607, 4662-4664`:
  ```
  if (Phase.PREFILL, "backend") in self._cuda_graph_config_locked:
      return
  ```
  ```
  (
      "non-CUDA hardware (HIP/NPU/CPU/MPS/XPU)",
      lambda: is_hip() or is_npu() or is_cpu() or is_mps() or is_xpu(),
  ),
  ```
  ```
  for _name, predicate in rules:
      if predicate():
          self.cuda_graph_config.prefill.backend = Backend.DISABLED
  ```
- Disabled prefill graph → EagerRunner — `srt/model_executor/model_runner_components/cuda_graph_setup.py:239-250`:
  ```
  if check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
  ```
  ```
  # Prefill cuda graph disabled: route eager prefill through the
  # EagerRunner (its can_run_graph returns False, so _forward_raw's
  # extend branch falls through to the eager path).
  if not model_runner.is_draft_worker:
      return result(eager_runner)
  ```
- Runtime routing — `srt/model_executor/model_runner.py:1653-1662, 1673-1678, 1703-1712, 1733-1736`:
  ```
  mode_check = (
      forward_batch.forward_mode.is_cpu_graph
      if self.device == "cpu"
      else forward_batch.forward_mode.is_cuda_graph
  )
  can_run_graph = bool(
      mode_check()
      and self.decode_cuda_graph_runner
      and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
  ```
  ```
  elif (
      forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
      and not isinstance(self.prefill_cuda_graph_runner, EagerRunner)
      and self.prefill_cuda_graph_runner is not None
  ```
  ```
  else:
      # Eager: decode / extend / idle dispatched inside the runner.
      ret = self.eager_runner.execute(
  ```
- Decode capture mode — `srt/model_executor/runner/decode_cuda_graph_runner.py:311, 323-326`:
  ```
  self.capture_forward_mode = ForwardMode.DECODE
  ```
  ```
      self.capture_forward_mode = ForwardMode.TARGET_VERIFY
  elif self.is_dllm:
      self.capture_forward_mode = ForwardMode.DLLM_EXTEND
  ```
- Default decode bucket list — `server_args.py:5141-5147`:
  ```
  elif self.speculative_algorithm is None:
      # Normal case:
      capture_bs = (
          [1, 2, 4, 8, 12]
          + list(range(16, 257, 8))
          + list(range(272, 512, 16))
          + list(range(512, max_bs + 1, 32))
  ```
  `--disable-cuda-graph-padding` → `capture_bs = list(range(1, max_bs + 1))` (:5139-5140). `max_bs` default by memory tier, e.g. ≥160 GB: `decode_cuda_graph_config.max_bs = 512` (:4882-4883); flags `cuda_graph_max_bs_decode` / `cuda_graph_bs_decode` override (:1879-1893, 4524-4529). The final list is further filtered by alignment and clamped to the request pool — `srt/model_executor/runner/base_cuda_graph_runner.py:92-94`:
  ```
  capture_bs = [bs for bs in capture_bs if bs * alignment_width % mul_base == 0]
  capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
  capture_bs = list(sorted(set(capture_bs)))
  ```
  (`mul_base = get_cuda_graph_batch_size_alignment` = 1 unless TBO/DP-attention/CP — `srt/utils/common.py:3804-3812`.)
- Padding — `base_cuda_graph_runner.py:137-152` (`_pad_to_bucket`: "Return the smallest buckets[i] >= raw_size") and `decode_cuda_graph_runner.py:1312-1315`:
  ```
      bs = self._pad_to_bucket(max_batch_size, self.capture_bs)
  else:
      bs = self._pad_to_bucket(raw_bs, self.capture_bs)
  padded_num_tokens = bs * self.captured_req_width
  ```
- Padded rows' seq_lens — `decode_cuda_graph_runner.py:368-369`: `self.seq_len_fill_value = (self.attn_backend.get_cuda_graph_seq_len_fill_value() ...`; `aiter_backend.py:1888-1889`:
  ```
  def get_cuda_graph_seq_len_fill_value(self):
      return 1 if self.num_draft_tokens is None else self.num_draft_tokens
  ```
  `srt/model_executor/cuda_graph_buffer_registry.py:587-594`:
  ```
  GraphSlot(
      "seq_lens",
      _bs,
      torch.int64,
      axis="bs",
      padding_policy=PaddingPolicy.FILL_SENTINEL,
      pad_value=seq_len_fill_value,
  ```
  Replay metadata is built over the padded bs (`build_replay_fb_view(... bs=bs ...)`, `decode_cuda_graph_runner.py:1352-1361`; `seq_lens_sum + (bs - raw_bs) * seq_len_fill_value` at :176), and aiter's `_apply_cuda_graph_metadata` does `kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)` (`aiter_backend.py:1553`) — so **padded rows do run attention, each over 1 KV token**, and `paged_attention_ragged` is launched with the padded batch.
- `can_run_graph` — `decode_cuda_graph_runner.py:648-650, 677-690`:
  ```
  def can_run_graph(self, forward_batch: ForwardBatch):
      # Disable for token embedding overrides (dynamic per-request)
      if forward_batch.replace_embeds is not None:
          return False
  ```
  ```
  is_bs_supported = (
      self.backend.can_run(forward_batch, graph_key)
      if self.disable_padding
      else cuda_graph_bs <= self.max_bs
  )
  ```
  (then `is_encoder_lens_supported`, `is_tbo_supported`, `is_ngram_supported`, :694-722).
- Prefill graph (only if enabled) maps MIXED→EXTEND for replay — `srt/model_executor/runner/prefill_cuda_graph_runner.py:1509-1512`:
  ```
  pcg_forward_mode = (
      ForwardMode.EXTEND
      if forward_batch.forward_mode == ForwardMode.MIXED
      else forward_batch.forward_mode
  ```
  and `can_replay_locally` rejects on non-CUDA BREAKABLE with prefix hits when the model has MHA companion layers (:1070-1077), caps padding waste at 2× (:1103-1105, `_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2` :140). Prefill `max_bs` default = `chunked_prefill_size` for non-MLA (`server_args.py:4919-4924`); token buckets from `_generate_prefill_cuda_graph_batch_sizes` (:5186-5202). Not active on ROCm by default.
- Other HIP lines near graphs in `server_args.py`: `6153-6155` (`if is_hip(): self.triton_attention_num_kv_splits = 16`), `8448-8455` (DLLM on HIP disables both graph phases). No HIP-specific decode-graph default was found.

---

## 3. Tensor-parallel communication placement

**Conclusion.** Qwen3-MoE builds `o_proj` with `reduce_results=False`, and the MoE experts layer with `reduce_results=False`; the all-reduces are issued by (a) `LayerCommunicator.prepare_mlp` → `CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual` → `attention_tensor_model_parallel_all_reduce(hidden_states)` followed by `post_attention_layernorm(hidden_states, residual)` (fused add+RMSNorm), and (b) inside `Qwen3MoeSparseMoeBlock.forward_normal` → `moe_tensor_model_parallel_all_reduce(final_hidden_states)` after the experts. `postprocess_layer` is a no-op (`_trivial`) without DP attention. Layer 0 differs only in `prepare_attn`: `residual is None` → `residual = hidden_states; hidden_states = self.input_layernorm(hidden_states)` (norm without residual add); all other layers' `prepare_attn` does `input_layernorm(hidden_states, residual)`. At TP1 all of these collapse to norm only (`_simple` / `tp_size > 1` guards). All-reduce+norm fusion exists in two forms: FlashInfer (CUDA SM90/SM100 only) and aiter (`apply_aiter_all_reduce_fusion`), the latter gated on `SGLANG_USE_AITER` **and** `--enable-aiter-allreduce-fusion`, which defaults to **False** and is not auto-enabled for Qwen3 (the auto-enable lines for DeepSeek/GPT-OSS are commented out). When enabled it (i) fuses the post-attention AR with the post-attention norm via `layernorm.forward_with_allreduce_fusion` → `tensor_model_parallel_fused_allreduce_rmsnorm`, and (ii) skips the post-MoE AR (`should_fuse_mlp_allreduce_with_next_layer` → `_sglang_needs_allreduce_fusion`) so the next layer's `prepare_attn` performs the fused AR+add+norm. `--enable-dp-attention` changes scatter modes (attn group vs full TP group, gather/scatter functions) and disables both fusions; `--moe-dense-tp-size` only affects dense MLP layers, which Qwen3-30B-A3B does not have (`is_layer_sparse = True` for all layers).

**Evidence.**

- `o_proj` — `srt/models/qwen3_moe.py:496-505`:
  ```
  self.o_proj = RowParallelLinear(
      self.total_num_heads * self.head_dim,
      hidden_size,
      bias=attention_bias,
  ```
  ```
      tp_rank=attn_tp_rank,
      tp_size=attn_tp_size,
      reduce_results=False,
      prefix=add_prefix("o_proj", prefix),
  ```
  (`srt/models/qwen2_moe.py:701-709` identical.) RowParallelLinear only all-reduces when `(self.reduce_results and self.tp_size > 1) or self.use_decode_attn_tp` (`srt/layers/linear.py:1641-1644`).
- Decoder forward — `qwen3_moe.py:811-856`:
  ```
  hidden_states, residual = (
      self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
  ```
  ```
  hidden_states, residual = self.layer_communicator.prepare_mlp(
      hidden_states, residual, forward_batch
  )
  ```
  ```
  if fuse_mlp_allreduce:
      hidden_states._sglang_needs_allreduce_fusion = True
  else:
      hidden_states, residual = self.layer_communicator.postprocess_layer(
  ```
  All layers sparse — `qwen3_moe.py:760-763`: `# Qwen3MoE all layers are sparse and have no nextn now` / `self.is_layer_sparse = True`.
- Scatter modes without DP attention: `LayerScatterModes.init_new` uses `attn_mode=TP_ATTN_FULL` (`srt/layers/communicator.py:373`), `mlp_mode=FULL` for sparse layers with no a2a backend (:386-400), `middle_residual_mode=TP_ATTN_FULL` (:418-423), `layer_output_mode=TP_ATTN_FULL` (:435-436); layer 0 input mode `ScatterMode.model_input_output()` (:380-382) = `TP_ATTN_FULL` when DP attention is off (:213-218). Group sizes: `TP_ATTN_FULL: attn_tp_size`, `FULL: tp_size // attn_cp_size` (:897-901) — equal without DP/CP.
- `prepare_mlp` selection — `communicator.py:997-1005, 1016-1025`:
  ```
  if (
      context.is_same_group_size(hidden_states_input_mode, hidden_states_output_mode)
      and context.is_same_group_size(residual_input_mode, residual_output_mode)
      and context.attn_tp_size == 1
  ):
      return CommunicateWithAllReduceAndLayerNormFn._simple
  ```
  ```
  and (hidden_states_output_mode == ScatterMode.FULL)
  and (residual_output_mode == ScatterMode.TP_ATTN_FULL)
  ):
      return partial(CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
  ```
  → TP1: `_simple` = `layernorm(hidden_states, residual)` only (:1075-1077). TP>1: `_gather_hidden_states_and_residual`, non-DP branch — :1155-1181:
  ```
  if (
      apply_aiter_all_reduce_fusion(hidden_states)
      or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
  ) and hasattr(layernorm, "forward_with_allreduce_fusion"):
      hidden_states, residual = layernorm.forward_with_allreduce_fusion(
  ```
  ```
      else:
          hidden_states = attention_tensor_model_parallel_all_reduce(
              hidden_states
          )
  ...
      hidden_states, residual = layernorm(hidden_states, residual)
  ```
  So the **post-o_proj all-reduce is here**, immediately followed by the fused add+RMSNorm.
- Post-MoE all-reduce — `qwen3_moe.py:331-343`:
  ```
  if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
      is_tp_path=True
  ):
      final_hidden_states = moe_tensor_model_parallel_all_reduce(
  ```
  (`self.tp_size = get_parallel().moe_tp_size`, :237). The experts layer itself is constructed with `reduce_results=False` default (`srt/layers/moe/fused_moe_triton/layer.py:245`) and only reduces if `self.reduce_results and (moe_tp_size > 1 or moe_ep_size > 1)` (:1508-1509). Skip predicate — `srt/layers/moe/utils.py:578-579, 608-609`: `return f.fuse_mlp_allreduce or f.mlp_reduce_scatter` / `if should_skip_mlp_all_reduce(): return True`.
- `postprocess_layer` → `_trivial` when group sizes match — `communicator.py:1315-1318`:
  ```
  if context.is_same_group_size(
      hidden_states_input_mode, output_mode
  ) and context.is_same_group_size(residual_input_mode, output_mode):
      return CommunicateSummableTensorPairFn._trivial
  ```
- Layer 0 (`residual is None`) — `communicator.py:619-620, 669-670` vs other layers :722-726:
  ```
  if residual is None:
      residual = hidden_states
  ```
  ```
  else:
      hidden_states = self.input_layernorm(hidden_states)
  ```
  ```
  else:
      hidden_states, residual = self.input_layernorm(
          hidden_states,
          residual,
  ```
  `Qwen2MoeModel.forward` passes `residual = None` into layer 0 (`qwen2_moe.py:940-944`).
- Fusion gates — `communicator.py:93, 182-194`:
  ```
  _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
  ```
  ```
  def apply_aiter_all_reduce_fusion(input_tensor: torch.Tensor):
  ...
      return (
          _use_aiter
          and total_bytes > 0
          and n <= 16384
          and total_bytes <= 8 * 1024 * 8192
          and get_parallel().tp_size != 6
          and not is_dp_attention_enabled()
          and get_exec().comm.enable_aiter_allreduce_fusion
  ```
  FlashInfer fusion requires `(_is_sm90_supported or _is_sm100_supported) and _is_flashinfer_available` (:165-172) — never on ROCm.
- Fuse-with-next-layer decision — `communicator.py:852-868`:
  ```
  apply_flashinfer_allreduce_fusion(batch_size)
  or (
      _use_aiter
      and batch_size > 0
      and get_parallel().tp_size != 6
      and not is_dp_attention_enabled()
      and get_moe_a2a_backend().is_none()
      and get_exec().comm.enable_aiter_allreduce_fusion
  ```
  `and (not self.is_last_layer) and (self._context.tp_size > 1)`. Consumed in the next layer's `prepare_attn` (:576-586): `if residual is not None and hasattr(hidden_states, "_sglang_needs_allreduce_fusion") ... self.input_layernorm.forward_with_allreduce_fusion(...)`, else `moe_tensor_model_parallel_all_reduce(hidden_states)` + norm (:614-617).
- Fused kernel — `srt/layers/layernorm.py:221-227`:
  ```
  # Prefer AITER fused AR+RMSNorm when enabled on AMD.
  if _use_aiter:
      fused_result = tensor_model_parallel_fused_allreduce_rmsnorm(
          x, residual, weight, norm_module.variance_epsilon
  ```
  → `get_tp_group().fused_allreduce_rmsnorm(...)` (`srt/distributed/communication_op.py:40`; GroupCoordinator implementation not inspected).
- Server flag default — `server_args.py:2051-2055`: `enable_aiter_allreduce_fusion: A[bool, Arg(help="Enable Aiter AllReduce Fusion.", resolvable=True), ...] = False`. Auto-enable is commented out — :5478-5480 (DeepSeek) and :5576-5577 (GPT-OSS): `# TODO (Hubert): Put this back later` / `# self.enable_aiter_allreduce_fusion = True`. Nothing for Qwen3.
- `--moe-dense-tp-size` — `server_args.py:2443-2449` ("TP size for MoE dense MLP layers"); only affects `_compute_mlp_mode` for non-sparse layers (`communicator.py:401-405`, `enable_moe_dense_fully_dp()` = `moe_dense_tp_size == 1` :441-442) → irrelevant for Qwen3-30B-A3B. `--enable-dp-attention` switches `ScatterMode.model_input_output()` to SCATTERED (:213-216), uses attn-TP-group collectives/dp_gather (:1141-1152), and disables both fusions (`not is_dp_attention_enabled()` in :171, :192, :859).
- TP2/TP8 note: head sharding — `qwen3_moe.py:463-473`: `self.num_heads = self.total_num_heads // attn_tp_size`; `if self.total_num_kv_heads >= attn_tp_size: assert ... % attn_tp_size == 0 else: # ... replicate the KV heads` / `self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)`. So TP2: 16 q / 2 kv heads per rank; TP4: 8 q / 1 kv; TP8: 4 q heads and the single KV head **replicated** (KV cache duplicated across pairs of ranks). Collective count per layer is unchanged (2 all-reduces: post-attention, post-MoE) at any TP>1; TP1 has none.

---

## 4. Embedding

**Conclusion.** `Qwen2MoeModel` (base of `Qwen3MoeModel`) builds `VocabParallelEmbedding(..., use_attn_tp_group=is_dp_attention_enabled())`. With `enable_tp=True` (default) and no DP attention it shards the (padded) vocab over the full TP group. At TP>1 the forward is: `get_masked_input_and_mask` (map out-of-shard ids to 0 and build `~vocab_mask`), local `embedding` gather, `masked_fill_(input_mask, 0)`, then `tensor_model_parallel_all_reduce`. A fused Triton kernel (`fused_vocab_parallel_embedding`) replaces mask+gather+fill when the input is CUDA-device, contiguous, int32/int64 and the method is unquantized — the all-reduce still follows. At TP1 it is a plain gather with no mask/all-reduce. Replication (`SGLANG_ENABLE_EMBED_REPLICATION`) is only used by DeepSeek-family models via `get_embedding_tp_kwargs`, not by Qwen. The embedding is called **once per forward**, only on the first PP rank, and only when `input_embeds is None`.

**Evidence.**

- Construction — `srt/models/qwen2_moe.py:886-895`:
  ```
  if self.pp_group.is_first_rank:
      self.embed_tokens = VocabParallelEmbedding(
          config.vocab_size,
          config.hidden_size,
          use_attn_tp_group=is_dp_attention_enabled(),
  ```
  ```
  else:
      self.embed_tokens = PPMissingLayer()
  ```
- Call site, once per forward — `qwen2_moe.py:939-948`:
  ```
  if self.pp_group.is_first_rank:
      if input_embeds is None:
          hidden_states = self.embed_tokens(input_ids)
      else:
          hidden_states = input_embeds
      residual = None
  else:
      assert pp_proxy_tensors is not None
      hidden_states = pp_proxy_tensors["hidden_states"]
  ```
- TP group selection — `srt/layers/vocab_parallel_embedding.py:250-260`:
  ```
  if self.enable_tp:
      if use_attn_tp_group:
          tp_rank = get_parallel().attn_tp_rank
          self.tp_size = get_parallel().attn_tp_size
      else:
          tp_rank = get_parallel().tp_rank
          self.tp_size = get_parallel().tp_size
  ```
- Mask — `vocab_parallel_embedding.py:149-163`:
  ```
  org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
  ```
  ```
  vocab_mask = org_vocab_mask | added_vocab_mask
  input_ = vocab_mask * (input_ - valid_offset)
  return input_, ~vocab_mask
  ```
- TP>1 local shard — `vocab_parallel_embedding.py:538-540, 553-564`:
  ```
  if self.tp_size == 1:
      with symm_alloc:
          return self.quant_method.embedding(self, input_.long())
  ```
  ```
  # Map out-of-shard ids to index 0, gather, then zero those rows.
  masked_input, input_mask = get_masked_input_and_mask(
  ```
  ```
  with symm_alloc:
      output_parallel = self.quant_method.embedding(self, masked_input.long())
  output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
  return output_parallel
  ```
  Triton alternative gate: `_use_triton_embedding` (:508-525): `if self.tp_size == 1: return False` ... `input_.is_cuda` (true for HIP tensors under torch) etc. → `fused_vocab_parallel_embedding(...)` (:542-550).
- All-reduce — `vocab_parallel_embedding.py:566-579`:
  ```
  output_parallel = self._embed_local_shard(input_)
  if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
      if self.use_attn_tp_group:
          output_parallel = attn_tp_all_reduce(output_parallel)
      else:
          # Reduce across all the model parallel GPUs.
          output_parallel = tensor_model_parallel_all_reduce(output_parallel)
  ```
- Replication option exists but is DeepSeek-only — `vocab_parallel_embedding.py:178-185` (`get_embedding_tp_kwargs`: `if envs.SGLANG_ENABLE_EMBED_REPLICATION.get(): return {"enable_tp": False}`); `qwen2_moe.py` does not call it. The unquantized gather is `F.embedding(input_, layer.weight)` (`srt/layers/quantization/unquant.py:182-183`).

---

## 5. Attention backend decode partition count and kernel selection

**Conclusion.** `max_num_partitions` is computed **once, server-wide** in `AiterAttnBackend.__init__` from `max_context_len` with a fixed 256-token partition (`_AITER_PARTITION_SIZE_ROCM = 256`), and the FP32 workspace is sized `max_bs * num_head * max_num_partitions * head_dim` + two `max_bs*num_head*max_num_partitions` fp32 arrays. For non-MLA models at page size 1 with the default NHD layout and `SGLANG_USE_AITER_UNIFIED_ATTN` unset, `forward_decode` calls `paged_attention_ragged(...)` passing `self.max_num_partitions` and `_AITER_PARTITION_SIZE_ROCM`, with K/V viewed as `(-1, 1, heads, head_dim)` ("page size 1"). `forward_extend` for non-MLA, non-spec, NHD, BF16 KV calls `mha_batch_prefill_func(q, k_cache, v_cache, qo_indptr, kv_indptr, page_table=kv_indices, max_q_len, max_kv_len, causal=True, ...)` — the **same call for a first chunk, a later chunk with prefix, a radix-hit extend, and MIXED**; nothing in the non-MLA path branches on `extend_prefix_lens` (the only prefix-dependent non-MLA branch is the gfx95 FP8/head_dim 256 flash path, inapplicable to BF16 head_dim 128).

**Evidence.**

- `aiter_backend.py:130`: `_AITER_PARTITION_SIZE_ROCM = 256`; :273-284:
  ```
  self.max_num_partitions = (
      self.max_context_len + _AITER_PARTITION_SIZE_ROCM - 1
  ) // _AITER_PARTITION_SIZE_ROCM
  ```
  ```
  if not (self.use_mla or self.use_triton_unified_attention):
      self.workspace_buffer = torch.empty(
          (max_bs * self.num_head * self.max_num_partitions * self.head_dim)
          * nbyes_per_qo_elem
          + 2 * (max_bs * self.num_head * self.max_num_partitions) * 4,
  ```
- Decode kernel selection gate — :262-264: `self.use_triton_unified_attention = get_bool_env_var("SGLANG_USE_AITER_UNIFIED_ATTN")` (env default False, `srt/environ.py:761`); then `forward_decode` non-MLA tail — `aiter_backend.py:2647-2667`:
  ```
  paged_attention_ragged(
      o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
      self.workspace_buffer,
      q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
  ```
  ```
      k_cache.view(-1, 1, layer.tp_k_head_num, layer.qk_head_dim),
      v_cache.view(-1, 1, layer.tp_v_head_num, layer.v_head_dim),
      self.scale,
      self.forward_metadata.kv_indptr,
  ```
  ```
      self.kv_last_page_len,
      1,
      self.max_num_partitions,
      None,
  ```
  ```
      "NHD",
      self.logits_soft_cap,
      self.k_scale,
      self.v_scale,
      None,
      _AITER_PARTITION_SIZE_ROCM,
  ```
  Decode metadata: `kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)` + `create_flashinfer_kv_indices_triton` (:928-940). Under graph replay the same is done over the padded bs (:1553-1562).
- Extend kernel — `aiter_backend.py:2399-2400, 2421-2431`:
  ```
  # NHD path — original aiter paged batch_prefill.
  # TODO kkhuang-amd need to remove it when mha_batch_prefill_func support fp8-kv
  ```
  ```
  o = mha_batch_prefill_func(
      q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
      k_cache,
      v_cache,
      self.qo_indptr[:bs0],
      self.forward_metadata.kv_indptr[:bs0],
      page_table,
      self.forward_metadata.max_q_len,
      self.forward_metadata.max_kv_len,
      causal=True,
  ```
  Layout default: `SGLANG_AITER_KV_CACHE_LAYOUT = EnvStr("nhd")` (`srt/environ.py:793`); the `vectorized_5d` path is taken only when the pool reports that layout (:245-252, 2385-2386). The gfx95 fp8 flash branch requires `layer.qk_head_dim == 256 ... and self.kv_cache_dtype == fp8_dtype` (:2349-2360) → not taken for BF16/128. Extend metadata (`max_q_len = max(extend_seq_lens_cpu)`, `max_kv_len = seq_lens_cpu.max()`, :1374-1379) and the qo/kv indptr derivation (§1) are prefix-agnostic, so a chunked-prefill extend with prefix uses the same kernel with longer `kv_indptr` spans. `extend_attention_fwd` (Triton) is used only for TARGET_VERIFY/spec (:2308-2324).

---

## 6. Linear GEMM dispatch (BF16 unquantized, ROCm)

**Conclusion.** `UnquantizedLinearMethod.apply` calls `aiter.tuned_gemm.tgemm.mm(x, weight, bias, otype=x.dtype)` when `_use_aiter` is true (module-level `get_bool_env_var("SGLANG_USE_AITER") and is_hip()`) and the weight is a plain tensor; otherwise it falls through (CuteDSL branch is CUDA-only) to `F.linear(x, weight, bias)`. The controlling switch is the **environment variable `SGLANG_USE_AITER`**, whose in-code default is **False** (`srt/environ.py` `EnvBool(False)`; `get_bool_env_var` default `"false"`). I found no code in `server_args.py`/`model_runner.py` that sets it programmatically on HIP — so unless the deployment (e.g. the ROCm Docker image) exports `SGLANG_USE_AITER=1`, BF16 linears run through `F.linear` (hipBLASLt via torch) even though the attention backend defaults to `aiter`. The same `_use_aiter` gate controls the aiter RMSNorm, the communicator fusions and the aiter MoE path.

**Evidence.**

- Gate — `srt/layers/quantization/unquant.py:62-69`:
  ```
  _is_hip = is_hip()
  ...
  _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

  if _use_aiter:
      from aiter.ops.shuffle import shuffle_weight
      from aiter.tuned_gemm import tgemm
  ```
- Branches — `unquant.py:235-236, 263-265`:
  ```
  elif _use_aiter and type(layer.weight.data) is torch.Tensor:
      return tgemm.mm(x, layer.weight, bias, otype=x.dtype)
  ```
  ```
      return F.linear(x, layer.weight, bias)

  return F.linear(x, layer.weight, bias)
  ```
  (the intervening `get_bf16_gemm_backend().is_cutedsl()` branch at :238-263 requires an SM10x GPU, `:109`.)
- Env default — `srt/environ.py:744-746`:
  ```
  # AMD, ROCm, and AITER
  # ===================================================================
  SGLANG_USE_AITER = EnvBool(False)
  ```
  `srt/utils/common.py:1247-1248`: `def get_bool_env_var(name: str, default: str = "false") -> bool:` / `value = os.getenv(name, default)`. `is_hip()` = `torch.version.hip is not None` (:133-134).
- No programmatic set: `grep` for `SGLANG_USE_AITER` assignment in `srt/server_args.py`, `srt/model_executor/model_runner.py`, `attention_backend_setup.py` returned only `envs.SGLANG_OPT_USE_AITER_INDEXER.set(True)` (`server_args.py:5532`, DSA-specific). Attention backend default on HIP is `aiter` regardless (`server_args.py:5916-5917`: `elif is_hip(): return "aiter"`).

---

## 7. Norm kernels (RMSNorm on ROCm)

**Conclusion.** `RMSNorm` extends `BaseFusedOp`; on HIP the platform chain is `("forward_hip", "forward_cuda")` then `forward_native`. If `SGLANG_USE_AITER` is set, `RMSNorm.__init__` pins `self._forward_method = self.forward_aiter`, which calls aiter `rmsnorm2d_fwd(x, weight, eps)` (no residual) or `rmsnorm2d_fwd_with_add(output, x, residual, residual_out, weight, eps)` (with residual, returns `(output, residual_out)`); it falls back to `forward_native` when weight dtype != activation dtype (gfx950 corruption note), on zero rows, or in batch-invariant mode. Without aiter, `forward_hip` runs the vLLM custom ops `rms_norm` / `fused_add_rms_norm` if `vllm._custom_ops` imports, else `forward_native` (pure torch). sgl_kernel's `fused_add_rmsnorm`/`rmsnorm` are imported only on CUDA/XPU/MUSA and are not used on HIP.

**Evidence.**

- Import-time selection — `srt/layers/layernorm.py:53, 110-125`:
  ```
  _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
  ```
  ```
  if _use_aiter:
      import aiter as _aiter
      from aiter import layernorm2d_fwd as layer_norm
      from aiter import rmsnorm2d_fwd as rms_norm
      from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
  ```
  ```
  elif _is_hip:
      try:
          from vllm._custom_ops import fused_add_rms_norm, rms_norm
          _has_vllm_rms_norm = True
  ```
  sgl_kernel import is under `if _is_cuda or _is_xpu or _is_musa:` (:60, :101-106).
- Pinning aiter path — `layernorm.py:456, 469`:
  ```
  if _use_aiter:
  ...
      self._forward_method = self.forward_aiter
  ```
- `forward_aiter` — `layernorm.py:601-604, 645-660`:
  ```
  if self.weight.data.dtype != x.dtype:
      # AITER's ROCm rmsnorm2d_fwd requires weight/activation dtypes to match;
      # FP32 weight + BF16 activation yields finite-but-corrupted output on gfx950.
      return self.forward_native(x, residual, post_residual_addition)
  ```
  ```
  if residual is not None:
      residual_out = torch.empty_like(x)
      output = torch.empty_like(x)
  ```
  ```
      fused_add_rms_norm(
          output,
          x,
          residual,
          residual_out,
          self.weight.data,
          self.variance_epsilon,
      )
      return output, residual_out
  output = rms_norm(x, self.weight.data, self.variance_epsilon)
  ```
- `forward_hip` (non-aiter) — `layernorm.py:672-674, 692-702`:
  ```
  # Fallback to native implementation if vllm is not available
  if not _has_vllm_rms_norm:
      return self.forward_native(x, residual, post_residual_addition)
  ```
  ```
  if residual is not None:
      out = torch.empty_like(x)
      residual_out = torch.empty_like(x)
  ```
  ```
      fused_add_rms_norm(
          out, x, residual_out, residual, self.weight.data, self.variance_epsilon
      )
      return out, residual_out
  out = torch.empty_like(x)
  rms_norm(out, x, self.weight.data, self.variance_epsilon)
  ```
- Platform dispatch — `kernels/fused_op.py:147-149, 560-561`:
  ```
  _PLATFORM_METHODS: Dict[str, Tuple[str, ...]] = {
      "cuda": ("forward_cuda",),
      "hip": ("forward_hip", "forward_cuda"),
  ```
  ```
  # 5) Platform-specific forward; 6) native fallback.
  method = self._platform_method(_platform_key())
  return method if method is not None else self.forward_native
  ```
  (`_platform_key()` returns `"hip"` when `is_hip()`, :182-183.) The fused AR+RMSNorm variants (`forward_with_allreduce_fusion`, :829-838) are only reached via the communicator fusion gates in §3.

---

## Not determinable from these files / flags

1. **Whether `SGLANG_USE_AITER` is set in the actual deployment.** In-code default is False; ROCm Docker images commonly export it, but no Dockerfile/launcher was in scope. This single variable decides GEMM (`tgemm.mm` vs `F.linear`), RMSNorm (aiter vs vLLM/native), and eligibility for aiter AR fusion — the profiling story must record its value.
2. **`gpu_mem` detection path** for the memory-tiered defaults (`chunked_prefill_size=16384`, decode `max_bs=512` on ≥160 GB): I quoted the tier table but did not inspect the helper that produces `gpu_mem`.
3. **`GroupCoordinator.fused_allreduce_rmsnorm`** (what aiter kernel it dispatches to, and its size limits) lives in `srt/distributed/parallel_state.py`, not inspected; irrelevant unless `--enable-aiter-allreduce-fusion` is passed.
4. **Whether Qwen3-MoE counts as "has_mha_companion_layers"** for the prefill BREAKABLE graph prefix-hit rejection on non-CUDA (`prefill_cuda_graph_runner.py:1070-1077`); only matters if someone explicitly enables `--cuda-graph-backend-prefill breakable` on ROCm.
5. **The `EagerRunner` decode/extend dispatch body** was not read; `_forward_raw`'s comment ("decode / extend / idle dispatched inside the runner") and `base_attn_backend.forward` establish the attention routing independently.
6. Attention-layer fusion gates (qk-norm/RoPE/KV write in `forward_extend` :1935-2010 and `forward_decode` :2469-2540) were deliberately skipped per the task split.
