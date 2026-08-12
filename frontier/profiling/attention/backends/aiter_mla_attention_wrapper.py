# Copyright 2023 The Sarathi team.

"""AMD `aiter` MLA attention wrapper for profiling (MI355X/ROCm).

Calls AMD's real, production `aiter` MLA kernels (the same ones real sglang serving
uses via `--attention-backend aiter`) instead of `TorchSdpaMlaAttentionWrapper`'s
portable-but-unfused reference implementation. Confirmed directly by reading real
sglang's own integration (`aiter_backend.py`, from the exact
`lmsysorg/sglang:v0.5.11-rocm700-mi35x` image used for the real DeepSeek-R1
benchmark this backend is meant to match) — including finding and correcting an
initial wrong assumption:

  - **Decode**: `q` arrives at `forward_decode` already absorbed into rank-space
    (same width as the compressed cache) — aiter's `mla_decode_fwd` does *not* do
    this absorption itself. So `attn_mla_decode_q_latent_proj` stays a real Python
    step here (`w_uk` reuse, same as `TorchSdpaMlaAttentionWrapper`), and
    `mla_decode_fwd` is called directly against the compressed cache
    (`aiter.mla_decode_stage1_asm_fwd` under the hood) with no separate V
    up-projection needed afterward — `attn_mla_v_up_proj` is a genuine no-op here
    (kernel output already lands at `v_head_dim`).
  - **Prefill is not the simple case it first looked like.** `aiter.mla_prefill_fwd`
    (the generic wrapper) asserts `head_size == kv_buffer.size(-1)` and has no
    separate V argument — it does not apply to this model's real shapes
    (`qk_head_dim=192 != kv_lora_rank+qk_rope_head_dim=576`), confirmed by hitting
    that exact assertion. Reading `forward_extend` in full shows real serving's
    actual dispatch for a fresh, no-cached-prefix request (matching this repo's own
    profiling workload — every request is independent, `random_input_len=8192` fits
    in one chunk) is a **persistent fp8 kernel**: `mla_fp8_prefill_attn` →
    `aiter.mla_prefill_ps_asm_fwd` (this *is* the same "asm" kernel family seen in
    the real Kineto trace, `aiter::mla_pfl_qh192_vh128_...`), fed by scheduling
    metadata from `aiter.get_ps_metadata_info_v1`/`get_ps_metadata_v1` and combined
    via `aiter.mla_reduce_v1` — mirrored here in `_gather_prefill_up_projected` +
    `_run_prefill`. Unlike the SDPA reference, up-projection to full K/V
    (`attn_mla_prefill_kv_up_proj`) is real, timed work here too — aiter's
    persistent kernel takes fully materialized K/V (cast to `aiter.dtypes.fp8`,
    matching the real quantized model), not the raw compressed cache.
  - This wrapper always reconstructs full K/V from the compressed cache per call
    (like the SDPA reference does), rather than maintaining a separate persistent
    full-width KV cache the way real sglang's paging does for genuine multi-chunk
    continuation — so for `kv_cache_size > 0` (chunked-continuation) profiling
    combos this measures a self-consistent but slightly pessimistic re-up-projection
    cost versus what a production continuation request would pay. Still a real
    aiter kernel end to end, not a reference fallback.
  - `aiter.concat_and_cache_mla` is the direct standalone analogue of the SDPA
    wrapper's manual `kv_cache[...] = c_kv`/`k_pe` cache-write slicing.

All six MLA scopes are still emitted (schema/training compatibility with
`LATENT_MLA_ATTENTION_FAMILY` requires this); only `attn_mla_v_up_proj` is a
genuine no-op for this backend (decode's V up-projection is fused into
`mla_decode_fwd`) — prefill's up-projection is real, unlike the initial design.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from frontier.profiling.attention.backends.base_attention_wrapper import (
    BaseAttentionWrapper,
)
from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
from frontier.attention.model_binding import bind_attention_family
from frontier.attention.ops import AttentionFamilySpec, AttentionMemoryLayout
from frontier.profiling.attention.sequence_metadata import SequenceMetadata
from frontier.profiling.common.constants import OperationMetrics
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig

_TILE_Q = 256  # fixed granularity constant in real sglang's ps-kernel scheduling


class AiterMlaAttentionWrapper(BaseAttentionWrapper):
    """MLA attention backend calling AMD's real `aiter` kernels.

    See module docstring for which scopes are real vs. fused-into-aiter no-ops,
    and for the prefill-path correction found while implementing this.
    """

    _inst = None

    def supports_attention_family(self, attention_family: AttentionFamilySpec) -> bool:
        return attention_family.family_id == LATENT_MLA_ATTENTION_FAMILY.family_id

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        self._attention_family = bind_attention_family(model_config).family
        if self._attention_family.family_id != LATENT_MLA_ATTENTION_FAMILY.family_id:
            raise NotImplementedError(
                "AiterMlaAttentionWrapper only supports the latent MLA "
                f"attention family; got {self._attention_family.family_id!r}."
            )

        super().init(model_config, parallel_config, block_size, device)
        self.dtype = model_config.dtype
        self.device = device
        self.block_size = block_size

        self.num_q_heads = model_config.get_num_q_heads(parallel_config)
        self.kv_lora_rank = int(model_config.kv_lora_rank)
        self.qk_rope_head_dim = int(model_config.qk_rope_head_dim)
        self.qk_nope_head_dim = int(model_config.qk_nope_head_dim)
        self.v_head_dim = int(model_config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.cache_width = self.kv_lora_rank + self.qk_rope_head_dim

        # W_UK/W_UV: per-head up-projection from the compressed latent to
        # K-nope/V. Needed for real prefill (aiter's persistent prefill kernel
        # takes materialized K/V, see module docstring) and W_UK is reused
        # transposed to absorb the decode-side query into rank space. Random
        # weights: only shapes matter for latency profiling.
        self.w_uk = torch.randn(
            self.num_q_heads,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            dtype=self.dtype,
            device=self.device,
        )
        self.w_uv = torch.randn(
            self.num_q_heads,
            self.kv_lora_rank,
            self.v_head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        self.softmax_scale = 1.0 / (self.qk_head_dim**0.5)

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.contains_decode = False
        self.slot_mapping: Optional[torch.Tensor] = None
        self._requests: List[Dict] = []

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        """Compressed MLA cache: (num_blocks, block_size, kv_lora_rank + qk_rope_head_dim).

        Same layout aiter's decode path expects directly — confirmed against
        `aiter/mla.py`'s `mla_decode_fwd` docstring comments.
        """
        return torch.randn(
            num_blocks,
            self.block_size,
            self.cache_width,
            **kwargs,
        )

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        """Identical bookkeeping to `TorchSdpaMlaAttentionWrapper.begin_forward`
        (block-table/context-length/query-offset tracking is backend-agnostic;
        only the attention math in `forward` differs). Prefill requests are
        always appended before decode requests, so the two phases occupy a
        contiguous prefix/suffix of the flattened query tensor — relied on by
        `forward`'s phase-batched slicing below.
        """
        self.is_profiling_iteration = False
        self.is_metadata_initialized = True
        self.contains_prefill = False
        self.contains_decode = False
        self._requests = []

        slot_mapping: List[int] = []
        q_offset = 0

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return
            self.contains_prefill = True

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + prompt_chunk_len
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size

            self._requests.append(
                {
                    "is_prefill": True,
                    "q_start": q_offset,
                    "query_len": prompt_chunk_len,
                    "context_len": current_total_len,
                    "block_table": list(seq_metadata.block_table[:num_blocks_in_use]),
                }
            )
            q_offset += prompt_chunk_len

            for token_idx in range(processed_prompt_len, current_total_len):
                block_number = seq_metadata.block_table[token_idx // self.block_size]
                block_offset = token_idx % self.block_size
                slot_mapping.append(block_number * self.block_size + block_offset)

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return
            self.contains_decode = True

            context_len = seq_metadata.seq.get_len()
            num_blocks_in_use = (context_len + self.block_size - 1) // self.block_size

            self._requests.append(
                {
                    "is_prefill": False,
                    "q_start": q_offset,
                    "query_len": 1,
                    "context_len": context_len,
                    "block_table": list(seq_metadata.block_table[:num_blocks_in_use]),
                }
            )
            q_offset += 1

            token_idx = context_len - 1
            block_number = seq_metadata.block_table[token_idx // self.block_size]
            block_offset = token_idx % self.block_size
            slot_mapping.append(block_number * self.block_size + block_offset)

        self.slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.long, device=self.device
        )

    def end_forward(self):
        self.is_metadata_initialized = False
        self.slot_mapping = None
        self._requests = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Same input contract as `TorchSdpaMlaAttentionWrapper.forward`:
        `query` is the final per-head nope++rope Q, `key` is the compressed
        latent `c_kv`, `value` carries the decoupled RoPE key `k_pe`.
        """
        assert self.is_metadata_initialized, "Metadata is not initialized."
        if self.is_profiling_iteration:
            return torch.zeros(
                query.shape[0],
                self.num_q_heads * self.v_head_dim,
                dtype=query.dtype,
                device=query.device,
            )
        if softmax_scale != self.softmax_scale:
            raise ValueError(
                f"softmax_scale mismatch: expected {self.softmax_scale}, got {softmax_scale}. "
                "Re-plan the wrapper if you need a different scale."
            )

        import aiter
        from aiter.mla import mla_decode_fwd
        from aiter import concat_and_cache_mla

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.qk_head_dim)
            c_kv = key.contiguous().reshape(-1, self.kv_lora_rank)
            k_pe = value.contiguous().reshape(-1, self.qk_rope_head_dim)

        output = torch.empty(
            query.shape[0],
            self.num_q_heads,
            self.v_head_dim,
            dtype=query.dtype,
            device=query.device,
        )

        with self.get_timer(OperationMetrics.ATTN_MLA_KV_CACHE_SAVE, layer_id):
            if self.slot_mapping is None:
                raise RuntimeError("slot_mapping is not initialized.")
            scale = torch.ones(1, dtype=torch.float32, device=self.device)
            concat_and_cache_mla(c_kv, k_pe, kv_cache, self.slot_mapping, "auto", scale)

        prefill_reqs = [r for r in self._requests if r["is_prefill"]]
        total_prefill_tokens = sum(r["query_len"] for r in prefill_reqs)

        with self.get_timer(OperationMetrics.ATTN_MLA_PREFILL_KV_UP_PROJ, layer_id):
            k_full, v_full, kv_indptr = (
                self._gather_prefill_up_projected(kv_cache, prefill_reqs)
                if self.contains_prefill
                else (None, None, None)
            )

        with self.get_timer(OperationMetrics.ATTN_MLA_PREFILL, layer_id):
            if self.contains_prefill:
                self._run_prefill(
                    query, output, prefill_reqs, total_prefill_tokens,
                    k_full, v_full, kv_indptr, aiter,
                )

        decode_reqs = [r for r in self._requests if not r["is_prefill"]]

        with self.get_timer(OperationMetrics.ATTN_MLA_DECODE_Q_LATENT_PROJ, layer_id):
            q_absorbed = (
                self._project_decode_query(query, decode_reqs, total_prefill_tokens)
                if self.contains_decode
                else None
            )

        with self.get_timer(OperationMetrics.ATTN_MLA_DECODE, layer_id):
            if self.contains_decode:
                self._run_decode(
                    q_absorbed, kv_cache, output, decode_reqs, total_prefill_tokens,
                    mla_decode_fwd,
                )

        # Fused into aiter's decode kernel above — real cost for this backend
        # is genuinely ~0 here, not omitted (see module docstring).
        with self.get_timer(OperationMetrics.ATTN_MLA_V_UP_PROJ, layer_id):
            pass

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.v_head_dim)

        return output

    def _gather_prefill_up_projected(
        self, kv_cache: torch.Tensor, prefill_reqs: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather each prefill request's full cached context (not just this
        call's new chunk — matching the SDPA reference's semantics) and
        up-project to full K (nope++rope) / V via `w_uk`/`w_uv`, concatenated
        into one flat `(sum(context_len), H, dim)` tensor per aiter's
        fp8-persistent-kernel calling convention (`fp8_prefill_kv_indices` is a
        flat `arange` in real sglang — see module docstring)."""
        k_chunks = []
        v_chunks = []
        kv_indptr = [0]
        for req in prefill_reqs:
            context_len = req["context_len"]
            block_table = req["block_table"]
            chunks = []
            remaining = context_len
            for block_number in block_table:
                take = min(self.block_size, remaining)
                chunks.append(kv_cache[block_number, :take])
                remaining -= take
                if remaining <= 0:
                    break
            cached = torch.cat(chunks, dim=0)  # (context_len, cache_width)
            c_kv = cached[:, : self.kv_lora_rank]
            k_pe = cached[:, self.kv_lora_rank :]

            k_nope = torch.einsum("tr,hdr->thd", c_kv, self.w_uk)
            v = torch.einsum("tr,hrd->thd", c_kv, self.w_uv)
            k_pe_broadcast = k_pe.unsqueeze(1).expand(-1, self.num_q_heads, -1)
            k_chunks.append(torch.cat([k_nope, k_pe_broadcast], dim=-1))
            v_chunks.append(v)
            kv_indptr.append(kv_indptr[-1] + context_len)

        k_full = torch.cat(k_chunks, dim=0).contiguous()
        v_full = torch.cat(v_chunks, dim=0).contiguous()
        kv_indptr_t = torch.tensor(kv_indptr, dtype=torch.int32, device=self.device)
        return k_full, v_full, kv_indptr_t

    def _run_prefill(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        prefill_reqs: List[Dict],
        total_prefill_tokens: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        kv_indptr: torch.Tensor,
        aiter,
    ) -> None:
        """Real aiter persistent fp8 prefill kernel, mirroring
        `AiterAttnBackend.mla_fp8_prefill_attn` + `make_mla_prefill_ps_meta_data`
        in real sglang's `aiter_backend.py` (see module docstring)."""
        q_slice = query[:total_prefill_tokens].contiguous()
        num_reqs = len(prefill_reqs)
        max_seqlen_q = max(r["query_len"] for r in prefill_reqs)

        qo_indptr = [0]
        for req in prefill_reqs:
            qo_indptr.append(qo_indptr[-1] + req["query_len"])
        qo_indptr_t = torch.tensor(qo_indptr, dtype=torch.int32, device=self.device)

        seq_lens = torch.tensor(
            [r["context_len"] for r in prefill_reqs], dtype=torch.int32, device=self.device
        )
        total_kv_tokens = k_full.shape[0]
        fp8_kv_indices = torch.arange(
            total_kv_tokens, dtype=torch.int32, device=self.device
        )

        num_kv_head = 1
        gqa_ratio = self.num_q_heads // num_kv_head
        qhead_granularity = gqa_ratio
        qlen_granularity = max(1, _TILE_Q // qhead_granularity)
        kvlen_granularity = max(128, self.block_size)

        shapes = aiter.get_ps_metadata_info_v1(
            batch_size=num_reqs,
            num_head_k=num_kv_head,
            max_qlen=max_seqlen_q,
            qlen_granularity=qlen_granularity,
        )
        (
            (work_meta_shape, work_meta_dtype),
            (work_indptr_shape, work_indptr_dtype),
            (work_info_shape, work_info_dtype),
            (reduce_indptr_shape, reduce_indptr_dtype),
            (reduce_final_map_shape, reduce_final_map_dtype),
            (reduce_partial_map_shape, reduce_partial_map_dtype),
        ) = shapes
        work_metadata = torch.empty(work_meta_shape, dtype=work_meta_dtype, device=self.device)
        work_indptr = torch.empty(work_indptr_shape, dtype=work_indptr_dtype, device=self.device)
        work_info = torch.empty(work_info_shape, dtype=work_info_dtype, device=self.device)
        reduce_indptr = torch.empty(
            reduce_indptr_shape, dtype=reduce_indptr_dtype, device=self.device
        )
        reduce_final_map = torch.empty(
            reduce_final_map_shape, dtype=reduce_final_map_dtype, device=self.device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_shape, dtype=reduce_partial_map_dtype, device=self.device
        )

        aiter.get_ps_metadata_v1(
            qo_indptr_t.cpu(),
            kv_indptr.cpu(),
            seq_lens.cpu(),
            gqa_ratio,
            num_kv_head,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=self.block_size,
            is_causal=True,
        )

        fp8 = aiter.dtypes.fp8
        q_fp8 = q_slice.to(fp8)
        k_fp8 = k_full.to(fp8)
        v_fp8 = v_full.to(fp8)
        one_scale = torch.ones((), dtype=torch.float32, device=self.device)

        logits = torch.empty(
            (reduce_partial_map.size(0) * _TILE_Q, self.num_q_heads, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        attn_lse = torch.empty(
            (reduce_partial_map.size(0) * _TILE_Q, self.num_q_heads),
            dtype=torch.float32,
            device=self.device,
        )
        final_lse = torch.empty(
            (total_prefill_tokens, self.num_q_heads),
            dtype=torch.float32,
            device=self.device,
        )
        o_slice = torch.empty(
            total_prefill_tokens,
            self.num_q_heads,
            self.v_head_dim,
            dtype=query.dtype,
            device=self.device,
        )

        aiter.mla_prefill_ps_asm_fwd(
            q_fp8,
            k_fp8,
            v_fp8,
            qo_indptr_t,
            kv_indptr,
            fp8_kv_indices,
            work_indptr,
            work_info,
            max_seqlen_q,
            self.softmax_scale,
            True,  # is_causal
            logits,
            attn_lse,
            o_slice,
            one_scale,
            one_scale,
            one_scale,
        )
        aiter.mla_reduce_v1(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            _TILE_Q,
            o_slice,
            final_lse,
        )
        output[:total_prefill_tokens] = o_slice

    def _project_decode_query(
        self,
        query: torch.Tensor,
        decode_reqs: List[Dict],
        total_prefill_tokens: int,
    ) -> torch.Tensor:
        """Absorb `W_UK` into the decode queries' nope component (batched
        across all decode requests at once) so they attend directly against
        the compressed cache — the one step real aiter decode still expects
        the caller to have done, confirmed via `aiter_backend.py`."""
        num_decode = len(decode_reqs)
        q_slice = query[total_prefill_tokens : total_prefill_tokens + num_decode]
        q_nope = q_slice[..., : self.qk_nope_head_dim]
        q_rope = q_slice[..., self.qk_nope_head_dim :]
        q_latent_nope = torch.einsum("thd,hdr->thr", q_nope, self.w_uk)
        return torch.cat([q_latent_nope, q_rope], dim=-1).contiguous()

    def _paged_metadata(
        self, reqs: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Build aiter's CSR-style paged-batch metadata
        (`qo_indptr`/`kv_indptr`/`kv_indices`/`kv_last_page_lens`) from
        Frontier's existing per-request `block_table`/`context_len`
        bookkeeping. Used for decode, which (unlike prefill) calls directly
        against the paged compressed cache."""
        qo_indptr = [0]
        kv_indptr = [0]
        kv_indices: List[int] = []
        kv_last_page_lens: List[int] = []
        max_seqlen_q = 0

        for req in reqs:
            qo_indptr.append(qo_indptr[-1] + req["query_len"])
            max_seqlen_q = max(max_seqlen_q, req["query_len"])

            block_table = req["block_table"]
            kv_indices.extend(block_table)
            kv_indptr.append(kv_indptr[-1] + len(block_table))

            context_len = req["context_len"]
            kv_last_page_lens.append(((context_len - 1) % self.block_size) + 1)

        to_i32 = lambda values: torch.tensor(  # noqa: E731
            values, dtype=torch.int32, device=self.device
        )
        return (
            to_i32(qo_indptr),
            to_i32(kv_indptr),
            to_i32(kv_indices),
            to_i32(kv_last_page_lens),
            max_seqlen_q,
        )

    def _run_decode(
        self,
        q_absorbed: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        decode_reqs: List[Dict],
        total_prefill_tokens: int,
        mla_decode_fwd,
    ) -> None:
        num_decode = len(decode_reqs)
        qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, _ = self._paged_metadata(
            decode_reqs
        )
        o_slice = torch.empty(
            num_decode,
            self.num_q_heads,
            self.v_head_dim,
            dtype=q_absorbed.dtype,
            device=self.device,
        )
        # aiter expects an explicit (size-1) num_kv_heads dim; Frontier's
        # get_cache_block/concat_and_cache_mla tolerate the 3D layout without
        # it, so add it here as a view (no copy) rather than changing the
        # cache's stored shape.
        mla_decode_fwd(
            q_absorbed,
            kv_cache.unsqueeze(2),
            o_slice,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            1,  # max_seqlen_q: always 1 query token per decode request
            # page_size/sm_scale passed by keyword: some aiter builds insert
            # page_size/nhead_kv positional params right after max_seqlen_q.
            # page_size defaults to 1 in that signature, which silently
            # misinterprets kv_indices as page_size=1-granularity offsets
            # instead of this wrapper's real block_size-granularity block
            # indices — only trips a hardware fault at batch>1/larger scale,
            # not the small single-request case, so pass it explicitly.
            page_size=self.block_size,
            nhead_kv=1,
            sm_scale=self.softmax_scale,
        )
        output[total_prefill_tokens : total_prefill_tokens + num_decode] = o_slice
