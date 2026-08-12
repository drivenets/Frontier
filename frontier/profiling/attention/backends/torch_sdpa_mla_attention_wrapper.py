# Copyright 2023 The Sarathi team.

"""Portable (CUDA + ROCm) MLA attention wrapper for profiling.

Implements DeepSeek-style Multi-head Latent Attention (MLA) using only
`torch.nn.functional.scaled_dot_product_attention` and plain tensor ops, so
it runs on any device PyTorch supports, including AMD ROCm (FlashInfer's MLA
kernel is NVIDIA-only and has no ROCm build).

MLA algorithm modeled here (matches DeepSeek-V2/V3's published design and
vLLM's weight-absorbed MLACommonImpl decode path):

  - The attention module receives a per-token compressed KV latent `c_kv`
    (width `kv_lora_rank`) and a decoupled RoPE key `k_pe` (width
    `qk_rope_head_dim`), not per-head dense K/V. Both are cached together as
    a single blob per token (kv_factor=1, num_kv_heads=1) — this compressed
    cache is the entire point of MLA and why it needs its own memory layout.
  - Prefill has no absorption benefit in real MLA either: `c_kv` is
    up-projected per-head to full K-nope and V via `W_UK`/`W_UV`, concatenated
    with the (broadcast, unprojected) `k_pe`, and run through ordinary causal
    SDPA. This is `attn_mla_prefill_kv_up_proj` + `attn_mla_prefill`.
  - Decode uses weight absorption to avoid materializing full per-head K: the
    query's nope component is projected through `W_UK` into the same rank
    as `c_kv` (`attn_mla_decode_q_latent_proj`), so decode attention
    (`attn_mla_decode`) runs directly against the compressed cache as an
    MQA-style op with head size `kv_lora_rank + qk_rope_head_dim` — matching
    `LATENT_MLA_ATTENTION_FAMILY.runtime_meta_contract`. The attention output
    then comes back in latent (rank) space and must be up-projected to real
    `v_head_dim` via `W_UV` (`attn_mla_v_up_proj`) before it leaves the
    module — this is the sixth and last scope.
  - `W_UK`/`W_UV` are randomly initialized once in `init()`. Their values are
    irrelevant to profiling (only real trained weights would matter for
    output *correctness*, not for matmul *latency*, which is all Frontier's
    compute profiling measures), but their shapes must be exact, and they are
    derived directly from `kv_lora_rank`/`qk_nope_head_dim`/`v_head_dim`.

Fidelity caveats (see TorchSdpaAttentionWrapper's module docstring for the
analogous dense-attention discussion — the same tradeoffs apply here):
  - Per-request Python loop + KV gather instead of one fused ragged-batch
    kernel. Slower than a real vendor MLA kernel; treat these numbers as a
    portable, correctness-of-algorithm-shape reference, not peak achievable
    hardware performance.
  - Q's own down/up-projection (`q_lora_rank`) happens outside this module in
    real vLLM (see `LATENT_MLA_ATTENTION_FAMILY.disjoint_model_projection_attrs`)
    and is out of scope here too — this wrapper receives `query` already in
    its final per-head (nope ++ rope) shape.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

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


class TorchSdpaMlaAttentionWrapper(BaseAttentionWrapper):
    """Portable MLA attention backend using torch SDPA.

    See module docstring for the algorithm and its fidelity tradeoffs.
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
                "TorchSdpaMlaAttentionWrapper only supports the latent MLA "
                f"attention family; got {self._attention_family.family_id!r}. "
                "Use TorchSdpaAttentionWrapper for dense attention models."
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
        # Compressed cache width: matches LATENT_MLA_ATTENTION_FAMILY's
        # runtime_meta_contract (kv_lora_rank + qk_rope_head_dim, 1 KV "head").
        self.cache_width = self.kv_lora_rank + self.qk_rope_head_dim

        # W_UK: per-head up-projection from the compressed latent to K-nope
        # (also reused, transposed, for the decode-side query absorption).
        # W_UV: per-head up-projection from the compressed latent to V.
        # Random weights: only shapes matter for latency profiling.
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

        Single blob per token (kv_factor=1), not a separate K/V pair —
        that compression is the entire point of MLA's memory layout.
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
        """Same bookkeeping shape as TorchSdpaAttentionWrapper.begin_forward:
        per-request block table / context length / query offset, plus the
        cache-write slot mapping. See that wrapper for the traversal-order
        rationale (prefill requests first, then decode).
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
        """`query`: (tokens, num_q_heads * qk_head_dim), already the final
        per-head nope++rope Q (see module docstring). `key`: (tokens,
        kv_lora_rank) is the compressed latent `c_kv`. `value`: (tokens,
        qk_rope_head_dim) carries the decoupled RoPE key `k_pe` — matching
        the shapes `AttentionWrapper._make_qkv_tensors` already builds for
        `_uses_latent_mla`.
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
            block_indices = self.slot_mapping // self.block_size
            block_offsets = self.slot_mapping % self.block_size
            kv_cache[block_indices, block_offsets, : self.kv_lora_rank] = c_kv
            kv_cache[block_indices, block_offsets, self.kv_lora_rank :] = k_pe

        with self.get_timer(OperationMetrics.ATTN_MLA_PREFILL_KV_UP_PROJ, layer_id):
            prefill_ctx = self._gather_prefill_up_projected(kv_cache) if self.contains_prefill else None

        with self.get_timer(OperationMetrics.ATTN_MLA_PREFILL, layer_id):
            if self.contains_prefill:
                self._run_prefill(query, prefill_ctx, output)

        with self.get_timer(OperationMetrics.ATTN_MLA_DECODE_Q_LATENT_PROJ, layer_id):
            q_latent = self._project_decode_query(query) if self.contains_decode else None

        with self.get_timer(OperationMetrics.ATTN_MLA_DECODE, layer_id):
            decode_latent_out = (
                self._run_decode(q_latent, kv_cache) if self.contains_decode else None
            )

        with self.get_timer(OperationMetrics.ATTN_MLA_V_UP_PROJ, layer_id):
            if self.contains_decode:
                self._up_project_decode_output(decode_latent_out, output)

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.v_head_dim)

        return output

    def _gather_prefill_up_projected(
        self, kv_cache: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Per prefill request: gather cached (c_kv, k_pe), up-project to
        full per-head K (nope ++ rope) and V. Returns request-index -> (K, V).
        """
        result: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for idx, req in enumerate(self._requests):
            if not req["is_prefill"]:
                continue
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

            # (context_len, H, dn) = (context_len, r) x (H, dn, r)^T per head
            k_nope = torch.einsum("tr,hdr->thd", c_kv, self.w_uk)
            v = torch.einsum("tr,hrd->thd", c_kv, self.w_uv)
            k_pe_broadcast = k_pe.unsqueeze(1).expand(-1, self.num_q_heads, -1)
            k = torch.cat([k_nope, k_pe_broadcast], dim=-1)  # (context_len, H, qk_head_dim)
            result[idx] = (k, v)
        return result

    def _run_prefill(
        self,
        query: torch.Tensor,
        prefill_ctx: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        output: torch.Tensor,
    ) -> None:
        for idx, req in enumerate(self._requests):
            if not req["is_prefill"]:
                continue
            q_start = req["q_start"]
            query_len = req["query_len"]
            k, v = prefill_ctx[idx]

            q = query[q_start : q_start + query_len].transpose(0, 1)  # (H, Lq, qk_head_dim)
            k = k.transpose(0, 1)  # (H, Lctx, qk_head_dim)
            v = v.transpose(0, 1)  # (H, Lctx, v_head_dim)

            out = F.scaled_dot_product_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                is_causal=True,
                scale=self.softmax_scale,
            ).squeeze(0)  # (H, Lq, v_head_dim)

            output[q_start : q_start + query_len] = out.transpose(0, 1)

    def _project_decode_query(self, query: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Weight-absorb W_UK into the decode query's nope component so it
        can attend directly against the compressed (rank-sized) cache.
        """
        result: Dict[int, torch.Tensor] = {}
        for idx, req in enumerate(self._requests):
            if req["is_prefill"]:
                continue
            q_start = req["q_start"]
            q = query[q_start]  # (H, qk_head_dim)
            q_nope = q[:, : self.qk_nope_head_dim]
            q_rope = q[:, self.qk_nope_head_dim :]
            # (H, r) = (H, dn) x (H, dn, r) per head
            q_latent_nope = torch.einsum("hd,hdr->hr", q_nope, self.w_uk)
            result[idx] = torch.cat([q_latent_nope, q_rope], dim=-1)  # (H, cache_width)
        return result

    def _run_decode(
        self,
        q_latent: Dict[int, torch.Tensor],
        kv_cache: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """MQA-style attention: query (per head, absorbed into rank space)
        against the single shared compressed cache. Only the c_kv slice of
        the attention *output* is meaningful in latent space — k_pe carries
        positional signal for scoring only, matching real MLA decode.
        """
        result: Dict[int, torch.Tensor] = {}
        for idx, req in enumerate(self._requests):
            if req["is_prefill"]:
                continue
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

            q = q_latent[idx].unsqueeze(1)  # (H, 1, cache_width)
            k = cached.unsqueeze(0).expand(self.num_q_heads, -1, -1)  # (H, Lctx, cache_width)
            v_latent = cached[:, : self.kv_lora_rank].unsqueeze(0).expand(
                self.num_q_heads, -1, -1
            )  # (H, Lctx, kv_lora_rank) — output stays in latent space

            out = F.scaled_dot_product_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v_latent.unsqueeze(0),
                is_causal=False,
                scale=self.softmax_scale,
            ).squeeze(0).squeeze(1)  # (H, kv_lora_rank)
            result[idx] = out
        return result

    def _up_project_decode_output(
        self,
        decode_latent_out: Dict[int, torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        for idx, req in enumerate(self._requests):
            if req["is_prefill"]:
                continue
            q_start = req["q_start"]
            latent_out = decode_latent_out[idx]  # (H, kv_lora_rank)
            # (H, dv) = (H, r) x (H, r, dv) per head
            out = torch.einsum("hr,hrd->hd", latent_out, self.w_uv)
            output[q_start] = out
