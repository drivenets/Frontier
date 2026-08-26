"""Dense/GQA attention profiling backend built on AMD's ``aiter`` kernels.

Why this backend exists
-----------------------
:class:`TorchSdpaAttentionWrapper` is a portable, correctness-first reference:
it fakes paged attention by gathering KV blocks into contiguous tensors and
calling ``torch.nn.functional.scaled_dot_product_attention``. That gather is
work a real paged kernel never does, and SDPA is not a tuned attention kernel,
so its numbers are conservative by construction.

Real sglang serving on MI355X does not run SDPA — it dispatches AMD's ``aiter``
kernels. This backend calls the same two entry points sglang's own
``aiter_backend.py`` uses for dense/GQA (non-MLA) models, so profiling rows
describe the kernels production actually executes:

===========  =====================================  ==============================
phase        kernel                                 sglang call site
===========  =====================================  ==============================
prefill      ``aiter.ops.mha.mha_batch_prefill_func``   ``aiter_backend.py`` L1524
decode       ``aiter.ops.attention.paged_attention_ragged``  ``aiter_backend.py`` L1633
===========  =====================================  ==============================

``AttentionBackend.AITER_MLA`` (where present) covers the MLA path via
``mla_decode_fwd``/``mla_prefill_fwd``; this backend is its dense/GQA
counterpart and refuses MLA models rather than mismeasuring them.

Validation
----------
Both kernels were checked against an fp32 oracle before this wrapper was
written, alongside torch SDPA measured against the same oracle. Relative error
was 2.5e-3 to 4.4e-3 for both — i.e. ``aiter`` is as close to the truth as SDPA
is, and the residual is bf16 rounding (bf16 eps is 3.9e-3), not a layout or
kernel bug. Cases covered: uniform and ragged batches, partial final pages,
``block_size`` 1 and 16, GQA and MHA head ratios, 8192-token contexts, and
chunked prefill where the query is the tail of a longer sequence.

That last case is the one worth naming: with chunked prefill a top-left-aligned
causal mask is wrong, and the SDPA backend needs an explicit bottom-right mask
to get it right. ``mha_batch_prefill_func`` derives the same alignment natively
from ``causal=True`` plus the paged KV extent, and matches the oracle to SDPA's
own precision.

Fidelity caveats — read before trusting the numbers
---------------------------------------------------
* **Real kernels, real paging.** Unlike the SDPA backend there is no gather:
  both kernels read the paged cache directly through ``kv_indptr`` /
  ``kv_page_indices``, so no synthetic work is timed inside the attention
  scopes.
* **Sliding-window and attention-sink cost is not exercised.** Both kernels
  accept ``window_size``/``sink_ptr``, and gpt-oss genuinely uses alternating
  128-token sliding-window attention with learned per-head sinks. Frontier has
  no sliding-window attention family and no sink concept, so this wrapper
  profiles every layer as full attention to stay consistent with what the
  simulator can represent. Long-context attention cost is overstated as a
  result — the same limitation the model config records, not a new one.
* **BF16 only.** ``logits_soft_cap`` is fixed at 0.0 and the fp8 KV-cache paths
  both kernels expose are unused; FP8 attention profiling is rejected upstream
  by ``_validate_precision``.
* **Decode workspace sizing follows sglang's own formula**, including its
  256-element partition size. A different partition size would change the
  split-K shape and therefore the timing.
"""

from typing import List, Optional, Tuple

import torch

from frontier.attention.model_binding import bind_attention_family
from frontier.attention.ops import AttentionMemoryLayout
from frontier.profiling.attention.backends.base_attention_wrapper import (
    BaseAttentionWrapper,
)
from frontier.profiling.attention.sequence_metadata import SequenceMetadata
from frontier.profiling.common.constants import OperationMetrics
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig

try:
    from aiter.ops.attention import paged_attention_ragged
    from aiter.ops.mha import mha_batch_prefill_func

    HAS_AITER = True
    _AITER_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    HAS_AITER = False
    _AITER_IMPORT_ERROR = exc
    paged_attention_ragged = None
    mha_batch_prefill_func = None


# sglang's aiter_backend.py hardcodes this as _AITER_PARTITION_SIZE_ROCM and
# uses it both to size the workspace and to pick the split-K partition count.
# Keep the two uses in sync — the kernel reads the value we pass, but the
# workspace must already be large enough for it.
_AITER_PARTITION_SIZE_ROCM = 256


class AiterAttentionWrapper(BaseAttentionWrapper):
    """Dense/GQA attention profiling backend using AMD ``aiter`` kernels.

    See the module docstring for the fidelity caveats that apply to every
    number this backend produces.
    """

    _inst = None

    def supports_attention_family(self, attention_family) -> bool:
        """Only dense-compatible families; MLA needs its own kernels and scopes."""
        return (
            attention_family.dense_compatible
            and attention_family.memory_layout is not AttentionMemoryLayout.LATENT_MLA
        )

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        """Initialize the aiter attention wrapper.

        Raises:
            ImportError: If ``aiter`` is not importable.
            NotImplementedError: If the model uses an MLA latent KV cache.
        """
        if not HAS_AITER:
            raise ImportError(
                "aiter is not importable, so AttentionBackend.AITER cannot be used. "
                "aiter is AMD's kernel library and is not available on CUDA hosts; "
                "on ROCm, run inside an image that ships it (e.g. the lmsysorg/sglang "
                "ROCm images) or add its source tree to PYTHONPATH."
            ) from _AITER_IMPORT_ERROR

        self._attention_family = bind_attention_family(model_config).family
        if self._attention_family.memory_layout is AttentionMemoryLayout.LATENT_MLA:
            raise NotImplementedError(
                "AiterAttentionWrapper implements the dense/GQA attention algorithm "
                "only. MLA uses a compressed latent KV cache and aiter's own "
                "mla_decode_fwd/mla_prefill_fwd kernels; use an MLA-specific backend "
                "instead of reusing this one."
            )

        super().init(model_config, parallel_config, block_size, device)

        self.softmax_scale = 1.0 / (self.head_dim**0.5)
        # Both kernels take K/V dequant scales as tensors. Profiling runs BF16
        # caches, so these are the identity.
        self._k_scale = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        self._v_scale = self._k_scale

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.contains_decode = False
        self.num_prefill_tokens = 0
        self.num_total_tokens = 0

        # Ragged paged-KV descriptors, built in begin_forward and consumed in
        # forward. Same layout for both phases: kv_indptr gives each request's
        # slice of kv_page_indices, and kv_last_page_lens says how much of the
        # final page is real.
        self._prefill_cu_seqlens_q: Optional[torch.Tensor] = None
        self._prefill_kv_indptr: Optional[torch.Tensor] = None
        self._prefill_kv_pages: Optional[torch.Tensor] = None
        self._prefill_kv_last_page_lens: Optional[torch.Tensor] = None
        self._prefill_max_q = 0
        self._prefill_max_kv = 0

        self._decode_kv_indptr: Optional[torch.Tensor] = None
        self._decode_kv_pages: Optional[torch.Tensor] = None
        self._decode_kv_last_page_lens: Optional[torch.Tensor] = None
        self._decode_batch_size = 0
        self._decode_max_num_partitions = 0

        # Write targets for this batch's new tokens.
        self._write_block_index: Optional[torch.Tensor] = None
        self._write_block_offset: Optional[torch.Tensor] = None

        # Decode workspace, grown on demand and reused across forwards so its
        # allocation is not timed inside the decode scope.
        self._workspace: Optional[torch.Tensor] = None

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        """Allocate a paged KV cache in the layout aiter expects.

        Shape is ``(2, num_blocks, block_size, num_kv_heads, head_dim)`` rather
        than the FlashInfer/SDPA backends' ``(num_blocks, 2, ...)``. Both
        kernels take K and V as two separate NHD-layout tensors, and putting the
        K/V dimension outermost makes ``kv_cache[0]`` and ``kv_cache[1]``
        contiguous views. With the K/V dimension in the middle they would be
        strided, and passing a strided cache would force a full copy per
        forward — the exact bug that made the SDPA backend's
        ``attn_kv_cache_save`` 150x too slow.
        """
        return torch.randn(
            2,
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    def _page_descriptors(
        self, seq_lens: List[int], block_tables: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(kv_indptr, kv_page_indices, kv_last_page_lens)``.

        ``block_table`` entries are the page ids a request occupies. A request
        holding ``n`` tokens occupies ``ceil(n / block_size)`` pages, of which
        the last holds ``((n - 1) % block_size) + 1`` real tokens.
        """
        indptr = [0]
        pages: List[int] = []
        last_lens: List[int] = []
        for seq_len, table in zip(seq_lens, block_tables):
            num_pages = (seq_len + self.block_size - 1) // self.block_size
            pages.extend(table[:num_pages])
            indptr.append(indptr[-1] + num_pages)
            last_lens.append(((seq_len - 1) % self.block_size) + 1)
        return (
            torch.tensor(indptr, dtype=torch.int32, device=self.device),
            torch.tensor(pages, dtype=torch.int32, device=self.device),
            torch.tensor(last_lens, dtype=torch.int32, device=self.device),
        )

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        """Build ragged paged-KV descriptors for the batch.

        Prefill sequences are ordered before decode sequences, matching the
        query-tensor layout the profiling harness passes to :meth:`forward`.
        """
        self.is_profiling_iteration = False
        self.is_metadata_initialized = True
        self.contains_prefill = False
        self.contains_decode = False

        write_blocks: List[int] = []
        write_offsets: List[int] = []

        prefill_query_lens: List[int] = []
        prefill_seq_lens: List[int] = []
        prefill_tables: List[List[int]] = []
        decode_seq_lens: List[int] = []
        decode_tables: List[List[int]] = []

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

            prefill_query_lens.append(prompt_chunk_len)
            prefill_seq_lens.append(current_total_len)
            prefill_tables.append(seq_metadata.block_table)

            for token_idx in range(processed_prompt_len, current_total_len):
                write_blocks.append(
                    seq_metadata.block_table[token_idx // self.block_size]
                )
                write_offsets.append(token_idx % self.block_size)

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            self.contains_decode = True
            context_len = seq_metadata.seq.get_len()
            decode_seq_lens.append(context_len)
            decode_tables.append(seq_metadata.block_table)

            token_idx = context_len - 1
            write_blocks.append(seq_metadata.block_table[token_idx // self.block_size])
            write_offsets.append(token_idx % self.block_size)

        if self.contains_prefill:
            indptr, pages, last_lens = self._page_descriptors(
                prefill_seq_lens, prefill_tables
            )
            self._prefill_kv_indptr = indptr
            self._prefill_kv_pages = pages
            self._prefill_kv_last_page_lens = last_lens
            cu = [0]
            for query_len in prefill_query_lens:
                cu.append(cu[-1] + query_len)
            self._prefill_cu_seqlens_q = torch.tensor(
                cu, dtype=torch.int32, device=self.device
            )
            self._prefill_max_q = max(prefill_query_lens)
            self._prefill_max_kv = max(prefill_seq_lens)
            num_prefill_tokens = cu[-1]
        else:
            num_prefill_tokens = 0

        if self.contains_decode:
            indptr, pages, last_lens = self._page_descriptors(
                decode_seq_lens, decode_tables
            )
            self._decode_kv_indptr = indptr
            self._decode_kv_pages = pages
            self._decode_kv_last_page_lens = last_lens
            self._decode_batch_size = len(decode_seq_lens)
            self._decode_max_num_partitions = (
                max(decode_seq_lens) + _AITER_PARTITION_SIZE_ROCM - 1
            ) // _AITER_PARTITION_SIZE_ROCM
            self._ensure_workspace(
                self._decode_batch_size, self._decode_max_num_partitions
            )

        self.num_prefill_tokens = num_prefill_tokens
        self.num_total_tokens = num_prefill_tokens + len(decode_seq_lens)
        self._write_block_index = torch.tensor(
            write_blocks, dtype=torch.long, device=self.device
        )
        self._write_block_offset = torch.tensor(
            write_offsets, dtype=torch.long, device=self.device
        )

    def _ensure_workspace(self, batch_size: int, max_num_partitions: int) -> None:
        """Allocate the decode split-K workspace, growing it as needed.

        Sizing follows sglang's own formula: one fp32 partial output per
        (request, head, partition, head_dim) element, plus fp32 exp-sum and
        max-logit scratch per (request, head, partition).
        """
        nbytes_per_qo_elem = torch.finfo(torch.float32).bits // 8
        required = (
            batch_size * self.num_q_heads * max_num_partitions * self.head_dim
        ) * nbytes_per_qo_elem + 2 * (
            batch_size * self.num_q_heads * max_num_partitions
        ) * 4
        if self._workspace is None or self._workspace.numel() < required:
            self._workspace = torch.empty(
                required, dtype=torch.uint8, device=self.device
            )

    def end_forward(self):
        """Release per-batch state. The workspace is kept and reused."""
        self.is_metadata_initialized = False
        self._write_block_index = None
        self._write_block_offset = None
        self._prefill_cu_seqlens_q = None
        self._prefill_kv_indptr = None
        self._prefill_kv_pages = None
        self._prefill_kv_last_page_lens = None
        self._decode_kv_indptr = None
        self._decode_kv_pages = None
        self._decode_kv_last_page_lens = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Run dense/GQA attention through aiter kernels and time each scope.

        Emits the same five scopes as the FlashInfer and SDPA backends
        (``ATTN_INPUT_RESHAPE``, ``ATTN_KV_CACHE_SAVE``, ``ATTN_PREFILL``,
        ``ATTN_DECODE``, ``ATTN_OUTPUT_RESHAPE``) so downstream training and
        simulation consume its CSV rows unchanged.
        """
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            # Memory-profiling iterations carry no block tables; nothing to compute.
            return torch.zeros_like(query)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        output = torch.empty_like(query)
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            if self._write_block_index is None:
                raise RuntimeError("KV cache write plan is not initialized.")
            # A true scatter into the paged cache. K and V are separate
            # contiguous tensors here (see get_cache_block), so neither index
            # expression materialises a copy of the cache.
            key_cache[self._write_block_index, self._write_block_offset] = key
            value_cache[self._write_block_index, self._write_block_offset] = value

        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            if self.contains_prefill:
                # ``out=`` writes straight into the output slice; assigning the
                # kernel's return value instead would time an extra full copy
                # of the prefill activations inside this scope.
                mha_batch_prefill_func(
                    query[: self.num_prefill_tokens],
                    key_cache,
                    value_cache,
                    self._prefill_cu_seqlens_q,
                    self._prefill_kv_indptr,
                    self._prefill_kv_pages,
                    self._prefill_max_q,
                    self._prefill_max_kv,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                    out=output[: self.num_prefill_tokens],
                    kv_last_page_lens=self._prefill_kv_last_page_lens,
                )

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            if self.contains_decode:
                decode_query = query[self.num_prefill_tokens :]
                paged_attention_ragged(
                    output[self.num_prefill_tokens :],
                    self._workspace,
                    decode_query,
                    key_cache,
                    value_cache,
                    self.softmax_scale,
                    self._decode_kv_indptr,
                    self._decode_kv_pages,
                    self._decode_kv_last_page_lens,
                    self.block_size,
                    self._decode_max_num_partitions,
                    None,  # alibi_slopes
                    "auto",  # kv_cache_dtype
                    "NHD",  # kv_cache_layout
                    0.0,  # logits_soft_cap
                    self._k_scale,
                    self._v_scale,
                    None,  # fp8_out_scale
                    _AITER_PARTITION_SIZE_ROCM,
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output
