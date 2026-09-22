# `linear_op.csv` provenance

Frontier's own `linear_op_input_file` config field hard-codes the filename
`linear_op.csv` (templated only by device/model directory) — exactly one
file at this path is ever "live," matching the same constraint
`moe_kernel_only.PROVENANCE.md` (this project's own established convention
for this situation) already documents for this model's `moe_kernel_only.csv`.

| file | dense MLP ops (`mlp_up_proj`/`mlp_down_proj`/`mlp_act`) | status | collected |
|---|---|---|---|
| `linear_op.csv` | **present** | **live** | Track B Step 30 (real hardware, `xai-3`/`amd-mi355x-3`, GPU index 1) |
| `linear_op.dense_mlp_missing.csv` | absent (`--is_moe` filtered them out) | archived | pre-Track-B, original collection |

## Why the original file was missing an entire op family, not one column

`deepseek-v3` declares `first_k_dense_replace=3`: its first three
transformer layers are ordinary dense feed-forward, every later layer is
mixture-of-experts. The original `linear_op.csv` was collected with
`--is_moe` set (`frontier/profiling/linear_op/main.py`'s own docstring:
"Skip dense MLP ops profiling (MoE models use expert layers profiled by
the MoE module)") — correct for the 58 real MoE layers, wrong for the 3
real dense ones, which use exactly the `MLP` module `--is_moe` disables.
Confirmed directly from `frontier/profiling/linear_op/linear_op_wrapper.py`'s
own default `enabled_ops` (includes `mlp_up_proj`/`mlp_down_proj`/`mlp_act`
unless overridden) and `linear_op_impl.py::GPTBlock` (builds a real, timed
`MLP` when `ffn_sharded_enabled`, a no-op `DummyMLP()` otherwise). See
Track B Step 29's report for how this was first traced, and Step 30's own
report for the full collection record.

## Why this is a merge, not a full recollection

`deepseek-v3`'s real quantization is block-wise FP8
(`weight_block_size=[128,128]`). The real, block-quantized custom op
Frontier's own profiling code calls
(`torch.ops.vllm.apply_w8a8_block_fp8_linear`) does not exist in the
pinned image's vLLM version (`0.27.1`) — genuinely removed, not renamed,
confirmed by grepping the installed package. `dc-sim`'s
`src/integration/profiling/fp8_linear_op_adapter.py` (new, Step 30)
bridges this by calling vLLM's own real replacement
(`torch.ops.vllm.w8a8_triton_block_scaled_mm_func`, the same op vLLM's
own current `Fp8BlockScaledMMLinearKernel` uses), but that replacement's
own kernel runs under an **untuned/default config** on this hardware
(vLLM's own runtime warning: *"Using default W8A8 Block FP8 kernel
config. Performance might be sub-optimal!"* — no tuned config file
exists yet for `AMD_Instinct_MI355_OAM` at these shapes). Recollecting
the *entire* file through this path would have silently replaced every
attention-projection timing this file already had with numbers from a
measurably different (and, checked directly, materially different —
up to ~2.3x at `tp=1`, non-uniformly across `tp`) kernel implementation.

**So this file is a column-wise merge**, keyed on `(num_tensor_parallel_workers,
num_tokens)`: every pre-existing column (`attn_pre_proj`, `attn_rope`,
`attn_post_proj`, `input_layernorm`, `post_attention_layernorm`, `emb`,
`mtp_fusion_proj`, `lm_head_linear`) is preserved bit-for-bit from the
original, untouched `linear_op.dense_mlp_missing.csv`. Only the three new
op families (`mlp_up_proj`, `mlp_act`, `mlp_down_proj`, 18 columns) come
from the new collection, run through the adapter above at the identical
grid (`tp∈{1,2,4,8}`, `num_tokens`: the same 19-point list the original
file used, `--profile_method cuda_event`, `--use_fp8 --block_shape 128 128`,
no `--is_moe`).

## Known limitation, stated plainly

The three new columns' **absolute magnitudes are measured through an
untuned kernel config** and are close to flat across both `tp` and
`num_tokens` (≈0.12–0.14ms throughout) — consistent with fixed
dispatch/launch overhead dominating a small, untuned Triton kernel at
these token counts, not necessarily the real compute-bound scaling a
tuned kernel would show. This is a real, disclosed fidelity gap, not a
silent one: these three columns should be treated as a first, working
approximation — real FP8 math, real hardware, wrong-tuned kernel —
pending either a tuned kernel config for this shape/device or a future
pinned image where `apply_w8a8_block_fp8_linear` (or an equally-tuned
equivalent) is restored.

See `docs/tasks/track-b-step30-deepseek-profiles-report.md` in `dc-sim`
for the full collection record.
