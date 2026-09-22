# Task: replace the torch RoPE fallback in linear_op profiling (opened 2026-09-17; **implemented the same day as TASK-1 of the recollection plan**)

**Status (2026-09-17, TASK-1 of `.claude/plans/linear-op-recollection-contract-a.md`).** The premise "vLLM's `get_rope()` signature
mismatches" is wrong for the linear_op image: vLLM `0.9.2rc2.dev2065` accepts Frontier's call (Slurm job 21402). But vLLM's own
`RotaryEmbedding` on ROCm dispatches `forward_hip` → `forward_native` (17 torch kernels, 45–160 µs) unless AITER is enabled, and never
calls the fused `rotary_embedding` op (jobs 21403/21405). SGLang's `RotaryEmbedding.forward_hip` on the same image launches the fused
`rotary_embedding_kernel` (job 21409: 4.0/5.6/27.9 µs at 8/512/4096 tokens, TP1), numerically identical to Frontier's own
`RotaryEmbedding` calling `vllm._custom_ops.rotary_embedding` (4.1/5.5/28.0 µs). Implemented: `get_rope` prefers Frontier's class with
the fused op; the torch fallback is per-head (correct) and no longer forced by the collection script; every row records `attn_rope_impl`;
`sanity_check.py` rejects non-`vllm_kernel` rows unless `--allow-rope-fallback`. Scope name `attn_rope` kept (the attention trainer
requires `time_stats.attn_rope.median`); disambiguation is by the column, not by a rename (deviation from step 2 below). AITER's own
`get_rope` is also correct (job 21409: 7.6/8.5/19.9 µs) but is not what SGLang launches for this config; the wrong AITER output seen in
job 21406 came from vLLM's `forward_hip_rocm_aiter` wrapper.


**Why it is the most serious open item.** `attn_rope` in every linear_op run is the pure-PyTorch fallback selected by
`FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1` (sbatch line 80; cookbook gotcha 4). `_apply_rotary_pos_emb` receives `cos` of width 64
(`cos_sin_cache.chunk(2)` of a 128-wide cache), sets `rotary_dim = 64`, and rotates `q[..., :64]` of the flattened `[N, heads·128]`
tensor — the first half of head 0 — passing every other head through (`rope_fallback_check.py`: untouched fraction 0.9844/0.9688/0.9376/
0.8751 at TP 1/2/4/8 = 1 − 64/(heads·128); max |fallback − reference neox RoPE| 6.6–8.8 for unit-normal inputs). Timing it — 13 launch-bound
torch kernels plus two full copies — produces a cost-model input for a computation that is not RoPE. Both timing columns and the
`record_function` trial inherit this.

**Scope of the fix (plan-loop input).**
1. Use vLLM's real op: adapt `frontier/profiling/common/layers/rotary_embedding.py::get_rope` to the bundled vLLM's `get_rope`
   signature (the mismatch that motivated the fallback; vLLM 0.9.2rc2 in `v0.5.11-rocm700-mi35x`) so `RotaryEmbedding.forward` calls
   `vllm._custom_ops.rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox_style)` — one kernel over all heads.
2. If the fallback must remain as a portable path, fix its numerics: `cos`/`sin` must be applied per head (`q.view(N, heads, head_dim)`),
   `rotary_dim` = head_dim (128), neox pairing (i, i+64) — and rename the scope so it is never confused with the vLLM kernel.
3. Numerical acceptance: fallback-vs-vLLM-kernel max abs diff ≤ 1e-2 (bf16) on random inputs at TP 1/2/4/8 and positions 0…4095; the
   existing CPU reference in `rope_fallback_check.py` becomes a unit test.
4. Re-time `attn_rope` on the validation grid in both columns; expect one kernel of a few µs to ≈ 100 µs (not 13 kernels) and update
   `09_` §4 (cross-validation scope) and the dataset README §5b.
5. Data handling: the existing `attn_rope` columns in `runs/*` are kept and marked invalid in their RUN.md (already done for the two
   2026-09-17 runs); the root `linear_op.csv` README gets a one-line warning.

**Out of scope:** any change to the timing method or to the other ops. **Owner/venue:** run through `/plan-loop` then `/implement-loop`;
a Jira story in the AIGPU project can be opened from this file if wanted.
