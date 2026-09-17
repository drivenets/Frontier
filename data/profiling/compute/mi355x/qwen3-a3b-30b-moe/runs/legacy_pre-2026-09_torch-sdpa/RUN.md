# Legacy data (before this workstream) — TORCH_SDPA backend

**Do not use for the AITER regressors.** Kept only for comparison.

- `attention_sdpa_block16.csv`, `attention_true_mixed_sdpa_block16.csv`, `attention_combined_sdpa_block16.csv`: TORCH_SDPA
  (portable reference kernel, not the production aiter kernels), block_size 16 only, `max_model_len` 9472, 5 timed runs per
  shape, no per-run samples, no `head_dim` column. Collected before 2026-09-10 (see `profiling_knowledge/MI355X_FOUR_MODEL_PROFILING.md`).
- `linear_op_maxtokens4096.csv`: linear ops on the profiler's default grid up to 4096 tokens, 20 timed runs, no samples.
- `moe.csv`: MoE ops (vLLM Triton fused_moe kernel, not aiter), up to 4096 tokens. MoE was **not** re-collected in this
  workstream (deferred: the profiler has no aiter MoE path).

How to tell legacy from new: legacy rows have `attention_backend == "TORCH_SDPA"` and lack `head_dim`, `warmup_steps`,
`active_steps`, `time_stats.<op>.samples`; `count` is 5 or 20 instead of 50.

Common to every 2026-09-15/16 run: repo `smatar/qwen3-30b-mi355-profiling` in `/home/dn/Frontier-qwen3-profiling`, rsync'd to
`/opt/shared/frontier-qwen3-profiling/Frontier` on the Memphis cluster; Slurm partition `XAI`, one node per job (8× MI355X gfx950);
Frontier profilers (`frontier.profiling.attention.main` / `linear_op.main`), BF16, HIP-event timing, 3 warm-up + 50 timed runs per
shape, every run recorded (`time_stats.<op>.samples`). Sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`.
Schema: see `../../README.md`. Times in this file are the cluster clock as reported by `sacct`.
