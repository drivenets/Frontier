# Run 2026-09-16 06:29 — linear ops, dense 3,327-token grid, RE-RUN

Job 21334, `amd-mi355x-8` (the first dense run was on node 9), 06:29:05–06:31:04 (1:59). Image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`.
Code `7f4dfa5`. `COLLECT_DIR=data/profiling_linear3k_rerun`.

## What was run
Byte-for-byte the same command as `2026-09-15_1308_linear_op_dense3327` (`STAGE=linear_op`, `LINEAR_TOKENS_PY` default dense grid).

| Shapes | Reps |
|---|---|
| 3,327 token counts (1…2048 dense, 2056…8192 step 8, 8208…16384 step 16, minus 4000) × TP {1,2,4,8} = 13,308 rows | 3 + 50 runs (`emb`: 6 + 100) |

## What is different and why
Nothing in the command. Purpose: the first dense run had one unexplained 150–270 ms `attn_pre_proj` run at timed index 11 for tokens
1968–1975 × every TP, and a 26-token repro (`2026-09-15_1359_linear_op_spike_repro`) did not trigger it. This run tests whether the
full sweep reproduces it (systematic) or not (one-off). Write-up: `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/03_linear_op_spike_investigation.md`.

## Results
`sanity_check.py` PASS (13,308 rows, 3,327 token values × 4 TP).

**The spike reproduced exactly**: `attn_pre_proj`, tokens 1968–1975, timed run 11, 150–263 ms, all four TP passes, on a different node
and a different day. It is therefore a **systematic event tied to the sweep**, not an environmental one-off. The 8 tokens are exactly
the 8 tasks at per-worker position 169 (0-based) of the descending token list, one per GPU worker, in every TP pass.

Medians are unaffected and agree with the first dense run (rerun/first median ratio: `attn_pre_proj` p5 0.914, p50 0.998, p95 1.064;
`attn_post_proj` p50 0.994; `attn_rope` p50 0.997; `input_layernorm` p50 1.004). Other >20× single-run outliers: one `emb` warm-up run
(run 0, tokens 12896, TP1: 1.47 ms vs 21 µs), as in the first run.

Canonical decision: root `linear_op.csv` stays the first dense run; this file is its independent replicate. Either is usable for
`median`-based training; for sample-level analysis drop timed run 11 of rows with `num_tokens` 1968–1975 in both.
