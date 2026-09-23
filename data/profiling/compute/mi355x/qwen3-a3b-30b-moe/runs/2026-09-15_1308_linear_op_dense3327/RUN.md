# Run 2026-09-15 13:08 — linear ops, dense 3,327-token grid

> **Data location (moved 2026-09-23).** The CSVs of this run are not in git. They live on the shared filesystem at
> `/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-15_1308_linear_op_dense3327/` with `SHA256SUMS` and a copy of this file;
> this RUN.md is the pointer. The dataset README's runs index lists the same path.

Job 21313, `amd-mi355x-9`, 13:08:59–13:11:01. Image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`. Code `da1ae75` (+ sbatch knob
`LINEAR_TOKENS_PY`, committed in `7f4dfa5`).

## What was run
Same command as `2026-09-15_1142_linear_op_grid386` with `--num_tokens_list` = every count 1…2048, every 8th from 2056 to 8192,
every 16th from 8208 to 16384, minus 4000.

| Shapes | Reps |
|---|---|
| 3,327 token counts × TP {1,2,4,8} = 13,308 rows | 3 + 50 runs (`emb`: 6 + 100) |

## What is different and why
User request: ~3k shapes for the linear ops ("collecting data is cheap"). The grid is a strict superset of the 386-value grid.

## Results
`sanity_check.py` PASS. **This file is the canonical `linear_op.csv` at the dataset root.**
Anomaly: 32 rows (tokens 1968–1975 × 4 TP) have one `attn_pre_proj` run at 150–270 ms (timed run 11), medians unaffected.
Investigated in `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/03_linear_op_spike_investigation.md`; not reproducible
(see `2026-09-15_1359_linear_op_spike_repro`). The re-run `2026-09-16_0629_linear_op_dense3327_rerun` reproduced the spike exactly (same tokens, same run index, other node).

Common to every 2026-09-15/16 run: repo `smatar/qwen3-30b-mi355-profiling` in `/home/dn/Frontier-qwen3-profiling`, rsync'd to
`/opt/shared/frontier-qwen3-profiling/Frontier` on the Memphis cluster; Slurm partition `XAI`, one node per job (8× MI355X gfx950);
Frontier profilers (`frontier.profiling.attention.main` / `linear_op.main`), BF16, HIP-event timing, 3 warm-up + 50 timed runs per
shape, every run recorded (`time_stats.<op>.samples`). Sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`.
Schema: see `../../README.md`. Times in this file are the cluster clock as reported by `sacct`.
