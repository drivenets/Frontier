# Run 2026-09-15 13:59 — linear ops, spike reproduction attempt (diagnostic, not for training)

Jobs 21323 (`linear_op_8workers_run1.csv`, 13:59:54–14:00:49), 21324 (`linear_op_8workers_run2.csv`, 14:00:49–14:01:46),
21325 (`linear_op_1worker.csv`, `--num_gpus 1`, 14:01:46–14:01:55); all `amd-mi355x-9`, image `v0.5.11-rocm700-mi35x`.

## What was run
Same linear_op command as the dense run, `--num_tokens_list` = 1960…1985 (26 values) × TP {1,2,4,8} = 104 rows per file;
8 GPU workers (twice, reproducing the dense run's lockstep dispatch) and 1 worker (serial).

## Why
The dense run (`2026-09-15_1308_linear_op_dense3327`) has one 150–270 ms `attn_pre_proj` run at timed index 11 in every row of
tokens 1968–1975 × 4 TP. Question: shape-triggered kernel event (would recur) or one-off stall (would not)?
Full write-up: `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/03_linear_op_spike_investigation.md`.

## Results
No run above 20× its row median in any of the three files (worst 1.2 ms vs 0.10 ms median). Not reproducible. Not a full grid:
do not feed these files to the sanity check or the regressor.

Common to every 2026-09-15/16 run: repo `smatar/qwen3-30b-mi355-profiling` in `/home/dn/Frontier-qwen3-profiling`, rsync'd to
`/opt/shared/frontier-qwen3-profiling/Frontier` on the Memphis cluster; Slurm partition `XAI`, one node per job (8× MI355X gfx950);
Frontier profilers (`frontier.profiling.attention.main` / `linear_op.main`), BF16, HIP-event timing, 3 warm-up + 50 timed runs per
shape, every run recorded (`time_stats.<op>.samples`). Sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`.
Schema: see `../../README.md`. Times in this file are the cluster clock as reported by `sacct`.
