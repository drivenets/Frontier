# Run 2026-09-15 13:15 — attention, AITER, 32k context, blocks 1 and 16

Jobs: 21314 (block 16, `amd-mi355x-8`, 13:15:53–13:22:31) and 21315 (block 1, `amd-mi355x-8`, 13:22:31–13:29:49).
Image `lmsysorg/sglang:v0.5.17-rocm720-mi35x`. Code `da1ae75` + sbatch env overrides (committed in `7f4dfa5`).

## What was run
As `2026-09-15_1142_attention_16k` except `MAX_SEQ_LEN=MAX_MODEL_LEN=32768`, decode KV list `128 512 1024 2048 4096 8192 16384 32768`
(also for true-mixed decode), true-mixed prefill chunks `1024 4096 8192 16384`.

| Rows | Shapes | Reps |
|---|---|---|
| prefill (even) | 5,359 attempts per (block, TP) | 3 + 50 |
| decode (even) | 144 shapes per (block, TP) minus memory-filtered | 3 + 50 |
| true-mixed | 560 per (block, TP) | 3 + 50 |

Per block: 21,979 standard + 2,240 true-mixed rows. `attention*.csv` in this folder = union of the two block files (built with
`float_precision="round_trip"`).

## What is different and why
User request: contexts beyond 16k ("let's also do 32k and 64k"). Job 21314's final copy into the collect dir failed (the per-context
collect dir was not world-writable for the root-squashed container); the files were copied by hand from the work tree. Fixed in the sbatch.

## Results
`sanity_check.py <this dir> --max-seq-len 32768` PASS. Memory-filtered decode shapes: TP1 18, TP2 9, TP4 3, TP8 3 (per block);
all 560 true-mixed shapes present at every cell.

Common to every 2026-09-15/16 run: repo `smatar/qwen3-30b-mi355-profiling` in `/home/dn/Frontier-qwen3-profiling`, rsync'd to
`/opt/shared/frontier-qwen3-profiling/Frontier` on the Memphis cluster; Slurm partition `XAI`, one node per job (8× MI355X gfx950);
Frontier profilers (`frontier.profiling.attention.main` / `linear_op.main`), BF16, HIP-event timing, 3 warm-up + 50 timed runs per
shape, every run recorded (`time_stats.<op>.samples`). Sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`.
Schema: see `../../README.md`. Times in this file are the cluster clock as reported by `sacct`.
