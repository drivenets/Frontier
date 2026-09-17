# Run 2026-09-15 13:15 — attention, AITER, 64k context, blocks 1 and 16

Jobs: 21316 (block 16, `amd-mi355x-9`, 13:15:53–13:37:46) and 21317 (block 1, `amd-mi355x-9`, 13:37:46–13:59:54).
Image `lmsysorg/sglang:v0.5.17-rocm720-mi35x`. Code `da1ae75` + sbatch env overrides (committed in `7f4dfa5`).

## What was run
As `2026-09-15_1142_attention_16k` except `MAX_SEQ_LEN=MAX_MODEL_LEN=65536`, decode KV list `128 … 32768 65536` (also for true-mixed
decode), true-mixed prefill chunks `1024 4096 8192 16384 32768`.

| Rows | Shapes | Reps |
|---|---|---|
| prefill (even) | 10,879 attempts per (block, TP) | 3 + 50 |
| decode (even) | 162 shapes per (block, TP) minus memory-filtered | 3 + 50 |
| true-mixed | 800 per (block, TP) | 3 + 50 |

Per block: 44,099 standard + 3,190 true-mixed rows. `attention*.csv` in this folder = union of the two block files.

## What is different and why
User request: contexts beyond 16k. Prefill work is ~38× the 16k cell (O(n²)); wall time 22 min per cell.

## Results
`sanity_check.py <this dir> --max-seq-len 65536` PASS. Memory-filtered decode shapes: TP1 29, TP2 18, TP4 9, TP8 9 (per block);
all 800 true-mixed shapes present at every cell.

Common to every 2026-09-15/16 run: repo `smatar/qwen3-30b-mi355-profiling` in `/home/dn/Frontier-qwen3-profiling`, rsync'd to
`/opt/shared/frontier-qwen3-profiling/Frontier` on the Memphis cluster; Slurm partition `XAI`, one node per job (8× MI355X gfx950);
Frontier profilers (`frontier.profiling.attention.main` / `linear_op.main`), BF16, HIP-event timing, 3 warm-up + 50 timed runs per
shape, every run recorded (`time_stats.<op>.samples`). Sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`.
Schema: see `../../README.md`. Times in this file are the cluster clock as reported by `sacct`.
