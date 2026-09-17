# Run 2026-09-15 11:42 — attention, AITER, 16k context, blocks 1 and 16

Jobs: 21310 (block 16, `amd-mi355x-8`, 11:42:46–11:46:28) and 21311 (block 1, `amd-mi355x-9`, 11:43:29–11:47:16).
Image `lmsysorg/sglang:v0.5.17-rocm720-mi35x`. Code `0fcac5f` + sbatch fixes (later committed in `b65842c`).

## What was run
`STAGE=attention`, one cell per job: `--attention_backend AITER --block_size {16|1} --num_tensor_parallel_workers 1 2 4 8
--max_seq_len 16384 --max_model_len 16384 --batch_size_list 1 2 4 8 16 24 32 48 64 96 128 160 192 256 320 384 448 512
--decode_kv_cache_size_list 128 512 1024 2048 4096 8192 16384 --enable_chunked_prefill_grid_search --enable_true_mixed
--true_mixed_prefill_batch_sizes 1 2 --true_mixed_prefill_chunk_sizes 1024 4096 8192 --true_mixed_decode_batch_sizes <batch list>
--true_mixed_decode_kv_cache_sizes <kv list> --precision BF16 --profile_method cuda_event --num_gpus 4`.

| Rows | Shapes | Reps |
|---|---|---|
| prefill (even): 2,646 attempts per (block, TP), batch 1, 178 chunk sizes × chunk positions; 269 shapes measured ×2 and 3 ×3 by the generator | 8 (block,TP) cells | 3 + 50 runs each |
| decode (even): 126 shapes per (block, TP) minus memory-filtered (TP1: 9, TP2: 3 dropped) | | 3 + 50 |
| true-mixed: 360 per (block, TP) | | 3 + 50 |

Totals: 11,076 standard + 1,440 true-mixed rows per block.

## What is different and why
First AITER collection for this model. The pinned image `v0.5.11-rocm700` was replaced by `v0.5.17-rocm720` after the pilot
showed aiter's `mha_batch_prefill` faulting for head_dim 128 with paged KV (see `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/02_run_record.md` (repo-relative)).

## Results
`sanity_check.py` PASS. The canonical `attention*.csv` at the dataset root are the union of this run's block-1 and block-16 files.
Slowest of the 53 runs is run 0 (warm-up) in ~88 % of prefill rows; decode outliers are spread uniformly.

Common to every 2026-09-15/16 run: repo `smatar/qwen3-30b-mi355-profiling` in `/home/dn/Frontier-qwen3-profiling`, rsync'd to
`/opt/shared/frontier-qwen3-profiling/Frontier` on the Memphis cluster; Slurm partition `XAI`, one node per job (8× MI355X gfx950);
Frontier profilers (`frontier.profiling.attention.main` / `linear_op.main`), BF16, HIP-event timing, 3 warm-up + 50 timed runs per
shape, every run recorded (`time_stats.<op>.samples`). Sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`.
Schema: see `../../README.md`. Times in this file are the cluster clock as reported by `sacct`.
