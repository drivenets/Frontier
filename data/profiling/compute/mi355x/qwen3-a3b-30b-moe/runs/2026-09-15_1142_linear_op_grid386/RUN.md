# Run 2026-09-15 11:42 — linear ops, default 386-token grid

Job 21309, `amd-mi355x-9`, 11:42:29–11:43:29. Image `lmsysorg/sglang:v0.5.11-rocm700-mi35x` (has vllm). Code `0fcac5f`.

## What was run
`STAGE=linear_op`: `--models qwen3-a3b-30b-moe --is_moe --num_tensor_parallel_workers 1 2 4 8 --max_tokens 16384
--num_tokens_list <get_num_tokens_to_profile(16384) minus 4000> --precision BF16 --profile_method cuda_event --num_gpus 8`.
Ops: `attn_pre_proj` (QKV GEMM + QK-norm), `attn_post_proj`, `attn_rope`, `input_layernorm`, `post_attention_layernorm`, `emb`.

| Shapes | Reps |
|---|---|
| 386 token counts (1…16384, profiler default grid, 4000 excluded) × TP {1,2,4,8} = 1,544 rows | 3 + 50 runs (`emb`: 6 + 100, called twice per forward) |

## What is different and why
First linear_op collection with 50 timed runs and per-run samples. 4000 is excluded because it faults the GPU at TP 1 on this
stack (`examples/profiling/profile_mi355x.sh`).

## Results
`sanity_check.py` PASS (as part of the dataset). Superseded as canonical by the dense run (`2026-09-15_1308_linear_op_dense3327`);
kept for comparison — on the 1,544 shared shapes the two runs' `attn_pre_proj` medians agree (ratio p50 1.001, p5 0.95, p95 1.07).
No single-run spikes > 20× anywhere in this file.

Common to every 2026-09-15/16 run: repo `smatar/qwen3-30b-mi355-profiling` in `/home/dn/Frontier-qwen3-profiling`, rsync'd to
`/opt/shared/frontier-qwen3-profiling/Frontier` on the Memphis cluster; Slurm partition `XAI`, one node per job (8× MI355X gfx950);
Frontier profilers (`frontier.profiling.attention.main` / `linear_op.main`), BF16, HIP-event timing, 3 warm-up + 50 timed runs per
shape, every run recorded (`time_stats.<op>.samples`). Sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`.
Schema: see `../../README.md`. Times in this file are the cluster clock as reported by `sacct`.
