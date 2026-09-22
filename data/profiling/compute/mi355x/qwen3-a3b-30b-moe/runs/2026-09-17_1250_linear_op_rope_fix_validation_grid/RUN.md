# Run 2026-09-17 12:50 — linear ops, validation grid, **attn_rope on the fused kernel**, both instruments (TASK-1 acceptance)

Job 21410, `amd-mi355x-7`, 118 s, 8 workers, automatic clock. Image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`. Code ec63f31 + TASK-1 of
`.claude/plans/linear-op-recollection-contract-a.md` (rope fix, `attn_rope_impl` column, `LINEAR_PROFILE_METHODS` loop, tracer cleanup).
Grid `[1,8,64,512,3072,4096,4192,6144,8192]` × TP {1,2,4,8} = 36 rows per file. `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK` **unset**
(first run without it); `FRONTIER_RF_KEEP_TRACES=1` (traces kept on the cluster scratch tree for the kernel-count check, not in this dir).
Files: `linear_op.csv` (cuda_event, two timing columns + probes, as `runs/2026-09-17_1016_…_probe/`), `linear_op_kernel_only.csv`
(record_function: per-scope kineto kernel time, one traced forward of 50 repeated blocks; probe columns NaN — added in TASK-2).
Collected into the scratch tree `data/profiling_rope_fix2` on the cluster checkout and copied here by hand (the staging script is TASK-3);
root `linear_op.csv` sha256 `4d1f399fa59c32a52e20dd3f884fe34198311912ea6d9fa6d67fcd88a4524239` before and after the copy (identical); `git diff --quiet HEAD -- <root>` clean.

## What changed for attn_rope
`attn_rope_impl = vllm_kernel` on 36/36 rows of both files. Kineto traces (two shapes inspected, largest and smallest file): exactly
**1 kernel per attn_rope annotation**, `vllm::rotary_embedding_kernel<c10::BFloat16, true>` (14.2 µs and 3.8 µs device time), against
17 torch kernels for vLLM's own object (job 21403, 41.7–112.7 µs) and 13 for the old forced fallback. Numerics: fused kernel vs float64
reference max |diff| 0.017–0.036 in bf16 (in-container test `test_vllm_kernel_matches_reference`, job 21411); the old fallback was
off by 6–9 on every head but head 0.

| tokens | GPU-bound event ms, TP1 / TP8 (this run) | kernel-only ms, TP1 / TP8 | old fallback GPU-bound ms, TP1 / TP8 (10:16 run) | old/new |
|---|---|---|---|---|
| 1 | 0.0065 / 0.0063 | 0.0038 / 0.0020 | 0.0314 / 0.0290 | 4.8 / 4.6 |
| 8 | 0.0064 / 0.0064 | 0.0054 / 0.0030 | 0.0432 / 0.0392 | 6.7 / 6.1 |
| 512 | 0.0081 / 0.0065 | 0.0066 / 0.0034 | 0.0549 / 0.0538 | 6.8 / 8.3 |
| 4096 | 0.0269 / 0.0085 | 0.0244 / 0.0074 | 0.0787 / 0.0563 | 2.9 / 6.6 |
| 8192 | 0.0465 / 0.0123 | 0.0440 / 0.0110 | 0.0997 / 0.0642 | 2.1 / 5.2 |

The event column exceeds the kernel-only column by ≈2.5–3 µs at every shape (the per-boundary record cost, `09_` §4). The legacy column
is host-bound for rope up to 512 tokens at every TP (legacy/GPU-bound 2.1–2.8) and up to 6144 tokens at TP8 — a 4–6 µs kernel behind a
~15 µs launch.

## Verification
`sanity_check.py <this dir> --tokens-grid "[1,8,64,512,3072,4096,4192,6144,8192]"` → PASS (36 rows, both gates, rope gate, SCLK note:
legacy_start median 793 [629–2403], legacy_end 2412, backlog 2384/2411 — the pre-ramp clock trajectory, unchanged from 10:16);
`test_measurement_validity.py <this>/linear_op.csv --expect green` → GREEN. The kernel-only file is not checked yet (TASK-2).

## Limitations
No clock ramp yet (TASK-2): the legacy pass still starts at ≈0.8 GHz. 50 forwards behind one spin (TASK-2 splits them). Kernels ≤ 10 µs
remain un-cross-validated between instruments (`09_` §4) — that now includes `attn_rope` below ≈3072 tokens.
