# `linear_op_kernel_only.csv` provenance

No `PROVENANCE.md` existed for this file before Track B Step 46.

| file | status | collected |
|---|---|---|
| `linear_op_kernel_only.tp1_2.csv` | archived | pre-Step-46 (`tp∈{1,2}`, 64 rows) |
| `linear_op_kernel_only.csv` | **live** | above + Track B Step 46 (`tp∈{4,8}` added, real hardware, `xai-5`/`amd-mi355x-5`) |

128 rows total: `tp∈{1,2,4,8}`, `num_tokens∈{1..32}`, `--is_moe` (matching
this model's own `linear_op.csv` convention — `qwen3-a3b-30b-moe` has no
dense-MLP layers), no `--use_fp8` (matches `attention_kernel_only.csv`'s
own finding for this model), `--profile_method record_function` (alias
`kernel_only`).

Step 46's `tp=4`/`tp=8` collection calls each echoed a second, sparser
`num_tensor_parallel_workers=1` row per `num_tokens` (same TP-invariant
"replicated op" behavior `deepseek-v3`'s own file documents) — discarded,
keeping only the genuine `tp=4`/`tp=8` rows; the pre-existing `tp=1`/`tp=2`
rows are untouched.

See `docs/tasks/track-b-step46-extend-collect-report.md` in `dc-sim` for
the Step 46 collection record.
