# `attention_kernel_only.csv` provenance

No `PROVENANCE.md` existed for this file before Track B Step 46 — it was
collected across Track B Steps 17/19/20 (per that era's own reports) but
never given a provenance record of its own; this file starts one now
rather than reconstructing that earlier history from CSV inspection
alone.

| file | status | collected |
|---|---|---|
| `attention_kernel_only.tp1_2.csv` | archived | pre-Step-46 (`tp∈{1,2}`, 352 rows — includes mixed-batch profiling rows beyond the plain decode/prefill grid; not re-derived here) |
| `attention_kernel_only.csv` | **live** | above + Track B Step 46 (`tp∈{4,8}` added, real hardware, `xai-5`/`amd-mi355x-5`, GPUs 0-3/0-7) |

490 rows total. Step 46's own addition: `attention_backend=TORCH_SDPA`
(dense GQA, not MLA — `qwen3-a3b-30b-moe` has no latent-MLA path), no
`--use_fp8` (this model's config has `quant_method=None`; passing
`--use_fp8` raises `ValueError: FP8 quantization config mismatch`,
confirmed live), `block_size=16`, `--profile_method record_function`
(alias `kernel_only`):

- decode: `tp∈{4,8}` × `batch∈{1,2,4,8,12,16,24,32}` ×
  `kv∈{0,8,16,24,32,48,64,96}` — 64 points per `tp`, 128 new rows.
- prefill: `tp∈{4,8}` × `total_tokens∈{32,64,96,128,160}` — 5 points per
  `tp`, 10 new rows, from `--max_seq_len 160` with `--profile_only_prefill`
  (same convention as `deepseek-v3`'s own file).

This model's file has no `batch_num_decode_tokens`/`batch_num_prefill_tokens`
columns (unlike `deepseek-v3`'s MLA-derived file) — decode vs. prefill rows
are instead distinguished by `total_prefill_tokens` (`>0` for a genuine
prefill row, `0` for decode), confirmed directly against the existing
`tp∈{1,2}` data before writing Step 46's own coverage checker.

Merged from four new collection calls (decode×tp4, decode×tp8,
prefill×tp4, prefill×tp8) by concatenation — no echo-row subtlety for this
family (confirmed directly, matching `deepseek-v3`'s own finding).

See `docs/tasks/track-b-step46-extend-collect-report.md` in `dc-sim` for
the Step 46 collection record.
