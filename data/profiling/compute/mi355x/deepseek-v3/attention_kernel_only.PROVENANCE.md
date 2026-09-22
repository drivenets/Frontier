# `attention_kernel_only.csv` provenance

Frontier's `attention_kernel_only_input_file` config field hard-codes the
filename `attention_kernel_only.csv` (templated only by device/model
directory) — same constraint this model's `attention.PROVENANCE.md`
documents for the eager file. This file did not exist for `deepseek-v3`
before this task; there is nothing archived to preserve.

| file | status | collected |
|---|---|---|
| `attention_kernel_only.tp1_2.csv` | archived | Track B Step 35 (real hardware, `xai-3`/`amd-mi355x-3`, GPU index 1) — `tp∈{1,2}`, 138 rows |
| `attention_kernel_only.csv` | **live** | Track B Step 35 (`tp∈{1,2}`) + Track B Step 46 (`tp∈{4,8}` added, real hardware, `xai-4`/`amd-mi355x-4`, GPUs 0-3/0-7) |

276 rows total (138 + 138), `block_size=16`, `attention_backend=TORCH_SDPA_MLA`,
`--profile_method record_function` (alias `kernel_only`), `--use_fp8
--block_shape 128 128`:

- decode: `tp∈{1,2,4,8}` × `batch∈{1,2,4,8,12,16,24,32}` ×
  `kv∈{0,8,16,24,32,48,64,96}` — 64 points per `tp`, 256 total. This is
  Step 34's own derived envelope (`tp∈{1,2}`), extended to `tp∈{4,8}` by
  Step 46 to unblock the Step 38 equal-hardware family (Step 44's own
  finding), run through `--profile_only_decode`.
- prefill: `tp∈{1,2,4,8}` × `total_tokens∈{32,64,96,128,160}` — 5 points per
  `tp`, 20 total, from `--max_seq_len 160` with
  `--profile_only_prefill` (the tool's own standard prefill grid is a
  fixed multiple-of-32 sweep driven by `max_seq_len`, not a directly
  settable token list; `160` was chosen to bracket Step 30's own 32-token
  prefill workload with one point at the value itself and four points of
  margin above, up to 5x).

Merged from eight separate collection calls total (Step 35's original four:
decode×tp1, decode×tp2, prefill×tp1, prefill×tp2; Step 46's four more:
decode×tp4, decode×tp8, prefill×tp4, prefill×tp8) by concatenation — column
sets were identical across all eight and no key collisions existed between
them (unlike `linear_op_kernel_only.csv`, this family has no TP-invariant
"replicated op" echo-row behavior to reconcile, confirmed again directly at
`tp∈{4,8}` before merging, Step 46).

The fix that made this collection possible at all:
`_get_allow_zero_cuda_ops_for_current_forward` (patched in
`dc-sim/src/integration/profiling/mla_attention_wrapper_adapter.py`,
Track B Step 35) was never taught about MLA's five conditionally-executed
operators, so a decode-only or prefill-only batch — which genuinely leaves
the other phase's op scopes empty — raised
`RecordFunctionTracer: operation '...' has zero CUDA kernel time` as fatal
rather than expected. See that module's own docstring and
`docs/tasks/track-b-step35-deepseek-kernel-only-report.md` in `dc-sim` for
the full diagnosis and the collection record.

See `docs/tasks/track-b-step46-extend-collect-report.md` in `dc-sim` for
the Step 46 collection record (wall-clock, compatibility fixes active,
coverage verification).
