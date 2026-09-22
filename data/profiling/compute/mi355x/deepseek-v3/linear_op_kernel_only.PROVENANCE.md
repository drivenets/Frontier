# `linear_op_kernel_only.csv` provenance

Frontier's `linear_op_kernel_only_input_file` config field hard-codes the
filename `linear_op_kernel_only.csv` (templated only by device/model
directory) — same constraint this model's `linear_op.PROVENANCE.md`
documents for the eager file. This file did not exist for `deepseek-v3`
before this task; there is nothing archived to preserve.

| file | status | collected |
|---|---|---|
| `linear_op_kernel_only.tp1_2.csv` | archived | Track B Step 35 (real hardware, `xai-3`/`amd-mi355x-3`, GPU index 1) — `tp∈{1,2}`, 64 rows |
| `linear_op_kernel_only.csv` | **live** | Track B Step 35 (`tp∈{1,2}`) + Track B Step 46 (`tp∈{4,8}` added, real hardware, `xai-4`/`amd-mi355x-4`) |

128 rows total: `tp∈{1,2,4,8}`, `num_tokens∈{1..32}`, no `--is_moe` (matching this
model's own `linear_op.csv` convention — `deepseek-v3` declares
`first_k_dense_replace=3`, so the dense-MLP ops (`mlp_up_proj`/`mlp_act`/
`mlp_down_proj`) are real for this model's first three layers and belong in
this file, per `linear_op.PROVENANCE.md`'s own finding). `--profile_method
record_function` (alias `kernel_only`), `--use_fp8 --block_shape 128 128`.

Step 46's `tp=4`/`tp=8` collection calls each echoed a second, sparser
`num_tensor_parallel_workers=1` row per `num_tokens` (same TP-invariant
"replicated op" behavior as Step 35's own `tp=2` call) — discarded, keeping
only the genuine `tp=4`/`tp=8` rows; the pre-existing `tp=1` rows (from
Step 35's own `tp=1` call) are untouched.

**One merge step was needed, not a straight concatenation.** A single
collection call at `tp=2` (or any `tp>1`) also emits, per `num_tokens`, a
second row tagged `num_tensor_parallel_workers=1` carrying only the
TP-invariant "replicated" op columns (`emb`/`input_layernorm`/
`post_attention_layernorm`) with the TP-sharded columns
(`attn_pre_proj`/`attn_rope`/`attn_post_proj`/`mlp_*`) left blank — this is
the tool's own behavior for `--disable_replicated`-eligible ops, not a
recollection artifact. Concatenating the separate `tp=1` and `tp=2`
collection outputs naively would have kept two same-keyed `tp=1` rows per
`num_tokens`: the genuine, fully-populated one from the `tp=1` collection
call, and the sparser, replicated-only echo emitted incidentally by the
`tp=2` call. The final file keeps the `tp=1` collection's own rows
wholesale and only the genuine `num_tensor_parallel_workers=2` rows from
the `tp=2` call — the echoed, sparser `tp=1` rows from that second call
were discarded as strictly redundant.

See `docs/tasks/track-b-step35-deepseek-kernel-only-report.md` in `dc-sim`
for the full collection record.

See `docs/tasks/track-b-step46-extend-collect-report.md` in `dc-sim` for
the Step 46 collection record.
