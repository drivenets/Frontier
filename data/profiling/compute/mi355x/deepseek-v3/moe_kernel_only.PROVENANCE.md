# `moe_kernel_only.csv` provenance

Frontier's `moe_kernel_only_input_file` config field hard-codes the filename
`moe_kernel_only.csv` (templated only by device/model directory) — same
constraint `moe_kernel_only.PROVENANCE.md` documents for
`qwen3-a3b-30b-moe`. This file did not exist for `deepseek-v3` before this
task; there is nothing archived to preserve.

| file | routing_runtime_path | gating_runtime_context | status | collected |
|---|---|---|---|---|
| `moe_kernel_only.ep1_2_4.csv` | `standard_fused_topk` | `standalone_legacy` | archived | Track B Step 35 (`ep∈{1,2}`) + Track B Step 37 (`ep=4`), real hardware, `xai-3`/`amd-mi355x-3`, GPU index 1 — 288 rows |
| `moe_kernel_only.csv` | `standard_fused_topk` | `standalone_legacy` | **live** | above + Track B Step 46 (`ep=8` added, real hardware, `xai-4`/`amd-mi355x-4`, GPUs 0-7) |

384 rows total: `ep∈{1,2,4,8}`, `num_tokens∈{1..32}`, 3 load-distribution samples
(`uniform`), `--num_tensor_parallel_workers 1`, `--profile_method
record_function` (alias `kernel_only`), `--use_fp8 --block_shape 128 128`.
**Only `ep=8` was collected by Step 46** — `ep=4` already existed from Step
37 (Step 45's own correction to a draft table that assumed `ep∈{4,8}` both
needed collecting; confirmed directly against the CSV before this
collection, per Step 46's own §2/§10 "known trap").

**`ep=4` added by Step 37, merged in rather than recollected from scratch**:
`ParamCounter.get_num_parameters_per_device()`, run directly against the real
`deepseek-v3.json` model config, found the frozen Track B Step 36 prediction's
own FFN shapes (`ep=2` for Arm A, `ep=1` for Arm B′) do not fit this model's
real FP8 weight footprint in a single `mi355x` GPU's 288 GiB HBM (`ep=2`:
307.10 GiB/device; `ep=1`: 611.60 GiB/device — both over capacity even before
KV cache/activations). `ep=4` (154.85 GiB/device, confirmed by the same
counter) is the smallest divisor of `n_routed_experts=256` that fits with
real headroom, so both arms' real-hardware topology in Step 37 needed
widening to `ep=4`, and this file needed the matching coverage before any
new simulated prediction could be produced at that shape. Column-merged by
name (not position) into the existing `ep∈{1,2}` file, no rows overwritten.

`routing_runtime_path`/`gating_runtime_context` match this model's existing
`moe.csv` exactly (Track B Step 20's correction of the routing-mode default,
applied here from the start rather than needing a later fix) — `PREFILL`'s
eager `moe.csv` has only `standard_fused_topk` rows, so `DECODE_FFN`'s
kernel-only file must match it for the same reason `qwen3-a3b-30b-moe`'s own
file does.

See `docs/tasks/track-b-step35-deepseek-kernel-only-report.md` and
`docs/tasks/track-b-step37-deepseek-hardware-report.md` in `dc-sim` for
the full collection record.

See `docs/tasks/track-b-step46-extend-collect-report.md` in `dc-sim` for
the Step 46 collection record.
