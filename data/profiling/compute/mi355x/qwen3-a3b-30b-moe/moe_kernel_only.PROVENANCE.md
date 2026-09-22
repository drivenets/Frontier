# `moe_kernel_only.csv` provenance

Frontier's `moe_kernel_only_input_file` config field hard-codes the filename
`moe_kernel_only.csv` (templated only by device/model directory) — there is
no way to point it at a differently-named file. So exactly one file at this
path is ever "live"; every other version must live under a different name.

| file | routing_runtime_path | status | collected |
|---|---|---|---|
| `moe_kernel_only.ep1_2.csv` | `standard_fused_topk` | archived | Track B Step 20 (real hardware, `xai-3`/`amd-mi355x-3`, GPU 0) — `ep∈{1,2}`, 192 rows |
| `moe_kernel_only.csv` | `standard_fused_topk` | **live** | above + Track B Step 46 (`ep∈{4,8}` added, real hardware, `xai-5`/`amd-mi355x-5`, GPUs 0-7) — 384 rows |
| `moe_kernel_only.uniform_topk.ep1_2.csv` | `uniform_topk` | archived | Track B Step 17 (real hardware, `xai-3`/`amd-mi355x-3`, GPU 0) — `ep∈{1,2}`, 192 rows |
| `moe_kernel_only.uniform_topk.csv` | `uniform_topk` | **live** | above + Track B Step 46 (`ep∈{4,8}` added) — 384 rows |

Both live files: 384 rows each (`ep∈{1,2,4,8}`, `num_tokens∈{1..32}`,
3 load-distribution samples), same model/device/profiler invocation shape.
**Unlike `deepseek-v3`, both `ep=4` and `ep=8` were new for this model** —
`qwen3-a3b-30b-moe` never went through a Step 37/38-style memory-feasibility
widening that reached `ep=4` early, so its own `moe_kernel_only` files
capped at `ep=2` until Step 46 (Step 45's own finding, confirmed directly
against the CSV before collecting).

Step 17 collected under `uniform_topk` because that matched
`tools/planner.py`'s then-current hard-coded default. Track B Step 19
corrected that default to Frontier's own `"simulation"` (`standard_fused_topk`
at the runtime-path level), which left the Step 17 file serving the wrong
mode — `PREFILL`'s eager `moe.csv` has only `standard_fused_topk` rows, so
under the corrected default `DECODE_FFN`'s kernel-only file must match it.
Step 20 recollected under `standard_fused_topk` and installed it as the live
file, archiving the original rather than discarding it: it is what a
`moe_routing_mode="uniform_random"` evaluation of this model would still
need if anything ever requires that mode again for `qwen3-a3b-30b-moe`.

See `AGENTS.md`'s "MoE routing mode" trap in `dc-sim` and
`docs/tasks/track-b-step{17,19,20}-*-report.md` there for the full history.

See `docs/tasks/track-b-step46-extend-collect-report.md` in `dc-sim` for
the Step 46 collection record.
