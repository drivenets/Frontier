# Run 2026-09-17 08:46 — linear ops, validation grid, two-column timing (legacy + GPU-bound)

Job 21389, `amd-mi355x-7`, 55 s. Image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`. Code: `5f4d244` + the two-column timing change
(implement-loop record `amd-playground/.claude/debug-reports/implement_linear_op_two_column_timing_2026-09-17.md`; handoff in
`debug_linear_op_host_bound_timing_2026-09-16.md`). Collected with `STAGE=linear_op LINEAR_TOKENS_PY="[1,8,64,3072,4096,4192,6144,8192]"`.

## What was run
| Shapes | Reps |
|---|---|
| 8 token counts × TP {1,2,4,8} = 32 rows | per shape **two passes**, each 3 + 50 runs: legacy → `time_stats_hostbound.*`, GPU-bound → `time_stats.*` |

## What is different and why
The legacy CUDA-event loop measures the host's launch span, not the kernel, whenever the GPU is idle when a scope starts
(host ≈0.45–0.68 ms/forward vs GPU 0.07–0.47 ms at TP>1 below ~6k tokens; `07_post_proj_rope_dip_root_cause.md`, validated in
`08_measurement_validation_preregistration.md`). The GPU-bound pass enqueues a spin (4× the legacy loop wall, event-measured,
calibration warmed by two 20M-cycle spins and re-fitted from every measured spin) before the timed forwards so every event pair
measures device time. New columns: `time_stats_hostbound.<op>.*`, `time_stats.forward_gpu_span.*` (one pair around each whole
forward), `host_wall_per_forward_ms` (legacy), `host_wall_per_forward_ms_backlog`, `gpu_backlog_ms` (requested), `gpu_backlog_ms_actual`
(measured), `legacy_host_bound_ratio.<op>` = legacy/GPU-bound median, `legacy_host_bound.<op>` = ratio > 1.15 (marks the LEGACY
column as host-bound; informational). `emb`/layernorm columns are NaN at TP>1 in all per-op fields (replicated ops live on TP=1 rows).

## Results
- `test_measurement_validity.py --expect green`: PASS — T1′ legacy/GPU-bound at 8/64 tokens 2.95–5.3 (≥ 2.0), max GPU-bound/legacy
  1.005 (≤ 1.05), T2 4.56/2.97/2.02 (≥ 2.0); legacy T1/T2/T3 fail on `time_stats_hostbound` (host-bound signature preserved).
- `sanity_check.py <dir> --tokens-grid "[1,8,64,3072,4096,4192,6144,8192]"`: PASS — backlog coverage min 3.99× (gate 3×),
  GPU-bound ≤ 1.05× legacy on all GEMM scopes; legacy column flagged host-bound on 25/32 `attn_post_proj`, 25 `attn_pre_proj`,
  22 `attn_rope` rows (all rows below ~6k tokens at TP>1 plus the 1–64-token rows at TP1).
- GPU-bound `attn_post_proj`, TP2 (ms): 0.0061 / 0.0107 / 0.0120 / 0.0306 / 0.0357 / 0.0383 / 0.0515 / 0.0584 at 1…8192 tokens —
  within +0.3…+4 µs of the rocprofv3 kernel durations of job 21376 (4.0 / 9.2 / 10.2 / 28.0 / 34.4 / 35.6 / 45.5 / 55.9 µs).

## The two columns differ by clock state as well as by queue state — read this before diffing them
The GPU-bound pass runs after an 80–120 ms device spin, the legacy pass on a device that has been mostly idle. The spin is a clock probe:
`torch.cuda._sleep` counts shader cycles, and the event-measured length of a fixed 20M-cycle spin gives **1.93–2.08 M cycles/ms when
started from idle (jobs 21356/21378/21385, 96 worker processes) and 2.38–2.40 M cycles/ms for a second spin after the first (job 21389)**,
i.e. roughly 2.0 GHz versus 2.4 GHz SCLK. So `time_stats.*` (GPU-bound) was measured at a boosted, steady clock and
`time_stats_hostbound.*` (legacy) at whatever clock a starved device sits at. The legacy/GPU-bound ratio therefore mixes two effects:
the instrument change (event pairs no longer span host idle) and the clock change. Where the legacy loop was already GPU-bound (TP1 ≥ 3072
tokens, TP2 8192) the two columns agree within 1–2 %, which bounds the pure-clock share of the large-GEMM numbers; for kernels under ~20 µs
the traced no-knob/knob duration ratio is 1.05–1.11 (o_proj at TP4/TP8) and up to 1.4 (MLP GEMMs ≤ 11 µs), part of which is clock. No
per-shape clock was logged in this run; a concurrent SCLK estimate per pass is a prerequisite for the next collection.

## Limitations / findings recorded
- **F6 whole-forward closure** (`forward_gpu_span×50 / (backlog wall×50 − spin)`): 0.985–0.991 at ≥ 3072 tokens, 0.965/0.971 at
  1/64 tokens, **0.74–0.79 at 8 tokens** (unexplained; also seen in job 21376 and 21385). 23/32 rows within ±3 %. The per-op
  numbers do not depend on it; it is a measurement, tracked by `test_forward_span_closure_is_recorded`.
- `attn_rope` is the torch fallback (`FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1`): 13 kernels, rotates only the first 64
  columns of the flattened q/k (numerically wrong) — its numbers do not represent vLLM's RoPE kernel in either column.
- 1-token rows take the `wvSplitK` skinny-GEMM path and have a lower host floor (0.020 ms legacy); the 1-token GEMM is fixed-cost bound.
- Deviations from the handoff recorded: calibration is re-fitted from every measured spin (handoff: "calibrate once per process");
  `--allow-legacy-schema` added to the checker so pre-2026-09-17 run folders remain checkable; the second pass runs for
  `--profile_method cuda_event` only.
- Only `--profile_method cuda_event` produces the second pass; kineto/perf_counter/record_function outputs have no
  `time_stats_hostbound.*`/`legacy_host_bound*` columns (they are dropped, not empty) and are checked with `--allow-legacy-schema`.
- This is the fixed profiler's validation run. The canonical dense sweep (`runs/2026-09-15_1308_linear_op_dense3327/`) is
  unchanged and single-column; re-collecting it with two columns is a separate task (≈26 min of one node).
