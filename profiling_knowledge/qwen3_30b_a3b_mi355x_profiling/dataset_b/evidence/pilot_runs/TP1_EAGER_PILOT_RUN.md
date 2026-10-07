# Dataset B — TP1 eager pilot, real GPU, 2026-10-05

Jobs 21772 → 21782 on `amd-mi355x-3`, approved image `sha256:dece6767...`, model
`/mnt/data/data/hai/qwen-blockwise-fp8-ab-20260810/models/Qwen3-30B-A3B` (verified BF16, §1 of
`evidence/instrumentation_phase_RUN.md`). Recipe: TP1, `--attention-backend aiter`, `SGLANG_USE_AITER=1`,
page size 1, radix cache disabled, chunked-prefill 16384, context 16640, max-running-requests 64, bf16,
eager (`--disable-cuda-graph`), `DEBUG_CLR_MAX_BATCH_SIZE=1000000`. Two requests: a 6-token warmup
(`max_new_tokens=2`) and an 8-token measurement request (`max_new_tokens=4`).

## Headline result

**A real served request through the fully patched SGLang stack produced the exact expected physical-scope
sequence, with zero missing or duplicate scopes, across all 48 layers and both prefill and decode phases.**
From job 21781's complete native trace (`evidence/pilot_runs/b-pilot-21781-partial/.../248_marker_api_trace.csv`):

- 5510 total marker rows: 19 `B|fwd|<n>` forward markers, 5491 `B|scope|...` scope markers.
- 5491 = 19 forwards × (1 `embedding` + 48 layers × 6 per-layer scopes) **exactly** — no gaps, no
  duplicates, confirmed by direct count, not by sampling.
- All 8 physical scopes fired: `embedding`, `input_norm`, `qkv_qk_norm`, `rope_kv`, `attention_prefill`,
  `attention_decode`, `output_projection`, `post_norm` — i.e. both prefill and decode-phase attention
  variants were exercised for real, not just the warmup prefill.
- All 48 layers (0–47) present for every per-layer scope.
- Real generation succeeded: prompt `[9707, 11, 1246, 525, 498, 30, 358, 2776]` ("Hi, how are you? I'm")
  → 4 greedy tokens `[4460, 311, 11625, 419]` = `" trying to solve this"`, with real per-token logprobs,
  `finish_reason: length`. Confirms the patch does not change model behavior in a way that breaks serving
  (not yet a formal numerical-equivalence check against an unpatched baseline — see Limitations).

This is the first real-hardware, full-stack confirmation that every call-site edit in
`sglang_b_markers.patch` is wired to the correct location, fires in the correct order, and survives a
real 48-layer MoE forward under the locked serving configuration.

## Two real bugs found and fixed via actual runs (not caught by any earlier unit/import test)

1. **`self.layer_id` AttributeError** (job 21772): `Qwen3MoeAttention` has no `self.layer_id`; only
   `Qwen3MoeDecoderLayer` does. The attention-side scope markers (`qkv_qk_norm`, `rope_kv`,
   `attention_{prefill,decode}`, `output_projection`) needed `self.attn.layer_id` (RadixAttention stores
   it) instead. My own import-level GPU tests (job `smatar-b-patchimport3`) had called `begin_scope`
   directly with a literal layer number and never actually invoked `forward_prepare_native`/`forward_core`,
   so they could not have caught this — only a real forward pass could. Fixed; re-verified by a clean
   later run reaching layer 47 (see headline result).
2. **Leftover port-30000 container** (jobs 21772→21773): cancelling a Slurm job does not reliably kill a
   `docker run --network=host --rm` container in time; the next job's container can fail to bind the same
   host port. Found a genuinely orphaned container (`docker ps` showed it `Up` after `scancel`) and
   stopped it directly; did not touch any other user's containers on the shared node.
3. **AITER JIT cache was not persisted** (jobs 21772–21779, ~5 extra minutes per attempt): `--rm`
   containers lose AITER's compiled-kernel cache every run. Fixed using the proven pattern from
   `profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh`: host `$HOME/.aiter` bind-mounted
   to `/root/.aiter`, `AITER_CACHE_DIR=/root/.aiter`, with stale `build/*/lock` file cleanup before each
   run (an interrupted compile's lock otherwise blocks the next run forever). `run_b.sbatch` updated
   accordingly. This matters for every future run, not just this pilot.

## Unresolved limitation: no single run yet has both a complete native trace AND complete CPU metadata

This is the concrete blocker before a `check_b.py` structural pass is possible, and the reason the pilot
matrix (B–G) and TP4/sweep have not started.

- **Default rocprofv3 signal handling** (job 21781): rocprofv3 installs its own SIGINT/SIGTERM handler
  per-process. When `run_b.py`'s `stop_owned()` sends SIGINT to the process group after a successful
  measurement, rocprofv3's handler treats it as "finalize and exit" for *each* process independently.
  SGLang's detokenizer subprocess dies first from this, SGLang's own supervisor logs
  `"Subprocess detokenizer (pid=249) crashed with exit code -2. Triggering SIGQUIT for cleanup..."`, and
  the resulting disorderly cascade needs a full SIGTERM/SIGKILL escalation (`forced_server_shutdown=true`,
  `status=failed_trace_drain`). **Result: the native trace for the scheduler process (PID 248) finished
  writing completely before the kill** (confirmed: full `kernel_trace.csv`, `hip_api_trace.csv`,
  `marker_api_trace.csv`, `agent_info.csv`, all internally consistent — this is the trace behind the
  headline result above) **but my own atexit-based CPU metadata flush never ran** (SIGKILL doesn't run
  Python atexit handlers), so `b_metadata/` is empty for that run.
- **`rocprofv3 --disable-signal-handlers`** (job 21782, this session's first attempted fix): did not
  actually stop rocprofv3 intercepting SIGINT for PID 248/249 (same `rocprofv3_error_signal_handler`
  log lines appear), and additionally triggered a *new* crash inside uvicorn's own shutdown path
  (`TypeError: signal handler must be signal.SIG_IGN, signal.SIG_DFL, or a callable object` from
  `uvicorn/server.py:344 capture_signals`) — plausibly because the native handler installed by rocprofv3
  leaves Python's signal table in a state `signal.signal()` cannot interpret. Net effect this run: the
  overall process group *did* exit cleanly (`forced_server_shutdown=false`), my atexit flush *did* run for
  PID 7 (the top-level launcher) — but PID 7 never calls `begin_scope`/`begin_forward` itself, so its
  flushed file is legitimately empty (`{"records": [], ...}`), while PID 248 (the real scheduler) stopped
  partway through finalizing its *own* rocprofv3 output — `248_kernel_trace.csv` is present and, by row
  count (20412 rows, identical to job 21781's), appears complete, but `248_hip_api_trace.csv`,
  `248_marker_api_trace.csv`, and `248_agent_info.csv` never got written, and `248`'s own CPU-metadata
  file never appeared either.

Neither attempt simultaneously produced a complete `{kernel, hip_api, marker, agent_info}` set **and** a
non-empty scheduler-process CPU metadata file for the same process. `assemble_bundle.py` (written this
session, not yet run against real data) needs both to build a bundle `check_b.py` can accept.

### Candidate next actions (smallest first)

1. Revert to **default** rocprofv3 signal handling (job 21781's behavior: complete native trace, forced
   shutdown) and instead make the CPU-metadata flush survive the forced shutdown — e.g. have
   `frontier_b_markers.py` install its *own* SIGINT/SIGTERM handler at import time that synchronously
   calls `flush_after_drain` the instant a signal arrives, before SGLang/uvicorn or rocprofv3 get a chance
   to tear anything down. Risk: a third party in an already-crowded signal-handling situation (uvicorn,
   rocprofv3, now this) could make things worse rather than better; needs to be tried and observed, not
   assumed safe.
2. Check whether SGLang exposes an HTTP-level graceful-shutdown path (not yet checked) that avoids OS
   signals for the measured shutdown entirely, only falling back to signals if that fails.
3. Ask whether rocprofv3 has a documented way to make `--disable-signal-handlers` actually take effect for
   *all* processes in the tree (not just the one it's literally invoked on) — current evidence suggests it
   does not propagate to the PIDs that matter here.

None of these require a protocol decision from the user — they're implementation investigation. Flagging
here because the pilot matrix cannot proceed to a `check_b.py`-validated cell until one of them works.

## What this pilot does and does not establish

**Does establish**, on real GPU hardware, real model, real serving config:
- The complete call-site patch (`sglang_b_markers.patch`, 5 files) is correctly wired and fires in the
  right order for every layer, every scope, both prefill and decode, across multiple real forwards.
- `librocprofiler-sdk-roctx.so` + the CSV marker dialect + the whole eager-attribution chain
  (dispatch → correlation → host launch → marker containment), already validated with a synthetic GEMM
  probe in the instrumentation phase, now also validated against a real 48-layer MoE serving forward.
- The locked TP1 eager serving recipe actually starts, serves, and generates correct-looking tokens.
- AITER JIT compilation completes and its cache now persists node-locally across repeated runs.

**Does not yet establish**:
- A `check_b.py` structural pass (needs the native-trace + CPU-metadata simultaneity problem above
  resolved first).
- Calibrated telemetry (telemetry sampler ran per `run_b.sbatch`'s wiring but was never joined/validated
  against a surviving CPU-metadata file in this session; the calibration-anchor mechanism in
  `marker_runtime.py`/`assemble_bundle.py` is implemented and unit-tested but not yet exercised on real
  data for the same reason).
- Numerical equivalence against an unpatched server (not run this session).
- Any TP4, graph-replay, chunking, mixed-execution, or Kineto-cross-check pilot case (A–G matrix
  items B onward) — blocked behind the same shutdown-reliability gap, since each of those also needs a
  complete bundle to be worth collecting.
- Any part of the D1/D2/D3 sweep.

## Measured timing (for the eventual sweep cost estimate)

- Model load (warm AITER cache, warm page cache from job 21779/21781): ~20s (`Load weight begin` →
  `Load weight end` in `server.log`), KV cache allocation ~0.05s, full engine startup ~22s total.
  **Cold AITER cache (no persisted `/root/.aiter`): add ~4–5 minutes** of JIT compilation (observed
  directly, jobs 21772–21779) — now avoided going forward.
- End-to-end measured request latency (8 prompt tokens, 4 greedy decode steps, TP1, eager): `e2e_latency`
  reported by the server as 0.2506s.
- These numbers are from a single tiny request and are not a throughput or per-scope latency estimate;
  they bound the smallest useful pilot cell's wall-clock cost, not the sweep's.

## Evidence paths

- `evidence/pilot_runs/b-pilot-21781-partial/` — complete native trace (the one behind the headline
  result), lost CPU metadata, `forced_server_shutdown=true`.
- `evidence/pilot_runs/b-pilot-21782/` — clean process-group exit, but incomplete native trace for PID 248
  and empty CPU metadata (only the non-scope-emitting top-level PID flushed).
- `evidence/pilot_runs/tp1_eager_static_manifest.json` — draft static manifest fields (image/source
  hashes, model dims, effective config cross-checked against `server.log`'s own
  `max_total_num_tokens=1816059, chunked_prefill_size=16384, ... context_len=16640` confirmation line);
  `patch_sha256` still needs a final recompute once the signal-handling fix lands (patch will change
  again).
- `assemble_bundle.py` — written and unit-syntax-checked this session; not yet run against real data
  (needs a run with both artifacts present).
- `run_b.sbatch` — now includes the AITER cache fix; the `--disable-signal-handlers` addition to
  `launch_plan_tp1_eager_pilot.json`'s `server_argv` should be reverted or replaced once a better fix is
  chosen (currently still present from the unsuccessful attempt).
