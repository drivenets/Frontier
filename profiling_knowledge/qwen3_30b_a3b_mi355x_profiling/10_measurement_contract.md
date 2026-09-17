# Measurement contract for the linear_op dataset (agreed 2026-09-17: option (a))

**Status: AGREED by the dataset owner on 2026-09-17 — option (a)**, kernel-only device time per op at a stated clock condition, plus a
separately measured launch/gap overhead term (§3). The recollection plan is derived from this contract (plan-loop); any change to the
contract reopens the plan.

Purpose: state exactly which two points on the host → queue → device → completion timeline each column subtracts, at which clock
condition, so that a column difference is never mistaken for a physical effect. No recollection starts before this is agreed.

## 1. The timeline of one timed scope
```
host: ...record(start_ev) ─ python/HIP launch work ─ launch(k1) ... launch(kn) ─ record(end_ev) ─ (next scope) ...
queue:      [start_ev pkt][k1 pkt]...[kn pkt][end_ev pkt]     (AQL packets, consumed in order; a full queue blocks the host)
device:  stamp(start_ev) ──idle?── run k1 ... run kn ──── stamp(end_ev)
```
`stamp(x)` happens when the device *reaches* packet x. If the device is idle when `start_ev` is enqueued, `stamp(start_ev)` ≈ host time
of `record(start_ev)`; the interval then contains host launch work. If the device is behind, both stamps are device-side and the interval
is `Σ kernel_i + Σ dispatch gaps + (n+1) record costs (~2.6–3.7 µs each)`.

## 2. What each column is
| column | subtraction | regime it is valid for | clock condition |
|---|---|---|---|
| `time_stats_hostbound.<op>` (legacy) | `stamp(end_ev) − stamp(start_ev)` with the device **not held behind** the host | equals kernel time only where the device happened to be behind (large shapes, TP1 ≥ 3k); elsewhere it is the host's launch span for the scope | uncontrolled: the timed loop starts at ≈0.8 GHz on 32/36 grid rows *even after the 3 warm-up forwards and a synchronize* and ramps to ≈2.4 GHz within the pass (probe columns, job 21400) |
| `time_stats.<op>` (GPU-bound) | same subtraction with the device held behind the host by a spin | `Σ kernels + dispatch gaps + ~3 µs record cost`; valid while the queue holds the remaining packets (≤ 25 forwards, or a re-spin) | busy device, ≈2.4 GHz automatic boost unless locked (probe columns) |
| `record_function` (`KERNEL_ONLY`) | Σ device durations of kernels correlated to the scope | kernel time regardless of queue state; sub-10 µs kernels ±20 % between tracers | tracer-loaded device; clock not yet logged for this method |
| `legacy_host_bound_ratio.<op>` | legacy / GPU-bound | > 1.15 ⇒ the legacy value is a host span | mixes clock trajectory and queue state — never read as a pure instrument delta |
| `forward_gpu_span` | one pair around a whole forward of the GPU-bound pass | device wall of a forward incl. 20 record costs; under-counts idle if the queue fills (≤ 8 tokens, 50 forwards) | as GPU-bound |
| `sclk_mhz_*_start/end` | 2M-cycle spin / elapsed | concurrent shader clock immediately before the timed loop (after warm-up) and immediately after it | — |
| `host_enqueue_per_forward_ms*` | host wall to the last launch call / N | host launch pace; ≈ wall ⇒ host-bound or queue-throttled | — |

## 3. What hetcalc's linear-op term should be — recommendation: (a), with two separately measured terms (AGREED)

**(a)** kernel-only device time per op at a **stated clock condition**, plus a separately measured **launch/gap overhead term**.
**(b)** inclusive per-op latency in a stated regime (host-bound or GPU-bound).

Argument for (a). A per-op sum is only meaningful if every summand is the same kind of quantity. Under (b) the summands change kind with
shape: at 8 tokens the "op latency" is a host span, at 8192 it is a kernel, and a model fitted across shapes learns the crossover of
the harness, not the hardware. (b) also bakes the harness's own launch pattern (one Python `torch` call per op, no CUDA graphs) into
the number; a serving engine's launch pattern differs. Low-batch decode *is* often launch-bound in production — the user's point stands —
but that is exactly why the overhead has to be its own term: the simulator can then combine "device time at this clock" with "launch
cost under the engine's regime" instead of inheriting whichever regime the profiler happened to be in.

If (a):
- **Kernel term.** Primary instrument: kernel-timestamp based (`record_function`/roctracer, §3c of `09_`), cross-checked by GPU-bound
  events. Reference clock condition: **two datasets, not one** — automatic clock with the concurrent probe logged per row (what the
  hardware does when busy), and one **locked-clock** run at a stated MHz (1900 available, the same node type) for a clock-independent
  baseline; the simulator scales between them with the probe value. State per row which one it is.
  *Precondition from job 21400:* three warm-up forwards do not bring the device up from its idle DPM clock (legacy timed loop starts at
  ≈0.8 GHz), so the automatic-clock dataset needs either a clock-ramp warm-up (spin until the probe reads the boost clock, then measure)
  or the GPU-bound pass's property that the preceding pass has already warmed the device; the start probe must read ≥ 2.2 GHz for a row
  to count as "automatic boost", otherwise the row is labelled by its probe value.
- **Overhead term.** Measured, not modelled: per op, `legacy − GPU-bound` at the shapes where the legacy loop is host-bound gives the
  harness's host span per scope (≈0.020 ms M=1 path, ≈0.035 ms GEMM scope, ≈0.09 ms the 13-kernel RoPE chain at TP>1); per kernel
  boundary, the ≈3 µs record/dispatch gap from the GPU-bound-minus-kernel difference; and, separately, the *engine's* launch cost from
  an engine step trace (not this profiler). The simulator decides the regime: `cost = kernel(clock) + overhead(regime)`, with `overhead`
  = 0 for a saturated device, the engine's launch span for launch-bound decode.
- **Scope of validity.** Kernel numbers below ≈ 10 µs carry a ±3–5 µs instrument floor and ±20 % tracer disagreement; mark them.
  `attn_rope` is excluded until the fallback is replaced (`11_attn_rope_task.md`).

Decision requested: (a) as above, or (b) with an explicit regime label per row. No recollection until this is answered.
