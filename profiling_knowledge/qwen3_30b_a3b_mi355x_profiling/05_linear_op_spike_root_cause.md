# Root cause of the ~200 ms `attn_pre_proj` stall at timed run 11 (answer to `04_linear_op_spike_specialist_brief.md`)

Investigation date: 2026-09-16. Node `amd-mi355x-8`, same image/command/code as jobs 21313/21334, plus an opt-in
diagnostic module described in §2. Evidence labels: **CONFIRMED** (direct evidence, cited), **RULED OUT** (direct
evidence against), **SPECULATIVE** (plausible, not tied to the effect by evidence).

## 1. Executive summary

**CONFIRMED:** the stall is a CPython generation-2 (full) garbage collection running on the worker's main thread
between the `attn_pre_proj` start event and the QKV GEMM launch. Across 168 instrumented worker processes in
six jobs, 184 gen-2 collections were recorded; 176 started inside forward 14 (timed run 11) of a task whose
`attn_pre_proj` run 11 spiked, with the collection's duration matching the CUDA-event gap to within 1 ms, and the
other 8 landed in untimed code of forward 36 and produced no spike. No 100 ms-class sample occurred without a
coincident gen-2 collection in any run. It is a pure host-side stall: the host wall time of that one forward call equals the
event gap (§3.2). Disabling or freezing the interpreter's GC removes or shrinks it (§3.5, §3.6).
**CONFIRMED:** the position is set by CPython's GC bookkeeping (tracked-container allocation counts and the
`long_lived_pending ≥ long_lived_total/4` rule), not by token count, TP, wall-clock, allocator state, hipBLASLt or the
HIP runtime; shifting the token list moves the spike to the same per-worker task index (§3.3), and the 8-process
contention explanation for the 150–290 ms magnitude is ruled out by a 2-worker run with identical durations (§3.4).
**SPECULATIVE:** the 146–290 ms cost is the traversal of the worker's ~0.5–1 M tracked objects (torch + vLLM import
graph); the spread between workers is CPU-side (cache/NUMA/frequency), not GPU-side (§4).

## 2. What was checked and how

Code read (Frontier fork `smatar/qwen3-30b-mi355-profiling`, `7f4dfa5`): `frontier/profiling/linear_op/main.py`
(`_worker_init`, `_worker_profile_linear_op_task`, TP loop and `ProcessPoolExecutor` dispatch, lines 151–228 and
561–735), `linear_op_wrapper.py` (`LinearOpWrapper.__init__/profile`), `linear_op_impl.py` (Qwen3 attention scope,
lines 427–545), `common/cuda_timer.py`, `common/timer_stats_store.py`, `utils/__init__.py::get_num_tokens_to_profile`,
`utils/singleton.py`. CPython 3.10 `Modules/gcmodule.c` (`gc_collect_generations`, `gc_collect_main`,
`gc_freeze_impl`) for the trigger rule quoted in §3.3.

Data re-checked independently: both original CSVs (`runs/2026-09-15_1308_linear_op_dense3327`,
`runs/2026-09-16_0629_linear_op_dense3327_rerun`) with the brief's own extraction; the 32-row band at tokens
1968–1975 / run 11 is reproduced in both. Two extra rows per run also exceed 20× the row median (§5.1).

Instrumentation added (opt-in, inert unless `FRONTIER_SPIKE_DIAG` is set; new file
`frontier/profiling/linear_op/spike_diag.py`, 3 call sites in `main.py`, 4 in `linear_op_wrapper.py`, one
`${LINEAR_DOCKER_ENV}`/`${LINEAR_LOG_SUFFIX}` passthrough in the sbatch; uncommitted). Per worker process it records:

- every GC collection via `gc.callbacks` (generation, start/stop `perf_counter`, objects collected);
- the host-side wall time of each of the 53 `self.model(...)` calls;
- per task: index, tokens, TP, `gc.get_count()` at begin/end, all per-op sample lists.

Recording inside the timed loop uses `array.array('d')` appends only (no GC-tracked allocations), so it cannot move the
GC schedule it observes inside a task; the per-task JSON write does add a few tracked allocations, which shifts the
spike to an earlier task (§3.3, "why 145 and not 169").

Jobs run on `amd-mi355x-8` (all full 3,327-token grid × TP {1,2,4,8} unless stated; ≈2–3 min each):

| job | id | change vs. original | result (attn_pre_proj rows >20× row median) |
|---|---|---|---|
| E1 | 21342 | diag on, plus a `gc.collect()` at worker init | **no** 100 ms-class spike anywhere; 0 gen-2 collections in 416 tasks (all 32 workers) |
| E1b | 21346 | diag on | 32 rows, tokens **2944–3000**, run 11, 146–277 ms; one gen-2 per worker, inside forward 14 |
| E2 | 21343 | diag on, `gc.collect()` + `gc.freeze()` at init | 32 rows at task 8 (tokens 15360…), run 11, 52–71 ms; one gen-2 per worker, inside forward 14 |
| E2b | 21347 | diag on, `gc.freeze()` at init | 32 rows, tokens 1616–1623 (task 213), run 11, 61–124 ms; one gen-2 per worker, inside forward 14 |
| E3 | 21344 | diag on + init collect, token list 1…2048 (brief's exp. A) | no spike, 0 gen-2 (same as E1) |
| E3b | 21348 | diag on, token list 1…2048 | 32 rows, tokens **881–888** (= per-worker task 145 of that list), run 11, 144–273 ms |
| E4b | 21349 | diag on, `LINEAR_NUM_GPUS=2` (brief's exp. C variant) | tokens **11744/11728** (= per-worker task 145), run 11, 148–210 ms; further gen-2 at tasks 1021 and 1508 |
| E6 | 21350 | diag module present but inert (baseline) | 32 rows, tokens **1968–1975**, run 11, 144–259 ms — the original position, third reproduction; 8 sporadic rows ≤2.1 ms |
| E5 | 21351 | `gc.disable()` at worker init only, no recording | **no spike**: max `attn_pre_proj` timed sample in 13,308 rows = 2.1 ms; band 1968–1975 run 11 max 0.135 ms |
| E7 | 21352 | diag on (replication of E1b) + heap size at task 0 | 32 rows, tokens 2944–3000, run 11, 149–271 ms; one gen-2 per worker inside forward 14; heap at task 0 = 480,215 tracked objects |
| E8 | 21353 | diag on, 200 timed repetitions per task (`FRONTIER_LINEAR_ACTIVE_STEPS=200`) | 64 rows: tokens 2112–2168 (task 158) and 912–919 (task 301), all run 11, 145–275 ms; 9 gen-2 per worker at runs 11/55/77/99/165/187, only the run-11 ones inside a timed scope (§3.9) |

Slurm outputs: `data/profiling/sweep_work/logs/spike-e*-<id>.out`; profiler logs `linear_op_e*.log`; diagnostics
`data/profiling/sweep_work/logs/spike_diag/<e>/worker_gpu<k>_pid<pid>.jsonl`; CSVs `data/profiling_spike_<e>/…/linear_op.csv`
(all under `/opt/shared/frontier-qwen3-profiling/Frontier`).

## 3. Findings

### 3.1 CONFIRMED — the stall is a host-side gap, not GPU execution

In E1b every spiking forward's host wall time (from before `self.model(...)` to after it returns) equals the
`attn_pre_proj` event gap plus ~1 ms, while the median host time per forward is 0.5–0.6 ms. Per worker (E1b,
`xworker.py` over the diagnostics; ms):

| TP | GPU | task | tokens | gen-2 duration | attn_pre_proj run 11 | host time of that forward |
|---|---|---|---|---|---|---|
| 1 | 0 | 145 | 3000 | 202.0 | 202.1 | 203.5 |
| 1 | 3 | 145 | 2976 | 165.9 | 166.6 | 167.4 |
| 1 | 6 | 145 | 2952 | 258.1 | 258.8 | 259.9 |
| 2 | 0 | 145 | 3000 | 251.8 | 251.9 | 253.3 |
| 2 | 3 | 145 | 2976 | 147.3 | 147.9 | 148.9 |
| 4 | 0 | 145 | 3000 | 276.0 | 276.7 | 277.6 |
| 4 | 2 | 145 | 2984 | 148.5 | 149.1 | 149.9 |
| 8 | 6 | 145 | 2952 | 252.2 | 252.8 | 253.8 |
| 8 | 7 | 145 | 2944 | 146.5 | 147.1 | 148.1 |

(all 32 rows have the same shape; full table in the E1b diagnostics). The GPU was idle during the gap; the mild
elevation of the following ops in that forward reported in the brief is the usual post-idle effect.

### 3.2 CONFIRMED — the gap is a CPython generation-2 collection

Evidence: in E1b, E3b and E4b each worker process recorded exactly one gen-2 collection in the 416-task pass (three in
the 1664-task E4b pass), and in every case where the collection landed in a timed scope (E1b 32, E2 32, E2b 32, E3b 32, E7 32, E4b 16 of 24
events = 176) the collection's start timestamp falls inside the host window of forward index 14 (= timed run 11) of
the spiking task, its duration matches the event gap (table above), and `collected` = 133 objects (20 after
`gc.freeze()`, 0 for the later E4b collections). The 8 remaining E4b events (task 1021) started in forward 36 in code
outside every `CudaTimer` scope; that task's samples are all normal, which is the expected behaviour when the stall
falls between scopes. Example record (E1b, GPU 0, TP 1, task 145):

```
gc_events: {"gen": 2, "start_ms": 8.84, "stop_ms": 210.81, "dur_ms": 201.97, "collected": 133}
forwards_ms[14]: [8.56, 212.06]          # host window of forward 14, ms since task begin
attn_pre_proj samples[14]: 202.1 ms      # the spike
```

Every instrumented run that changed the GC state changed the spike accordingly (§3.5, §3.6), and no run showed a
100 ms-class `attn_pre_proj` sample without a coincident gen-2 collection.

### 3.3 CONFIRMED — the position is a GC-schedule position, not a token, TP, time or allocator position

CPython 3.10 `gc_collect_generations` (quoted from `Modules/gcmodule.c`):

```c
for (int i = NUM_GENERATIONS-1; i >= 0; i--) {
    if (gcstate->generations[i].count > gcstate->generations[i].threshold) {
        if (i == NUM_GENERATIONS - 1
            && gcstate->long_lived_pending < gcstate->long_lived_total / 4)
            continue;
        n = gc_collect_with_callback(tstate, i);
        break;
    }
}
```

`long_lived_pending` grows by the survivors of each gen-1 collection; `long_lived_total` is reset to the heap size at
each full collection. Thresholds are (700, 10, 10) in the workers (recorded). What the diagnostics show:

- The gen-0 counter is a *net* count (it is decremented when tracked objects are freed). Each task starts with it at
  1 and ends at 6 (E1/E1b, every task), because dropping the previous wrapper frees the model's objects. Within a task
  the counter then grows deterministically: ≈255 net objects for model build + setup, then ≈32 per forward, so it
  crosses the threshold of 700 at **forward 14** and again 22 forwards later at **forward 36** (E1 GPU 0: 345 and 414
  of the gen-0 events after task 0; 1 each at forwards 13 and 35). Gen-1 events (every 10th gen-0) therefore land at
  forward 14 or 36, and the gen-2 event fires at the first gen-1 opportunity after
  `long_lived_pending ≥ long_lived_total/4` — at forward 14 in 176 of 184 observed cases and at forward 36 in the
  other 8. This is why it is always timed run 11. With more repetitions per task the crossings continue every ~22
  forwards (runs 11, 33, 55, 77, …), so the spike would land on one of those runs, not on an arbitrary one. This is why it is always timed run 11.
- The per-worker task index, not the token count, selects the task: E1b (full grid) → task 145 = tokens 2944–3000;
  E3b (list 1…2048) → task 145 = tokens 881–888; E4b (2 workers) → task 145 = tokens 11744/11728. All three were
  predicted from the list order before the jobs ran (`get_num_tokens_to_profile` sorts descending; task `idx` → worker
  `idx % n_gpus`).
- Same task at all four TP passes (fresh processes replay the same allocation sequence), as in the original runs.
- Why task 145 in the instrumented runs and 169 in the originals: the instrumentation adds ≈50–100 tracked
  allocations per task (per-task JSON record, `gc.get_count()` tuples, two callback dicts per collection). That
  advances the gen-0/gen-1 cadence and hence `long_lived_pending`, so the 25 % rule is met ~24 tasks earlier. The
  position is deterministic for a fixed code path and moves whenever the allocation pattern changes — exactly the
  brief's "history-dependent in-process event". E6 (module present, recording off) is the check that the
  un-instrumented position 1968–1975 is unchanged: it reproduced the original 32 rows exactly (run 11, 144–259 ms; table below), so the hooks themselves do not move it.

  E6 `attn_pre_proj` run 11 (ms), module present, recording off:

  | tokens | TP1 | TP2 | TP4 | TP8 |
  |---|---|---|---|---|
  | 1968 | 258 | 253 | 144 | 150 |
  | 1969 | 193 | 155 | 163 | 144 |
  | 1970 | 144 | 154 | 152 | 161 |
  | 1971 | 192 | 144 | 149 | 237 |
  | 1972 | 193 | 251 | 165 | 146 |
  | 1973 | 159 | 216 | 155 | 252 |
  | 1974 | 165 | 198 | 202 | 205 |
  | 1975 | 259 | 199 | 151 | 148 |
- Allocator: `torch.cuda.memory_stats()` read after each spiking task shows `num_alloc_retries = 0`,
  `num_device_free = 0`, `num_device_alloc = 12`, `segment.all.current = 12` (E1b, all TPs) — the caching allocator had
  made its 12 `hipMalloc`s early in the pass and never freed or retried. **RULED OUT** as the trigger.

### 3.4 RULED OUT — 8-process contention as the cause of the 150–290 ms magnitude

E4b ran 2 workers instead of 8: gen-2 durations 148–210 ms at task 145, and 146–258 ms at tasks 1021 and 1508 (when
`collected` = 0). Same range as with 8 workers (E1b 146–277 ms, E3b 144–273 ms). In addition, in the 8-worker runs the
workers' gen-2 collections were spread over 0.4–1.7 s within a TP pass (wall-clock start times in the diagnostics),
i.e. mostly not overlapping, and the longest collections were not the overlapping ones. The duration is intrinsic to
one process's full collection.

### 3.5 CONFIRMED — interventions on the interpreter's GC move or remove the stall

- E1/E3: one `gc.collect()` at worker init (before the lazy vLLM import) resets `long_lived_total` to the whole heap
  and `long_lived_pending` to 0; no gen-2 collection then occurs in the entire pass (gen-2 counter climbs 6 → 75
  without reset) and no 100 ms-class sample exists anywhere in 13,308 rows (max `attn_pre_proj` timed sample 2.39 ms).
- E2/E2b: `gc.freeze()` at init (233k objects moved to the permanent generation) — the gen-2 still fires once, still
  inside forward 14, but costs 52–124 ms instead of 146–277 ms because the frozen objects are not traversed.
- E5: `gc.disable()` at init with no recording at all (the only difference from E6): the band is gone (run 11 at
  tokens 1968–1975 ≤0.135 ms) and the largest `attn_pre_proj` timed sample in the whole file is 2.1 ms, i.e. only the
  sporadic class remains. E6 → E5 is the clean intervention on the un-instrumented code path.

### 3.6 CONFIRMED — the same mechanism leaves a small, systematic fingerprint in the original data (brief Q4)

Gen-0 collections cost 0.02 ms (p50; p99 0.04, max 0.16) and gen-1 0.03 ms (p50; max 0.18) in E1b. Because they land
in forwards 14 and 36 of almost every task, timed runs 11 and 33 are slightly high in *every* row. In the two original
(un-instrumented) CSVs the median over rows of `sample/row-median` is 1.31 and 1.32 for run 11 and 1.06 and 1.04 for
run 33, against 0.998 for the other runs; run 11 is the row maximum in 13 % / 23 % of rows and run 33 in 20 % / 14 %,
versus 1.4 % for a typical run. Medians are unaffected (2 of 50 samples). Runs 49 (1.36–1.41×), 0 (1.14–1.18×) and 16
(1.08–1.10×) are also systematically high; those are not GC (no collections land there) and are outside this brief's
scope — **SPECULATIVE:** run 0 follows the post-warm-up `synchronize()`/`mark_warmup_end()` host gap.

### 3.7 RULED OUT — the other candidates in the brief §8

- PyTorch caching allocator (§8.1): see 3.3, allocator counters flat across the spike; also the stall is on the
  Python side of the launch, not in a HIP call.
- hipBLASLt lazy work (§8.2) and HIP/ROCr housekeeping (§8.3): the gap is fully accounted for by the gen-2
  collection (durations match within 1 ms in 88/88 cases), and E1/E5 remove the gap without touching the GPU stack.
  No separate HIP-side event is needed; none was observed.
- Environment / node / shape: already excluded by the brief; consistent with all runs here.

### 3.8 The 2 ms sporadic class is different (complication, §5.1)

The four extra >20× rows per original run and the sporadic >20× rows in `attn_rope`/`attn_post_proj` (all ≤2.2 ms,
random run index, not reproducible between runs) are also host-side (host time 2.4–2.7 ms vs 0.6 ms median in E1)
but have **no** GC event overlapping them. **SPECULATIVE:** ordinary host jitter (scheduler/NFS/IPC). Not the same
phenomenon.

### 3.9 CONFIRMED — prediction test with 200 timed repetitions per task (E8, job 21353)

Prediction made before the run: gen-0 crossings continue every ~22 forwards, so collections can only land at timed
runs 11, 33, 55, 77, … and the gen-2 will hit one of those, earlier in the pass and more than once. Observed, on all
32 workers identically: gen-0 collections at forwards 14, 36, 58, 80, 102, 124, 146, 168, 190 only (plus one straggler
each at the preceding forward); nine gen-2 collections per worker at tasks 12, 69, 117, 158, 215, 262, 301, 358, 405,
landing at timed runs 77, 165, 187, 11, 99, 55, 11, 99, 55; every one stalls the host 145–275 ms (host time of that
forward call = collection duration + ~1 ms). Only the two at run 11 (tasks 158 → tokens 2112–2168, 301 → tokens
912–919) show up in the CSV, as 64 `attn_pre_proj` rows. The other seven per worker fell in code between the
`CudaTimer` scopes of their forward (the largest scope sample in those forwards is 0.1–0.6 ms), so they are invisible in
every timed column even though the GPU idled for ~200 ms. Reason: the crossing point drifts 4 allocations earlier per
crossing (700 mod 32 = 28), so only the first crossing of a task (offset ≈ 29 into the forward) lands on the QKV
projection.

## 4. Open points and what would settle them

1. **Why 146–290 ms and why bimodal (~150 / ~250 ms)?** SPECULATIVE: a full collection traverses every tracked
   object; the worker heap after the lazy vLLM/model imports is 480,215 tracked objects (E7, `len(gc.get_objects())` at task 0), against 233k before that import, where a full collection already took 41–78 ms (E1 init measurement). The two modes are likely the CPU
   the worker landed on (NUMA distance to its heap, boost frequency), since 2-worker runs show the same spread. Test:
   pin each worker with `taskset` to a fixed core/NUMA node and see whether each worker's duration becomes stable;
   or `perf stat` the worker around task 145.
2. **Serving relevance (brief Q3).** SPECULATIVE: any long-running CPython process with a large heap and steady
   container allocation will take full collections of this cost periodically; the period here (~1 per 400–900
   tasks in E4b) depends on the allocation pattern. Whether SGLang/vLLM engines already mitigate it
   (`gc.freeze()` after model load, raised thresholds) was not checked in this image and should be, before
   claiming either way.

## 5. Contradictions / complications for the brief

1. The brief's §4 statement "nothing else in the file exceeds 20× its row median except the warm-up-run-0 `emb`
   outlier" is not exact: run 1 has 2 further `attn_pre_proj` rows (tokens 1992, 2004; 2.1 / 1.9 ms), 9 `attn_rope`
   and 8 `attn_post_proj` rows; the re-run has 2 (1921, 3264), 3 and 10. All are ≤2.2 ms and belong to the sporadic
   class (§3.8). The 32-row band is correctly isolated by magnitude (≥146 ms), not by the 20× filter alone.
2. The brief's §7.5 inference "a first-use event would land in warm-up run 0; whatever it is must be counting something
   that crosses a threshold mid-task" is right; the counter is CPython's, not the GPU stack's.
3. The brief's §8.1 "eight processes … serialize on the KFD/driver, which would explain 150–290 ms" is ruled out (§3.4).
4. The brief's §3 note that the CUDA-event gap includes host stalls is the crucial reading: this is a 100 % host stall.
5. `n_tracked_objects_after_imports` in the E1/E1b headers (233k) is measured *before* the lazy vLLM import that the
   first task triggers (the import happens at `main.py:189`, before `task_begin`); the heap the gen-2 traverses is the
   larger post-import heap (E7).

## 6. Recommendation for the dataset (not a fix request in the brief, recorded for completeness)

The 32 affected samples are host artefacts and should be excluded from any max/tail statistic; medians are
unaffected. If future sweeps should be free of it, calling `gc.freeze()` once after the first task's imports, or
`gc.disable()` for the timed loop (cyclic garbage is 133 objects per 416 tasks), removes it; both are one-line,
worker-init changes and were exercised here (E2b, E5).
