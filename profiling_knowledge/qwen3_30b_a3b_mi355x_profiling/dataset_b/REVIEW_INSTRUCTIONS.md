# Dataset B — review instructions (TP1 eager canary)

This document is for a reviewer who was not in the implementation session.
It explains the goal, what was built, what was tested, and how to read the
code and evidence in this directory. It does not assume prior context.

Commit to review: `237dd3e` ("dataset_b: TP1 eager pilot validated
end-to-end ... + Dataset-A-scope canary check"), on branch
`smatar/qwen3-30b-mi355-profiling`.

---

## 1. The goal

### 1.1 What Dataset A is

Dataset A (`data/profiling/compute/<device>/<model>/{linear_op,attention}.csv`)
measures individual model operators **in isolation**: one op (e.g. RMSNorm,
RoPE, an attention kernel) is run by itself, repeatedly, under controlled
synthetic shapes (batch size, sequence length, etc.), and its wall-clock
duration is recorded as a distribution (min/max/mean/median/std/count per
shape). This is how Frontier's execution-time predictor is currently
trained: it looks up (or regresses from) Dataset A numbers to predict how
long an operator will take.

### 1.2 The open question

Dataset A never runs inside a real server. It never competes with other
requests for the GPU, never sits inside SGLang's real scheduler loop, never
experiences the batching, memory-bandwidth contention, or clock-frequency
behavior that shows up when the model actually serves traffic. So a
fundamental question is open:

> **Does an op's Dataset A timing distribution actually predict what that
> same op costs when it runs as part of a real, full LLM inference request?**

And if the two don't match exactly (they won't, exactly), a follow-up
question:

> **When we use Dataset A as a predictor, which summary statistic should we
> use — mean? median? max? min? something else? Does it depend on the op,
> the phase (prefill vs decode), or the shape?**

Neither question can be answered by looking at Dataset A alone. Answering
it requires a second, independent measurement: the same physical operators,
identified the same way, but measured **while the model is actually
serving**, inside the real SGLang process, under the real scheduler. That
second measurement is **Dataset B**.

### 1.3 What Dataset B is

Dataset B is collected by attaching an AMD ROCm profiler (`rocprofv3`) to a
real, running SGLang server process serving the real Qwen3-30B-A3B model,
and instrumenting the SGLang model code (via a source patch) to push a
named marker (`B|scope|<physical_scope>|<layer>|<variant>`) around each
occurrence of each operator we care about, for every real forward pass. The
profiler's raw GPU kernel trace is then matched against those markers, so
every GPU kernel dispatch gets attributed to the specific operator
invocation that caused it. The result is a per-kernel, per-operator-
invocation duration measurement taken from inside a real request's
execution — directly comparable, op-for-op, to the matching Dataset A rows.

`B_SCHEMA.md` is the authoritative schema document for the Dataset B record
format (scopes, forwards, kernels, clock calibration, etc.). Read it before
the code if you want the full data model; this document instead walks
through what was *built and validated* in this session.

---

## 2. Scope of this canary (what this is **not** yet)

This is explicitly a **first canary**, not the production Dataset B sweep.
Two deliberate scope limits:

1. **Only the ops Dataset A already has data for.** Dataset A has 9 column
   groups: `emb`, `input_layernorm`, `attn_pre_proj`, `attn_rope`,
   `attn_kv_cache_save`, `attn_post_proj`, `post_attention_layernorm`,
   `attn_prefill`, `attn_decode`. `scope_spec.json` maps each of these to a
   Dataset B `physical_scope` name (see `a_scopes` field per rule). Only
   these mapped scopes are in scope right now. Everything else the model
   does on the GPU — MoE expert routing/sorting/GEMM, sampling,
   quantization, scheduler-support kernels — is **deliberately ignored**
   for this canary. Covering those is a separate, already-open decision
   (the operator-boundary investigation, decisions D1–D5), not something
   this session resolved.

2. **This validates the collection mechanism, not the A-vs-B answer.** The
   actual statistical question from §1.2 (does Dataset A predict Dataset B;
   which statistic to use) is **not answered by this work**. This session's
   job was narrower and prior to that: *can we reliably collect Dataset B
   at all, with clean, trustworthy, unambiguous per-op timing data, with no
   contamination from unrelated server activity?* Only once that is true
   does it make sense to run the real sweep and do the A-vs-B statistical
   comparison. That comparison, and the full pilot matrix (short/long
   prompts, chunking, mixed execution, graph replay, TP4, Kineto
   cross-check, output equivalence, instrumentation-perturbation
   sensitivity), are the next steps, gated on this review.

---

## 3. What was done

Starting from an existing (but never fully run end-to-end) Dataset B
collector, this session drove the smallest real case — **TP1, eager mode,
one warmup request + one short measured request, against the real
Qwen3-30B-A3B model on an MI355X node** — all the way through the pipeline
(collect → assemble → normalize → extract → check), across repeated real
cluster runs (Slurm jobs 21791, 21792, ..., 21798), fixing real bugs found
at each stage:

1. **Unexplained server activity.** SGLang's own internal startup
   self-warmup and a hidden 1-token `generate()` call inside `/health`
   both occurred inside every capture window with no independently-tracked
   ground truth, so capture reconciliation started with 11 of 19 observed
   forwards unexplained. Fixed with two launch flags
   (`--skip-server-warmup`, `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`).
   Reconciliation is now 100% clean.
2. **Missing warmup ground truth.** The assembly script only read the
   measurement phase's request log, silently dropping the warmup request's
   independently-tracked input/output tokens.
3. **Wrong decode-count convention.** The correctness check assumed
   `decodes == generated_tokens - 1` (prefill directly yields one token).
   Real SGLang with overlap scheduling produces exactly one decode step per
   generated token. Corrected and confirmed on two independent requests.
4. **Missing clock policy.** The manifest's clock-validity bounds
   (max sample age/gap/uncertainty) were never computed; now derived from
   the run's own telemetry sampler interval and calibration uncertainty.
5. **Two incompatible GPU-identity schemes.** Kernel records used
   rocprofv3's internal `gpuid:N`, telemetry records used a PCI bus id;
   these never matched. Both now derive the same PCI bus id from
   `agent_info.csv`.
6. **Unscoped native trace window.** The raw GPU trace included ~9,000
   kernels from weight loading and first-dispatch JIT compilation, before
   any operator marker existed, which is why clean, scoped collection is
   necessary — see `marker_runtime.py`/`run_b.sbatch`'s `--selected-regions`
   note for the fix.

After these fixes, a clean run (job 21798) reconciles every observed
forward to a known request, and 3,659/3,659 kernels mapped to a
Dataset-A-backed scope resolve cleanly and unambiguously. 2,150 kernels per
run remain unattributed; all of them are confirmed (by name) to be MoE or
other out-of-scope kernels — i.e., exactly the kernels §2 says are
out of scope for now, not a sign of a broken measurement.

A new, narrow **canary check** (`check_dataset_a_scope.py`) was added to
assert exactly that: every kernel inside a Dataset-A-mapped scope is
cleanly resolved, and nothing structurally broken (ambiguous markers, or an
unverifiable graph replay) is hiding anywhere in the bundle, regardless of
scope.

---

## 4. Files to review

### 4.1 Start here (context and schema)

| File | Purpose |
|---|---|
| `REVIEW_INSTRUCTIONS.md` | This document. |
| `B_SCHEMA.md` | The Dataset B record schema: scopes, forwards, kernels, clock calibration, windows. Read this to understand the data model the code below implements. |
| `scope_spec.json` | The provisional physical-scope ↔ Dataset-A-column mapping. Each rule has `physical_scope`, `variant`, `layers`, `phases`, `a_scopes` (the Dataset A column name(s) it corresponds to), and `compatibility` ("provisional" everywhere right now). This is the file that defines what "ops we have Dataset A data for" means, precisely. |

### 4.2 The collector (runs inside the SGLang process)

| File | Purpose |
|---|---|
| `marker_runtime.py` | The core instrumentation module. Defines `begin_scope`/`end_scope` (ROCTx marker push/pop around one operator invocation), `begin_forward`/`end_forward` (marks one model forward pass), `record_scheduler_batch` (independent scheduler-side record of what each forward will process, for later cross-checking), and the explicit two-rank drain protocol (`confirmed_drain`, `emergency_flush`) that closes a capture safely. Also owns the `roctxProfilerResume`/`Pause` calls that bracket rocprofv3's recording window to exactly the measured region (fixed this session — see §3.6). This file becomes `frontier_b_markers.py` inside the patched SGLang image. |
| `sglang_b_markers.patch` | The unified diff adding `marker_runtime.py` as a new SGLang module and wiring its calls into 8 real SGLang source files (scheduler, schedule_batch, io_struct, tokenizer_manager, http_server, the Qwen3/Qwen2 MoE model files, and the communicator layer). This is what actually gets applied to the approved serving image before launch. |
| `evidence/patch_source_hashes.json` | SHA-256 hashes of every original and patched file the patch touches, plus the patch's own hash. `run_b.py`'s preflight step verifies these at container-launch time, so a stale or tampered patch fails loud instead of silently running. |

### 4.3 Launching a pilot run

| File | Purpose |
|---|---|
| `launch_plan_tp1_eager_pilot.json` | The exact, pinned configuration for the TP1 eager pilot: approved image id, server CLI flags (including `--skip-server-warmup`, `--selected-regions`), environment variables, source hashes to verify, and the request manifest to drive. The `notes` field documents *why* each non-obvious flag is there. |
| `requests_tp1_eager_pilot.json` | The two requests this pilot drives: `warmup-0` (absorbs first-dispatch JIT-compile cost) and `r-0` (the actual measured request). |
| `run_b.py` | The driver that runs inside the container: verifies the launch plan (`preflight`), starts the server and telemetry sampler, waits for readiness, drives the warmup then measurement requests, and — critically — performs the **explicit drain protocol**: call `/frontier_b_drain`, verify every rank's acknowledgement (record count, no overflow, matching capture id) *before* sending any termination signal. |
| `run_b.sbatch` | The Slurm batch script: resolves the allocated GPU's PCI bus id for telemetry, handles the shared-account AITER JIT cache directory (only removing a lock file after confirming via `flock -n` that it is not currently held by a live process — this machine's AITER cache is shared with other users' concurrent jobs), and launches the approved container with every patched file bind-mounted at its exact in-image path. |
| `drive_b.py` | The HTTP request driver `run_b.py` invokes once per phase (warmup, measurement); records every request's real input/output tokens and timing as independent ground truth. |
| `telemetry_b.py` | The out-of-process GPU clock (sclk/mclk) sysfs sampler, and `ClockJoin`, which joins a kernel's calibrated host timestamp to the nearest clock sample and reports a `clock_status` (valid, stale, uncertain, etc.). |

### 4.4 The pipeline (run after a pilot, off the cluster)

Run in this order:

| Step | File | Purpose |
|---|---|---|
| 1 | `assemble_bundle.py` | Combines one run's artifacts — the rocprofv3 CSV trace triple, the drained CPU-side metadata, the driver's request logs (now reading *both* the warmup and measurement phase directories — see §3.2), and the telemetry samples — into one JSON bundle. Fits a clock calibration per physical GPU (`calibrate_b.py`) and derives `clock_policy` from this run's own telemetry/calibration evidence (§3.4). Also derives a single, consistent PCI-bus GPU identity for both kernels and telemetry (§3.5). |
| 2 | `normalize_b.py` | Parses the native rocprofv3 CSV dialects (`b_traces.py`) into normalized kernel/launch/range records, and binds each static `B|scope|...` marker label to its specific metadata invocation (`bind_labels`), using the enclosing `B|fwd|...` forward range to disambiguate repeated scope labels across layers/forwards. |
| 3 | `extract_b.py` → `b_check.py`/`b_extract.py` | The full structural-correctness gate. `b_extract.py`'s `extract()` attributes every kernel to a scope (`eager_correlation`), an exclusion reason (`excluded`), or leaves it `unresolved`/`ambiguous`/`graph_unresolved`. `b_check.py`'s `check()` additionally reconciles every forward against the independent scheduler record and the independent request ledger, checks clock-join quality, and — this is the gate that is **not yet fully satisfiable** for this pilot, because of the MoE/out-of-scope kernels from §2 — requires every kernel in the *entire* bundle to be `excluded` or `eager_correlation`. |
| — | `check_b.py` | A standalone CLI that just re-runs `b_check.py`'s `check()` against an already-normalized bundle and prints/saves the report; useful for re-checking without re-running extraction. |

### 4.5 The new canary check (this session's main deliverable)

| File | Purpose |
|---|---|
| `check_dataset_a_scope.py` | **New.** A narrower check, run *alongside* (not instead of) `b_check.py`. It calls the same `extract()`, then restricts attention to kernels whose `(physical_scope, variant)` is one of `scope_spec.json`'s mapped rules (i.e., has Dataset A backing). It requires every such kernel to be `eager_correlation` (fully, unambiguously resolved) and every matching scope invocation to have at least one attributed kernel. It separately requires **zero** `ambiguous`/`graph_unresolved` kernels anywhere in the whole bundle (those are always real defects, never a legitimate "no coverage yet" case, so they can never be waved off as "out of scope"). Everything else is reported, not failed on, under `out_of_scope_kernels`/`out_of_scope_attribution`. |

### 4.6 Tests

| File | Purpose |
|---|---|
| `tests/fixtures.py` | Builds small, fully synthetic (never real-GPU) bundles exercising every structural case: prefill-only, decode-only, mixed batches, TP>1, padded batches, chunked prefill. Explicitly documented as never GPU measurements. |
| `tests/test_collector.py` | 146 tests. Covers the marker buffer and drain protocol, native-trace parsing for both rocprofv3's CSV dialect and Kineto, the full `b_check.py` correctness-check matrix (including ~38 deliberately-damaged-record cases, each expected to be rejected), and — new this session — 3 tests for `check_dataset_a_scope.py`: a clean pass, correctly ignoring a real out-of-scope unresolved kernel, and correctly failing on an ambiguous Dataset-A-scope kernel. |

### 4.7 Evidence from the real pilot run (job 21798, the clean run)

These are small, derived, human-readable artifacts — not raw traces (raw
traces and large derived JSON dumps are intentionally excluded from git;
see `.gitignore`):

| File | Purpose |
|---|---|
| `evidence/pilot_runs/tp1_eager_assemble_config_21798.json` | The exact config used to assemble job 21798's bundle (paths, capture id, static manifest fields). |
| `evidence/pilot_runs/b-pilot-21798-bundle.json.calibration_holdout.json` | Clock calibration quality report: 7 anchors fit, 1 held out, **29ns** max residual. |
| `evidence/pilot_runs/b-pilot-21798-extracted/{inputs.json,failure.json}` | The full `b_check.py` gate's output for this run: still fails, specifically and only on the MoE/out-of-scope attribution gap from §2 (`failure.json` records the exact error). |
| `evidence/pilot_runs/b-pilot-21798-dataset-a-scope-check.json` | The new canary's report for this run: **passes** — 3,659/3,659 in-scope kernels resolved, 2,150 out-of-scope kernels correctly reported, not blocking. |
| `evidence/pilot_runs/TP1_EAGER_PILOT_RUN.md` | Narrative notes from an earlier pilot attempt (before this session's fixes); kept for historical context on what the original symptoms looked like. |

---

## 5. Suggested walkthrough

1. Read §1–§2 above (or just skim `B_SCHEMA.md`'s first section) so the
   vocabulary (scope, forward, physical_scope, a_scopes) is familiar.
2. Read `scope_spec.json` in full — it is short and is the actual
   definition of "the ops Dataset A has data for."
3. Read `marker_runtime.py` top-to-bottom. Its module docstring and inline
   comments explain the drain protocol and the native-trace-window
   bracketing in detail; the rest of the pipeline exists to consume what
   this module produces.
4. Skim `sglang_b_markers.patch`'s hunks for `schedule_batch.py` and
   `scheduler.py` to see how `marker_runtime.py`'s functions get called
   from real SGLang code.
5. Read `launch_plan_tp1_eager_pilot.json`'s `notes` field — it documents
   every non-default flag and why it's there.
6. Read `run_b.py`, focusing on `preflight()` and the drain sequence in
   `run()`'s `try`/`finally` block.
7. Read `assemble_bundle.py` top-to-bottom; it is the file most changed
   this session (clock_policy derivation, GPU-identity unification,
   dual-phase request loading).
8. Read `check_dataset_a_scope.py` (short, ~70 lines) and its 3 tests in
   `tests/test_collector.py` (search for `dataset_a_scope`).
9. Look at the job 21798 evidence files in §4.7 to see real numbers behind
   every claim above.
10. Run the test suite and the canary check yourself (see §6).

---

## 6. Reproducing the checks locally

From `dataset_b/`, with a Python environment that has this repo on
`PYTHONPATH` (no GPU or cluster access needed — everything below runs on
already-collected evidence or pure-Python fixtures):

```bash
# Full test suite (146 tests, all synthetic/pure-Python, ~1.5s)
python -m pytest tests/test_collector.py -q

# The new canary, against the real job 21798 data
python check_dataset_a_scope.py evidence/pilot_runs/b-pilot-21798-normalized.json
# -> structural_pass: true, in_scope_kernels: 3659 (all resolved),
#    out_of_scope_kernels: 2150 (all "unresolved", none ambiguous)
```

Note: `evidence/pilot_runs/b-pilot-21798-normalized.json` is a large
derived file excluded from git (see `.gitignore`); it must be regenerated
locally via `assemble_bundle.py` + `normalize_b.py` from a fresh cluster
run, or obtained from whoever ran job 21798, if you want to reproduce this
exact check rather than take the committed small-report files in §4.7 on
trust.

---

## 7. Open items (explicitly out of scope for this review)

- **MoE / out-of-scope kernel coverage.** Not decided in this session.
  Overlaps the operator-boundary-investigation decisions (D1–D5) already
  pending separately. `b_check.py`'s full gate will not pass until this is
  resolved one way or another.
- **The actual Dataset A vs Dataset B comparison** (§1.2's mean/median/
  max/min question). Not started. Depends on running the real sweep, which
  is gated on this canary review.
- **The remaining pilot matrix**: short/long prompts, chunking, mixed
  execution, graph replay and padding, all-rank TP4, Kineto cross-check,
  output equivalence, instrumentation-perturbation sensitivity. Not yet
  run; planned next, pending this review.
