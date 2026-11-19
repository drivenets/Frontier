# Qwen3-30B-A3B on MI355X: how the linear_op profiling data evolved — corrected story

Prepared 2026-10-11 from an independent re-verification. Where this document disagrees with the Confluence page
(id 7166984214, published 2026-09-24) or with `STORY.md`, **this version is the corrected one**; every difference is
itemised in `CORRECTIONS_TO_EXISTING.md`, and the evidence is in `FINDINGS_LOG.md`.

Covers all three sharded ops — `attn_pre_proj`, `attn_post_proj`, `attn_rope` — plus the replicated scopes
(`emb`, `input_layernorm`, `post_attention_layernorm`). Nothing below was re-collected: every number is recomputed
from existing CSVs and traces.

---

## 0. Summary

The simulator's per-operator regressors train on `time_stats.<op>.median` from the MI355X linear-op profiler. We
re-collected that data, found three defects in how it was measured, fixed them, and re-collected again. The defects
were: a CPython garbage collection landing inside the timed loop; CUDA-event pairs measuring the host's launch span
instead of device time whenever the GPU was idle; and two run-position artefacts in the corrected data. Along the way
we found that `attn_rope` had been timing a torch fallback that is not RoPE.

Against the final dataset (job 21539), the original file (job 21313) overstated the GEMM scopes by **1.38–2.87×**
(`attn_pre_proj`) and **1.61–2.88×** (`attn_post_proj`) as a full-grid per-TP median — up to **3.25×** and **3.08×**
for tokens ≤2,048 — and `attn_rope` by **4.50–13.02×**, that last figure confounded by the kernel change. Between 1 %
and 11 % of shapes run the other way.

**What is still open:** a pre-registered acceptance test (T2 at TP4) fails reproducibly at 1.9724 against a 2.0
threshold; the `emb` label pools two different call sites; the agreement with kernel traces is not the constant it was
described as; and four of the eight datasets exist only in a temporary directory outside git.

---

## 1. What the label is

Before any iteration: the quantity.

`time_stats.<op>.median` is the **median over timed runs of a CUDA-event interval spanning the scope**, in
milliseconds (`frontier/profiling/linear_op/linear_op_wrapper.py`, `_timed_pass`). It is not a sum of kernel
durations and it is not a production latency. The interval includes the inter-kernel dispatch gaps inside a
multi-kernel scope.

From job 21483 onward the profiler runs **two full passes** per shape:

| column | pass | conditions |
|---|---|---|
| `time_stats.<op>.*` (primary) | GPU-bound | a calibrated backlog is enqueued before the timed forwards, so the device is behind the host and every event pair measures device time |
| `time_stats_hostbound.<op>.*` | legacy | the loop exactly as before |

The two columns differ by **clock state as well as queue state**. A fixed 20M-cycle spin measures 1.93–2.08 M
cycles/ms from idle versus 2.38–2.40 M cycles/ms for a second spin — roughly 2.0 GHz versus 2.4 GHz SCLK
(`runs/2026-09-17_0846_linear_op_validation_grid_two_column/RUN.md`). Any legacy ÷ GPU-bound ratio therefore mixes the
instrument change with a clock change. **This was recorded in a RUN.md but never stated on the published page.**

---

## 2. The steps

Every collection lives in the shared datasets store — **not in git**:

```
/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/
```

Each directory holds the CSV, a self-contained `README.md` and `SHA256SUMS`. Job ids are Slurm ids and increase with
submission time, so they order the work; they belong in the explaining text, not in headings.

| Step | Job | Date | Rows | Token grid | reps | Timer | RoPE | Directory |
|---|---|---|---|---|---|---|---|---|
| Where we started | — | before 09-10 | 1,032 | 258 values, ≤4,096 | 20 | legacy | fallback | `legacy_pre-2026-09_torch-sdpa` |
| First re-collection | 21313 | 09-15 13:08 | 13,308 | 3,327 (1–16,384) | 3+50 | legacy | fallback | `2026-09-15_1308_linear_op_dense3327` |
| The spike, reproduced | 21334 | 09-16 06:29 | 13,308 | 3,327 | 3+50 | legacy | fallback | `2026-09-16_0629_dense3327_rerun` |
| *Diagnostic: holding the GPU behind the host* | *21356* | *09-16 08:32* | *4,800* | *1,200 (1,024–16,384)* | *3+50* | *legacy + knob* | *fallback* | `2026-09-16_0832_dip_x1_backlog_knob_diagnostic` |
| Garbage collection removed | 21361 | 09-16 08:53 | 13,308 | 3,327 | 3+50 | legacy | fallback | `2026-09-16_0853_gcfix_verification` |
| Timing the GPU, not the host | 21483 | 09-22 06:47 | 13,308 | 3,327 | 3+25 | two-column | **fused** | `2026-09-22_0647_dense_two_column_rope_fix` |
| Queue barrier removed | 21500 | 09-22 15:25 | 13,308 | 3,327 | 3+25 | two-column | fused | `2026-09-22_1525_dense_fixed_batchflush` |
| A real workload as backlog | 21519 | 09-22 17:35 | 13,308 | 3,327 | 3+25 | two-column | fused | `2026-09-22_1735_dense_workbacklog_settle3` |
| **Longer settling — the dataset we handed over** | **21539** | 09-23 09:49 | 13,308 | 3,327 | 3+25 | two-column | fused | `2026-09-23_0949_dense_workbacklog_settle8` |

md5s are in `FINDINGS_LOG.md`. Final dataset md5 `3635e4d305ae5cf84884e275d3ff37ef`.

**Two naming corrections carried into this table.** The row in italics, job 21356, is a **diagnostic experiment, not a
step in the dataset** — 4,800 rows, tokens ≥1,024, the old single-column timer plus an opt-in backlog knob
(`07_post_proj_rope_dip_root_cause.md:45`). It proved the host-bound hypothesis; it produced no dataset. And the two
rows "Queue barrier removed" and "A real workload as backlog" are genuine steps that both the evolution notebook and
the October-11 presentation scripts omitted entirely. Because job ids order submissions, note also that 21356
precedes 21361: the conceptual order of the fixes is not the chronological order of the runs.

**Iteration 0 is Qwen3-30B-A3B.** Verified from the inherited file's own architecture columns — n_head 32,
n_kv_head 4, n_embd 2048, n_expanded_embd 768, vocab_size 151936, use_qk_norm true — identical to job 21313 and to
`Qwen3MoeForCausalLM` in `data/config/models/qwen3-a3b-30b-moe.json`, and different on every field from gpt-oss-120b
(64 / 8 / 2880 / 2880 / 201088). The `model_arch` column reads `generic` in both files because the profiler did not
record an architecture name then; that is a metadata gap, not a different model. What *was* different about the
inherited data is the **attention** companion files, which used the portable TORCH_SDPA kernel instead of the
production AITER kernels — that is what prompted the whole re-collection.

**Two naming corrections.** Job **21356 is a diagnostic, not an iteration** — 4,800 rows, tokens ≥1,024, single-column
legacy timer plus an opt-in `FRONTIER_GPU_BACKLOG_MS` knob (`07_post_proj_rope_dip_root_cause.md:45`). It proved the
host-bound hypothesis; it did not produce a dataset. And jobs **21500 and 21519 are genuine iterations** that both the
evolution notebook and the October-11 presentation scripts omit entirely. Because job ids order submissions, note also
that **21356 precedes 21361**: the conceptual order of the fixes is not the chronological order of the runs.

---

## Iteration 1 — a garbage collection inside the timed loop

**Issue.** In the original dense grid (job 21313), exactly 32 rows — `num_tokens` 1968…1975 × TP {1,2,4,8} — contained
one timed run of `attn_pre_proj` that took **150.4–268.6 ms** where the other 49 took ≈0.09–0.14 ms. The spike was
always at **0-based timed run 11** — the 12th timed run counting from one, the 15th forward including three warm-ups.
Medians were untouched; means and maxima were destroyed. No other op showed it.

**How we verified it.** Re-ran the full dense grid byte-for-byte on a different node 17 hours later (job 21334): the
same eight token counts, the same op, the same run position, **146.4–287.7 ms**. An earlier 26-token repro
(jobs 21323–21325) had *failed* to reproduce it, which is what made the full re-run the decisive experiment. Twelve
instrumented experiments (E1–E8, jobs 21342–21353) then traced it to a CPython generation-2 garbage collection
running on the worker's main thread between the timer's start event and the GEMM launch. The diagnostic run recorded
every collection: all 32 fell on the identity line against the event gap, and all started in forward 14.
`gc.freeze()` left the spike in place but reduced it to 52–124 ms; `gc.disable()` removed it.

**Fix.** `gc.disable()` / `gc.enable()` around the timed loop (commit `74eee7c`), verified on the full grid in job
21361 against five pre-stated acceptance criteria. Largest `attn_pre_proj` sample afterwards: **2.086 ms**
(down from 268.645 ms).

**Impact.** Medians were unaffected throughout, so nothing downstream of the medians changed. The value of the fix was
to make per-sample work possible.

> **Correction to the published page.** Chapter 2 gives the rerun's range as "150–263 ms". The true range across all
> four TPs is **146.4–287.7 ms**; 263.9 ms is the TP8 maximum alone. The §0 figure "150–290 ms" is a fair rounding of
> the union across both jobs. The October-11 presentation's "150–243 ms" is TP1 of job 21313 only.
>
> **Correction.** "Timed run 11" is 0-based, as is "absolute sample 14"; "the 15th forward" in the same sentence is
> 1-based. The page states the base only in one figure caption, and `evolution.ipynb` plots a 1-based axis.

---

## Iteration 2 — the profiler was measuring the host

**Issue.** In the GC-fixed data (job 21361), `attn_rope` at TP 2/4/8 and `attn_pre_proj` at TP8 showed medians that
**fell** 25–28 % below their low-token plateau before rising again, at a token count that moved with TP;
`attn_post_proj` showed flat plateaus and steps instead of a rise. A cost that falls as the work grows cannot be the
hardware.

**How we verified it.** Two independent lines, neither of which is "the curve looked wrong":

1. **Intervention (X1, job 21356).** One opt-in knob — a calibrated `torch.cuda._sleep` enqueued after warm-up so the
   GPU runs ≈100 ms behind the host — with no change to any kernel. The dip and the plateaus disappeared for every op
   and TP.
2. **Traces (X2/X2d, rocprofv3).** The GPU was idle 45–60 % of every forward at TP 2/4/8 between 3,000 and 8,000
   tokens; the idle gap before the `o_proj` kernel was 47 µs at 3,000 tokens, shrinking to 5.8 µs from 7,008 tokens,
   and 5.9–6.0 µs everywhere with the backlog. The traced `o_proj` kernel itself was monotonic through the dip band,
   which refuted the leading hypothesis (a faster kernel being chosen in that band).

So the CUDA-event pairs were measuring the host's launch span whenever the device was idle at the start event.

**Fix.** Time every shape twice — legacy pass and GPU-bound pass — and make the GPU-bound pass the primary column
(commit `d4469be`); add a per-row shader-clock probe and host-enqueue columns; agree a measurement contract
(`10_measurement_contract.md`). First full dense collection with the fix: job 21483.

**Impact — per op, original (21313) ÷ final (21539), full grid, per-token ratio then median:**

| TP | attn_pre_proj | attn_post_proj | attn_rope |
|---|---|---|---|
| 1 | 1.381 [0.952–4.673] | 1.607 [0.957–3.205] | 4.500 [1.820–13.795] |
| 2 | 1.725 [0.956–5.389] | 2.101 [0.933–3.556] | 9.674 [2.093–15.783] |
| 4 | 2.359 [0.967–5.496] | 2.648 [0.972–4.744] | 12.367 [2.336–15.888] |
| 8 | 2.873 [0.953–5.656] | 2.879 [0.962–5.503] | 13.021 [3.539–15.744] |

For tokens ≤2,048 the GEMM ratios are higher: `attn_pre_proj` 1.530–3.247, `attn_post_proj` 1.805–3.083.

**Shapes where the old file was faster**, which the published page does not mention: `attn_pre_proj` 352 / 230 / 170 /
187 shapes at TP 1/2/4/8, `attn_post_proj` 222 / 122 / 69 / 41. `attn_rope` has none.

> **Correction.** The page's "1.3–2.9× and 4–12×" holds only for the full grid, and the RoPE upper bound is ≈13, not
> 12. These are label differences across a non-uniform token grid, not workload-weighted serving speed-ups.

### 2a. `attn_rope` was never RoPE

**Issue.** Every run up to and including job 21356 timed a forced torch fallback: 13 launch-bound kernels that rotated
only the first 64 columns of head 0. It was not RoPE, numerically or structurally.

**How we verified it.** Source inspection of the dispatch path plus a validation grid (jobs 21402–21412): vLLM's own
object dispatches to a 17-kernel torch path on ROCm, while Frontier's class with
`vllm._custom_ops.rotary_embedding` is the fused kernel SGLang also launches.

**Fix.** `get_rope` prefers the fused kernel; `attn_rope_impl` is recorded per row and `sanity_check` rejects anything
but `vllm_kernel` (commit `4b8fce3`).

**Impact.** Across the same protocol change (job 21356 → 21483) `attn_rope` moved by **2.585× / 3.440× / 4.447× /
6.063×** at TP 1/2/4/8, while `attn_pre_proj` and `attn_post_proj` moved by at most 1.2 % — which is what makes them
usable as controls here. **The original ÷ final RoPE ratio of 4.5–13.0× is therefore not an instrument artefact and
not a pure kernel speed-up: it is both, and it cannot be decomposed from this data.**

> **Correction.** Calling the GEMM scopes "unaffected controls" from code identity alone is not sound — jobs 21356 and
> 21483 differ in repetitions (50 vs 25), timer (single- vs two-column) and grid. The control claim survives because
> the *measured* agreement is within 1.2 %, and should be stated that way.

---

## Iteration 3 — run-position artefacts in the corrected data

**Issue.** With the medians clean, the per-run samples of job 21483 showed two patterns: a 2–8 % within-block drift
(samples declining from run ≈3 to run 25 of every block), and a fixed-position bump of ≈5.5 µs at run 20 for TP>1.

**How we verified it.** Single-GPU probes (`posprobe/`, jobs 21491–21519) isolated each: the drift tracked the clock
domains idling during the backlog **spin**; the bump matched the ROCm runtime's 1000-command batch-flush barrier.
Settle-after-spin was tried and ruled out; `gemm`, `stream` and `gemm_alloc` backlogs were compared.

**Fix.** Three changes, in three separate collections:
1. `DEBUG_CLR_MAX_BATCH_SIZE=1000000` removes the barrier → job **21500**.
2. Replace the spin with an allocating GEMM chain (`BACKLOG_KIND=gemm_alloc`) plus 3 settle forwards → job **21519**.
3. Raise to 8 untimed settle forwards (`SETTLE_STEPS=8`) → job **21539**, the final dataset.

**Impact — which of the three actually moved the medians** (later ÷ earlier, pooled over TP, full grid):

| transition | attn_pre_proj | attn_post_proj | attn_rope |
|---|---|---|---|
| 21483 → 21500 (batch-flush fix) | 0.9990 | 0.9992 | 1.0000 |
| 21500 → 21519 (**gemm_alloc backlog**) | **0.9769** | **0.9755** | **0.9845** |
| 21519 → 21539 (settle 8) | 1.0005 | 1.0017 | 1.0000 |

**Only the backlog change moved the medians**, by 1.5–2.4 % downward, for all three ops. The barrier removal and the
settle increase changed medians by ≤0.2 % — consistent with their documented purpose, which was to fix run-position
artefacts, not the median. This quantifies the page's "medians were taken mid-ramp, 2–3 % too high" as **≈2.3–2.4 %**
and attributes it specifically to the backlog change.

> **Correction.** Both the evolution notebook and the October-11 presentation omit jobs 21500 and 21519, which is
> exactly the pair that separates "fixed a position artefact" from "moved the median".

---

## 3. The final dataset: what it is, and what is still wrong with it

Job 21539, 13,308 rows, 3,327 token counts × TP {1,2,4,8}, 3 warm-up + 25 timed runs (`emb`: 6 + 50),
`settle_steps=8`, `backlog_kind=gemm_alloc`, `attn_rope_impl=vllm_kernel`, `measurement_type=CUDA_EVENT`,
md5 `3635e4d305ae5cf84884e275d3ff37ef`.

### Variability

Within-row relative MAD over the timed samples, median across shapes:

| op | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| attn_pre_proj | 0.555 % | 0.552 % | 0.539 % | 0.541 % |
| attn_post_proj | 0.684 % | 0.733 % | 0.859 % | 0.918 % |
| attn_rope | 0.769 % | 0.669 % | 0.616 % | 0.613 % |
| input_layernorm (TP1) | 0.781 % | — | — | — |
| post_attention_layernorm (TP1) | 0.756 % | — | — | — |
| **emb (TP1)** | **2.469 %** | — | — | — |

> **Correction.** The handoff document's "~0.55 %" is the `attn_pre_proj` figure, not a dataset-wide one.

### The curves are not monotonic

43–48 % of consecutive-token steps decrease in the final dataset, with a median down-step of ≈1.1 % and isolated
steps as large as **−42.00 %** (`attn_post_proj` TP8, 12288→12304) and **−15.83 %** (`attn_pre_proj` TP1, 256→257).

The published claim "the new collections are monotonic" is **false**. Monotonicity was never an acceptance criterion,
and it should not be offered as evidence of correctness. The diagnosis in Iteration 2 rests on the X1 intervention and
the traces. **The cause of the individual large steps is not documented** — tile selection is plausible given the
traced tile-switch behaviour, but establishing it would need a kernel trace at those shapes, which has not been run.

### `emb` pools two different call sites

`linear_op_impl.py:1089` and `:1092` both call `self.embed_tokens(input_ids)`; the first result is overwritten
without being used. The scope therefore records **two invocations per forward**, and the stored median pools them.
The two populations differ systematically:

| token band | median(first) ÷ median(second) |
|---|---|
| 1–512 | 1.058 |
| 512–2,048 | 1.025 |
| 2,048–4,096 | 1.124 |
| 4,096–8,192 | **1.232** |
| 8,192–16,384 | 1.127 |

Each sequence is flat over the block (slope ≈1e-5 per step), so this is a stable offset between two call sites, **not
progressive drift**. Anyone fitting `emb` is fitting a bimodal mixture, half of which is dead code. **Why the two
sites differ, and why the gap peaks at 4–8k tokens, is not established.**

### A pre-registered test fails, reproducibly

`test_measurement_validity.py` T2 requires median(`attn_post_proj`, 4096 tok) ÷ median(…, 64 tok) ≥ 2.00 at TP 1, 2, 4.

| dataset | TP1 | TP2 | TP4 | TP8 (not tested) |
|---|---|---|---|---|
| 21483 | 4.6964 | 3.0132 | 2.0375 | 1.8156 |
| 21500 | 4.5631 | 2.9803 | 2.0346 | 1.8144 |
| 21519 | 4.5163 | 2.9932 | **1.9724** | 1.7634 |
| 21539 | 4.5119 | 2.9371 | **1.9724** | 1.7719 |

T2 TP4 is **identical to four decimals in two independent collections** on different days with different settle
protocols. It is a reproducible systematic result, **not a flake**, and the handoff's "Known flake item" wording is
unsupported.

**What is actually wrong is the criterion, not the data.** T2's two ends behave differently with TP. In job 21539,
in microseconds:

| | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| 64 tokens (denominator) | 13.52 | 12.08 | 11.60 | 10.52 |
| 4,096 tokens (numerator) | 61.00 | 35.48 | 22.88 | 18.64 |
| T2 | 4.51 | 2.94 | **1.97** | 1.77 |

The denominator barely moves with TP (−22 %) because at 64 tokens the scope is launch- and fixed-cost bound. The
numerator scales with 1/TP (−69 %) because it is work bound. **T2 therefore falls with TP by construction**, so a
single TP-independent threshold of 2.00 cannot be right across TP 1/2/4/8. At TP4 both ends already sit in the
sub-10 µs regime this project's own documents flag as the instrument-floor zone. It crossed at job 21519 because the
backlog change lowered the work-bound end 2.72 % while the fixed-cost end did not move at all (+0.35 %).

**And the test it was proxying for already exists and passes.** T1′ = legacy ÷ GPU-bound at 8 and 64 tokens is the
direct host-bound detector, and it *rises* with TP — the opposite direction:

| tokens | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| 8 | 2.60 | 2.96 | 3.64 | 4.10 |
| 64 | 2.44 | 2.73 | 2.70 | 2.91 |

So the measurement at exactly the shapes T2 complains about is confirmed sound by T1′. T2 cannot distinguish the
defect it was written to catch (a small-token end inflated by host time) from a true property of the hardware (at
high TP the large shape has itself shrunk toward the fixed-cost floor). **Recommendation: keep T1′ as the gate, and
re-specify or retire T2 — with per-TP thresholds and TP8 actually tested if it is kept. Silently re-thresholding it
to accommodate 1.9724 would be the wrong fix.** Separately, T2 at TP8 is below 2.0 in every post-fix dataset
(1.76–1.82) and the validator never tests it there.

### Agreement with kernel traces is not a constant

Comparing job 21400's GPU-bound `attn_post_proj` TP2 medians against the eight hard-coded rocprofv3 values from job
21376 that the figures use:

| tokens | 1 | 8 | 64 | 3072 | 4096 | 4192 | 6144 | 8192 |
|---|---|---|---|---|---|---|---|---|
| event − trace (µs) | +2.08 | +1.48 | +1.76 | +2.62 | +1.40 | +3.24 | **+5.40** | +2.66 |

Range **+1.4 … +5.4 µs**, median **+2.35 µs**. The published "a constant ≈3 µs per scope boundary" overstates both the
constancy and the evidence base: eight shapes, one op, one TP, values rounded to 0.1 µs and hard-coded into the figure
generator, compared across two different jobs. **The broader trace-agreement claims of Chapter 7 (job 21389's 10.7 µs
gap, job 21400's event/kernel ratio spreads, the closure arithmetic) were NOT re-verified here** — the raw evidence
still exists in the historical scratch directory but the trace aggregation was not reconstructed. Treat them as
neither confirmed nor refuted.

---

## 4. What the regressors established

Trained on the final dataset's labels, random-shape split, 499 test token values × 15 (op, TP) targets × 3 methods:

| method | mean APE | p95 | max |
|---|---|---|---|
| Random forest | 1.249 % | 3.068 % | 93.857 % |
| XGBoost exact | 1.272 % | 3.089 % | 92.332 % |
| Linear interpolation | 1.372 % | 3.448 % | 62.925 % |

**Scope.** This is **interpolation between immediate neighbours on a dense grid**: 498 of 499 test tokens lie inside
the training range, and the median distance to the nearest training token is one grid step. It is not evidence of
extrapolation, of kernel-time accuracy, or of end-to-end simulator accuracy.

**The tail is one shape.** `num_tokens=1` is the only test token below the training minimum of 2. It produces all four
of RF's errors above 50 %. Excluding it, RF's maximum error falls from 93.86 % to **17.86 %**. Only 15 of 7,485 RF
rows exceed 10 % error.

**The "noise floor" is a cross-collection discrepancy.** The 1.188 % figure is the mean test-shape MAPE **between job
21519 (settle 3) and job 21539 (settle 8)** — two collections on different days. It is 2–3× larger than the
within-collection sampling noise of the median, so a run-level component dominates. The code divides it by `√2`
(a quadrature argument) and then subtracts it linearly (`baseline_v2.py:150-152`, `:198`), which is internally
inconsistent; under its own premise the residual would be ≈2.2× larger. With the undivided figure, 14 of 45
(op, TP, model) rows would "beat" the floor, which is incoherent. **Call it a cross-collection discrepancy, not an
irreducible instrument floor.** The resolving check — an independent same-protocol pair — does not exist.

**A caution about the holdout.** The reserved holdout was genuinely never scored by this benchmark. But the `random`
and `block` geometries partition the same tokens independently, and an earlier run (`full_v1`) scored both:
**281 of 332 random-holdout shapes (84.6 %) were already trained on or scored under the block geometry.** The holdout
is therefore not independent of what has already been seen. (The holdout was not evaluated in producing this
document; this is a comparison of token lists only.)

---

## 5. What this data does and does not say about serving

The operator-boundary investigation, re-audited against the approved image's own SGLang source and a real rocprofv3
trace:

- **RoPE and the KV-cache write are one kernel in serving.** On HIP the fusion gate needs no env var and no server
  flag (`models/utils.py:289-305`), and in the pilot trace the `rope_kv` scope contains exactly one kernel,
  `_fused_qk_rope_reshape_and_cache_kernel`, in all 384 occurrences. No separate cache-write kernel appears anywhere.
  Dataset A times these as two separate labels, neither of which is the deployed kernel, so **summing them is a cost
  hypothesis, not a measurement of the fused unit.**
- **Kernel providers depend on configuration, not on the image.** `SGLANG_USE_AITER` defaults to False
  (`environ.py:746`); the pilot harness had to set it explicitly. One qualification found during this audit: with
  AITER on, the QKV GEMMs still run as hipBLASLt Tensile kernels, so that fork changes the *dispatcher* and only
  sometimes the kernel.
- **A fused all-reduce + norm would double-count.** Frontier adds CC-backend all-reduce terms flat alongside the
  profiled norm predictions, with no de-duplication.
- **`emb` is trained but never enters the execution-time composition** (`families.py:294-304`:
  `predictor_target=False`, `execution_time_attr=None`), and it runs once per forward, not per layer. Final norm and
  lm_head GPU work are not modelled; a CPU-overhead `sampler_e2e_time` term does exist, so "sampling is unmodelled"
  would be wrong without the "GPU work" qualifier.
- **Three harness differences from serving** are confirmed at the cited lines: aliased input/residual tensors into the
  timed `fused_add_rms_norm`; every sequence sharing one attention block table; and a KV write implemented as two
  torch scatters that serving never performs.

**Dataset B — the serving-side pilot — has not validated any of this.** Its live evidence covers one TP1, eager run
with **two requests**. For the nine Dataset-A-mapped scopes the restricted structural check passes (3,659 kernels
resolved: 2,261 measurement + 1,398 warm-up), but the **full gate fails** (`structural_pass=false`, 2,150 out-of-scope
kernels unresolved), and **0 of 3,659 kernels reached `valid_context`**, so there is no accepted clock-qualified
timing. The Dataset A versus Dataset B comparison has **not started**. Collector-side fixes made on 2026-10-07 have
no live evidence at all: the re-run attempt (job 21837) failed because the model directory is now an empty stub on
every node checked.

---

## 6. Open items

| # | Item | What would resolve it |
|---|---|---|
| 1 | T2 is mis-specified: it falls with TP by construction, and T1′ already tests the same defect directly and passes | Re-specify T2 with per-TP thresholds and test TP8, or retire it in favour of T1′. Not a re-threshold-and-move-on. |
| 2 | Chapter 7's trace-agreement claims were not re-verified | Reconstruct the trace aggregation from the historical scratch (`f6/`, `f6b/`, `trace_metrics.json`, `analyze_trace.py`) with both estimators |
| 3 | Large isolated down-steps (−42 % at `attn_post_proj` TP8 12288→12304) | A kernel trace at those shapes |
| 4 | `emb` pools two call sites, one of which is dead code | Decide whether to drop the first call, record the sites separately, or exclude `emb`; it is currently unused downstream anyway |
| 5 | The "noise floor" cannot separate noise from the settle-protocol or day effect | An independent same-protocol recollection pair |
| 6 | The random holdout is 84.6 % contaminated across geometries | Decide before any future unseal |
| 7 | ~~Jobs 21356, 21483, 21500, 21519 live only in a temporary directory~~ | **Done 2026-10-11**: all ten collections staged to the shared datasets store with READMEs and `SHA256SUMS`; nothing in git |
| 8 | Whether the served checkpoint is block-FP8, which would change the provider analysis | Read the checkpoint's `quantization_config` |
| 9 | Dataset B: no accepted timing, full gate never passed, no A-vs-B comparison | Re-stage the model, re-run with the fixed collector, resolve the out-of-scope kernel policy |

---

## 7. Answers to the four questions

**1. Is the third quantity rocprofv3 kernel duration, compared with legacy and backlogged event measurements?**
Yes. Three quantities appear: the legacy event interval (`time_stats_hostbound.*`), the backlogged/GPU-bound event
interval (`time_stats.*`, the label), and rocprofv3 kernel durations from separate traced jobs. The third is used as a
reference, and in the published figures it enters as eight hard-coded, rounded TP2 values from job 21376.

**2. For which quantity, if any, is it justified as ground truth?**
For the **duration of individual kernels on the device**, rocprofv3 is the closer instrument, because it timestamps
kernels rather than an interval that also contains dispatch gaps. It is not ground truth for the label: the label
deliberately spans a *scope*, so it includes the inter-kernel gaps the trace excludes, and tracing perturbs execution.
It is best described as an independent reference for a related but different quantity.

**3. When the backlogged event measurement is smaller, what does that show?**
On the eight shapes recomputed here the GPU-bound event is always **larger** than the trace, by +1.4 to +5.4 µs. Where
below-reference cells have been reported elsewhere (Chapter 7 cites ratios below one in 9/32 and 15/32 cells), this
audit did **not** re-verify them, so no explanation is offered — variability, a systematic boundary difference, and a
trace-condition difference all remain open. What can be said: the comparisons cross jobs and clock states, so a
signed discrepancy cannot be attributed to any one of those without further work.

**4. What does agreement near one establish, and what does it leave unverified?**
It establishes that for the tested shapes the two instruments report compatible magnitudes. It does **not** establish
that the scope boundaries are equivalent, that the offset is constant (it is not: 1.4–5.4 µs here), that untested
shapes agree, or that either number equals the cost of the same work in serving — where, as section 5 shows, the
operator is sometimes a different kernel entirely.

---

*Terminology note: H0/H1 in the pre-registration documents name **hypotheses** about what the profiler was measuring,
not measurement methods. The measurement passes are "legacy" and "GPU-bound"/"backlogged".*
