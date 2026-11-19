# FINDINGS_LOG — independent verification of the Qwen3-30B-A3B / MI355X profiling story

Prepared 2026-10-11. Phase 1 only: **nothing was published, and the old Confluence page was not changed.**

Scope: the findings F01–F12 of `CONFLUENCE_STORY_REVIEW_HANDOFF.md`, plus five points raised separately by the user
(recorded here as U1–U5). Every finding was treated as a hypothesis and checked against data or code.

## Source versions actually used

| Item | Value |
|---|---|
| Repo | `/home/dn/Frontier-qwen3-profiling`, branch `smatar/qwen3-30b-mi355-profiling`, HEAD `25e163c` |
| Live Confluence page | id 7166984214, space DAD, title "Qwen3-30B-A3B on MI355X — the linear_op profiling story", lastModified **Sept 24, 2026**, author Shaked Matar. Retrieved 2026-10-11 via the Atlassian MCP; body 76,512 chars saved to the session scratchpad. Its own summary line says it was generated from `STORY.md` at commit **`4d18e8d`**. |
| `STORY.md` | 585 lines, mtime 2026-09-23 14:03, tracked |
| `evolution/`, `presentation_11_10_26/`, `CONFLUENCE_STORY_REVIEW_HANDOFF.md` | **untracked** (`git status` `??`) |
| F08 historical scratch | `/tmp/claude-1001/-home-dn-amd-playground/0de4f46f-.../scratchpad/` — **still present** (not ephemeral as feared) |
| D3/D4 cluster-scratch copies | `/tmp/claude-1001/-home-dn-Frontier-qwen3-profiling/839301bf-.../scratchpad/cluster_data/` — **still present** |

### Dataset inventory (recomputed; md5 over the whole file)

| Job | Role | Path | Rows | Tokens | reps (warm+act) | settle | backlog | rope impl | cols | two-column | md5 (10) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 21313 | original dense | `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` | 13,308 | 3,327 (1–16,384) | 3+50 | – | – | – | 69 | no | `97d6255414` |
| 21334 | rerun (spike repro) | `…/runs/2026-09-16_0629_linear_op_dense3327_rerun/linear_op.csv` | 13,308 | 3,327 | 3+50 | – | – | – | 69 | no | `3986255253` |
| 21356 | **X1 backlog-knob DIAGNOSTIC** | scratch `profiling_dip_x1/linear_op.csv` | 4,800 | 1,200 (1,024–16,384) | 3+50 | – | – | – | 69 | no | `93704fef54` |
| 21361 | GC fix | `data/profiling_gcfix/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` | 13,308 | 3,327 | 3+50 | – | – | – | 69 | no | `9381d3edaa` |
| 21483 | two-column + fused RoPE | scratch `profiling_dense_fixed/linear_op.csv` | 13,308 | 3,327 | 3+25 | – | – | `vllm_kernel` | 148 | yes | `1d84b3d195` |
| 21500 | batch-flush fix | scratch `profiling_dense_fixed_batchflush/linear_op.csv` | 13,308 | 3,327 | 3+25 | – | – | `vllm_kernel` | 148 | yes | `cd405d6428` |
| 21519 | gemm_alloc backlog, settle 3 | scratch `profiling_dense_fixed_workbacklog/linear_op.csv` | 13,308 | 3,327 | 3+25 | 3 | `gemm_alloc` | `vllm_kernel` | 150 | yes | `7016bd6260` |
| 21539 | **FINAL** settle 8 | `data/local_datasets/.../2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv` | 13,308 | 3,327 | 3+25 | 8 | `gemm_alloc` | `vllm_kernel` | 150 | yes | `3635e4d305` |

Reproduce: `/usr/bin/python3` + pandas; script kept at `scratchpad/phase1/inv.py`.

---

## Verdict table

| ID | Claim in one line | Verdict |
|---|---|---|
| F01 | Monotonicity is not a correctness criterion; the final data itself has downward steps | **CONFIRMED (strengthened)** |
| F02 | The label is an event interval under a changed protocol, not a kernel-duration sum | **CONFIRMED** |
| F03 | Final validation is not uniformly GREEN; T2 TP4 = 1.972 < 2.0 | **CONFIRMED (strengthened: not a flake)** |
| F04 | emb is invoked twice per forward; run-index labelling and MAD figures need care | **CONFIRMED (exact)** |
| F05 | The "noise floor" is a cross-collection discrepancy, not an instrument constant | **CONFIRMED (strengthened)** |
| F06a | Original per-repetition samples *were* retained | **CONFIRMED — my own README was wrong** |
| F06b | pre/post_proj are not automatically "unaffected controls" | **REVISED** — caution valid, controls hold to ≤1.2 % |
| F06c | Original/final RoPE ratios mix a kernel change with timing changes | **CONFIRMED** |
| F06d | D3 (21356) precedes D2 (21361); D0–D5 is not chronological | **CONFIRMED** |
| F07 | Recompute the ratio tables | **CONFIRMED (exact to 3 dp)** |
| F08 | Chapter 7 re-audit: f14 sourcing, closure arithmetic, trace agreement | **PARTLY CONFIRMED / PARTLY UNRESOLVED** |
| F09 | Regression success is label interpolation with important tails | **CONFIRMED + new holdout finding** |
| F10 | Operator-boundary report needs care | **CONFIRMED with revisions** |
| F11 | Dataset B is a restricted structural pilot | **CONFIRMED (exact)** |
| F12 | `linear_op_kernel_only.csv` missing next to D5 | **REVISED — exists elsewhere** |
| U1 | Step-naming conflict between artifacts | **CONFIRMED — and worse than described** |
| U2 | Deck ratio claims vs data | **CONFIRMED** |
| U3 | GC spike "150–290 ms" | **REVISED** |
| U4 | Slide-2 history is a gpt-oss → qwen3 move | **REJECTED** |
| U5 | Later iterations did not change attn_pre_proj | **REVISED — one step did** |

---

## F01 — monotonicity is not a correctness criterion

**Method.** Read `time_stats.<op>.median` from job 21539 for all three ops × TP {1,2,4,8}; evaluated the two cited
token pairs and then took a census of all consecutive-token steps. Script `scratchpad/phase1/f01_spike.py`.

**Result — both cited numbers reproduce exactly:**

- `attn_pre_proj`, TP1, 256 → 257 tokens: 0.038400 → 0.032320 ms = **−15.83 %**
- `attn_post_proj`, TP8, 12288 → 12304 tokens: 0.068001 → 0.039440 ms = **−42.00 %**

Census over the final dataset (3,326 steps per op/TP):

| op | TP1 | TP2 | TP4 | TP8 | worst single step |
|---|---|---|---|---|---|
| attn_pre_proj | 47 % down | 48 % | 46 % | 48 % | −15.83 % |
| attn_post_proj | 46 % | 48 % | 45 % | 47 % | **−42.00 %** |
| attn_rope | 47 % | 45 % | 43 % | 44 % | −7.42 % |

Median down-step is ≈ −1.1 % in every cell.

**Verdict: CONFIRMED and strengthened.** Between 43 % and 48 % of consecutive steps in the *final* dataset decrease.
The page's §0 sentence "the new collections are **monotonic**" is false as written. Most decreases are ≈1 % and
consistent with measurement scatter on a dense grid; a few are large. The diagnosis of the 2026-09 dip rests on the X1
intervention (job 21356) and the rocprofv3 traces, **not** on monotonicity.

**Limitation.** I did not establish the cause of the −42 % step. Attributing it to tile selection would require a
kernel trace at those shapes, which I did not run.

---

## F02 — what the label is

**Method.** Read `frontier/profiling/linear_op/linear_op_wrapper.py` (`_timed_pass` docstring at :291 ff.; two-pass
block at :422 ff.) and the event mechanics at :72–77, :122, :139–152.

**Result.** The primary label is `torch.cuda.Event.elapsed_time` between a start event recorded before a scope and an
end event after it, median over the timed runs. The docstring states the order explicitly: synchronize → warm-up →
synchronize → `mark_warmup_end` → clock probe → backlog (GEMM chain or spin) → `SETTLE_STEPS` untimed forwards →
timed forwards → synchronize → clock probe.

The two-column design runs **two full passes**: pass 1 legacy (`backlog_ms=0.0`) → `time_stats_hostbound.*`;
pass 2 GPU-bound (backlog enqueued) → `time_stats.*`. It is **not** a subtraction of CPU overhead.

**Verdict: CONFIRMED.** The label is a device-timestamp interval spanning the scope under a backlogged queue and a
settled clock, in ms. It is not a sum of kernel durations and not a production latency. The event interval also
contains the inter-kernel dispatch gaps inside a multi-kernel scope.

The two columns differ by **clock state as well as queue state** — the repository already says so, in
`runs/2026-09-17_0846_…_two_column/RUN.md`: a fixed 20M-cycle spin measures 1.93–2.08 M cycles/ms from idle versus
2.38–2.40 M cycles/ms for a second spin, i.e. ≈2.0 GHz vs ≈2.4 GHz SCLK. So any legacy/GPU-bound ratio mixes the
instrument change with a clock change. That caveat is in a RUN.md but **not** on the Confluence page.

---

## F03 — final validation is not uniformly GREEN

**Method.** Read `test_measurement_validity.py` (T2 defined at :48 and :64) and `14_regressor_dataset_handoff.md:109-111`;
recomputed T2 = median(attn_post_proj, 4096 tok) / median(…, 64 tok) for every dataset and TP.

| dataset | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| 21313 original | 1.6842 ✗ | 1.6868 ✗ | 1.3368 ✗ | 1.2701 ✗ |
| 21361 gcfix | 1.6257 ✗ | 1.0710 ✗ | 1.2561 ✗ | 1.3283 ✗ |
| 21483 | 4.6964 | 3.0132 | 2.0375 | 1.8156 ✗ |
| 21500 | 4.5631 | 2.9803 | 2.0346 | 1.8144 ✗ |
| 21519 | 4.5163 | 2.9932 | **1.9724 ✗** | 1.7634 ✗ |
| 21539 FINAL | 4.5119 | 2.9371 | **1.9724 ✗** | 1.7719 ✗ |

(✗ = below the 2.00 threshold. The validator applies T2 only at TP 1, 2, 4.)

**Verdict: CONFIRMED and strengthened.** The handoff asked for repeatability evidence before calling it a flake.
There is such evidence, and it points the other way: T2 TP4 is **1.9724 in both job 21519 and job 21539** — two
independent collections, on different days, with different settle protocols, agreeing to four decimals. That is a
reproducible, systematic result, not a flake. Calling it a "Known flake item" in
`14_regressor_dataset_handoff.md:110-111` is unsupported.

Two further facts the documents do not state: TP4 passed (2.0375 / 2.0346) in jobs 21483 and 21500 and only fell below
2.0 once the `gemm_alloc` backlog arrived (job 21519) — consistent with that change lowering medians ≈2.4 % (see U5);
and **TP8 is below 2.0 in every post-fix dataset** (1.76–1.82), but the validator never tests T2 at TP8.

### Why T2 TP4 is 1.97 — the mechanism (added 2026-10-11)

T2's two ends behave completely differently with TP. In microseconds, `attn_post_proj` in job 21539:

| | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| 64 tokens (denominator) | 13.52 | 12.08 | 11.60 | 10.52 |
| 4,096 tokens (numerator) | 61.00 | 35.48 | 22.88 | 18.64 |
| **T2** | 4.51 | 2.94 | **1.97** | 1.77 |

The denominator barely moves with TP (13.5 → 10.5 µs, −22 %) because at 64 tokens the scope is launch- and
fixed-cost bound, not work bound. The numerator scales roughly with 1/TP (61.0 → 18.6 µs, −69 %) because it is work
bound. **T2 therefore falls with TP by construction**, and a single TP-independent threshold of 2.00 cannot be right
for all of TP 1/2/4/8. At TP4 both ends are already inside the regime the project's own documents call the ≈3 µs
instrument floor / sub-10 µs caution zone.

Why it crossed at job 21519: the `gemm_alloc` backlog lowered the work-bound end by 2.72 % at TP4 (23.52 → 22.88 µs)
while the fixed-cost end did not move (11.56 → 11.60 µs, +0.35 %). That alone pushed 2.035 → 1.972.

**The decisive point: T2 is a weak proxy for something another test already measures directly.** T1′ =
legacy ÷ GPU-bound at 8 and 64 tokens is the actual host-bound detector, and in job 21539 it passes everywhere and
*rises* with TP — the opposite direction from T2:

| tokens | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| 8 | 2.60 | 2.96 | 3.64 | 4.10 |
| 64 | 2.44 | 2.73 | 2.70 | 2.91 |

So the measurement at exactly the shapes T2 complains about is confirmed sound by T1′. **The T2 failure is a defect in
the acceptance criterion, not in the data**: T2 cannot distinguish "the small-token end is inflated by host time"
(the thing it was meant to catch, which T1′ catches properly) from "at high TP the large shape has itself shrunk
toward the fixed-cost floor" (a true property of the hardware). Recommended: re-specify or retire T2, keep T1′ as the
gate, and if T2 is kept, give it per-TP thresholds and actually test TP8. Re-thresholding it silently to accommodate
1.9724 would be the wrong fix.

---

## F04 — run index, embedding invocations, and variability

**Method.** Read `linear_op_impl.py:1088-1097`; parsed the raw `time_stats.<op>.samples` arrays of job 21539.
Script `scratchpad/phase1/f04.py`.

**Embedding is invoked twice per forward — confirmed in code:**

```python
def forward(self, input_ids, positions):
    hidden_states = self.embed_tokens(input_ids)      # :1089
    residual = None
    for _ in range(self.num_repeat_steps):
        hidden_states = self.embed_tokens(input_ids)  # :1092  <- overwrites :1089
```

The first result is overwritten without being used. Sample-array lengths confirm the consequence: in job 21539 the
three GEMM scopes and both norms store 28 samples (3 warm-up + 25 timed) while `emb` stores **56** (6 + 50).

**Position profile, TP1, 3,327 rows:**

| | value |
|---|---|
| median over FIRST invocations (even positions) | **1.0166** |
| median over SECOND invocations (odd positions) | **0.9699** |
| slope of the first-invocation sequence | +0.00001 / step (range 1.0147–1.0178) |
| slope of the second-invocation sequence | −0.00002 / step (range 0.9691–0.9708) |

**Verdict: CONFIRMED, exactly.** Both numbers reproduce to four decimals, and the reviewer's caution is right: each
invocation sequence is **flat**. The pattern is a stable ≈4.8 % offset between two call sites that alternate, not
progressive drift. Reading the interleaved array as a 50-point "run position" curve would show a zigzag that is an
artefact of the call-site alternation.

**New, beyond the handoff.** The gap is token-dependent and much larger than 4.8 % in places:

| token band | median (first / second) |
|---|---|
| 1–512 | 1.0584 |
| 512–2,048 | 1.0247 |
| 2,048–4,096 | 1.1243 |
| 4,096–8,192 | **1.2321** |
| 8,192–16,384 | 1.1266 |

The stored `time_stats.emb.median` is the **pooled** median over both call sites (verified: stored 0.006560 =
pooled 0.006560, vs first-only 0.006680 and second-only 0.006320). So the published `emb` label is the median of a
bimodal mixture of two call sites, one of which is dead code. Anyone fitting `emb` is fitting that mixture.

**Within-row relative MAD** (per shape over timed samples, median over shapes), job 21539:

| op | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| attn_pre_proj | 0.555 % | 0.552 % | 0.539 % | 0.541 % |
| attn_post_proj | 0.684 % | 0.733 % | 0.859 % | 0.918 % |
| attn_rope | 0.769 % | 0.669 % | 0.616 % | 0.613 % |
| emb (TP1, pooled) | **2.469 %** | – | – | – |
| input_layernorm (TP1) | 0.781 % | – | – | – |
| post_attention_layernorm (TP1) | 0.756 % | – | – | – |

**Verdict: CONFIRMED.** All three reported ranges reproduce (0.54–0.55 %, 0.68–0.92 %, 2.47 %). The handoff's point
stands: `14_regressor_dataset_handoff.md:100` says "~0.55 %" without qualification, but that is the pre-projection
value; post-projection reaches 0.92 % and `emb` 2.47 %.

**Run-index base — a documentation defect I add to this finding.** For the GC spike I located the maximum inside the
raw arrays: absolute 0-based index **14**, i.e. the 15th forward, i.e. 0-based timed run **11** = 1-based timed run
**12**. `STORY.md:93` and the live page say "timed run 11 (absolute sample 14, i.e. the 15th forward including 3
warm-ups)" — internally consistent only by mixing 0-based and 1-based counting in one sentence. The page discloses
the base in exactly one place (Fig. 4's caption, "0-based, as in the investigation documents"); Chapter 1 does not,
and `evolution.ipynb`'s run-position figures plot a **1-based** "timed run index". A reader matching "run 11" against
those plots lands one position early.

---

## F05 / F09 — regressor bench (delegated audit, results reproduced by that audit from source)

**F05 — the "noise floor".** CONFIRMED and strengthened.
- Jobs 21519/21539 differ in `settle_steps` 3 vs 8 — verified as the *only* differing configuration column across the
  two 13,308 × 150 CSVs. They also differ in job id and calendar day, so the comparison is confounded with
  run-to-run state.
- Mean test-shape discrepancy recomputed **twice independently**: **1.188254 % MAPE** (from the derived column, and
  from the two raw CSVs). Per-regressor 0.889 %–1.524 %.
- `baseline_v2.py:150-152` divides the pairwise MAPE by `np.sqrt(2)`; `:198` then subtracts that linearly as
  `learnable_mape_above_floor`.
- **The code is internally inconsistent**: `√2` is a quadrature argument (variances add), but line 198 subtracts
  linearly. Under its own premise the consistent estimate is `sqrt(test² − floor²)`, which is ≈2.2× larger
  (rf: reported 0.409 vs consistent 0.924).
- No assumption is stated anywhere (grep for `gaussian|independent|i.i.d|assum|quadrature` finds only colloquial uses).
- Empirically the discrepancy is **2–3× larger** than the within-collection sampling noise of the median
  (e.g. attn_pre_proj TP1: implied 0.309 % vs observed 1.074 %), so a run-level component dominates.
- Using the raw pairwise floor, **14 of 45** (op, TP, model) rows would "beat" the noise floor — incoherent; the `√2`
  is exactly what reduces that count to 0.
- `learnable_mape_above_floor` is written to `results.csv` but published nowhere.

**Verdict:** call it a **cross-collection discrepancy between job 21519 (settle 3) and job 21539 (settle 8)**, not an
irreducible noise floor. Resolving check (not run): an independent same-protocol pair (settle 8 vs settle 8).

**F09 — regression results.** CONFIRMED exactly, with one important addition.
- 22,455 prediction rows = 499 test tokens × 15 (op,TP) × 3 methods. Confirmed.
- RF 1.249138 / 3.068267 / 93.857161; XGB 1.271686 / 3.089002 / 92.332174; interp 1.371881 / 3.447971 / 62.925173 —
  all nine figures reproduce.
- RF's worst row is `attn_pre_proj` TP8 `num_tokens=1`, label 0.01092 ms, prediction 0.021169, APE 93.857161 %.
- The split is the **random** geometry (`baseline_v2.py:134`), described in `splits.py:12-13` as interpolation between
  immediate neighbours. 498 of 499 test tokens lie strictly inside the train range; median nearest-train-neighbour
  distance is 1 grid step.
- **The headline max error is a single-shape extrapolation artefact**: `num_tokens=1` is the only test token outside
  the train range (train minimum 2). Excluding it, RF max falls 93.86 % → 17.86 %.
- Tail concentration: only 15 of 7,485 RF rows exceed 10 % error; tokens {1, 4} (30 rows, 0.4 % of data) carry
  15–19 % MAPE versus ≈1.2 % everywhere else.

**NEW — the reserved holdout is not pristine, and the user should see this.** For `baseline_v2` the holdout was
genuinely never read (`include_holdout=False`; no `holdout_metrics.csv`; no `splits/HOLDOUT_LOG.md`, so the unseal
command has never run). **But** the `random` and `block` geometries partition the same 3,327 tokens independently, and
`results/full_v1/` scored both. By token-set membership alone: **57 of 332** random-holdout tokens (17.2 %) were
scored as *block-test* rows in `full_v1`, and a further **224** (67.5 %) were used as *block-train* tokens — i.e.
**281 of 332 (84.6 %)** random-holdout shapes have already been trained on or scored under the block geometry. The
holdout tier was **not** evaluated during this audit; this is a set-comparison of token lists only.

---

## F06 — presentation sample availability and experimental controls

**(a) Samples were retained. CONFIRMED — and this is an error in my own work.**
Job 21313's CSV has `time_stats.<op>.samples` populated: 53 values per shape for the three GEMM scopes (3 warm-up +
50 timed) and 106 for `emb`. `presentation_11_10_26/README.md` (the s04 bullet) says "per-repetition samples are not
kept for this dataset". That is wrong. The figure itself is still correct — it plots the stored min/max/median — but
the stated reason is false, and a per-sample version of that figure is possible from the raw file.

**(b) "Unaffected controls". REVISED.**
The caution is methodologically right: D3 (21356) and D4 (21483) differ in repetitions (50 vs 25), timing
implementation (single-column legacy + backlog knob vs two-column), and grid (1,200 vs 3,327 token counts), so
"unchanged operator code" does not by itself make them controls. Empirically, however, the controls hold:

| op | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| attn_pre_proj (D3/D4 median) | 0.988 | 0.989 | 0.989 | 0.994 |
| attn_post_proj | 1.003 | 1.002 | 0.999 | 0.997 |
| attn_rope | **2.585** | **3.440** | **4.447** | **6.063** |

So the GEMM scopes agree within 1.2 % across that protocol change while `attn_rope` moves 2.6–6.1×. The control claim
survives, but it should be stated as a measured agreement with the differing conditions named, not asserted from code
identity.

**(c) RoPE ratios mix a kernel change with timing changes. CONFIRMED** — see F07 and U2.

**(d) D3 precedes D2. CONFIRMED.** Job 21356's output file is dated **2026-09-16 08:32**; job 21361's is
**2026-09-16 08:55**. Slurm ids are monotonic in submission order and 21356 < 21361. The D0–D5 labels in
`evolution.ipynb` are therefore a *conceptual* ordering, not a chronological one, and the notebook does not say so.

---

## F07 — the ratio tables

**Method.** For each token count present in both files, ratio = job 21313 median ÷ job 21539 median, then summarise
across token counts. Script `scratchpad/phase1/ratios.py`.

**Full range (n = 3,327 shared token counts per TP):**

| TP | pre_proj median [min–max] | post_proj median [min–max] |
|---|---|---|
| 1 | **1.381** [0.952–4.673] | **1.607** [0.957–3.205] |
| 2 | **1.725** [0.956–5.389] | **2.101** [0.933–3.556] |
| 4 | **2.359** [0.967–5.496] | **2.648** [0.972–4.744] |
| 8 | **2.873** [0.953–5.656] | **2.879** [0.962–5.503] |

**Verdict: CONFIRMED to three decimal places** — every one of the eight medians and all sixteen extremes match the
prior review's table. This is an independent reproduction, not a copy.

**Added, as the handoff asked.** Counts below one (the old file was *faster* than the final one at those shapes):

| op | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| attn_pre_proj | 352 | 230 | 170 | 187 |
| attn_post_proj | 222 | 122 | 69 | 41 |

So 1–11 % of shapes run counter to the headline direction. Quantiles are in `numbers_v2.md`.

`attn_rope`, original ÷ final, full range: TP1 **4.500** [1.820–13.795], TP2 **9.674** [2.093–15.783],
TP4 **12.367** [2.336–15.888], TP8 **13.021** [3.539–15.744]; zero shapes below one.

These are label differences across a non-uniform token grid, not workload-weighted serving speed-ups.

---

## F08 — Chapter 7 re-audit

**(a) Figure f14's sources. CONFIRMED.** `make_story_figures.py:383` hard-codes
`rocprof_tp2 = {1: 4.0, 8: 9.2, 64: 10.2, 3072: 28.0, 4096: 34.4, 4192: 35.6, 6144: 45.5, 8192: 55.9}` — eight values,
TP2 only, rounded to 0.1 µs. Line 390 labels them "rocprofv3 kernel (job **21376**)"; line 395's caption says the
event data is job **21400**. So the figure is a cross-job comparison of job 21400 event points against eight
hard-coded job-21376 trace values. The same eight numbers appear in
`runs/2026-09-17_0846_…_two_column/RUN.md`, where they are compared against job **21389**.

**(b) "Agrees with kernel traces to a constant ≈3 µs per scope boundary". REVISED — it is not constant.**
Recomputing the gap from the actual job-21400 CSV against those hard-coded values:

| tokens | 1 | 8 | 64 | 3072 | 4096 | 4192 | 6144 | 8192 |
|---|---|---|---|---|---|---|---|---|
| GPU-bound − trace (µs) | +2.08 | +1.48 | +1.76 | +2.62 | +1.40 | +3.24 | **+5.40** | +2.66 |

Range **+1.40 … +5.40 µs**, median **+2.35 µs** — a 3.9× spread across eight shapes of one op at one TP. "A constant
≈3 µs per scope boundary" overstates both the constancy and the evidence base.

**(c) The closure arithmetic. UNRESOLVED.** I could not verify the claim that median-based closure
0.779/0.908/0.738 becomes ≈0.982/0.979/0.982 using sums. Reason: the closure identity compares the forward span
against the **traced kernel-duration sum**, not against the profiled scopes. The CSV carries only six timed scopes
plus `forward_gpu_span`; summing those six gives 0.33–0.36 of the forward span at 8 tokens, because the forward also
runs attention and the MoE/MLP block, which this profiler does not time. The check itself lives in
`scratchpad/scorecard.py:74-75` as `check E: W′/G_sum(cb)` with a (0.9, 1.1) band, where `W′` comes from the event
JSONL and `G_sum` from the trace. `09_measurement_validation_results.md:59` does report "closure 0.76–0.80 in jobs
21389/21397, 0.74–0.91 in job 21400". **Missing check:** re-run the trace-side aggregation from
`scratchpad/f6/*.jsonl` and `trace_metrics.json` with both estimators. The raw files still exist, so this is doable;
I did not do it.

**(d) Other F08 sub-claims. UNRESOLVED.** The job-21389 10.7005 µs gap, the job-21400 event/kernel ratio spreads
(0.9206–1.3032 etc. with 9/32 and 15/32 below one), and the persistence of below-reference cells across three event
runs all require the same trace pipeline. I located the evidence (`scratchpad/f6/`, `f6b/`, `fixedclk/`,
`trace_metrics.json`, `analyze_trace.py`) but did not reconstruct it. **Do not treat these as either confirmed or
refuted.**

---

## F10 — operator-boundary report (delegated audit)

CONFIRMED with revisions. `frontier/` is byte-identical between the report's base commit `284130b` and HEAD, so every
Frontier citation is testable as written. Work committed *after* the report added the unpacked SGLang source from the
approved image and real rocprofv3 traces, so several `[source]` claims are now **measured**:

- **Fused RoPE + KV write (CONFIRMED, upgraded to measured).** `models/utils.py:289-305` gates fusion on HIP with no
  env var and no server flag. In the pilot trace the `rope_kv` scope contains exactly one kernel,
  `_fused_qk_rope_reshape_and_cache_kernel`, in all 384 occurrences; no separate cache-write kernel appears anywhere.
  Summing Dataset A's two separate labels is a cost hypothesis, as the report says.
- **Providers depend on configuration (CONFIRMED, two qualifications).** `environ.py:746` `SGLANG_USE_AITER` defaults
  False; the pilot harness had to set it itself. Qualification (a): "the image has no vLLM" is classed observed but
  E5 §5 marks it *inferred* for the approved digest. Qualification (b): the F1 GEMM fork is narrower than stated —
  in the AITER-on pilot the QKV GEMMs are hipBLASLt Tensile kernels, i.e. `tgemm.mm` routed back to hipBLASLt, so F1
  changes the dispatcher and only sometimes the kernel.
- **Fused all-reduce + norm double-counts (CONFIRMED).** But the line citations are wrong: `execution_time.py:884-907`
  covers only the MoE all-reduce; the attention one is at `:932-941`. A systematic ≈48-line offset affects the helper
  methods cited from that file.
- **`emb` trained but unused (CONFIRMED); "sampler unmodeled" (REVISED).** `families.py:294-304` sets
  `predictor_target=False`, `execution_time_attr=None`, and `emb` appears nowhere in `execution_time.py`. But a
  CPU-overhead `sampler_e2e_time` does exist (`time_components.py:631`), so the "GPU work" qualifier must be kept.
- **Harness defects (all three CONFIRMED)** at the exact cited lines.
- **Source/measured discipline (REVISED).** Mostly good, but `REPORT.md:190` "Trace-confirmed five kernels (two
  copies)" is not backed by any trace — E1 explicitly marks those kernel internals as assumed, and the real trace
  shows 3 kernels for the SGLang analogue in decode (5 only in prefill). That sentence must be rewritten or dropped.
- **New, UNRESOLVED:** the pilot's model path is a block-FP8 checkpoint, which would mean `unquant.py` is not the GEMM
  path at all. Missing check: read the served checkpoint's `quantization_config`.

---

## F11 — Dataset B (delegated audit)

CONFIRMED, every number exact.
- Restricted Dataset-A-scope check: `in_scope_kernels = 3659` = 2,261 measurement + 1,398 warm-up,
  `structural_pass = true` **for the restricted check only**.
- Clock states: `{unknown_resolution: 2793, outside_calibration: 866}` — sums to 3,659, i.e. **0 of 3,659 in-scope
  kernels reached `valid_context`**, while `B_SCHEMA.md:70` requires valid context for acceptance.
- `out_of_scope_kernels = 2150`, all `unresolved`; the full gate returns
  `{"structural_pass": false, "dataset_accepted": false, "error": "unresolved/ambiguous target attribution"}`
  (re-run and reproduced).
- The A-vs-B comparison has **not started** (`REVIEW_INSTRUCTIONS.md:299-301`). The live run was TP1, eager, one
  tracer, **two requests** (6 input/2 output and 8 input/4 output).
- The Oct-7 fixes were applied to *analysis* code and re-run against the Oct-5 raw capture (extractor source hashes
  match the working tree). **Collector-side fixes have no live evidence at all**: the re-run attempt, job 21837,
  failed because the model directory is now an empty stub on every node checked, and it left no evidence file.

Status by layer: (a) source inspection strong; (b) synthetic tests strong but narrow (152 collected; the only
committed log is stale, Oct 4, 136 tests); (c) live structural **restricted pass only**, full gate never passed;
(d) live timing **none**; (e) serving equivalence **zero**.

The commit title `237dd3e` "TP1 eager pilot validated end-to-end" overclaims; its body is honest.

---

## F12 — evidence availability

**REVISED.** `linear_op_kernel_only.csv` is **not** next to D5 (that directory holds only `linear_op.csv`), but the
artifact exists in at least four places, including
`/tmp/qwen3-pr2-consolidation-20261011T110831Z/evidence/linear_op/2026-09-17_1250_linear_op_rope_fix_validation_grid/`
and the historical scratch. So the proposed cross-check was not impossible — it was looking in the wrong place. I did
not run the cross-check itself.

`14_regressor_dataset_handoff.md:113-118` describes the file as the same grid under `--profile_method record_function`,
one sample per shape, and explicitly says not to train on it.

**Provenance — RESOLVED 2026-10-11.** At the time of the audit, jobs 21356, 21483, 21500 and 21519 existed only
under `/tmp/claude-1001/.../scratchpad/cluster_data/`, a session-scoped temporary directory outside git. On the
user's instruction ("datasets should all live under shared storage, nothing should go to github") every collection
has now been staged into the shared datasets store, alongside the three that were already there:

```
/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/
  legacy_pre-2026-09_torch-sdpa/                   dbc872076f   inherited (iteration 0)
  2026-09-15_1308_linear_op_dense3327/             97d6255414   job 21313   (already present)
  2026-09-16_0629_dense3327_rerun/                 3986255253   job 21334
  2026-09-16_0832_dip_x1_backlog_knob_diagnostic/  93704fef54   job 21356   (diagnostic, not an iteration)
  2026-09-16_0853_gcfix_verification/              9381d3edaa   job 21361
  2026-09-22_0647_dense_two_column_rope_fix/       1d84b3d195   job 21483
  2026-09-22_0849_dense_200fwd_all_fixes/          4c5f40f7de   job 21486   (already present)
  2026-09-22_1525_dense_fixed_batchflush/          cd405d6428   job 21500
  2026-09-22_1735_dense_workbacklog_settle3/       7016bd6260   job 21519
  2026-09-23_0949_dense_workbacklog_settle8/       3635e4d305   job 21539   (FINAL, already present)
```

Each new directory carries the CSV, a self-contained `README.md` (what / why / what it does **not** fix / known
remaining defects) and `SHA256SUMS`. Copies were made server-side from the cluster checkout
(`/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling_*`), whose md5s were first confirmed to match the
locally audited files exactly; only the inherited file had to be uploaded. `sha256sum -c` passes in all ten
directories. Nothing was added to git and no existing directory was modified.

---

## U1 — the step-naming conflict

**CONFIRMED, and the situation is worse than the user described: four conventions are in use.**

| Convention | Where | "it 1" | "it 2" | "it 3" | "it 4" | final |
|---|---|---|---|---|---|---|
| A — deck | `build_story_deck.py:146,195,245` | GC spike (21313→21361) | dip/host-bound + RoPE (→21483) | run-position (→21539) | – | 21539 |
| B — evolution notebook | `evolution.ipynb` S2_RUNS | 21313 | 21361 | **21356** | 21483 | 21539 |
| C — presentation script | `make_presentation_figures.py` STEPS | 21361 | **21356** | 21483 | *"does not exist"* | 21539 |
| D — as the user described it | – | – | 21483 | 21500 | 21519 | 21539 |

**Which is correct?** Convention **D is the right sequence of dataset iterations** and B and C are both wrong, for two
independent reasons:

1. **Job 21356 is not a story iteration.** `07_post_proj_rope_dip_root_cause.md:45` documents it as experiment **X1**:
   an opt-in `FRONTIER_GPU_BACKLOG_MS` knob, tokens 1,024–10,240 step 8 plus 10,368–16,384 step 128, **4,800 rows**,
   single-column legacy timer. It is a *diagnostic that proved the host-bound hypothesis*, not a re-collection of the
   dataset. Both B and C place it in the iteration sequence, and C labels it "it 2".
2. **Jobs 21500 and 21519 are real dataset iterations and both B and C omit them entirely.** Both are full 13,308-row
   dense grids with the two-column timer (21500 = batch-flush fix, 21519 = gemm_alloc backlog + settle 3).

My own `make_presentation_figures.py` additionally asserts in `README.md` and `numbers.md` that **"There is no 'it 4'"**.
That statement is false: under convention D, it 4 = job 21519.

Convention A (the deck) is defensible as a *narrative* framing — three problems, three fixes — but it collides with
C: the deck's "Iteration 2" is the host-bound fix, whereas C's "it 2" is job 21356. A reader moving between the deck
and the figure files will mismatch them.

**Recommendation.** Drop bare "it N" everywhere. Name every step by job id, and keep the deck's three-act narrative as
narrative only. Affected artifacts are listed in `CORRECTIONS_TO_EXISTING.md`; regenerated figures carry the `_v2`
suffix.

---

## U2 — deck claims versus data

**CONFIRMED.** Using the per-token-ratio-then-median definition:

| claim | op | ≤2,048 tokens | ≤5,000 | full range | consistent? |
|---|---|---|---|---|---|
| GEMM "1.3–2.9×" | attn_pre_proj | 1.53–3.25 | 1.48–3.14 | **1.38–2.87** | full range only |
| GEMM "1.3–2.9×" | attn_post_proj | 1.81–3.08 | 1.76–3.03 | **1.61–2.88** | full range and ≤5,000 |
| RoPE "4–12×" | attn_rope | 7.35–13.54 | 6.41–13.44 | **4.50–13.02** | **no — upper bound is ≈13, not 12** |

(Ranges are across TP 1/2/4/8 of the per-TP median ratio.)

So "1.3–2.9×" is right for the full grid and understates the small-token regime, where TP8 pre-projection reaches
3.25×. "4–12×" understates the top end at every range; the honest statement is **4.5–13.0×** full-range, and it is
confounded by the kernel change (F06c). The old page's Fig. 20 caption says "1.3–3×" for GEMMs, which is closer than
§0's "1.3–2.9×", and describes a rolling median over 15 shapes — a smoothed curve, not the raw ratio.

---

## U3 — the GC spike magnitude

**REVISED.** Computed from the raw sample arrays over all four TPs:

| job | per-TP ranges | overall |
|---|---|---|
| 21313 original | TP1 150.4–242.8, TP2 151.0–255.1, TP4 156.4–268.6, TP8 152.9–263.9 | **150.4–268.6 ms** |
| 21334 rerun | TP1 149.7–262.9, TP2 151.6–266.0, TP4 **146.4**–**287.7**, TP8 148.5–244.5 | **146.4–287.7 ms** |

All 32 rows per job, tokens 1968–1975 × TP {1,2,4,8}, spike always at 0-based timed run 11 (= 15th forward).
Row medians 0.0887–0.1377 ms, so the spike is ≈1,100–3,000× the median.

Assessment of each published figure:
- §0 / `data/profiling_gcfix/RUN.md` **"150–290 ms"** — a fair rounding of the union across both jobs (146.4–287.7).
- Chapter 1 **"150–270 ms"** for job 21313 — **correct** (150.4–268.6).
- Chapter 2 **"150–263 ms"** for job 21334 — **wrong**; the rerun's true range is 146.4–287.7 ms. The quoted upper
  bound appears to be the TP8 figure (263.9) rather than the maximum over TPs.
- My `numbers.md` / `manifest.json` **"150–243 ms"** — correct for **TP1 of job 21313 only**, but presented without
  that qualifier.

---

## U4 — the slide-2 history

**REJECTED as stated.** Three parts of the description do not hold.

**Model identity, verified directly (2026-10-11).** The inherited file's own architecture columns are
n_head 32, n_kv_head 4, n_embd 2048, n_expanded_embd 768, vocab_size 151936, use_gated_mlp true, use_qk_norm true,
attn_output_gate false. These are **identical to job 21313's** and match
`data/config/models/qwen3-a3b-30b-moe.json` (`Qwen3MoeForCausalLM`, num_attention_heads 32, num_key_value_heads 4,
hidden_size 2048, moe_intermediate_size 768, vocab_size 151936). gpt-oss-120b's own profile in this repo
(`data/profiling/compute/mi355x/openai/gpt-oss-120b/linear_op.csv`) differs on **every** one of those fields:
64 / 8 / 2880 / 2880 / 201088. **Iteration 0 is Qwen3-30B-A3B.** The `model_arch` column reads `generic` in both the
inherited file and job 21313 because the profiler did not record an architecture name at the time — a metadata gap,
not a different model.

1. **It is not a gpt-oss sweep.** The inherited file is
   `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/legacy_pre-2026-09_torch-sdpa/linear_op_maxtokens4096.csv` —
   1,032 rows, tokens 1–4,096, `count = 20`, same six linear-op scopes. It is **qwen3-a3b-30b-moe** data. The string
   "gpt-oss" appears nowhere in the live page or in `STORY.md`; it appears in a *different* document,
   `docs/profiling_slides_kit/profiling_project_summary.md`, which covers a broader multi-model project. The repo does
   hold a separate `data/profiling/compute/mi355x/openai/` tree, but the story does not use it.
   What *is* true is that the inherited **attention** data used the portable TORCH_SDPA backend rather than AITER —
   that is the "four model profiling" context, and it is probably the source of the conflation.
2. **The op is attn_pre_proj, not attn_post_proj.** The page's Fig. 1 plots "Median attn_pre_proj time against
   num_tokens, one panel per tensor-parallel degree".
3. **The comparison run is job 21309, not 21313.** Fig. 1's caption: "Blue is our first re-collection on the
   profiler's default **386-token** grid with 50 timed runs per shape (job 21309, 2026-09-15)". Job 21313 is the dense
   3,327-token grid collected later the same day.

The one part that holds: the inherited data is 20 timed repetitions, ≤4,096 tokens, aggregates only, and the decision
was to re-collect at 50 repetitions keeping every run in order.

---

## U5 — did the later iterations change anything?

**REVISED — the answer differs per step, and one step did matter.** Isolating each transition (later ÷ earlier,
pooled over TP, full grid):

| transition | attn_pre_proj | attn_post_proj | attn_rope |
|---|---|---|---|
| 21483 → 21500 (batch-flush fix) | 0.9990 | 0.9992 | 1.0000 |
| 21500 → 21519 (gemm_alloc backlog + settle 3) | **0.9769** | **0.9755** | **0.9845** |
| 21519 → 21539 (settle 8) | 1.0005 | 1.0017 | 1.0000 |

And against the final dataset:

| step ÷ 21539 | attn_pre_proj | attn_post_proj | attn_rope |
|---|---|---|---|
| 21483 | 1.022–1.026 | 1.019–1.030 | 1.012–1.024 |
| 21500 | 1.020–1.026 | 1.016–1.029 | 1.006–1.027 |
| 21519 | 0.999–1.000 | 0.997–0.999 | 0.998–1.004 |

(ranges across TP; max |ratio−1| 0.046–0.164 in all cells, p95 ≤ 0.065.)

**Conclusion.** Of the three post-21483 changes, **only the backlog change (spin → `gemm_alloc`, job 21519) moved the
medians**, by 1.5–2.4 % downward, and it did so for all three ops. The batch-flush barrier removal and the settle-3 →
settle-8 change left medians unchanged to within 0.2 %. This is consistent with the documented purposes — both of
those fixed *run-position* artefacts rather than the median — and it quantifies `STORY.md`'s "medians were taken
mid-ramp, 2–3 % too high" as **≈2.3–2.4 %**, attributable specifically to the backlog change.

The user's starting figures ("it 2 / it 3 within about 2–3 % of final, max |ratio−1| 0.05–0.09") match jobs
21483/21500 under convention C/D; what they miss is that job 21519 is already at 1.000.

---

## Remaining uncertainties (nothing below is settled)

1. **F08's closure arithmetic and the job-21389/21400 trace comparisons** — UNRESOLVED. Evidence exists in the
   historical scratch; the trace aggregation was not reconstructed. Until then, "agree with kernel traces to ≈3 µs" is
   supported only by the eight hard-coded TP2 values I recomputed against (range +1.40…+5.40 µs).
2. **The cause of individual large downward steps** (e.g. −42 % at `attn_post_proj` TP8 12288→12304) — not established
   here. Needs a kernel trace at those shapes.
3. **Why the two `emb` call sites differ**, and why the gap peaks at 4–8k tokens — observed, not explained.
4. **Whether the served checkpoint is block-FP8**, which would change the operator-boundary provider analysis.
5. **The random holdout's independence** — 84.6 % of its shapes were touched under the block geometry in `full_v1`.
6. ~~Four of eight datasets live only in a temporary directory~~ — **resolved 2026-10-11**, see F12.
7. I did not inspect the page's **version history**; I compared only its current rendering against `STORY.md`.

---

## Decisions taken after the verification (2026-10-11)

| Question | Decision |
|---|---|
| Step naming | Human-readable titles; job ids appear in the explaining text, never in a heading. Table below. |
| T2 TP4 | The criterion is mis-specified, not the data — see the mechanism added under F03. Do not silently re-threshold. |
| Dataset location | All ten collections staged to the shared datasets store; nothing to git. See F12. |
| Figure set | The three **sharded** ops only: 18 section-1 figures (6 blocks × 3 ops) and 12 section-2 figures (3 ops × 4 TP). The 9 replicated-op figures (`emb`, two norms, TP1 only) are omitted from the narrative; the `emb` double-invocation finding is covered in text instead. |
| Iteration 0 | Verified Qwen3-30B-A3B from the file's own architecture columns — see U4. |

### Step names for the page

| Title | Job | What it was |
|---|---|---|
| Where we started | inherited | 20 reps, ≤4,096 tokens, aggregates only, TORCH_SDPA-era |
| First re-collection | 21313 | dense 3,327-token grid, 50 reps, every run kept in order |
| The spike, reproduced | 21334 | byte-identical re-run on another node |
| Garbage collection removed | 21361 | `gc.disable()` around the timed loop |
| *Diagnostic: holding the GPU behind the host* | *21356* | *the X1 knob experiment — not a dataset iteration* |
| Timing the GPU, not the host | 21483 | two-column timer; also the first run with the real RoPE kernel |
| The RoPE kernel that was not RoPE | 21483 | same run, separate finding |
| Queue barrier removed | 21500 | `DEBUG_CLR_MAX_BATCH_SIZE`; no effect on medians |
| A real workload as backlog | 21519 | `gemm_alloc`; **the only later step that moved the medians** |
| Longer settling — the dataset we handed over | 21539 | 8 settle forwards; medians unchanged |
