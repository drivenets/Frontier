# Side comparison: RF / XGBoost baselines on the pre-fix MI355X linear_op collection (commit 3d1cabd)

**Standalone side investigation. Not part of `regressor_bench/` results, not an input to the current regressors.**
Nothing in `regressor_bench/`, its sealed splits, or the current `linear_op.csv` pipeline was modified; the main harness
was only read (its `results/baseline_rf_xgb/results.csv`, the random-split train/test token lists, and the current
dataset mirror). **The current dataset's holdout tier was not read or scored.**

Reproduce: `git show 3d1cabd:data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv > data/nuzan_original_linear_op.csv`
(md5 dbc872076ff145d8a518e69673a52a6b), then `python run_sidecar.py` (~8 min on 6 cores). Outputs in `results/`
(git-ignored): `old_split_assignment.csv`, `old_split_audit.csv`, `old_results.csv`, `old_cv_folds.csv`, `old_predictions.csv`, `comparison.csv`.

## 1. What the old file is

| property | old (3d1cabd, pre-fix) | current (2026-09-23 settle8) |
|---|---|---|
| rows × cols | 1,032 × 55 | 13,308 × 150 |
| `num_tokens` grid | **258 values, 1..4096**: 1, 2, 4, 8, then step 8 to 2048 (127 steps), step 16 to ~3072, step 32 to 4096 | 3,327 values, 1..16384 (step 1 to 2048, 8 to 8192, 16 to 16384) |
| TP | 1, 2, 4, 8, every pair once | same |
| ops | same six; replicated ops at TP1 only | same |
| timed samples per shape | 20 (single pass, no settle, no work backlog) | 25 (GPU-bound pass after 8 settle forwards) |
| `time_stats_hostbound.*`, `settle_steps`, `backlog_kind`, `sclk_*`, `samples` | **absent** (verified by column scan) | present |
| label level, attn_pre_proj at 1 token | 52 µs | 11 µs |
| label level, attn_rope | 56-97 µs (13-kernel torch fallback, wrong computation) | 6-96 µs (fused kernel) |

Two consequences before any modelling:

* **None of the five tile-step cliff windows is inside the old grid.** The grid ends at 4096; the windows start at 4352.
  A cliff MAPE is therefore not measurable on this data and is not reported.
* The old label is a different quantity, not a noisier copy: at TP>1 below ~5k tokens it is the host launch span
  (handoff §1, `12_`), and `attn_rope` is a different kernel path. Its per-shape relative std (over 20 samples) is
  **12-30 % for the TP>1 GEMMs and 23-26 % for rope at TP2-8**, versus 1.1-3.5 % (GEMMs) and 2.8-5.9 % (rope) in the
  current file. The old label's own noise exceeds the current model's entire error.

## 2. Method (mirrors `regressor_bench/baseline_rf_xgb.py`)

Feature `num_tokens` only; label `time_stats.<op>.median`. Shape-level split on the old file's own 258 token values,
stratified random draw (same scheme and seed as the main split): **train 194 / test 40 / holdout 24** values, identical
for all 15 (op, TP) pairs; disjointness and coverage audited per pair (`results/old_split_audit.csv`, all 15 pass:
pairwise intersections 0, union = 258). 10 shape-level CV folds on train for the search. RF: full 48-point grid
(n_estimators {200,500,1000} × max_depth {None,10,20,30} × min_samples_leaf {1,2,5,10}). XGBoost: the same 48 sampled
configurations as the main baseline, early stopping on a fixed shape-level 15 % split of the fitting rows, refit at the
best iteration. Winner by mean CV MAPE, refit on train, scored once on test and once on the **old** data's holdout.

**Plateau rule** (applied identically to both datasets, per pair, on the full curve): the flat low-token region is the
prefix of the token grid over which the label stays within +10 % of the median of the 8 smallest shapes, ending where
three consecutive shapes exceed that. Its size is reported, and MAPE is given with and without it.

## 3. Comparison table

MAPE in %. "old" = 3d1cabd file; "current" = settle8 file, numbers from the main baseline (test) plus a read-only refit
of its recorded winners to get the restricted views. The old holdout is this side split's own holdout (24 values);
the current holdout remains sealed.

## Random Forest

| op | TP | old test | old holdout | old plateau (≤ tok, % grid) | old test excl. plateau (n) | old holdout excl. plateau (n) | old label rel. std % | current test | current test ≤4096 tok | current test excl. plateau (plateau % grid) | current label rel. std % | current holdout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| attn_post_proj | 1 | 4.88 | 4.81 | ≤16 (2%) | 3.79 (38) | 4.81 (24) | 3.55 | 1.35 | 1.50 | 1.26 (0%) | 1.55 | sealed, not scored |
| attn_post_proj | 2 | 3.88 | 2.65 | ≤32 (3%) | 2.76 (38) | 2.65 (24) | 19.82 | 1.42 | 1.38 | 1.40 (0%) | 2.03 | sealed, not scored |
| attn_post_proj | 4 | 3.76 | 4.32 | ≤32 (3%) | 2.66 (38) | 4.32 (24) | 22.83 | 1.24 | 1.22 | 1.21 (0%) | 2.62 | sealed, not scored |
| attn_post_proj | 8 | 3.56 | 2.74 | ≤2080 (76%) | 2.93 (10) | 2.36 (5) | 26.43 | 1.32 | 1.22 | 1.30 (0%) | 3.48 | sealed, not scored |
| attn_pre_proj | 1 | 1.72 | 1.88 | ≤672 (34%) | 1.20 (27) | 1.99 (16) | 12.16 | 1.33 | 1.42 | 1.18 (2%) | 1.10 | sealed, not scored |
| attn_pre_proj | 2 | 2.74 | 2.03 | ≤184 (10%) | 2.14 (35) | 2.02 (23) | 27.59 | 1.25 | 1.28 | 1.10 (2%) | 1.24 | sealed, not scored |
| attn_pre_proj | 4 | 2.57 | 1.71 | ≤32 (3%) | 1.93 (38) | 1.71 (24) | 30.38 | 1.20 | 1.08 | 1.08 (0%) | 1.49 | sealed, not scored |
| attn_pre_proj | 8 | 2.52 | 2.74 | ≤128 (7%) | 1.88 (37) | 2.07 (23) | 29.23 | 1.32 | 1.41 | 1.11 (0%) | 1.80 | sealed, not scored |
| attn_rope | 1 | 2.28 | 2.86 | ≤4096 (100%) | – (0) | – (0) | 4.21 | 1.25 | 1.24 | 1.20 (8%) | 2.80 | sealed, not scored |
| attn_rope | 2 | 2.49 | 2.75 | ≤4096 (100%) | – (0) | – (0) | 26.41 | 1.33 | 1.49 | 1.25 (23%) | 3.71 | sealed, not scored |
| attn_rope | 4 | 4.15 | 3.64 | ≤2304 (79%) | 5.31 (8) | 4.96 (5) | 26.12 | 1.02 | 1.00 | 1.14 (41%) | 4.34 | sealed, not scored |
| attn_rope | 8 | 1.99 | 2.63 | ≤232 (12%) | 1.78 (35) | 1.99 (21) | 23.06 | 0.99 | 1.01 | 1.00 (64%) | 5.92 | sealed, not scored |
| emb | 1 | 5.76 | 7.10 | ≤8 (2%) | 5.76 (38) | 7.10 (24) | 26.41 | 1.45 | 1.19 | 1.42 (0%) | 10.38 | sealed, not scored |
| input_layernorm | 1 | 4.44 | 5.00 | ≤1392 (60%) | 3.53 (17) | 2.31 (10) | 15.84 | 1.24 | 1.12 | 1.27 (8%) | 2.82 | sealed, not scored |
| post_attention_layernorm | 1 | 1.84 | 2.84 | ≤1472 (62%) | 1.61 (15) | 1.85 (10) | 6.30 | 1.02 | 1.00 | 1.01 (8%) | 2.71 | sealed, not scored |
| **mean** | | **3.24** | **3.31** | | | | | **1.25** | **1.24** | | | |

## XGBoost

| op | TP | old test | old holdout | old plateau (≤ tok, % grid) | old test excl. plateau (n) | old holdout excl. plateau (n) | old label rel. std % | current test | current test ≤4096 tok | current test excl. plateau (plateau % grid) | current label rel. std % | current holdout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| attn_post_proj | 1 | 5.09 | 4.82 | ≤16 (2%) | 3.96 (38) | 4.82 (24) | 3.55 | 1.82 | 1.90 | 1.67 (0%) | 1.55 | sealed, not scored |
| attn_post_proj | 2 | 4.09 | 2.82 | ≤32 (3%) | 2.80 (38) | 2.82 (24) | 19.82 | 2.05 | 1.89 | 1.86 (0%) | 2.03 | sealed, not scored |
| attn_post_proj | 4 | 3.86 | 3.13 | ≤32 (3%) | 2.69 (38) | 3.13 (24) | 22.83 | 2.01 | 1.68 | 1.90 (0%) | 2.62 | sealed, not scored |
| attn_post_proj | 8 | 3.24 | 2.73 | ≤2080 (76%) | 2.57 (10) | 2.46 (5) | 26.43 | 2.42 | 2.07 | 2.36 (0%) | 3.48 | sealed, not scored |
| attn_pre_proj | 1 | 1.74 | 1.62 | ≤672 (34%) | 1.32 (27) | 1.80 (16) | 12.16 | 1.61 | 1.68 | 1.44 (2%) | 1.10 | sealed, not scored |
| attn_pre_proj | 2 | 2.87 | 1.81 | ≤184 (10%) | 2.24 (35) | 1.82 (23) | 27.59 | 1.46 | 1.48 | 1.26 (2%) | 1.24 | sealed, not scored |
| attn_pre_proj | 4 | 2.49 | 1.72 | ≤32 (3%) | 1.77 (38) | 1.72 (24) | 30.38 | 1.49 | 1.29 | 1.28 (0%) | 1.49 | sealed, not scored |
| attn_pre_proj | 8 | 2.46 | 2.25 | ≤128 (7%) | 1.69 (37) | 1.96 (23) | 29.23 | 1.43 | 1.46 | 1.20 (0%) | 1.80 | sealed, not scored |
| attn_rope | 1 | 2.19 | 2.65 | ≤4096 (100%) | – (0) | – (0) | 4.21 | 1.37 | 1.36 | 1.27 (8%) | 2.80 | sealed, not scored |
| attn_rope | 2 | 2.57 | 2.64 | ≤4096 (100%) | – (0) | – (0) | 26.41 | 1.35 | 1.46 | 1.30 (23%) | 3.71 | sealed, not scored |
| attn_rope | 4 | 3.92 | 3.59 | ≤2304 (79%) | 5.54 (8) | 4.90 (5) | 26.12 | 1.05 | 1.04 | 1.12 (41%) | 4.34 | sealed, not scored |
| attn_rope | 8 | 2.47 | 3.06 | ≤232 (12%) | 2.16 (35) | 2.52 (21) | 23.06 | 1.04 | 1.05 | 1.09 (64%) | 5.92 | sealed, not scored |
| emb | 1 | 5.65 | 6.31 | ≤8 (2%) | 5.53 (38) | 6.31 (24) | 26.41 | 1.56 | 1.36 | 1.48 (0%) | 10.38 | sealed, not scored |
| input_layernorm | 1 | 4.50 | 4.91 | ≤1392 (60%) | 3.58 (17) | 2.13 (10) | 15.84 | 1.58 | 1.40 | 1.47 (8%) | 2.82 | sealed, not scored |
| post_attention_layernorm | 1 | 2.50 | 2.97 | ≤1472 (62%) | 2.35 (15) | 2.14 (10) | 6.30 | 1.40 | 1.39 | 1.26 (8%) | 2.71 | sealed, not scored |
| **mean** | | **3.31** | **3.13** | | | | | **1.58** | **1.50** | | | |

## 4. Answer: is the old data easier to predict?

**No. It is harder on every one of the 30 (op, TP, model) cells.** Mean test MAPE over the 15 pairs is 3.24 % (RF) /
3.31 % (XGB) on the old file against 1.25 % / 1.58 % on the current file, and restricting the current test set to
≤ 4096 tokens (the old grid's range) changes nothing (1.24 % / 1.50 %). The old holdout agrees with the old test (3.31 % /
3.13 %), so this is not a test-draw artefact.

What does *not* explain the difference:

* **Grid range.** Same conclusion on ≤ 4096 tokens.
* **The plateau.** Excluding the flat low-token region lowers the old MAPE by 0-1.2 points where the plateau is small
  (GEMMs at TP1-4) and leaves the current numbers essentially unchanged; the gap persists.

What does explain it: **label noise.** The old per-shape relative std is 12-30 % on the TP>1 GEMMs and rope, i.e. the
median of 20 single-pass, host-bound samples is itself uncertain by several percent, and no regressor on `num_tokens`
can predict that scatter. The current file's 1-4 % relative std (0.3-1.5 % standard error of the median) is what makes
1.2 % MAPE reachable.

Where the old data *looks* easy, it is an artefact, not signal:

* **`attn_rope` at TP1 and TP2: the whole curve is plateau (100 % of the grid).** The old rope label is the launch-bound
  torch fallback (13 kernels, flat in tokens); predicting a constant is trivial, yet the MAPE is still 2.2-2.9 % because
  of the 4-26 % sample noise. There is no ramp to fit and nothing to learn about the real kernel.
* **`attn_post_proj` TP8 (76 % plateau), `attn_rope` TP4 (79 %), the two norms (60-62 %).** The flat region is the
  host-bound launch span. Only 5-17 test/holdout points remain outside it, so the "excl. plateau" numbers for these
  pairs are means over a handful of points and should not be over-read (e.g. rope TP4 goes *up* to 5.3-5.5 % outside
  the plateau: the ramp part of the old curve is the hardest part).
* **Coarseness.** 40 test / 24 holdout values per pair make every old MAPE a small-sample mean; test and holdout differ
  by up to 1.6 points on the same pair (attn_post_proj TP2: 3.88 vs 2.65). Treat old numbers as ±1 point.
* **Cliffs.** Not measurable on the old grid (§1). The current data's cliff behaviour has no counterpart here.

Bottom line for the comparison: the current dataset is the easier *and* the more meaningful target. Any impression
that the pre-fix collection was easier would come from its flat, host-bound or wrong-kernel curves, and even those
carry 2-6 % MAPE because the single-pass label is noisy.
