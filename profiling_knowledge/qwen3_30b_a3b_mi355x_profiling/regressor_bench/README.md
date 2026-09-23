# Regressor benchmark for the Qwen3-30B-A3B / MI355X linear_op dataset

A harness that answers one question with evidence instead of taste: **which regressor family should predict
`time_stats.<op>.median` from `num_tokens`, per (op, TP), for the 15 regressors the simulator needs?**
Input contract: `../14_regressor_dataset_handoff.md`. This document explains the protocol and why each step is
there; `results/<run>/summary.md` holds the numbers.

## Quick start

```bash
source /home/dn/.virtualenvs/qwen3-profiling/bin/activate      # numpy, pandas, scikit-learn, scipy, lightgbm, matplotlib, pytest
cd <repo root>
B=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench

# 0. data: mirror the shared-store file (md5 3635e4d305ae5cf84884e275d3ff37ef) to the git-ignored local path
#    data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv
python -m pytest $B/tests -q                                   # invariants of splits, metrics, every zoo member
python $B/bench.py make-splits                                 # once; refuses to overwrite a sealed split
python $B/bench.py run --out $B/results/full_v1                # ~1 h on 6 cores; --quick for a 1-minute smoke
python $B/bench.py report --results $B/results/full_v1         # summary.md + figs/
python $B/baseline_rf_xgb.py                                   # RF + XGBoost baselines, train/test only (~25 min)
python $B/build_eval_notebook.py --execute                     # EVAL_baseline_rf_xgb.ipynb: tails (p95/p99/max), where, in µs
python $B/bench.py holdout --results $B/results/full_v1 --unseal --models <the 1-2 finalists>   # once, only when the user says so
```

## The protocol, and the reason for each step

### 1. Decide what one sample is before splitting

One sample is one **(op, TP, num_tokens)** cell with its median over 25 timed forwards (`dataset.tidy`). The
25 raw samples are *replicates of the same x*, not independent observations: training on them as rows and
scoring on other replicates of the same shape would be a leak that flatters every model. The label noise that
remains (relative std of the 25 samples: median 1.4 % for `attn_pre_proj`, 2.3 % `attn_post_proj`, 4 % `attn_rope`;
standard error of the median about a quarter of that) is the floor below which MAPE differences are not real.

Features: `num_tokens` only. Every other column is a model constant or instrument metadata and is barred
(`dataset.FORBIDDEN_FEATURE_PREFIXES`). Per-(op, TP) fits mirror how the simulator consumes the model.

### 2. Split on the token value, three tiers, sealed

`splits.py` assigns each of the 3,327 token values to **train / test / holdout** (75 / 15 / 10 %), the same
assignment for all 15 regressors so comparisons are paired. The holdout file is written once with its sha256 in
`splits/manifest.json`; `run` never opens it, `make-splits` refuses to overwrite a sealed directory, and
`holdout --unseal` appends every use to `splits/HOLDOUT_LOG.md`. Test is used once per model for the reported
numbers; all tuning happens inside train with cross-validation.

Two **geometries** from the same seed, because the honest question depends on how the model is used:

| geometry | what is held out | what it measures |
|---|---|---|
| `random` | token values sampled at random, stratified by log2(tokens) | interpolation between immediate neighbours on the dense grid: the deployment case if the simulator only ever asks for measured shapes |
| `block` | whole contiguous bands of 10 % relative width (1000-1100, 8000-8800, ...) | filling a gap in the grid: a sparser profiling grid, or a query between two measured shapes. Harder, and the one that separates methods |

Relative band width keeps the gap equally hard at 100 and 10,000 tokens (a fixed count of grid points would be a
2x gap at 64 tokens and a 3 % gap at 8k). Stratification guarantees every size regime lands in every tier.
Within-train CV folds mirror the geometry (`splits.cv_folds`).

### 3. Baselines that are hard to beat

`interp` (piecewise-linear interpolation of the training grid) and `isotonic`. On a dense grid these are what
any smooth model is implicitly competing with. A candidate that does not beat `interp` on the block geometry
is not worth deploying, whatever its random-split score. `rf_frontier` and `polylr_frontier` reproduce the
simulator's two production configurations with their exact grids, so "what we ship today" is a row in the table.

### 4. Candidates (`models.zoo`)

| family | models | why |
|---|---|---|
| baseline | `interp`, `isotonic` | what to beat |
| neighbours | `knn` | local averaging with tunable smoothing |
| trees | `rf_frontier` (simulator grid), `extratrees` | piecewise-constant, capture the hipBLASLt tile steps |
| boosting | `hgb`, `hgb_log`, `lgbm` | piecewise-constant with better bias/variance trade |
| polynomial | `polylr_frontier` (simulator grid, raw tokens), `poly_ridge`, `poly_ridge_log` | global smooth, cheap |
| spline | `spline_lin(_log)`, `spline_cub(_log)` | local smooth with quantile knots; linear extrapolation |
| physical | `roofline` | `c + b*max(0, n-n0)`, three parameters, relative residuals |
| hybrid | `linear_plus_hgb` | linear trend (extrapolates) + trees on the residual (steps) |

`_log` variants fit `log(y)` and predict `exp`, which makes the squared-error objective behave like a relative
error, the metric we care about. Every transform sits inside an sklearn pipeline, so it is fit on training folds
only. The Frontier polynomial is left unscaled on purpose: that is how it is shipped.

### 5. Selection, assessment, test: three separate uses of the data

For each (geometry, op, TP, model), `bench.py run`:

1. **selection** – `GridSearchCV` over the model's grid on train, 5 geometry-matched folds, scoring = the
   simulator's MAPE. Result: `best_params` (`cv_selection.csv`).
2. **assessment** – repeated 5-fold on train with those params, 3 fresh fold seeds, identical folds for every
   model, so differences are paired (`cv_assessment.csv`, mean ± std).
3. **test** – fit on all of train, score the test tier once (`test_metrics.csv`, per regime in
   `test_metrics_by_regime.csv`, predictions in `test_predictions.csv`).

Selecting hyper-parameters on the same folds that produce the headline CV number would bias it; the separate
assessment pass with new seeds avoids that. A CV/test disagreement in the summary is the signal of a test-set
fluke or a selection leak.

### 6. Metrics on the ms scale, distribution not just mean

MAPE (for continuity with the simulator), median APE, **p95 APE**, max APE, fraction of points above 5 %,
signed bias, MAE and RMSE in ms, all broken down by token regime (1-64, 65-512, 513-2048, 2049-8192,
8193-16384). Errors are heteroscedastic: labels under ~10 µs carry a ±3-5 µs instrument floor, so a 30 % APE at
8 tokens and a 3 % APE at 8k tokens can be the same 3 µs. The regime table keeps that visible. The leaderboard
ranks by mean MAPE but shows the worst regressor: a model that is 1 % on average and 40 % on one op is not
deployable.

### 7. Things the data says you must not do

* Do not smooth away the 15-30 % steps at ~4.3k-5.4k and ~12k tokens at TP>1; they are hipBLASLt tile
  selection, real, and a good model reproduces them (handoff §5).
* Do not read sub-10 µs labels as precise.
* Do not compare numbers here with models fitted on the 200-forward file of 2026-09-22 without expecting its
  2-3 % clock-ramp offset.

### 8. Reproducibility

Fixed seeds everywhere; split assignments as CSV with sha256 in the manifest; input md5 verified on every load;
library versions and wall time in `run_meta.json`; one command regenerates each stage. Results directories are
git-ignored except the one promoted to a decision record.

## Layout

```
regressor_bench/
├── README.md          this file
├── dataset.py         load + md5 check + tidy per-regressor table
├── splits.py          strata, random/block geometries, sealed I/O, geometry-matched CV folds
├── models.py          the zoo (ModelSpec, custom estimators)
├── metrics.py         Frontier MAPE, summary, regime breakdown
├── bench.py           CLI: make-splits | run | holdout | report
├── report.py          leaderboard, per-regressor table, regime table, figures
├── tests/             pytest: split invariants, metric definition, every zoo member fits
├── splits/            sealed splits (manifest.json, <geometry>/{train,test,holdout}_tokens.csv, HOLDOUT_LOG.md)
└── results/<run>/     cv_selection.csv, cv_assessment.csv, test_metrics*.csv, test_predictions.csv, run_meta.json, summary.md, figs/
```
