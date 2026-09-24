#!/usr/bin/env python3
"""Build (and optionally execute) EVAL_baseline_rf_xgb.ipynb: tail-aware evaluation of the baseline v2 regressors.

    python build_eval_notebook.py            # writes the notebook
    python build_eval_notebook.py --execute  # also runs it in place (needs nbconvert + ipykernel in the venv)

Reads results/baseline_v2 (baseline_v2.py: RF, XGBoost with exact splits, linear interpolation, two-collection noise
floor). The earlier histogram-XGBoost run in results/baseline_rf_xgb is not used: diagnose_xgb.py traced its tail
deficit to the 256-bin histogram, and baseline_v2 replaced it.

The notebook reads only the train and test tiers of the sealed random-geometry split. It never opens the holdout.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "EVAL_baseline_rf_xgb.ipynb"

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(new_code_cell(s.strip("\n")))  # noqa: E731

md(r"""
# Baseline v2 regressors on the linear_op dataset: MAPE is not enough

Per (op, TP) Random Forest, XGBoost (`tree_method="exact"`) and piecewise-linear interpolation baselines
(`baseline_v2.py`, sealed random-geometry shape-level split, feature `num_tokens`, label `time_stats.<op>.median`).
This notebook looks past the mean error at the **tails**: median, p95, p99 and max absolute percentage error, where in
`num_tokens` they occur, and what they are in microseconds.

**Reference: the measurement noise floor.** Every test row also carries the median of a second, independent collection
(job 21519, settle 3) next to the training label (job 21539, settle 8). The error of one collection predicting the other is
plotted in green wherever the models are compared. A model whose tail sits on the green curve is limited by the
measurement, not by the model.

**Tiers used:** train (fit, in `baseline_v2.py`) and test (all numbers). The holdout is never read here.

**How to read the four views per op**

1. **ECDF of |error| %** (one panel per TP, three models + noise floor). Every quantile is on the curve: read p95 / p99
   where the dashed lines cross, the max is the right end. Curves compare models at *every* quantile, not one number.
2. **Signed error vs `num_tokens`.** Where the tail lives: hipBLASLt tile-step windows are shaded, sub-10 µs labels
   (instrument floor, ±3-5 µs) are hollow markers. A bias shows as a one-sided cloud.
3. **Absolute error in µs vs `num_tokens`.** The simulator sums per-layer times, so a 10 % miss on a 7 µs label
   (0.7 µs) and a 3 % miss at 16k tokens (15 µs) are not the same problem. Relative and absolute tails rank pairs differently.
4. **Measured vs predicted time.** Green is the label, blue the RF prediction, red the XGBoost prediction, on the same rows.

**Cautions.** With 499 test points p99 is roughly the 5th-worst point: real, but noisy. The two-collection noise floor
is 0.9-1.5 % MAPE per regressor (≈ 0.6-1.1 % for a single collection), so errors at that level are measurement, not model.

**Superseded run.** The first XGBoost baseline (`results/baseline_rf_xgb`, histogram tree method) had p99 tails of 12-19 %
on attn_post_proj. `diagnose_xgb.py` traced that to the 256-bin histogram quantising `num_tokens`; it is not shown here.
""")

code(r"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = Path.cwd()
while not (HERE / "splits" / "manifest.json").exists():   # allow launching from the repo root
    cand = [p for p in HERE.rglob("regressor_bench/splits/manifest.json")]
    if cand: HERE = cand[0].parents[1]; break
    HERE = HERE.parent
sys.path.insert(0, str(HERE.parent))
from regressor_bench.splits import load_split
from regressor_bench.metrics import ape, INSTRUMENT_FLOOR_MS
from regressor_bench.baseline_rf_xgb import CLIFFS, in_cliff
from regressor_bench.report import SERIES, INK, INK2, MUTED, GRID, BASELINE, SURFACE
%matplotlib inline
# (after the report import: that module selects the Agg backend for file output; the magic switches back)

RESULTS = HERE / "results" / "baseline_v2"
MODELS = {"rf": ("Random Forest", SERIES[0]), "xgb_exact": ("XGBoost (exact)", SERIES[1]), "interp": ("linear interp", SERIES[6])}
NOISE = ("noise floor (21519 vs 21539)", SERIES[2])       # green: second collection predicting the first
REFS = {**MODELS, "noise": NOISE}                          # models + noise, for the comparison views
OPS = ["attn_pre_proj", "attn_post_proj", "attn_rope", "input_layernorm", "post_attention_layernorm", "emb"]
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
meta = json.loads((RESULTS / "run_meta.json").read_text())
print("package:", HERE); print("results:", RESULTS)
print("train csv:", meta["csv"], meta["csv_md5"]); print("noise csv:", meta["noise_csv"], meta["noise_csv_md5"])
print("xgb tree_method:", meta["xgb"]["tree_method"], "| finished:", meta["finished_utc"])
""")

code(r"""
split = load_split(HERE / "splits", "random", include_holdout=False)     # train + test token lists only
assert set(split) == {"train", "test"}
results = pd.read_csv(RESULTS / "results.csv")
print(f"train tokens {len(split['train'])}, test tokens {len(split['test'])}, "
      f"regressors {results[['op','tp']].drop_duplicates().shape[0]}, models {sorted(results.model.unique())}")
""")

md(r"""
## Per-row test predictions

`baseline_v2.py` saves `test_predictions.csv`: one row per (op, TP, model, test token) with the label `y` (job 21539), the
second collection's median `y_other_collection` (job 21519) and the prediction. The per-model MAPE recomputed from these
rows is asserted against `results.csv` before anything is plotted. The noise floor is added as a fourth pseudo-model
whose "prediction" is the other collection's median.
""")

code(r"""
saved = RESULTS / "test_predictions.csv"
assert saved.exists(), f"{saved} missing: run baseline_v2.py first"
preds = pd.read_csv(saved)
assert set(preds.tier) == {"test"}
preds = preds.drop(columns="tier")
noise = preds[preds.model == "rf"].assign(model="noise", pred=lambda d: d.y_other_collection)
preds = pd.concat([preds, noise], ignore_index=True)

preds["ape"] = ape(preds.y, preds.pred)
preds["signed_pct"] = (preds.pred - preds.y) / preds.y * 100
preds["abs_err_us"] = (preds.pred - preds.y).abs() * 1e3
preds["cliff"] = in_cliff(preds.num_tokens.to_numpy())
preds["sub10us"] = preds.y < INSTRUMENT_FLOOR_MS

chk = preds[preds.model != "noise"].groupby(["op", "tp", "model"]).ape.mean().rename("row_mape").reset_index() \
    .merge(results[["op", "tp", "model", "test_mape"]])
chk["diff"] = (chk.row_mape - chk.test_mape).abs()
assert len(chk) == 45 and chk["diff"].max() < 1e-6, chk.sort_values("diff", ascending=False).head()
nchk = preds[preds.model == "noise"].groupby(["op", "tp"]).ape.mean().rename("row_noise").reset_index() \
    .merge(results[results.model == "rf"][["op", "tp", "noise_floor_mape"]])
assert (nchk.row_noise - nchk.noise_floor_mape).abs().max() < 1e-6
print("per-row MAPE reproduces results.csv for all 45 (op, TP, model) and the 15 noise floors "
      "(max |diff| = %.2e)" % max(chk["diff"].max(), (nchk.row_noise - nchk.noise_floor_mape).abs().max()))
print(preds.groupby("model").size().rename("test rows"))
""")

md(r"""
## Tail table

`argmax_tokens` is the `num_tokens` of the worst test point; `max_abs_us` is the worst absolute miss in microseconds.
The `noise` column is the second collection scored against the first on the same rows.
""")

code(r"""
def tails(g):
    i = g.ape.idxmax()
    return pd.Series({"n": len(g), "MAPE": g.ape.mean(), "MdAPE": g.ape.median(), "p95": g.ape.quantile(.95),
                      "p99": g.ape.quantile(.99), "max": g.ape.max(), "argmax_tokens": int(g.loc[i, "num_tokens"]),
                      "y_at_max_us": g.loc[i, "y"] * 1e3, "bias_%": g.signed_pct.mean(),
                      "p95_abs_us": g.abs_err_us.quantile(.95), "max_abs_us": g.abs_err_us.max()})

tab = preds.groupby(["op", "tp", "model"]).apply(tails).reset_index()
tab["n"] = tab.n.astype(int)
tab["model"] = pd.Categorical(tab.model, categories=list(REFS))
show = tab.pivot(index=["op", "tp"], columns="model", values=["MAPE", "p95", "p99", "max", "argmax_tokens", "max_abs_us"])
show = show.reindex(columns=["MAPE", "p95", "p99", "max", "argmax_tokens", "max_abs_us"], level=0)
show.round(2).style.background_gradient(subset=[c for c in show.columns if c[0] in ("MAPE", "p95", "p99", "max")], cmap="Blues", axis=None)
""")

code(r"""
# same table, long form, sortable
tab.sort_values(["model", "p99"], ascending=[True, False]).round(2)
""")

md(r"""
### What the table says (baseline v2, 2026-09-24)

* **The max error is still one point: `num_tokens = 1`.** The random draw put the grid's minimum into the test tier, so
  for the four attn_pre_proj regressors the worst miss is 52-94 % at 1 token, for all three models. The label there is a
  different kernel regime (attn_pre_proj TP8: 10.9 µs at 1 token vs 18.5 at 4 and ~22 from 5 on) and 1 is below the
  training range, so every model predicts the 4-8-token plateau. That is extrapolation off the grid's edge, not a tail of
  the interpolation error. For the simulator it matters only if decode steps of exactly one token per forward are
  predicted; otherwise read `p99`, not `max`.
* **Excluding that point, the two tree models have the same tail: p99 is 3.5-7.2 % for RF and 3.5-6.9 % for XGBoost
  (exact).** The 12-19 % attn_post_proj tails of the first XGBoost run are gone; they were the histogram binning, not the
  data. Linear interpolation is worse at the tail on every pair (p99 3.8-10.8 %, worst on attn_post_proj TP8), which is
  the price of following each measured point exactly.
* **The tails are close to the noise floor.** The two-collection noise has p95 of 2.2-4.1 % and p99 of 2.9-6.8 % on the
  same rows; the tree models' p95 is 2.5-4.0 % and their p99 3.5-7.2 %. Roughly one point of p99 above the
  measurement-vs-measurement tail is what is left for the model to lose.
* **XGBoost's remaining outliers sit on the second grid point.** Its worst non-1-token misses are 34 % and 25 % at
  `num_tokens = 4` on attn_post_proj TP4 and TP2 (labels 6-8 µs, below the instrument floor). RF's are 18 % at 128 tokens
  (attn_post_proj TP8) and 17 % at 18 tokens (emb), also sub-12 µs labels.
* **In microseconds the tails are small.** p95 absolute error is 0.2-7 µs for every model; the largest single miss is 23 µs
  (RF, attn_pre_proj TP1, the 1-token point). The relative tail is dominated by small labels near the instrument floor.
* **Bias is within ±0.3 % everywhere**, so the errors are scatter, not a systematic offset. XGBoost is consistently slightly
  positive (+0.02 to +0.28 %).
""")

md(r"""
## View 1: ECDF of |error| %, per op (panels = TP)

Dashed verticals mark p95 and p99 of each series; the curve's right end is the max. The green curve is the noise floor:
the second collection scored against the training labels on the same rows. Log x-axis so the 1 % body and the 10 %+
tail are both legible.
""")

code(r"""
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8); ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)

def ecdf_panel(ax, g, label, color, ref=False):
    x = np.sort(g.ape.to_numpy()); y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", lw=1.4 if ref else 1.6, color=color, label=label, ls="-" if not ref else (0, (3, 1.5)))
    for q, ls in ((.95, "--"), (.99, ":")):
        ax.axvline(np.quantile(x, q), color=color, lw=.9, ls=ls, alpha=.8)
    ax.scatter([x[-1]], [1.0], s=22, color=color, zorder=3)

def ecdf_op(op):
    tps = sorted(preds[preds.op == op].tp.unique())
    fig, axes = plt.subplots(1, len(tps), figsize=(4.2 * len(tps), 3.4), dpi=120, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, tp in zip(axes[0], tps):
        style(ax)
        for m, (name, color) in REFS.items():
            g = preds[(preds.op == op) & (preds.tp == tp) & (preds.model == m)]
            ecdf_panel(ax, g, name, color, ref=(m == "noise"))
        ax.set_xscale("log"); ax.set_xlim(0.01, 100); ax.set_ylim(0, 1.02)
        ax.set_title(f"TP{tp}", loc="left", fontsize=9, color=INK); ax.set_xlabel("|error| % (log)", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("fraction of test points", fontsize=8, color=INK2)
    axes[0][-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(f"{op}: ECDF of |error| on the test tier  (dashed = p95, dotted = p99, dot = max; green = noise floor)", x=.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); plt.show()

for op in OPS:
    ecdf_op(op)
""")

md(r"""
## View 2: signed error vs `num_tokens` (where the tail lives)

Shaded bands are the known hipBLASLt tile-step windows (4352-4360, 4480-4488, 4864-4872, 5376-5384, 11800-12300).
Hollow markers are rows whose label is below 10 µs, where ±3-5 µs of event-record cost is a large relative error by itself.
Note the random draw left **no test values in the four narrow windows**; only the 12k window has test rows.
""")

code(r"""
def signed_op(op):
    tps = sorted(preds[preds.op == op].tp.unique())
    fig, axes = plt.subplots(1, len(tps), figsize=(4.2 * len(tps), 3.4), dpi=120, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, tp in zip(axes[0], tps):
        style(ax)
        for lo, hi in CLIFFS: ax.axvspan(lo, hi, color=GRID, alpha=.9, lw=0)
        for m, (name, color) in MODELS.items():
            g = preds[(preds.op == op) & (preds.tp == tp) & (preds.model == m)]
            solid, hollow = g[~g.sub10us], g[g.sub10us]
            ax.scatter(solid.num_tokens, solid.signed_pct, s=9, color=color, alpha=.75, lw=0, label=name)
            ax.scatter(hollow.num_tokens, hollow.signed_pct, s=12, facecolors="none", edgecolors=color, lw=.7, alpha=.8)
        ax.axhline(0, color=BASELINE, lw=.8); ax.set_xscale("log")
        ax.set_title(f"TP{tp}", loc="left", fontsize=9, color=INK); ax.set_xlabel("num_tokens", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("signed error %  (pred - y)/y", fontsize=8, color=INK2)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", markerscale=2)
    fig.suptitle(f"{op}: signed test error vs num_tokens  (hollow = label < 10 µs, shaded = tile-step windows)", x=.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); plt.show()

for op in OPS:
    signed_op(op)
""")

md(r"""
## View 3: absolute error in µs vs `num_tokens`

Same points, y-axis in microseconds on a linear scale (the x-axis stays log, it is `num_tokens`). This is the error
that propagates into a layer-summed forward time. Panels share the y-axis per op, so the largest miss sets the range.
""")

code(r"""
def abs_op(op):
    tps = sorted(preds[preds.op == op].tp.unique())
    fig, axes = plt.subplots(1, len(tps), figsize=(4.2 * len(tps), 3.4), dpi=120, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, tp in zip(axes[0], tps):
        style(ax)
        for lo, hi in CLIFFS: ax.axvspan(lo, hi, color=GRID, alpha=.9, lw=0)
        for m, (name, color) in MODELS.items():
            g = preds[(preds.op == op) & (preds.tp == tp) & (preds.model == m)]
            ax.scatter(g.num_tokens, g.abs_err_us, s=9, color=color, alpha=.75, lw=0, label=name)
        ax.set_xscale("log"); ax.set_ylim(bottom=0)
        ax.set_title(f"TP{tp}", loc="left", fontsize=9, color=INK); ax.set_xlabel("num_tokens", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("|pred - y|  (µs)", fontsize=8, color=INK2)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", markerscale=2)
    fig.suptitle(f"{op}: absolute test error vs num_tokens", x=.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); plt.show()

for op in OPS:
    abs_op(op)
""")

md(r"""
## View 4: measured vs predicted time, per op (panels = TP)

Green dots are the measured test labels (`time_stats.<op>.median`, µs, job 21539). Blue is the Random Forest prediction
and red the XGBoost (exact) prediction for the same test rows, so a miss shows as a coloured dot sitting off its green
dot. Linear interpolation is left out of this view to keep the three-colour reading. Both axes are linear; `num_tokens`
is log because the grid is.
""")

code(r"""
MEASURED = SERIES[2]   # green
FIT_MODELS = {k: MODELS[k] for k in ("rf", "xgb_exact")}

def fit_op(op):
    tps = sorted(preds[preds.op == op].tp.unique())
    fig, axes = plt.subplots(1, len(tps), figsize=(4.2 * len(tps), 3.4), dpi=120, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, tp in zip(axes[0], tps):
        style(ax)
        for lo, hi in CLIFFS: ax.axvspan(lo, hi, color=GRID, alpha=.9, lw=0)
        g0 = preds[(preds.op == op) & (preds.tp == tp) & (preds.model == "rf")]      # labels are identical across models
        ax.scatter(g0.num_tokens, g0.y * 1e3, s=26, color=MEASURED, alpha=.9, lw=0, label="measured", zorder=2)
        for m, (name, color) in FIT_MODELS.items():
            g = preds[(preds.op == op) & (preds.tp == tp) & (preds.model == m)]
            ax.scatter(g.num_tokens, g.pred * 1e3, s=7, color=color, alpha=.9, lw=0, label=name, zorder=3)
        ax.set_xscale("log"); ax.set_ylim(bottom=0)
        ax.set_title(f"TP{tp}", loc="left", fontsize=9, color=INK); ax.set_xlabel("num_tokens", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("time (µs)", fontsize=8, color=INK2)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", markerscale=1.5)
    fig.suptitle(f"{op}: measured (green) vs predicted (blue = RF, red = XGBoost exact), test tier", x=.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); plt.show()

for op in OPS:
    fit_op(op)
""")

md(r"""
## Worst points

The ten largest relative misses per model, with the label so the µs size of the miss is visible, and the second
collection's value (`y2_us`) so a measurement artefact shows as a label that the other collection does not reproduce.
""")

code(r"""
cols = ["model", "op", "tp", "num_tokens", "y_us", "y2_us", "pred_us", "ape", "abs_err_us", "cliff", "sub10us"]
w = preds[preds.model != "noise"].assign(y_us=lambda d: d.y * 1e3, y2_us=lambda d: d.y_other_collection * 1e3, pred_us=lambda d: d.pred * 1e3)
pd.concat([w[w.model == m].nlargest(10, "ape") for m in MODELS])[cols].round(2).reset_index(drop=True)
""")

md(r"""
## One-glance comparison: p95 / p99 / max across all 15 regressors

Dot plots on a log axis, one row per (op, TP). The three models and the noise floor sit side by side, so a tail
regression on one pair is visible without scanning the table.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), dpi=120, sharey=True)
fig.patch.set_facecolor(SURFACE)
order = tab[tab.model == "rf"].sort_values(["op", "tp"]).apply(lambda r: f"{r.op}/TP{r.tp}", axis=1).tolist()
off = np.linspace(-.3, .3, len(REFS))
for ax, metric in zip(axes, ["p95", "p99", "max"]):
    style(ax); ax.set_xscale("log")
    for k, (m, (name, color)) in enumerate(REFS.items()):
        d = tab[tab.model == m].assign(key=lambda d: d.op + "/TP" + d.tp.astype(str)).set_index("key").loc[order]
        ax.scatter(d[metric], np.arange(len(order)) + off[k], s=26 if m != "noise" else 18, color=color, label=name, zorder=3,
                   marker="o" if m != "noise" else "D")
    ax.set_title(f"{metric} |error| %", loc="left", fontsize=10, color=INK)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=8, color=INK)
axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); plt.show()
""")

md(r"""
## Notes

* Numbers here are the test tier of the sealed random-geometry split (interpolation between neighbouring grid
  points). The block-geometry results in `RESULTS_full_v1.md` are the gap-filling case and are 1-2 points worse.
* `p99` on 499 points is the 5th-worst point; compare it with the max, not with p95, when judging stability.
* Summary tables for this run (CV, cliff rows, XGBoost winners, detected steps) are in `RESULTS_baseline_v2.md`.
* To rebuild this notebook after a new run: `python build_eval_notebook.py --execute`.
""")

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "qwen3-profiling", "display_name": "qwen3-profiling", "language": "python"},
                                         "language_info": {"name": "python"}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    nbformat.write(nb, OUT)
    print("wrote", OUT)
    if a.execute:
        cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               "--ExecutePreprocessor.timeout=1200", "--ExecutePreprocessor.kernel_name=qwen3-profiling", str(OUT)]
        return subprocess.call(cmd, cwd=HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
