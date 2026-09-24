"""Diagnostics for the RF-vs-XGBoost gap in baseline_rf_xgb (train + test tiers only; the holdout is never read).

Answers, with numbers written to results/diagnose_xgb/:
  A. configuration facts: tree_method, max_bin, actual histogram cut points around the cliff windows and per grid region,
     objective / target scale / metrics, best_iteration vs cap, hyper-parameters at the edge of their range;
  B. underfit or overfit: train-tier MAPE of RF and XGB side by side; per-pair gap; residual/overlay figures for the
     worst pairs;
  C. fairness and sanity: full-train refit (already the case in the baseline; verified here), interpolation reference on
     the same test tier, CV-vs-test agreement, test MAPE spread across model seeds, and the binning experiment
     (max_bin 256 -> 1024 -> exact) scored by 10-fold CV on train first, test second (exploratory, flagged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from xgboost import QuantileDMatrix, XGBRegressor  # noqa: E402

from regressor_bench.baseline_rf_xgb import CLIFFS, ES_FRAC, ES_ROUNDS, ES_SEED, FOLD_SEED, N_FOLDS, XGB_SPACE, fit_rf, in_cliff  # noqa: E402
from regressor_bench.dataset import DEFAULT_CSV, load_raw, regressor_keys, tidy  # noqa: E402
from regressor_bench.metrics import mape  # noqa: E402
from regressor_bench.models import InterpRegressor  # noqa: E402
from regressor_bench.report import BASELINE, GRID, INK, INK2, MUTED, SERIES, SURFACE  # noqa: E402
from regressor_bench.splits import cv_folds, load_split  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "baseline_rf_xgb"
OUT = HERE / "results" / "diagnose_xgb"
FIGS = OUT / "figs"
RF_SPACE = {"n_estimators": [200, 500, 1000], "max_depth": [None, 10, 20, 30], "min_samples_leaf": [1, 2, 5, 10]}


def subset(t, op, tp, tokens):
    g = t[(t.op == op) & (t.tp == tp) & t.num_tokens.isin(tokens)].sort_values("num_tokens")
    return g[["num_tokens"]].to_numpy(float), g.y.to_numpy(float), g.num_tokens.to_numpy()


def make_xgb(params, n_estimators, seed=0, **extra):
    p = {k: v for k, v in params.items() if k != "n_estimators"}
    return XGBRegressor(objective="reg:squarederror", n_estimators=n_estimators, random_state=seed, n_jobs=1, verbosity=0, **p, **extra)


def fit_xgb_es(params, X, y, tokens, seed=0, es_seed=ES_SEED, **extra):
    """Same procedure as the baseline: early-stop on a fixed 15 % shape-level carve-out, refit at best_n on all rows."""
    rng = np.random.default_rng(es_seed)
    uniq = np.unique(tokens)
    val = np.isin(tokens, rng.choice(uniq, size=max(1, int(round(ES_FRAC * len(uniq)))), replace=False))
    probe = make_xgb(params, params["n_estimators"], seed, early_stopping_rounds=ES_ROUNDS, eval_metric="mape", **extra)
    probe.fit(X[~val], y[~val], eval_set=[(X[val], y[val])], verbose=False)
    best_n = int(probe.best_iteration) + 1
    m = make_xgb(params, best_n, seed, **extra).fit(X, y)
    m.best_n_ = best_n
    return m


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    t = tidy(load_raw(DEFAULT_CSV))
    split = load_split(HERE / "splits", "random", include_holdout=False)
    res = pd.read_csv(RES / "results.csv")
    interp_ref = pd.read_csv(HERE / "results" / "full_v1" / "test_metrics.csv")
    interp_ref = interp_ref[(interp_ref.geometry == "random") & (interp_ref.model == "interp")][["op", "tp", "mape"]].rename(columns={"mape": "interp_test_mape"})
    keys = regressor_keys(t)
    report = {}

    # ------------------------------------------------------------------ A. configuration facts
    probe = XGBRegressor(objective="reg:squarederror", n_estimators=1, verbosity=0)
    Xtr, ytr, ttr = subset(t, "attn_pre_proj", 2, split["train"])
    probe.fit(Xtr, ytr)
    cfg = json.loads(probe.get_booster().save_config())
    tm = cfg["learner"]["gradient_booster"]["gbtree_train_param"]["tree_method"]
    updater = cfg["learner"]["gradient_booster"].get("updater", [{}])
    max_bin = None
    for u in updater if isinstance(updater, list) else [updater]:
        hp = u.get("hist_train_param") or u.get("train_param") or {}
        if "max_bin" in hp:
            max_bin = int(hp["max_bin"])
    report["xgboost_version"] = xgboost.__version__
    report["tree_method_resolved"] = tm
    report["max_bin_default"] = max_bin
    report["unique_train_tokens_per_pair"] = int(len(np.unique(ttr)))
    # actual histogram cut points for one feature (identical for every pair: same train token values)
    qdm = QuantileDMatrix(Xtr, ytr, max_bin=max_bin or 256)
    indptr, cuts = qdm.get_quantile_cut()
    cuts = np.asarray(cuts[indptr[0]:indptr[1]], dtype=float)
    report["n_cut_points"] = int(len(cuts))
    bins = pd.DataFrame({"lo": np.r_[Xtr.min(), cuts[:-1]], "hi": cuts})
    bins["n_train_tokens"] = [int(((ttr > lo) & (ttr <= hi)).sum()) if i else int((ttr <= hi).sum()) for i, (lo, hi) in enumerate(zip(bins.lo, bins.hi))]
    bins["width_tokens"] = bins.hi - bins.lo
    region = pd.cut(bins.hi, [0, 2048, 8192, 16384 + 1], labels=["<=2048", "2049-8192", ">8192"])
    report["bin_width_tokens_by_region"] = bins.groupby(region, observed=True).width_tokens.agg(["count", "median", "min", "max"]).round(1).to_dict("index")
    cliff_bins = []
    for lo, hi in CLIFFS:
        idx = np.flatnonzero((bins.hi >= lo) & (bins.lo <= hi))
        cliff_bins.append({"window": f"{lo}-{hi}", "n_bins_overlapping": int(len(idx)),
                           "bin_ranges": [f"({bins.lo[i]:.0f}, {bins.hi[i]:.0f}]" for i in idx],
                           "train_tokens_sharing_those_bins": int(bins.n_train_tokens[idx].sum())})
    report["cliff_window_binning"] = cliff_bins
    bins.to_csv(OUT / "hist_bins.csv", index=False)
    report["objective"] = "reg:squarederror on the label in ms (no target transform)"
    report["early_stopping_metric"] = "mape (xgboost eval_metric) on a fixed 15 % shape-level carve-out of the fitting rows"
    report["cv_selection_metric"] = "Frontier MAPE on 10 shape-level folds"
    report["metrics_match"] = "early stopping and selection both use MAPE; the training objective is squared error in ms, which weights large labels more"

    xr = res[res.model == "xgb"].copy()
    xr["cap"] = xr.best_params.map(lambda s: json.loads(s)["n_estimators"])
    xr["early_stopped"] = xr.xgb_best_iteration < xr.cap
    xr["hit_cap"] = xr.xgb_best_iteration >= xr.cap - 1
    es = xr[["op", "tp", "xgb_best_iteration", "cap", "early_stopped", "hit_cap"]]
    es.to_csv(OUT / "early_stopping.csv", index=False)
    edge_rows = []
    for r in res.itertuples(index=False):
        p = json.loads(r.best_params)
        space = XGB_SPACE if r.model == "xgb" else RF_SPACE
        for k, v in p.items():
            vals = space[k]
            if v == vals[0] or v == vals[-1]:
                edge_rows.append(dict(model=r.model, op=r.op, tp=r.tp, param=k, value=v, edge="min" if v == vals[0] else "max"))
    edge = pd.DataFrame(edge_rows)
    edge.to_csv(OUT / "edge_params.csv", index=False)

    # ------------------------------------------------------------------ B/C. refits: train MAPE, seeds, binning, interp
    rows, bin_rows, seed_rows, pred_frames = [], [], [], []
    for op, tp in keys:
        Xtr, ytr, ttr = subset(t, op, tp, split["train"])
        Xte, yte, tte = subset(t, op, tp, split["test"])
        rr = res[(res.op == op) & (res.tp == tp)].set_index("model")
        p_rf = json.loads(rr.loc["rf", "best_params"])
        p_xg = json.loads(rr.loc["xgb", "best_params"])
        rf = fit_rf(p_rf, Xtr, ytr, ttr)
        xg = fit_xgb_es(p_xg, Xtr, ytr, ttr)
        interp = InterpRegressor().fit(Xtr, ytr)
        assert abs(mape(yte, xg.predict(Xte)) - rr.loc["xgb", "test_mape"]) < 1e-6
        for name, m in (("rf", rf), ("xgb", xg), ("interp", interp)):
            ptr, pte = m.predict(Xtr), m.predict(Xte)
            rows.append(dict(op=op, tp=tp, model=name, train_mape=mape(ytr, ptr), test_mape=mape(yte, pte),
                             test_mape_gt64=mape(yte[tte > 64], pte[tte > 64]), test_mape_gt8192=mape(yte[tte > 8192], pte[tte > 8192]),
                             test_mape_cliff=mape(yte[in_cliff(tte)], pte[in_cliff(tte)]),
                             cv_mape_mean=float(rr.loc[name, "cv_mape_mean"]) if name in rr.index else np.nan))
            pred_frames.append(pd.DataFrame(dict(op=op, tp=tp, model=name, num_tokens=tte, y=yte, pred=pte)))
        # model-seed spread (split fixed): RF random_state, XGB random_state and early-stopping carve-out seed
        for s in range(5):
            seed_rows.append(dict(op=op, tp=tp, model="rf", seed=s, test_mape=mape(yte, fit_rf({**p_rf}, Xtr, ytr, ttr).predict(Xte)) if s == 0 else
                             mape(yte, __import__("sklearn.ensemble", fromlist=["RandomForestRegressor"]).RandomForestRegressor(random_state=s, n_jobs=1, **p_rf).fit(Xtr, ytr).predict(Xte))))
            seed_rows.append(dict(op=op, tp=tp, model="xgb", seed=s, test_mape=mape(yte, fit_xgb_es(p_xg, Xtr, ytr, ttr, seed=s, es_seed=ES_SEED + s).predict(Xte))))
        # binning experiment: CV on train (clean), then test (exploratory)
        folds = cv_folds(ttr, "random", N_FOLDS, seed=FOLD_SEED, band_rel_width=0.10)
        for label, extra in (("hist_256", {}), ("hist_1024", {"max_bin": 1024}), ("hist_4096", {"max_bin": 4096}), ("exact", {"tree_method": "exact"})):
            cv = Parallel(n_jobs=6)(delayed(lambda tr, va: mape(ytr[va], fit_xgb_es(p_xg, Xtr[tr], ytr[tr], ttr[tr], **extra).predict(Xtr[va])))(tr, va) for tr, va in folds)
            m = fit_xgb_es(p_xg, Xtr, ytr, ttr, **extra)
            bin_rows.append(dict(op=op, tp=tp, variant=label, cv_mape_mean=float(np.mean(cv)), cv_mape_std=float(np.std(cv, ddof=1)),
                                 train_mape=mape(ytr, m.predict(Xtr)), test_mape=mape(yte, m.predict(Xte)), best_n=m.best_n_))
        print(f"{op:25s} TP{tp}  train MAPE rf={rows[-3]['train_mape']:.2f} xgb={rows[-2]['train_mape']:.2f} interp={rows[-1]['train_mape']:.2f} | "
              f"test rf={rows[-3]['test_mape']:.2f} xgb={rows[-2]['test_mape']:.2f} interp={rows[-1]['test_mape']:.2f} | "
              f"xgb CV by binning: " + " ".join(f"{b['variant']}={b['cv_mape_mean']:.2f}" for b in bin_rows[-4:]), flush=True)

    fit = pd.DataFrame(rows)
    fit.to_csv(OUT / "train_vs_test.csv", index=False)
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(OUT / "seed_spread.csv", index=False)
    binning = pd.DataFrame(bin_rows)
    binning.to_csv(OUT / "binning_experiment.csv", index=False)
    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(OUT / "test_predictions.csv", index=False)

    # gap per pair
    piv = fit.pivot(index=["op", "tp"], columns="model", values="test_mape").reset_index()
    piv["gap_xgb_minus_rf"] = piv.xgb - piv.rf
    piv["rf_minus_interp"] = piv.rf - piv.interp
    piv.to_csv(OUT / "gap_per_pair.csv", index=False)

    # figures: worst three pairs by gap, overlay + residuals
    worst = piv.sort_values("gap_xgb_minus_rf", ascending=False).head(3)
    for r in worst.itertuples(index=False):
        g = preds[(preds.op == r.op) & (preds.tp == r.tp)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=130)
        fig.patch.set_facecolor(SURFACE)
        ax = axes[0]
        style(ax)
        gl = g[g.model == "rf"].sort_values("num_tokens")
        ax.scatter(gl.num_tokens, gl.y * 1e3, s=10, color=SERIES[2], label="measured (test)", zorder=3)
        for k, (m, name) in enumerate((("rf", "Random Forest"), ("xgb", "XGBoost"))):
            gm = g[g.model == m].sort_values("num_tokens")
            ax.plot(gm.num_tokens, gm.pred * 1e3, lw=1.1, color=SERIES[k], label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("num_tokens", fontsize=8, color=INK2)
        ax.set_ylabel("time (µs)", fontsize=8, color=INK2)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{r.op} TP{r.tp}: predictions over labels", loc="left", fontsize=9, color=INK)
        ax = axes[1]
        style(ax)
        for lo, hi in CLIFFS:
            ax.axvspan(lo, hi, color=GRID, lw=0)
        for k, (m, name) in enumerate((("rf", "Random Forest"), ("xgb", "XGBoost"))):
            gm = g[g.model == m]
            ax.scatter(gm.num_tokens, (gm.pred - gm.y) / gm.y * 100, s=9, color=SERIES[k], alpha=.8, lw=0, label=name)
        ax.axhline(0, color=BASELINE, lw=.8)
        ax.set_xscale("log")
        ax.set_xlabel("num_tokens", fontsize=8, color=INK2)
        ax.set_ylabel("signed error %", fontsize=8, color=INK2)
        ax.set_title("residuals (shaded = cliff windows)", loc="left", fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(FIGS / f"worst_{r.op}_TP{r.tp}.png", facecolor=SURFACE)
        plt.close(fig)

    (OUT / "config_facts.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    print("\nearly stopping:\n", es.to_string(index=False))
    print("\nedge params:\n", edge.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
