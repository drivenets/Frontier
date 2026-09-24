"""Baseline v2: Random Forest, XGBoost (exact splits, fresh search) and linear interpolation per (op, TP), plus the
measurement noise floor from an independent collection. Sealed random-geometry shape-level split; train + test only.
The holdout tier is never read.

Changes from baseline_rf_xgb.py (kept unchanged for the record):
* XGBoost uses tree_method="exact" (diagnose_xgb.py showed the 256-bin histogram was the cause of its deficit) and a
  fresh search: max_depth {4,6,8,10,12} x learning_rate {0.02,0.05,0.1,0.2} x reg_lambda {0,0.01,0.1,1} x
  min_child_weight {1,2,5} x subsample {0.8,1.0} = 480 configs, 48 sampled without replacement; n_estimators cap 10000
  with early stopping (50 rounds) on the fixed 15 % shape-level carve-out, then refit on all fitting rows at best_n.
* linear interpolation (InterpRegressor: piecewise-linear over the training grid) is a third baseline row everywhere.
* RF rows are taken from results/baseline_rf_xgb (same protocol, same folds, deterministic) and re-fit for predictions.
* noise floor: MAPE between the medians of two independent collections with the same measurement fixes on the
  test-tier tokens (job 21519, settle 3, 2026-09-22 vs job 21539, settle 8, 2026-09-23; 13_ s8: medians identical,
  p50 ratio 1.000-1.002). That number is the error of one collection "predicting" the other, so it carries the noise
  of both; the single-collection floor is about that / sqrt(2).
* step detection on the train tier for attn_post_proj (and all pairs, written to CSV): sustained jumps of the label
  between neighbouring grid points, compared with the five known cliff windows.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from regressor_bench.baseline_rf_xgb import CLIFFS, CLIFF_MARGIN, ES_FRAC, ES_ROUNDS, ES_SEED, FOLD_SEED, N_FOLDS, fit_rf, in_cliff  # noqa: E402
from regressor_bench.dataset import DEFAULT_CSV, load_raw, md5_of, regressor_keys, tidy  # noqa: E402
from regressor_bench.metrics import INSTRUMENT_FLOOR_MS, mape  # noqa: E402
from regressor_bench.models import InterpRegressor  # noqa: E402
from regressor_bench.report import BASELINE, GRID, INK, INK2, MUTED, SERIES, SURFACE  # noqa: E402
from regressor_bench.splits import cv_folds, load_split  # noqa: E402

HERE = Path(__file__).resolve().parent
PREV = HERE / "results" / "baseline_rf_xgb"
NOISE_CSV = Path("data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-22_job21519_dense_workbacklog_settle3/linear_op.csv")
NOISE_MD5 = "7016bd62"  # prefix, checked
XGB_CAP = 10000
XGB_SPACE = dict(max_depth=[4, 6, 8, 10, 12], learning_rate=[0.02, 0.05, 0.1, 0.2], reg_lambda=[0, 0.01, 0.1, 1],
                 min_child_weight=[1, 2, 5], subsample=[0.8, 1.0])
STEP_MIN_TOKENS = 256      # below this, sub-10 us labels and the 1-token regime make jump detection meaningless
STEP_WINDOW = 4            # neighbours on each side used for the before/after level
STEP_THRESHOLD = 0.06      # sustained relative jump between the two levels


def xgb_grid(n: int = 48, seed: int = 0) -> List[Dict]:
    full = [dict(zip(XGB_SPACE, v)) for v in itertools.product(*XGB_SPACE.values())]
    idx = np.random.default_rng(seed).choice(len(full), size=n, replace=False)
    return [full[i] for i in sorted(idx)]


def fit_xgb_exact(params: Dict, X, y, tokens):
    rng = np.random.default_rng(ES_SEED)
    uniq = np.unique(tokens)
    val = np.isin(tokens, rng.choice(uniq, size=max(1, int(round(ES_FRAC * len(uniq)))), replace=False))
    common = dict(objective="reg:squarederror", tree_method="exact", random_state=0, n_jobs=1, verbosity=0, **params)
    probe = XGBRegressor(n_estimators=XGB_CAP, early_stopping_rounds=ES_ROUNDS, eval_metric="mape", **common)
    probe.fit(X[~val], y[~val], eval_set=[(X[val], y[val])], verbose=False)
    best_n = int(probe.best_iteration) + 1
    m = XGBRegressor(n_estimators=best_n, **common).fit(X, y)
    m.best_n_ = best_n
    return m


def cv_xgb(params, X, y, tokens, folds):
    scores, cliff = [], []
    for tr, va in folds:
        p = fit_xgb_exact(params, X[tr], y[tr], tokens[tr]).predict(X[va])
        scores.append(mape(y[va], p))
        c = in_cliff(tokens[va])
        cliff.append(mape(y[va][c], p[c]) if c.any() else np.nan)
    return dict(params=params, fold_mape=scores, fold_cliff=cliff, cv_mape=float(np.mean(scores)))


def subset(t, op, tp, tokens):
    g = t[(t.op == op) & (t.tp == tp) & t.num_tokens.isin(tokens)].sort_values("num_tokens")
    return g[["num_tokens"]].to_numpy(float), g.y.to_numpy(float), g.num_tokens.to_numpy()


def detect_steps(tokens: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Sustained jumps between the median level of the STEP_WINDOW points before and after each grid gap."""
    order = np.argsort(tokens)
    tk, yy = tokens[order], y[order]
    rows = []
    w = STEP_WINDOW
    for i in range(w, len(tk) - w):
        if tk[i] < STEP_MIN_TOKENS:
            continue
        before, after = np.median(yy[i - w:i]), np.median(yy[i:i + w])
        jump = after / before - 1
        if abs(jump) >= STEP_THRESHOLD:
            rows.append(dict(from_tokens=int(tk[i - 1]), to_tokens=int(tk[i]), jump_pct=jump * 100))
    if not rows:
        return pd.DataFrame(columns=["from_tokens", "to_tokens", "jump_pct", "known_window"])
    df = pd.DataFrame(rows)
    # merge runs of adjacent detections into one step (keep the largest jump)
    df["grp"] = (df.to_tokens.diff().fillna(1e9) > 4 * STEP_WINDOW * 8).cumsum()
    df = df.loc[df.groupby("grp").jump_pct.apply(lambda s: s.abs().idxmax())].drop(columns="grp")
    df["known_window"] = [next((f"{lo}-{hi}" for lo, hi in CLIFFS if lo - CLIFF_MARGIN <= t <= hi + CLIFF_MARGIN), "") for t in df.to_tokens]
    return df.reset_index(drop=True)


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
    out = HERE / "results" / "baseline_v2"
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    t = tidy(load_raw(DEFAULT_CSV))
    assert md5_of(NOISE_CSV).startswith(NOISE_MD5)
    t2 = tidy(load_raw(NOISE_CSV, verify=False))
    split = load_split(HERE / "splits", "random", include_holdout=False)
    prev = pd.read_csv(PREV / "results.csv")
    prev_folds = pd.read_csv(PREV / "cv_folds.csv")
    keys = regressor_keys(t)
    grid = xgb_grid()
    rows, fold_rows, pred_frames, search_rows, step_frames = [], [], [], [], []
    t_start = time.time()
    for op, tp in keys:
        Xtr, ytr, ttr = subset(t, op, tp, split["train"])
        Xte, yte, tte = subset(t, op, tp, split["test"])
        folds = cv_folds(ttr, "random", N_FOLDS, seed=FOLD_SEED, band_rel_width=0.10)
        cliff_te = in_cliff(tte)
        floor_te = yte >= INSTRUMENT_FLOOR_MS
        coverage = {f"train_cov_{lo}-{hi}": int(np.sum((ttr >= lo - CLIFF_MARGIN) & (ttr <= hi + CLIFF_MARGIN))) for lo, hi in CLIFFS}
        # noise floor on the test tokens: collection 21519 vs 21539
        _, y2te, _ = subset(t2, op, tp, split["test"])
        noise = dict(noise_floor_mape=mape(yte, y2te), noise_floor_single=mape(yte, y2te) / np.sqrt(2),
                     noise_floor_cliff=mape(yte[cliff_te], y2te[cliff_te]) if cliff_te.any() else np.nan,
                     noise_floor_excl_sub10us=mape(yte[floor_te], y2te[floor_te]))
        # steps on the train tier
        st = detect_steps(ttr, ytr)
        st.insert(0, "tp", tp)
        st.insert(0, "op", op)
        step_frames.append(st)

        models = {}
        # RF: previous winner (identical protocol), refit for predictions
        pr = prev[(prev.op == op) & (prev.tp == tp) & (prev.model == "rf")].iloc[0]
        rf_params = json.loads(pr.best_params)
        models["rf"] = dict(est=fit_rf(rf_params, Xtr, ytr, ttr), best_params=rf_params, n_candidates=48,
                            fold_mape=prev_folds[(prev_folds.op == op) & (prev_folds.tp == tp) & (prev_folds.model == "rf")].sort_values("fold").mape.tolist(),
                            fold_cliff=prev_folds[(prev_folds.op == op) & (prev_folds.tp == tp) & (prev_folds.model == "rf")].sort_values("fold").cliff_mape.tolist(),
                            search_seconds=float(pr.search_seconds), xgb_best_iteration=np.nan)
        # interp: no search; folds for spread
        fm, fc = [], []
        for tr, va in folds:
            p = InterpRegressor().fit(Xtr[tr], ytr[tr]).predict(Xtr[va])
            fm.append(mape(ytr[va], p))
            c = in_cliff(ttr[va])
            fc.append(mape(ytr[va][c], p[c]) if c.any() else np.nan)
        models["interp"] = dict(est=InterpRegressor().fit(Xtr, ytr), best_params={}, n_candidates=1, fold_mape=fm, fold_cliff=fc, search_seconds=0.0, xgb_best_iteration=np.nan)
        # XGB exact: fresh search
        t0 = time.time()
        cv = Parallel(n_jobs=6)(delayed(cv_xgb)(p, Xtr, ytr, ttr, folds) for p in grid)
        for r in cv:
            search_rows.append(dict(op=op, tp=tp, model="xgb_exact", params=json.dumps(r["params"]), cv_mape=r["cv_mape"]))
        best = min(cv, key=lambda r: r["cv_mape"])
        final = fit_xgb_exact(best["params"], Xtr, ytr, ttr)
        edges = [f"{k}={v}({'min' if v == XGB_SPACE[k][0] else 'max'})" for k, v in best["params"].items() if v in (XGB_SPACE[k][0], XGB_SPACE[k][-1])]
        models["xgb_exact"] = dict(est=final, best_params=best["params"], n_candidates=len(grid), fold_mape=best["fold_mape"], fold_cliff=best["fold_cliff"],
                                   search_seconds=time.time() - t0, xgb_best_iteration=final.best_n_, edge_params=";".join(edges), hit_cap=final.best_n_ >= XGB_CAP)

        for name, m in models.items():
            p = m["est"].predict(Xte)
            fm = np.asarray(m["fold_mape"], float)
            fc = np.asarray(m["fold_cliff"], float)
            row = dict(op=op, tp=tp, model=name, best_params=json.dumps(m["best_params"], default=str), n_candidates=m["n_candidates"],
                       xgb_best_iteration=m["xgb_best_iteration"], edge_params=m.get("edge_params", ""), hit_cap=m.get("hit_cap", False),
                       cv_mape_mean=float(fm.mean()), cv_mape_std=float(fm.std(ddof=1)),
                       cv_cliff_mape_mean=float(np.nanmean(fc)) if np.isfinite(fc).any() else np.nan,
                       train_mape=mape(ytr, m["est"].predict(Xtr)),
                       test_mape=mape(yte, p), test_cliff_mape=mape(yte[cliff_te], p[cliff_te]) if cliff_te.any() else np.nan, test_n_cliff_rows=int(cliff_te.sum()),
                       test_mape_excl_sub10us=mape(yte[floor_te], p[floor_te]), test_n_sub10us_rows=int((~floor_te).sum()),
                       n_train=len(ytr), n_test=len(yte), search_seconds=m["search_seconds"], **noise, **coverage)
            row["learnable_mape_above_floor"] = row["test_mape"] - row["noise_floor_single"]
            rows.append(row)
            fold_rows += [dict(op=op, tp=tp, model=name, fold=k, mape=s, cliff_mape=c) for k, (s, c) in enumerate(zip(fm, fc))]
            pred_frames.append(pd.DataFrame(dict(tier="test", op=op, tp=tp, model=name, num_tokens=tte, y=yte, y_other_collection=y2te, pred=p)))
        r = {m: next(x for x in rows if x["op"] == op and x["tp"] == tp and x["model"] == m) for m in models}
        print(f"{op:25s} TP{tp}  noise={noise['noise_floor_mape']:.2f} (single {noise['noise_floor_single']:.2f}) | "
              f"interp test={r['interp']['test_mape']:.2f}  rf={r['rf']['test_mape']:.2f}  xgb_exact cv={r['xgb_exact']['cv_mape_mean']:.2f} test={r['xgb_exact']['test_mape']:.2f} "
              f"best_n={final.best_n_} {best['params']} edges=[{models['xgb_exact']['edge_params']}] ({models['xgb_exact']['search_seconds']:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(out / "cv_folds.csv", index=False)
        pd.DataFrame(search_rows).to_csv(out / "search_xgb_exact.csv", index=False)
        pd.concat(pred_frames, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
        pd.concat(step_frames, ignore_index=True).to_csv(out / "train_steps.csv", index=False)

    # ---- attn_post_proj: interpolation residuals vs num_tokens, with noise floor band and detected steps
    preds = pd.concat(pred_frames, ignore_index=True)
    steps = pd.concat(step_frames, ignore_index=True)
    g = preds[(preds.op == "attn_post_proj") & (preds.model == "interp")]
    tps = sorted(g.tp.unique())
    fig, axes = plt.subplots(2, len(tps), figsize=(4.4 * len(tps), 6.4), dpi=130, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, tp in enumerate(tps):
        d = g[g.tp == tp].sort_values("num_tokens")
        res_model = (d.pred - d.y) / d.y * 100
        res_noise = (d.y_other_collection - d.y) / d.y * 100
        for i, (vals, title, color) in enumerate(((res_model, "interp residual (pred - y)/y", SERIES[0]), (res_noise, "collection 21519 vs 21539 (y' - y)/y", SERIES[2]))):
            ax = axes[i][j]
            style(ax)
            for lo, hi in CLIFFS:
                ax.axvspan(lo, hi, color=GRID, lw=0)
            for s in steps[(steps.op == "attn_post_proj") & (steps.tp == tp)].itertuples():
                ax.axvline(s.to_tokens, color=SERIES[1], lw=.7, alpha=.7)
            ax.scatter(d.num_tokens, vals, s=8, color=color, alpha=.8, lw=0)
            ax.axhline(0, color=BASELINE, lw=.8)
            ax.set_xscale("log")
            ax.set_ylim(-25, 25)
            ax.set_title(f"TP{tp}: {title}", loc="left", fontsize=8.5, color=INK)
            ax.set_xlabel("num_tokens", fontsize=8, color=INK2)
        axes[0][j].set_xlabel("")
    axes[0][0].set_ylabel("%", fontsize=8, color=INK2)
    axes[1][0].set_ylabel("%", fontsize=8, color=INK2)
    fig.suptitle("attn_post_proj, test tier: interpolation residual (top) vs measurement noise between two collections (bottom). "
                 "Shaded = known cliff windows; orange lines = steps detected on the train tier", x=.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(figs / "attn_post_proj_interp_residuals.png", facecolor=SURFACE)
    plt.close(fig)

    meta = dict(csv=str(DEFAULT_CSV), csv_md5=md5_of(Path(DEFAULT_CSV)), noise_csv=str(NOISE_CSV), noise_csv_md5=md5_of(NOISE_CSV),
                xgb=dict(tree_method="exact", space=XGB_SPACE, n_sampled=len(grid), cap=XGB_CAP, es_rounds=ES_ROUNDS, es_frac=ES_FRAC, es_seed=ES_SEED),
                rf="rows taken from results/baseline_rf_xgb (identical protocol), refit for predictions", folds=N_FOLDS, fold_seed=FOLD_SEED,
                step_detection=dict(min_tokens=STEP_MIN_TOKENS, window=STEP_WINDOW, threshold=STEP_THRESHOLD),
                wall_seconds=time.time() - t_start, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"done in {meta['wall_seconds']:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
