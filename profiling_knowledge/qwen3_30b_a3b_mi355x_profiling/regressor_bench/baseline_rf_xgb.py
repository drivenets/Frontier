"""Baseline regressors (Random Forest, XGBoost) per (op, TP) on the sealed random-geometry shape-level split.

Protocol (fixed before running; nothing is iterated after test/holdout are seen):

* feature ``num_tokens`` only; label ``time_stats.<op>.median`` (ms); hostbound / kernel-only inputs never read;
* tiers: the sealed ``splits/random`` token-value assignment, as is (75/15/10);
* 10 shape-level CV folds on the train tier (``splits.cv_folds``, the mechanism verify_splits.py audits);
* RF: full grid n_estimators {200,500,1000} x max_depth {None,10,20,30} x min_samples_leaf {1,2,5,10} = 48;
* XGB: 48 configs sampled without replacement from max_depth {3,4,6,8} x learning_rate {0.01,0.05,0.1} x
  n_estimators {200,500,1000} x subsample {0.6,0.8,1.0} x min_child_weight {1,5,10} (324). Early stopping uses a
  separate fixed shape-level 15 % validation split of the fitting rows (own random_state), never the CV folds;
  the best iteration is then refit on all fitting rows;
* winner = lowest mean CV MAPE; refit on the full train tier; scored once on test.

The holdout tier is never read by this script (only the train and test token lists are loaded).

Metrics per (op, TP, model): overall test MAPE; test MAPE inside the known hipBLASLt cliff windows; test MAPE
excluding sub-10 us labels; train-tier coverage of each cliff window (values inside or within 8 tokens); CV
fold-level MAPE (per fold, mean, std) overall and inside the cliff windows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from regressor_bench.dataset import DEFAULT_CSV, load_raw, md5_of, regressor_keys, tidy  # noqa: E402
from regressor_bench.metrics import INSTRUMENT_FLOOR_MS, mape  # noqa: E402
from regressor_bench.splits import cv_folds, load_split  # noqa: E402

HERE = Path(__file__).resolve().parent
CLIFFS: List[Tuple[int, int]] = [(4352, 4360), (4480, 4488), (4864, 4872), (5376, 5384), (11800, 12300)]
CLIFF_MARGIN = 8
N_FOLDS = 10
FOLD_SEED = 0
ES_SEED = 12345  # internal early-stopping validation split (XGB only)
ES_FRAC = 0.15
ES_ROUNDS = 50

RF_GRID = [dict(n_estimators=n, max_depth=d, min_samples_leaf=m)
           for n, d, m in itertools.product([200, 500, 1000], [None, 10, 20, 30], [1, 2, 5, 10])]
XGB_SPACE = dict(max_depth=[3, 4, 6, 8], learning_rate=[0.01, 0.05, 0.1], n_estimators=[200, 500, 1000],
                 subsample=[0.6, 0.8, 1.0], min_child_weight=[1, 5, 10])


def xgb_grid(n: int, seed: int = 0) -> List[Dict]:
    full = [dict(zip(XGB_SPACE, v)) for v in itertools.product(*XGB_SPACE.values())]
    idx = np.random.default_rng(seed).choice(len(full), size=n, replace=False)
    return [full[i] for i in sorted(idx)]


# ------------------------------------------------------------------------------------------------ helpers


def in_cliff(tokens: np.ndarray, margin: int = 0) -> np.ndarray:
    tokens = np.asarray(tokens)
    m = np.zeros(len(tokens), dtype=bool)
    for lo, hi in CLIFFS:
        m |= (tokens >= lo - margin) & (tokens <= hi + margin)
    return m


def safe_mape(y, p, mask=None) -> float:
    if mask is not None:
        y, p = np.asarray(y)[mask], np.asarray(p)[mask]
    return mape(y, p) if len(y) else float("nan")


def fit_rf(params: Dict, X, y, _tokens) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=0, n_jobs=1, **params).fit(X, y)


def fit_xgb(params: Dict, X, y, tokens) -> XGBRegressor:
    """Early-stop on a fixed shape-level 15 % split of the fitting rows, then refit at the best iteration."""
    rng = np.random.default_rng(ES_SEED)
    uniq = np.unique(tokens)
    val_tokens = rng.choice(uniq, size=max(1, int(round(ES_FRAC * len(uniq)))), replace=False)
    v = np.isin(tokens, val_tokens)
    p = dict(params)
    max_rounds = p.pop("n_estimators")
    probe = XGBRegressor(objective="reg:squarederror", eval_metric="mape", n_estimators=max_rounds,
                         early_stopping_rounds=ES_ROUNDS, random_state=0, n_jobs=1, verbosity=0, **p)
    probe.fit(X[~v], y[~v], eval_set=[(X[v], y[v])], verbose=False)
    best_n = int(probe.best_iteration) + 1
    model = XGBRegressor(objective="reg:squarederror", n_estimators=best_n, random_state=0, n_jobs=1, verbosity=0, **p)
    model.fit(X, y)
    model.best_n_ = best_n
    return model


FITTERS = {"rf": fit_rf, "xgb": fit_xgb}


def cv_one(model: str, params: Dict, X, y, tokens, folds) -> Dict:
    scores, cliff_scores = [], []
    for tr, va in folds:
        m = FITTERS[model](params, X[tr], y[tr], tokens[tr])
        p = m.predict(X[va])
        scores.append(mape(y[va], p))
        cliff_scores.append(safe_mape(y[va], p, in_cliff(tokens[va])))
    return dict(params=params, fold_mape=scores, fold_cliff_mape=cliff_scores, cv_mape=float(np.mean(scores)))


# --------------------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--splits", default=str(HERE / "splits"))
    ap.add_argument("--out", default=str(HERE / "results" / "baseline_rf_xgb"))
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--n-xgb-configs", type=int, default=48)
    ap.add_argument("--ops", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true", help="2 configs per model (pipeline check only)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t = tidy(load_raw(a.csv))
    split = load_split(a.splits, "random", include_holdout=False)  # holdout tokens are never loaded
    tiers = {k: set(map(int, v)) for k, v in split.items()}
    assert set(tiers) == {"train", "test"} and not (tiers["train"] & tiers["test"])
    keys = regressor_keys(t)
    if a.ops:
        keys = [k for k in keys if k[0] in a.ops]
    grids = {"rf": RF_GRID, "xgb": xgb_grid(a.n_xgb_configs)}
    score_tiers = ("test",)
    if a.smoke:
        grids = {k: v[:2] for k, v in grids.items()}

    results, fold_rows, search_rows, pred_frames = [], [], [], []
    t_start = time.time()
    for op, tp in keys:
        g = t[(t.op == op) & (t.tp == tp)].sort_values("num_tokens")
        part = {k: g[g.num_tokens.isin(v)] for k, v in tiers.items()}
        X = {k: d[["num_tokens"]].to_numpy(float) for k, d in part.items()}
        y = {k: d["y"].to_numpy(float) for k, d in part.items()}
        tok = {k: d["num_tokens"].to_numpy() for k, d in part.items()}
        folds = cv_folds(tok["train"], "random", N_FOLDS, seed=FOLD_SEED, band_rel_width=0.10)
        coverage = {f"{lo}-{hi}": int(np.sum((tok["train"] >= lo - CLIFF_MARGIN) & (tok["train"] <= hi + CLIFF_MARGIN))) for lo, hi in CLIFFS}
        for model, grid in grids.items():
            t0 = time.time()
            cv = Parallel(n_jobs=a.n_jobs)(delayed(cv_one)(model, p, X["train"], y["train"], tok["train"], folds) for p in grid)
            for r in cv:
                search_rows.append(dict(op=op, tp=tp, model=model, params=json.dumps(r["params"]), cv_mape=r["cv_mape"]))
            best = min(cv, key=lambda r: r["cv_mape"])
            for k, (s, c) in enumerate(zip(best["fold_mape"], best["fold_cliff_mape"])):
                fold_rows.append(dict(op=op, tp=tp, model=model, fold=k, mape=s, cliff_mape=c,
                                      n_cliff_rows=int(in_cliff(tok["train"][folds[k][1]]).sum())))
            final = FITTERS[model](best["params"], X["train"], y["train"], tok["train"])
            row = dict(op=op, tp=tp, model=model, best_params=json.dumps(best["params"]),
                       xgb_best_iteration=getattr(final, "best_n_", np.nan),
                       cv_mape_mean=float(np.mean(best["fold_mape"])), cv_mape_std=float(np.std(best["fold_mape"], ddof=1)),
                       cv_cliff_mape_mean=float(np.nanmean(best["fold_cliff_mape"])) if not all(np.isnan(best["fold_cliff_mape"])) else np.nan,
                       cv_cliff_mape_std=float(np.nanstd(best["fold_cliff_mape"], ddof=1)) if np.sum(~np.isnan(best["fold_cliff_mape"])) > 1 else np.nan,
                       n_train=len(y["train"]), n_test=len(y["test"]))
            for tier in score_tiers:
                p = final.predict(X[tier])
                pred_frames.append(pd.DataFrame(dict(tier=tier, op=op, tp=tp, model=model, num_tokens=tok[tier], y=y[tier], pred=p)))
                cliff = in_cliff(tok[tier])
                floor = y[tier] >= INSTRUMENT_FLOOR_MS
                row[f"{tier}_mape"] = mape(y[tier], p)
                row[f"{tier}_cliff_mape"] = safe_mape(y[tier], p, cliff)
                row[f"{tier}_n_cliff_rows"] = int(cliff.sum())
                row[f"{tier}_mape_excl_sub10us"] = safe_mape(y[tier], p, floor)
                row[f"{tier}_n_sub10us_rows"] = int((~floor).sum())
            for k, v in coverage.items():
                row[f"train_cov_{k}"] = v
            row["cliff_gt_2x_overall"] = bool(row["test_cliff_mape"] > 2 * row["test_mape"]) if not np.isnan(row["test_cliff_mape"]) else False
            row["search_seconds"] = time.time() - t0
            results.append(row)
            print(f"{op:25s} TP{tp} {model:3s} cv={row['cv_mape_mean']:5.2f}+-{row['cv_mape_std']:4.2f}  test={row['test_mape']:5.2f}  "
                  f"cliff(test)={row['test_cliff_mape']:5.2f} [{row['test_n_cliff_rows']} rows]  "
                  f"{best['params']}  ({row['search_seconds']:.0f}s)", flush=True)
        pd.DataFrame(results).to_csv(out / "results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(out / "cv_folds.csv", index=False)
        pd.DataFrame(search_rows).to_csv(out / "search.csv", index=False)
        pd.concat(pred_frames, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)

    meta = dict(csv=str(a.csv), csv_md5=md5_of(Path(a.csv)), splits=str(a.splits), geometry="random", n_folds=N_FOLDS,
                fold_seed=FOLD_SEED, es_seed=ES_SEED, es_frac=ES_FRAC, es_rounds=ES_ROUNDS, rf_grid_size=len(RF_GRID),
                xgb_configs=len(grids["xgb"]), cliffs=CLIFFS, cliff_margin=CLIFF_MARGIN, wall_seconds=time.time() - t_start,
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    write_summary(out)
    print(f"done in {meta['wall_seconds']:.0f}s -> {out}")
    return 0


# ------------------------------------------------------------------------------------------------- summary


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("–" if np.isnan(v) else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(out: Path) -> None:
    r = pd.read_csv(out / "results.csv")
    folds = pd.read_csv(out / "cv_folds.csv")
    parts = ["# Baseline RF / XGBoost per (op, TP), random-geometry shape-level split", "",
             "Feature `num_tokens`; label `time_stats.<op>.median` (ms); 10 shape-level CV folds on train; test scored once; holdout not read. "
             "MAPE in %. `cliff` = rows with num_tokens in 4352-4360, 4480-4488, 4864-4872, 5376-5384, 11800-12300; "
             "`excl<10us` = rows with label >= 0.010 ms only; `cov_*` = train-tier num_tokens values inside or within 8 tokens of the window.", ""]
    metric_cols = ["op", "tp", "cv_mape_mean", "cv_mape_std", "test_mape", "test_cliff_mape", "test_n_cliff_rows",
                   "test_mape_excl_sub10us", "test_n_sub10us_rows", "cv_cliff_mape_mean", "cv_cliff_mape_std", "cliff_gt_2x_overall"]
    cov_cols = [c for c in r.columns if c.startswith("train_cov_")]
    for model, name in (("rf", "Random Forest"), ("xgb", "XGBoost")):
        rm = r[r.model == model].sort_values(["op", "tp"])
        parts += [f"## {name}: metrics", "", _md(rm[metric_cols]), ""]
        bp = rm[["op", "tp", "best_params", "cv_mape_mean"] + (["xgb_best_iteration"] if model == "xgb" else [])]
        parts += [f"## {name}: selected hyper-parameters", "", _md(bp), ""]
    parts += ["## Train-tier coverage of the cliff windows (values inside or within 8 tokens; identical for every op at a TP)", "",
              _md(r[r.model == "rf"][["op", "tp"] + cov_cols].rename(columns={c: c.replace("train_cov_", "") for c in cov_cols})), ""]
    flagged = r[r.cliff_gt_2x_overall]
    parts += ["## Flags: cliff-window test MAPE > 2x overall test MAPE", ""]
    parts += [_md(flagged[["op", "tp", "model", "test_mape", "test_cliff_mape", "test_n_cliff_rows"]]) if len(flagged) else "none", ""]
    parts += ["## CV fold-level MAPE (winning config), per fold", ""]
    piv = folds.pivot_table(index=["model", "op", "tp"], columns="fold", values="mape").reset_index()
    piv.columns = [str(c) for c in piv.columns]
    parts += [_md(piv), ""]
    parts += ["### CV fold-level cliff-window MAPE (winning config), per fold (– = no cliff rows in that fold)", ""]
    piv = folds.pivot_table(index=["model", "op", "tp"], columns="fold", values="cliff_mape", dropna=False).reset_index()
    piv.columns = [str(c) for c in piv.columns]
    parts += [_md(piv), ""]
    (out / "summary.md").write_text("\n".join(parts))


if __name__ == "__main__":
    sys.exit(main())
