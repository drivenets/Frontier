"""SIDE INVESTIGATION (standalone): RF / XGBoost baselines on the pre-fix MI355X linear_op collection (commit 3d1cabd).

Everything here is self-contained: split logic, fold mechanism, model grids and metrics are copied from
regressor_bench/ (not imported) so nothing in the main harness is touched or depended on at run time. The main
harness is only *read* (its results.csv, its sealed random-split token lists, and the current dataset mirror) to
build the comparison table. The current dataset's holdout tier is never read or scored.

Data:  git show 3d1cabd:data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv > data/nuzan_original_linear_op.csv

Method mirrors regressor_bench/baseline_rf_xgb.py: feature num_tokens, label time_stats.<op>.median, shape-level
75/15/10 split on this file's own grid (stratified random draw, same scheme and seed as the main split), 10
shape-level CV folds on train, RF full grid (48) and XGBoost 48 sampled configs with early stopping on a fixed
internal shape-level split, refit on train, scored once on test and once on the OLD data's holdout.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD_CSV = HERE / "data" / "nuzan_original_linear_op.csv"
OLD_MD5 = "dbc872076ff145d8a518e69673a52a6b"
OUT = HERE / "results"
# read-only inputs from the main harness
MAIN = REPO / "profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench"
CUR_CSV = REPO / "data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv"
CUR_RESULTS = MAIN / "results/baseline_rf_xgb/results.csv"
CUR_SPLIT = MAIN / "splits/random"

SHARDED = ("attn_pre_proj", "attn_post_proj", "attn_rope")
REPLICATED = ("input_layernorm", "post_attention_layernorm", "emb")
CLIFFS = [(4352, 4360), (4480, 4488), (4864, 4872), (5376, 5384), (11800, 12300)]
N_FOLDS, FOLD_SEED, SPLIT_SEED = 10, 0, 20260923
ES_SEED, ES_FRAC, ES_ROUNDS = 12345, 0.15, 50
PLATEAU_TOL = 0.10  # label within +10 % of the small-token level counts as plateau
N_JOBS = 6

RF_GRID = [dict(n_estimators=n, max_depth=d, min_samples_leaf=m)
           for n, d, m in itertools.product([200, 500, 1000], [None, 10, 20, 30], [1, 2, 5, 10])]
XGB_SPACE = dict(max_depth=[3, 4, 6, 8], learning_rate=[0.01, 0.05, 0.1], n_estimators=[200, 500, 1000],
                 subsample=[0.6, 0.8, 1.0], min_child_weight=[1, 5, 10])


def xgb_grid(n: int = 48, seed: int = 0) -> List[Dict]:
    full = [dict(zip(XGB_SPACE, v)) for v in itertools.product(*XGB_SPACE.values())]
    idx = np.random.default_rng(seed).choice(len(full), size=n, replace=False)
    return [full[i] for i in sorted(idx)]


# ------------------------------------------------------------------------------------------ data / splits (copied)


def md5_of(path: Path) -> str:
    import hashlib
    return hashlib.md5(path.read_bytes()).hexdigest()


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for op in SHARDED + REPLICATED:
        d = pd.DataFrame({"op": op, "tp": raw["num_tensor_parallel_workers"].astype(int), "num_tokens": raw["num_tokens"].astype(int),
                          "y": raw[f"time_stats.{op}.median"].astype(float), "y_std": raw[f"time_stats.{op}.std"].astype(float),
                          "n_samples": raw[f"time_stats.{op}.count"]})
        frames.append(d[d.y.notna()])
    return pd.concat(frames, ignore_index=True).sort_values(["op", "tp", "num_tokens"]).reset_index(drop=True)


def log_strata(values, n_strata=12, min_size=40):
    values = np.asarray(values, dtype=float)
    edges = np.linspace(np.log2(values.min()), np.log2(values.max()), n_strata + 1)
    raw = np.clip(np.digitize(np.log2(values), edges[1:-1]), 0, n_strata - 1)
    labels, current, count = np.zeros_like(raw), 0, 0
    for b in range(n_strata):
        m = raw == b
        labels[m] = current
        count += int(m.sum())
        if count >= min_size:
            current, count = current + 1, 0
    if 0 < count < min_size and current > 0:
        labels[labels == current] = current - 1
    return labels


def make_random_split(tokens, holdout_frac=0.10, test_frac=0.15, seed=SPLIT_SEED, min_size=40) -> pd.DataFrame:
    tokens = np.sort(np.asarray(tokens))
    rng = np.random.default_rng(seed)
    strata = log_strata(tokens, min_size=min_size)
    tier = np.array(["train"] * len(tokens), dtype=object)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        n_h, n_t = int(round(len(idx) * holdout_frac)), int(round(len(idx) * test_frac))
        tier[idx[:n_h]] = "holdout"
        tier[idx[n_h:n_h + n_t]] = "test"
    return pd.DataFrame({"num_tokens": tokens, "stratum": strata, "tier": tier})


def cv_folds(train_tokens, n_splits=N_FOLDS, seed=FOLD_SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_tokens))
    chunks = np.array_split(perm, n_splits)
    all_idx = np.arange(len(train_tokens))
    return [(np.setdiff1d(all_idx, np.sort(v), assume_unique=True), np.sort(v)) for v in chunks]


# ------------------------------------------------------------------------------------------------ models (copied)


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = np.zeros_like(y)
    nz = y != 0
    out[nz] = np.abs((y[nz] - p[nz]) / y[nz]) * 100
    return float(out.mean()) if len(y) else float("nan")


def fit_rf(params, X, y, _tokens):
    return RandomForestRegressor(random_state=0, n_jobs=1, **params).fit(X, y)


def fit_xgb(params, X, y, tokens):
    rng = np.random.default_rng(ES_SEED)
    uniq = np.unique(tokens)
    val = np.isin(tokens, rng.choice(uniq, size=max(1, int(round(ES_FRAC * len(uniq)))), replace=False))
    p = dict(params)
    max_rounds = p.pop("n_estimators")
    probe = XGBRegressor(objective="reg:squarederror", eval_metric="mape", n_estimators=max_rounds, early_stopping_rounds=ES_ROUNDS,
                         random_state=0, n_jobs=1, verbosity=0, **p)
    probe.fit(X[~val], y[~val], eval_set=[(X[val], y[val])], verbose=False)
    best_n = int(probe.best_iteration) + 1
    m = XGBRegressor(objective="reg:squarederror", n_estimators=best_n, random_state=0, n_jobs=1, verbosity=0, **p).fit(X, y)
    m.best_n_ = best_n
    return m


FITTERS = {"rf": fit_rf, "xgb": fit_xgb}


def cv_one(model, params, X, y, tokens, folds):
    scores = [mape(y[va], FITTERS[model](params, X[tr], y[tr], tokens[tr]).predict(X[va])) for tr, va in folds]
    return dict(params=params, fold_mape=scores, cv_mape=float(np.mean(scores)))


# ------------------------------------------------------------------------------------------------------ plateau


def plateau_end(g: pd.DataFrame, tol=PLATEAU_TOL, n_ref=8, run=3) -> int:
    """Largest num_tokens of the flat low-token region: label stays within +tol of the median of the n_ref smallest
    shapes until `run` consecutive shapes exceed it. Returns 0 if the curve ramps immediately."""
    g = g.sort_values("num_tokens")
    ref = float(np.median(g.y.iloc[:n_ref]))
    above = (g.y.to_numpy() > ref * (1 + tol)).astype(int)
    for i in range(len(above) - run + 1):
        if above[i:i + run].all():
            return int(g.num_tokens.iloc[i - 1]) if i > 0 else 0
    return int(g.num_tokens.max())


# --------------------------------------------------------------------------------------------------------- main


def subset(t, op, tp, tokens):
    g = t[(t.op == op) & (t.tp == tp) & t.num_tokens.isin(tokens)].sort_values("num_tokens")
    return g[["num_tokens"]].to_numpy(float), g.y.to_numpy(float), g.num_tokens.to_numpy()


def main() -> int:
    OUT.mkdir(exist_ok=True)
    assert md5_of(OLD_CSV) == OLD_MD5, "old CSV changed; re-extract from 3d1cabd"
    raw = pd.read_csv(OLD_CSV, low_memory=False)
    forbidden = [c for c in raw.columns if c.startswith("time_stats_hostbound.") or c in ("settle_steps", "backlog_kind")]
    print(f"old file: {raw.shape[0]} rows x {raw.shape[1]} cols; hostbound/settle/backlog columns present: {forbidden or 'none'}")
    t = tidy(raw)
    grid = np.sort(t.num_tokens.unique())
    print(f"grid: {len(grid)} token values, {grid.min()}..{grid.max()}; step counts: {dict(zip(*np.unique(np.diff(grid), return_counts=True)))}")
    for lo, hi in CLIFFS:
        print(f"  cliff window {lo}-{hi}: grid values inside = {[int(v) for v in grid if lo <= v <= hi]}  (grid max {grid.max()})")

    # ---- split on the old grid + disjointness audit per (op, TP)
    assign = make_random_split(grid, min_size=20)  # 258 values: smaller minimum stratum than the 3,327-value grid
    assign.to_csv(OUT / "old_split_assignment.csv", index=False)
    tiers = {k: set(map(int, assign[assign.tier == k].num_tokens)) for k in ("train", "test", "holdout")}
    print(f"split: train={len(tiers['train'])} test={len(tiers['test'])} holdout={len(tiers['holdout'])}  "
          f"overlaps train&test={len(tiers['train'] & tiers['test'])} train&holdout={len(tiers['train'] & tiers['holdout'])} test&holdout={len(tiers['test'] & tiers['holdout'])}")
    keys = [(o, int(p)) for o, p in t[["op", "tp"]].drop_duplicates().itertuples(index=False)]
    audit = []
    for op, tp in keys:
        toks = set(map(int, t[(t.op == op) & (t.tp == tp)].num_tokens))
        a, b, c = toks & tiers["train"], toks & tiers["test"], toks & tiers["holdout"]
        ok = not (a & b) and not (a & c) and not (b & c) and (a | b | c) == toks
        audit.append(dict(op=op, tp=tp, rows=len(toks), train=len(a), test=len(b), holdout=len(c), disjoint_and_covering=ok))
    audit = pd.DataFrame(audit)
    audit.to_csv(OUT / "old_split_audit.csv", index=False)
    assert audit.disjoint_and_covering.all()
    print(audit.to_string(index=False))

    # ---- plateau per pair (from the full old curve) and label noise
    plat = {}
    for op, tp in keys:
        g = t[(t.op == op) & (t.tp == tp)]
        end = plateau_end(g)
        plat[(op, tp)] = end
    # ---- baselines
    grids = {"rf": RF_GRID, "xgb": xgb_grid()}
    rows, fold_rows, preds = [], [], []
    for op, tp in keys:
        X, y, tok = {}, {}, {}
        for k, v in tiers.items():
            X[k], y[k], tok[k] = subset(t, op, tp, v)
        folds = cv_folds(tok["train"])
        g_full = t[(t.op == op) & (t.tp == tp)]
        end = plat[(op, tp)]
        for model, grid in grids.items():
            t0 = time.time()
            cv = Parallel(n_jobs=N_JOBS)(delayed(cv_one)(model, p, X["train"], y["train"], tok["train"], folds) for p in grid)
            best = min(cv, key=lambda r: r["cv_mape"])
            fold_rows += [dict(op=op, tp=tp, model=model, fold=k, mape=s) for k, s in enumerate(best["fold_mape"])]
            final = FITTERS[model](best["params"], X["train"], y["train"], tok["train"])
            row = dict(op=op, tp=tp, model=model, best_params=json.dumps(best["params"]), xgb_best_iteration=getattr(final, "best_n_", np.nan),
                       cv_mape_mean=float(np.mean(best["fold_mape"])), cv_mape_std=float(np.std(best["fold_mape"], ddof=1)),
                       n_train=len(y["train"]), n_test=len(y["test"]), n_holdout=len(y["holdout"]),
                       plateau_end_tokens=end, plateau_rows=int((g_full.num_tokens <= end).sum()), plateau_pct=float((g_full.num_tokens <= end).mean() * 100),
                       label_rel_std_pct_median=float((g_full.y_std / g_full.y).median() * 100))
            for tier in ("test", "holdout"):
                p = final.predict(X[tier])
                preds.append(pd.DataFrame(dict(tier=tier, op=op, tp=tp, model=model, num_tokens=tok[tier], y=y[tier], pred=p)))
                nonplat = tok[tier] > end
                row[f"{tier}_mape"] = mape(y[tier], p)
                row[f"{tier}_mape_excl_plateau"] = mape(y[tier][nonplat], p[nonplat])
                row[f"{tier}_n"] = len(p)
                row[f"{tier}_n_excl_plateau"] = int(nonplat.sum())
            row["seconds"] = time.time() - t0
            rows.append(row)
            print(f"{op:25s} TP{tp} {model:3s} cv={row['cv_mape_mean']:5.2f}+-{row['cv_mape_std']:4.2f} test={row['test_mape']:5.2f} "
                  f"holdout={row['holdout_mape']:5.2f} | excl plateau (<= {end} tok, {row['plateau_pct']:.0f}% of grid): "
                  f"test={row['test_mape_excl_plateau']:5.2f} [{row['test_n_excl_plateau']}] holdout={row['holdout_mape_excl_plateau']:5.2f} [{row['holdout_n_excl_plateau']}]", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "old_results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUT / "old_cv_folds.csv", index=False)
        pd.concat(preds, ignore_index=True).to_csv(OUT / "old_predictions.csv", index=False)

    # ---- current-data reference: refit the recorded winners (read-only inputs), test tier only; holdout not read
    cur = pd.read_csv(CUR_RESULTS)
    cur_raw = pd.read_csv(CUR_CSV, low_memory=False)
    ct = tidy(cur_raw)
    cur_train = set(pd.read_csv(CUR_SPLIT / "train_tokens.csv").num_tokens)
    cur_test = set(pd.read_csv(CUR_SPLIT / "test_tokens.csv").num_tokens)
    assert not (cur_train & cur_test)
    cur_rows = []
    for r in cur.itertuples(index=False):
        Xtr, ytr, ttr = subset(ct, r.op, r.tp, cur_train)
        Xte, yte, tte = subset(ct, r.op, r.tp, cur_test)
        params = json.loads(r.best_params)
        if r.model == "xgb":
            params["n_estimators"] = int(r.xgb_best_iteration)
            m = XGBRegressor(objective="reg:squarederror", random_state=0, n_jobs=1, verbosity=0, **params).fit(Xtr, ytr)
        else:
            m = fit_rf(params, Xtr, ytr, ttr)
        p = m.predict(Xte)
        assert abs(mape(yte, p) - r.test_mape) < 1e-6
        g_full = ct[(ct.op == r.op) & (ct.tp == r.tp)]
        end = plateau_end(g_full)
        nonplat, le4096 = tte > end, tte <= 4096
        cur_rows.append(dict(op=r.op, tp=r.tp, model=r.model, cur_test_mape=r.test_mape,
                             cur_plateau_end_tokens=end, cur_plateau_pct=float((g_full.num_tokens <= end).mean() * 100),
                             cur_test_mape_excl_plateau=mape(yte[nonplat], p[nonplat]), cur_test_n_excl_plateau=int(nonplat.sum()),
                             cur_test_mape_le4096=mape(yte[le4096], p[le4096]), cur_test_n_le4096=int(le4096.sum()),
                             cur_test_mape_le4096_excl_plateau=mape(yte[le4096 & nonplat], p[le4096 & nonplat]),
                             cur_label_rel_std_pct_median=float((g_full.y_std / g_full.y).median() * 100)))
    cur_df = pd.DataFrame(cur_rows)
    comp = pd.DataFrame(rows).merge(cur_df, on=["op", "tp", "model"])
    comp["cur_holdout_mape"] = "sealed, not scored"
    comp.to_csv(OUT / "comparison.csv", index=False)
    print("wrote", OUT / "comparison.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
