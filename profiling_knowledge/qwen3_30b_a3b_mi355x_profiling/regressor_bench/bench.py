"""CLI for the regressor benchmark.

    python .../regressor_bench/bench.py make-splits --splits <dir>
    python .../regressor_bench/bench.py run         --splits <dir> --out results/<name> [--quick] [--models a b]
    python .../regressor_bench/bench.py holdout     --splits <dir> --results results/<name> --unseal
    python .../regressor_bench/bench.py report      --results results/<name>

Protocol per (geometry, op, TP, model):

1. **selection**   GridSearchCV over the model's grid on the *train* tier, K folds that mirror the split
                   geometry, scoring = Frontier's MAPE. Picks ``best_params``.
2. **assessment**  repeated K-fold on train with the chosen params, fresh fold seeds, the same folds for every
                   model (paired). Gives mean +- std MAPE.
3. **test**        fit on all of train, predict the *test* tier once. Metrics on the ms scale + per regime.
4. **holdout**     separate command, explicit ``--unseal``: refit on train+test with the selected params and
                   score the sealed tier once. Every use is appended to ``<splits>/HOLDOUT_LOG.md``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402  make `regressor_bench` importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

from regressor_bench.dataset import DEFAULT_CSV, load_raw, md5_of, regressor_keys, tidy, token_grid  # noqa: E402
from regressor_bench.metrics import by_regime, mape, mape_scorer, summarize  # noqa: E402
from regressor_bench.models import FEATURE_COLS, zoo  # noqa: E402
from regressor_bench.splits import GEOMETRIES, SplitSpec, cv_folds, load_manifest, load_split, write_splits  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SPLITS = HERE / "splits"
DEFAULT_RESULTS = HERE / "results"


# ------------------------------------------------------------------------------------------------- helpers


def _subset(t: pd.DataFrame, op: str, tp: int, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = t[(t.op == op) & (t.tp == tp) & t.num_tokens.isin(tokens)].sort_values("num_tokens")
    return g[FEATURE_COLS].to_numpy(dtype=float), g["y"].to_numpy(dtype=float), g["num_tokens"].to_numpy()


def _versions() -> Dict[str, str]:
    import scipy
    import sklearn

    v = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    try:
        import lightgbm

        v["lightgbm"] = lightgbm.__version__
    except Exception:
        pass
    return v


def _select(est, grid, X, y, folds, n_jobs) -> Tuple[Dict, float, int]:
    if not grid:
        return {}, float("nan"), 1
    gs = GridSearchCV(est, grid, scoring=mape_scorer, cv=folds, n_jobs=n_jobs, refit=False, error_score="raise")
    gs.fit(X, y)
    return dict(gs.best_params_), float(-gs.best_score_), len(gs.cv_results_["params"])


def _fold_score(est, X, y, tr, va) -> float:
    m = clone(est).fit(X[tr], y[tr])
    return mape(y[va], m.predict(X[va]))


def _write(out: Path, sel, cv, test, regime, preds) -> None:
    pd.DataFrame(sel).to_csv(out / "cv_selection.csv", index=False)
    pd.DataFrame(cv).to_csv(out / "cv_assessment.csv", index=False)
    pd.DataFrame(test).to_csv(out / "test_metrics.csv", index=False)
    if regime:
        pd.concat(regime, ignore_index=True).to_csv(out / "test_metrics_by_regime.csv", index=False)
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)


# ------------------------------------------------------------------------------------------------ commands


def cmd_make_splits(a) -> None:
    raw = load_raw(a.csv, verify=not a.no_verify)
    t = tidy(raw)
    spec = SplitSpec(holdout_frac=a.holdout_frac, test_frac=a.test_frac, seed=a.seed, band_rel_width=a.band_rel_width)
    m = write_splits(Path(a.splits), token_grid(t), spec, Path(a.csv), md5_of(Path(a.csv)))
    for g in GEOMETRIES:
        sizes = {k: v["n"] for k, v in m["geometries"][g]["files"].items()}
        print(f"{g:7s} {sizes}  bands={m['geometries'][g]['n_bands']}")
    print(f"sealed: {Path(a.splits) / 'manifest.json'}")


def cmd_run(a) -> None:
    raw = load_raw(a.csv, verify=not a.no_verify)
    t = tidy(raw)
    specs = [s for s in zoo(quick=a.quick) if not a.models or s.name in a.models]
    if a.models:
        unknown = set(a.models) - {s.name for s in specs}
        if unknown:
            raise SystemExit(f"unknown models: {sorted(unknown)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(a.splits)
    band_w = float(manifest["spec"]["band_rel_width"])
    geoms = list(GEOMETRIES) if a.geometry == "both" else [a.geometry]
    keys = regressor_keys(t)
    if a.ops:
        keys = [k for k in keys if k[0] in a.ops]

    sel_rows: List[Dict] = []
    cv_rows: List[Dict] = []
    test_rows: List[Dict] = []
    regime_frames: List[pd.DataFrame] = []
    pred_frames: List[pd.DataFrame] = []
    t_start = time.time()
    for geom in geoms:
        split = load_split(a.splits, geom)  # never touches the holdout file
        for op, tp in keys:
            Xtr, ytr, toktr = _subset(t, op, tp, split["train"])
            Xte, yte, tokte = _subset(t, op, tp, split["test"])
            sel_folds = cv_folds(toktr, geom, a.folds, seed=a.seed, band_rel_width=band_w)
            assess_folds = [cv_folds(toktr, geom, a.folds, seed=a.seed + 1000 + r, band_rel_width=band_w) for r in range(a.repeats)]
            for spec in specs:
                est, grid = spec.build()
                t0 = time.time()
                best, cv_sel, n_cand = _select(est, grid, Xtr, ytr, sel_folds, a.n_jobs)
                est.set_params(**best)
                sel_s = time.time() - t0
                sel_rows.append(dict(geometry=geom, op=op, tp=tp, model=spec.name, family=spec.family, n_candidates=n_cand,
                                     cv_select_mape=cv_sel, best_params=json.dumps(best, default=str), select_seconds=sel_s))
                for r, folds in enumerate(assess_folds):
                    scores = Parallel(n_jobs=a.n_jobs)(delayed(_fold_score)(est, Xtr, ytr, tr, va) for tr, va in folds)
                    cv_rows.extend(dict(geometry=geom, op=op, tp=tp, model=spec.name, repeat=r, fold=k, mape=s) for k, s in enumerate(scores))
                t0 = time.time()
                fitted = clone(est).fit(Xtr, ytr)
                fit_s = time.time() - t0
                pred = fitted.predict(Xte)
                test_rows.append(dict(geometry=geom, op=op, tp=tp, model=spec.name, family=spec.family, fit_seconds=fit_s, **summarize(yte, pred)))
                rg = by_regime(yte, pred, tokte)
                for col, val in (("model", spec.name), ("tp", tp), ("op", op), ("geometry", geom)):
                    rg.insert(0, col, val)
                regime_frames.append(rg)
                pred_frames.append(pd.DataFrame(dict(geometry=geom, op=op, tp=tp, model=spec.name, num_tokens=tokte, y=yte, pred=pred)))
                cv_m = np.mean([r["mape"] for r in cv_rows if r["model"] == spec.name and r["op"] == op and r["tp"] == tp and r["geometry"] == geom])
                print(f"[{geom:6s}] {op:25s} TP{tp} {spec.name:16s} cv={cv_m:6.2f}%  test={test_rows[-1]['mape']:6.2f}%  "
                      f"p95={test_rows[-1]['p95_ape']:6.2f}%  ({sel_s + fit_s:5.1f}s)", flush=True)
            _write(out, sel_rows, cv_rows, test_rows, regime_frames, pred_frames)  # checkpoint per regressor
    meta = dict(
        csv=str(a.csv), csv_md5=md5_of(Path(a.csv)), splits=str(a.splits), splits_manifest=manifest,
        models=[s.name for s in specs], quick=a.quick, folds=a.folds, repeats=a.repeats, seed=a.seed,
        geometries=geoms, ops=[k[0] for k in keys], n_jobs=a.n_jobs, wall_seconds=time.time() - t_start,
        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), versions=_versions(),
    )
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"done in {meta['wall_seconds']:.0f}s -> {out}")


def cmd_holdout(a) -> None:
    if not a.unseal:
        raise SystemExit("Refusing to read the holdout tier without --unseal. This is meant to be done once, "
                         "after model selection is frozen.")
    results = Path(a.results)
    sel = pd.read_csv(results / "cv_selection.csv")
    if a.models:
        sel = sel[sel.model.isin(a.models)]
    raw = load_raw(a.csv, verify=not a.no_verify)
    t = tidy(raw)
    specs = {s.name: s for s in zoo(quick=json.loads((results / "run_meta.json").read_text()).get("quick", False))}
    rows, regime_frames, pred_frames = [], [], []
    for geom in sorted(sel.geometry.unique()):
        split = load_split(a.splits, geom, include_holdout=True)
        fit_tokens = np.concatenate([split["train"], split["test"]])
        for rec in sel[sel.geometry == geom].itertuples(index=False):
            Xfit, yfit, _ = _subset(t, rec.op, rec.tp, fit_tokens)
            Xho, yho, tokho = _subset(t, rec.op, rec.tp, split["holdout"])
            est, _ = specs[rec.model].build()
            est.set_params(**json.loads(rec.best_params))
            pred = clone(est).fit(Xfit, yfit).predict(Xho)
            rows.append(dict(geometry=geom, op=rec.op, tp=rec.tp, model=rec.model, family=rec.family, **summarize(yho, pred)))
            rg = by_regime(yho, pred, tokho)
            for col, val in (("model", rec.model), ("tp", rec.tp), ("op", rec.op), ("geometry", geom)):
                rg.insert(0, col, val)
            regime_frames.append(rg)
            pred_frames.append(pd.DataFrame(dict(geometry=geom, op=rec.op, tp=rec.tp, model=rec.model, num_tokens=tokho, y=yho, pred=pred)))
            print(f"[{geom:6s}] {rec.op:25s} TP{rec.tp} {rec.model:16s} holdout={rows[-1]['mape']:6.2f}%  p95={rows[-1]['p95_ape']:6.2f}%", flush=True)
    pd.DataFrame(rows).to_csv(results / "holdout_metrics.csv", index=False)
    pd.concat(regime_frames, ignore_index=True).to_csv(results / "holdout_metrics_by_regime.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(results / "holdout_predictions.csv", index=False)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with open(Path(a.splits) / "HOLDOUT_LOG.md", "a") as f:
        f.write(f"- {stamp}  results={results}  geometries={sorted(sel.geometry.unique())}  models={sorted(sel.model.unique())}  "
                f"regressors={sel[['op', 'tp']].drop_duplicates().shape[0]}\n")
    print(f"holdout scored once; logged in {Path(a.splits) / 'HOLDOUT_LOG.md'}")


def cmd_report(a) -> None:
    from regressor_bench.report import build_report

    out = build_report(Path(a.results), top=a.top, csv=None if a.no_noise else Path(a.csv), verify=not a.no_verify)
    print(f"report -> {out}")


# ---------------------------------------------------------------------------------------------------- main


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--no-verify", action="store_true", help="skip the md5 check of the input CSV")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-splits", help="create and seal train/test/holdout token splits (both geometries)")
    s.add_argument("--splits", default=str(DEFAULT_SPLITS))
    s.add_argument("--holdout-frac", type=float, default=SplitSpec.holdout_frac)
    s.add_argument("--test-frac", type=float, default=SplitSpec.test_frac)
    s.add_argument("--seed", type=int, default=SplitSpec.seed)
    s.add_argument("--band-rel-width", type=float, default=SplitSpec.band_rel_width,
                   help="block geometry: relative width of a held-out band, e.g. 0.10 = tokens t..1.1t")
    s.set_defaults(fn=cmd_make_splits)

    s = sub.add_parser("run", help="select, assess and test every model on train/test")
    s.add_argument("--splits", default=str(DEFAULT_SPLITS))
    s.add_argument("--out", required=True)
    s.add_argument("--geometry", choices=list(GEOMETRIES) + ["both"], default="both")
    s.add_argument("--models", nargs="*", default=None)
    s.add_argument("--ops", nargs="*", default=None)
    s.add_argument("--quick", action="store_true", help="one grid point per model (smoke run)")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-jobs", type=int, default=6)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("holdout", help="score the sealed holdout once with the selected configurations")
    s.add_argument("--splits", default=str(DEFAULT_SPLITS))
    s.add_argument("--results", required=True)
    s.add_argument("--models", nargs="*", default=None)
    s.add_argument("--unseal", action="store_true")
    s.set_defaults(fn=cmd_holdout)

    s = sub.add_parser("report", help="summary tables and figures for a results directory")
    s.add_argument("--results", required=True)
    s.add_argument("--top", type=int, default=3, help="models shown in regime tables and residual plots")
    s.add_argument("--no-noise", action="store_true", help="skip the label-noise-floor table (does not read the CSV)")
    s.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
