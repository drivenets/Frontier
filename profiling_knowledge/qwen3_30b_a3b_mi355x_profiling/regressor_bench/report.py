"""Summary tables (markdown) and figures (PNG) for a results directory produced by ``bench.py run``.

Leaderboard logic: a model's headline number is its mean test MAPE over the 15 regressors, but the table also
shows the worst regressor and the p95 APE, because a predictor that is 1 % on average and 40 % on one op is
not deployable. Cross-validation numbers (mean +- std across repeats x folds) sit next to the test numbers so
a test-set fluke is visible as a CV/test disagreement.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-3 validate all-pairs; chrome from the ink table.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BASELINE_MODELS = ("interp", "isotonic")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def leaderboard(test: pd.DataFrame, cv: pd.DataFrame, geom: str) -> pd.DataFrame:
    tg = test[test.geometry == geom]
    cg = cv[cv.geometry == geom]
    cv_stats = cg.groupby(["model", "op", "tp"]).mape.agg(["mean", "std"]).reset_index()
    rows = []
    for model, g in tg.groupby("model"):
        c = cv_stats[cv_stats.model == model]
        worst = g.loc[g.mape.idxmax()]
        rows.append(
            {
                "model": model,
                "family": g.family.iloc[0],
                "test MAPE mean": g.mape.mean(),
                "test MAPE median": g.mape.median(),
                "test MAPE worst": worst.mape,
                "worst regressor": f"{worst.op}/TP{worst.tp}",
                "test p95 APE mean": g.p95_ape.mean(),
                "test |bias| mean": g.bias_pct.abs().mean(),
                "CV MAPE mean": c["mean"].mean(),
                "CV MAPE std": c["std"].mean(),
                "wins": 0,
                "fit s": g.fit_seconds.mean(),
            }
        )
    lb = pd.DataFrame(rows)
    wins = tg.loc[tg.groupby(["op", "tp"]).mape.idxmin(), "model"].value_counts()
    lb["wins"] = lb.model.map(wins).fillna(0).astype(int)
    # paired comparison against interpolation on identical CV folds: mean difference in MAPE points and the
    # fraction of (regressor, repeat, fold) cells where the model is better
    if (cg.model == "interp").any():
        key = ["op", "tp", "repeat", "fold"]
        base = cg[cg.model == "interp"][key + ["mape"]].rename(columns={"mape": "base"})
        paired = cg.merge(base, on=key)
        paired["d"] = paired.mape - paired.base
        agg = paired.groupby("model").d.agg(["mean", lambda d: float((d < 0).mean())])
        agg.columns = ["CV dMAPE vs interp", "CV beats interp"]
        lb = lb.merge(agg.reset_index(), on="model", how="left")
    return lb.sort_values("test MAPE mean").reset_index(drop=True)


def label_noise(csv: Path, verify: bool = True) -> pd.DataFrame:
    """Relative standard error of the label (median of 25 samples) per regressor: the floor for MAPE claims."""
    from .dataset import load_raw, tidy

    t = tidy(load_raw(csv, verify=verify))
    t["rel_se_pct"] = 1.2533 * t.y_std / np.sqrt(t.n_samples.astype(float)) / t.y * 100.0
    t["rel_std_pct"] = t.y_std / t.y * 100.0
    g = t.groupby(["op", "tp"]).agg(**{"label rel std % (median)": ("rel_std_pct", "median"), "label rel SE % (median)": ("rel_se_pct", "median"), "label rel SE % (p95)": ("rel_se_pct", lambda s: s.quantile(0.95))})
    return g.reset_index()


def per_regressor(test: pd.DataFrame, geom: str, top: int = 3) -> pd.DataFrame:
    tg = test[test.geometry == geom]
    rows = []
    for (op, tp), g in tg.groupby(["op", "tp"]):
        g = g.sort_values("mape")
        base = g[g.model == "interp"].mape
        rows.append(
            {
                "op": op,
                "tp": tp,
                "best": g.model.iloc[0],
                "best MAPE": g.mape.iloc[0],
                "best p95": g.p95_ape.iloc[0],
                "runner-up": g.model.iloc[1] if len(g) > 1 else "",
                "runner-up MAPE": g.mape.iloc[1] if len(g) > 1 else np.nan,
                "interp MAPE": float(base.iloc[0]) if len(base) else np.nan,
                "rf_frontier MAPE": float(g[g.model == "rf_frontier"].mape.iloc[0]) if (g.model == "rf_frontier").any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def regime_table(regime: pd.DataFrame, geom: str, models: List[str]) -> pd.DataFrame:
    rg = regime[(regime.geometry == geom) & regime.model.isin(models)]
    piv = rg.groupby(["model", "regime"]).mape.mean().unstack("regime")
    order = [r for r in ["1-64", "65-512", "513-2048", "2049-8192", "8193-16384"] if r in piv.columns]
    piv = piv[order].loc[[m for m in models if m in piv.index]]
    return piv.reset_index().rename(columns={c: f"{c} tok" for c in order})


# --------------------------------------------------------------------------------------------------- figures


def fig_leaderboard(test: pd.DataFrame, lb: pd.DataFrame, geom: str, path: Path) -> None:
    tg = test[test.geometry == geom]
    order = lb.model.tolist()[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(order) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.grid(False)
    for i, m in enumerate(order):
        vals = tg[tg.model == m].mape
        ax.scatter(vals, np.full(len(vals), i), s=18, color=MUTED, alpha=0.55, linewidths=0, zorder=2)
        ax.scatter([vals.mean()], [i], s=46, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.text(vals.max() * 1.08, i, f"{vals.mean():.2f}%", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8, color=INK)
    ax.set_xscale("log")
    n_reg = tg[["op", "tp"]].drop_duplicates().shape[0]
    ax.set_xlabel(f"test MAPE per regressor (%, log scale); blue = mean over the {n_reg} regressors", color=INK2, fontsize=8)
    ax.set_title(f"Test-tier MAPE by model, {geom} split", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_residuals(preds: pd.DataFrame, geom: str, op: str, models: List[str], path: Path) -> None:
    pg = preds[(preds.geometry == geom) & (preds.op == op) & preds.model.isin(models)]
    tps = sorted(pg.tp.unique())
    fig, axes = plt.subplots(1, len(tps), figsize=(4.2 * len(tps), 3.4), dpi=150, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, tp in zip(axes[0], tps):
        _style(ax)
        for k, m in enumerate(models):
            d = pg[(pg.tp == tp) & (pg.model == m)].sort_values("num_tokens")
            if d.empty:
                continue
            spe = (d.pred - d.y) / d.y * 100.0
            # markers, not lines: in the block geometry the test tokens come in separated bands and a line
            # would draw fictitious error between them
            color = MUTED if m == "interp" else SERIES[k % len(SERIES)]
            ax.scatter(d.num_tokens, spe, s=7, color=color, alpha=0.8, linewidths=0, label=m, zorder=3 if m != "interp" else 2)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"TP{tp}", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("num_tokens", color=INK2, fontsize=8)
    axes[0][0].set_ylabel("signed % error on test tier", color=INK2, fontsize=8)
    axes[0][-1].legend(frameon=False, fontsize=7.5, loc="upper right", markerscale=2.5)
    fig.suptitle(f"{op}: signed prediction error vs num_tokens ({geom} split, test tier)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_curves(preds: pd.DataFrame, geom: str, op: str, model: str, path: Path) -> None:
    pg = preds[(preds.geometry == geom) & (preds.op == op) & (preds.model == model)]
    tps = sorted(pg.tp.unique())
    fig, axes = plt.subplots(1, len(tps), figsize=(4.2 * len(tps), 3.4), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, tp in zip(axes[0], tps):
        _style(ax)
        d = pg[pg.tp == tp].sort_values("num_tokens")
        ax.scatter(d.num_tokens, d.y * 1e3, s=9, color=MUTED, linewidths=0, label="measured (test tier)")
        ax.plot(d.num_tokens, d.pred * 1e3, lw=1.4, color=SERIES[0], label=model)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"TP{tp}", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("num_tokens", color=INK2, fontsize=8)
    axes[0][0].set_ylabel("median time (us)", color=INK2, fontsize=8)
    axes[0][-1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(f"{op}: {model} on the test tier ({geom} split)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- report


def build_report(results: Path, top: int = 3, csv: Optional[Path] = None, verify: bool = True) -> Path:
    results = Path(results)
    test = pd.read_csv(results / "test_metrics.csv")
    cv = pd.read_csv(results / "cv_assessment.csv")
    regime = pd.read_csv(results / "test_metrics_by_regime.csv")
    preds = pd.read_csv(results / "test_predictions.csv")
    sel = pd.read_csv(results / "cv_selection.csv")
    holdout: Optional[pd.DataFrame] = None
    if (results / "holdout_metrics.csv").exists():
        holdout = pd.read_csv(results / "holdout_metrics.csv")
    figs = results / "figs"
    figs.mkdir(exist_ok=True)

    parts = [f"# Regressor benchmark: {results.name}", ""]
    parts.append("Numbers are MAPE in percent on the ms-scale label unless stated. `wins` = regressors (op/TP) where the model has the "
                 "lowest test MAPE. `CV dMAPE vs interp` = mean (model - interp) MAPE on identical folds, negative is better; "
                 "`CV beats interp` = share of folds where the model is lower.")
    if csv is not None:
        ln = label_noise(csv, verify=verify)
        parts += ["", "## Label noise floor", "",
                  "Relative standard error of the 25-sample median per regressor. A MAPE difference between two models that is "
                  "smaller than this is not resolvable by this dataset.", "", _md(ln)]
    for geom in sorted(test.geometry.unique()):
        lb = leaderboard(test, cv, geom)
        top_models = lb.model.tolist()[:top]
        n_reg = test[test.geometry == geom][["op", "tp"]].drop_duplicates().shape[0]
        parts += [f"\n## Geometry: {geom}", "", f"### Leaderboard (mean over the {n_reg} regressors)", "", _md(lb)]
        parts += ["", "### Best model per regressor", "", _md(per_regressor(test, geom))]
        parts += ["", f"### MAPE by token regime (top {top} + baselines)", "", _md(regime_table(regime, geom, top_models + [m for m in BASELINE_MODELS if m not in top_models]))]
        chosen = sel[(sel.geometry == geom) & sel.model.isin(top_models)][["model", "op", "tp", "best_params", "cv_select_mape"]]
        parts += ["", f"### Selected hyper-parameters (top {top})", "", _md(chosen.sort_values(["model", "op", "tp"]))]
        if holdout is not None and (holdout.geometry == geom).any():
            hg = holdout[holdout.geometry == geom]
            h = hg.groupby("model").agg(**{"holdout MAPE mean": ("mape", "mean"), "holdout MAPE worst": ("mape", "max"), "holdout p95 mean": ("p95_ape", "mean")}).reset_index()
            h = h.merge(lb[["model", "test MAPE mean"]], on="model").sort_values("holdout MAPE mean")
            parts += ["", "### Sealed holdout (scored once)", "", _md(h)]
        fig_leaderboard(test, lb, geom, figs / f"leaderboard_{geom}.png")
        parts += ["", f"![leaderboard]({figs.name}/leaderboard_{geom}.png)"]
        for op in sorted(test.op.unique()):
            series = top_models + [m for m in ("interp",) if m not in top_models]
            fig_residuals(preds, geom, op, series, figs / f"residuals_{geom}_{op}.png")
            fig_curves(preds, geom, op, top_models[0], figs / f"curves_{geom}_{op}_{top_models[0]}.png")
            parts += [f"![residuals {op}]({figs.name}/residuals_{geom}_{op}.png)"]
    out = results / "summary.md"
    out.write_text("\n".join(parts) + "\n")
    return out
