"""Error metrics on the original (ms) scale, plus a breakdown by token regime.

MAPE follows Frontier's ``SklearnExecutionTimePredictor.mean_absolute_percentage_error`` (percent, zero labels
skipped) so numbers are comparable with the simulator's training log. Percentage errors on labels below ~10 us
are inflated by the +-3-5 us instrument floor (handoff s5); the regime table makes that visible instead of
hiding it.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

REGIMES: List[Tuple[str, int, int]] = [
    ("1-64", 1, 64),
    ("65-512", 65, 512),
    ("513-2048", 513, 2048),
    ("2049-8192", 2049, 8192),
    ("8193-16384", 8193, 16384),
]
INSTRUMENT_FLOOR_MS = 0.010


def ape(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = np.zeros_like(y_true)
    nz = y_true != 0
    out[nz] = np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]) * 100.0
    return out


def mape(y_true, y_pred) -> float:
    return float(np.mean(ape(y_true, y_pred)))


mape_scorer = make_scorer(mape, greater_is_better=False)


def summarize(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    a = ape(y_true, y_pred)
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mape": float(a.mean()),
        "mdape": float(np.median(a)),
        "p95_ape": float(np.percentile(a, 95)),
        "max_ape": float(a.max()),
        "frac_ape_gt5": float(np.mean(a > 5.0)),
        "bias_pct": float(np.mean(err / y_true) * 100.0),
        "mae_ms": float(np.mean(np.abs(err))),
        "rmse_ms": float(np.sqrt(np.mean(err**2))),
    }


def regime_of(tokens) -> np.ndarray:
    tokens = np.asarray(tokens)
    lab = np.empty(len(tokens), dtype=object)
    for name, lo, hi in REGIMES:
        lab[(tokens >= lo) & (tokens <= hi)] = name
    return lab


def by_regime(y_true, y_pred, tokens) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "regime": regime_of(tokens)})
    rows = []
    for name, _, _ in REGIMES:
        g = df[df.regime == name]
        if len(g) == 0:
            continue
        rows.append({"regime": name, **summarize(g.y, g.p)})
    return pd.DataFrame(rows)
