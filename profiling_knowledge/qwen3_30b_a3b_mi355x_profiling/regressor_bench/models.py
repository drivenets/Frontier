"""Model zoo: every candidate is an sklearn estimator over the single feature ``num_tokens``.

Each :class:`ModelSpec` carries a factory, a (small) hyper-parameter grid searched on the training tier only,
and an optional log-target switch (fit on log(y), predict exp). Frontier's two production configurations
(``rf_frontier``, ``polylr_frontier``) are reproduced with the simulator's own grids so the comparison has a
faithful "what we ship today" row.

Baselines that a candidate must beat to be interesting: ``interp`` (piecewise-linear interpolation of the
training grid) and ``isotonic``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, SplineTransformer

try:  # optional
    from lightgbm import LGBMRegressor

    HAVE_LGBM = True
except Exception:  # pragma: no cover
    HAVE_LGBM = False

FEATURE_COLS = ["num_tokens"]
TOKEN_SCALE = 16384.0


def _x1d(X) -> np.ndarray:
    return np.asarray(X, dtype=float).reshape(-1)


# ------------------------------------------------------------------------------------------ custom estimators


class InterpRegressor(BaseEstimator, RegressorMixin):
    """Piecewise-linear interpolation of the training grid, clamped at both ends. The baseline to beat."""

    def fit(self, X, y):
        x, y = _x1d(X), np.asarray(y, dtype=float)
        order = np.argsort(x)
        ux, inv = np.unique(x[order], return_inverse=True)
        uy = np.bincount(inv, weights=y[order]) / np.bincount(inv)  # average duplicate x
        self.x_, self.y_ = ux, uy
        return self

    def predict(self, X):
        return np.interp(_x1d(X), self.x_, self.y_)


class RooflineRegressor(BaseEstimator, RegressorMixin):
    """``y = c + b * max(0, n - n0)``: fixed cost while the kernel is weight-read / launch bound, then linear.

    Three physical parameters, fitted on *relative* residuals so small and large labels weigh equally
    (MAPE-shaped objective). ``loss`` is passed to ``scipy.optimize.least_squares``.
    """

    def __init__(self, loss: str = "linear"):
        self.loss = loss

    def fit(self, X, y):
        x, y = _x1d(X), np.asarray(y, dtype=float)
        c0 = float(np.median(y[x <= np.quantile(x, 0.05)]))
        hi = x >= np.quantile(x, 0.5)
        slope, intercept = np.polyfit(x[hi], y[hi], 1)
        b0 = max(float(slope), 1e-9)
        n0 = float(np.clip((c0 - intercept) / b0, 0.0, x.max()))

        def resid(p):
            c, b, k = p
            return (c + b * np.maximum(0.0, x - k)) / y - 1.0

        res = least_squares(
            resid, x0=[c0, b0, n0], bounds=([0, 0, 0], [np.inf, np.inf, float(x.max())]), loss=self.loss
        )
        self.c_, self.b_, self.n0_ = (float(v) for v in res.x)
        return self

    def predict(self, X):
        return self.c_ + self.b_ * np.maximum(0.0, _x1d(X) - self.n0_)


class LinearPlusTreeRegressor(BaseEstimator, RegressorMixin):
    """Linear trend in ``num_tokens`` plus gradient-boosted trees on the residual.

    The linear part carries the roofline slope; the trees only have to learn the flat start and the hipBLASLt
    tile steps, which are local. Trees cannot extrapolate, the linear part can.
    """

    def __init__(self, max_leaf_nodes: int = 15, min_samples_leaf: int = 10, max_iter: int = 300, learning_rate: float = 0.1):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X2 = np.asarray(X, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float)
        self.lin_ = LinearRegression().fit(X2 / TOKEN_SCALE, y)
        resid = y - self.lin_.predict(X2 / TOKEN_SCALE)
        self.tree_ = HistGradientBoostingRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=0,
        ).fit(X2, resid)
        return self

    def predict(self, X):
        X2 = np.asarray(X, dtype=float).reshape(-1, 1)
        return self.lin_.predict(X2 / TOKEN_SCALE) + self.tree_.predict(X2)


# ------------------------------------------------------------------------------------------------- the zoo


@dataclass
class ModelSpec:
    name: str
    family: str
    make: Callable[[], BaseEstimator]
    grid: Dict[str, List[Any]] = field(default_factory=dict)
    log_target: bool = False
    note: str = ""

    def build(self) -> Tuple[BaseEstimator, Dict[str, List[Any]]]:
        """Estimator (wrapped for log-target if requested) and its grid with matching parameter prefixes."""
        est = self.make()
        grid = dict(self.grid)
        if self.log_target:
            est = TransformedTargetRegressor(regressor=est, func=np.log, inverse_func=np.exp)
            grid = {f"regressor__{k}": v for k, v in grid.items()}
        return est, grid

    def n_candidates(self) -> int:
        n = 1
        for v in self.grid.values():
            n *= len(v)
        return n


def _poly_ridge():
    return make_pipeline(MinMaxScaler(), PolynomialFeatures(), Ridge())


def _spline(degree: int):
    return make_pipeline(
        SplineTransformer(degree=degree, knots="quantile", extrapolation="linear"), Ridge(alpha=1e-6)
    )


def zoo(quick: bool = False) -> List[ModelSpec]:
    """All candidates. ``quick`` collapses the grids to one point each for smoke runs."""
    if quick:
        rf_grid = {"n_estimators": [250], "max_depth": [16], "min_samples_split": [5]}
        hgb_grid = {"max_iter": [300], "learning_rate": [0.1], "max_leaf_nodes": [15], "min_samples_leaf": [10]}
        lgbm_grid = {"n_estimators": [300], "learning_rate": [0.1], "num_leaves": [15], "min_child_samples": [10]}
        polylr_grid = {"polynomialfeatures__degree": [3], "polynomialfeatures__include_bias": [True], "linearregression__fit_intercept": [True]}
        poly_ridge_grid = {"polynomialfeatures__degree": [4], "ridge__alpha": [1e-4]}
        spl_lin_grid = {"splinetransformer__n_knots": [32]}
        spl_cub_grid = {"splinetransformer__n_knots": [16]}
        knn_grid = {"n_neighbors": [3], "weights": ["distance"]}
        et_grid = {"min_samples_leaf": [2]}
        lpt_grid = {"max_leaf_nodes": [15], "min_samples_leaf": [10]}
    else:
        # Frontier's RandomForrestExecutionTimePredictorConfig defaults.
        rf_grid = {"n_estimators": [250, 500, 750], "max_depth": [8, 16, 32], "min_samples_split": [2, 5, 10]}
        hgb_grid = {"max_iter": [300, 800], "learning_rate": [0.05, 0.1], "max_leaf_nodes": [7, 15, 31], "min_samples_leaf": [3, 10, 20]}
        lgbm_grid = {"n_estimators": [300, 800], "learning_rate": [0.05, 0.1], "num_leaves": [7, 15, 31], "min_child_samples": [3, 10, 20]}
        # Frontier's LinearRegressionExecutionTimePredictorConfig defaults (interaction_only is a no-op with one feature).
        polylr_grid = {"polynomialfeatures__degree": [1, 2, 3, 4, 5], "polynomialfeatures__include_bias": [True, False], "linearregression__fit_intercept": [True, False]}
        poly_ridge_grid = {"polynomialfeatures__degree": [2, 3, 4, 6, 8], "ridge__alpha": [1e-6, 1e-4, 1e-2]}
        spl_lin_grid = {"splinetransformer__n_knots": [8, 16, 32, 64, 128]}
        spl_cub_grid = {"splinetransformer__n_knots": [8, 16, 32, 64]}
        knn_grid = {"n_neighbors": [1, 2, 3, 5, 8], "weights": ["uniform", "distance"]}
        et_grid = {"min_samples_leaf": [1, 2, 5]}
        lpt_grid = {"max_leaf_nodes": [7, 15], "min_samples_leaf": [5, 20]}

    specs = [
        ModelSpec("interp", "baseline", InterpRegressor, note="piecewise-linear interpolation of the training grid"),
        ModelSpec("isotonic", "baseline", lambda: IsotonicRegression(out_of_bounds="clip"), note="monotone non-parametric fit"),
        ModelSpec("knn", "neighbours", KNeighborsRegressor, knn_grid),
        ModelSpec("rf_frontier", "trees", lambda: RandomForestRegressor(random_state=0, n_jobs=1), rf_grid, note="Frontier RandomForrest predictor, simulator grid"),
        ModelSpec("extratrees", "trees", lambda: ExtraTreesRegressor(n_estimators=300, random_state=0, n_jobs=1), et_grid),
        ModelSpec("hgb", "boosting", lambda: HistGradientBoostingRegressor(random_state=0), hgb_grid),
        ModelSpec("hgb_log", "boosting", lambda: HistGradientBoostingRegressor(random_state=0), hgb_grid, log_target=True),
        ModelSpec("polylr_frontier", "polynomial", lambda: make_pipeline(PolynomialFeatures(), LinearRegression()), polylr_grid, note="Frontier LinearRegression predictor on raw num_tokens (unscaled, as shipped)"),
        ModelSpec("poly_ridge", "polynomial", _poly_ridge, poly_ridge_grid),
        ModelSpec("poly_ridge_log", "polynomial", _poly_ridge, poly_ridge_grid, log_target=True),
        ModelSpec("spline_lin", "spline", lambda: _spline(1), spl_lin_grid, note="piecewise-linear spline, quantile knots"),
        ModelSpec("spline_lin_log", "spline", lambda: _spline(1), spl_lin_grid, log_target=True),
        ModelSpec("spline_cub", "spline", lambda: _spline(3), spl_cub_grid, note="cubic spline, quantile knots"),
        ModelSpec("spline_cub_log", "spline", lambda: _spline(3), spl_cub_grid, log_target=True),
        ModelSpec("roofline", "physical", RooflineRegressor, note="c + b*max(0, n-n0), relative residuals"),
        ModelSpec("linear_plus_hgb", "hybrid", LinearPlusTreeRegressor, lpt_grid, note="linear trend + boosted trees on residual"),
    ]
    if HAVE_LGBM:
        specs.append(ModelSpec("lgbm", "boosting", lambda: LGBMRegressor(random_state=0, verbose=-1, n_jobs=1), lgbm_grid))
    return specs


def zoo_by_name(quick: bool = False) -> Dict[str, ModelSpec]:
    return {s.name: s for s in zoo(quick)}
