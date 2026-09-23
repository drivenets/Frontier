"""Metrics match Frontier's MAPE; every zoo member fits and predicts a synthetic roofline-with-steps curve."""
import json

import numpy as np
import pytest

from regressor_bench.dataset import reference_token_grid
from regressor_bench.metrics import ape, by_regime, mape, summarize
from regressor_bench.models import InterpRegressor, RooflineRegressor, zoo


def test_mape_matches_frontier_definition():
    y = np.array([1.0, 2.0, 0.0, 4.0])
    p = np.array([1.1, 1.8, 0.5, 4.0])
    # Frontier: zero labels contribute 0 to the mean, not skipped
    assert mape(y, p) == pytest.approx((10 + 10 + 0 + 0) / 4)
    assert ape(y, p).tolist() == pytest.approx([10, 10, 0, 0])


def test_summarize_and_regimes():
    tok = reference_token_grid()
    y = 0.01 + 3e-5 * tok
    p = y * 1.02
    s = summarize(y, p)
    assert s["mape"] == pytest.approx(2.0) and s["bias_pct"] == pytest.approx(2.0)
    r = by_regime(y, p, tok)
    assert list(r.regime) == ["1-64", "65-512", "513-2048", "2049-8192", "8193-16384"]
    assert r.n.sum() == len(tok)


def _synthetic():
    x = reference_token_grid().astype(float)
    rng = np.random.default_rng(0)
    y = 0.012 + 3.5e-5 * np.maximum(0, x - 300)
    y = y * np.where(x > 5000, 1.2, 1.0)  # a tile step
    y = y * (1 + rng.normal(0, 0.005, len(x)))
    return x.reshape(-1, 1), y


def test_interp_reproduces_training_points():
    X, y = _synthetic()
    m = InterpRegressor().fit(X, y)
    assert np.allclose(m.predict(X), y)


def test_roofline_recovers_parameters():
    x = reference_token_grid().astype(float)
    y = 0.012 + 3.5e-5 * np.maximum(0, x - 300)
    m = RooflineRegressor().fit(x.reshape(-1, 1), y)
    assert m.c_ == pytest.approx(0.012, rel=1e-3)
    assert m.b_ == pytest.approx(3.5e-5, rel=1e-3)
    assert m.n0_ == pytest.approx(300, abs=2)


@pytest.mark.parametrize("spec", zoo(quick=True), ids=lambda s: s.name)
def test_every_zoo_member_fits(spec):
    X, y = _synthetic()
    est, grid = spec.build()
    params = {k: v[0] for k, v in grid.items()}
    est.set_params(**params)
    json.dumps(params, default=str)  # best_params must be serialisable
    tr = np.arange(len(X)) % 5 != 0
    est.fit(X[tr], y[tr])
    p = est.predict(X[~tr])
    assert p.shape == (int((~tr).sum()),)
    assert np.isfinite(p).all()
    assert mape(y[~tr], p) < 30.0, spec.name
