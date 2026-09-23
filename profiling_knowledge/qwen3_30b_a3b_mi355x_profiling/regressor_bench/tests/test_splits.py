"""Split invariants: disjoint tiers that cover the grid, every regime represented, block geometry band-aligned,
determinism, and the sealed-holdout guard. Uses the profiler grid rebuilt from its definition (no CSV needed)."""
import numpy as np
import pandas as pd
import pytest

from regressor_bench.dataset import EXPECTED_TOKEN_COUNT, reference_token_grid
from regressor_bench.splits import (
    GEOMETRIES,
    SplitSpec,
    band_ids,
    cv_folds,
    load_split,
    log_strata,
    make_split,
    write_splits,
)

GRID = reference_token_grid()


def test_reference_grid_matches_handoff():
    assert len(GRID) == EXPECTED_TOKEN_COUNT
    assert 4000 not in GRID and GRID.min() == 1 and GRID.max() == 16384


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_tiers_partition_the_grid(geometry):
    a = make_split(GRID, geometry, SplitSpec())
    assert sorted(a.num_tokens) == sorted(GRID)
    counts = a.tier.value_counts()
    assert set(counts.index) == {"train", "test", "holdout"}
    n = len(GRID)
    assert abs(counts["holdout"] / n - 0.10) < 0.03
    assert abs(counts["test"] / n - 0.15) < 0.03


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_every_regime_present_in_every_tier(geometry):
    a = make_split(GRID, geometry, SplitSpec())
    regime = pd.cut(a.num_tokens, [0, 64, 512, 2048, 8192, 16384])
    table = pd.crosstab(regime, a.tier)
    assert (table > 0).all().all(), table


def test_band_ids_have_fixed_relative_width():
    b = band_ids(GRID, 0.10)
    assert b[0] == 0 and np.all(np.diff(b) >= 0) and np.array_equal(np.unique(b), np.arange(b.max() + 1))
    df = pd.DataFrame({"t": GRID, "b": b})
    span = df.groupby("b").t.agg(["min", "max"])
    assert (span["max"] / span["min"] < 1.10 + 1e-9).all()
    # in the dense region a band holds ~10 % of its token value worth of grid points
    mid = span[(span["min"] > 1000) & (span["max"] < 2048)]
    counts = df[df.b.isin(mid.index)].groupby("b").size()
    assert counts.between(90, 210).all(), counts


def test_block_split_holds_out_whole_bands():
    spec = SplitSpec()
    a = make_split(GRID, "block", spec)
    assert (a.band.to_numpy() == band_ids(GRID, spec.band_rel_width)).all()
    per_band = a.groupby("band").tier.nunique()
    assert (per_band == 1).all()
    # every held-out band is a contiguous run of grid points that spans < 10 % in token value
    for b, g in a[a.tier != "train"].groupby("band"):
        idx = np.flatnonzero(np.isin(GRID, g.num_tokens))
        assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))
        assert g.num_tokens.max() / g.num_tokens.min() < 1.10 + 1e-9


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_split_is_deterministic(geometry):
    a = make_split(GRID, geometry, SplitSpec(seed=7))
    b = make_split(GRID, geometry, SplitSpec(seed=7))
    c = make_split(GRID, geometry, SplitSpec(seed=8))
    assert a.equals(b)
    assert not a.tier.equals(c.tier)


def test_log_strata_merges_small_bins():
    labels = log_strata(GRID, n_strata=12, min_size=40)
    sizes = pd.Series(labels).value_counts()
    assert (sizes >= 40).all()
    assert labels.min() == 0 and np.array_equal(np.unique(labels), np.arange(labels.max() + 1))


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_cv_folds_partition_train(geometry):
    a = make_split(GRID, geometry, SplitSpec())
    train = a.loc[a.tier == "train", "num_tokens"].to_numpy()
    folds = cv_folds(train, geometry, n_splits=5, seed=1, band_rel_width=0.10)
    assert len(folds) == 5
    valid = np.concatenate([va for _, va in folds])
    assert sorted(valid) == list(range(len(train)))
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0
    if geometry == "block":
        bands = band_ids(train, 0.10)
        for _, va in folds:  # a band is never split across the fold boundary
            for b in set(bands[va]):
                assert (bands[va] == b).sum() == (bands == b).sum()


def test_write_and_load_are_sealed(tmp_path):
    root = tmp_path / "splits"
    m = write_splits(root, GRID, SplitSpec(), source_csv="x.csv", source_md5="0" * 32)
    assert set(m["geometries"]) == set(GEOMETRIES)
    s = load_split(root, "random")
    assert set(s) == {"train", "test"}
    s2 = load_split(root, "random", include_holdout=True)
    assert len(s2["holdout"]) == m["geometries"]["random"]["files"]["holdout"]["n"]
    assert not np.intersect1d(s2["holdout"], s2["train"]).size
    with pytest.raises(FileExistsError):
        write_splits(root, GRID, SplitSpec(), source_csv="x.csv", source_md5="0" * 32)
    f = root / "random" / "holdout_tokens.csv"
    f.write_text(f.read_text() + "5\n")
    with pytest.raises(ValueError):
        load_split(root, "random", include_holdout=True)
