"""Train / test / sealed-holdout splits on the ``num_tokens`` grid.

Why split on the token value and not on the row: the 15 regressors share the same 3,327 token values, so
holding out the *same* token values for every regressor keeps the comparison paired and makes it impossible
for a value to be "seen" through a sibling regressor. There are no repeated measurements to leak (one row per
cell), so the grouping unit is the token value itself.

Two geometries are produced from the same seed:

* ``random`` - token values sampled at random, stratified by log2(num_tokens) so every size regime is
  represented in every tier. On a dense grid this measures interpolation between immediate neighbours.
* ``block``  - the token axis is cut into contiguous bands of fixed *relative* width (``band_rel_width``, e.g.
  10 %: 1000-1100, 8000-8800, ...) and whole bands are held out. This measures how a method fills a gap in the
  profiling grid, which is the harder and more deployment-relevant question (a sparser grid, or a query between
  two measured shapes). Relative width keeps the gap equally hard at 100 and at 10,000 tokens; at the very low
  end (< ~10 tokens) a band is a single grid point, so block degenerates to random there, as it should.

Tiers: ``holdout`` (sealed; read only by the ``holdout`` command), ``test`` (used once per model for the final
numbers), ``train`` (all selection and tuning happen inside it with cross-validation).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .dataset import sha256_of

GEOMETRIES: Tuple[str, ...] = ("random", "block")
TIERS: Tuple[str, ...] = ("train", "test", "holdout")


@dataclass(frozen=True)
class SplitSpec:
    holdout_frac: float = 0.10
    test_frac: float = 0.15
    seed: int = 20260923
    band_rel_width: float = 0.10  # relative width of a contiguous band (block geometry): [t, t*(1+w))
    n_strata: int = 12  # log2 bins before merging small ones
    min_stratum_size: int = 40  # tokens per stratum (random geometry) ...
    min_stratum_bands: int = 4  # ... or bands per stratum (block geometry)


# ----------------------------------------------------------------------------------------------- strata


def log_strata(values: np.ndarray, n_strata: int, min_size: int) -> np.ndarray:
    """Equal-width bins in log2 space, merged left-to-right until each holds at least ``min_size`` items."""
    values = np.asarray(values, dtype=float)
    lo, hi = np.log2(values.min()), np.log2(values.max())
    edges = np.linspace(lo, hi, n_strata + 1)
    raw = np.clip(np.digitize(np.log2(values), edges[1:-1]), 0, n_strata - 1)
    labels = np.zeros_like(raw)
    current, count = 0, 0
    for b in range(n_strata):
        mask = raw == b
        labels[mask] = current
        count += int(mask.sum())
        if count >= min_size:
            current, count = current + 1, 0
    if 0 < count < min_size and current > 0:  # trailing small bin -> merge into the previous stratum
        labels[labels == current] = current - 1
    return labels


def _allocate(n: int, holdout_frac: float, test_frac: float) -> Tuple[int, int]:
    return int(round(n * holdout_frac)), int(round(n * test_frac))


# ----------------------------------------------------------------------------------------------- random


def make_random_split(tokens: np.ndarray, spec: SplitSpec) -> pd.DataFrame:
    tokens = np.sort(np.asarray(tokens))
    rng = np.random.default_rng(spec.seed)
    strata = log_strata(tokens, spec.n_strata, spec.min_stratum_size)
    tier = np.array(["train"] * len(tokens), dtype=object)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        n_hold, n_test = _allocate(len(idx), spec.holdout_frac, spec.test_frac)
        tier[idx[:n_hold]] = "holdout"
        tier[idx[n_hold : n_hold + n_test]] = "test"
    return pd.DataFrame({"num_tokens": tokens, "stratum": strata, "band": -1, "tier": tier})


# ----------------------------------------------------------------------------------------------- block


def band_ids(tokens: np.ndarray, rel_width: float) -> np.ndarray:
    """Band index of each token value: consecutive integers, band k covers [(1+w)^k, (1+w)^(k+1))."""
    tokens = np.asarray(tokens, dtype=float)
    raw = np.floor(np.log(tokens) / np.log1p(rel_width) + 1e-9).astype(int)
    _, dense = np.unique(raw, return_inverse=True)
    return dense


def make_block_split(tokens: np.ndarray, spec: SplitSpec) -> pd.DataFrame:
    tokens = np.sort(np.asarray(tokens))
    rng = np.random.default_rng(spec.seed + 1)
    bands = band_ids(tokens, spec.band_rel_width)
    uniq = np.unique(bands)
    band_mid = np.array([np.median(tokens[bands == b]) for b in uniq])
    band_strata = log_strata(band_mid, spec.n_strata, spec.min_stratum_bands)
    band_size = np.bincount(bands)
    band_tier = {int(b): "train" for b in uniq}
    for s in np.unique(band_strata):
        idx = uniq[band_strata == s].copy()
        rng.shuffle(idx)
        # Bands hold very different numbers of grid points (1 at the low end, ~200 near 2048), so allocate
        # whole bands greedily until the tier holds its share of the stratum's *tokens*, not its bands.
        total = float(band_size[idx].sum())
        cum = 0.0
        for b in idx:
            size = float(band_size[b])
            if cum + size / 2 <= spec.holdout_frac * total:
                band_tier[int(b)] = "holdout"
            elif cum + size / 2 <= (spec.holdout_frac + spec.test_frac) * total:
                band_tier[int(b)] = "test"
            else:
                break
            cum += size
    tier = np.array([band_tier[int(b)] for b in bands], dtype=object)
    stratum = np.array([band_strata[int(b)] for b in bands])
    return pd.DataFrame({"num_tokens": tokens, "stratum": stratum, "band": bands, "tier": tier})


def make_split(tokens: np.ndarray, geometry: str, spec: SplitSpec) -> pd.DataFrame:
    if geometry == "random":
        return make_random_split(tokens, spec)
    if geometry == "block":
        return make_block_split(tokens, spec)
    raise ValueError(f"unknown geometry {geometry!r}; choose from {GEOMETRIES}")


# ----------------------------------------------------------------------------------------------- I/O


def write_splits(
    root: Path, tokens: np.ndarray, spec: SplitSpec, source_csv: Path, source_md5: str
) -> Dict:
    """Write ``<root>/<geometry>/{train,test,holdout}_tokens.csv`` + ``assignment.csv`` and a manifest.

    The holdout file is written once and its sha256 recorded; ``load_split`` refuses to hand it out unless
    asked explicitly, and the ``holdout`` command logs every use.
    """
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError(
            f"{root}/manifest.json exists; splits are sealed. Choose another --splits directory instead of "
            "overwriting (re-splitting after looking at results invalidates the holdout)."
        )
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_csv": str(source_csv),
        "source_md5": source_md5,
        "spec": asdict(spec),
        "n_tokens": int(len(tokens)),
        "geometries": {},
    }
    for geom in GEOMETRIES:
        gdir = root / geom
        gdir.mkdir(exist_ok=True)
        assign = make_split(tokens, geom, spec)
        assign.to_csv(gdir / "assignment.csv", index=False)
        files = {}
        for tier in TIERS:
            f = gdir / f"{tier}_tokens.csv"
            assign.loc[assign.tier == tier, ["num_tokens"]].to_csv(f, index=False)
            files[tier] = {
                "path": str(f.relative_to(root)),
                "n": int((assign.tier == tier).sum()),
                "sha256": sha256_of(f),
            }
        manifest["geometries"][geom] = {
            "files": files,
            "n_bands": int(assign.band.nunique()) if geom == "block" else None,
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(root: Path) -> Dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def load_split(root: Path, geometry: str, include_holdout: bool = False) -> Dict[str, np.ndarray]:
    """Return ``{'train': tokens, 'test': tokens[, 'holdout': tokens]}``; verifies sha256 against the manifest."""
    root = Path(root)
    manifest = load_manifest(root)
    tiers = ["train", "test"] + (["holdout"] if include_holdout else [])
    out = {}
    for tier in tiers:
        entry = manifest["geometries"][geometry]["files"][tier]
        f = root / entry["path"]
        if sha256_of(f) != entry["sha256"]:
            raise ValueError(f"{f} does not match the manifest sha256; splits were modified after sealing")
        out[tier] = pd.read_csv(f)["num_tokens"].to_numpy()
    return out


def load_assignment(root: Path, geometry: str) -> pd.DataFrame:
    return pd.read_csv(Path(root) / geometry / "assignment.csv")


# ----------------------------------------------------------------------------------------------- CV folds


def cv_folds(
    train_tokens: np.ndarray, geometry: str, n_splits: int, seed: int, band_rel_width: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Index folds over ``train_tokens`` (in the order given) that mirror the split geometry.

    random -> shuffled K-fold on token values; block -> whole relative-width bands per fold, bands shuffled
    across folds. Returned as (train_idx, valid_idx) pairs usable as GridSearchCV ``cv``.
    """
    train_tokens = np.asarray(train_tokens)
    rng = np.random.default_rng(seed)
    if geometry == "random":
        perm = rng.permutation(len(train_tokens))
        chunks = np.array_split(perm, n_splits)
    elif geometry == "block":
        bands = band_ids(train_tokens, band_rel_width)
        uniq = rng.permutation(np.unique(bands))
        band_chunks = np.array_split(uniq, n_splits)
        chunks = [np.flatnonzero(np.isin(bands, bc)) for bc in band_chunks]
    else:
        raise ValueError(geometry)
    folds = []
    all_idx = np.arange(len(train_tokens))
    for valid in chunks:
        valid = np.sort(valid)
        train = np.setdiff1d(all_idx, valid, assume_unique=True)
        folds.append((train, valid))
    return folds
