#!/usr/bin/env python3
"""Regenerate the figures of STORY.md (the Qwen3-30B-A3B MI355X linear_op profiling storyline).

Usage:
    python3 make_story_figures.py [--cluster-data DIR] [--out DIR]

Inputs
- repo data:   data/profiling/compute/mi355x/qwen3-a3b-30b-moe/{linear_op.csv, runs/*}, data/profiling_gcfix/
- cluster data (not in git; rsync from /opt/shared/frontier-qwen3-profiling/Frontier/ on the Memphis cluster):
      <DIR>/profiling_dense_fixed/linear_op.csv               job 21483 (25 fwd, all fixes)
      <DIR>/profiling_dense200/linear_op.csv                  job 21486 (200 fwd = 8 spun blocks of 25)
      <DIR>/profiling_dense_fixed_batchflush/linear_op.csv    job 21500 (+ DEBUG_CLR_MAX_BATCH_SIZE=1000000)
      <DIR>/profiling_dip_x1/linear_op.csv                    job 21356 (FRONTIER_GPU_BACKLOG_MS=100 experiment X1)
      <DIR>/spike_diag_e1b/worker_gpu*_pid*.jsonl             job 21346 (GC diagnostics)
      <DIR>/posprobe/R15_clock_s25_tp8.jsonl, R17/R19 ...     job 21494 (run-position probes)
Figures that need a missing input are skipped with a note. Only pandas + matplotlib are required (system python3).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
D = os.path.join(REPO, "data/profiling/compute/mi355x/qwen3-a3b-30b-moe")

# ---- palette (dataviz skill reference instance) -------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CRITICAL = "#d03b3b"          # status: the anomaly being discussed
TP_COLOR = {1: BLUE, 2: ORANGE, 4: AQUA, 8: YELLOW}   # fixed order, never cycled
# entity colours used consistently across every comparison figure:
C_OLD = ORANGE      # the original / legacy / host-bound measurement
C_NEW = BLUE        # the fixed / GPU-bound measurement
C_TRACE = AQUA      # rocprofv3 / kineto kernel time
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal",
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.constrained_layout.use": True, "legend.fontsize": 9, "text.color": INK,
})

OPS3 = ["attn_pre_proj", "attn_post_proj", "attn_rope"]
TPS = [1, 2, 4, 8]


def load(path, **kw):
    return pd.read_csv(path, low_memory=False, float_precision="round_trip", **kw)


def samples(df, op, prefix="time_stats"):
    """Timed samples per row (warm-ups dropped) as a list of np arrays, aligned with df.index."""
    col, wc = f"{prefix}.{op}.samples", f"{prefix}.{op}.warmup_count"
    out = []
    for s, w in zip(df[col], df[wc]):
        if isinstance(s, str):
            a = np.asarray(json.loads(s), dtype=float)
            out.append(a[int(w):])
        else:
            out.append(None)
    return out


def position_profile(df, op, prefix="time_stats"):
    """median over rows of sample_k / row-median, for k = 1..N (timed positions)."""
    rows = [s / np.median(s) for s in samples(df, op, prefix) if s is not None and len(s) > 0]
    n = min(len(r) for r in rows)
    m = np.vstack([r[:n] for r in rows])
    return np.arange(0, n), np.median(m, axis=0)


def tokfmt(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else f"{v:g}"))


def finish(fig, name, out, note=None):
    if note:
        fig.text(0.01, -0.035, note, fontsize=7.5, color=MUTED, ha="left", va="top")
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, REPO))


def band(ax, lo, hi, label=None):
    ax.axvspan(lo, hi, color=CRITICAL, alpha=0.08, lw=0)
    if label:
        ax.text(lo, ax.get_ylim()[1], label, fontsize=8, color=CRITICAL, va="top", ha="left")


def tp_legend(ax, tps=TPS, **kw):
    for tp in tps:
        ax.plot([], [], color=TP_COLOR[tp], label=f"TP {tp}")
    ax.legend(**kw)


# =================================================================================================
def fig00_baseline(out, old_legacy, grid386):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharey=True)
    for ax, tp in zip(axes.flat, TPS):
        a = old_legacy[old_legacy.num_tensor_parallel_workers == tp].sort_values("num_tokens")
        b = grid386[grid386.num_tensor_parallel_workers == tp].sort_values("num_tokens")
        ax.plot(a.num_tokens, a["time_stats.attn_pre_proj.median"], color=C_OLD, label="inherited dataset (20 timed runs, ≤4,096 tokens)")
        ax.plot(b.num_tokens, b["time_stats.attn_pre_proj.median"], color=C_NEW, label="2026-09-15 re-collection (50 timed runs, ≤16,384 tokens)")
        ax.set_title(f"attn_pre_proj median, TP {tp}")
        tokfmt(ax); ax.set_yscale("log"); ax.set_xlabel("num_tokens")
    axes[0, 0].set_ylabel("ms (log)"); axes[1, 0].set_ylabel("ms (log)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Chapter 0 — what we started from: the inherited linear_op data vs our first 50-repetition run (386-token grid)", x=0.01, ha="left")
    finish(fig, "f00_baseline_inherited_vs_50rep.png", out,
           "inherited: runs/legacy_pre-2026-09_torch-sdpa/linear_op_maxtokens4096.csv · new: runs/2026-09-15_1142_linear_op_grid386 (job 21309)")


def fig01_spike_envelope(out, old):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharey=True)
    for ax, tp in zip(axes.flat, TPS):
        d = old[old.num_tensor_parallel_workers == tp].sort_values("num_tokens")
        ax.fill_between(d.num_tokens, d["time_stats.attn_pre_proj.min"], d["time_stats.attn_pre_proj.max"],
                        color=BLUE, alpha=0.18, lw=0, label="min–max over 50 timed runs")
        ax.plot(d.num_tokens, d["time_stats.attn_pre_proj.mean"], color=BLUE, lw=1.2, label="mean")
        ax.plot(d.num_tokens, d["time_stats.attn_pre_proj.median"], color=INK, lw=1.2, ls="--", label="median")
        sp = d[d["time_stats.attn_pre_proj.max"] > 50]
        ax.scatter(sp.num_tokens, sp["time_stats.attn_pre_proj.max"], color=CRITICAL, s=18, zorder=5,
                   label="run-11 spike (tokens 1968–1975)")
        ax.set_title(f"attn_pre_proj, TP {tp}")
        tokfmt(ax); ax.set_yscale("log"); ax.set_xlabel("num_tokens")
    axes[0, 0].set_ylabel("ms (log)"); axes[1, 0].set_ylabel("ms (log)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Chapter 1 — first dense grid (job 21313): one timed run of attn_pre_proj at 150–290 ms in exactly 32 rows; medians untouched", x=0.01, ha="left")
    finish(fig, "f01_spike_envelope_dense_original.png", out, "data: runs/2026-09-15_1308_linear_op_dense3327/linear_op.csv (canonical linear_op.csv)")


def fig02_spike_samples(out, old):
    r = old[(old.num_tokens == 1970) & (old.num_tensor_parallel_workers == 1)].iloc[0]
    s = np.asarray(json.loads(r["time_stats.attn_pre_proj.samples"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(s))
    ax.axvspan(-0.5, 2.5, color=MUTED, alpha=0.12, lw=0)
    ax.text(0, s.max() * 0.6, "warm-up\n(3 runs)", fontsize=8, color=INK2, ha="left")
    ax.vlines(x, 0.05, s, color=BLUE, lw=2)
    ax.scatter(x, s, color=BLUE, s=14, zorder=4)
    ax.vlines([14], 0.05, s[14], color=CRITICAL, lw=2.5); ax.scatter([14], [s[14]], color=CRITICAL, s=28, zorder=5)
    ax.annotate(f"timed run 11 (absolute sample 14): {s[14]:.0f} ms\n≈1,100× the row median of {np.median(s[3:]):.3f} ms",
                (14, s[14]), xytext=(20, s[14] * 0.5), fontsize=9, color=CRITICAL,
                arrowprops=dict(arrowstyle="-", color=CRITICAL, lw=0.8))
    ax.set_yscale("log"); ax.set_ylim(0.05, 400)
    ax.set_xlabel("sample index in execution order (0–2 warm-up, 3–52 timed)"); ax.set_ylabel("attn_pre_proj time, ms (log)")
    ax.set_title("Chapter 1 — the 53 recorded samples of one affected row (tokens 1970, TP 1, job 21313)")
    finish(fig, "f02_spike_one_row_samples.png", out)


def fig03_spike_band_two_runs(out, old, rerun):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (df, title) in zip(axes, [(old, "first dense run — job 21313, node 9, 2026-09-15"),
                                      (rerun, "re-run, same command — job 21334, node 8, 2026-09-16")]):
        for tp in TPS:
            d = df[(df.num_tensor_parallel_workers == tp) & df.num_tokens.between(1940, 2010)].sort_values("num_tokens")
            s = samples(d, "attn_pre_proj")
            y = [a[11] / np.median(a) for a in s]
            ax.plot(d.num_tokens, y, color=TP_COLOR[tp], marker="o", ms=3.5, lw=1.2, label=f"TP {tp}")
        ax.axvspan(1967.5, 1975.5, color=CRITICAL, alpha=0.10, lw=0)
        ax.set_yscale("log"); ax.set_xlabel("num_tokens"); ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("timed run 11 ÷ row median (log)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Chapter 2 — the spike reproduces exactly: same 8 token counts, same run index, every TP, another node 17 h later", x=0.01, ha="left")
    finish(fig, "f03_spike_band_first_vs_rerun.png", out,
           "tokens 1968–1975 are the eight tasks at per-worker position 169 of the descending token list (one per GPU worker)")


def fig04_fingerprint(out, old, label, fname, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    for tp in TPS:
        d = old[old.num_tensor_parallel_workers == tp]
        k, p = position_profile(d, "attn_pre_proj")
        ax.plot(k, p, color=TP_COLOR[tp], marker="o", ms=3, label=f"TP {tp}")
    ax.axhline(1, color=AXIS, lw=0.8)
    for kk, txt in [(11, "run 11: GC collections land here"), (33, "run 33: gen-0 GC"), (49, "run 49: not GC")]:
        ax.axvline(kk, color=CRITICAL if kk in (11, 33) else MUTED, lw=0.8, ls=":")
        ax.text(kk + 0.3, 2.1, txt, fontsize=8, color=INK2, rotation=90, va="top")
    ax.set_xlabel("timed run index k (0-based, 0–49; as in the documents)"); ax.set_ylabel("median over rows of sample_k ÷ row median")
    ax.set_ylim(0.9, 2.2); ax.legend(loc="upper right", ncol=4)
    ax.set_title(title)
    finish(fig, fname, out, label)


def fig05_gc_gen2_vs_spike(out, e1b_dir):
    files = sorted(glob.glob(os.path.join(e1b_dir, "worker_gpu*_pid*.jsonl")))
    if not files:
        print("skip f05: no spike_diag e1b jsonl"); return
    pts = []
    for f in files:
        for line in open(f):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ev = [e for e in rec.get("gc_events", []) if e.get("gen") == 2]
            if not ev:
                continue
            fw = rec.get("forwards_ms")
            smp = (rec.get("samples_ms") or {}).get("attn_pre_proj")
            for e in ev:
                # which forward window contains the collection start?
                idx = next((i for i, (a, b) in enumerate(fw) if a <= e["start_ms"] <= b), None) if fw else None
                pts.append(dict(dur=e["dur_ms"], fwd=idx, tp=rec.get("tp"), tokens=rec.get("num_tokens"),
                                spike=(smp[idx] if (idx is not None and smp and idx < len(smp)) else np.nan)))
    p = pd.DataFrame(pts)
    if p.empty:
        print("skip f05: no gen-2 events parsed"); return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    inwin = p.dropna(subset=["spike"])
    ax.plot([100, 300], [100, 300], color=AXIS, lw=0.8)
    ax.scatter(inwin.dur, inwin.spike, color=CRITICAL, s=22, zorder=4)
    ax.set_xlabel("gen-2 garbage collection duration, ms (gc.callbacks)"); ax.set_ylabel("attn_pre_proj sample in the same forward, ms")
    ax.set_title(f"{len(inwin)} gen-2 collections inside a timed forward: duration = event gap (identity line)", fontsize=10)
    ax = axes[1]
    vc = p.fwd.value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color=BLUE, width=0.6)
    ax.set_xlabel("forward index the collection started in (0–2 warm-up; 14 = timed run 11)"); ax.set_ylabel("gen-2 collections")
    ax.set_title("where in the 53-forward task the full collections fired", fontsize=10)
    fig.suptitle("Chapter 3 — root cause: a CPython generation-2 (full) garbage collection on the worker's main thread (job 21346 diagnostics)", x=0.01, ha="left")
    finish(fig, "f05_gc_gen2_duration_vs_spike.png", out, "data: data/profiling/sweep_work/logs/spike_diag/e1b/*.jsonl on the cluster (E1b, diag on)")


def fig06_gc_counter_model(out):
    fwd = np.arange(0, 53)
    count = 255 + 32 * fwd
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(fwd, count, color=BLUE, label="gen-0 net allocation count within one task (≈255 after model build, +≈32 per forward)")
    ax.axhline(700, color=CRITICAL, lw=1, ls="--"); ax.text(0.3, 712, "gen-0 threshold 700", color=CRITICAL, fontsize=8)
    for f in (14, 36):
        ax.axvline(f, color=MUTED, lw=0.8, ls=":")
        ax.text(f + 0.3, 300, f"forward {f}\n= timed run {f-3}", fontsize=8, color=INK2)
    ax.axvspan(-0.5, 2.5, color=MUTED, alpha=0.12, lw=0)
    ax.set_xlabel("forward index inside one profiler task"); ax.set_ylabel("tracked-object count")
    ax.set_title("Chapter 3 — why always run 11: the gen-0 counter crosses 700 at forward 14 (and 36); a collection of any generation can only fire there")
    ax.legend(loc="lower right")
    finish(fig, "f06_gc_counter_model.png", out, "illustration of the mechanism in 05 §3.3; slope/offset measured in E1/E1b, not fitted here")


def fig07_gcfix_before_after(out, rerun, gcfix):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (df, ttl, c) in zip(axes, [(rerun, "before the fix — job 21334", C_OLD), (gcfix, "gc.disable() around the timed loop — job 21361", C_NEW)]):
        for tp in TPS:
            d = df[df.num_tensor_parallel_workers == tp]
            ax.scatter(d.num_tokens, d["time_stats.attn_pre_proj.max"] / d["time_stats.attn_pre_proj.median"],
                       s=5, color=c, alpha=0.5, lw=0)
        ax.axhline(20, color=MUTED, lw=0.8, ls=":"); ax.text(1.1, 22, "20× row median", fontsize=8, color=INK2)
        ax.set_yscale("log"); tokfmt(ax); ax.set_xlabel("num_tokens"); ax.set_title(ttl, fontsize=10)
    axes[0].set_ylabel("max timed sample ÷ row median (log)")
    fig.suptitle("Chapter 4 — the fix removes the 150–290 ms class; only the sporadic ≤2 ms host-jitter class remains", x=0.01, ha="left")
    finish(fig, "f07_gcfix_max_over_median_before_after.png", out, "attn_pre_proj, all four TPs pooled per panel · after: data/profiling_gcfix/…/linear_op.csv")


def fig08_fingerprint_before_after(out, rerun, gcfix):
    fig, ax = plt.subplots(figsize=(10, 4))
    for df, ttl, c in [(rerun, "before (job 21334)", C_OLD), (gcfix, "after gc.disable() (job 21361)", C_NEW)]:
        k, p = position_profile(df, "attn_pre_proj")
        ax.plot(k, p, color=c, marker="o", ms=3, label=ttl)
    ax.axhline(1, color=AXIS, lw=0.8)
    ax.axvline(11, color=CRITICAL, lw=0.8, ls=":"); ax.axvline(33, color=CRITICAL, lw=0.8, ls=":")
    ax.text(11.3, 1.3, "run 11: 1.31 → 1.00", fontsize=8, color=INK2); ax.text(33.3, 1.3, "run 33", fontsize=8, color=INK2)
    ax.set_ylim(0.9, 1.5); ax.set_xlabel("timed run index k (0-based)"); ax.set_ylabel("median of sample_k ÷ row median")
    ax.set_title("Chapter 4 — the GC fingerprint on runs 11/33 is gone; the bumps at runs 0, 16, 49 are not GC and stay (attn_pre_proj, all rows)")
    ax.legend(loc="upper right")
    finish(fig, "f08_fingerprint_before_after_gcfix.png", out)


def fig09_dip(out, old):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, op in zip(axes, OPS3):
        for tp in TPS:
            d = old[(old.num_tensor_parallel_workers == tp) & old.num_tokens.between(1000, 16384)].sort_values("num_tokens")
            ax.plot(d.num_tokens, d[f"time_stats.{op}.median"], color=TP_COLOR[tp], lw=1.2, label=f"TP {tp}")
        ax.axvspan(4000, 6000, color=CRITICAL, alpha=0.08, lw=0)
        tokfmt(ax); ax.set_yscale("log"); ax.set_xlabel("num_tokens"); ax.set_title(f"{op} median")
    axes[0].set_ylabel("ms (log)"); axes[0].legend(loc="upper left")
    axes[1].text(4050, axes[1].get_ylim()[0] * 1.05, "the “dip”", fontsize=8, color=CRITICAL)
    fig.suptitle("Chapter 5 — the V-shaped dip: attn_post_proj and attn_rope medians *fall* 17–37 % around 4,000–6,000 tokens at TP 2/4/8; flat plateaus below", x=0.01, ha="left")
    finish(fig, "f09_dip_medians_original.png", out, "data: canonical linear_op.csv (job 21313); reproduced within ~100 tokens by job 21334")


def fig10_dip_samples(out, old):
    fig, ax = plt.subplots(figsize=(10, 4))
    for tok, c, lab in [(4128, CRITICAL, "tokens 4128 (in the dip): bimodal, drifting within the task"),
                        (6432, BLUE, "tokens 6432 (past the dip): tight")]:
        r = old[(old.num_tokens == tok) & (old.num_tensor_parallel_workers == 2)].iloc[0]
        s = np.asarray(json.loads(r["time_stats.attn_post_proj.samples"]))[3:]
        ax.plot(np.arange(1, 51), s * 1000, color=c, marker="o", ms=3.5, lw=1, label=lab)
    ax.axhline(35.8, color=AQUA, lw=1, ls="--"); ax.text(1, 36.3, "traced o_proj kernel at M=4128: 35.8 µs (rocprofv3)", fontsize=8, color=AQUA)
    ax.set_xlabel("timed run index"); ax.set_ylabel("attn_post_proj sample, µs"); ax.legend(loc="upper right")
    ax.set_title("Chapter 5 — the fingerprint the brief missed: dip-row samples are bimodal (lower mode = kernel, upper = host span), TP 2, job 21313")
    finish(fig, "f10_dip_row_samples_bimodal.png", out)


def fig11_dip_backlog(out, old, x1):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for row, op in enumerate(["attn_post_proj", "attn_rope"]):
        for col, tp in enumerate([2, 4, 8]):
            ax = axes[row, col]
            a = old[(old.num_tensor_parallel_workers == tp) & old.num_tokens.between(1000, 16384)].sort_values("num_tokens")
            b = x1[(x1.num_tensor_parallel_workers == tp)].sort_values("num_tokens")
            ax.plot(a.num_tokens, a[f"time_stats.{op}.median"] * 1000, color=C_OLD, lw=1.1, label="original sweep (job 21313)")
            ax.plot(b.num_tokens, b[f"time_stats.{op}.median"] * 1000, color=C_NEW, lw=1.1, label="GPU held 100 ms behind the host (X1, job 21356)")
            ax.axvspan(4000, 6000, color=CRITICAL, alpha=0.08, lw=0)
            tokfmt(ax); ax.set_title(f"{op}, TP {tp}", fontsize=10)
            if col == 0: ax.set_ylabel("median, µs")
            if row == 1: ax.set_xlabel("num_tokens")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Chapter 6 — one knob, no dip: with the device kept busy before the timed loop every curve becomes monotonic and matches the traced kernel + ≈6 µs", x=0.01, ha="left")
    finish(fig, "f11_dip_original_vs_gpu_backlog.png", out, "X1 grid: 1024…10240 step 8, 10368…16384 step 128 · data: data/profiling_dip_x1/…/linear_op.csv (cluster)")


def fig12_inflation_heatmap(out, old, x1):
    wins = list(range(1024, 10241, 1024))
    rows, labels = [], []
    for op in OPS3:
        for tp in TPS:
            a = old[old.num_tensor_parallel_workers == tp]; b = x1[x1.num_tensor_parallel_workers == tp]
            m = a.merge(b, on="num_tokens", suffixes=("_o", "_x"))
            vals = []
            for w in wins:
                mm = m[m.num_tokens.between(w - 256, w + 256)]
                vals.append((mm[f"time_stats.{op}.median_o"] / mm[f"time_stats.{op}.median_x"]).median() if len(mm) else np.nan)
            rows.append(vals); labels.append(f"{op}  TP {tp}")
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(M, cmap=cmap, vmin=1.0, vmax=3.0, aspect="auto")
    ax.set_xticks(range(len(wins))); ax.set_xticklabels([f"{w:,}" for w in wins], fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK if M[i, j] < 2.0 else "#ffffff")
    ax.set_xlabel("num_tokens window (±256)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("original median ÷ GPU-bound (X1) median")
    ax.set_title("Chapter 6 — the artefact is not confined to the dip band: the original medians are 1.2–3× the GPU time over most of the TP>1 grid")
    finish(fig, "f12_inflation_original_over_gpu_bound.png", out, "reproduces 07 §3.6 from the two CSVs (windows of ±256 tokens)")


def fig13_oproj_tiles(out):
    M = [3000, 3504, 3928, 4032, 4080, 4128, 4176, 4320, 4328, 4360, 4400, 4840, 4872, 5000, 5504, 5912, 6432, 7008, 8000]
    tp2 = [28.6, 32.0, 33.8, 34.6, 34.8, 35.8, 35.5, 35.9, 44.9, 44.8, 45.1, 47.2, 38.6, 41.0, 42.8, 47.0, 48.9, 51.3, 56.0]
    tp4 = [18.7, 21.3, 22.0, 21.3, 21.6, 23.3, 23.2, 23.3, 23.6, 31.7, 31.8, 32.6, 24.3, 24.4, 26.7, 27.2, 28.2, 28.6, 33.6]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(M, tp2, color=TP_COLOR[2], marker="o", ms=4, label="TP 2 (K = 2048)")
    ax.plot(M, tp4, color=TP_COLOR[4], marker="o", ms=4, label="TP 4 (K = 1024)")
    ax.axvspan(4328, 4840, color=CRITICAL, alpha=0.08, lw=0)
    ax.text(4340, 50, "hipBLASLt picks tile MT160x256x64 here:\n+25–34 % slower — a real effect, opposite in sign to the dip", fontsize=8, color=INK2)
    ax.set_xlabel("M = num_tokens"); ax.set_ylabel("traced o_proj GEMM kernel duration, µs")
    ax.set_title("Chapter 6 — the GPU truth behind attn_post_proj (rocprofv3 kernel trace, X2d): monotonic through the dip band, with one slow tile band")
    ax.legend(loc="upper left")
    finish(fig, "f13_oproj_kernel_trace_tiles.png", out, "values from 07 §3.3 (job 21357/X2d traces, data/profiling_dip_x2/rocprof on the cluster)")


def fig14_validation_two_column(out, probe):
    rocprof_tp2 = {1: 4.0, 8: 9.2, 64: 10.2, 3072: 28.0, 4096: 34.4, 4192: 35.6, 6144: 45.5, 8192: 55.9}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, tp in zip(axes, TPS):
        d = probe[probe.num_tensor_parallel_workers == tp].sort_values("num_tokens")
        ax.plot(d.num_tokens, d["time_stats_hostbound.attn_post_proj.median"] * 1000, color=C_OLD, marker="o", ms=4, label="legacy loop (host-bound where the device idles)")
        ax.plot(d.num_tokens, d["time_stats.attn_post_proj.median"] * 1000, color=C_NEW, marker="o", ms=4, label="GPU-bound pass (device held behind the host)")
        if tp == 2:
            ax.scatter(list(rocprof_tp2), list(rocprof_tp2.values()), color=C_TRACE, marker="D", s=26, zorder=5, label="rocprofv3 kernel (job 21376)")
        tokfmt(ax); ax.set_yscale("log"); ax.set_xlabel("num_tokens"); ax.set_title(f"attn_post_proj, TP {tp}", fontsize=10)
    axes[0].set_ylabel("median, µs (log)")
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Chapter 7 — the fixed profiler times every shape twice; the legacy column sits on a ≈35 µs host floor while the kernel is 4–12 µs at 1–64 tokens", x=0.01, ha="left")
    finish(fig, "f14_validation_grid_two_columns.png", out, "data: runs/2026-09-17_1016_linear_op_validation_grid_two_column_probe (job 21400)")


def fig15_clock_probe(out, probe):
    cols = [("sclk_mhz_legacy_start", "legacy pass: start of timed loop\n(after 3 warm-ups + synchronize)"),
            ("sclk_mhz_legacy_end", "legacy pass: end"),
            ("sclk_mhz_backlog_start", "GPU-bound pass: start"),
            ("sclk_mhz_backlog_end", "GPU-bound pass: end")]
    fig, ax = plt.subplots(figsize=(10, 4))
    rng = np.random.default_rng(0)
    for i, (c, lab) in enumerate(cols):
        y = probe[c].values
        x = i + rng.uniform(-0.18, 0.18, len(y))
        ax.scatter(x, y, s=18, color=BLUE if "backlog" in c else ORANGE, alpha=0.8, lw=0)
        ax.hlines(np.median(y), i - 0.3, i + 0.3, color=INK, lw=1.2)
        ax.text(i + 0.32, np.median(y), f"median {np.median(y):.0f}", fontsize=8, color=INK2, va="center")
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in cols], fontsize=8.5)
    ax.set_ylabel("shader clock estimate, MHz"); ax.set_ylim(500, 2600)
    ax.set_title("Chapter 7 — warm-up does not warm the clock: the legacy timed loop starts at ≈0.8 GHz on 32 of 36 rows and ramps to ≈2.4 GHz; the GPU-bound pass runs at boost throughout")
    finish(fig, "f15_clock_probe_per_pass.png", out, "probe = fixed 2M-cycle torch.cuda._sleep timed by an event pair; validated at a 1900 MHz lock (reads 1890–1916) · job 21400, 36 rows")


def fig16_rope(out, probe, ropefix, ropeko):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tp in zip(axes, [1, 8]):
        a = probe[probe.num_tensor_parallel_workers == tp].sort_values("num_tokens")
        b = ropefix[ropefix.num_tensor_parallel_workers == tp].sort_values("num_tokens")
        c = ropeko[ropeko.num_tensor_parallel_workers == tp].sort_values("num_tokens")
        ax.plot(a.num_tokens, a["time_stats.attn_rope.median"] * 1000, color=C_OLD, marker="o", ms=4, label="forced torch fallback (13 kernels; rotates only head 0's first 64 columns)")
        ax.plot(b.num_tokens, b["time_stats.attn_rope.median"] * 1000, color=C_NEW, marker="o", ms=4, label="fused vllm rotary_embedding kernel, GPU-bound event")
        ax.plot(c.num_tokens, c["time_stats.attn_rope.median"] * 1000, color=C_TRACE, marker="D", ms=4, lw=1, label="same, kineto kernel-only (record_function)")
        tokfmt(ax); ax.set_yscale("log"); ax.set_xlabel("num_tokens"); ax.set_title(f"attn_rope, TP {tp}", fontsize=10)
    axes[0].set_ylabel("median, µs (log)"); axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Chapter 8 — RoPE was never RoPE: replacing the forced fallback by the fused kernel makes attn_rope 2–8× cheaper and numerically correct", x=0.01, ha="left")
    finish(fig, "f16_rope_fallback_vs_fused.png", out, "runs/2026-09-17_1016_…_probe (fallback, job 21400) vs runs/2026-09-17_1250_linear_op_rope_fix_validation_grid (job 21410)")


def fig17_dense_fixed_two_columns(out, nf):
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
    for r, op in enumerate(OPS3):
        for c, tp in enumerate(TPS):
            ax = axes[r, c]
            d = nf[nf.num_tensor_parallel_workers == tp].sort_values("num_tokens")
            ax.plot(d.num_tokens, d[f"time_stats_hostbound.{op}.median"] * 1000, color=C_OLD, lw=0.9, label="legacy loop (same method as the old dataset)")
            ax.plot(d.num_tokens, d[f"time_stats.{op}.median"] * 1000, color=C_NEW, lw=0.9, label="GPU-bound pass (primary column)")
            tokfmt(ax); ax.set_yscale("log"); ax.set_title(f"{op}, TP {tp}", fontsize=9.5)
            if c == 0: ax.set_ylabel("median, µs (log)")
            if r == 2: ax.set_xlabel("num_tokens")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Chapter 9 — the re-collected dense grid (job 21483, 2026-09-22): both columns on every row; the gap between them is the host span the old data reported as GPU time", x=0.01, ha="left")
    finish(fig, "f17_dense_fixed_gpu_bound_vs_legacy.png", out, "data: data/profiling_dense_fixed/…/linear_op.csv (cluster scratch; 25 timed forwards per shape, fused RoPE, GC disabled, clock probe)")


def fig18_dip_gone(out, old, nf):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for r, op in enumerate(["attn_post_proj", "attn_rope"]):
        for c, tp in enumerate([2, 4, 8]):
            ax = axes[r, c]
            a = old[(old.num_tensor_parallel_workers == tp) & old.num_tokens.between(1000, 16384)].sort_values("num_tokens")
            b = nf[(nf.num_tensor_parallel_workers == tp) & nf.num_tokens.between(1000, 16384)].sort_values("num_tokens")
            ax.plot(a.num_tokens, a[f"time_stats.{op}.median"] * 1000, color=C_OLD, lw=1, label="old canonical (job 21313)")
            ax.plot(b.num_tokens, b[f"time_stats.{op}.median"] * 1000, color=C_NEW, lw=1, label="fixed collection, GPU-bound (job 21483)")
            ax.axvspan(4000, 6000, color=CRITICAL, alpha=0.08, lw=0)
            tokfmt(ax); ax.set_title(f"{op}, TP {tp}", fontsize=10)
            if c == 0: ax.set_ylabel("median, µs")
            if r == 1: ax.set_xlabel("num_tokens")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Chapter 9 — the dip is gone and so are the plateaus; what remains are 15–30 % steps at the hipBLASLt tile switches (real kernel behaviour)", x=0.01, ha="left")
    finish(fig, "f18_dip_gone_old_vs_fixed.png", out, "attn_rope also changed implementation (fallback → fused kernel), so its old/new gap is measurement + kernel")


def fig19_inflation_ratio(out, old, nf):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, op in zip(axes, OPS3):
        for tp in TPS:
            a = old[old.num_tensor_parallel_workers == tp]; b = nf[nf.num_tensor_parallel_workers == tp]
            m = a.merge(b, on="num_tokens", suffixes=("_o", "_n")).sort_values("num_tokens")
            ratio = (m[f"time_stats.{op}.median_o"] / m[f"time_stats.{op}.median_n"]).rolling(15, center=True, min_periods=1).median()
            ax.plot(m.num_tokens, ratio, color=TP_COLOR[tp], lw=1.1, label=f"TP {tp}")
        ax.axhline(1, color=AXIS, lw=0.8); tokfmt(ax); ax.set_yscale("log"); ax.set_xlabel("num_tokens"); ax.set_title(op)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    axes[0].set_ylabel("old ÷ fixed GPU-bound median (log)"); axes[0].legend(loc="upper right")
    fig.suptitle("Chapter 9 — how inflated the old data was: 1.3–2.9× for the GEMMs at small/mid token counts, 4–12× for attn_rope (artefact + wrong kernel)", x=0.01, ha="left")
    finish(fig, "f19_old_over_fixed_ratio.png", out)


def fig20_tile_steps(out, nf):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for tp in [2, 4, 8]:
        d = nf[(nf.num_tensor_parallel_workers == tp) & nf.num_tokens.between(3800, 5800)].sort_values("num_tokens")
        ax.plot(d.num_tokens, d["time_stats.attn_post_proj.median"] * 1000, color=TP_COLOR[tp], marker="o", ms=2.5, lw=1, label=f"TP {tp}")
    for x in (4352, 4480, 4864, 5376):
        ax.axvline(x + 4, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("num_tokens"); ax.set_ylabel("attn_post_proj GPU-bound median, µs"); ax.legend(loc="upper left")
    ax.set_title("Chapter 9 — zoom on 3,800–5,800 tokens in the fixed data: sharp steps at 4352→4360, 4480→4488, 4864→4872, 5376→5384 = tile-selection boundaries")
    finish(fig, "f20_tile_steps_zoom_fixed.png", out, "a cost model has to represent these, not smooth them (12 §5)")


def fig21_collection_diag(out, nf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for tp in TPS:
        d = nf[nf.num_tensor_parallel_workers == tp]
        ax.hist(d["sclk_mhz_backlog_start"], bins=np.arange(2000, 2460, 10), color=TP_COLOR[tp], alpha=0.75, label=f"TP {tp}", histtype="stepfilled", lw=0)
    ax.set_xlabel("shader clock at the start of the GPU-bound timed loop, MHz"); ax.set_ylabel("rows"); ax.legend()
    ax.set_title("device at ≈2.4 GHz on all but a few of 13,308 rows (back-to-back tasks keep it warm)", fontsize=9.5)
    ax = axes[1]
    r = nf["gpu_backlog_ms_actual"] / nf["gpu_backlog_ms"]
    ax.hist(r, bins=np.linspace(0.9, 1.1, 81), color=BLUE, lw=0)
    ax.set_xlabel("delivered spin ÷ requested spin"); ax.set_ylabel("rows")
    ax.set_title("the backlog spin was delivered as requested (0.97–1.03)", fontsize=9.5)
    fig.suptitle("Chapter 9 — collection diagnostics recorded on every row of the fixed dense grid (job 21483)", x=0.01, ha="left")
    finish(fig, "f21_collection_diagnostics_fixed.png", out)


def fig22_stall_gone(out, rerun, dense200):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for df, lab, c in [(rerun, "old (job 21334, 50 timed runs)", C_OLD), (dense200, "fixed (job 21486, 200 timed runs)", C_NEW)]:
        for tp in TPS:
            d = df[df.num_tensor_parallel_workers == tp]
            ax.scatter(d.num_tokens, d["time_stats.attn_pre_proj.max"] / d["time_stats.attn_pre_proj.median"], s=5, color=c, alpha=0.5, lw=0, label=lab if tp == 1 else None)
    ax.axhline(10, color=MUTED, lw=0.8, ls=":"); ax.text(1.1, 11, "10× row median", fontsize=8, color=INK2)
    ax.set_yscale("log"); tokfmt(ax); ax.set_xlabel("num_tokens"); ax.set_ylabel("max timed sample ÷ row median (log)"); ax.legend(loc="upper right")
    ax.set_title("Chapter 9 — with 200 samples per row, no attn_pre_proj row has a sample above 10× its median (old file: 47 rows, worst 268 ms)")
    finish(fig, "f22_stall_gone_200_reps.png", out, "data: data/profiling_dense200/…/linear_op.csv (cluster scratch)")


def fig23_position_profile_25(out, nf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    ax = axes[0]
    for tp in TPS:
        k, p = position_profile(nf[nf.num_tensor_parallel_workers == tp], "attn_pre_proj")
        ax.plot(k + 1, p, color=TP_COLOR[tp], marker="o", ms=3, label=f"TP {tp}")
    ax.axhline(1, color=AXIS, lw=0.8); ax.axvline(20, color=CRITICAL, lw=0.8, ls=":")
    ax.text(20.3, 1.10, "run 20: ROCclr batch-flush barrier\n(1000th command since synchronize)", fontsize=8, color=INK2)
    ax.set_xlabel("timed run index (1–25)"); ax.set_ylabel("median of sample_k ÷ row median"); ax.legend(loc="upper right", ncol=2)
    ax.set_title("GPU-bound column: 2–3 % downward ramp within the block + a spike at run 20 (TP>1)", fontsize=9.5)
    ax = axes[1]
    for tp in TPS:
        k, p = position_profile(nf[nf.num_tensor_parallel_workers == tp], "attn_pre_proj", "time_stats_hostbound")
        ax.plot(k + 1, p, color=TP_COLOR[tp], marker="o", ms=3, label=f"TP {tp}")
    ax.axhline(1, color=AXIS, lw=0.8)
    ax.set_xlabel("timed run index (1–25)"); ax.set_title("legacy column, same rows: the same barrier costs the host-bound loop +50–100 µs at runs 21/22", fontsize=9.5)
    fig.suptitle("Chapter 10 — run-position anomalies in the fixed collection (job 21483, attn_pre_proj)", x=0.01, ha="left")
    finish(fig, "f23_position_profile_fixed_25.png", out)


def fig24_position_profile_200(out, d200):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for ax, op in zip(axes, ["attn_pre_proj", "forward_gpu_span"]):
        for tp in [1, 8]:
            k, p = position_profile(d200[d200.num_tensor_parallel_workers == tp], op)
            ax.plot(k + 1, p, color=TP_COLOR[tp], lw=1, marker="o", ms=2, label=f"TP {tp}")
        for b in range(25, 200, 25):
            ax.axvline(b + 0.5, color=MUTED, lw=0.6, ls=":")
        for h in (20, 40, 59, 79, 98, 118, 138, 157, 177, 196):
            ax.axvline(h, color=CRITICAL, lw=0.6, alpha=0.6)
        ax.axhline(1, color=AXIS, lw=0.8); ax.set_ylabel("sample_k ÷ row median"); ax.set_title(op, fontsize=10)
    axes[0].legend(loc="upper right"); axes[1].set_xlabel("timed run index (1–200; dotted = re-armed spin every 25; red = predicted barrier positions at TP 8)")
    fig.suptitle("Chapter 10 — 200 repetitions as 8 spun blocks of 25 (job 21486): the ramp restarts after every spin; the barrier hits exactly where 'one per 1000 packets' predicts", x=0.01, ha="left")
    finish(fig, "f24_position_profile_200_reps.png", out, "data: data/profiling_dense200/…/linear_op.csv (cluster scratch)")


def fig25_batchflush(out, nf, bf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, prefix, ttl in [(axes[0], "time_stats", "GPU-bound column, attn_pre_proj, TP 8"),
                            (axes[1], "time_stats_hostbound", "legacy column, attn_pre_proj, TP 8")]:
        for df, lab, c in [(nf, "default runtime (job 21483)", C_OLD), (bf, "DEBUG_CLR_MAX_BATCH_SIZE=1000000 (job 21500)", C_NEW)]:
            k, p = position_profile(df[df.num_tensor_parallel_workers == 8], "attn_pre_proj", prefix)
            ax.plot(k + 1, p, color=c, marker="o", ms=3, label=lab)
        ax.axhline(1, color=AXIS, lw=0.8); ax.set_xlabel("timed run index (1–25)"); ax.set_title(ttl, fontsize=10)
    axes[0].set_ylabel("median of sample_k ÷ row median"); axes[0].legend(loc="upper left")
    fig.suptitle("Chapter 10 — fix for the barrier verified: the run-20 spike and the runs-21/22 host stall vanish; the within-block clock ramp stays (needs its own fix)", x=0.01, ha="left")
    finish(fig, "f25_batchflush_before_after.png", out, "data: data/profiling_dense_fixed_batchflush/…/linear_op.csv (cluster scratch); medians unchanged (ratio 1.000, p1–p99 0.96–1.05)")


def _posprobe_tasks(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def fig26_clock_ramp(out, pp_dir):
    """posprobe R15 (device-side clock probe after every forward) and R17/R19 (GEMM backlog vs _sleep spin), TP 8."""
    r15 = os.path.join(pp_dir, "R15_clock_s25_tp8.jsonl")
    r17 = os.path.join(pp_dir, "R17_busy_s25_tp8.jsonl"); r19 = os.path.join(pp_dir, "R19_default_s25_tp8.jsonl")
    if not (os.path.exists(r15) and os.path.exists(r17) and os.path.exists(r19)):
        print("skip f26: posprobe files missing"); return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for rec in _posprobe_tasks(r15):
        tok = rec.get("tokens")
        clk = rec.get("sclk_mhz_by_forward")
        if not clk or tok not in (6144, 2048, 64):
            continue
        y = np.asarray([c[1] for c in clk], dtype=float)[-25:]      # the 25 timed forwards of the GPU-bound pass
        ax.plot(np.arange(1, len(y) + 1), y, marker="o", ms=3, lw=1, color={6144: BLUE, 2048: ORANGE, 64: AQUA}[tok], label=f"{tok:,} tokens")
    ax.set_xlabel("timed forward after the spin (GPU-bound pass)"); ax.set_ylabel("device-side shader-clock probe, MHz"); ax.legend()
    ax.set_title("R15: the shader clock rises ≈1.5 % over the first 10–15 forwards after a _sleep spin", fontsize=9.5)
    ax = axes[1]
    for path, lab, c in [(r19, "default: _sleep spin as backlog (R19)", C_OLD), (r17, "backlog = chain of real GEMMs (R17)", C_NEW)]:
        profs = []
        for rec in _posprobe_tasks(path):
            d = rec.get("gpu", {}).get("attn_pre_proj")
            if d:
                s = np.asarray(d["samples"], dtype=float)[int(d["warmup_count"]):]
                profs.append(s / np.median(s))
        if profs:
            m = np.median(np.vstack(profs), axis=0)
            ax.plot(np.arange(1, len(m) + 1), m, marker="o", ms=3, color=c, label=lab)
    ax.axhline(1, color=AXIS, lw=0.8); ax.set_xlabel("timed run index (1–25)"); ax.set_ylabel("median over 7 tasks of sample_k ÷ task median"); ax.legend()
    ax.set_title("R17 vs R19 (attn_pre_proj, TP 8): a work backlog instead of the spin removes the drift", fontsize=9.5)
    fig.suptitle("Chapter 10 — Effect 1 root cause: during the low-activity _sleep spin the clock domains idle; the ramp is the device warming up again", x=0.01, ha="left")
    finish(fig, "f26_clock_ramp_after_spin.png", out, "data: data/profiling_posprobe/R15,R17,R19 (cluster scratch; job 21494, single process on GPU 0)")


# =================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-data", default=os.environ.get("STORY_CLUSTER_DATA", ""))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cd = args.cluster_data

    old = load(f"{D}/linear_op.csv")
    rerun = load(f"{D}/runs/2026-09-16_0629_linear_op_dense3327_rerun/linear_op.csv")
    grid386 = load(f"{D}/runs/2026-09-15_1142_linear_op_grid386/linear_op.csv")
    legacy = load(f"{D}/runs/legacy_pre-2026-09_torch-sdpa/linear_op_maxtokens4096.csv")
    gcfix_p = os.path.join(REPO, "data/profiling_gcfix/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv")
    gcfix = load(gcfix_p) if os.path.exists(gcfix_p) else None
    probe = load(f"{D}/runs/2026-09-17_1016_linear_op_validation_grid_two_column_probe/linear_op.csv")
    ropefix = load(f"{D}/runs/2026-09-17_1250_linear_op_rope_fix_validation_grid/linear_op.csv")
    ropeko = load(f"{D}/runs/2026-09-17_1250_linear_op_rope_fix_validation_grid/linear_op_kernel_only.csv")

    fig00_baseline(args.out, legacy, grid386)
    fig01_spike_envelope(args.out, old)
    fig02_spike_samples(args.out, old)
    fig03_spike_band_two_runs(args.out, old, rerun)
    fig04_fingerprint(args.out, old, "data: canonical linear_op.csv (job 21313)", "f04_run_position_fingerprint_original.png",
                      "Chapter 1/3 — a second, subtler fingerprint: fixed run positions (11, 16–17, 33, 49) sit systematically above the row median in every row")
    fig06_gc_counter_model(args.out)
    if gcfix is not None:
        fig07_gcfix_before_after(args.out, rerun, gcfix)
        fig08_fingerprint_before_after(args.out, rerun, gcfix)
    else:
        print("skip f07/f08: data/profiling_gcfix missing")
    fig09_dip(args.out, old)
    fig10_dip_samples(args.out, old)
    fig13_oproj_tiles(args.out)
    fig14_validation_two_column(args.out, probe)
    fig15_clock_probe(args.out, probe)
    fig16_rope(args.out, probe, ropefix, ropeko)

    def cpath(*p):
        q = os.path.join(cd, *p)
        return q if cd and os.path.exists(q) else None

    x1 = cpath("profiling_dip_x1", "linear_op.csv")
    if x1:
        x1 = load(x1); fig11_dip_backlog(args.out, old, x1); fig12_inflation_heatmap(args.out, old, x1)
    else:
        print("skip f11/f12: profiling_dip_x1 missing")
    e1b = cpath("spike_diag_e1b")
    if e1b:
        fig05_gc_gen2_vs_spike(args.out, e1b)
    nf = cpath("profiling_dense_fixed", "linear_op.csv")
    if nf:
        nf = load(nf)
        fig17_dense_fixed_two_columns(args.out, nf); fig18_dip_gone(args.out, old, nf); fig19_inflation_ratio(args.out, old, nf)
        fig20_tile_steps(args.out, nf); fig21_collection_diag(args.out, nf); fig23_position_profile_25(args.out, nf)
        bf = cpath("profiling_dense_fixed_batchflush", "linear_op.csv")
        if bf:
            fig25_batchflush(args.out, nf, load(bf))
    else:
        print("skip f17–f23/f25: profiling_dense_fixed missing")
    d200 = cpath("profiling_dense200", "linear_op.csv")
    if d200:
        d200 = load(d200); fig22_stall_gone(args.out, rerun, d200); fig24_position_profile_200(args.out, d200)
    else:
        print("skip f22/f24: profiling_dense200 missing")
    pp = cpath("posprobe")
    if pp:
        fig26_clock_ramp(args.out, pp)


if __name__ == "__main__":
    sys.exit(main())
