#!/usr/bin/env python3
"""Publication figure set for the corrected story (story/review/STORY_CORRECTED.md).

Why this exists instead of re-running evolution/evolution.ipynb: that notebook's figures carry the superseded step
naming (its "iter 3" is job 21356, which is a DIAGNOSTIC, and jobs 21500/21519 are missing entirely). Publishing them
beside the corrected text would contradict it. The notebook is left untouched as the historical artifact.

Naming rule (user, 2026-10-11): titles and legends use human-readable step names; job ids go in the explaining text.
Each figure carries a small grey provenance footnote naming the job, so an extracted PNG is still traceable.

Read-only on all data. Writes only into story/review/figs/. Run: /usr/bin/python3 make_corrected_figures.py
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCR = Path("/tmp/claude-1001/-home-dn-Frontier-qwen3-profiling/839301bf-3ea8-4d67-9dad-d3ac176a2add/scratchpad/cluster_data")
FIGS = HERE / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
SHARED = "/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op"

# step key -> (human name, job, local path, shared dir)
STEPS = {
 "start":   ("Where we started",                   "inherited", REPO/"data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/legacy_pre-2026-09_torch-sdpa/linear_op_maxtokens4096.csv", "legacy_pre-2026-09_torch-sdpa"),
 "first":   ("First re-collection",                "21313", REPO/"data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv", "2026-09-15_1308_linear_op_dense3327"),
 "rerun":   ("The spike, reproduced",              "21334", REPO/"data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/2026-09-16_0629_linear_op_dense3327_rerun/linear_op.csv", "2026-09-16_0629_dense3327_rerun"),
 "diag":    ("Holding the GPU behind the host",    "21356", SCR/"profiling_dip_x1/linear_op.csv", "2026-09-16_0832_dip_x1_backlog_knob_diagnostic"),
 "gc":      ("Garbage collection removed",         "21361", REPO/"data/profiling_gcfix/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv", "2026-09-16_0853_gcfix_verification"),
 "gpu":     ("Timing the GPU, not the host",       "21483", SCR/"profiling_dense_fixed/linear_op.csv", "2026-09-22_0647_dense_two_column_rope_fix"),
 "barrier": ("Queue barrier removed",              "21500", SCR/"profiling_dense_fixed_batchflush/linear_op.csv", "2026-09-22_1525_dense_fixed_batchflush"),
 "backlog": ("A real workload as backlog",         "21519", SCR/"profiling_dense_fixed_workbacklog/linear_op.csv", "2026-09-22_1735_dense_workbacklog_settle3"),
 "final":   ("The dataset we handed over",         "21539", REPO/"data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv", "2026-09-23_0949_dense_workbacklog_settle8"),
}
# Section B shows the dataset ITERATIONS, oldest -> newest. The diagnostic is deliberately absent (different grid and
# timer); the re-run is absent too (same protocol as "First re-collection", it would overlap exactly).
EVOLUTION = ["start", "first", "gc", "gpu", "barrier", "backlog", "final"]
OPS = ["attn_pre_proj", "attn_post_proj", "attn_rope"]
TPS = [1, 2, 4, 8]
STAT = ["min", "max", "mean", "median"]

D = {}
for k, (_, _, p, _) in STEPS.items():
    cols = ["num_tokens", "num_tensor_parallel_workers"] + [f"time_stats.{o}.{s}" for o in OPS for s in STAT]
    d = pd.read_csv(p, usecols=cols)
    D[k] = d.rename(columns={"num_tensor_parallel_workers": "tp"})

def ser(k, op, tp, stat="median", xm=None):
    g = D[k][D[k].tp == tp].sort_values("num_tokens").dropna(subset=[f"time_stats.{op}.{stat}"])
    if xm: g = g[g.num_tokens <= xm]
    return g.num_tokens.values, g[f"time_stats.{op}.{stat}"].values

FONT = 12
plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT + 1, "axes.labelsize": FONT, "xtick.labelsize": FONT,
    "ytick.labelsize": FONT, "legend.fontsize": FONT - 1, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
TP_COLOR = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}
# oldest (red) -> newest (green), 7 steps
EVO_COLORS = ["#ff0000", "#ff7a00", "#e6b800", "#b5d334", "#7ac62e", "#3fae2a", "#00a000"]
DPI, FULLX, ZOOM = 150, 16384, 2048
def light(c, f=.6): return tuple(1 - f * (1 - v) for v in to_rgb(c))
def fmt_x(ax, xmax, narrow=False):
    if xmax and xmax <= 2048: ax.set_xticks([0, 1000, 2000] if narrow else [0, 500, 1000, 1500, 2000])
    else: ax.set_xticks([0, 8000, 16000] if narrow else [0, 4000, 8000, 12000, 16000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

MAN = []
def finish(fig, name, title, prov, ops, tps, steps, caveats):
    fig.subplots_adjust(bottom=max(0.22, fig.subplotpars.bottom))
    fig.text(0.005, 0.005, prov, fontsize=FONT - 3, color="#777", ha="left", va="bottom")
    fig.savefig(FIGS / name, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    MAN.append(dict(file=name, title=title, ops=ops, tps=tps,
                    steps=[dict(name=STEPS[s][0], job=STEPS[s][1], shared_dir=f"{SHARED}/{STEPS[s][3]}") for s in steps],
                    caveats=caveats))
    print("  ", name)

# ======================= SECTION A - the narrative blocks =======================
def a0_start(op):   # Where we started
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (ttl, xm) in zip(axes, (("the inherited grid stops at 4,096 tokens", None), (f"num_tokens <= {ZOOM:,}", ZOOM))):
        for tp in TPS:
            x, y = ser("start", op, tp, xm=xm); ax.plot(x, y, color=TP_COLOR[tp], lw=1.4, label=f"TP {tp}")
        ax.set_xlim(0, FULLX if xm is None else xm); ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlabel("num_tokens"); ax.set_ylabel("median time (ms)"); ax.set_title(ttl, loc="left"); fmt_x(ax, xm or 4096)
    axes[0].legend(ncol=2, frameon=False)
    finish(fig, f"a0_where_we_started__{op}.png", f"Where we started - {op}",
           "inherited collection (job n/a, before 2026-09-10): 258 token values, <=4,096, 20 timed runs, aggregates only",
           [op], TPS, ["start"],
           ["Qwen3-30B-A3B, verified from the file's own architecture columns.",
            "Host-bound at TP>1; attn_rope here timed a torch fallback that is not RoPE."])

def a1_spike(op):   # The spike
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0), sharey=True)
    n = 0
    for ax, tp in zip(axes, TPS):
        x, lo = ser("first", op, tp, "min"); _, hi = ser("first", op, tp, "max"); _, md = ser("first", op, tp, "median")
        ax.fill_between(x, lo, hi, color="#a3bfe6", alpha=.6, lw=0, label="min-max over 50 timed runs")
        ax.plot(x, md, color="k", lw=1.0, ls="--", label="median")
        bad = hi > 50
        if bad.any():
            ax.scatter(x[bad], hi[bad], s=16, color="#d03b3b", zorder=5, label="one run > 50 ms"); n += int(bad.sum())
        ax.set_yscale("log"); ax.set_xlim(0, FULLX); ax.set_title(f"TP {tp}", loc="left"); ax.set_xlabel("num_tokens"); fmt_x(ax, None, narrow=True)
    axes[0].set_ylabel("time (ms, log)"); axes[0].legend(loc="center right", fontsize=FONT - 3)
    finish(fig, f"a1_the_spike__{op}.png", f"The spike - {op}",
           f"first re-collection (job 21313): 3,327 token values, 3+50 runs kept in execution order; {n} affected rows in this figure",
           [op], TPS, ["first"],
           ["Log y: the outlier is ~1,100x the row median, so a linear axis would hide everything else.",
            "Medians are untouched by the spike; only mean and max are destroyed.",
            "It sits at 0-based timed run 11 (the 12th of 50, the 15th forward)."])

def a2_hostbound(op):   # The cost that fell as the work grew
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, logy in zip(axes, (False, True)):
        for tp in TPS:
            x, y = ser("gc", op, tp); ax.plot(x, y, color=TP_COLOR[tp], lw=1.3, label=f"TP {tp}")
        ax.set_xlim(0, FULLX); ax.set_xlabel("num_tokens"); ax.set_ylabel("median time (ms)"); fmt_x(ax, None)
        if logy: ax.set_yscale("log"); ax.set_title("same data, log y", loc="left")
        else: ax.set_ylim(0, ax.get_ylim()[1]); ax.set_title("linear y", loc="left")
    axes[0].legend(ncol=2, frameon=False)
    finish(fig, f"a2_cost_that_fell__{op}.png", f"The cost that fell as the work grew - {op}",
           "after the GC fix (job 21361): same measurement as the first re-collection, so the dip is still present",
           [op], TPS, ["gc"],
           ["A median that falls as the token count rises cannot be the hardware; this is what prompted the investigation.",
            "Monotonicity was never an acceptance criterion - the final dataset still has downward steps."])

def a3_diag(op):   # Holding the GPU behind the host
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0), sharey=False)
    for ax, tp in zip(axes, TPS):
        x1, y1 = ser("gc", op, tp); x2, y2 = ser("diag", op, tp)
        ax.plot(x1, y1, color=light(TP_COLOR[tp]), lw=1.4, label="timed as before")
        ax.plot(x2, y2, color=TP_COLOR[tp], lw=2.0, label="GPU held behind the host")
        ax.set_xlim(0, FULLX); ax.set_ylim(0, max(y1.max(), y2.max()) * 1.08)
        ax.set_title(f"TP {tp}", loc="left"); ax.set_xlabel("num_tokens"); fmt_x(ax, None, narrow=True)
    axes[0].set_ylabel("median time (ms)"); axes[0].legend(loc="upper left", fontsize=FONT - 3, frameon=False)
    finish(fig, f"a3_holding_gpu_behind_host__{op}.png", f"Holding the GPU behind the host - {op}",
           "diagnostic experiment (job 21356) against the GC-fixed collection (job 21361). NOT a dataset step",
           [op], TPS, ["gc", "diag"],
           ["This is a diagnostic, not an iteration: 4,800 rows, tokens >=1,024 only, the old single-column timer plus one opt-in backlog knob.",
            "One change that touches no kernel removes the dip at every op and TP - that is the evidence the event pairs had been timing the host.",
            "The two passes differ in clock state as well as queue state (~2.0 GHz idle-start vs ~2.4 GHz after the spin)."])

def a4_rope():   # The RoPE kernel that was not RoPE  (one figure, attn_rope only, all TP)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0))
    for ax, tp in zip(axes, TPS):
        x1, y1 = ser("diag", "attn_rope", tp); x2, y2 = ser("gpu", "attn_rope", tp)
        ax.plot(x1, y1, color=light(TP_COLOR[tp]), lw=1.4, label="torch fallback")
        ax.plot(x2, y2, color=TP_COLOR[tp], lw=2.0, label="fused kernel")
        ax.set_xlim(0, FULLX); ax.set_ylim(0, max(y1.max(), y2.max()) * 1.08)
        ax.set_title(f"TP {tp}", loc="left"); ax.set_xlabel("num_tokens"); fmt_x(ax, None, narrow=True)
    axes[0].set_ylabel("median time (ms)"); axes[0].legend(loc="upper left", fontsize=FONT - 3, frameon=False)
    finish(fig, "a4_rope_kernel_that_was_not_rope__attn_rope.png", "The RoPE kernel that was not RoPE - attn_rope",
           "torch fallback (job 21356) vs the fused vLLM kernel (job 21483); the fallback line starts at 1,024 tokens",
           ["attn_rope"], TPS, ["diag", "gpu"],
           ["Different computation, not a timing fix: the fallback was 13 launch-bound kernels rotating only the first 64 columns of head 0.",
            "The two jobs also differ in repetitions, timer and grid, so this ratio mixes the kernel change with the timing change.",
            "attn_pre_proj and attn_post_proj agree within 1.2 % across the same pair, which is what makes them controls."])

def a5_runpos(op):   # Patterns inside the repetitions
    def prof(k, op):
        p = STEPS[k][2]
        d = pd.read_csv(p, usecols=["num_tensor_parallel_workers", f"time_stats.{op}.samples", f"time_stats.{op}.warmup_count"], low_memory=False)
        out = {}
        for tp, g in d.groupby("num_tensor_parallel_workers"):
            rel = []
            for s, wu in zip(g[f"time_stats.{op}.samples"], g[f"time_stats.{op}.warmup_count"]):
                if not isinstance(s, str): continue
                a = np.asarray(json.loads(s), float)[int(wu):]
                if len(a): rel.append(a / np.median(a))
            if rel: out[int(tp)] = np.median(np.vstack(rel), axis=0)
        return out
    pg, pf = prof("gpu", op), prof("final", op)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (pr, ttl) in zip(axes, ((pg, "before the backlog and settle fixes"), (pf, "the dataset we handed over"))):
        for tp in TPS:
            if tp in pr: ax.plot(np.arange(1, len(pr[tp]) + 1), pr[tp], color=TP_COLOR[tp], lw=1.4, marker="o", ms=3, label=f"TP {tp}")
        ax.axhline(1.0, color="#888", lw=.6); ax.set_ylim(0.95, 1.13)
        ax.set_xlabel("timed run index (1 = first timed run)"); ax.set_title(ttl, loc="left")
    axes[0].set_ylabel("sample / row median"); axes[0].legend(ncol=2, frameon=False)
    finish(fig, f"a5_patterns_inside_repetitions__{op}.png", f"Patterns inside the repetitions - {op}",
           "job 21483 (left) vs job 21539 (right); each line is the median over all rows of sample_k / row median",
           [op], TPS, ["gpu", "final"],
           ["x is 1-based here. The investigation documents count the same positions from 0, which is why the spike is called 'run 11' there.",
            "Two separate defects: a within-block drift (the backlog spin let the clock domains idle) and a fixed-position bump (a queue barrier).",
            "Fixing them changed the medians by at most 0.2 % - the backlog change did that, not these."])

def a6_final(op):   # The dataset we handed over
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (ttl, xm) in zip(axes, (("full range", None), (f"num_tokens <= {ZOOM:,}", ZOOM))):
        for tp in TPS:
            x, y = ser("final", op, tp, xm=xm); ax.plot(x, y, color=TP_COLOR[tp], lw=1.4, label=f"TP {tp}")
        ax.set_xlim(0, FULLX if xm is None else xm); ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlabel("num_tokens"); ax.set_ylabel("median time (ms)"); ax.set_title(ttl, loc="left"); fmt_x(ax, xm)
    axes[0].legend(ncol=2, frameon=False)
    finish(fig, f"a6_dataset_we_handed_over__{op}.png", f"The dataset we handed over - {op}",
           "final collection (job 21539): 13,308 rows, 3+25 runs after 8 settle forwards, md5 3635e4d3",
           [op], TPS, ["final"],
           ["The visible steps are real hardware behaviour (tile selection), not measurement artefacts.",
            "43-48 % of consecutive-token steps still decrease, median -1.1 %, worst -42 % (attn_post_proj TP8).",
            "Scopes below ~10 us sit inside the instrument's few-microsecond floor."])

print("Section A:")
for op in OPS: a0_start(op)
for op in OPS: a1_spike(op)
for op in OPS: a2_hostbound(op)
for op in OPS: a3_diag(op)
a4_rope()
for op in OPS: a5_runpos(op)
for op in OPS: a6_final(op)

# ======================= SECTION B - evolution per (op, TP) =======================
print("Section B:")
for op in OPS:
    for tp in TPS:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.2))
        for ax, (kind, xm) in zip(axes, (("full range (log y)", None), (f"num_tokens <= {ZOOM:,} (linear y)", ZOOM))):
            vals = []
            for i, k in enumerate(EVOLUTION):
                x, y = ser(k, op, tp, xm=xm)
                if not len(x): continue
                vals.append(y)
                name, job, _, _ = STEPS[k]
                lab = name
                if k == "start": lab += "  [to 4,096 tokens]"
                ax.plot(x, y, color=EVO_COLORS[i], lw=2.4 if k in ("start", "final") else 1.3, label=lab,
                        zorder=3 + i)
            allv = np.concatenate(vals)
            if xm is None:
                ax.set_yscale("log"); ax.set_ylim(allv[allv > 0].min() / 1.15, allv.max() * 1.3)
            else:
                ax.set_ylim(0, allv.max() * 1.08)
            ax.set_xlim(0, FULLX if xm is None else xm); fmt_x(ax, xm)
            ax.set_xlabel("num_tokens"); ax.set_ylabel("median time (ms)"); ax.set_title(kind, loc="left")
        h, l = axes[1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=FONT - 1)
        fig.suptitle(f"{op}  |  TP {tp}", fontsize=FONT + 4, y=0.995)
        fig.tight_layout(rect=(0, 0.1, 1, 0.98))
        cav = ["The first three curves measure the host's launch span, the last four measure device time. "
               "Below the crossover they differ because of what is being measured, not because the hardware changed.",
               "The diagnostic (job 21356) is not shown: different grid and timer. The spike re-run is not shown: "
               "same protocol as the first re-collection, it would overlap exactly."]
        if op == "attn_rope":
            cav.append("attn_rope: the first three curves timed a torch fallback that is not RoPE; the last four use the fused kernel.")
        finish(fig, f"b_evolution__{op}_tp{tp}.png", f"How the data changed - {op}, TP {tp}",
               "seven dataset steps, oldest (red) to newest (green): jobs n/a, 21313, 21361, 21483, 21500, 21519, 21539",
               [op], [tp], EVOLUTION, cav)

json.dump(dict(generated="story/review/make_corrected_figures.py",
               naming="human-readable step names in titles and legends; job ids in the provenance footnote and captions",
               shared_store=SHARED, steps={k: dict(name=v[0], job=v[1], shared_dir=v[3]) for k, v in STEPS.items()},
               evolution_sequence=[STEPS[k][0] for k in EVOLUTION],
               excluded_from_evolution={"21356": "diagnostic, different grid and timer", "21334": "same protocol as the first re-collection"},
               figures=MAN), open(HERE / "figs_manifest.json", "w"), indent=1)
print(f"\n{len(MAN)} figures -> {FIGS}")
