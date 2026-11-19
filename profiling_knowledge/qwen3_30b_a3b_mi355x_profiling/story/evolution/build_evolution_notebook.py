#!/usr/bin/env python3
"""Build evolution.ipynb (nbformat). Run: python3 build_evolution_notebook.py"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Qwen3-30B-A3B / MI355X `linear_op` — the story of the data, iteration by iteration

Iteration names and numbers follow the Confluence page *"Qwen3-30B-A3B on MI355X: the linear_op profiling story"*
(source: `story/STORY.md`). Fixed scope: Qwen3-30B-A3B, ops `attn_pre_proj`, `attn_post_proj`, `attn_rope`, TP 1/2/4/8,
statistic = median per (op, TP, num_tokens) in ms unless stated. The three *replicated* ops (`emb`, `input_layernorm`,
`post_attention_layernorm`; not split by tensor parallelism, profiled once at TP 1) are added after the three sharded ops, at TP 1 only.

* **Section 1** — six blocks (Block 0-5). Each block: *Problem*, six figures (one per op) from the dataset that shows it, *Fix*.
* **Section 2** — one figure per op, one line per iteration (D0-D5): sharded ops with TP in columns and full range / `num_tokens <= 2048` in rows;
  replicated ops (TP 1 only) as two panels (full range | `<= 2048`).

Edit paths only in the config cell. Read-only on all data; no network.
""")

code(r"""
# ============================ CONFIG (edit here) ============================
from pathlib import Path

def find_repo(start=Path.cwd()):
    p = start.resolve()
    while not (p / "profiling_knowledge").exists():
        if p.parent == p: raise RuntimeError("repo root (dir containing profiling_knowledge/) not found")
        p = p.parent
    return p

REPO = find_repo()
EVO = REPO / "profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/story/evolution"
RUNS = REPO / "data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs"
LOCAL = REPO / "data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op"
# Cluster-scratch copies (not in git; STORY.md "Reproducing the figures"). If a source is missing the notebook falls back to
# the small cache that a previous run wrote to evolution/cache/.
SCRATCH = Path("/tmp/claude-1001/-home-dn-Frontier-qwen3-profiling/839301bf-3ea8-4d67-9dad-d3ac176a2add/scratchpad/cluster_data")

SHARDED_OPS = ["attn_pre_proj", "attn_post_proj", "attn_rope"]
REPLICATED_OPS = ["emb", "input_layernorm", "post_attention_layernorm"]   # not split by TP: profiled at TP 1 only
OPS = SHARDED_OPS + REPLICATED_OPS
TPS = [1, 2, 4, 8]
TPS_OF = {op: ([1] if op in REPLICATED_OPS else TPS) for op in OPS}
ZOOM = 2048
TP_COL = "num_tensor_parallel_workers"
COL = "time_stats.{op}.median"          # primary column at every stage (host-bound in D0-D2, GPU-bound in D3-D5)

# kind: what the primary column measures at that stage.  samples=True -> per-run samples are read (run-position profile).
DATASETS = [
    dict(id="D0", label="Start", block="Starting point", ch="Ch. 0", job="n/a (inherited)", commit="before 2026-09-10",
         path=RUNS / "legacy_pre-2026-09_torch-sdpa/linear_op_maxtokens4096.csv", kind="host-bound", samples=False),
    dict(id="D1", label="First re-collection", block="Spikes", ch="Ch. 1", job="21313", commit="da1ae75",
         path=REPO / "data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv", kind="host-bound", samples=False),
    dict(id="D2", label="GC fixed", block="V-shaped dip / host time", ch="Ch. 4", job="21361 (21362 diag)", commit="74eee7c",
         path=REPO / "data/profiling_gcfix/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv", kind="host-bound", samples=False),
    dict(id="D3", label="Host fix", block="RoPE", ch="Ch. 5-7", job="21356", commit="(knob FRONTIER_GPU_BACKLOG_MS=100; uncommitted)",
         path=SCRATCH / "profiling_dip_x1/linear_op.csv", kind="GPU-bound (backlog knob)", samples=False),
    dict(id="D4", label="RoPE fix", block="Run-position anomalies", ch="Ch. 8-9", job="21483", commit="4b8fce3 (d4469be)",
         path=SCRATCH / "profiling_dense_fixed/linear_op.csv", kind="GPU-bound", samples=True),
    dict(id="D5", label="Run-position fixes (final)", block="Where we are", ch="Ch. 10-11", job="21539", commit="3a8eb55, 88da6a4",
         path=LOCAL / "2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv", kind="GPU-bound", samples=True),
]
EXPECTED_MD5 = {"D5": "3635e4d305ae5cf84884e275d3ff37ef"}
REF = "D1"            # the dense 3,327-token grid the others are compared against

FIGS1, FIGS2, CACHE = EVO / "figs/section1", EVO / "figs/section2", EVO / "cache"
BLOCKS = ["b0_start", "b1_spikes", "b2_dip", "b3_rope", "b4_runpos", "b5_now"]
DPI, FONT, W, H = 200, 12, 13.3, 3.8
SPIKE_MS = 50.0
TP_COLOR = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}   # fixed per TP in every figure
ITER_COLOR = ["#cfd9e8", "#a3bfe6", "#7aa3dd", "#4f7fcb", "#2a56a3", "#0b2f66"]  # light (D0) -> dark (D5)
""")

code(r"""
import hashlib, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Image
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT + 1, "axes.labelsize": FONT, "xtick.labelsize": FONT,
                     "ytick.labelsize": FONT, "legend.fontsize": FONT, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
for d in (FIGS1, FIGS2, CACHE): d.mkdir(parents=True, exist_ok=True)
STAT = ["min", "max", "mean", "median"]
""")

md("## Dataset discovery table (D0-D5)")

code(r"""
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def run_position_profile(df, op):
    # {tp: [median over rows of sample_k / row median, k = 1..N timed runs]}
    out = {}
    for tp, g in df[df[f"time_stats.{op}.samples"].notna()].groupby(TP_COL):
        rel = []
        for s, wu in zip(g[f"time_stats.{op}.samples"], g[f"time_stats.{op}.warmup_count"]):
            a = np.asarray(json.loads(s), float)[int(wu):]
            rel.append(a / np.median(a))
        out[int(tp)] = np.median(np.vstack(rel), axis=0).tolist()
    return out

DATA, META = {}, {}
for ds in DATASETS:
    k, src = ds["id"], Path(ds["path"])
    cache, cmeta = CACHE / f"{k}.csv", CACHE / f"{k}.meta.json"
    if src.exists():
        head = pd.read_csv(src, nrows=1)
        need = ["num_tokens", TP_COL] + [f"time_stats.{op}.{s}" for op in OPS for s in STAT] + [f"time_stats.{op}.count" for op in OPS]
        extra = [c for c in ["time_stats.attn_pre_proj.warmup_count", "attn_rope_impl", "settle_steps", "backlog_kind", "attention_backend"] if c in head.columns]
        samp = [f"time_stats.{op}.{s}" for op in OPS for s in ("samples", "warmup_count")] if ds["samples"] else []
        full = pd.read_csv(src, usecols=need + extra + samp)
        df = full[need].rename(columns={TP_COL: "tp"})
        df.columns = [c.replace("time_stats.", "").replace(".", "__") for c in df.columns]
        meta = dict(source=str(src), md5=md5(src), rows=len(full), from_cache=False,
                    reps=sorted(full["time_stats.attn_pre_proj.count"].unique().tolist()),
                    reps_by_op={op: sorted(full[f"time_stats.{op}.count"].dropna().unique().tolist()) for op in OPS},
                    rope_impl=sorted(full["attn_rope_impl"].dropna().unique().tolist()) if "attn_rope_impl" in full else None,
                    settle=sorted(full["settle_steps"].dropna().unique().tolist()) if "settle_steps" in full else None,
                    backlog_kind=sorted(full["backlog_kind"].dropna().unique().tolist()) if "backlog_kind" in full else None,
                    profile={op: run_position_profile(full, op) for op in OPS} if ds["samples"] else None)
        df.to_csv(cache, index=False); json.dump(meta, open(cmeta, "w"))
    elif cache.exists() and cmeta.exists():
        df, meta = pd.read_csv(cache), json.load(open(cmeta)); meta["from_cache"] = True
        print(f"[{k}] source missing -> using cache {cache.name}")
    else:
        raise FileNotFoundError(f"[{k}] neither {src} nor a cache exists; edit DATASETS in the config cell")
    if k in EXPECTED_MD5 and meta["md5"] != EXPECTED_MD5[k]: raise AssertionError(f"[{k}] md5 {meta['md5']} != {EXPECTED_MD5[k]}")
    if meta["profile"]: meta["profile"] = {op: {int(tp): v for tp, v in d.items()} for op, d in meta["profile"].items()}
    DATA[k], META[k] = df, meta

rows = []
for ds in DATASETS:
    d, m = DATA[ds["id"]], META[ds["id"]]
    rows.append({"id": ds["id"], "iteration": ds["label"], "story": ds["ch"], "job": ds["job"], "commit": ds["commit"], "rows": m["rows"],
                 "token values": d.num_tokens.nunique(), "token range": f"{d.num_tokens.min()}-{d.num_tokens.max()}", "timed reps/shape": m["reps"], "reps (emb)": m["reps_by_op"]["emb"],
                 "time column": f"time_stats.<op>.median ({ds['kind']})", "md5": m["md5"][:8], "path": m["source"] + (" [cache]" if m["from_cache"] else "")})
TABLE = pd.DataFrame(rows); TABLE.to_csv(EVO / "dataset_table.csv", index=False)
pd.set_option("display.width", 260, "display.max_colwidth", 60)
display(TABLE.drop(columns="path"))
for r in rows: print(f"{r['id']} {r['path']}")
""")

code(r"""
# helpers
LABEL = {d["id"]: d["label"] for d in DATASETS}
JOB = {d["id"]: d["job"] for d in DATASETS}
REG = {}      # figure name -> list of per-panel facts, for the verification cell

def ser(did, op, tp, stat="median", zoom=False):
    d = DATA[did]; d = d[d.tp == tp].sort_values("num_tokens")
    if zoom: d = d[d.num_tokens <= ZOOM]
    return d.num_tokens.values, d[f"{op}__{stat}"].values

def xmax(did): return int(DATA[did].num_tokens.max())

def finish(fig, path, name, show):
    fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    if show: display(Image(filename=str(path), width=1100))

def median_lines(ax, did, op, zoom, kind="plot"):
    for tp in TPS_OF[op]:
        x, y = ser(did, op, tp, zoom=zoom)
        ax.plot(x, y, color=TP_COLOR[tp], lw=1.1, label=f"TP {tp}")

def two_panel(did, op, block_key, block_name, full_log=False, note=None):
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    for ax, zoom in zip(axes, (False, True)):
        median_lines(ax, did, op, zoom)
        ax.set_xlim(0, ZOOM if zoom else xmax(did)); ax.set_xlabel("num_tokens"); ax.set_ylabel("median time (ms)")
        ax.set_title(f"num_tokens <= {ZOOM}" if zoom else "full range" + (" (log y)" if full_log else ""), loc="left")
        if full_log and not zoom: ax.set_yscale("log")
        else: ax.set_ylim(0, ax.get_ylim()[1])
    axes[0].legend(loc="upper left", ncol=2)
    fig.suptitle(f"{op} | {block_name} | job {JOB[did]}", fontsize=FONT + 3, y=1.02)
    if note: fig.text(0.01, -0.03, note, fontsize=FONT - 1, color="#555", ha="left", va="top")
    fig.tight_layout()
    name = f"{block_key}__{op}"
    REG[name] = [dict(lines=len(a.lines)) for a in axes]
    finish(fig, FIGS1 / f"{name}.png", name, True)
""")

md(r"""
# Section 1 — what each iteration did
""")

# ------------------------------------------------------------------ Block 0
md(r"""
## Block 0 — Starting point *(Ch. 0, data D0: inherited, `<= 4,096` tokens, 20 reps)*

**Problem.** The data does not show the clean "more tokens, more work" trend we expect (maybe in steps, but rising). It is not random noise either:
there is a signal, but not the one we want to measure.
""")
code(r"""
for op in OPS: two_panel("D0", op, "b0_start", "Starting point", note="D0 stops at 4,096 tokens (inherited grid, 20 timed reps); host-bound primary column.")
""")
md(r"""
**Fix.** Re-collect: 50 timed + 3 warm-up repetitions (was 20), every run kept in execution order, token range extended from 4,096 to 16,384,
dense grid of 3,327 token counts, AITER attention kernels.
""")

# ------------------------------------------------------------------ Block 1
md(r"""
## Block 1 — Spikes *(Ch. 1, data D1: job 21313, dense grid, 50 reps)*

**Problem.** In 32 rows (tokens 1968-1975 x 4 TP) one timed run of `attn_pre_proj` (run 11) took 150-290 ms vs ~0.13 ms; reproduced exactly on a re-run
(job 21334). Medians are untouched; mean and max are destroyed. *(Medians alone hide the spike, so the figures show the min-max envelope, mean and median.)*
""")
code(r"""
def envelope_fig(did, op, block_key, block_name):
    fig, axes = plt.subplots(1, len(TPS_OF[op]), figsize=(W, H), sharey=True); axes = np.atleast_1d(axes)
    facts, n_dots = [], 0
    for ax, tp in zip(axes, TPS_OF[op]):
        x, lo = ser(did, op, tp, "min"); _, hi = ser(did, op, tp, "max"); _, mean = ser(did, op, tp, "mean"); _, med = ser(did, op, tp)
        ax.fill_between(x, lo, hi, color="#a3bfe6", alpha=.6, lw=0, label="min-max envelope")
        ax.plot(x, mean, color=TP_COLOR[tp], lw=1.0, label="mean")
        ax.plot(x, med, color="k", lw=1.0, ls="--", label="median")
        bad = hi > SPIKE_MS
        if bad.any(): ax.scatter(x[bad], hi[bad], s=14, color="#d03b3b", zorder=5, label=f"max > {SPIKE_MS:g} ms"); n_dots += int(bad.sum())
        ax.set_yscale("log"); ax.set_xlim(0, xmax(did)); ax.set_xlabel("num_tokens"); ax.set_title(f"TP {tp}", loc="left")
        facts.append(dict(lines=len(ax.lines), envelope=len(ax.collections) >= 1, labels=[l.get_label() for l in ax.lines], dots=int(bad.sum())))
    axes[0].set_ylabel("time (ms, log)"); axes[0].legend(loc="center right", fontsize=FONT - 2)
    fig.suptitle(f"{op} | {block_name} | job {JOB[did]}", fontsize=FONT + 3, y=1.02)
    fig.tight_layout()
    name = f"{block_key}__{op}"; REG[name] = facts
    finish(fig, FIGS1 / f"{name}.png", name, True)
    return n_dots

dots = {op: envelope_fig("D1", op, "b1_spikes", "Spikes") for op in OPS}
print("red dots (rows with max >", SPIKE_MS, "ms):", dots, "| expected attn_pre_proj: 32 (8 tokens x 4 TP)")
""")
md(r"""
**Fix.** Root cause was a CPython generation-2 garbage collection inside the timed loop. `gc.disable()` around the timed loop (commit `74eee7c`).
Verified on the full grid: largest sample 2.09 ms (jobs 21361/21362).
""")
code(r"""
print("D1 largest attn_pre_proj sample:", round(DATA["D1"]["attn_pre_proj__max"].max(), 3), "ms")
print("D2 largest attn_pre_proj sample:", round(DATA["D2"]["attn_pre_proj__max"].max(), 3), "ms   (STORY.md: 2.09 ms)")
""")

# ------------------------------------------------------------------ Block 2
md(r"""
## Block 2 — V-shaped dip / host time *(Ch. 4-7, data D2: GC fixed, jobs 21361/21362, host-bound column)*

**Problem.** `attn_rope` (TP 2/4/8) and `attn_pre_proj` (TP 8) medians FALL about 25-28% below their low-token plateau, at a token count that moves with TP (`attn_rope`: about 3.7k tokens at TP 2, 6.8k at TP 4, 9k at TP 8), then climb again; `attn_post_proj` shows flat plateaus and steps instead of a steady rise. More tokens cannot mean less time: a bug in the measurement.
""")
code(r"""
for op in OPS: two_panel("D2", op, "b2_dip", "V-shaped dip", full_log=True)
""")
md(r"""
**Fix.** The CUDA-event pairs measured the host's launch span whenever the GPU was idle. Every shape is now timed twice; the GPU-bound pass (device kept behind
the host by a calibrated backlog) is the primary column, the legacy loop is kept as `time_stats_hostbound` (commit `d4469be`).
""")

# ------------------------------------------------------------------ Block 3
md(r"""
## Block 3 — RoPE *(Ch. 8, data D3: job 21356, GPU-bound via the backlog knob, `attn_rope` still the torch fallback)*

**Problem.** `attn_rope` timed a forced torch fallback: 13 launch-bound kernels that rotated only the first 64 columns of head 0, so it was timing a
computation that is not RoPE. `attn_pre_proj` and `attn_post_proj` are unaffected (they act as controls in these figures).

*D3 has no data below 1,024 tokens, so a `<= 2048` panel would show only 1,024-2,048 and not the problem: these figures use one full-range panel.*
""")
code(r"""
D3_NOTE = "D3 has no data below 1,024 tokens."
def single_panel(did, op, block_key, block_name, note):
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    median_lines(ax, did, op, False)
    ax.set_xlim(0, xmax(did)); ax.set_ylim(0, ax.get_ylim()[1]); ax.set_xlabel("num_tokens"); ax.set_ylabel("median time (ms)")
    ax.set_title("full range", loc="left"); ax.legend(loc="upper left", ncol=4)
    fig.suptitle(f"{op} | {block_name} | job {JOB[did]}", fontsize=FONT + 3, y=1.02)
    fig.text(0.01, -0.03, note, fontsize=FONT - 1, color="#555", ha="left", va="top")
    fig.tight_layout()
    name = f"{block_key}__{op}"; REG[name] = [dict(lines=len(ax.lines))]
    finish(fig, FIGS1 / f"{name}.png", name, True)
for op in OPS:
    extra = {"attn_rope": " attn_rope here is still the torch fallback.", "attn_pre_proj": " Control: unaffected by the RoPE fix.", "attn_post_proj": " Control: unaffected by the RoPE fix."}.get(op, " Replicated op (TP 1 only), unaffected by the RoPE fix.")
    single_panel("D3", op, "b3_rope", "RoPE", D3_NOTE + extra)
""")
md(r"""
**Fix.** `get_rope` prefers Frontier's `RotaryEmbedding` with `vllm._custom_ops.rotary_embedding` (the fused kernel); `attn_rope_impl` is recorded per row and
`sanity_check` rejects anything but `vllm_kernel` (commit `4b8fce3`). The fused kernel is 2-8x cheaper.
""")

# ------------------------------------------------------------------ Block 4
md(r"""
## Block 4 — Run-position anomalies *(Ch. 9-10, data D4: job 21483, left; D5: job 21539 after the fix, right)*

**Problem.** Hidden in per-run samples: (1) a 2-8% within-block drift (the spin backlog lets clocks idle); (2) a ~5.5 us spike at run 20 for TP>1
(ROCm runtime barrier every 1000 commands). Medians were taken mid-ramp, 2-3% too high.

*Figures: x = timed run index 1..25, y = median over rows of (sample / row median), one line per TP.*
""")
code(r"""
def runpos_fig(op, block_key, block_name):
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)
    facts = []
    for ax, did, ttl in zip(axes, ("D4", "D5"), ("D4 (job 21483): problem", "D5 (job 21539): after the fix")):
        for tp in TPS_OF[op]:
            v = META[did]["profile"][op][tp]
            ax.plot(np.arange(1, len(v) + 1), v, color=TP_COLOR[tp], lw=1.4, marker="o", ms=3, label=f"TP {tp}")
        ax.axhline(1.0, color="#888", lw=.6)   # reference, not counted as a data line below
        ax.set_ylim(0.95, 1.15); ax.set_xlim(0.5, len(v) + 0.5); ax.set_xlabel("timed run index"); ax.set_title(ttl, loc="left")
        facts.append(dict(lines=len(ax.lines) - 1))
    axes[0].set_ylabel("sample / row median"); axes[0].legend(loc="upper left", ncol=2)
    fig.suptitle(f"{op} | {block_name} | job 21483 -> 21539", fontsize=FONT + 3, y=1.02)
    fig.tight_layout()
    name = f"{block_key}__{op}"; REG[name] = facts
    finish(fig, FIGS1 / f"{name}.png", name, True)
for op in OPS: runpos_fig(op, "b4_runpos", "Run-position anomalies")
""")
md(r"""
**Fix.** `DEBUG_CLR_MAX_BATCH_SIZE=1000000`; allocating-GEMM backlog (`BACKLOG_KIND=gemm_alloc`) instead of the spin; 8 untimed settle forwards
(`SETTLE_STEPS=8`) (commits `3a8eb55`, `88da6a4`).
""")

# ------------------------------------------------------------------ Block 5
md(r"""
## Block 5 — Where we are *(Ch. 11, data D5: job 21539, the handed-off dataset)*

Not done, there may be more bugs, but we are on the right track. Still open: slow 0.9-1.3% decline per block at TP 1/2; tile-selection steps of 15-30% are real
hardware behavior; kernels below ~10 us sit inside the +-3-5 us instrument floor; locked-clock dataset not collected.
""")
code(r"""
for op in OPS: two_panel("D5", op, "b5_now", "Where we are", note="Final dataset (md5 3635e4d3); GPU-bound primary column.")
""")

# ------------------------------------------------------------------ Section 2
md(r"""
# Section 2 — per-op, per-TP evolution (one figure per (op, TP))

One figure per (op, TP); never mixed. Each figure has two stacked panels with the same x meaning and y label: **full range** and a **zoom** into the small-token region.
One curve per run, chronological: baseline (D0) -> iter 1 (D1) -> ... -> iter 5 (D5); colour goes red -> green. The plotted values are the per-shape **medians**
(the data is already aggregated, so outliers are filtered with a rolling-median rule rather than a global one). Clipped points are never hidden: they are marked
with a triangle at the top edge and counted in the legend.
""")

code(r"""
# ============================ SECTION 2 CONFIG (tune per op here) ============================
S2_RUNS = [("D0", "baseline"), ("D1", "iter 1"), ("D2", "iter 2"), ("D3", "iter 3"), ("D4", "iter 4"), ("D5", "iter 5")]   # chronological
S2_FIGSIZE = (14, 11.5)      # two stacked panels + a legend row underneath
S2_FULL_YSCALE = "log"       # "log": spikes stay visible, nothing clipped | "clipped": same robust limits as the zoom panel
S2_DEFAULT = dict(
    zoom=(1, 2048),          # (x_lo, x_hi) of the zoom panel, in num_tokens
    roll_win=21,             # rolling-median window (points along x, per run)
    F=3.0,                   # outlier if y > rolling_median * F ...
    k=5.0,                   # ... or y > rolling_median + k * MAD (rolling, scaled 1.4826); the more permissive (larger) threshold wins
    margin=0.08,             # head-room on the robust y-limits
    N=100,                   # sanity bound: first/last N points of each run in the panel
    tail_mult=1.3,           # ymax2 = tail_mult * p95(last N points across runs)
)
S2_OVERRIDE = {              # per-op overrides, e.g. "attn_rope": dict(zoom=(1, 4096), F=2.5)
}
S2_COLORS6 = ["#ff0000", "#ff7a00", "#e6b800", "#b5d334", "#5fb52e", "#00a000"]   # baseline pure red ... latest clear green
def s2_params(op): return {**S2_DEFAULT, **S2_OVERRIDE.get(op, {})}
def s2_palette(n):
    if n == len(S2_COLORS6): return S2_COLORS6
    cm = plt.get_cmap("RdYlGn"); c = [matplotlib.colors.to_hex(cm(v)) for v in np.linspace(0, 1, n)]; c[0], c[-1] = "#ff0000", "#00a000"; return c
""")

code(r"""
def rolling_flags(y, win, F, k):
    # True where y is an upward outlier vs its local rolling median (threshold = max(F*rm, rm + k*MAD))
    s = pd.Series(y); mp = max(3, win // 2)
    rm = s.rolling(win, center=True, min_periods=mp).median()
    mad = (s - rm).abs().rolling(win, center=True, min_periods=mp).median() * 1.4826
    thr = np.maximum(rm * F, rm + k * mad)
    return (s > thr).values

def robust_ylim(runs_xy, P):
    # runs_xy: list of (x, y) already restricted to the panel's x-range, sorted by x
    good = np.concatenate([y[~rolling_flags(y, P["roll_win"], P["F"], P["k"])] for x, y in runs_xy if len(y)])
    lo1, hi1 = good.min() * (1 - P["margin"]), good.max() * (1 + P["margin"])
    tail = np.concatenate([y[-P["N"]:] for x, y in runs_xy if len(y)]); head = np.concatenate([y[:P["N"]] for x, y in runs_xy if len(y)])
    hi2, lo2 = P["tail_mult"] * np.percentile(tail, 95), head.min()
    return min(lo1, lo2), max(hi1, hi2)          # union: the wider of the two

def panel_data(op, tp, did, xr=None):
    x, y = ser(did, op, tp); m = ~np.isnan(y)
    x, y = x[m], y[m]
    if xr is not None: k = (x >= xr[0]) & (x <= xr[1]); x, y = x[k], y[k]
    return x, y

S2_COUNTS, S2_REG = [], {}
for op in OPS:
    P = s2_params(op)
    for tp in TPS_OF[op]:
        pal = s2_palette(len(S2_RUNS))
        fig, axes = plt.subplots(2, 1, figsize=S2_FIGSIZE)
        panels = [("full", None), ("zoom", P["zoom"])]
        facts = []
        for ax, (kind, xr) in zip(axes, panels):
            data = {did: panel_data(op, tp, did, xr) for did, _ in S2_RUNS}
            have = {did: xy for did, xy in data.items() if len(xy[0])}
            logmode = kind == "full" and S2_FULL_YSCALE == "log"
            allpos = np.concatenate([y for x, y in have.values()])
            if logmode:
                lo, hi = allpos[allpos > 0].min() / 1.15, allpos.max() * 1.3
            else:
                lo, hi = robust_ylim(list(have.values()), P)
            nlines = 0
            for i, (did, rname) in enumerate(S2_RUNS):
                x, y = data[did]
                lab = f"{rname} - {LABEL[did]}"
                if not len(x): lab += " (no data for this op/TP)"; ax.plot([], [], color=pal[i], lw=1.4, label=lab); continue
                fx, _ = panel_data(op, tp, did); gx1 = max(panel_data(op, tp, d)[0].max() for d, _ in S2_RUNS if len(panel_data(op, tp, d)[0]))
                if fx.min() > 100 or fx.max() < 0.9 * gx1: lab += f" [data {int(fx.min()):,}-{int(fx.max()):,} tokens]"
                over = y > hi
                nflag = int(rolling_flags(y, P["roll_win"], P["F"], P["k"]).sum())
                if not logmode and over.any():
                    ax.scatter(x[over], np.full(over.sum(), 0.985), marker="^", s=38, color=pal[i], transform=ax.get_xaxis_transform(), clip_on=False, zorder=20)
                if not logmode: lab += f": {int(over.sum())} points above range"
                ax.plot(x, y, color=pal[i], lw=2.6 if i == 0 else 1.3, label=lab, zorder=3 + i); nlines += 1
                if kind == "zoom" or (kind == "full" and not logmode): S2_COUNTS.append(dict(op=op, tp=tp, panel=kind, run=rname, did=did, clipped=int(over.sum()), flagged=nflag, points=len(y)))
            if kind == "full" and logmode:
                for did, rname in S2_RUNS: S2_COUNTS.append(dict(op=op, tp=tp, panel="full(log)", run=rname, did=did, clipped=0, flagged=-1, points=len(data[did][0])))
            ax.set_ylim(lo, hi)
            if logmode: ax.set_yscale("log")
            xs = np.concatenate([x for x, y in have.values()])
            ax.set_xlim((0, xs.max()) if xr is None else (xr[0] - 0.01 * xr[1], xr[1]))
            ax.set_ylabel("median time (ms" + (", log)" if logmode else ")")); ax.set_xlabel("num_tokens")
            ax.set_title((f"Full range: num_tokens {int(xs.min()):,}-{int(xs.max()):,} ({'log' if logmode else 'linear, robust limits'} y)" if kind == "full"
                          else f"Zoom: num_tokens {xr[0]:,}-{xr[1]:,} (linear y, robust limits [{lo:.4g}, {hi:.4g}] ms)"), loc="left")
            if kind == "zoom": zoom_handles = ax.get_legend_handles_labels()
            facts.append(dict(kind=kind, lines=nlines))
        tag = "replicated op, TP 1 only" if op in REPLICATED_OPS else f"TP {tp}"
        fig.suptitle(f"{op} | {tag}", fontsize=FONT + 5, y=0.995)
        fig.legend(*zoom_handles, loc="lower center", ncol=2, frameon=False, fontsize=FONT - 1, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.13, 1, 0.98))
        S2_REG[(op, tp)] = facts
        pth = FIGS2 / f"{op}_tp{tp}.png"; fig.savefig(pth, dpi=DPI // 2, bbox_inches="tight"); plt.close(fig)
        display(Image(filename=str(pth), width=950))

CNT = pd.DataFrame(S2_COUNTS)
CNT.to_csv(EVO / "section2_clipped_counts.csv", index=False)
ORDER_RUNS = [r for _, r in S2_RUNS]
print("clipped points (above the y-range) per (op, TP, panel, run):")
display(CNT[CNT.panel != "full(log)"].pivot_table(index=["op", "tp", "panel"], columns="run", values="clipped", aggfunc="sum")[ORDER_RUNS])
print("points flagged as outliers by the rolling-median rule (flagged points are excluded when the y-limits are computed):")
display(CNT[CNT.panel != "full(log)"].pivot_table(index=["op", "tp", "panel"], columns="run", values="flagged", aggfunc="sum")[ORDER_RUNS])
print("chronological run order (colour red -> green):", " -> ".join(f"{r} = {d} {LABEL[d]} (job {JOB[d]})" for d, r in S2_RUNS))
""")

md("# Checks")
code(r"""
# verification: files + lines per panel (fails loudly)
exp1 = [f"{b}__{op}.png" for b in BLOCKS for op in OPS]; exp2 = [f"{op}_tp{tp}.png" for op in OPS for tp in TPS_OF[op]]
got1 = sorted(p.name for p in FIGS1.glob("*.png")); got2 = sorted(p.name for p in FIGS2.glob("*.png"))
print(f"section1: {len(got1)} figures (expected {len(exp1)} = 6 blocks x 6 ops); section2: {len(got2)} figures (expected {len(exp2)})")
assert got1 == sorted(exp1), f"section1 mismatch: missing {set(exp1)-set(got1)} extra {set(got1)-set(exp1)}"
assert got2 == sorted(exp2), f"section2 mismatch: missing {set(exp2)-set(got2)} extra {set(got2)-set(exp2)}"
for (op, tp), f in S2_REG.items():
    assert len(f) == 2 and all(p["lines"] == len(S2_RUNS) for p in f), f"section2 {op} TP{tp}: lines per panel {[p['lines'] for p in f]}"
print(f"section2: {len(S2_REG)} figures, 2 stacked panels each, {len(S2_RUNS)} curves per panel OK")
for op in OPS:
    f = REG[f"b1_spikes__{op}"]; assert len(f) == len(TPS_OF[op])
    for p in f: assert p["envelope"] and "mean" in p["labels"] and "median" in p["labels"], f"block1 {op}: {p}"
print("block1: envelope + mean + median present in every panel OK")
print("block1: spike rows (max >", SPIKE_MS, "ms) per op:", {op: sum(p["dots"] for p in REG[f"b1_spikes__{op}"]) for op in OPS})
assert sum(p["dots"] for p in REG["b1_spikes__attn_pre_proj"]) == 32, "expected 32 spike rows for attn_pre_proj"
for op in OPS:
    assert all(p["lines"] == len(TPS_OF[op]) for p in REG[f"b4_runpos__{op}"]), f"block4 {op}"
    assert REG[f"b3_rope__{op}"][0]["lines"] == len(TPS_OF[op])
print("blocks 3-4: one line per TP of the op OK")
""")

md("# Inconsistencies between the datasets")
code(r"""
issues = []
ORDER_IDS = [d["id"] for d in DATASETS]
ref_tok = set(DATA[REF].num_tokens.unique())
for ds in DATASETS:
    k, d, m = ds["id"], DATA[ds["id"]], META[ds["id"]]
    toks = set(d.num_tokens.unique())
    if toks != ref_tok:
        issues.append(f"{k} {ds['label']}: token grid differs from {REF} - {len(toks)} values in {min(toks)}-{max(toks)} ({len(toks & ref_tok)} shared with {REF}, {len(toks - ref_tok)} only here)")
    for op in OPS:
        if f"{op}__median" not in d: issues.append(f"{k}: op {op} missing"); continue
        for tp in TPS_OF[op]:
            n = int(d[(d.tp == tp)][f"{op}__median"].notna().sum())
            if n == 0: issues.append(f"{k}: {op} TP {tp} missing")
            elif n != len(toks): issues.append(f"{k}: {op} TP {tp} has {n} non-NaN medians for {len(toks)} token values")
        if op in REPLICATED_OPS and d[(d.tp != 1)][f"{op}__median"].notna().any(): issues.append(f"{k}: {op} has values at TP>1 (expected NaN, replicated op)")
reps = {ds["id"]: META[ds["id"]]["reps"] for ds in DATASETS}
issues.append(f"timed repetitions per shape differ (medians are over different counts): {reps}")
issues.append("timed repetitions of the replicated ops differ from the sharded ops (emb: 100 in the 50-rep runs): " + str({k: META[k]["reps_by_op"] for k in ORDER_IDS}))
issues.append("Replicated ops (emb, input_layernorm, post_attention_layernorm) exist at TP 1 only, so they have no TP comparison; their medians are NaN at TP>1 by design.")
issues.append("D3 (job 21356) is NOT a full-grid run: 1,200 token counts, 1,024-16,384 tokens only; no data below 1,024, so D3 never appears in a `<= 2048` panel except for 1,024-2,048 (Section 2).")
issues.append("D0 is the inherited file: <= 4,096 tokens, 258 token values, TORCH_SDPA-era collection, 20 reps, aggregates only (no per-run samples).")
issues.append("Time column semantics: D0-D2 primary column is host-bound (CUDA-event span of the host launch); D3 is GPU-bound via the opt-in FRONTIER_GPU_BACKLOG_MS=100 knob; D4-D5 are GPU-bound with the two-column method. Host-bound vs GPU-bound steps differ in what is measured, not just how well.")
issues.append("attn_rope: D0-D3 timed the forced torch fallback (no attn_rope_impl column); D4-D5 use the fused kernel (attn_rope_impl = " + str(META['D4']['rope_impl']) + "). attn_rope is not comparable across D3 -> D4.")
issues.append(f"clock/backlog protocol: settle_steps / backlog_kind per dataset: " + str({k: (META[k]['settle'], META[k]['backlog_kind']) for k in ORDER_IDS}) + " (None = column absent; D4 uses the spin backlog, D5 the gemm_alloc backlog + 8 settle forwards).")
issues.append("D2 is job 21361; the diag twin 21362 is not kept as a CSV. D3, D4 sources are cluster-scratch copies (not in git).")
for i, t in enumerate(issues, 1): print(f"{i}. {t}")
open(EVO / "INCONSISTENCIES.txt", "w").write("\n".join(f"{i}. {t}" for i, t in enumerate(issues, 1)))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, HERE / "evolution.ipynb")
print("wrote", HERE / "evolution.ipynb")
