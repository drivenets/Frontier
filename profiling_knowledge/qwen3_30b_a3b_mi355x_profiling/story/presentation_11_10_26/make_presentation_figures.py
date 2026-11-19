#!/usr/bin/env python3
"""Regenerate every presentation figure + numbers.md/json + manifest.json from scratch.

Read-only on all data and on evolution/; writes only into this folder.
Data loading mirrors evolution/evolution.ipynb: the DATASETS/TP_COLOR config cell is exec'd from the notebook itself, a source CSV
is read when it exists (same columns, same renaming), otherwise evolution/cache/<id>.csv is used and the fallback is recorded.

Run:  python make_presentation_figures.py        (from anywhere)
"""
import hashlib, json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import to_rgb

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
EVO = OUT.parent / "evolution"

# ------------------------------------------------------------------ config from the notebook (single source of truth)
nb = json.load(open(EVO / "evolution.ipynb"))
cfg_src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "CONFIG (edit here)" in "".join(c["source"]))
_cwd = os.getcwd(); os.chdir(EVO)                      # the notebook's find_repo() uses cwd at definition time
CFG = {}; exec(cfg_src, CFG); os.chdir(_cwd)
DATASETS = {d["id"]: d for d in CFG["DATASETS"]}
TP_COLOR, TP_COL, EXPECTED_MD5 = CFG["TP_COLOR"], CFG["TP_COL"], CFG["EXPECTED_MD5"]
TPS = [1, 2, 4, 8]

# Step names used in the deck -> notebook dataset ids. (The brief says "original, it 1..it 4, final" but only four non-final full
# steps exist after the original; see README.md "Step mapping".)
STEPS = {
    "original": dict(ds="D1", plain="original",              job="21313", note="first re-collection, host-bound, torch-fallback RoPE"),
    "it1":      dict(ds="D2", plain="GC removed",            job="21361", note="host-bound, torch-fallback RoPE"),
    "it2":      dict(ds="D3", plain="GPU-bound (host fix)",  job="21356", note="GPU-bound, torch-fallback RoPE, tokens >= 1,024 only"),
    "it3":      dict(ds="D4", plain="RoPE fix",              job="21483", note="GPU-bound, fused RoPE"),
    "final":    dict(ds="D5", plain="final",                 job="21539", note="GPU-bound, fused RoPE, run-position fixes"),
}
OPS3 = ["attn_pre_proj", "attn_post_proj", "attn_rope"]
STATS = ["min", "max", "median"]

# ------------------------------------------------------------------ data loading
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

DATA, SRC = {}, {}
for sid, st in STEPS.items():
    did = st["ds"]; src = Path(DATASETS[did]["path"]); cache = EVO / "cache" / f"{did}.csv"; cmeta = json.load(open(EVO / "cache" / f"{did}.meta.json"))
    if src.exists():
        need = ["num_tokens", TP_COL] + [f"time_stats.{op}.{s}" for op in OPS3 for s in STATS]
        df = pd.read_csv(src, usecols=need).rename(columns={TP_COL: "tp"})
        df.columns = [c.replace("time_stats.", "").replace(".", "__") for c in df.columns]
        m5 = md5(src)
        SRC[sid] = dict(dataset=did, source="source CSV", path=str(src), md5=m5, cache_md5=cmeta["md5"], source_matches_cache=(m5 == cmeta["md5"]), job=st["job"])
    else:
        df = pd.read_csv(cache); m5 = cmeta["md5"]
        SRC[sid] = dict(dataset=did, source="CACHE FALLBACK (source CSV missing)", path=str(cache), original_source_path=cmeta["source"], md5=m5,
                        cache_md5=m5, source_matches_cache=None, job=st["job"])
    if did in EXPECTED_MD5 and m5 != EXPECTED_MD5[did]: raise AssertionError(f"{did} md5 {m5} != {EXPECTED_MD5[did]}")
    DATA[sid] = df
FALLBACKS = [s for s, v in SRC.items() if v["source"].startswith("CACHE")]
MISMATCH = [s for s, v in SRC.items() if v["source_matches_cache"] is False]

def ser(sid, op, tp, stat="median", xmax=None):
    d = DATA[sid]; d = d[d.tp == tp].sort_values("num_tokens"); d = d[d[f"{op}__{stat}"].notna()]
    if xmax is not None: d = d[d.num_tokens <= xmax]
    return d.num_tokens.values, d[f"{op}__{stat}"].values

# ------------------------------------------------------------------ style
FONT = 12
plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
                     "legend.fontsize": FONT, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
ONE, TWO, DPI = (6.5, 3.8), (10, 3.8), 200
ZOOM = 2048
LW_OLD, LW_FINAL = 1.5, 2.6
def light(c, f=0.62): return tuple(1 - f * (1 - v) for v in to_rgb(c))        # blend toward white
def tp_lab(tp): return f"TP {tp}"

MANIFEST = []
def save(fig, name, slide, op, tps, steps, xrange, ylims, caveats, extra=None):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    ent = dict(file=name, slide=slide, op=op, tps=tps, steps=[dict(step=s, plain=STEPS[s]["plain"], dataset=STEPS[s]["ds"], **{k: SRC[s][k] for k in ("source", "path", "md5")}, job=STEPS[s]["job"]) for s in steps],
               token_range=xrange, y_limits_ms=ylims, caveats=caveats)
    if extra: ent.update(extra)
    MANIFEST.append(ent)

def ylim_for(curves, pad=1.08):
    return (0.0, float(max(np.max(y) for y in curves if len(y))) * pad)

def fmt_x(ax, xmax):
    # few, comma-formatted ticks so labels never collide on 5-inch panels
    ax.set_xticks([0, 500, 1000, 1500, 2000] if xmax and xmax <= 2048 else [0, 4000, 8000, 12000, 16000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

def style_ax(ax, title, xmax=None, ylim=None):
    ax.set_title(title, loc="left"); ax.set_xlabel("num_tokens"); ax.set_ylabel("median time (ms)")
    if xmax is not None: ax.set_xlim(0, xmax)
    if ylim is not None: ax.set_ylim(*ylim)
    fmt_x(ax, xmax)

FULLX = 16384

# ------------------------------------------------------------------ figures
def fig_all_tp(sid, name, slide, loc_full="upper left"):
    """Two panels (full / <=2048), TP 1/2/4/8, one step. Y-limits per panel are shared across s02/s03 (see PANEL_Y)."""
    op = "attn_pre_proj"
    fig, axes = plt.subplots(1, 2, figsize=TWO)
    for ax, (ttl, xm) in zip(axes, (("full range", None), (f"num_tokens ≤ {ZOOM:,}", ZOOM))):
        for tp in TPS:
            x, y = ser(sid, op, tp, xmax=xm); ax.plot(x, y, color=TP_COLOR[tp], lw=LW_FINAL if sid == "final" else LW_OLD, label=tp_lab(tp))
        style_ax(ax, ttl, xm if xm else FULLX, PANEL_Y[ttl])
    axes[0].legend(loc=loc_full, ncol=1, frameon=False); fig.tight_layout(w_pad=2.5)
    save(fig, name, slide, op, TPS, [sid], f"full (1-{FULLX:,}) and 1-{ZOOM:,}", {k: list(v) for k, v in PANEL_Y.items()},
         ["y-limits are shared with the other all-TP figure (s02/s03) panel-by-panel, but differ between the full and zoom panels (otherwise the zoom is unreadable)."])

# shared y-limits for s02 / s03 panels (original vs final, same op)
PANEL_Y = {}
_full = [ser(s, "attn_pre_proj", tp)[1] for s in ("original", "final") for tp in TPS]
_zoom = [ser(s, "attn_pre_proj", tp, xmax=ZOOM)[1] for s in ("original", "final") for tp in TPS]
PANEL_Y["full range"] = ylim_for(_full); PANEL_Y[f"num_tokens ≤ {ZOOM:,}"] = ylim_for(_zoom)

fig_all_tp("original", "s02_pre_proj_original.png", 2)
fig_all_tp("final", "s03_pre_proj_final_all_tp.png", 3)

# ---- s04: GC spike (min-max band over 50 reps), TP 1, from the cached min/max columns of the original step
def fig_s04():
    op, tp, sid = "attn_pre_proj", 1, "original"
    x, lo = ser(sid, op, tp, "min"); _, hi = ser(sid, op, tp, "max"); _, med = ser(sid, op, tp)
    fig, ax = plt.subplots(figsize=ONE)
    c = TP_COLOR[tp]
    ax.fill_between(x, lo, hi, color=light(c, .7), lw=0, label="min–max over 50 timed runs")
    ax.plot(x, med, color=c, lw=LW_OLD, label="median")
    bad = hi > 50
    ax.scatter(x[bad], hi[bad], s=16, color="#d03b3b", zorder=5, label=f"max > 50 ms ({int(bad.sum())} token counts)")
    ax.set_yscale("log"); style_ax(ax, "TP 1, original data", FULLX); ax.set_ylabel("time (ms, log)")
    ax.legend(loc="center right", frameon=True, framealpha=.9)
    ylim = ax.get_ylim()
    save(fig, "s04_gc_spike_per_repetition.png", 4, op, [1], [sid], f"1-{FULLX:,}", {"log_y": [float(ylim[0]), float(ylim[1])]},
         ["DELIBERATE EXCEPTION to 'linear y from 0, medians only': log y, because the spike (150-290 ms) is ~1,000x the median (~0.1 ms) and a single sample.",
          "Built from the min/max/median columns (per-shape aggregates over 50 reps); per-repetition samples are not stored for this step.",
          f"Spike token counts at TP 1: {int(x[bad].min())}-{int(x[bad].max())}."],
         extra=dict(spike_tokens=[int(v) for v in x[bad]], spike_max_ms=[round(float(v), 1) for v in hi[bad]]))
    return dict(tokens=[int(v) for v in x[bad]], max_ms=[float(v) for v in hi[bad]], median_ms=[float(m) for m in med[bad]])
S04 = fig_s04()

# ---- s05: host-bound vs GPU-bound, TP 8 : find where the host-bound curve decreases (detection only, plotted curve is raw)
def decreasing_region(x, y, win=501):
    r = pd.Series(y).rolling(win, center=True, min_periods=win // 2).median().values
    imin = int(np.nanargmin(r)); imax = int(np.nanargmax(r[:imin + 1]))
    return int(x[imax]), int(x[imin]), float(r[imax]), float(r[imin])

def fig_s05():
    op, tp = "attn_pre_proj", 8
    xh, yh = ser("it1", op, tp); xg, yg = ser("it2", op, tp)
    a, b, ra, rb = decreasing_region(xh, yh)
    fig, ax = plt.subplots(figsize=ONE); c = TP_COLOR[tp]
    ax.axvspan(a, b, color="#f2c6c6", alpha=.55, lw=0, label=f"host-bound curve decreasing ({a:,}–{b:,} tokens)")
    ax.plot(xh, yh, color=light(c), lw=LW_OLD, label="it 1: GC removed (host-bound, CPU launch time)")
    ax.plot(xg, yg, color=c, lw=LW_FINAL, label="it 2: GPU-bound")
    ylim = ylim_for([yh, yg]); style_ax(ax, "TP 8", FULLX, ylim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    save(fig, "s05_pre_proj_tp8_hostbound_vs_gpubound.png", 5, op, [8], ["it1", "it2"], f"full (it 2 has data from 1,024 only)", {"all": list(ylim)},
         [f"Shaded region = {a:,}-{b:,} tokens, computed from the data: argmax -> argmin of a 501-point rolling median (~15% of the grid) of the host-bound curve (used for detection only; the plotted curves are the raw medians). Rolling-median values at the ends: {ra:.4f} -> {rb:.4f} ms.",
          "it 2 (GPU-bound) has no data below 1,024 tokens, so its line starts at 1,024.",
          "The old slide shaded 4,000-6,000."], extra=dict(shaded_tokens=[a, b], rolling_median_ms=[ra, rb]))
    return dict(start=a, end=b, rm_start=ra, rm_end=rb)
S05 = fig_s05()

# ---- s06: RoPE torch fallback vs fused
def fig_s06():
    op, tp = "attn_rope", 1
    xf, yf = ser("it2", op, tp, xmax=ZOOM); xu, yu = ser("it3", op, tp, xmax=ZOOM)
    fig, ax = plt.subplots(figsize=ONE); c = TP_COLOR[tp]
    ax.plot(xf, yf, color=light(c), lw=LW_OLD, label="torch fallback (wrong kernel)")
    ax.plot(xu, yu, color=c, lw=LW_FINAL, label="fused kernel")
    ylim = ylim_for([yf, yu]); style_ax(ax, f"attn_rope, TP 1, num_tokens ≤ {ZOOM:,}", ZOOM, ylim)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.55, "different computation,\nnot a timing fix", transform=ax.transAxes, ha="right", va="center", fontsize=FONT, style="italic")
    save(fig, "s06_rope_fallback_vs_fused.png", 6, op, [1], ["it2", "it3"], f"1-{ZOOM:,}", {"all": list(ylim)},
         ["'torch fallback' = GPU-bound dataset D3 (job 21356), 'fused' = D4 (job 21483). D3 only has tokens >= 1,024, so the fallback line covers 1,024-2,048 only.",
          "The host-bound fallback data (D1/D2) cover the full 1-2,048 range but measure host launch time, not GPU time; they are deliberately not used here."])
fig_s06()

# ---- s07: before vs after
def fig_s07(xm, name):
    op = "attn_pre_proj"
    curves = {(tp, s): ser(s, op, tp, xmax=xm) for tp in (1, 8) for s in ("original", "final")}
    ylim = ylim_for([v[1] for v in curves.values()])
    fig, axes = plt.subplots(1, 2, figsize=TWO)
    for ax, tp in zip(axes, (1, 8)):
        c = TP_COLOR[tp]
        ax.plot(*curves[(tp, "original")], color=light(c), lw=LW_OLD, label="original")
        ax.plot(*curves[(tp, "final")], color=c, lw=LW_FINAL, label="final")
        style_ax(ax, tp_lab(tp), xm if xm else FULLX, ylim)
        ax.legend(loc=("lower right" if tp == 1 else ("upper right" if xm else "upper left")), frameon=False)
    fig.tight_layout(w_pad=2.5)
    save(fig, name, 7, op, [1, 8], ["original", "final"], f"1-{xm:,}" if xm else f"full (1-{FULLX:,})", {"both_panels": list(ylim)},
         ["Both panels share the same y-limits (as requested), so TP 8 looks compressed on the full-range version."] if not xm else
         ["Both panels share the same y-limits (as requested); TP 1 sets the scale."])
fig_s07(ZOOM, "s07_pre_proj_before_after.png")
fig_s07(None, "s07_pre_proj_before_after_full.png")

# ---- appendix: ratio to final
def fig_app():
    op = "attn_pre_proj"
    seq = [("original", "-"), ("it1", (0, (5, 2))), ("it2", (0, (3, 1, 1, 1))), ("it3", (0, (1, 1)))]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharex=True)
    allr = []
    for ax, tp in zip(axes.ravel(), TPS):
        c = TP_COLOR[tp]; xf, yf = ser("final", op, tp)
        for i, (s, ls) in enumerate(seq):
            x, y = ser(s, op, tp); m = pd.Series(y, index=x).rename("a").to_frame().join(pd.Series(yf, index=xf).rename("f"), how="inner")
            r = (m.a / m.f); allr.append(r.values)
            ax.plot(r.index, r.values, color=light(c, .35 + .2 * i), lw=1.5, ls=ls,
                    label=f"{STEPS[s]['plain']} (job {STEPS[s]['job']})")
        ax.axhline(1.0, color="k", lw=1)
        ax.set_yscale("log"); ax.set_yticks([1, 1.5, 2, 3, 4, 6]); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); fmt_x(ax, None)
        ax.set_title(f"{tp_lab(tp)}: median time / final (job 21539)", loc="left"); ax.set_xlabel("num_tokens"); ax.set_xlim(0, FULLX)
        ax.set_ylabel("ratio to final (log)")
    allv = np.concatenate(allr); lo, hi = float(allv.min() / 1.1), float(allv.max() * 1.1)
    for ax in axes.ravel(): ax.set_ylim(lo, hi)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, handlelength=3.5, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "appendix_iteration_ratio_to_final.png", "appendix", op, TPS, ["original", "it1", "it2", "it3", "final"], f"1-{FULLX:,} (it 2: from 1,024)",
         {"log_y_all_panels": [lo, hi]},
         ["Diagnostic, not for the main deck. Ratios are per-token (step median / final median) on the token counts both datasets share.",
          "The brief asks for 'it 4'; no such full-grid step exists in the notebook (see README 'Step mapping'), so the four non-final steps are original, it 1, it 2, it 3."])
fig_app()

# ------------------------------------------------------------------ numbers
RANGES = [("<=2048", 2048), ("<=5000", 5000), ("full", None)]
def ratios(num, den, op, tp, xm):
    xa, ya = ser(num, op, tp, xmax=xm); xb, yb = ser(den, op, tp, xmax=xm)
    m = pd.DataFrame({"a": ya}, index=xa).join(pd.DataFrame({"b": yb}, index=xb), how="inner")
    r = (m.a / m.b).values
    return dict(n_tokens=int(len(r)), median_of_per_token_ratio=float(np.median(r)), min=float(r.min()), max=float(r.max()),
                ratio_of_medians=float(m.a.median() / m.b.median()), max_abs_ratio_minus_1=float(np.max(np.abs(r - 1))))
NUM = {"definition": "Primary statistic: for every num_tokens present in BOTH datasets compute step_median/final_median, then take the median over tokens "
                     "('median_of_per_token_ratio'); min/max are over the same per-token ratios. 'ratio_of_medians' = median(step medians)/median(final medians) over the same tokens (alternative). "
                     "'max_abs_ratio_minus_1' = max |ratio-1|. Medians are time_stats.<op>.median. Ranges are on num_tokens.",
       "step_to_dataset": {k: dict(dataset=v["ds"], job=v["job"], meaning=v["plain"]) for k, v in STEPS.items()}}
for op in ("attn_pre_proj", "attn_post_proj"):
    NUM[f"{op}__original_over_final"] = {f"TP{tp}": {rn: ratios("original", "final", op, tp, xm) for rn, xm in RANGES} for tp in TPS}
NUM["attn_rope__over_final"] = {"CAVEAT": "original, it1 and it2 timed the forced torch-fallback RoPE (a different computation); final uses the fused kernel. These ratios mix a kernel change with timing changes. it2 only has tokens >= 1,024, so its '<=2048' range covers 1,024-2,048.",
                                **{s: {f"TP{tp}": {rn: ratios(s, "final", "attn_rope", tp, xm) for rn, xm in RANGES} for tp in TPS} for s in ("original", "it1", "it2", "it3")}}
NUM["attn_pre_proj__steps_over_final"] = {s: {f"TP{tp}": {rn: ratios(s, "final", "attn_pre_proj", tp, xm) for rn, xm in RANGES} for tp in TPS} for s in ("it1", "it2", "it3")}
NUM["attn_pre_proj__steps_over_final"]["NOTE"] = "No 'it 4' exists in the notebook datasets; it 3 = RoPE fix (D4) is the last step before final. it 2 only has tokens >= 1,024."
NUM["s04_spike"] = S04; NUM["s05_decreasing_region"] = S05

# claims check
def band(op, rn):
    v = [NUM[f"{op}__original_over_final"][f"TP{tp}"][rn]["median_of_per_token_ratio"] for tp in TPS] if op != "attn_rope" else [NUM["attn_rope__over_final"]["original"][f"TP{tp}"][rn]["median_of_per_token_ratio"] for tp in TPS]
    return min(v), max(v)
claims = []
for op, lo, hi, txt in (("attn_pre_proj", 1.3, 2.9, "GEMM 1.3-2.9x"), ("attn_post_proj", 1.3, 2.9, "GEMM 1.3-2.9x"), ("attn_rope", 4, 12, "RoPE 4-12x")):
    for rn, _ in RANGES:
        a, b = band(op, rn); ok = (a >= lo * 0.95 and b <= hi * 1.05)
        claims.append(dict(claim=txt, op=op, range=rn, computed_median_ratio_over_TPs=[round(a, 2), round(b, 2)], consistent=bool(ok)))
NUM["deck_claims_check"] = claims
json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1)

def f2(v): return f"{v:.2f}"
md = ["# numbers.md — verification of the deck's overstatement claims", "", "Generated by `make_presentation_figures.py`. Same content as `numbers.json`.", "",
      "## Definitions", "", NUM["definition"], "",
      "Step → dataset: " + "; ".join(f"**{k}** = {v['ds']} ({v['plain']}, job {v['job']})" for k, v in STEPS.items()), ""]
if FALLBACKS: md += [f"**Cache fallback used for:** {', '.join(FALLBACKS)} (source CSV missing; numbers come from evolution/cache/).", ""]
if MISMATCH: md += [f"**WARNING: source CSV md5 differs from the cache for {', '.join(MISMATCH)}.**", ""]
def table(title, block, extra_cols=True):
    out = [f"### {title}", "", "| TP | range | n tokens | median per-token ratio | min | max | ratio of medians |", "|---|---|---|---|---|---|---|"]
    for tp in TPS:
        for rn, _ in RANGES:
            r = block[f"TP{tp}"][rn]; out.append(f"| {tp} | {rn} | {r['n_tokens']} | {f2(r['median_of_per_token_ratio'])} | {f2(r['min'])} | {f2(r['max'])} | {f2(r['ratio_of_medians'])} |")
    return out + [""]
md += ["## Original / final", ""]
md += table("attn_pre_proj — original / final", NUM["attn_pre_proj__original_over_final"])
md += table("attn_post_proj — original / final", NUM["attn_post_proj__original_over_final"])
md += ["## attn_rope", "", "> **Caveat:** original, it 1 and it 2 timed the forced torch fallback, which is a different computation from the fused kernel in the final step. These ratios therefore mix a kernel change with timing changes. it 2 has tokens >= 1,024 only.", ""]
for s in ("original", "it1", "it2", "it3"): md += table(f"attn_rope — {STEPS[s]['plain']} ({s}) / final", NUM["attn_rope__over_final"][s])
md += ["## Did the later iterations change anything? (attn_pre_proj, step / final)", ""]
md += ["| step | TP | range | n | median ratio | max abs(ratio-1) |", "|---|---|---|---|---|---|"]
for s in ("it1", "it2", "it3"):
    for tp in TPS:
        for rn, _ in RANGES:
            r = NUM["attn_pre_proj__steps_over_final"][s][f"TP{tp}"][rn]; md.append(f"| {s} ({STEPS[s]['plain']}) | {tp} | {rn} | {r['n_tokens']} | {f2(r['median_of_per_token_ratio'])} | {f2(r['max_abs_ratio_minus_1'])} |")
md += ["", "No 'it 4' exists; it 3 (RoPE fix) is the last step before final.", "", "## Deck claims vs data", "", "| claim | op | range | computed median ratio across TPs (min–max) | consistent with claim (±5%) |", "|---|---|---|---|---|"]
for c in claims: md.append(f"| {c['claim']} | {c['op']} | {c['range']} | {c['computed_median_ratio_over_TPs'][0]}–{c['computed_median_ratio_over_TPs'][1]} | {'yes' if c['consistent'] else '**NO**'} |")
md += ["", "## Other facts used in the figures", "", f"- s04 spike token counts (TP 1): {S04['tokens']}; max {min(S04['max_ms']):.0f}–{max(S04['max_ms']):.0f} ms vs median ≈ {np.median(S04['median_ms']):.3f} ms.",
       f"- s05 host-bound TP 8 curve decreases between {S05['start']:,} and {S05['end']:,} tokens (501-point rolling median, {S05['rm_start']:.4f} → {S05['rm_end']:.4f} ms). The old slide shaded 4,000–6,000."]
(OUT / "numbers.md").write_text("\n".join(md) + "\n")

# ------------------------------------------------------------------ manifest
json.dump(dict(generated_by="make_presentation_figures.py", step_mapping={k: dict(dataset=v["ds"], job=v["job"], meaning=v["plain"], note=v["note"]) for k, v in STEPS.items()},
               sources=SRC, cache_fallbacks=FALLBACKS, figures=MANIFEST), open(OUT / "manifest.json", "w"), indent=1)
print("wrote", len(MANIFEST), "figures; cache fallbacks:", FALLBACKS or "none"); print("claims inconsistent:", [(c["op"], c["range"], c["computed_median_ratio_over_TPs"]) for c in claims if not c["consistent"]])
