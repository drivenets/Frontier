#!/usr/bin/env python3
"""v2 of the presentation outputs, after the 2026-10-11 verification (story/review/FINDINGS_LOG.md).

What changed vs make_presentation_figures.py (originals are kept, v2 files are new):
  * STEP NAMING. Job 21356 is the X1 backlog-knob DIAGNOSTIC (4,800 rows, tokens >=1024, single-column legacy timer),
    not a story iteration; jobs 21500 and 21519 are real iterations that v1 omitted entirely. v1's claim "there is no
    it 4" was wrong. Every step is now named by job id.
  * s04 is rebuilt from the REAL per-repetition samples. v1's README said they were not retained; they are
    (53 per shape in job 21313, 106 for emb).
  * s05/s06 legends name jobs and disclose that the compared passes differ in clock state as well as queue state.
  * appendix adds jobs 21500 and 21519.
  * numbers_v2 adds quantiles, below-one counts, the per-TP spike ranges, and the per-transition step effects.

Read-only on all data; writes only *_v2.* into this folder. Run: python make_presentation_figures_v2.py
"""
import hashlib, json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib.colors import to_rgb

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
SCR = Path("/tmp/claude-1001/-home-dn-Frontier-qwen3-profiling/839301bf-3ea8-4d67-9dad-d3ac176a2add/scratchpad/cluster_data")

# job -> (plain label, path, in-git?)
JOBS = {
 "21313": ("original dense grid",            REPO/"data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv", True),
 "21334": ("rerun, different node",          REPO/"data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/2026-09-16_0629_linear_op_dense3327_rerun/linear_op.csv", True),
 "21356": ("X1 backlog-knob diagnostic",     SCR/"profiling_dip_x1/linear_op.csv", False),
 "21361": ("GC fix",                         REPO/"data/profiling_gcfix/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv", True),
 "21483": ("two-column + fused RoPE",        SCR/"profiling_dense_fixed/linear_op.csv", False),
 "21500": ("batch-flush fix",                SCR/"profiling_dense_fixed_batchflush/linear_op.csv", False),
 "21519": ("gemm_alloc backlog, settle 3",   SCR/"profiling_dense_fixed_workbacklog/linear_op.csv", False),
 "21539": ("FINAL - settle 8",               REPO/"data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv", True),
}
ITER_SEQ = ["21313", "21334", "21361", "21483", "21500", "21519", "21539"]   # dataset iterations (21356 is a diagnostic)
OPS = ["attn_pre_proj", "attn_post_proj", "attn_rope"]; TPS = [1, 2, 4, 8]; ZOOM = 2048; FULLX = 16384
EXPECTED_MD5 = {"21539": "3635e4d305ae5cf84884e275d3ff37ef", "21313": "97d6255414", "21519": "7016bd6260"}

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

SRC, D = {}, {}
for j, (lab, p, ingit) in JOBS.items():
    if not p.exists(): raise FileNotFoundError(f"job {j}: {p}")
    m = md5(p)
    if j in EXPECTED_MD5 and not m.startswith(EXPECTED_MD5[j]): raise AssertionError(f"job {j} md5 {m}")
    d = pd.read_csv(p, usecols=["num_tokens", "num_tensor_parallel_workers"] + [f"time_stats.{o}.median" for o in OPS])
    D[j] = d.rename(columns={"num_tensor_parallel_workers": "tp"})
    SRC[j] = dict(label=lab, path=str(p), md5=m, in_git=ingit)

def ser(j, op, tp, xm=None):
    g = D[j][D[j].tp == tp].sort_values("num_tokens").dropna(subset=[f"time_stats.{op}.median"])
    if xm: g = g[g.num_tokens <= xm]
    return g.set_index("num_tokens")[f"time_stats.{op}.median"]

def ratio(a, b, op, tp, xm=None):
    x, y = ser(a, op, tp, xm), ser(b, op, tp, xm); k = x.index.intersection(y.index)
    return (x[k] / y[k]).values

# ---------------- style (identical to v1) ----------------
FONT = 12
plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT,
    "ytick.labelsize": FONT, "legend.fontsize": FONT, "axes.grid": True, "grid.alpha": .3,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
ONE, TWO, DPI = (6.5, 3.8), (10, 3.8), 200
TP_COLOR = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}
def light(c, f=0.62): return tuple(1 - f * (1 - v) for v in to_rgb(c))
def fmt_x(ax, xmax):
    ax.set_xticks([0, 500, 1000, 1500, 2000] if xmax and xmax <= 2048 else [0, 4000, 8000, 12000, 16000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
def style_ax(ax, title, xmax=None, ylim=None):
    ax.set_title(title, loc="left"); ax.set_xlabel("num_tokens"); ax.set_ylabel("median time (ms)")
    if xmax is not None: ax.set_xlim(0, xmax)
    if ylim is not None: ax.set_ylim(*ylim)
    fmt_x(ax, xmax)
MAN = []
def save(fig, name, slide, op, tps, jobs, xr, yl, caveats, extra=None):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    e = dict(file=name, slide=slide, op=op, tps=tps,
             steps=[dict(job=j, label=SRC[j]["label"], path=SRC[j]["path"], md5=SRC[j]["md5"], in_git=SRC[j]["in_git"]) for j in jobs],
             token_range=xr, y_limits_ms=yl, caveats=caveats)
    if extra: e.update(extra)
    MAN.append(e)

# ---------------- s04 v2: real per-repetition samples ----------------
def fig_s04():
    p = JOBS["21313"][1]
    d = pd.read_csv(p, usecols=["num_tokens", "num_tensor_parallel_workers", "time_stats.attn_pre_proj.samples",
                                "time_stats.attn_pre_proj.warmup_count", "time_stats.attn_pre_proj.max"])
    d = d.rename(columns={"num_tensor_parallel_workers": "tp"})
    spike_rows = d[d["time_stats.attn_pre_proj.max"] > 50]
    row = d[(d.num_tokens == 1970) & (d.tp == 1)].iloc[0]
    a = np.asarray(json.loads(row["time_stats.attn_pre_proj.samples"]), float)
    wu = int(row["time_stats.attn_pre_proj.warmup_count"])
    fig, ax = plt.subplots(figsize=ONE); c = TP_COLOR[1]
    x = np.arange(len(a))
    ax.axvspan(-0.5, wu - 0.5, color="#e8e8e8", lw=0, label=f"{wu} warm-up forwards")
    ax.plot(x, a, color=c, lw=1.2, marker="o", ms=3)
    j = int(np.argmax(a))
    ax.scatter([j], [a[j]], s=60, color="#d03b3b", zorder=6,
               label=f"0-based timed run {j-wu} (= run {j-wu+1} of {len(a)-wu} counting from 1)")
    ax.set_yscale("log"); ax.set_xlabel("sample index in execution order (0-based, incl. warm-up)")
    ax.set_ylabel("time (ms, log)"); ax.set_title("attn_pre_proj, 1,970 tokens, TP 1, job 21313", loc="left")
    ax.legend(loc="center right", frameon=True, framealpha=.92, fontsize=FONT - 2)
    yl = list(ax.get_ylim())
    per_tp = {int(tp): [round(float(g["time_stats.attn_pre_proj.max"].min()), 1),
                        round(float(g["time_stats.attn_pre_proj.max"].max()), 1)]
              for tp, g in spike_rows.groupby("tp")}
    save(fig, "s04_gc_spike_per_repetition_v2.png", 4, "attn_pre_proj", [1], ["21313"], "single shape, 53 samples",
         {"log_y": yl},
         ["Rebuilt from the REAL per-repetition samples. v1's README wrongly said job 21313 kept no samples; it stores 53 per shape (106 for emb).",
          "DELIBERATE style exception: log y, because the spike is ~1,100x the row median.",
          f"Spike is at absolute 0-based sample {j} (the {j+1}th forward) = 0-based timed run {j-wu}. The docs' 'timed run 11' is 0-BASED.",
          f"Per-TP max over the 32 affected rows of job 21313: {per_tp}. Job 21334 (rerun) spans 146.4-287.7 ms."],
         extra=dict(spike_abs_index_0based=j, spike_timed_run_0based=j - wu, per_tp_max_ms_job21313=per_tp,
                    affected_rows=int(len(spike_rows))))
    return per_tp

# ---------------- s05 v2: host-bound vs GPU-bound, named by job ----------------
def decreasing_region(x, y, win=501):
    r = pd.Series(y).rolling(win, center=True, min_periods=win // 2).median().values
    i = int(np.nanargmin(r)); k = int(np.nanargmax(r[:i + 1]))
    return int(x[k]), int(x[i]), float(r[k]), float(r[i])

def fig_s05():
    op, tp = "attn_pre_proj", 8
    xh, yh = ser("21361", op, tp).index.values, ser("21361", op, tp).values
    xg, yg = ser("21356", op, tp).index.values, ser("21356", op, tp).values
    a, b, ra, rb = decreasing_region(xh, yh)
    fig, ax = plt.subplots(figsize=ONE); c = TP_COLOR[tp]
    ax.axvspan(a, b, color="#f2c6c6", alpha=.55, lw=0, label=f"host-bound curve decreasing ({a:,}-{b:,} tokens)")
    ax.plot(xh, yh, color=light(c), lw=1.5, label="job 21361 - GC fix, host-bound (legacy timer)")
    ax.plot(xg, yg, color=c, lw=2.6, label="job 21356 - X1 backlog-knob diagnostic, GPU-bound")
    yl = (0.0, float(max(yh.max(), yg.max())) * 1.08)
    style_ax(ax, "attn_pre_proj, TP 8", FULLX, yl)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=FONT - 2)
    save(fig, "s05_pre_proj_tp8_hostbound_vs_gpubound_v2.png", 5, op, [8], ["21361", "21356"], "full range (21356 from 1,024)",
         {"all": list(yl)},
         ["RELABELLED: v1 called these 'it 1' and 'it 2'. Job 21356 is the X1 DIAGNOSTIC (4,800 rows, tokens >=1,024, single-column legacy timer + FRONTIER_GPU_BACKLOG_MS=100), not a dataset iteration.",
          "The two passes differ in CLOCK STATE as well as queue state: a fixed 20M-cycle spin measures 1.93-2.08 M cycles/ms from idle vs 2.38-2.40 after a first spin (~2.0 vs ~2.4 GHz). The gap is therefore not purely the instrument.",
          f"Shaded region {a:,}-{b:,} tokens computed from the data (argmax->argmin of a 501-point rolling median; {ra:.4f} -> {rb:.4f} ms). The old slide shaded 4,000-6,000.",
          "The two jobs also differ in grid (3,327 vs 1,200 token counts)."],
         extra=dict(shaded_tokens=[a, b], rolling_median_ms=[ra, rb]))
    return dict(start=a, end=b, rm_start=ra, rm_end=rb)

# ---------------- s06 v2: RoPE, named by job ----------------
def fig_s06():
    op, tp = "attn_rope", 1
    f, u = ser("21356", op, tp, ZOOM), ser("21483", op, tp, ZOOM)
    fig, ax = plt.subplots(figsize=ONE); c = TP_COLOR[tp]
    ax.plot(f.index, f.values, color=light(c), lw=1.5, label="job 21356 - torch fallback (wrong kernel)")
    ax.plot(u.index, u.values, color=c, lw=2.6, label="job 21483 - fused vLLM kernel")
    yl = (0.0, float(max(f.max(), u.max())) * 1.08)
    style_ax(ax, f"attn_rope, TP 1, num_tokens <= {ZOOM:,}", ZOOM, yl)
    ax.legend(loc="upper left", frameon=False, fontsize=FONT - 2)
    ax.text(0.98, 0.52, "different computation,\nnot a timing fix", transform=ax.transAxes, ha="right", va="center",
            fontsize=FONT, style="italic")
    save(fig, "s06_rope_fallback_vs_fused_v2.png", 6, op, [1], ["21356", "21483"], f"1-{ZOOM:,}", {"all": list(yl)},
         ["RELABELLED by job. The fallback line covers only 1,024-2,048 tokens because job 21356's grid starts at 1,024.",
          "The two jobs differ in repetitions (50 vs 25), timer (single- vs two-column) and grid, so the ratio mixes the KERNEL change with the TIMING change. It cannot be decomposed from this data.",
          "Controls across the same job pair: attn_pre_proj and attn_post_proj agree within 1.2 % (see numbers_v2.md)."])

# ---------------- appendix v2: all iterations incl. 21500 / 21519 ----------------
def fig_app():
    op = "attn_pre_proj"
    seq = [("21313", "-"), ("21361", (0, (5, 2))), ("21483", (0, (3, 1, 1, 1))), ("21500", (0, (1, 1))), ("21519", (0, (6, 1, 1, 1, 1, 1)))]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharex=True)
    allr = []
    for ax, tp in zip(axes.ravel(), TPS):
        c = TP_COLOR[tp]
        for i, (j, ls) in enumerate(seq):
            r = ratio(j, "21539", op, tp); allr.append(r)
            x = ser(j, op, tp).index.intersection(ser("21539", op, tp).index)
            ax.plot(x, r, color=light(c, .30 + .16 * i), lw=1.5, ls=ls, label=f"job {j} - {SRC[j]['label']}")
        ax.axhline(1.0, color="k", lw=1); ax.set_yscale("log")
        ax.set_yticks([1, 1.5, 2, 3, 4, 6]); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); fmt_x(ax, None)
        ax.set_title(f"TP {tp}: median time / final (job 21539)", loc="left")
        ax.set_xlabel("num_tokens"); ax.set_ylabel("ratio to final (log)"); ax.set_xlim(0, FULLX)
    v = np.concatenate(allr); lo, hi = float(v.min() / 1.1), float(v.max() * 1.1)
    for ax in axes.ravel(): ax.set_ylim(lo, hi)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, handlelength=3.5, bbox_to_anchor=(0.5, -0.09), fontsize=FONT - 1)
    fig.tight_layout()
    save(fig, "appendix_iteration_ratio_to_final_v2.png", "appendix", op, TPS, [j for j, _ in seq] + ["21539"],
         f"1-{FULLX:,}", {"log_y_all_panels": [lo, hi]},
         ["ADDS jobs 21500 and 21519, which v1 omitted. Job 21519 is the step that first reaches ~1.000, so it answers the appendix's question.",
          "Job 21356 is NOT shown: it is a diagnostic on a different grid and timer, not an iteration.",
          "Ratios are per-token on the token counts each pair shares."])

PER_TP_SPIKE = fig_s04(); S05 = fig_s05(); fig_s06(); fig_app()

# ---------------- numbers_v2 ----------------
RANGES = [("<=2048", 2048), ("<=5000", 5000), ("full", None)]
def summ(a, b, op, tp, xm):
    r = ratio(a, b, op, tp, xm)
    return dict(n=int(len(r)), median=float(np.median(r)), min=float(r.min()), max=float(r.max()),
                p05=float(np.percentile(r, 5)), p25=float(np.percentile(r, 25)), p75=float(np.percentile(r, 75)),
                p95=float(np.percentile(r, 95)), below_one=int((r < 1).sum()),
                ratio_of_medians=float(np.median(ser(a, op, tp, xm).reindex(ser(b, op, tp, xm).index).dropna()) /
                                       np.median(ser(b, op, tp, xm))),
                max_abs_ratio_minus_1=float(np.max(np.abs(r - 1))))
N = {"definition": "Per shared num_tokens, ratio = job_A median / job_B median; then the median over tokens. min/max/percentiles are over the same per-token ratios. Labels are time_stats.<op>.median in ms.",
     "step_naming_correction": "Job 21356 is the X1 backlog-knob DIAGNOSTIC, not an iteration. The dataset iterations are 21313 -> 21334 -> 21361 -> 21483 -> 21500 -> 21519 -> 21539. v1's 'there is no it 4' was wrong (it 4 = job 21519).",
     "sources": SRC}
N["original_over_final"] = {op: {f"TP{tp}": {rn: summ("21313", "21539", op, tp, xm) for rn, xm in RANGES} for tp in TPS} for op in OPS}
N["attn_rope_caveat"] = "Jobs 21313/21334/21361/21356 timed a torch fallback that is not RoPE; 21483 onward use the fused kernel. The original/final rope ratio mixes a kernel change with the timing change."
N["step_over_final"] = {j: {op: {f"TP{tp}": summ(j, "21539", op, tp, None) for tp in TPS} for op in OPS} for j in ["21483", "21500", "21519"]}
trans = {}
for a, b, lbl in [("21483", "21500", "batch-flush fix"), ("21500", "21519", "gemm_alloc backlog + settle 3"), ("21519", "21539", "settle 8")]:
    trans[f"{a}->{b} ({lbl})"] = {op: dict(median=float(np.median(np.concatenate([ratio(b, a, op, tp) for tp in TPS]))),
        p95_abs=float(np.percentile(np.abs(np.concatenate([ratio(b, a, op, tp) for tp in TPS]) - 1), 95)),
        max_abs=float(np.max(np.abs(np.concatenate([ratio(b, a, op, tp) for tp in TPS]) - 1)))) for op in OPS}
N["transition_effect_on_medians"] = trans
N["controls_21356_over_21483"] = {op: {f"TP{tp}": float(np.median(ratio("21356", "21483", op, tp))) for tp in TPS} for op in OPS}
N["gc_spike_per_tp_max_ms_job21313"] = PER_TP_SPIKE
N["gc_spike_job21334_range_ms"] = [146.4, 287.7]
N["s05_decreasing_region"] = S05
claims = []
for op, lo, hi, txt in (("attn_pre_proj", 1.3, 2.9, "GEMM 1.3-2.9x"), ("attn_post_proj", 1.3, 2.9, "GEMM 1.3-2.9x"), ("attn_rope", 4, 12, "RoPE 4-12x")):
    for rn, _ in RANGES:
        vals = [N["original_over_final"][op][f"TP{tp}"][rn]["median"] for tp in TPS]
        a, b = min(vals), max(vals)
        claims.append(dict(claim=txt, op=op, range=rn, computed=[round(a, 2), round(b, 2)], consistent=bool(a >= lo * .95 and b <= hi * 1.05)))
N["deck_claims_check"] = claims
json.dump(N, open(OUT / "numbers_v2.json", "w"), indent=1)

f2 = lambda v: f"{v:.2f}"
md = ["# numbers_v2.md - recomputed 2026-10-11 after independent verification", "",
      "Supersedes `numbers.md`. See `../review/FINDINGS_LOG.md`. Originals kept for comparison.", "",
      "## Step naming (corrected)", "", N["step_naming_correction"], "",
      "| job | role | in git? | md5 |", "|---|---|---|---|"]
for j in ITER_SEQ + ["21356"]:
    md.append(f"| {j} | {SRC[j]['label']}{' *(diagnostic, not an iteration)*' if j=='21356' else ''} | {'yes' if SRC[j]['in_git'] else '**no - temp dir**'} | `{SRC[j]['md5'][:10]}` |")
md += ["", "## Definition", "", N["definition"], "", "## Original (21313) / final (21539)", ""]
for op in OPS:
    md += [f"### {op}", "", "| TP | range | n | median | min | p05 | p25 | p75 | p95 | max | ratio<1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tp in TPS:
        for rn, _ in RANGES:
            r = N["original_over_final"][op][f"TP{tp}"][rn]
            md.append(f"| {tp} | {rn} | {r['n']} | **{f2(r['median'])}** | {f2(r['min'])} | {f2(r['p05'])} | {f2(r['p25'])} | {f2(r['p75'])} | {f2(r['p95'])} | {f2(r['max'])} | {r['below_one']} |")
    md.append("")
md += ["> " + N["attn_rope_caveat"], "", "## Which later step actually moved the medians?", "",
       "later / earlier, pooled over TP, full grid:", "", "| transition | attn_pre_proj | attn_post_proj | attn_rope |", "|---|---|---|---|"]
for k, v in trans.items():
    md.append(f"| {k} | {v['attn_pre_proj']['median']:.4f} | {v['attn_post_proj']['median']:.4f} | {v['attn_rope']['median']:.4f} |")
md += ["", "Only the `gemm_alloc` backlog change moved the medians (1.5-2.4 % down, all three ops). The batch-flush fix and settle-8 left them within 0.2 %.", "",
       "## Controls across the RoPE change (job 21356 / job 21483, median per TP)", "", "| op | TP1 | TP2 | TP4 | TP8 |", "|---|---|---|---|---|"]
for op in OPS:
    c = N["controls_21356_over_21483"][op]
    md.append(f"| {op} | {c['TP1']:.3f} | {c['TP2']:.3f} | {c['TP4']:.3f} | {c['TP8']:.3f} |")
md += ["", "The GEMM scopes agree within 1.2 % across that protocol change while attn_rope moves 2.6-6.1x. That is what makes them controls - a measured agreement, not an assumption from code identity.", "",
       "## Deck claims vs data", "", "| claim | op | range | computed median ratio across TPs | consistent (+/-5 %) |", "|---|---|---|---|---|"]
for c in claims:
    md.append(f"| {c['claim']} | {c['op']} | {c['range']} | {c['computed'][0]}-{c['computed'][1]} | {'yes' if c['consistent'] else '**NO**'} |")
md += ["", "## GC spike (corrected)", "",
       "| job | TP1 | TP2 | TP4 | TP8 | overall |", "|---|---|---|---|---|---|",
       f"| 21313 | {PER_TP_SPIKE[1][0]}-{PER_TP_SPIKE[1][1]} | {PER_TP_SPIKE[2][0]}-{PER_TP_SPIKE[2][1]} | {PER_TP_SPIKE[4][0]}-{PER_TP_SPIKE[4][1]} | {PER_TP_SPIKE[8][0]}-{PER_TP_SPIKE[8][1]} | **150.4-268.6 ms** |",
       "| 21334 | 149.7-262.9 | 151.6-266.0 | 146.4-287.7 | 148.5-244.5 | **146.4-287.7 ms** |", "",
       "v1's `numbers.md` reported 150-243 ms, which is TP1 of job 21313 only. The page's Chapter 2 '150-263 ms' for job 21334 is wrong.", "",
       f"- s05 shaded region: {S05['start']:,}-{S05['end']:,} tokens (501-point rolling median, {S05['rm_start']:.4f} -> {S05['rm_end']:.4f} ms)."]
(OUT / "numbers_v2.md").write_text("\n".join(md) + "\n")
json.dump(dict(generated="make_presentation_figures_v2.py", note="v2 after the 2026-10-11 verification; see ../review/FINDINGS_LOG.md",
               iteration_sequence=ITER_SEQ, diagnostic_not_an_iteration="21356", sources=SRC, figures=MAN),
          open(OUT / "manifest_v2.json", "w"), indent=1)
print("v2 figures:", [m["file"] for m in MAN])
print("claims inconsistent:", [(c["op"], c["range"], c["computed"]) for c in claims if not c["consistent"]])
