"""Charts for docs/profiling_walkthrough.md. Numbers come from docs/profiling_project_summary.md
(sections 4 and 9). Run: /usr/bin/python3 make_charts.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, OCHRE, BLUE, GREY = "#10141B", "#7A5200", "#0A3FC4", "#8A9099"
plt.rcParams.update({"font.size": 20, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

# 1. Dataset A: test MAPE vs noise floor (RESULTS_baseline_v2.md, random split, test tier)
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
names = ["interp", "rf", "xgb_exact"]
vals = [1.37, 1.25, 1.27]
bars = ax.bar(names, vals, color=[GREY, BLUE, BLUE], width=0.55)
ax.axhline(1.19, color=OCHRE, ls="--", lw=3)
ax.text(2.4, 1.19, "noise floor\n1.19%", color=OCHRE, ha="left", va="center", fontweight="bold")
ax.set_xlim(-0.5, 3.3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}%", ha="center", fontweight="bold", color=INK)
ax.set_ylim(1.0, 1.5)
ax.set_ylabel("Mean MAPE, 15 regressors (%)")
ax.set_title("Test tier, random split", loc="left", fontsize=18, color=INK)
fig.tight_layout(); fig.savefig("dataset_a_mape.png"); plt.close(fig)

# 2. Old data vs fixed data: ratio of medians (12_dip_investigation_summary.md section 5)
fig, ax = plt.subplots(figsize=(9, 4.4), dpi=160)
rows = [("GEMM operators", 1.3, 2.9), ("attn_rope", 4.3, 12.0)]
for i, (n, lo, hi) in enumerate(rows):
    ax.barh(i, hi - lo, left=lo, color=OCHRE, height=0.42)
    ax.text(hi + 0.25, i, f"{lo:g} to {hi:g}x", va="center", fontweight="bold", color=INK)
ax.axvline(1, color=BLUE, lw=3)
ax.text(1.1, 1.52, "fixed data = 1x", color=BLUE, fontweight="bold", va="center")
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 15); ax.set_ylim(-0.6, 1.8)
ax.set_xlabel("Old median / fixed median (median over the grid)")
fig.tight_layout(); fig.savefig("old_vs_fixed.png"); plt.close(fig)
