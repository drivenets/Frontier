"""Dense-grid shape check: per op x TP, flag non-monotone drops (a drop > 15 % between consecutive token counts that is not recovered
within the next 8 grid points) and print the worst ones, plus the new-vs-old ratio. Monotonicity is NOT a pass criterion here; it is the
symptom the dip investigation started from, so we simply list where it still happens and how big it is."""
import sys, pandas as pd, numpy as np
new = pd.read_csv(sys.argv[1]); old = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else None
ops = [c[len("time_stats."):-len(".median")] for c in new.columns if c.startswith("time_stats.") and c.endswith(".median") and "forward_gpu_span" not in c]
print(f"rows={len(new)} tokens={new.num_tokens.nunique()} TP={sorted(new.num_tensor_parallel_workers.unique())} active_steps={new.active_steps.iloc[0]} rope_impl={new.attn_rope_impl.value_counts().to_dict()}")
print(f"SCLK backlog_start median {new.sclk_mhz_backlog_start.median():.0f} [{new.sclk_mhz_backlog_start.min():.0f}-{new.sclk_mhz_backlog_start.max():.0f}]")
for op in ops:
    for tp, g in new.groupby("num_tensor_parallel_workers"):
        s = g.sort_values("num_tokens").set_index("num_tokens")[f"time_stats.{op}.median"].dropna()
        if len(s) < 10: continue
        v = s.values; t = s.index.values; dips = []
        for i in range(1, len(v)):
            if v[i] < 0.85 * v[i-1] and v[i-1] > 0.02:  # ignore the sub-20 us noise floor
                rec = v[i:i+8].max()
                if rec < 0.95 * v[i-1]: dips.append((int(t[i-1]), int(t[i]), v[i-1], v[i]))
        line = f"{op:26s} TP{tp}: {len(dips)} unrecovered drops"
        if dips: line += " | worst " + ", ".join(f"{a}->{b} tok {x:.4f}->{y:.4f} ms" for a, b, x, y in sorted(dips, key=lambda d: d[3]/d[2])[:3])
        print(line)
if old is not None:
    m = new.merge(old, on=["num_tokens", "num_tensor_parallel_workers"], suffixes=("", "_old"))
    print("\nnew/old GPU-bound median ratio by TP (median over tokens; <1 = old was inflated):")
    for op in ("attn_pre_proj", "attn_post_proj", "attn_rope"):
        r = (m[f"time_stats.{op}.median"] / m[f"time_stats.{op}.median_old"]).groupby(m.num_tensor_parallel_workers).median().round(2)
        print(f"  {op:16s} {r.to_dict()}")
