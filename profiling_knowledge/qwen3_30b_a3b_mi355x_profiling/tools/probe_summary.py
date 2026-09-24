import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1])
print(f"rows={len(df)} tokens={sorted(df.num_tokens.unique())} tp={sorted(df.num_tensor_parallel_workers.unique())}")
print("| probe | median MHz | min–max |\n|---|---|---|")
for c, lab in [("sclk_mhz_legacy_start","legacy pass, start"),("sclk_mhz_legacy_end","legacy pass, end"),("sclk_mhz_backlog_start","GPU-bound pass, start"),("sclk_mhz_backlog_end","GPU-bound pass, end")]:
    print(f"| {lab} | {df[c].median():.0f} | {df[c].min():.0f}–{df[c].max():.0f} |")
lo = (df.sclk_mhz_legacy_start < 1500).sum(); print(f"legacy start < 1500 MHz on {lo}/{len(df)} rows")
tpcol = 'num_tensor_parallel_workers'
df["enq_share_backlog"] = df.host_enqueue_per_forward_ms_backlog / df.host_wall_per_forward_ms_backlog
df["enq_share_legacy"] = df.host_enqueue_per_forward_ms / df.host_wall_per_forward_ms
if "time_stats.forward_gpu_span.median" in df:
    ops = [c[len("time_stats."):-len(".median")] for c in df.columns if c.startswith("time_stats.") and c.endswith(".median") and "forward_gpu_span" not in c]
    df["closure"] = df[[f"time_stats.{o}.median" for o in ops]].sum(axis=1) / df["time_stats.forward_gpu_span.median"]
g = df.groupby("num_tokens")
print("\n| tokens | enqueue/wall GPU-bound (min–max over TP) | enqueue/wall legacy | closure Σops/span (min–max) |\n|---|---|---|---|")
for t, s in g:
    cl = f"{s.closure.min():.2f}–{s.closure.max():.2f}" if "closure" in s else "n/a"
    print(f"| {t} | {s.enq_share_backlog.min():.2f}–{s.enq_share_backlog.max():.2f} | {s.enq_share_legacy.min():.2f}–{s.enq_share_legacy.max():.2f} | {cl} |")
print("\nper-row legacy start MHz by tokens/tp:")
print(df.pivot_table(index="num_tokens", columns=tpcol, values="sclk_mhz_legacy_start", aggfunc="first").round(0).to_string())
