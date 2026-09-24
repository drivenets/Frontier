"""E8 (200 timed reps): gen-0 landing forwards, every gen-2 event with its timed-run index, and attn_pre_proj spikes."""
import json, sys, glob, collections, statistics as st
root = sys.argv[1]
files = sorted(glob.glob(root + "/worker_*.jsonl"))
pos = collections.Counter(); g2 = []; spikes = []; nfw = None
for f in files:
    for l in open(f):
        r = json.loads(l)
        if r.get("kind") != "task": continue
        nfw = len(r["forwards_ms"])
        for ev in r["gc_events"]:
            if ev["start_ms"] is None: continue
            fw = [k for k, (b, e) in enumerate(r["forwards_ms"]) if b <= ev["start_ms"] <= e]
            k = fw[0] if fw else None
            if ev["gen"] == 0 and r["task_idx"] > 0: pos[k if k is not None else "outside"] += 1
            if ev["gen"] == 2:
                pre = r["samples_ms"].get("attn_pre_proj", [])
                g2.append((r["tp"], r["gpu"], r["task_idx"], r["num_tokens"], k, None if k is None else k - 3, round(ev["dur_ms"], 1), ev["collected"], None if k is None or k >= len(pre) else round(pre[k], 1)))
        pre = r["samples_ms"].get("attn_pre_proj")
        if pre:
            t = pre[3:]; med = st.median(t); mx = max(t)
            if mx > 20 * med and mx > 20: spikes.append((r["tp"], r["gpu"], r["task_idx"], r["num_tokens"], t.index(mx), round(mx, 1)))
print("workers:", len(files), " forwards per task:", nfw)
print("gen-0 landing forwards (forward idx -> count):", sorted(pos.items(), key=lambda x: (x[0] is None, x[0]) if not isinstance(x[0], str) else (True, 0)))
print("gen-2 events: (tp, gpu, task, tokens, forward, timed_run, dur_ms, collected, attn_pre_proj_ms_at_that_forward)")
for g in sorted(g2): print("  ", g)
print("100ms-class attn_pre_proj spikes: (tp, gpu, task, tokens, timed_run, ms)")
for s_ in sorted(spikes): print("  ", s_)
print("timed-run indices of gen-2 events:", sorted(collections.Counter(g[5] for g in g2).items(), key=lambda x: (x[0] is None, x[0])))
