"""For every gen-2 collection recorded inside a task: absolute wall time, duration, task idx, forward idx, tokens, tp,
plus the attn_pre_proj value of that forward, per worker. Shows whether the 8 workers collect simultaneously."""
import json, sys, glob, statistics as st
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/worker_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["kind"] != "task": continue
        for e in r["gc_events"]:
            if e["gen"] != 2 or e["start_ms"] is None: continue
            fw = [k for k, (b, en) in enumerate(r["forwards_ms"]) if b <= e["start_ms"] <= en]
            pre = r["samples_ms"].get("attn_pre_proj", [])
            val = pre[fw[0]] if fw and fw[0] < len(pre) else None
            host = (r["forwards_ms"][fw[0]][1] - r["forwards_ms"][fw[0]][0]) if fw else None
            rows.append((r["tp"], r["gpu"], r["task_idx"], r["num_tokens"], fw, r["wall_start"] + e["start_ms"] / 1e3, e["dur_ms"], e["collected"], val, host, r["gc_count_begin"]))
rows.sort()
print("tp gpu task tokens fwd_idx  wall_start(gc)      gc_dur_ms collected attn_pre_proj_ms host_fwd_ms gc_count_at_task_begin")
for tp, gpu, ti, nt, fw, w, d, c, v, h, gcb in rows:
    print(f"{tp:2d} {gpu:3d} {ti:4d} {nt:6d} {str(fw):6s} {w:17.3f} {d:9.1f} {c:9d} {v if v is None else round(v,1)!s:>16} {h if h is None else round(h,1)!s:>11} {gcb}")
for tp in sorted({r[0] for r in rows}):
    ws = [r[5] for r in rows if r[0] == tp]
    if len(ws) > 1: print(f"tp{tp}: gen-2 start spread across workers = {(max(ws)-min(ws))*1e3:.0f} ms")
