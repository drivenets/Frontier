"""Cost of gen-0 / gen-1 collections and their effect on the attn_pre_proj sample they land in (brief Q4)."""
import json, sys, glob, statistics as st
d0=[]; d1=[]; hit=[]; miss=[]
for f in sorted(glob.glob(sys.argv[1] + "/worker_*.jsonl")):
    for l in open(f):
        r=json.loads(l)
        if r["kind"]!="task" or r["task_idx"]==0: continue
        pre=r["samples_ms"].get("attn_pre_proj")
        if not pre: continue
        med=st.median(pre[3:])
        landed=set()
        for ev in r["gc_events"]:
            if ev["start_ms"] is None: continue
            fw=[k for k,(b,e) in enumerate(r["forwards_ms"]) if b<=ev["start_ms"]<=e]
            if ev["gen"]==0: d0.append(ev["dur_ms"])
            elif ev["gen"]==1: d1.append(ev["dur_ms"]); 
            if fw and ev["gen"]==1: landed.add(fw[0]); hit.append(pre[fw[0]]/med)
        for k in (14,36):
            if k not in landed and any(ev["gen"]==0 and ev["start_ms"] is not None and r["forwards_ms"][k][0]<=ev["start_ms"]<=r["forwards_ms"][k][1] for ev in r["gc_events"]):
                miss.append(pre[k]/med)
q=lambda a,p: sorted(a)[int(p*(len(a)-1))]
print(f"gen-0 collections: n={len(d0)} dur ms p50={q(d0,.5):.3f} p99={q(d0,.99):.3f} max={max(d0):.3f}")
print(f"gen-1 collections: n={len(d1)} dur ms p50={q(d1,.5):.3f} p99={q(d1,.99):.3f} max={max(d1):.3f}")
print(f"attn_pre_proj sample / row-median where a gen-1 landed in that forward: n={len(hit)} p50={q(hit,.5):.2f} p90={q(hit,.9):.2f} max={max(hit):.2f}")
print(f"same ratio where only a gen-0 landed in forward 14/36: n={len(miss)} p50={q(miss,.5):.2f} p90={q(miss,.9):.2f} max={max(miss):.2f}")
