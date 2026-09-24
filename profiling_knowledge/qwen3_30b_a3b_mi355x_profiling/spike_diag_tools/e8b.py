import json, sys, glob, collections
f = sorted(glob.glob(sys.argv[1] + "/worker_gpu0_*.jsonl"))
res = collections.defaultdict(list)
for fn in f:
    for l in open(fn):
        r = json.loads(l)
        if r.get("kind") != "task": continue
        for ev in r["gc_events"]:
            if ev["gen"] != 2 or ev["start_ms"] is None: continue
            fw = [k for k, (b, e) in enumerate(r["forwards_ms"]) if b <= ev["start_ms"] <= e][0]
            host = r["forwards_ms"][fw][1] - r["forwards_ms"][fw][0]
            top = sorted(((v[fw], k) for k, v in r["samples_ms"].items() if fw < len(v)), reverse=True)[:2]
            res[(r["task_idx"], fw)].append((r["tp"], round(ev["dur_ms"]), round(host), [(k, round(v, 1)) for v, k in top]))
for (t, fw), rows in sorted(res.items()):
    print(f"task {t:3d} forward {fw:3d} (run {fw-3:3d}):", rows[0], "| tp2/4/8 top scope:", [r[3][0] for r in rows[1:]])
