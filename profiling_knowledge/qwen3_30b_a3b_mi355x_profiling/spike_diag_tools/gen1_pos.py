import json, sys, glob, collections
f = sorted(glob.glob(sys.argv[1] + "/worker_gpu0_*.jsonl"))[0]
tasks = [json.loads(l) for l in open(f) if l.strip()]
tasks = [r for r in tasks if r["kind"] == "task"]
pos = collections.Counter(); alloc_per_task = []
for r in tasks:
    b, e = r["gc_count_begin"], r["gc_count_end"]
    for ev in r["gc_events"]:
        if ev["gen"] >= 1 and ev["start_ms"] is not None:
            fw = [k for k, (bb, en) in enumerate(r["forwards_ms"]) if bb <= ev["start_ms"] <= en]
            pos[(ev["gen"], fw[0] if fw else ("pre-fwd" if ev["start_ms"] < r["forwards_ms"][0][0] else "post"))] += 1
print(f.split("/")[-1], "tp", tasks[0]["tp"])
print("gen>=1 collection positions (gen, forward_idx) -> count:", sorted(pos.items(), key=lambda x: -x[1])[:12])
# gen0 count delta per task = allocations mod 700 progression; estimate tracked allocs per task from gen0 count change and #gen0 collections
g0 = [sum(1 for ev in r["gc_events"] if ev["gen"] == 0) for r in tasks]
est = [(r["gc_count_end"][0] - r["gc_count_begin"][0]) + 700 * n for r, n in zip(tasks, g0)]
print("gen0 collections per task (tasks 1..10):", g0[1:11], " est. tracked allocations per task (tasks 1..10):", est[1:11], "median", sorted(est[1:])[len(est)//2])
# gen0 positions
p0 = collections.Counter()
for r in tasks[1:]:
    for ev in r["gc_events"]:
        if ev["gen"] == 0 and ev["start_ms"] is not None:
            fw = [k for k, (bb, en) in enumerate(r["forwards_ms"]) if bb <= ev["start_ms"] <= en]
            p0[fw[0] if fw else "outside-fwd"] += 1
print("gen0 collection forward positions:", sorted(p0.items(), key=lambda x: (str(type(x[0])), x[0]))[:60])
