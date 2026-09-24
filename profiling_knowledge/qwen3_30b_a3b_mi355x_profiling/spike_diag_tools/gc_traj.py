"""GC counter trajectory per worker: gen2 count at task begin/end (resets to 0 only when a gen-2 collection ran),
gen-1 events per task, and detect gen-2 collections that fell between tasks (count drops between task_end and next task_begin)."""
import json, sys, glob
for f in sorted(glob.glob(sys.argv[1] + "/worker_gpu0_*.jsonl"))[:2] + sorted(glob.glob(sys.argv[1] + "/worker_gpu5_*.jsonl"))[:1]:
    recs = [json.loads(l) for l in open(f) if l.strip()]
    tasks = [r for r in recs if r["kind"] == "task"]
    print("==", f.split("/")[-1], "tp", tasks[0]["tp"], "tasks", len(tasks))
    prev_end = None
    between = []
    for r in tasks:
        b, e = r["gc_count_begin"], r["gc_count_end"]
        if prev_end is not None and b[2] < prev_end[2]:
            between.append((r["task_idx"], prev_end, b))
        prev_end = e
    g1 = [sum(1 for ev in r["gc_events"] if ev["gen"] == 1) for r in tasks]
    g2in = [(r["task_idx"], [ (ev["start_ms"], ev["dur_ms"], ev["collected"]) for ev in r["gc_events"] if ev["gen"] == 2]) for r in tasks if any(ev["gen"] == 2 for ev in r["gc_events"])]
    print("  gen2-count at begin of tasks 0,1,2,8,9,50,100,168,169,170,300,415:", [tasks[i]["gc_count_begin"][2] for i in (0,1,2,8,9,50,100,168,169,170,300,415) if i < len(tasks)])
    print("  gen1 collections per task: min %d max %d, task0 %d" % (min(g1), max(g1), g1[0]))
    print("  gen-2 inside tasks:", g2in[:6])
    print("  gen-2 that fell between tasks (task_idx, prev_end_counts, begin_counts):", between[:8], "total", len(between))
    # host time per forward summary & task durations
    import statistics as st
    print("  task_ms task0 %.0f task1 %.0f median %.0f; model-build (first fwd begin) task1 %.0f ms" % (tasks[0]["task_ms"], tasks[1]["task_ms"], st.median(r["task_ms"] for r in tasks), tasks[1]["forwards_ms"][0][0]))
