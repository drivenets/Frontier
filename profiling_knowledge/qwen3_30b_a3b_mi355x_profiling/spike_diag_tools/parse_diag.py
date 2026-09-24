"""Parse spike_diag JSONL: for every task with a >20x attn_pre_proj run, print the spike run, the host-side
duration of that forward, and every GC event overlapping that forward's host window. Also list all gen-2 events."""
import json, sys, glob, statistics as st
files = sorted(glob.glob(sys.argv[1] + "/worker_*.jsonl"))
for f in files:
    recs = [json.loads(l) for l in open(f) if l.strip()]
    init = [r for r in recs if r["kind"] == "worker_init"]
    tasks = [r for r in recs if r["kind"] == "task"]
    if init:
        i = init[0]
        print(f"== {f.split('/')[-1]}: gc_mode={i['gc_mode']} thr={i['gc_threshold']} tracked_after_imports={i['n_tracked_objects_after_imports']} init_collect={i.get('init_collect')} full_collect_at_init={i['full_collect_ms_at_init']} frozen={i['frozen']} tasks={len(tasks)}")
    gen2 = [(r["task_idx"], r["num_tokens"], r["tp"], e) for r in tasks for e in r["gc_events"] if e["gen"] == 2]
    print(f"   gen2 collections: {len(gen2)}")
    for ti, nt, tp, e in gen2:
        # which forward contains it
        r = next(x for x in tasks if x["task_idx"] == ti)
        fw = [k for k, (b, en) in enumerate(r["forwards_ms"]) if e["start_ms"] is not None and b <= e["start_ms"] <= en]
        print(f"     task {ti:3d} tokens {nt:5d} tp {tp} gen2 start {e['start_ms']:.1f}ms dur {e['dur_ms']:.1f}ms collected {e['collected']} in_forward={fw}")
    for r in tasks:
        pre = r["samples_ms"].get("attn_pre_proj")
        if not pre: continue
        timed = pre[3:]; med = st.median(timed); mx = max(timed); k = timed.index(mx)
        if mx > 20 * med:
            fb, fe = r["forwards_ms"][3 + k]
            host = fe - fb
            hosts = [e - b for b, e in r["forwards_ms"]]
            ov = [e for e in r["gc_events"] if e["start_ms"] is not None and not (e["stop_ms"] < fb or e["start_ms"] > fe)]
            print(f"   SPIKE task {r['task_idx']} tokens {r['num_tokens']} tp {r['tp']}: run {k} gpu-gap {mx:.1f}ms (median {med:.3f}); host time of that forward {host:.1f}ms (median host fwd {st.median(hosts):.2f}ms); gc overlapping: {ov}; mem={r.get('memory_stats_after')}")
