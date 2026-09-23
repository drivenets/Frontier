import csv, sys, bisect, collections, numpy as np
pfx = sys.argv[1]  # e.g. res/rocprof/R9_tp8
WU = int(sys.argv[2]) if len(sys.argv) > 2 else 3
marks = [r for r in csv.DictReader(open(pfx + "_marker_api_trace.csv"))]
tasks = [(r["Function"], int(r["Start_Timestamp"]), int(r["End_Timestamp"])) for r in marks if r["Function"].startswith("task_")]
fwds = [(int(r["Function"][3:]), int(r["Start_Timestamp"]), int(r["End_Timestamp"])) for r in marks if r["Function"].startswith("fwd")]
api = [r for r in csv.DictReader(open(pfx + "_hip_api_trace.csv")) if "HIP_RUNTIME" in r["Domain"]]
kern = [r for r in csv.DictReader(open(pfx + "_kernel_trace.csv"))]
print(f"{pfx}: tasks={len(tasks)} fwd ranges={len(fwds)} api={len(api)} kernels={len(kern)}")
corr2api = {int(r["Correlation_Id"]): r for r in api}
def task_of(t): 
    for name, s, e in tasks:
        if s <= t <= e: return name
    return None
for tname, ts, te in tasks:
    tf = sorted([(k, s, e) for k, s, e in fwds if ts <= s <= te])
    starts = [s for _, s, _ in tf]
    def fwd_of(t):
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and tf[i][1] <= t <= tf[i][2]: return tf[i][0]
        return None
    tapi = [r for r in api if ts <= int(r["Start_Timestamp"]) <= te]
    per = collections.defaultdict(lambda: collections.Counter())
    for r in tapi:
        per[fwd_of(int(r["Start_Timestamp"]))][r["Function"]] += 1
    tk = []
    for r in kern:
        a = corr2api.get(int(r["Correlation_Id"]))
        if a is None or not (ts <= int(a["Start_Timestamp"]) <= te): continue
        tk.append((fwd_of(int(a["Start_Timestamp"])), int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"][:70], a["Function"]))
    tk.sort(key=lambda x: x[1])
    nf = len(tf)
    print(f"\n=== {tname}: {nf} forwards; kernels correlated={len(tk)}")
    # packets per forward: kernel launches + hipEventRecord + memcpy
    rows = []
    for k, s, e in tf:
        c = per[k]; nk = sum(1 for x in tk if x[0] == k)
        rows.append((k, nk, c["hipEventRecord"], c["hipMemcpyAsync"], c["hipLaunchKernel"] + c["hipExtModuleLaunchKernel"]))
    legacy = [r for r in rows if r[0] < nf // 2]; gpu = [r for r in rows if r[0] >= nf // 2]
    def summ(rs): 
        a = np.array([[r[1], r[2], r[3], r[4]] for r in rs]); return f"kernels/fwd={collections.Counter(a[:,0]).most_common(3)} records/fwd={collections.Counter(a[:,1]).most_common(2)} memcpy/fwd={collections.Counter(a[:,2]).most_common(2)} launchcalls/fwd={collections.Counter(a[:,3]).most_common(2)}"
    print("   legacy pass:", summ(legacy)); print("   gpu-bound  :", summ(gpu))
    outside = per[None]; print("   calls outside fwd ranges:", {k: v for k, v in outside.items() if k in ("hipEventRecord","hipLaunchKernel","hipDeviceSynchronize","hipMemcpyAsync","hipExtModuleLaunchKernel")})
    # device-side per forward in GPU-bound pass
    print("   GPU-bound pass, per timed run: device span (first kernel start -> last kernel end) us, sum kernel us, total gap us, max gap us & where; period to next fwd")
    byf = collections.defaultdict(list)
    for x in tk: byf[x[0]].append(x)
    base = nf // 2 + WU
    stats = []
    for k in range(base, nf):
        ks = byf[k]
        if not ks: continue
        span = (ks[-1][2] - ks[0][1]) / 1e3; ksum = sum(e - s for _, s, e, _, _ in ks) / 1e3
        gaps = [(ks[i+1][1] - ks[i][2]) / 1e3 for i in range(len(ks) - 1)]
        mg = max(gaps); mi = gaps.index(mg)
        nxt = byf.get(k + 1); period = (nxt[0][1] - ks[0][1]) / 1e3 if nxt else float("nan")
        stats.append((k - base + 1, span, ksum, sum(gaps), mg, mi, ks[mi][3][:28], ks[mi+1][3][:28], period, len(ks)))
    for st in stats:
        print(f"      run{st[0]:3d}: span={st[1]:7.1f} ksum={st[2]:7.1f} gaps={st[3]:6.1f} maxgap={st[4]:5.1f} after k#{st[5]:2d} [{st[6]} -> {st[7]}] period={st[8]:7.1f} nk={st[9]}")
    # legacy pass host side: long API calls / gaps around all forwards
    print("   LEGACY pass host side: API calls > 40us and inter-call gaps > 40us, by timed run:")
    for k, s, e in tf:
        if k >= nf // 2: continue
        calls = sorted([(int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Function"]) for r in tapi if fwd_of(int(r["Start_Timestamp"])) == k])
        longc = [(f, (e_ - s_) / 1e3) for s_, e_, f in calls if (e_ - s_) > 40_000]
        gaps = [((calls[i+1][0] - calls[i][1]) / 1e3, calls[i][2], calls[i+1][2]) for i in range(len(calls) - 1) if calls[i+1][0] - calls[i][1] > 40_000]
        if longc or gaps:
            print(f"      fwd{k} (timed run {k - WU + 1}): wall={(e - s)/1e3:.0f}us long calls={[(f, round(d,1)) for f, d in longc]} gaps={[(round(g,1), a, b) for g, a, b in gaps]}")
    print("   GPU-BOUND pass host side: API calls > 40us / gaps > 40us:")
    for k, s, e in tf:
        if k < nf // 2: continue
        calls = sorted([(int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Function"]) for r in tapi if fwd_of(int(r["Start_Timestamp"])) == k])
        longc = [(f, (e_ - s_) / 1e3) for s_, e_, f in calls if (e_ - s_) > 40_000]
        gaps = [((calls[i+1][0] - calls[i][1]) / 1e3, calls[i][2], calls[i+1][2]) for i in range(len(calls) - 1) if calls[i+1][0] - calls[i][1] > 40_000]
        if longc or gaps:
            print(f"      fwd{k} (timed run {k - nf//2 - WU + 1}): wall={(e - s)/1e3:.0f}us long calls={[(f, round(d,1)) for f, d in longc]} gaps={[(round(g,1), a, b) for g, a, b in gaps]}")
