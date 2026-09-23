import csv, sys, bisect
pfx, task, runs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
WU=3
marks = list(csv.DictReader(open(pfx + "_marker_api_trace.csv")))
ts, te = [(int(r["Start_Timestamp"]), int(r["End_Timestamp"])) for r in marks if r["Function"] == task][0]
fw = sorted([(int(r["Function"][3:]), int(r["Start_Timestamp"]), int(r["End_Timestamp"])) for r in marks if r["Function"].startswith("fwd") and ts <= int(r["Start_Timestamp"]) <= te])
nf = len(fw); starts=[s for _,s,_ in fw]
def fwd_of(t):
    i = bisect.bisect_right(starts, t) - 1
    return fw[i][0] if i >= 0 and fw[i][1] <= t <= fw[i][2] else None
api = {int(r["Correlation_Id"]): r for r in csv.DictReader(open(pfx + "_hip_api_trace.csv")) if "HIP_RUNTIME" in r["Domain"] and ts <= int(r["Start_Timestamp"]) <= te}
ks = []
for r in csv.DictReader(open(pfx + "_kernel_trace.csv")):
    a = api.get(int(r["Correlation_Id"]))
    if a: ks.append((fwd_of(int(a["Start_Timestamp"])), int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
ks.sort(key=lambda x: x[1])
for run in runs:
    k = nf//2 + WU + run - 1
    seq = [x for x in ks if x[0] == k]
    prev_end = max([x[2] for x in ks if x[1] < seq[0][1]])
    print(f"--- {task} GPU-bound timed run {run} (fwd{k}): {len(seq)} kernels; gap from previous forward's last kernel = {(seq[0][1]-prev_end)/1e3:.1f} us")
    for i, (_, s, e, n) in enumerate(seq):
        gap = (s - (seq[i-1][2] if i else prev_end)) / 1e3
        print(f"   k{i:2d} gap_before={gap:5.1f} dur={(e-s)/1e3:6.1f}  {n[:90]}")
