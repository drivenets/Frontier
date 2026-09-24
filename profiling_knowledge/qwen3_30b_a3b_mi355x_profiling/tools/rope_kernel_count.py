"""Count kernels correlated to each vidur_<scope> annotation in a kineto chrome trace (same matching as RecordFunctionTracer)."""
import json, sys, collections
trace = json.load(open(sys.argv[1]))["traceEvents"]
scopes = [e for e in trace if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("vidur_")]
launches = [e for e in trace if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})]
kernels = {e["args"]["correlation"]: e for e in trace if e.get("cat") == "kernel" and "correlation" in e.get("args", {})}
per_scope = collections.defaultdict(list)
for s in scopes:
    ks = [kernels[l["args"]["correlation"]] for l in launches
          if l["ts"] > s["ts"] and l["ts"] + l["dur"] < s["ts"] + s["dur"] and l["args"]["correlation"] in kernels]
    per_scope[s["name"].replace("vidur_", "")].append((len(ks), sum(k["dur"] for k in ks), sorted({k["name"][:60] for k in ks})))
for name, rows in sorted(per_scope.items()):
    counts = sorted({r[0] for r in rows}); names = sorted({n for r in rows for n in r[2]})
    durs = [r[1] for r in rows]
    print(f"{name}: annotations={len(rows)} kernels/annotation={counts} device_us median={sorted(durs)[len(durs)//2]:.1f} kernel names={names[:6]}")
