"""Per-position normalized profile + anomaly characterization. Usage: python profile.py <csv> <prefix> <n_expected>"""
import sys, json, numpy as np, pandas as pd
path, prefix = sys.argv[1], sys.argv[2]
df = pd.read_csv(path, low_memory=False)
def load(op, tp, pfx=prefix):
    sc, wc = f"{pfx}.{op}.samples", f"{pfx}.{op}.warmup_count"
    sub = df[(df.num_tensor_parallel_workers==tp) & df[sc].notna()].sort_values("num_tokens")
    A = np.array([json.loads(r)[int(w):] for r, w in zip(sub[sc], sub[wc])])
    return sub.num_tokens.values, A
np.set_printoptions(linewidth=250, precision=3, suppress=True)
print("=== A. per-position mean of sample/row-median (x1000, i.e. permille), positions 1..n")
for op in ["attn_pre_proj","attn_post_proj","attn_rope","forward_gpu_span"]:
    for tp in [1,2,4,8]:
        try: tok, A = load(op, tp)
        except Exception as e: continue
        if A.size==0: continue
        med = np.median(A,1,keepdims=True); R = A/med
        prof = np.nanmean(R,0)*1000; profmed = np.median(R,0)*1000
        print(f"{op:16s} TP{tp} mean  :", " ".join(f"{v:5.0f}" for v in prof))
        print(f"{op:16s} TP{tp} median:", " ".join(f"{v:5.0f}" for v in profmed))
print("\n=== B. run-20 (and neighbours) excess in attn_pre_proj GPU-bound: sample - row median, microseconds")
for tp in [1,2,4,8]:
    tok, A = load("attn_pre_proj", tp)
    n = A.shape[1]
    med = np.median(A,1)
    for k in [19,20,21,22]:
        ex = (A[:,k-1]-med)*1000
        print(f"TP{tp} run{k}: excess us p10={np.percentile(ex,10):.1f} p50={np.percentile(ex,50):.1f} p90={np.percentile(ex,90):.1f} p99={np.percentile(ex,99):.1f}  frac>+5us={np.mean(ex>5):.2f} frac>+20us={np.mean(ex>20):.2f}")
    is20 = A.argmax(1)==19
    # token dependence
    bins=[0,64,256,1024,2048,4096,8192,16385]
    print(f"TP{tp} P(argmax=20) by token bin:", " ".join(f"[{bins[i]}-{bins[i+1]}):{np.mean(is20[(tok>=bins[i])&(tok<bins[i+1])]):.2f}" for i in range(len(bins)-1)))
    # other scopes in same forward
    for op2 in ["attn_rope","attn_post_proj","forward_gpu_span"]:
        tok2, B = load(op2, tp)
        assert (tok2==tok).all()
        medB = np.median(B,1)
        for k in [20,21]:
            exB=(B[:,k-1]-medB)*1000
            print(f"   TP{tp} {op2:16s} run{k} excess us: p50={np.percentile(exB,50):.1f} p90={np.percentile(exB,90):.1f} | rows where preproj argmax=20: p50={np.percentile(exB[is20],50):.1f} p90={np.percentile(exB[is20],90):.1f}")
    ex20 = (A[:,19]-med)*1000
    print(f"TP{tp} pre_proj run20 excess where argmax=20: p10={np.percentile(ex20[is20],10):.1f} p50={np.percentile(ex20[is20],50):.1f} p90={np.percentile(ex20[is20],90):.1f} us; row median p50={np.median(med)*1000:.1f} us")
print("\n=== C. legacy column: run 21/22 excess in attn_pre_proj (us)")
for tp in [1,2,4,8]:
    tok, A = load("attn_pre_proj", tp, "time_stats_hostbound")
    med = np.median(A,1)
    for k in [1,20,21,22,23]:
        ex=(A[:,k-1]-med)*1000
        print(f"TP{tp} legacy run{k}: excess us p10={np.percentile(ex,10):.1f} p50={np.percentile(ex,50):.1f} p90={np.percentile(ex,90):.1f} p99={np.percentile(ex,99):.1f}")
    am = A.argmax(1)+1
    bins=[0,64,256,1024,2048,4096,8192,16385]
    print(f"TP{tp} legacy P(argmax in 21..22) by token bin:", " ".join(f"[{bins[i]}-{bins[i+1]}):{np.mean(np.isin(am,[21,22])[(tok>=bins[i])&(tok<bins[i+1])]):.2f}" for i in range(len(bins)-1)))
