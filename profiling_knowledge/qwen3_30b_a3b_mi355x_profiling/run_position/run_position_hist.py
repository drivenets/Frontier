"""Re-derive the argmin/argmax position histograms per op x TP from a linear_op.csv.
Usage: python hist.py <csv> [prefix=time_stats]"""
import sys, json, numpy as np, pandas as pd
path = sys.argv[1]; prefix = sys.argv[2] if len(sys.argv) > 2 else "time_stats"
df = pd.read_csv(path, low_memory=False)
print(path, df.shape)
ops = ["attn_pre_proj","attn_post_proj","attn_rope","input_layernorm","post_attention_layernorm","emb","forward_gpu_span"]
for op in ops:
    sc, wc = f"{prefix}.{op}.samples", f"{prefix}.{op}.warmup_count"
    if sc not in df.columns: print(f"-- {op}: no column {sc}"); continue
    for tp in sorted(df.num_tensor_parallel_workers.unique()):
        sub = df[(df.num_tensor_parallel_workers==tp) & df[sc].notna()]
        if len(sub)==0: continue
        arrs=[]; 
        for raw,w in zip(sub[sc], sub[wc]):
            s=json.loads(raw)[int(w):]; arrs.append(s)
        lens=set(map(len,arrs)); 
        if len(lens)!=1: print(f"-- {op} TP{tp}: mixed lengths {lens}"); continue
        A=np.array(arrs); n=A.shape[1]; N=A.shape[0]
        amin=A.argmin(1)+1; amax=A.argmax(1)+1
        # ties
        tmin=(A==A.min(1,keepdims=True)).sum(1); tmax=(A==A.max(1,keepdims=True)).sum(1)
        hmin=np.bincount(amin,minlength=n+1)[1:]/N*100; hmax=np.bincount(amax,minlength=n+1)[1:]/N*100
        base=100/n
        top_min=sorted(range(n),key=lambda i:-hmin[i])[:6]; top_max=sorted(range(n),key=lambda i:-hmax[i])[:6]
        print(f"== {op} TP{tp}: rows={N} n_timed={n} baseline={base:.2f}%  ties(min>1)={int((tmin>1).sum())} ties(max>1)={int((tmax>1).sum())}")
        print("   argmin top: " + ", ".join(f"run{i+1}={hmin[i]:.1f}%" for i in top_min))
        print("   argmax top: " + ", ".join(f"run{i+1}={hmax[i]:.1f}%" for i in top_max))
        if n>=25:
            b=[k for k in range(25,n+1,25)]
            print("   argmin at block ends " + " ".join(f"r{k}={hmin[k-1]:.1f}%" for k in b) + f" | r{n-1}={hmin[n-2]:.1f}% (n-1)")
