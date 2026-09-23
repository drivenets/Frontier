import json, sys, glob, numpy as np
np.set_printoptions(linewidth=250)
def timed(d): return np.array(d["samples"][d["warmup_count"]:])
def profile(recs, pas, op):
    A = np.array([timed(r[pas][op]) for r in recs if op in r[pas]])
    med = np.median(A,1,keepdims=True); return np.median(A/med,0)*1000, A
files = sys.argv[1:]
for f in files:
    recs = [json.loads(l) for l in open(f)]
    r0 = recs[0]; n = len(timed(r0["gpu"]["attn_pre_proj"]))
    print(f"\n######## {f}: {len(recs)} tasks, mode={r0['mode']}, tp={r0['tp']}, tokens={[r['tokens'] for r in recs]}, n_timed={n}, warmup={r0['warmup_steps']}, block={r0['block_steps']}")
    print(f"   backlog req/actual ms: {[ (round(r['gpu_backlog_ms'],1), round(r['gpu_backlog_ms_actual'],1)) for r in recs][:3]} ; host enqueue/fwd backlog ms {[round(r['host_enqueue_per_forward_ms_backlog'],3) for r in recs][:3]}; legacy host wall/fwd {[round(r['host_wall_per_forward_ms'],3) for r in recs][:3]}")
    print(f"   sclk legacy start/end, backlog start/end (first task): {r0['sclk_mhz_legacy_start']:.0f}/{r0['sclk_mhz_legacy_end']:.0f} {r0['sclk_mhz_backlog_start']:.0f}/{r0['sclk_mhz_backlog_end']:.0f}")
    for op in ["attn_pre_proj","attn_post_proj","forward_gpu_span"]:
        prof, A = profile(recs, "gpu", op)
        if n <= 25:
            print(f"   GPU {op:16s} profile: {np.round(prof).astype(int)}")
        else:
            spikes = [(i+1,int(v)) for i,v in enumerate(prof) if v>1006 and (i%25)>3]
            print(f"   GPU {op:16s} spike positions (permille>1006, not first 4 of block): {spikes}")
            print(f"   GPU {op:16s} block-avg profile: {np.round(prof.reshape(-1,25).mean(0)).astype(int)}")
        # drift: median over rows of (mean pos 1-5)/(mean pos 21-25) per block
        R = A/np.median(A,1,keepdims=True)
        blocks = R.reshape(R.shape[0], -1, 25)
        early = np.median(blocks[:,:,0:5],axis=2); late = np.median(blocks[:,:,19:25],axis=2)
        print(f"   GPU {op:16s} early(1-5)/late(20-25) per task: {np.round(np.median(early/late,1),4)}  (row medians us: {np.round(np.median(A,1)*1000,1)})")
    A = np.array([timed(r["gpu"]["attn_pre_proj"]) for r in recs]); med=np.median(A,1)
    print(f"   GPU pre_proj argmax pos per task: {list(A.argmax(1)+1)}")
    if n<=25:
        for k in [19,20,21]:
            print(f"   GPU pre_proj excess at run{k} us per task: {np.round((A[:,k-1]-med)*1000,1)}")
    else:
        prof,_ = profile(recs,"gpu","attn_pre_proj"); print(f"   GPU pre_proj positions permille>1015: {[(i+1,int(v)) for i,v in enumerate(prof) if v>1015 and (i%25)>2]}")
    for op in ["attn_pre_proj","attn_post_proj"]:
        prof, A = profile(recs, "legacy", op)
        spikes = [(i+1,int(v)) for i,v in enumerate(prof) if v>1030]
        print(f"   LEGACY {op:14s} spike positions (permille>1030): {spikes}")
        if n<=25: print(f"   LEGACY {op:14s} profile: {np.round(prof).astype(int)}")
