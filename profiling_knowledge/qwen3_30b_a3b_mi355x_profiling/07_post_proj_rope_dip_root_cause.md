# Root cause of the `attn_post_proj` / `attn_rope` mid-range "dip" at TP>1 (answer to `06_post_proj_rope_dip_specialist_brief.md`)

Investigation date: 2026-09-16, node `amd-mi355x-8`, image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`, code `7f4dfa5` plus the
opt-in knobs in §2. Evidence labels: **CONFIRMED** (direct evidence, cited), **RULED OUT** (direct evidence against),
**SPECULATIVE** (plausible, not tied to the effect by evidence).

## 1. Executive summary

**CONFIRMED:** the dip is not a GPU effect. In the sweep the GPU is idle 45–60 % of every forward at TP 2/4/8 in the
3000–8000-token band (kernel trace, §3.3), so the CUDA-event pair around a short op measures the *host's* Python/HIP launch
span (≈0.05 ms for `attn_post_proj`, ≈0.07–0.09 ms for the 13-kernel RoPE fallback) rather than the kernel. The
"plateau" is that host floor; the "dip" is the token count at which the GPU work queued *earlier in the same forward*
(QKV GEMM, QK-norms, RoPE chain, `randn_like`) becomes long enough that the GPU is still busy when `attn_post_proj`'s start
event is recorded, so the pair collapses to the true kernel time (28–40 µs), which then grows monotonically with tokens.
Holding the GPU 100 ms behind the host for the timed loop (a one-knob change, job 21356) removes the dip and the plateau for
every op at every TP and yields monotonic, low-variance curves that match the traced kernel durations to within one
inter-kernel gap (§3.1, §3.2). **RULED OUT:** a hipBLASLt kernel-selection speed-up keyed to `K = 4096/TP` (brief candidate 1):
the traced `o_proj` kernel duration is monotonic through the dip band and the only selection effects found are *slow-downs*
(§3.4). **RULED OUT:** a separate RoPE-kernel occupancy mechanism (candidate 2): the same host-floor mechanism, acting on a
13-kernel torch fallback chain, reproduces every `attn_rope` observation (§3.5). **CONFIRMED, beyond the brief:** the same
artefact inflates `attn_pre_proj`, `attn_rope` and `attn_post_proj` medians by 1.2–3.0× over large parts of the TP>1 dataset
(up to ≈9000–10000 tokens at TP8) and TP1 below ≈2000–3000 tokens (§3.6, §5).

## 2. What was checked and how

Code read (worktree `smatar/qwen3-30b-mi355-profiling`): `frontier/profiling/common/cuda_timer.py` (CUDA_EVENT path: two
`torch.cuda.Event.record()` calls, no synchronisation), `linear_op_wrapper.py::profile` (3 warm-up + 50 timed forwards
enqueued back-to-back; the only `torch.cuda.synchronize()` calls are before and after the timed loop),
`common/timer_stats_store.py` (`elapsed_time` between the two events), `linear_op_impl.py::CausalSelfAttention.forward`
(scope order: `attn_pre_proj` = QKV GEMM + split + q/k RMSNorm; `attn_rope`; untimed `torch.randn_like(q)`; `attn_post_proj`
= `RowParallelLinear.apply_weights` → `_run_unquantized_gemm`, `reduce_results=False` as the brief says),
`common/layers/rotary_embedding.py` (`_should_prefer_torch_rope_fallback`, `_apply_rotary_pos_emb`, `get_rope`),
`profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` line 79 (`-e FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1`),
`MI355X_ROCM_COOKBOOK.md` gotcha 4 (why the fallback is forced), `linear_op/main.py` (task → GPU = index in the descending
token list mod 8; `--is_moe` drops MLP columns at output but the MLP still runs).

Data re-analysed: both original CSVs (jobs 21313/21334), with the brief's §8 extraction reproduced exactly, plus the
50-sample lists per row, a per-GPU split (token → worker via `idx % 8`), and the E1b/E7 diagnostics of the previous
investigation (`spike_diag/e1b`, `e7`: host wall time of every forward of every task, all GPUs, all TPs).

Experiments (all on `amd-mi355x-8`, same image; artefacts under `/opt/shared/frontier-qwen3-profiling/Frontier`):

| id | what | how | result |
|---|---|---|---|
| X1 | GPU held 100 ms behind the host for every timed loop | new opt-in `FRONTIER_GPU_BACKLOG_MS` in `linear_op_wrapper.py` (a `torch.cuda._sleep` calibrated per process, enqueued after `mark_warmup_end()`); job 21356, tokens 1024…10240 step 8 + 10368…16384 step 128, TP 1/2/4/8, 4800 rows; `data/profiling_dip_x1/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv`, log `sweep_work/logs/linear_op_dip_x1.log` (32 `[gpu_backlog]` calibration lines, 1.93–2.04 M cycles/ms) | dip and plateau gone for every op/TP (§3.1) |
| X2 | pool profiler under `rocprofv3` | job 21357 (`LINEAR_CMD_PREFIX`, new sbatch passthrough) | no trace written (rocprofv3 aborts creating `.rocprofv3/` in the root-squashed NFS cwd); its CSV is still useful: with the tracer slowing the host, the same shapes report `attn_post_proj` 0.069–0.094 ms and `attn_rope` 0.11–0.18 ms at TP2 (§3.2) |
| X2d | kernel + marker trace of one worker's task loop | new `trace_ops.py` (same `LinearOpWrapper.profile`, same profiling plan, roctx range per task), `srun` + docker on GPU 0, `rocprofv3 --kernel-trace --marker-trace`; TP2, TP4, and TP2 with the backlog; 20 token counts each; `data/profiling_dip_x2/rocprof/{tp2,tp4,tp2_backlog}_kernel_trace.csv` (28 MB each) | kernel names, durations, GPU idle gaps per forward (§3.3–3.5) |

Code changes are uncommitted and inert unless the env var / prefix is set: `linear_op_wrapper.py` (`_enqueue_gpu_backlog`),
sbatch (`LINEAR_CMD_PREFIX`, `LINEAR_TPS`), `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/trace_ops.py`. Note: while this
ran, another session edited `linear_op_wrapper.py` (08:16 UTC, `FRONTIER_LINEAR_ACTIVE_STEPS` override); both edits are
present in the file and were rsync'd to the cluster together.

## 3. Findings

### 3.1 CONFIRMED — with the GPU kept behind the host, the dip and the plateau disappear for every op at every TP

Job 21356 vs. the original dense run 21313, median of the row medians (ms) per token window; `std` is the median row std.

`attn_post_proj`:

| TP | run | 1024–1400 | 2800–3200 | 3600–4000 | 4000–4400 | 4400–5000 | 5000–6000 | 6400–7000 | 7500–8200 | 9600–10240 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | orig | 0.0481 | 0.0583 | 0.0594 | **0.0490** | 0.0534 | 0.0474 | 0.0534 | 0.0583 | 0.0912 |
| 2 | X1 | 0.0229 | 0.0334 | 0.0361 | 0.0389 | 0.0416 | 0.0473 | 0.0544 | 0.0588 | 0.0926 |
| 2 | std | 0.0137 / 0.0009 | 0.0137 / 0.0009 | 0.0138 / 0.0009 | 0.0185 / 0.0011 | 0.0183 / 0.0009 | 0.0197 / 0.0011 | 0.0128 / 0.0011 | 0.0009 / 0.0009 | 0.0010 / 0.0012 |
| 4 | orig | 0.0421 | 0.0486 | 0.0496 | 0.0511 | 0.0605 | 0.0555 | 0.0571 | 0.0585 | 0.0516 |
| 4 | X1 | 0.0153 | 0.0222 | 0.0239 | 0.0256 | 0.0347 | 0.0295 | 0.0315 | 0.0357 | 0.0519 |
| 8 | orig | 0.0397 | 0.0472 | 0.0444 | 0.0456 | 0.0459 | 0.0489 | 0.0493 | 0.0511 | 0.0729 |
| 8 | X1 | 0.0138 | 0.0215 | 0.0188 | 0.0197 | 0.0202 | 0.0234 | 0.0240 | 0.0259 | 0.0486 |
| 1 | orig | 0.0559 | 0.0511 | 0.0599 | 0.0662 | 0.0709 | 0.0797 | 0.0904 | 0.0945 | 0.1475 |
| 1 | X1 | 0.0329 | 0.0522 | 0.0611 | 0.0675 | 0.0724 | 0.0817 | 0.0917 | 0.0965 | 0.1507 |

`attn_rope`:

| TP | run | 1024–1400 | 2800–3200 | 3600–4000 | 4000–4400 | 4400–5000 | 5000–6000 | 6400–7000 | 7500–8200 | 9600–10240 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | orig | 0.0940 | 0.0773 | 0.0656 | 0.0654 | 0.0668 | 0.0692 | 0.0745 | 0.0836 | 0.0894 |
| 2 | X1 | 0.0577 | 0.0622 | 0.0642 | 0.0658 | 0.0675 | 0.0704 | 0.0771 | 0.0836 | 0.0900 |
| 4 | orig | 0.0919 | 0.0929 | 0.0882 | 0.0870 | 0.0843 | 0.0775 | 0.0678 | 0.0673 | 0.0757 |
| 4 | X1 | 0.0548 | 0.0585 | 0.0606 | 0.0611 | 0.0621 | 0.0628 | 0.0646 | 0.0692 | 0.0765 |
| 8 | orig | 0.0887 | 0.0917 | 0.0918 | 0.0916 | 0.0922 | 0.0923 | 0.0874 | 0.0828 | 0.0665 |
| 8 | X1 | 0.0537 | 0.0560 | 0.0566 | 0.0572 | 0.0577 | 0.0594 | 0.0605 | 0.0625 | 0.0693 |

Shape test (largest drop between consecutive 400-token windows of a 9-row rolling median, 2000–9000 tokens):
`attn_post_proj` TP2 orig 15.8 % → X1 −0.5 % (i.e. never drops); `attn_rope` TP2 orig 10.8 % → X1 −0.6 %; `attn_rope` TP4 orig
4.5 %/window (a 27 % decline overall) → X1 −0.4 %; `attn_rope` TP8 orig 5.9 % → X1 0.5 %. The X1 curves are monotonic in
tokens for both ops at every TP, and the row std falls from 0.012–0.030 ms to ≈0.001 ms everywhere in the band. Where the
original converges to X1 (TP2 `attn_post_proj` from ≈5000 tokens, `attn_rope` TP2 from ≈3600, TP4 from ≈6600) the two agree
to 1–3 %, so X1 is not a different measurement of the kernels, it is the same measurement with the host removed.
Per-GPU uniformity: X1 TP2 `attn_post_proj` 4000–6400 per worker 0.0457–0.0477 ms (original: 0.0465–0.0548).

### 3.2 CONFIRMED — the sweep's event intervals are host launch spans; the dip is the point where the GPU backlog inside the forward exceeds them

Three independent lines of evidence:

1. **Host time per forward is constant and larger than the GPU work.** From the E1b diagnostics (host `perf_counter` around
   each of the 50 timed `self.model()` calls, all 13,308 tasks): TP1 0.45–0.47 ms, TP2 0.54–0.57 ms, TP4 0.56–0.57 ms, TP8
   0.55–0.58 ms, flat in tokens. The sum of the op medians in the same task is 0.33–0.35 ms at 3000–4000 tokens for TP2/4/8
   (ratio 0.60–0.63) and only reaches the host time at ≈10000 tokens (TP2) or ≈16000 tokens (TP8). The GPU cannot be the
   bottleneck of the loop in the band the brief looks at.
2. **The trace shows the idle directly** (X2d, TP2, traced host period ≈750 µs): GPU idle per forward 58 % at 3000 tokens,
   50 % at 4128, 46 % at 5000, 31 % at 6432, 21 % at 8000; TP4: 63 % at 3000, still 42 % at 8000. The idle gap immediately
   before the `o_proj` kernel is 47 µs at 3000 tokens, 42 µs at 4128–5000, 40 µs at 5504, 24.5 µs at 6432, 5.8 µs from 7008
   (TP2); at TP4 it stays at 46–50 µs through 8000 tokens. With the backlog the same gap is 5.9–6.0 µs at every token count.
3. **Slowing the host raises the "op time" of unchanged kernels.** Under rocprofv3 (job 21357, pool run, per-launch tracer
   overhead) the same TP2 shapes report `attn_post_proj` 0.069–0.094 ms and `attn_rope` 0.11–0.18 ms (sweep: 0.047–0.060 and
   0.065–0.077; GPU time: 0.033–0.059 and 0.062–0.084), and the transition to GPU time moves out to ≈8000 tokens (0.0620).
   In the single-process traced run (X2d) `attn_post_proj` at 3000 tokens reads 0.066 ms while the kernel is 28.6 µs.

Why it is a *dip* for `attn_post_proj` and not a flat floor: the pair measures `max(host span, GPU completion of the scope)`
minus the start-event time, and the start event is stamped only when the GPU reaches it. At small M the GPU has drained
everything queued before `attn_post_proj` (QKV GEMM ≈33 µs, QK-norms ≈40 µs, RoPE chain ≈58 µs, `randn_like` ≈13 µs at TP2/3000
— trace, §3.5) long before the host records the start event, so both events are stamped at host time and the interval is the
host span of `apply_weights` (≈0.05–0.06 ms). As M grows those preceding kernels grow (QKV 33 → 92 µs, norms 40 → 88 µs, RoPE
58 → 90 µs, randn 13 → 28 µs from 3000 to 8000 tokens) while the host span does not, so the start event begins to be stamped
late, the interval shrinks toward the kernel time, and from there it grows with M like the kernel (X1 column). The
sample-level fingerprint the brief did not look at: in the trough rows the 50 samples are bimodal and drift downward within
the task (row 4128, TP2, job 21313: first ten samples 0.051, 0.047, 0.054, 0.041, 0.052, 0.041, 0.044, 0.038, 0.038, 0.038;
p10 = 0.0377, p90 = 0.0507), while rows from 6432 up are tight (p10–p90 0.0514–0.0537). The kernel for M = 4128 is 35.8 µs
(trace) and X1 gives 0.0389–0.0416 ms for that window, so the lower mode is the kernel and the upper values are the host.

Per-worker dependence, predicted by this mechanism and observed: workers with a faster host reach the transition at fewer
tokens. In E1b (TP2) the host forward time and the first token at which the rolling `attn_post_proj` median falls below
0.046 ms are: GPU3 0.526 ms → 4008; GPU2 0.537 → 4016; GPU6 0.544 → 4048; GPU5 0.550 → 4120; GPU4 0.571, GPU0 0.593, GPU7
0.597 → not reached in 2000–9000 tokens (GPU1, 0.578 ms, is an exception: its values are low from 3056 on and its `attn_rope`
host span is anomalously large, 0.10–0.23 ms, which is not explained here). In the two original runs the set of workers
that dip at 4000–4400 differs (21313: GPUs 1,2,3,4 at 0.041–0.046 ms, the rest 0.052–0.059; 21334: GPUs 0,1,2,4,5 at
0.038–0.045, GPUs 3,6 at 0.054–0.057), so it is a property of the worker process, not of a GPU. This is also why the row std
is high in the band (host jitter) and why the "trough" is at "≈4000–4200 across TPs" in the brief: it is set by host speed
and by the GPU work that precedes the scope, which are similar across TP 2/4/8, not by K.

### 3.3 CONFIRMED — kernel durations behind the two ops (GPU truth, X2d)

`o_proj` GEMM kernel (M = tokens, K = 4096/TP, N = 2048, BF16), median over the 50 timed forwards, µs;
identical (±1 µs) in the traced-original and traced-backlog runs, so the backlog does not change kernel behaviour or clocks:

| tokens | 3000 | 3504 | 3928 | 4032 | 4080 | 4128 | 4176 | 4320 | 4328 | 4360 | 4400 | 4840 | 4872 | 5000 | 5504 | 5912 | 6432 | 7008 | 8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP2 | 28.6 | 32.0 | 33.8 | 34.6 | 34.8 | 35.8 | 35.5 | 35.9 | **44.9** | 44.8 | 45.1 | **47.2** | 38.6 | 41.0 | 42.8 | 47.0 | 48.9 | 51.3 | 56.0 |
| TP4 | 18.7 | 21.3 | 22.0 | 21.3 | 21.6 | 23.3 | 23.2 | 23.3 | 23.6 | **31.7** | 31.8 | **32.6** | 24.3 | 24.4 | 26.7 | 27.2 | 28.2 | 28.6 | 33.6 |

X1's event medians equal these plus one inter-kernel gap (TP2/3000: 28.5 µs + 5.9 µs ≈ 0.0334 ms; TP2/8000: 56.6 + 6.0 ≈
0.0617 vs 0.0588–0.0617 measured). Nothing in this table is faster in 4000–6000 than at 3000–4000.

### 3.4 RULED OUT — a hipBLASLt kernel-selection speed-up tied to K (brief candidate 1)

The `o_proj` kernel names change with M (Tensile macro-tile in the name), but never toward a faster kernel in the dip band:

| M range (TP2) | kernel | effect |
|---|---|---|
| 3000 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x192x64_MI16x16x1_SN` | — |
| 3504–4080 | `…MT128x256x64…` | continuous (32.0 → 34.8 µs) |
| 4128–4320 | `…MT256x160x64…` | continuous (35.8 → 35.9 µs) |
| **4328–4840** | `…MT160x256x64…` | **+25 % slower** (35.9 → 44.9 µs at 4320 → 4328), back to `MT256x160` at 4872 (38.6 µs) |
| 5504–5912 | `…MT256x192x64…` | continuous |
| ≥ 6432 | `Custom_Cijk_…_NTD_SK3_UserArgs_MT256x256x…` (stream-K) | continuous |

TP4 has the same tile swap one step later (4328 → 4360: `MT256x160` → `MT160x256`, 23.6 → 31.7 µs, +34 %, back at 4872) and
`MT256x128` at 4032–4080. These swaps are real, K-dependent, and visible in the original data as *bumps* (job 21313 TP2:
4320 → 0.0409, 4328 → 0.0472 ms; TP4 4400–5000 window 0.0605 vs 0.0511/0.0555; X1 TP4 4360–4840 0.0343–0.0356 vs 0.0257/0.0270),
and X1 shows two more at TP8 (2832–2928: 0.0214 vs 0.0166; 3088–3184: 0.0240 vs 0.0166 ms, not traced). They are the opposite
sign of the phenomenon in the brief and account for at most 10 µs; the 35–37 % drop from 0.058 to 0.037 ms has no kernel
counterpart. Brief question 7.1 ("is the fast kernel a better choice the library fails to pick elsewhere?") therefore has no
object: there is no fast kernel; there is a slow tile band at 4328–4840 (TP2) / 4360–4840 (TP4) that hipBLASLt's heuristic
picks and which would be worth reporting upstream separately.

### 3.5 CONFIRMED — `attn_rope` is the same host-floor artefact acting on a 13-kernel torch fallback, not a RoPE kernel

`attn_rope` in every linear_op run of this dataset is the pure-PyTorch fallback, because the sbatch (line 79) sets
`FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1` (cookbook gotcha 4: the bundled vLLM's `get_rope()` signature changed). The
brief's §3 ("a RoPE kernel via `get_rope`, not a GEMM") is therefore wrong in an important way: `_apply_rotary_pos_emb`
launches, per forward (trace, TP2/3000, µs): `vectorized_gather` 2.9 (index_select), then for q and for k: mul 4.3,
neg 4.5, cat 5.1 (`_rotate_half`), mul 3.0, add 3.8, and finally two `CatArrayBatchedCopy` of 12.5 (q) and 4.8 (k) that copy the
whole tensors — 13 kernels, 2–5 µs each except the copies, ≈58 µs of GPU time at 3000 tokens, ≈90 µs at 8000. Its event
interval in the sweep (0.065–0.093 ms) is the host span of issuing 13 torch ops; the GPU time (X1) is 0.055–0.062 ms at
3000 tokens and only exceeds the host span above ≈3600 tokens (TP2) / ≈6600 (TP4) / ≈9200 (TP8). The TP4 "dip" is a monotonic
decline from 0.093 to 0.067 ms over 3000–7000 tokens (fine-grained print in the brief's own §4 note and here), which is the
start event being stamped progressively later as the QKV GEMM + norms grow (TP4 QKV 22.6 → 40.5 µs, norms ≈25 → 55 µs);
the chain itself is too short to keep the GPU busy, so `attn_post_proj` right after it is still starved at TP4 (idle gap
before `o_proj` 46–50 µs at every token count), which is why `attn_post_proj` shows no dip at TP4 (brief: "2 % negligible").
The two ops are therefore one mechanism, not two (brief question 7.2), and the different "trough" locations (post_proj ≈4100,
rope TP4 ≈5900) are where each scope's own host span is first exceeded by the GPU backlog in front of it (7.3).

Code-verified complication for the dataset: the fallback also computes the wrong thing. `RotaryEmbedding._compute_cos_sin_cache`
builds a `[max_pos, 128]` cache of 64 cos + 64 sin, `forward` does `cos, sin = cos_sin.chunk(2, dim=-1)` (each `[N, 64]`), and
`_apply_rotary_pos_emb` sets `rotary_dim = cos.shape[-1] = 64` and rotates `q[..., :64]` — the first half of head 0 of the
flattened `[N, heads·128]` tensor — passing the other heads through via `torch.cat`. So neither the sweep's nor X1's
`attn_rope` numbers represent vLLM's `rotary_embedding` kernel (one launch over all heads); they measure 13 launch-bound
torch kernels plus two full copies. Whether `attn_rope` should be kept in the dataset at all is a decision for the dataset
owner (cookbook gotcha 4 calls the fallback "portable", not "equivalent").

### 3.6 CONFIRMED — the same artefact contaminates far more of the dataset than the dip band (beyond the brief)

Ratio of the original median to X1's GPU-time median (median over 512-token windows), and the first window from which all
later windows are within 5 %:

| op | TP | 1024 | 2048 | 3072 | 4096 | 5120 | 6144 | 7168 | 8192 | 9216 | 10240–16384 | clean from |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `attn_pre_proj` | 1 | 1.42 | 1.25 | 1.01 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.97 | 3072 |
| `attn_pre_proj` | 2 | 2.01 | 1.50 | 1.36 | 1.24 | 1.19 | 1.03 | 0.99 | 1.00 | 0.99 | 0.98 | 6144 |
| `attn_pre_proj` | 4 | 2.73 | 1.88 | 1.59 | 1.41 | 1.33 | 1.22 | 1.13 | 1.03 | 0.99 | 0.98 | 8192 |
| `attn_pre_proj` | 8 | 2.98 | 2.42 | 1.99 | 1.68 | 1.46 | 1.29 | 1.16 | 1.00 | 0.98 | 0.98 | 7680 |
| `attn_rope` | 2 | 1.67 | 1.40 | 1.11 | 0.99 | 0.99 | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 | 3584 |
| `attn_rope` | 4 | 1.77 | 1.61 | 1.54 | 1.41 | 1.25 | 1.11 | 0.99 | 0.98 | 0.96 | 0.99 | 6656 |
| `attn_rope` | 8 | 1.58 | 1.66 | 1.66 | 1.59 | 1.53 | 1.48 | 1.39 | 1.21 | 1.00 | 0.99 | 9216 |
| `attn_post_proj` | 1 | 1.70 | 1.01 | 0.98 | 0.98 | 0.98 | 0.99 | 0.98 | 0.99 | 0.98 | 0.97 | 2048 |
| `attn_post_proj` | 2 | 2.08 | 1.86 | 1.73 | 1.23 | 1.01 | 0.98 | 0.98 | 0.99 | 0.98 | 0.97 | 5120 |
| `attn_post_proj` | 4 | 2.63 | 2.37 | 2.15 | 1.91 | 1.84 | 1.84 | 1.72 | 1.17 | 1.00 | 0.97 | 9216 |
| `attn_post_proj` | 8 | 2.96 | 2.56 | 2.39 | 2.27 | 2.09 | 2.05 | 2.00 | 1.74 | 1.52 | 0.98 | > 10240 |

(`attn_rope` TP1 is clean from 1536.) The brief's "clean negative control" `attn_pre_proj` is not clean: it is the first timed
scope of the forward, so nothing is ever queued in front of it and it sits on its host floor without a dip until its own
kernels outgrow the floor (TP8: 2.4× at 2048 tokens, 1.16× at 7168). Its slow TP8 "decline" in the brief's §4 note is the
floor being approached from above as the GEMM+norm time grows, exactly the `attn_rope` TP4 shape.

### 3.7 RULED OUT / not applicable — remaining brief candidates and questions

- Candidate 3 (wall-clock/scheduling): the E1b host forward times are flat in tokens and in pass position; the transition
  is set by the GPU work queued in the same forward (§3.2), and the per-worker onset ordering follows host speed, not
  elapsed time.
- Candidate 4 items: GC (05_) only slows single samples; the allocator is not involved (no `hipMalloc` in the traces after
  warm-up: kernel counts per forward are a constant 41 at every token count).
- Question 7.4 (other hardware): H100 (`data/profiling/compute/h100/qwen3-a3b-30b-moe/linear_op.csv`, TP1 only) and A800
  (TP 1/2/4/8, ≤4096 tokens) show no dip; A800 `attn_post_proj` shows tile-quantisation *steps up* at 1680 and 3488 tokens at
  every TP (e.g. TP2 0.071 → 0.135 → 0.201 ms). **SPECULATIVE:** those GPUs are 2–4× slower on these GEMMs, so the kernels
  exceed the host span already at small M and the floor is not visible; no host timing exists for those runs to confirm it.

## 4. Open points (none block the conclusion)

1. **SPECULATIVE:** the ≈6 µs inter-kernel gap that remains with a full queue (X1 event = kernel + 6 µs; trace idle 5.9 µs
   before every kernel) is the GPU's dispatch latency on this stack; it inflates sub-10 µs kernels by up to 2× even in X1.
   A `--profile_method kineto`/rocprof-based measurement would remove it if per-kernel numbers are wanted.
2. E1b GPU1's early low `attn_post_proj` values with a slow host (§3.2) are unexplained; not needed for the mechanism.
3. The TP8 tile bumps (2832–2928, 3088–3184 tokens) were not traced; kernel names would come from the same `trace_ops.py`
   with `--tp 8`.

## 5. What contradicts or complicates the brief

1. §1/§3 "clean, TP-dependent structural difference … `attn_post_proj`'s K changes with TP": true but causally irrelevant;
   the K-dependence tracks the dip only because a larger K makes the o_proj kernel exceed its host span sooner (TP1) and the
   preceding QKV GEMM is TP-dependent too. The traced kernel is monotonic through the dip (§3.3–3.4).
2. §3 "`attn_rope` … is a RoPE kernel": it is the torch fallback, 13 launch-bound kernels, and numerically incorrect (§3.5).
3. §5 "not a within-shape outlier … normal variance": the trough rows are bimodal with a within-task downward drift and the
   band's row std is 10–30× that of the clean rows (§3.1–3.2); the brief's std scan did not compare against the clean region.
4. §5 "not present in `attn_pre_proj` … rules out anything that would uniformly affect the whole forward": `attn_pre_proj` is
   affected, more than the two ops in the brief, just without a dip (§3.6).
5. §1 "climbs back sharply … within a few hundred tokens": the post-trough rise is the kernel growing with M plus the
   `MT160x256` tile band (4328–4840); no snap back to the floor occurs (§3.3).
6. §4 tables: every "plateau" value in them is a host span, and "recovered = plateau" at TP2 (0.0583 both) is a coincidence
   between the host span and the kernel time at ≈8000 tokens.

## 6. Recommendation for the dataset (recorded, not requested by the brief)

For TP>1 the sweep's `attn_pre_proj`/`attn_rope`/`attn_post_proj` medians below the "clean from" bounds in §3.6 (and TP1 below
2048–3072 tokens) measure host launch overhead, up to 3× the kernel time, and should not be used as GPU op costs. Re-collecting
with `FRONTIER_GPU_BACKLOG_MS=100` (one env var in `LINEAR_DOCKER_ENV`; +100 ms per task, the full 13,308-task grid then takes
≈13 min instead of 2) yields monotonic kernel-plus-dispatch-gap times; job 21356's CSV already covers 1024–10240 (step 8) and
10368–16384 (step 128) for all four TPs. `attn_rope` additionally needs a decision about the fallback (§3.5).
