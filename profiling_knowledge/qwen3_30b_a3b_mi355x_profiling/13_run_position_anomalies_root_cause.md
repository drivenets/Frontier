# Run-position anomalies in the fixed linear_op collections: root cause

Qwen3-30B-A3B on MI355X, linear_op profiler, the two 2026-09-22 dense collections (job 21483: 25 timed forwards per shape;
job 21486: 200 = 8 spun blocks of 25). Investigation run 2026-09-22 on `amd-mi355x-1` (jobs 21491, 21494, 21495; 21492 and
21493 failed at startup on a `chmod` of files owned by `nobody` and were resubmitted as 21494/21495). Labels: **CONFIRMED** /
**RULED OUT** / **SPECULATIVE** as defined in the brief. Every number is reproducible from the files listed in §3.

## 1. Executive summary

**Effect 1 ("last run of every 25-run block is the minimum").** CONFIRMED as a *within-block drift*, not a boundary event: in
every op and in the whole-forward span, samples decline monotonically by 2–8 % from run ~3 to run 25 of every spun block,
restarting after every re-armed spin; the block-end argmin is the tail of that ramp. CONFIRMED cause: the `torch.cuda._sleep`
spin used as the queue backlog. Replacing it with an equal-length chain of real GEMMs removes the drift entirely (early/late
ratio 1.079 → 1.001 at TP1/6144 tokens, flat profiles in every task), while a device-side shader-clock probe shows the shader
clock rising ≈1.5 % (2290 → 2330 MHz) over the first 10–15 forwards after a spin. SPECULATIVE: the remaining 2–5 % of the drift
at large shapes is another DPM domain (memory/fabric clock) that also idles during the spin; the probe cannot see it.

**Effect 2 (`attn_pre_proj` argmax at run 20 / run 118 at TP>1).** CONFIRMED: the ROCm runtime (ROCclr) inserts a barrier-AND
packet with system-scope acquire/release fences after every 1000 commands submitted since the last synchronize
(`HostQueue::FlushSubmissionBatch`, `DEBUG_CLR_MAX_BATCH_SIZE = 1000`). The device pays ≈5.5 µs for it, inside whichever scope
holds the 1000th packet. With 29 kernels + 22 event records per forward at TP>1 (15 + 22 at TP1), the 1000th packet of the
GPU-bound loop is the copy kernel just before k-norm in forward 20's `attn_pre_proj`; the rule predicts all ten hit positions
of the 200-rep data (20, 40, 59, 79, 98, 118, 138, 157, 177, 196) exactly, the TP1 positions (27k), and the much larger
legacy-pass stalls at runs 21/22, 42/43, … (TP>1) and 30, 58, 87, … (TP1). Runs 40, 59, 79, … land in other scopes, which is
why only 20 and 118 show in `attn_pre_proj` and why `attn_post_proj` has an unreported 40–49 % argmax at run 59.

## 2. Verification of the reported numbers (re-derived from the raw CSVs)

Files: cluster `data/profiling_dense_fixed/…/linear_op.csv` (45,122,465 B) and `data/profiling_dense200/…/linear_op.csv`
(218,776,785 B), copied by rsync; script `run_position/run_position_hist.py` (argmin/argmax position histogram per op × TP,
warm-up records dropped by `warmup_count`, ties counted separately). Baseline 4 % (25) / 0.5 % (200).

| claim in the brief | re-derived | verdict |
|---|---|---|
| 25-rep: run 25 is the argmin in 12–33 % of rows (run 24 for `forward_gpu_span`) | `attn_pre_proj` 33.2/30.5/27.2/22.4 % (TP1/2/4/8); `attn_post_proj` 24.2/22.0/17.1/16.4; `attn_rope` 21.9/14.7/11.1/7.9; `input_layernorm` 12.5; span run 24 = 24.1/24.6/21.7/20.8 %, run 25 = 23.0/20.9/17.0/14.2 % | matches, **but** run 24 = 22 %, run 23 = 15 %, run 22 = 8–10 %: a ramp, not a boundary spike (§4.1) |
| 200-rep: block ends 25, 50, …, 200 each 2–6 % | `attn_pre_proj` TP1: 6.0/5.8/4.4/5.2/4.7/4.2/5.5/4.6 %; other ops 1–5 % | matches |
| 25-rep: run 20 argmax in `attn_pre_proj` 42/54/60 % at TP2/4/8, 0 % at TP1 (max at run 3) | 42.2/53.6/59.5 %; TP1: run 20 = 0 %, run 3 = 12.6 % | matches |
| 200-rep: run 20 at 7.4/9.6/10.2 %, run 118 at 4.0/4.8/6.0 %; 45, 69, 70, 95, 120, 145, 167, 170, 195 at baseline | identical; 45/70/95/145/170/195 excess −0.6…−0.2 µs (p50) | matches |
| other ops peak at early runs only | `attn_post_proj`/`attn_rope`/span 25-rep: argmax at runs 3–5 (7–13 %) — yes; **200-rep `attn_post_proj` TP2/4/8: argmax run 59 in 40.4/47.8/48.6 %, argmin run 157 in 14.3/34.1/41.0 %; span argmax run 79 in 7–8 %** | not reported in the brief; same mechanism as Effect 2 (§4.2) |

Magnitudes (`run_position/run_position_profile.py`): the run-20 excess is +3.0/+3.7/+3.9 µs (p50, TP2/4/8; p10–p90 −0.4…+4.7 µs)
on a 32–56 µs scope, present in 73–84 % of rows below 2048 tokens and fading above 4k only because the row's noise grows; the
whole-forward span at run 20 is +1.4…+2.9 µs with no compensating deficit in `attn_rope`/`attn_post_proj`. The legacy column
(`time_stats_hostbound`) has its own, much larger positional effect: at TP>1 runs 21 and 22 are +52…62 µs and +88…97 µs (p50) in
`attn_pre_proj` and every other attention scope, in 97–100 % of rows below 4k tokens; 0–2 % at TP1.

## 3. What was checked and how

**Code read.** `frontier/profiling/linear_op/linear_op_wrapper.py` (`_timed_pass`, `_enqueue_gpu_backlog`, `_sclk_probe_mhz`:
per pass: synchronize → 3 warm-up → synchronize → probe (contains its own synchronize) → [spin] → timed forwards → synchronize
→ probe), `common/cuda_timer.py`, `common/timer_stats_store.py` (samples include the warm-up records; `warmup_count` says how
many), `linear_op_impl.py` (scope order: emb ×2, input_layernorm, attn_pre_proj = QKV GEMM + q/k RMSNorm, attn_rope,
attn_post_proj, post_attention_layernorm, then the MLP scopes, which run and are timed but whose columns main.py drops for
MoE), `main.py` (round-robin tasks, `filter_mlp_columns`), `profiling_plan.py`, `tensor_parallel_layers.py` (no collectives at
TP>1: `gather_output=False`, `reduce_results=False`). Cluster copy of the wrapper is byte-identical to the worktree
(md5 8594a978…). ROCclr source at tag rocm-7.0.0 (the image is `lmsysorg/sglang:v0.5.11-rocm700-mi35x`):
`rocclr/platform/command.cpp`, `commandqueue.hpp`, `utils/flags.hpp`, `device/rocm/rocvirtual.cpp/.hpp`.

**Data.** Both CSVs, all rows (13,308 each); per-position profiles = median over rows of sample/row-median (permille).
Backlog columns: `gpu_backlog_ms_actual/gpu_backlog_ms` = 0.969–1.031 in every row of both files, so the spin was delivered as
requested; the CSV holds one value per row, so there is no per-block backlog information in it (see §4.1 for the direct test).
`host_enqueue_per_forward_ms_backlog / host_wall_per_forward_ms_backlog` = 0.21 (25-rep, all TP): the host enqueues all 25
forwards in ≈12 ms while the device is still in the ≈52 ms spin, so nothing the host does during the loop can reach the device
timeline. In the 200-rep file the ratio is 0.50 at TP>1 (host throttled by the queue between blocks).

**Probe script** `posprobe.py` (this directory; `scripts/slurm/qwen3_posprobe{,2,3}.sbatch`): one worker's task loop in a
single process on GPU 0 via the unmodified `LinearOpWrapper.profile`, dumping every sample of both passes, with interventions
(`--mode`): `busy_backlog` (replace the `_sleep` spin by a chain of 4096² bf16 GEMMs of the same length, calibrated after 20
warm-up GEMMs), `warmup6`, `hostsleep` (0.5 ms `time.sleep` per forward), `clockprobe` (a 200k-cycle `_sleep` timed by an
event pair after every forward = device-side shader-clock sample), `--marks` (roctx range per forward). Runs, all TP8 unless
stated, node `amd-mi355x-1`, results in cluster `data/profiling_posprobe/` (jsonl per run, `rocprof/`, `amdlog_R11_tp8.log`):

| run | job | what | steps |
|---|---|---|---|
| R1/R2/R3 | 21491 | default, TP8 (10 tokens) / TP1 (10) / TP2 (3) | 50 (2 blocks) |
| R4, R5, R6 | 21491 | GEMM backlog (miscalibrated, host-bound after run 22: discarded), spin 10 ms (host-bound after run 17: discarded), spin 400 ms | 50 |
| R7, R8 | 21491 | 6 warm-up forwards; 0.5 ms host sleep per forward | 50 |
| R9, R10 | 21491 | rocprofv3 `--kernel-trace --hip-trace --marker-trace`, TP8 / TP1, tokens 64/512/2048 | 50 |
| R11 | 21491 | `AMD_LOG_LEVEL=4` runtime log, TP8, 64 tokens | 50 |
| R12 | 21491 | default, 200 steps (8 blocks), tokens 64/512/2048 | 200 |
| R13, R14 | 21495 | rocprofv3 traces at 200 steps, TP8 / TP2 | 200 |
| R15, R16, R21 | 21494 | clock probe, TP8 / TP1 (7 tokens) / TP8 | 25, 25, 50 |
| R17, R18, R22 | 21494 | GEMM backlog (corrected calibration), TP8 / TP1 / TP8 | 25, 25, 50 |
| R19, R20 | 21494 | default TP8 / TP1 (same 7 tokens) | 25 |

Analysis scripts: `run_position/posprobe_analyze.py` (profiles, spikes, early/late drift per task),
`posprobe_trace_analyze.py` (kernels and HIP calls per forward via the roctx marks; per-run device span, gaps, host stalls),
`posprobe_trace_detail.py` (kernel-by-kernel list with inter-kernel gaps for chosen runs).

## 4. Findings

### 4.1 Effect 1 — within-block drift caused by the `_sleep` backlog spin

**CONFIRMED (shape).** Median profile of sample/row-median across the 25 positions, dense 25-rep, `attn_pre_proj` (permille):
TP1 `1013 1013 1014 1014 1014 1013 1012 1009 1007 1004 1002 1000 1000 997 996 994 993 992 990 989 988 987 985 985 983`;
TP8 `1009 1009 1009 1008 1007 1006 1005 1004 1002 1001 1000 999 998 997 996 995 994 993 993 [1123] 991 990 989 989 988`.
The same monotone ramp appears in `attn_post_proj`, `attn_rope`, both norms and `forward_gpu_span`, and in each of the eight
blocks of the 200-rep file with identical shape (e.g. `attn_pre_proj` TP8 block 2: `1010 1010 … 989 988`). A ramp explains
the whole argmin histogram (run 25 > 24 > 23 > 22 …) and why the span's minimum sits at run 24 as often as 25 (the last two
positions differ by <0.2 %).

**CONFIRMED (cause = the spin).** Corrected GEMM-backlog runs vs the default spin, same tokens, 25 forwards, ratio of the median
of runs 1–5 to the median of runs 20–25 per task (tokens 6144, 4096, 2048, 1024, 256, 64, 32):

| | TP1 default (R20) | TP1 GEMM backlog (R18) | TP8 default (R19) | TP8 GEMM backlog (R17) |
|---|---|---|---|---|
| `attn_pre_proj` | 1.079 1.035 1.030 1.025 1.008 1.009 1.010 | 1.001 0.986 1.017 1.011 1.000 1.002 1.018 | 1.020 1.028 1.010 1.009 0.998 1.000 1.009 | 0.993 1.008 1.001 1.002 1.021 1.016 1.013 |
| `forward_gpu_span` | 1.064 1.030 1.020 1.018 1.007 1.004 1.003 | 1.004 1.001 1.007 0.998 1.002 1.011 1.002 | 1.020 1.000 1.002 1.008 0.996 0.991 1.018 | 1.003 1.006 1.004 1.001 1.009 1.010 1.001 |

With the GEMM backlog the position profile is flat after a 2–3 run cold-cache transient (the GEMMs evict the weights; TP1
profile `1012 1010 1005 1002 1002 1000 999 …`), i.e. the ordering is reversed from the spin case (first runs slowest for a
different, short-lived reason). The GEMM backlog delivered 85 % of the requested length (44 ms at TP8), still ≥3× the legacy
loop wall, and the host finished enqueueing in 17 ms, so the loop stayed device-bound. Spin length does not matter above
≈50 ms: 400 ms spins (R6) give the same drift as 100 ms (1.019 vs 1.025 at 4096 tokens). Prediction stated before running R17/R18:
"if the drift disappears with a work backlog, the spin's low-activity state is the cause; if it persists, it is not." It disappeared.

**CONFIRMED (shader clock rises after the spin, partially).** Device-side probe after every timed forward (R15, TP8, 25
forwards; MHz): 6144 tokens `2293 2280 2292 2293 2305 2309 2313 2315 2321 2319 2321 2319 2323 2323 2326 …`; 2048 tokens
`2301 2299 2311 2313 … 2333 2327`; 64 tokens `2297 2302 2299 2295 2311 … 2302` (flat). TP1 (R16) 1024 tokens `2288 2293 2304
2304 2306 2312 2310 2315 2318 … 2326`. The clock climbs 1.3–1.7 % over the first 10–15 forwards after a spin wherever the
drift exists, and is flat where the drift is absent (TP8 ≤ 256 tokens). The drift itself is 2–8 %, largest where the device
time per forward is largest (TP1 6144 tokens: 7.9 %), so the shader clock accounts for roughly a third of it.

**SPECULATIVE.** The remainder is consistent with another activity-managed clock domain (memory or fabric clock) sitting at an
idle DPM level during the single-wave spin and ramping under load; `_sleep` cannot probe it. A sysfs/SMI clock sampler at
≥1 kHz during a block would decide this; not run.

**RULED OUT.** A boundary-specific event at run 25 (no step in any profile; run 24 ≈ run 25); the backlog spin being cut short
(actual/requested 0.97–1.03 on every row); a timestamp artefact (the span, which contains all scopes, drifts by the same
fraction, and rocprofv3 kernel durations follow it); GC (disabled in the loop); host effects (all packets are queued 40 ms
before the device executes them in the 25-rep regime).

### 4.2 Effect 2 — ROCclr's 1000-command batch-flush barrier

**CONFIRMED at kernel level (rocprofv3, R9, TP8, 64 tokens, GPU-bound timed runs 19/20/21, kernel-by-kernel).** All 29
kernels have identical durations in the three runs (QKV GEMM 7.7–7.9 µs, RMSNorm 4.1–4.8 µs). The only difference is one
inter-kernel gap inside `attn_pre_proj`:

```
run 19  k20 gap_before= 0.0 dur=4.6  elementwise_kernel_manual_unroll<…direct_copy…>
        k21 gap_before= 0.0 dur=4.1  vllm::rms_norm_kernel<BFloat16>          (k-norm)
run 20  k20 gap_before= 0.0 dur=4.1
        k21 gap_before= 5.5 dur=4.0  vllm::rms_norm_kernel<BFloat16>          <-- 5.5 us idle, nowhere else
run 21  k20 gap_before= 0.0 dur=4.6
        k21 gap_before= 0.0 dur=4.1
```
The 200-step trace (R13) shows the same for the other hits: run 59 has a 26.6 µs gap before its first kernel (18.8 µs in
runs 58 and 60); run 157's gap falls between a kernel and an event record. Per forward the trace counts 29 kernels
(TP8; TP1: 15) and 20 `hipEventRecord` (legacy pass) or 22 (GPU-bound pass, span pair added), i.e. 51/49 packets per forward at
TP8 and 37/35 at TP1.

**CONFIRMED in the runtime log (R11, `AMD_LOG_LEVEL=4`, 202,825 lines).** Counting every AQL packet the runtime logs
(`Dispatch Header`, `BarrierValue Header` = event records, `BarrierAND Header`) since the last `hipDeviceSynchronize`:

```
legacy loop   : BarrierAND after 1003 packets (597 kernels + 405 records), again after 2008
GPU-bound loop: BarrierAND after 1003 packets (573 kernels + 428 records), again after 2005
```
Each of these four barriers is issued *inside a `hipLaunchKernel` call*, after the kernel's own dispatch packet, and is not
one of the 13 synchronize barriers or the 3 kernarg-chunk barriers ("Issue barrier to flush chunk N", a separate mechanism,
§6). The GPU-bound one at 1003 packets follows the `direct_copy` kernel (k20) — exactly the packet before k-norm where the trace
shows the gap. Log excerpt (line 136610 ±):
```
:3:rocvirtual.cpp :3344 ShaderName : void at::native::elementwise_kernel_manual_unroll<…direct_copy…>
:4:rocvirtual.cpp :1076 SWq=…, HWq=…, Dispatch Header = 0xb02 (type=2, barrier=1, acquire=1, release=1) …
:4:command.cpp    :357  Command (InternalMarker) enqueued: 0x5bbf32719c80 to queue: 0x5bbf2a895da0
:3:rocvirtual.cpp :551  Set Handler: handle(0x784deabfeb80), timestamp(0x5bbf3271e4e0)
:4:rocvirtual.cpp :1260 SWq=…, HWq=…, BarrierAND Header = 0x1503 (type=3, barrier=1, acquire=2, release=2) …
:3:hip_module.cpp :813  hipLaunchKernel: Returned hipSuccess : : duration: 22 us
```
`acquire=2, release=2` = system-scope fences (L2 writeback + invalidate), which is what costs the device ≈5.5 µs.

**CONFIRMED in the ROCclr source (ROCm/clr, tag rocm-7.0.0).** `rocclr/platform/command.cpp`, `Command::enqueue`
(direct-dispatch path): every command is appended to the host queue's submission batch (`FormSubmissionBatch`, `size_++`);
profiling markers such as `hipEventRecord` do not reset it ("Enqueue flushes, except profiling markers to avoid frequent
expensive callbacks"); any other command then calls `queue_->FlushSubmissionBatch(this)`:
```cpp
  //! Flushes submitted commands if the batch size significantly grew
  void FlushSubmissionBatch(Command* command) {
    if (size_ > DEBUG_CLR_MAX_BATCH_SIZE) {      // rocclr/platform/commandqueue.hpp
      command->notifyCmdQueue();                 // -> new amd::Marker(...) with a completion callback
    }
  }
```
`release(uint, DEBUG_CLR_MAX_BATCH_SIZE, 1000, "Forces the callback to clean-up CPU submission queue")` (`utils/flags.hpp`).
The marker is a `CL_COMMAND_MARKER`, which takes the branch that calls `ResetSubmissionBatch()` (size back to 0) and is
dispatched by `VirtualGPU::dispatchBarrierPacket(kBarrierPacketHeader …)` with
`kBarrierPacketHeader = BARRIER_AND | BARRIER | SYSTEM acquire | SYSTEM release` (`rocvirtual.cpp:84`). A synchronize is a marker
too, so the count restarts at every `torch.cuda.synchronize()`; the last one before each timed loop is inside the clock probe.

**CONFIRMED by pre-stated predictions.** (i) Position arithmetic: after the probe's synchronize the GPU-bound loop starts
with the 3 spin packets, then 51 per forward, so packet 1000 is the 28th packet of forward 20 = the copy kernel before k-norm
(forward layout: span record, emb 1+8+1 ×2, input_layernorm 1+1+1, `attn_pre_proj` record + GEMM + copy + q-norm + copy | k-norm).
(ii) Applying "one barrier per 1000 packets, restart at synchronize, spins count 3" to the 200-rep layout predicts
GPU-bound TP8 hits at runs **20, 40, 59, 79, 98, 118, 138, 157, 177, 196** — the observed span-spike positions are exactly
these ten; TP1 (37 packets/forward) predicts 27, 54, 81, 108, 135, 162, 189 — observed 81, 108, 135, 162, 189 (27 and 54 fall
in the block-start ramp); legacy TP8 (49/forward) predicts 21, 41, 62, 82, 103, 123, 143, 164, 184 vs observed 21/22, 42/43,
62/63, 83/84, 103/104, 124/125, 144/145, 164/165, 185/186; legacy TP1 (35/forward) predicts 29, 58, 86, 115, 143, 172 vs
observed 30/31, 58–60, 87/88, 116/117, 144/145, 173/174. (iii) Falsification tests stated in advance: adding 3 warm-up forwards
(R7) must not move the hit if the count restarts at the post-warm-up synchronize (it stayed at run 20; a per-pass count would
have moved it to 17); adding 0.5 ms of host time per forward (R8, host wall 0.52 → 1.08 ms) must not move it if the mechanism is a
count (it stayed at 20; a 12-ms timer would have moved it to ~10); replacing the 1-kernel spin by a ≈349-packet GEMM chain (R17/R22)
must move the first hit earlier by 349/51 ≈ 7 forwards (observed: span spike at run 12–13, `attn_pre_proj` no longer hit).
(iv) The single-process 200-step run (R12) reproduces `attn_pre_proj` +4…+8 µs at runs 20 and 118 and `attn_post_proj` +5 µs at
run 59 in every task, with `forward_gpu_span` +4…+14 µs at all four predicted runs.

**Why TP>1 only, and why `attn_pre_proj`.** TP>1 forwards carry 14 more kernels than TP1 (the `.split()`/`.view()` copies and
two `hipMemcpyAsync` blits of the sharded QKV path: `__amd_rocclr_copyBuffer` ×2, `vectorized_elementwise_kernel<4>` ×8,
`elementwise_kernel_manual_unroll` ×2, `vectorized_elementwise_kernel<16>` ×2, identical at TP2/4/8), so packet 1000 lands
in forward 20 at all TP>1 and in forward 27 at TP1 — outside a 25-run loop, hence "0 % at TP1" in the 25-rep file. Which scope
is hit depends only on the packet layout, not on shape, so the same scope is hit in every row.

**SPECULATIVE.** The 200-rep `attn_post_proj` *minimum* at run 157 (−0.4…−0.6 µs) is consistent with the barrier landing
immediately before that scope's start record (R13 trace: the gap at 157 sits between a kernel and an event record), whose own
fence then finds clean caches; not measured directly. The legacy-pass stall size (+50…100 µs per scope over two forwards vs the
5.5 µs device fence) is the host-side cost of the marker's completion callback in a host-bound loop; the R9 trace shows the
affected forward 350 µs longer with the time spread over many HIP calls rather than one long call. The mechanism is fully
identified; only this host-side breakdown is not.

**RULED OUT.** Queue backpressure (host finishes 12 ms into a 52 ms spin; no throttling in the 25-rep regime); a time-periodic
host or device event (R8); a per-pass packet counter (R7); a different kernel or tile at run 20 (identical kernel names and
durations); the kernarg chunk flush (`rocvirtual.cpp:1642`, period 977 kernels, not reset by synchronize, positions drift
across passes: 856/1450 kernels into the legacy loop, 287 and 1264 into the GPU-bound loop in R11); GC (disabled).

## 5. Consequences for the dataset and next steps

- **Medians are unaffected by Effect 2** (one sample in ~20 shifted by ≈5 µs) and only marginally by Effect 1 (the median is
  the mid-ramp value). `min`, `max`, `mean` and `std` carry both effects: `max` at TP>1 is the barrier sample in 40–60 % of rows,
  `min` is the last-of-block sample. Anything that reads per-run samples (the run-group notebook) sees them.
- **Effect 1 is a clock-condition problem**: the contract's "kernel time at the automatic boost clock" is the *late-block* value;
  the first ~10 forwards after each spin run 1.5–8 % slow. Fix options, cheapest first: (a) keep the spin but add 5–10 untimed
  forwards after it before the timed block (prediction: profile flat from run 1; TP8 ≤256 tokens unchanged); (b) replace the
  spin with a work backlog such as the GEMM chain in `posprobe.py --mode busy_backlog` plus 3 untimed forwards to refill caches.
  Either can be validated with `posprobe.py --mode clockprobe` (expect a flat probe series).
- **Effect 2 fix options**: set `DEBUG_CLR_MAX_BATCH_SIZE` (an env flag of the runtime) to a value larger than the packets per
  block (e.g. 100000) in `LINEAR_DOCKER_ENV`; prediction: the run-20/118 hits and the legacy runs-21/22 stall vanish in
  `posprobe.py` output and the runtime log shows no `InternalMarker` inside `hipLaunchKernel`. Not yet run. Alternatively keep
  blocks of ≤19 forwards (1000/51) so the barrier never falls inside a timed block at TP>1 — TP1 (27) is already safe at 25.
  Or accept and document it as part of the ≈3 µs/packet instrument term, since production engines hit the same barrier.
- **Unresolved, with the experiment that would settle it**: the non-shader-clock part of the drift (sample mclk/fclk via
  `/sys/class/drm/card*/device/pp_dpm_mclk|pp_dpm_fclk` at ≥1 kHz during one task, or lock the memory clock and rerun R15;
  if the drift persists at locked mclk/fclk, the shader-clock probe under-reads and a second probe kernel type is needed).

## 6. What contradicts or complicates the brief

1. Effect 1 is not "the last run is the minimum" but a smooth 2–8 % ramp over the whole block; runs 22–24 are elevated in
   the argmin histogram almost as much as 25, and the legacy pass shows a milder ramp too.
2. Effects 1 and 2 are indeed separate mechanisms, but Effect 2 is not specific to `attn_pre_proj` or to runs 20/118: it hits
   every ≈19.6 forwards at TP>1 (`attn_post_proj` at run 59 is the largest instance, 40–49 % of rows), every ≈27 at TP1
   (runs 27k, so it *is* present at TP1, first at run 27 > 25), and it is the same event as the legacy column's +50…100 µs
   stalls at runs 21/22 — the prompt's "separate, already-understood warm-up-adjacent" TP1 argmax at run 3 is part of Effect 1.
3. The backlog/backpressure story does not apply to the 25-rep regime at all (host finishes at 21 % of the loop wall); in the
   200-rep run the host *is* throttled between blocks (50 %), which is worth keeping in mind for whole-forward quantities.
4. The kernarg-pool chunk barrier (`Issue barrier to flush chunk`, every ≈977 kernels, not synchronize-aligned) is a second,
   position-drifting source of ≈5 µs single-sample bumps and is the likely origin of the small scattered spikes in the
   single-process profiles; it does not produce fixed positions and was not the cause of either effect.
5. The MLP scopes are executed and timed in the MoE runs (their columns are dropped by `filter_mlp_columns`), which is why the
   per-forward packet counts above include them; the published columns understate what the loop actually enqueues.

## 7. Addendum (2026-09-22, later the same day): Effect-2 fix applied and verified

`DEBUG_CLR_MAX_BATCH_SIZE=1000000` in the container removes the batch-flush barrier from the timed loops. Validation job 21499
(`scripts/slurm/qwen3_posprobe4.sbatch`): run-20 sample 997 permille vs 1132 in the same-job control, no legacy stall at runs
21/22, and the runtime log shows only synchronize and kernarg-chunk barriers. A full 25-rep dense collection with the flag
(job 21500, cluster `data/profiling_dense_fixed_batchflush/`, README there and in `run_position/README_dense_fixed_batchflush.md`):
`attn_pre_proj` argmax at run 20 drops from 42–60 % to baseline, the legacy runs-21/22 stall disappears, medians agree with job
21483 to 1.000 (p1–p99 0.96–1.05), and Effect 1 is unchanged. Not staged into the canonical dataset.

## 8. Addendum (2026-09-22, later): Effect-1 fix — what did not work, what did

All runs single-process on GPU 0 of `amd-mi355x-1`, `posprobe.py` default mode (the unmodified profiler loop), 25 timed
forwards, 7 token counts, with the Effect-2 flag on. "early/late" = median of runs 1–5 over median of runs 20–25 of
`attn_pre_proj`, listed for tokens 6144, 4096, 2048, 1024, 256, 64, 32.

| variant (job) | TP8 early/late | TP1 early/late | verdict |
|---|---|---|---|
| spin, no settle (controls in 21513/21515/21517) | 1.026 1.017 1.012 1.017 1.003 0.998 0.989 | 1.061 1.036 1.027 1.023 1.009 1.011 1.005 | the drift |
| spin + 10 untimed settle forwards (21513, `scripts/slurm/qwen3_posprobe5.sbatch`) | 1.025 1.018 1.013 1.012 1.003 0.997 1.027 | 1.066 1.085 1.027 1.060 1.012 1.003 1.002 | **RULED OUT**: unchanged. Clock probe: shader clock stays at 2286–2295 MHz through all 10 settle forwards and only starts rising at the last one; at TP1/6144 it *dips* to 2103–2155 MHz when load begins; and at TP1 the samples keep falling 5–6 % while the shader clock is already flat at 2327–2337, so a slower domain is involved. A settle long enough for that (tens of ms) would exceed the queue capacity at small shapes |
| dense GEMM chain into a preallocated output (`gemm`) + 3 settle (21515) | 0.874 1.210 1.020 1.003 0.999 0.899 0.910 | 1.173 1.180 0.906 0.987 0.964 1.005 1.297 | heavy shapes flat and at the settled value; small shapes noisy (27–49 µs samples around a 22.7 µs kernel), clock 1687–1909 MHz right after the chain. **Not reproduced** in 21517 (`gemm`, settle 0: 1.028 1.001 1.004 1.037 1.003 1.018 1.011, no noise): job/thermal-state dependent |
| memory-streaming `add` over 1 GB operands (`stream`) + 3 settle (21517) | 1.028 1.011 1.014 1.006 0.992 0.987 0.998 | 1.032 0.998 1.025 1.018 1.011 0.993 0.994 | flat at TP8 but small kernels 4–5 % slower than any spin control (23.8 vs 22.7 µs) and TP1 heavy shapes still drift 3.5 % with the shader clock flat |
| **allocating GEMM chain (`gemm_alloc`) + 3 settle** (21517, 21518; prototype 21494) | 0.998 0.996 0.995 1.001 1.003 0.995 1.024 | 0.999 0.985 1.006 1.010 0.997 1.001 0.997 | **chosen**: flat profiles (997–1005 permille at every position), shader clock 2330–2340 MHz from run 1, medians at the settled value (TP1/6144: 229 µs vs 240–268 with the spin; TP8/64: 22.6 vs 22.7), 200-step run flat |

Code: `linear_op_wrapper.BACKLOG_KIND` (env `FRONTIER_GPU_BACKLOG_KIND`, default `gemm_alloc`; `sleep`, `gemm`, `stream`
selectable), `SETTLE_STEPS` (env `FRONTIER_LINEAR_SETTLE_STEPS`, default 3) with `TimerStatsStore.pause()/resume()` so the
settle forwards record no events and no samples; new CSV columns `settle_steps`, `backlog_kind`. Why `gemm` and `gemm_alloc`
differed in job 21515 but not in 21517 is not understood (SPECULATIVE: power-management state after a full-power GEMM burst
depends on the GPU's recent history); the allocating variant was flat in all three jobs it ran in.

Dense collections with both fixes: `data/profiling_dense_fixed_workbacklog/` (job 21519; README there and in
`run_position/README_dense_fixed_workbacklog.md`). Abandoned on the way: `data/profiling_dense_fixed_settle/` (job 21514,
cancelled) and `data/profiling_dense_fixed_gemmbacklog/` (job 21516, cancelled), each with a README saying why.

**Dense collection with the chosen fix (job 21519, `data/profiling_dense_fixed_workbacklog/`, 15 min, `sanity_check.py` PASS
on every gate).** Effect 2 absent. Effect 1 halved, not removed: `attn_pre_proj` argmin at run 25 falls from 33/31/27/22 %
(TP1/2/4/8) to 17/13/9/6 %; the TP1 median profile ramps 1012 → 994 permille instead of 1013 → 983, TP8 is flat after a 1 %
transient in runs 1–4; early/late 1.028/1.026/1.020/1.016 → 1.014/1.010/1.007/1.005. Medians are 2–3 % below job 21483 (the
settled clock) and the per-row spread is 40 % smaller (relative MAD 0.98 → 0.59 % at TP1). The single-GPU validation was flat
where the 8-GPU dense run still shows the runs-1–4 transient and a slow 0.7 % TP1 decline — SPECULATIVE: clock/thermal
settling under full-node load is slower than on an otherwise idle node. Next step if it matters: `FRONTIER_LINEAR_SETTLE_STEPS=8`
for the transient; a clock probe under full-node load for the residual.

**Settle 8 (job 21539, 2026-09-23, `data/profiling_dense_fixed_workbacklog_settle8/`, `sanity_check.py` PASS).** Raising the
settle from 3 to 8 untimed forwards removes the runs-1–4 transient at TP4/TP8 (early/late 1.002/1.000, position profile flat to
±0.1 % from run 1) and halves it at TP1/TP2 (1.011/1.007). Medians are identical to job 21519 (p50 ratio 1.000–1.002). The
residual at TP1/TP2 is a slow linear 0.9–1.3 % decline over the whole block, independent of the settle count and absent in
single-GPU probes: SPECULATIVE, clock/thermal settling under full-node load on a timescale longer than a block. The profiler
default is now `SETTLE_STEPS = 8`.
