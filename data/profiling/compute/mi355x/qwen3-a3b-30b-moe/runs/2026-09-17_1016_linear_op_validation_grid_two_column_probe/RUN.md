# Run 2026-09-17 10:16 — linear ops, validation grid + 512 tokens, two-column timing **with the concurrent clock probe**

Job 21400, `amd-mi355x-7`, 57 s, 8 workers. Image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`. Code `d4469be` + the clock-probe follow-up
(`sclk_mhz_{legacy,backlog}_{start,end}`, `host_enqueue_per_forward_ms{,_backlog}`). Grid `[1,8,64,512,3072,4096,4192,6144,8192]` × TP {1,2,4,8} = 36 rows.
Same two-pass method as `runs/2026-09-17_0846_…_two_column/` (read that RUN.md first); this run adds the per-row clock estimate the
follow-up review made mandatory, so it supersedes the 08:46 run as the reference validation-grid file. It replaces an earlier collection of
the same grid (job 21397, 10:10) whose start probe sat *before* the warm-up forwards; the committed placement is immediately before the timed
loop (after warm-up + synchronize + `mark_warmup_end`, before the backlog spin) and immediately after the trailing synchronize.

Verification: `sanity_check.py <this dir> --tokens-grid "[1,8,64,512,3072,4096,4192,6144,8192]"` → PASS (36 rows, both gates run, SCLK note printed);
`test_measurement_validity.py <this>/linear_op.csv --expect green` → PASS (T1′/T2 as revised in `09_…results.md` §1).

## Clock state, measured per row (fixed 2M-cycle `torch.cuda._sleep` timed by an event pair on the idle device)
| probe | median MHz | min–max |
|---|---|---|
| legacy pass, start of timed loop (after warm-up) | **798** | 648–2413 |
| legacy pass, end | 2409 | 2325–2424 |
| GPU-bound pass, start of timed loop | 2387 | 2243–2423 |
| GPU-bound pass, end | 2409 | 2368–2424 |

Legacy start reads 648–948 MHz on 32/36 rows; the four 1-token rows (2381–2413 MHz) are the exception — they follow another task's
activity on the same worker without an idle gap. Reading: **three warm-up forwards plus a synchronize do not bring the device up from its
idle DPM clock**; the legacy pass runs its first timed forwards at ≈⅓ of the clock and ramps to ≈2.4 GHz during the 50-forward loop, while the
GPU-bound pass, which follows it immediately, runs at ≈2.4 GHz throughout. The two columns therefore differ by clock trajectory as well as by
queue state (the 10:10 collection, probe before warm-up, read 803–983 MHz on the same 32 rows — the warm-up changed nothing). Unit check:
with the shader clock locked at 1900 MHz (job 21395, `rocm-smi --setperfdeterminism 1900`) the same probe reads 1890–1916 MHz.

## Host enqueue vs wall (F6 backpressure test)
GPU-bound pass: the host finished enqueueing at 97–100 % of the loop wall at 8 tokens, 94–100 % at 64/512, 86–96 % at ≥ 3072 — the host is
throttled by a full HIP command queue for most of the spin and, at the fastest forwards, the device catches it before the loop ends
(closure 0.74–0.91 at 8 tokens for TP 2/4/8 — 0.78/0.91/0.74 —, 0.95–0.99 elsewhere). Legacy pass: enqueue == wall on every row up to 3072 tokens (host-bound); 0.57–0.93 at ≥ 4096 tokens where the device is the bottleneck. See
`09_measurement_validation_results.md` §3 for the arithmetic and the 25/50/100-forward test.

## Limitations
As for the 08:46 run: `attn_rope` is the numerically wrong torch fallback; kernels ≤ 10 µs are not cross-validated; the legacy column is
host-bound below ~5k tokens at TP>1 (that is what it documents). The 8-token rows' GPU-bound column is affected by the backpressure
effect at the end of the 50-forward loop (span closure 0.74–0.91); per-op medians there still sit within +0.3…+3.9 µs of the traced kernels.
