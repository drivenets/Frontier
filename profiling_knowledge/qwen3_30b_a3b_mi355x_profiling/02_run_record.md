# Run record — Qwen3-30B-A3B MI355X profiling (2026-09-15)

Code: worktree `/home/dn/Frontier-qwen3-profiling`, branch `smatar/qwen3-30b-mi355-profiling`, commits `deef568` (Steps 1+1b),
`0fcac5f` (Steps 2–4) plus the sbatch fixes below (uncommitted at the time of the run, labelled `0fcac5f+sbatch-fix5` in the job
banners). GitHub push blocked: `smatar-dn` has no write access to `drivenets/Frontier`; the cluster got the code by rsync to
`/opt/shared/frontier-qwen3-profiling/Frontier` (owner `dn`).

## Cluster facts (Memphis, Slurm cluster `xai`, partition `XAI`, nodes `amd-mi355x-[1-9]`, 8× MI355X gfx950 each)
- `ssh cluster` = `amd-mi355x-1` (login user `matars`; jobs submitted with `sudo -u dn sbatch`).
- `/opt/shared` is the only NFS path shared across nodes and it is exported with **root_squash**: a container running as root
  writes as `nobody`, so every output directory must be pre-created and `chmod a+rwX` before `docker run` (pilot 21280 died on the
  first `mkdir`). Files the container writes come back owned `nobody:nogroup`, world-readable.
- Docker images are per node and not everywhere: `lmsysorg/sglang:v0.5.11-rocm700-mi35x` on nodes 1, 4, 8, 9;
  `lmsysorg/sglang:v0.5.17-rocm720-mi35x` on nodes 1, 8, 9; node 6 has neither. Jobs must be pinned with `-w`.
- `/mnt/data/aiter-cache` exists root-owned on some nodes and is not writable by `dn`; the sbatch falls back to
  `/tmp/aiter-cache-dn` (per node). The aiter decode template compiles in ~12 s; the prefill template (~25 s) is built inside the
  image's own `jit/build` dir and is not cached across jobs.
- 13 GPU nodes in total (9 XAI + 4 TEST); all were allocated most of the day, so idle-node pinning (8, 9) was what made the run possible.

## The aiter bug that changed the image pin
Plan pinned `v0.5.11-rocm700-mi35x` (aiter commit a6bb49937, 2026-04-29). Its `mha_batch_prefill` faults for this model:
`Memory access fault by GPU ... HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on the second chunked-prefill shape
(chunk 80, KV 80, batch 1, TP 1) with 4 workers (job 21300) and with 1 worker (21303); the same in a 13-point repro on one GPU
(21302), at block 16 and at block 1 (21304). TORCH_SDPA runs the same grid clean, and the identical shapes exist in the gpt-oss
AITER data (head_dim 64), so the fault is specific to head_dim 128 with paged KV in that aiter version.
`v0.5.17-rocm720-mi35x` (torch 2.9.1+rocm7.2.0) runs the same grid clean at both block sizes (21304, 21305). Attention now uses
v0.5.17. v0.5.17 does **not** ship vllm, which the linear_op profiler imports, so linear_op uses v0.5.11 (`LINEAR_IMAGE`); it never
calls the faulty kernel. Kernel signatures used by the wrapper are unchanged between the two images.

## Jobs
| Job | Stage | Node | Image | Result |
|---|---|---|---|---|
| 21280 | pilot | 4 | v0.5.11 | died on mkdir (root_squash), exit 0 (bug: pilot had no CSV check) |
| 21283 | pilot | 6 | v0.5.11 | `mkdir /mnt/data/aiter-cache` denied |
| 21284 | pilot | 6 | v0.5.11 | image not on node |
| 21300 | pilot (4 workers) | 8 | v0.5.11 | GPU memory fault at attempt 24/440 |
| 21303 | pilot (1 worker) | 8 | v0.5.11 | GPU memory fault at attempt 18/440 |
| 21302 | repro (chunk 80 grid) | 9 | v0.5.11 | AITER fault at 2/13; TORCH_SDPA 13/13 |
| 21304 | repro | 9 | v0.5.11 block 1: fault; v0.5.17 block 16 and block 1: 13/13 | |
| 21305 | pilot (4 workers) | 9 | v0.5.17 | **PASS**, 440 attempts, 1:06 wall |
| 21309 | linear_op | 9 | v0.5.11 | **PASS**, 1:00 wall |
| 21310 | attention block 16 | 8 | v0.5.17 | **PASS**, 3:42 wall |
| 21311 | attention block 1 | 9 | v0.5.17 | **PASS**, 3:47 wall |

Settings: AITER backend, BF16, cuda_event, TP 1 2 4 8, `max_seq_len = max_model_len = 16384`, batch list 1…512 (18), decode KV
128…16384 (7), chunked-prefill grid search, true-mixed on (prefill bs {1,2} × chunk {1024,4096,8192} × decode grids),
3 warm-up + 50 timed runs per shape, `AITER_NUM_GPUS=4`; linear_op `--is_moe --max_tokens 16384`, 386 token values (4000 excluded).

## Dataset (`data/profiling/compute/mi355x/qwen3-a3b-30b-moe/`)
```
attention_aiter_block1                      11076 rows   29.8 MB
attention_aiter_block16                     11076 rows   29.8 MB
attention_true_mixed_aiter_block1            1440 rows    4.1 MB
attention_true_mixed_aiter_block16           1440 rows    4.1 MB
attention_combined_aiter_block1             12516 rows   34.1 MB
attention_combined_aiter_block16            12516 rows   34.2 MB
attention                                   22152 rows   59.6 MB
attention_true_mixed                         2880 rows    8.2 MB
attention_combined                          25032 rows   68.4 MB
linear_op                                    1544 rows    3.2 MB
```
Legacy TORCH_SDPA trio renamed `*_sdpa_block16.csv`; legacy `linear_op.csv` → `linear_op_maxtokens4096.csv`; `moe.csv` untouched.
Canonical trio = per-cell block-1 + block-16 files concatenated with `float_precision="round_trip"`.

## sanity_check.py — PASS
- Schema incl. `head_dim == 128`, AITER, block {1,16}, TP {1,2,4,8}, `max_model_len` 16384, `count == 50`, 53 samples,
  `warmup_count == 3` per scope; canonical == per-cell and combined == attention + true_mixed by row hash; no symlinks.
- Grid coverage: true-mixed 360/360 at every (block, TP). Standard 2772 attempts per (block, TP) except TP1 (2763) and TP2 (2769):
  9 and 3 decode shapes dropped by the profiler's memory filter, exactly the expected TP-1/2 pattern (96 / 48 KB per token).
- Repeated shapes (2176 groups per block): relative spread median 0.2 %, max 19.7 %.
- Where the slowest of the 53 runs sits (0–2 = warm-up): `attn_prefill` run 0 in 19,422 of ~22k rows, run 3 (first timed run) in
  1,504, otherwise flat with a bump at run 49 (137); `attn_kv_cache_save` run 0 in 11,856, run 3 in 9,299; `attn_decode` nearly
  flat (run 0: 1,237 of ~22k; runs 48–49 elevated: 630, 779). Reading: warm-up is real and 3 warm-ups cover it for the kernels;
  the first *timed* run after the synchronize is often the slowest for the cheap scopes; decode outliers are noise-like.
- max/median over the 50 timed runs: prefill p50 1.09 p95 1.54 (one extreme outlier, max 1117×); decode p50 1.44 p95 1.96 max 38.6;
  kv_cache_save p50 1.70 p95 3.73 max 86.8.
- Decode median non-decreasing in KV for 42 % and in batch for 7 % of series: at these sizes decode is microseconds and noise-dominated;
  informational only. block1/block16 decode ratio at the same shape p50 0.98 (p5 0.93, p95 1.06).
- True-mixed vs even `attn_decode` median ratio −61 %: the two scopes are not like-for-like (known; informational).

## Open items
- Push to GitHub needs write access for `smatar-dn` (or a push from an account that has it).
- The sbatch fixes (root_squash dirs, aiter-cache fallback, per-stage images, pilot CSV check) and this record are uncommitted.
- Decide whether block 16 stays in the canonical trio: SGLang runs the AITER prefill kernel at page size 1 only, so block 16 is a
  kernel path production never exercises (the v0.5.17 kernel handled it, but it is not validated against sglang output).
- Cold-start timing was never measured (every shape runs warm-up first in a hot process); separate experiment if wanted.
