# Run 2026-09-22 08:49 — linear ops, dense 3,327-token grid × TP {1,2,4,8}, 200 forwards (8 spun blocks of 25), all measurement fixes

**The CSV is not in git.** Data on the shared filesystem (convention: datasets live on `/opt/shared`, repositories carry pointers):
```
/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-22_0849_dense_200fwd_all_fixes/
    linear_op.csv   sha256 73fb3e8f6295f498a0b922c823033d080b957af84b5a6ed6ea9145e4f3f2f143   (13,308 rows, 209 MB)
    README.md       schema, caveats, verification commands, references to 05_-12_
    SHA256SUMS, collection.log, slurm-21486.out
```
Job 21486, `amd-mi355x-1`, 19 min, 8 workers, automatic clock (≈2.4 GHz on all but 35 rows, per-row probe). Image
`lmsysorg/sglang:v0.5.11-rocm700-mi35x`. Code `4b8fce3` (two-column timing, spun blocks of 25, clock probe, GC guard, fused RoPE kernel).
`FRONTIER_LINEAR_ACTIVE_STEPS=200`, `LINEAR_PROFILE_METHODS=cuda_event`.

**Supersedes the root `linear_op.csv`** (2026-09-15, jobs 21313/21334) as the file to train on: that file's TP>1 rows below ~5k tokens
are host launch spans (1.3–2.9× too high for the GEMMs, 4–12× for RoPE) and its `attn_rope` timed the wrong computation. The root file
is kept unchanged as the record of the pre-fix method.

Verification (2026-09-22, on the shared copy): `sanity_check.py <dir>` PASS with the 1.10 GPU-bound-over-legacy gate (raised from 1.05
this day, reason in `sanity_check.py` and `12_` §5); `test_measurement_validity.py <dir>/linear_op.csv --expect green` GREEN
(T1′ 2.5–4.4, max GPU-bound/legacy 1.094, T2 4.7/3.0). Whole-forward closure 0.96–0.99 on every row. No sample above 10× its row's
median in any op. Agrees row for row with the 25-forward collection of the same day (job 21483, scratch tree `data/profiling_dense_fixed`
on the cluster checkout; not staged).

Read `12_dip_investigation_summary.md` first; the README next to the data lists every document.
