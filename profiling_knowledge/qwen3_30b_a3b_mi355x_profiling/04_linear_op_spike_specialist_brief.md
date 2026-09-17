# Brief for a specialist: the deterministic ~200 ms stall in `attn_pre_proj` at timed run 11, tokens 1968–1975

Prepared 2026-09-16. Goal: hand a ROCm / PyTorch / hipBLASLt specialist everything needed to explain **why** this happens.
We do not need the data fixed (medians are unaffected); we need the root cause.

## 1. One-paragraph summary

We profile Qwen3-30B-A3B's attention-layer linear ops on an 8× MI355X node with Frontier's `linear_op` profiler: for each
`num_tokens` in a 3,327-value grid and each TP ∈ {1,2,4,8}, a fresh model is built on one GPU and run 3 warm-up + 50 timed
forwards; every op is timed with HIP events and all 53 per-run times are stored. In the QKV-projection scope (`attn_pre_proj`,
median ≈ 0.13 ms at ~2k tokens) **exactly one** timed run — index 11 — takes **150–290 ms** (≈1,000–2,000× the median) for
**exactly eight consecutive token counts, 1968–1975, at every TP**. A full re-run on a different node 17 h later reproduced it
to the token and to the run index. A 26-token repro around the same tokens did not. The eight tokens are the eight tasks at
per-GPU-worker position 169 of the sweep, one per worker, and each TP pass starts fresh worker processes that replay the same
history, so the trigger is a deterministic, history-dependent in-process event rather than a shape or environmental effect.
We do not know what the event is.

## 2. Environment

| | |
|---|---|
| Node | Memphis Slurm cluster, partition `XAI`; job 21313 on `amd-mi355x-9`, job 21334 on `amd-mi355x-8`; 8× AMD MI355X (gfx950), 288 GB HBM each |
| Container | `lmsysorg/sglang:v0.5.11-rocm700-mi35x`; ROCm 7.0.0; Python 3.10.12; torch `2.9.0a0+git7bcbafe` (HIP 7.0.51831); triton 3.4.0; vllm `0.9.2rc2.dev2065+g4f43dae12.rocm700`; hipBLASLt `libhipblaslt.so.1.0.70000`; `torch.backends.cuda.preferred_blas_library() == Cublaslt` (i.e. hipBLASLt) |
| Image env | `HIP_FORCE_DEV_KERNARG=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `TORCHINDUCTOR_MAX_AUTOTUNE=1`, `TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1`, `PYTORCH_ROCM_ARCH=gfx942;gfx950` (image defaults; no torch.compile is used) |
| Docker | `--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G --cap-add=SYS_PTRACE --security-opt seccomp=unconfined`, repo bind-mounted from NFS (`/opt/shared`, root_squash) |
| Extra env we set | `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1`, `HSA_ENABLE_COREDUMP=0`, `CUDA_VISIBLE_DEVICES=HIP_VISIBLE_DEVICES=0..7` |
| Code | Frontier fork, branch `smatar/qwen3-30b-mi355-profiling`, commit `7f4dfa5` (job 21334) / `da1ae75` (job 21313; identical profiler code) |

## 3. What exactly is measured

Entry point: `python -m frontier.profiling.linear_op.main --disable_ray --yes --device mi355x --models qwen3-a3b-30b-moe --is_moe
--num_gpus 8 --num_tensor_parallel_workers 1 2 4 8 --precision BF16 --profile_method cuda_event --max_tokens 16384
--num_tokens_list <3,327 values> --output_dir <dir>` (sbatch: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`, `STAGE=linear_op`).

Dispatch (`frontier/profiling/linear_op/main.py`):
- Outer loop over TP (line 561). **Inside** it, one `ProcessPoolExecutor(max_workers=1, mp_context=spawn)` per GPU is created
  (line ~688), i.e. 8 fresh Python processes per TP pass, each pinned to one GPU by `torch.cuda.set_device` in the initializer.
- The token list is sorted **descending** by `get_num_tokens_to_profile` (`frontier/profiling/utils/__init__.py`, `sorted(..., reverse=True)`),
  so the sweep runs 16384 → 1, and tasks are assigned round-robin: task `idx` → worker `idx % 8`.
  Every worker therefore processes 416 tasks per TP pass, in a fixed order, identical across TP passes and across nodes.
- Per task (`_worker_profile_linear_op_task`): a **new** `LinearOpWrapper` is built — a fresh `GPTModel` with dummy weights
  (includes the 151,936 × 2048 embedding, ≈0.6 GB bf16 at TP1, plus QKV 2048 × 5120 and O 4096 × 2048 for TP1), moved to the GPU,
  `eval()`. Then `profile(num_tokens)`: random `input_ids`, positions, `clear_stats()`, 3 warm-up forwards, `torch.cuda.synchronize()`,
  `mark_warmup_end()`, 50 timed forwards, `torch.cuda.synchronize()`, collect. The wrapper is dropped when the task returns
  (no explicit `empty_cache`), so the caching allocator's state accumulates across the worker's 416 tasks.

The `attn_pre_proj` scope (`frontier/profiling/linear_op/linear_op_impl.py:~500`, Qwen3 path with `use_qk_norm=True`):
```python
with self._attn_pre_proj_timer:                 # CudaTimer: start_event.record() ... end_event.record()
    qkv, _ = self.qkv_proj(hidden_states)       # ColumnParallelLinear -> vllm dispatch_unquantized_gemm
    q, k, v = qkv.split([...], dim=-1)          #   ROCm: rocm_unquantized_gemm_impl; for n_tokens > 4 it falls through to
    q = self.q_norm(q.view(..., head_dim))      #   torch.nn.functional.linear -> hipBLASLt (bf16, M=n_tokens, K=2048, N=5120/TP-ish)
    k = self.k_norm(k.view(..., head_dim))      # RMSNorm per head (QKNorm)
```
Timing (`frontier/profiling/common/cuda_timer.py`): `torch.cuda.Event(enable_timing=True)` recorded on the current stream at
scope entry and exit; `start.elapsed_time(end)` (ms) is read after the final synchronize. So the number is the **GPU-timeline
gap between the two events**: it includes host-side stalls that happen between recording the start event and enqueuing the GEMM
(e.g. a blocking `hipMalloc`, a lazy kernel/module load, a hipBLASLt solution search), as well as GPU-side execution time.

Stored per row: `time_stats.attn_pre_proj.samples` = JSON list of all 53 run times in **milliseconds**, in order; indices 0–2 are the
warm-ups, index 3 + k is timed run k. `warmup_count` = 3, `count` = 50, `median` etc. are over the 50 timed runs only.

## 4. First observation — job 21313, 2026-09-15 13:08:59–13:11:01, node 9

Data: `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/2026-09-15_1308_linear_op_dense3327/linear_op.csv` (root `linear_op.csv` is a copy).
Grid: every count 1…2048, every 8th 2056…8192, every 16th 8208…16384, minus 4000 (a known GPU fault at TP1) → 3,327 values × 4 TP = 13,308 rows.

Anomaly: 32 rows = tokens 1968–1975 × TP {1,2,4,8}. In each, `attn_pre_proj` timed run 11 (sample index 14) is 150–270 ms;
all other 52 runs are normal. Nothing else in the file exceeds 20× its row median except the usual warm-up-run-0 `emb` outlier.

Example row, tokens 1971, TP 1, `attn_pre_proj` samples in ms (warm-ups first):
```
0.2486 0.1408 0.1328 | 0.1737 0.1324 0.1311 0.1306 0.1312 0.1324 0.1318 0.1382 0.1307 0.1341 0.1311 167.577 0.1499 0.1430 0.1426
0.1434 0.1402 0.1393 0.1381 0.1381 0.1364 0.1394 0.1661 0.1387 0.1343 0.1362 0.1352 0.1339 0.1318 0.1343 0.1344 0.1336 0.1351
0.1372 0.1358 0.1344 0.1342 0.1356 0.1352 0.1366 0.1383 0.1352 0.1365 0.1794 0.1360 0.1352 0.1348 0.1384 0.1362 0.1353
```
Other ops in the **same forward** (timed run 11) of that row are elevated only mildly, i.e. what you expect right after a
long idle gap (clocks down / caches cold), not stalled themselves:

| op (same row, run 10 → 11 → 12, ms) | median |
|---|---|
| `input_layernorm` 0.0175 → 0.0142 → 0.0205 | 0.0182 |
| `attn_rope` 0.0659 → **0.2279** → 0.0891 | 0.0659 |
| `attn_post_proj` 0.0428 → **0.1186** → 0.0690 | 0.0533 |
| `post_attention_layernorm` 0.0130 → **0.0305** → 0.0133 | 0.0132 |

Neighbouring tokens (1960–1967, 1976–1984) have max timed runs of 0.14–0.23 ms — unremarkable.

## 5. Reproduction attempt — jobs 21323/21324/21325, 2026-09-15 13:59–14:02, node 9

Folder `runs/2026-09-15_1359_linear_op_spike_repro/`. Same command, `--num_tokens_list` = 1960…1985 (26 values) × 4 TP:
twice with 8 workers (same round-robin lockstep) and once with `--num_gpus 1`. **No run above 20× its row median in any of the
three files** (worst 1.2 ms vs 0.10 ms). Conclusion at the time: not shape-triggered; the 26-token sweep starts at a different
point in the worker's history than the full sweep does.

## 6. Full re-run — job 21334, 2026-09-16 06:29:05–06:31:04, node 8

Folder `runs/2026-09-16_0629_linear_op_dense3327_rerun/`. Byte-identical command to job 21313, different node, 17 h later.
**Reproduced exactly**: same 32 rows, same op, same timed index 11, no other >20× runs. Medians of all ops agree with the first
run (rerun/first median ratio p50 0.998, p5–p95 0.91–1.06).

Run-11 value of `attn_pre_proj` in ms, first run → re-run:

| tokens | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| 1968 | 150 → 165 | 247 → 259 | 269 → 149 | 155 → 205 |
| 1969 | 243 → 256 | 151 → 168 | 157 → 265 | 250 → 157 |
| 1970 | 151 → 153 | 241 → 266 | 256 → 161 | 251 → 159 |
| 1971 | 168 → 263 | 151 → 259 | 202 → 146 | 264 → 164 |
| 1972 | 151 → 245 | 255 → 255 | 156 → 288 | 154 → 244 |
| 1973 | 198 → 197 | 213 → 219 | 162 → 174 | 153 → 149 |
| 1974 | 155 → 187 | 214 → 192 | 159 → 155 | 163 → 193 |
| 1975 | 200 → 150 | 205 → 152 | 196 → 212 | 242 → 209 |

The **position** is deterministic; the **duration** is not (150–290 ms, no pattern across tokens/TPs/runs), and clusters
loosely around ~150 and ~250 ms.

## 7. What we have established

1. **Lockstep across 8 GPUs.** Tokens 1968–1975 are 8 consecutive grid values → one task per worker, all at the same position in
   each worker's list. In the descending list, 1975 is task 1352 = worker 0, per-worker index 169; 1968 is task 1359 = worker 7,
   index 169 (0-based; the band is exactly 8-aligned).
   Each worker has 416 tasks per TP pass, so the stall is ~40 % into the pass. Cumulative GPU-kernel time on a worker up to that
   point is ≈5.7 s (TP1) / 3.0 s (TP2) / 2.2 s (TP4) / 2.0 s (TP8), plus host overhead; a whole pass is ~25–30 s wall.
2. **Same position in all four TP passes** because each pass starts fresh worker processes (fresh HIP runtime, fresh allocator,
   fresh hipBLASLt handle) and replays the same task sequence. Cumulative *time* differs 3× between TP1 and TP8, so a wall-clock
   or periodic trigger is excluded; the trigger is keyed to the *sequence* (number of tasks/forwards/allocations), reached at
   task 169, forward 3 + 11 = the 15th forward of that task.
3. **Same on two nodes, 17 h apart, and absent in the 26-token sweep** → deterministic in-process/software event, not
   environmental, not shape-specific. Sizes at the band are unremarkable: input 1971 × 2048 bf16 = 8.1 MB, QKV output 1971 × 5120
   = 20.2 MB (TP1), `randn_like(q)` 16.1 MB; no power-of-two boundary in the band.
4. **Inside the `attn_pre_proj` scope only.** The other ops in that forward are 2–4× slower (post-stall effect), not stalled.
   The scope contains one hipBLASLt GEMM, a `split`, two `view`s and two per-head RMSNorm kernels. The QKV GEMM is the only
   candidate for a library-side event; the RMSNorm kernels are trivial elementwise ops.
5. It happens at the **15th forward of the task**, not the first. A first-use event (kernel load, autotune) for the shape would
   land in warm-up run 0. Whatever it is must be counting something that crosses a threshold mid-task.

## 8. Candidate causes (our ranking — please confirm or refute)

1. **PyTorch caching allocator event.** Each task allocates ~0.65 GB of fresh weights + activations and frees them on return;
   over 169 tasks the cache holds many freed segments. A specific allocation in forward 15 (the allocator's block reuse depends
   on the exact history of the 53 identical forwards, plus the fresh weights) could miss the cache and trigger `hipMalloc` of a
   new segment, or — if the allocator decided to release cached blocks (`release_cached_blocks` → `hipFree` of many segments) —
   a long blocking host call. Eight processes on the same node doing this at the same moment would serialize on the KFD/driver,
   which would explain 150–290 ms rather than a few ms, and the bimodal-ish durations. Cheap test: `PYTORCH_NO_HIP_MEMORY_CACHING=1`
   (or `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`) and see whether the spike moves/disappears; or record
   `torch.cuda.memory_stats()` deltas (`num_alloc_retries`, `num_device_alloc`, `num_device_free`, `segment.*`) per forward.
2. **hipBLASLt lazy work** (Tensile library / code-object load, or a solution-selection/tuning step) triggered by an internal
   counter. Against: it should occur on the first call for a given problem, not the 15th. Cheap test: `HIPBLASLT_LOG_LEVEL=5`
   / `HIPBLASLT_LOG_MASK` and `TENSILE_DB` in the worker around the band.
3. **HIP runtime / ROCr housekeeping keyed to a queue or event count** (e.g. signal-pool growth, stream/event object pool
   reallocation after N `hipEventCreate`s — we create two timing events per scope per forward, six scopes → ~12 events per
   forward, never destroyed until the wrapper is dropped; kernarg pool with `HIP_FORCE_DEV_KERNARG=1`). Test: `AMD_LOG_LEVEL=3`
   or `rocprof --sys-trace` for one worker in the band.
4. **Python GC** of the previous wrappers' tensors (gen-2 collection reached at that allocation count, freeing 168 models'
   worth of tensors → many `hipFree`). Test: `gc.disable()` in the worker, or `gc.set_debug(gc.DEBUG_STATS)`.

Not candidates: environmental stalls (ruled out by determinism across nodes/passes), shape (ruled out by the 26-token repro),
NFS (nothing is written during the timed loop), other jobs on the node (nodes were exclusively ours).

## 9. Questions we would like answered

1. What is the event? (A per-worker trace of one TP pass around task 169 should show it directly.)
2. Why is its duration 150–290 ms — is that 8 processes contending on one host resource?
3. Is it purely an artefact of this profiler's allocate-a-fresh-model-per-task pattern, or could it hit a serving process
   (SGLang/vLLM) that has been running for a while?
4. Do we need to worry about smaller versions of the same event polluting other rows (e.g. the 1.7× run-0 warm-up effect)?

## 10. How to reproduce / experiment (≈2 min of GPU per job)

Submit from the Frontier checkout on the cluster (`/opt/shared/frontier-qwen3-profiling/Frontier`):
```
sudo -u dn sbatch -w amd-mi355x-8 -J spike-x --export=ALL,STAGE=linear_op,COLLECT_DIR=data/profiling_spike_x \
  profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch
```
Nodes with the `v0.5.11` image: 1, 4, 8, 9. Useful knobs (env, all optional): `LINEAR_TOKENS_PY` — a Python expression for
the token list (default = the dense grid above); `LINEAR_NUM_GPUS` (default 8). To pass extra env into the container add
`-e VAR=...` in the `linear_op)` block of the sbatch (the `docker run` line).

Discriminating experiments, in order of information per minute:
- **A. Shift the history**: `LINEAR_TOKENS_PY='sorted(set(range(1,2049))-set([4000]))'` (starts at 2048 instead of 16384). If
  the spike moves to whatever tokens sit at per-worker position 169 of *that* list (i.e. tokens 2048−8·169−k ≈ 690–697 if
  descending), the trigger is positional; if it stays at 1968–1975 or vanishes, it is not.
- **B. Allocator off**: full grid with `PYTORCH_NO_HIP_MEMORY_CACHING=1` (slower, but 2 min becomes maybe 5). Spike gone → allocator.
- **C. One worker**: full grid with `LINEAR_NUM_GPUS=1` (≈15 min). Same position but ~10–20 ms instead of 200 → the 8-way
  contention explains the magnitude; unchanged 200 ms → single-process cost.
- **D. Trace**: run A or the full grid with `AMD_LOG_LEVEL=3` / `HIPBLASLT_LOG_LEVEL=5` for one worker, or wrap forward 15 of
  task 169 in `torch.profiler` (CPU + HIP activities) — the profiler code path to add it is `LinearOpWrapper.profile`.

## 11. Files

- Data (all with per-run samples): `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/runs/2026-09-15_1308_linear_op_dense3327/linear_op.csv`,
  `.../runs/2026-09-16_0629_linear_op_dense3327_rerun/linear_op.csv`, `.../runs/2026-09-15_1359_linear_op_spike_repro/*.csv`,
  `.../runs/2026-09-15_1142_linear_op_grid386/linear_op.csv` (386-token grid, first run, also spike-free — a different history).
  Each folder has a `RUN.md`; the dataset `README.md` explains the columns.
- Earlier write-up (pre-re-run reasoning): `03_linear_op_spike_investigation.md`.
- Code: `frontier/profiling/linear_op/main.py` (dispatch), `linear_op_wrapper.py` (per-task lifecycle), `linear_op_impl.py`
  (scopes), `frontier/profiling/common/cuda_timer.py`, `timer_stats_store.py`; sbatch `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`.
- Slurm logs on the cluster: `/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/q3-linear3k-21313.out`, `q3-linear3k-r2-21334.out`, and `linear_op.log` (profiler stdout, overwritten per job).

Extracting the spike rows:
```python
import pandas as pd, json, numpy as np
df = pd.read_csv("runs/2026-09-16_0629_linear_op_dense3327_rerun/linear_op.csv", low_memory=False)
s = df["time_stats.attn_pre_proj.samples"].map(json.loads).map(np.array)
timed = s.map(lambda a: a[3:])                                   # drop the 3 warm-ups
bad = df[timed.map(lambda a: a.max() > 20 * np.median(a))]       # -> 32 rows, tokens 1968..1975
print(bad[["num_tokens", "num_tensor_parallel_workers"]].assign(idx=timed[bad.index].map(np.argmax), ms=timed[bad.index].map(np.max)))
```
