# `attention.csv` provenance

Frontier's own `atten_input_file` config field hard-codes the filename
`attention.csv` (templated only by device/model directory) — exactly one
file at this path is ever "live," matching the same constraint
`moe_kernel_only.PROVENANCE.md`/`linear_op.PROVENANCE.md` already document
for this model's other profile files.

| file | `block_size` | decode grid | `tp` | status | collected |
|---|---|---|---|---|---|
| `attention.csv` | **16** | `batch∈{1,2,4,8,12,16,24,32}`×`kv∈{0,8,16,24,32,48,64,96}` (tp≤2), sparse (tp∈{4,8}, see below) | merged, see below | **live** | Track B Step 35 (dense tp1/2 rows) + Step 34 (tp4/8 rows preserved) |
| `attention.sparse_block16.csv` | 16 | `batch∈{1,2}`×`kv∈{0,32,64,96}`, all `tp∈{1,2,4,8}` | — | archived | Track B Step 34 (real hardware, `xai-3`/`amd-mi355x-3`, GPU index 1) |
| `attention.block32.csv` | 32 | — | — | archived | pre-Track-B, original collection (provenance below) |

## Why the live file is a merge, not a straight recollection (Track B Step 35)

Step 34's own MAPE figures for the MLA decode-attention operators were
20–76%, roughly an order of magnitude worse than qwen3's true-mixed models
(2.6–3.9%) — traced to grid sparsity: Step 34's own decode grid (`batch∈{1,2}`
× `kv∈{0,32,64,96}`, 8 points) was eight times sparser than the
`attention_kernel_only`/`linear_op_kernel_only`/`moe_kernel_only` dispatch-free
envelope Step 34 itself derived for the same model (64 decode points). Step 35
recollected `attention.csv` at that same dense decode grid, restricted to
`tp∈{1,2}` (the only `tp` values the dispatch-free families and this track's
own disaggregated arms ever use), plus a 5-point prefill sweep
(`total_tokens∈{32,64,96,128,160}`, bracketing Step 30's own 32-token
workload) at the same two `tp` values.

**The live file keeps Step 34's own `tp∈{4,8}` rows from
`attention.sparse_block16.csv` unchanged, merged in alongside the new dense
`tp∈{1,2}` rows** — Step 35's own task scope never needed `tp>2`, so a
straight overwrite would have silently dropped real, correctly-labeled
`tp=4`/`tp=8` coverage nothing asked to remove. `attention.sparse_block16.csv`
is archived as Step 34 collected it, unmodified, including two pre-existing
duplicate-keyed rows at `tp∈{4,8}` (the chunked-prefill sweep and the
full-prefill sweep both independently producing a `(chunk_size=64, kv=0)`
point) — a benign quirk of Step 34's own collection, not introduced or
corrected here.

## Why the block size changed

Track B Step 31 established that `16` is correct: this project's own
working `qwen3-a3b-30b-moe` profile is collected at `block_size=16` (the
profiler's own unoverridden default, never a deliberate choice), the
evaluator's own MLA structural filter reads `block_size=16` with no CLI
override that reaches it, and — independently — vLLM's own
`CacheConfig.DEFAULT_BLOCK_SIZE` is `16`, unconditional, and the AFD
plugin's own real recipes never override it. `32` was never a documented
real-deployment target; it is what this model's *original* profiling
collection happened to use.

## Why the file could be recollected at all

Track B Step 32 found that recollecting through this checkout's own
`frontier.profiling.attention.main` failed for a *different* reason than
`block_size`: the merged `TorchSdpaAttentionWrapper` backend explicitly
refuses `LATENT_MLA` models ("MLA needs its own algorithm and scopes"), and
the recorded `attention_backend=TORCH_SDPA_MLA` on the original file
matched no code path anywhere in this pinned checkout — grepped and found
nowhere. Track B Step 33 traced this to a real, unmerged branch
(`origin/task/deepseek-mla-attention-port-block-sweep`, commit `8c87017`)
that ports a working, portable (ROCm-compatible) `TorchSdpaMlaAttentionWrapper`
from `server3`/`server1`'s own local checkouts — never merged into this
project's own pinned `main`, and not merged here either. Track B Step 34
recovers *only* the wrapper (plus the minimum driver-side plumbing to make
its output pass validation) as a guarded, runtime adapter —
`dc-sim/src/integration/profiling/mla_attention_wrapper_adapter.py` — the
same "runtime-patch a pinned external dependency, guarded" pattern this
project already uses for Frontier elsewhere (`rope_api_adapter.py`,
`fp8_linear_op_adapter.py`). None of the branch's own CSV files or its
`attention.csv` deletion were touched; Frontier's own `main` remains
unmodified and uncommitted to.

**The batch-aliasing correctness fix the branch also carries was not part
of this recovery** — it didn't need to be. `src/integration/profiling/attention_block_table_fix.py`
("Task 53 Fix B") already cherry-picked that exact fix from this same
branch, independently, before this task started, and every attention-
profiling wrapper script in this track (including every `qwen3` collection)
already installs it. This collection used it too, unchanged.

## The original file's own provenance, now understood rather than mysterious

`attention.block32.csv`'s own `attention_backend=TORCH_SDPA_MLA` label was,
until Step 33, unexplained: no code in this checkout could have produced
it. It was measured using the same wrapper this checkout has now recovered
— but on `server3`'s or `server1`'s own local checkout, never this
project's own pinned `main`, per Track B Step 33's own reading of that
branch's commit history and its own edit to `profiling_knowledge/INFRASTRUCTURE_MAP.md`.
The data is real (measured on real hardware); the record of how to
reproduce it was not, until this branch was found and read. A real,
disclosed instance of "a profile collected by code that was never part of
the checkout it now lives in."

See `docs/tasks/track-b-step34-mla-wrapper-report.md` and
`docs/tasks/track-b-step35-deepseek-kernel-only-report.md` in `dc-sim` for
the full collection record.
