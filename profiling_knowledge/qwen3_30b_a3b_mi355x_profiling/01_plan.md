# Plan: Qwen3-30B-A3B MI355X profiling — five operators, AITER attention + attention-layer GEMMs

Goal: collect fresh MI355X profiling CSVs for `qwen3-a3b-30b-moe` covering `attn_prefill`,
`attn_decode`, `attn_kv_cache_save` (attention profiler, AITER backend) and `attn_pre_proj`,
`attn_post_proj` (linear_op profiler; `attn_rope`, layernorms, `emb` come for free), with `head_dim`
recorded and every timed run kept in iteration order (50 timed + 3 warm-up per shape), so a regressor can
be trained on them later and warm-up vs. noise outliers can be told apart. Regressor training, simulator changes and MoE
are out of scope.

Established (not re-derived): Frontier binds this model to dense GQA attention + the attention-layer
GEMMs; each operator's time is `time_stats.<op>.median`, filtered by exact match on model dims /
`block_size` / TP, into a per-operator RandomForest. Sources: `frontier/operators/binding.py`,
`frontier/attention/families.py:51-98`, `sklearn_execution_time_predictor.py:1222-1228`,
`linear_op_impl.py:501-526` (QK-norm inside `attn_pre_proj`), `data/config/models/qwen3-a3b-30b-moe.json`
(32 q heads, 4 kv heads, head_dim 128, 48 layers, bf16). Stage 1–2 record: `00_requirements_and_source_map.md`.

Decisions (user): branch based on `feat/gptoss120b-mi355` (c68096c); `head_dim` required, no legacy
migration; MoE deferred; canonical attention files = union of block 1 and 16 AITER rows, legacy
TORCH_SDPA trio renamed `*_sdpa_block16.csv`; grid kept exactly as the profiler measures it.

## Step 1 — record `head_dim`

- `frontier/attention/families.py:79-91`: insert `"head_dim"` after `"n_kv_head"` in
  `DENSE_ATTENTION_FAMILY.required_profiling_feature_columns`.
- `frontier/profiling/attention/attention_wrapper.py:388-402, 493-503, 563-573`: add
  `"head_dim": int(self._model_config.get_head_size())` next to `"n_kv_head"` in the three result
  dicts (`get_head_size()` already returns the explicit 128, `model_config.py:455-463`).
- Update the pinned tuple/fixtures so they carry `head_dim`: `tests/unit/test_attention_family_specs.py:765-777` and the second dense tuple at `:820-832`,
  `test_attention_family_spec_data.py:25-37`, `test_attention_main_true_mixed_contract.py` (`_dense_standard_row`),
  `test_attention_tp_effective_mapping.py`, `test_execution_time_predictor_max_tokens_budget.py`,
  `test_shared_prediction_model_manager_eager_attention_decode.py`, `test_spec_decode_predictor_verify_prefill_path.py`,
  `test_mha_stage2_profile_modeling.py`, `test_mqa_stage2_profile_modeling.py`.
- Implementation note (2026-09-14): only the two pinned-tuple tests needed `head_dim`; the other seven fixture
  tests bypass the write-time validator (the dense loader only filters columns) and passed unchanged.
- Cost of strictness: none for the simulator. The predictor validates the schema only for the MLA
  family (`sklearn_execution_time_predictor.py:1233-1238`, `shared_prediction_model_manager.py:1682-1686`);
  the dense loader only filters columns. Enforcement is at profiler write time (`attention/main.py:886`).
- Check: existing `test_profiling_schema_columns_are_derived_from_family_spec` (updated) plus one new
  assert in `test_attention_main_true_mixed_contract.py`: a dense row without `head_dim` is rejected by
  `_prepare_standard_attention_output_dataframe`.

## Step 1b — 50 samples per shape, every run recorded in order

Today: 3 warm-up forwards, `clear_stats()`, then 5 (attention) / 20 (linear_op) timed forwards, and
`TimerStatsStore.get_stats()` reduces them to min/max/mean/median/std/count (`timer_stats_store.py:24-41`,
`attention_wrapper.py:30-31, 355-369`, `linear_op_wrapper.py:23-24, 161-179`). Iteration order is lost.

- `attention_wrapper.py:31` and `linear_op_wrapper.py:24`: `ACTIVE_STEPS = 50`. `WARMUP_STEPS` stays 3.
- Move the `clear_stats()` that sits between the warm-up and the timed loop (`attention_wrapper.py:361, 476, 551`,
  `linear_op_wrapper.py:169`) to just before the warm-up loop of the same cuda_event branch (in the attention
  wrapper that branch has no other clear; the one at `:342` belongs to the record_function branch). The singleton
  store then starts empty for every shape and holds all 53 runs of that shape in order.
- `TimerStatsStore.mark_warmup_end()` is called by both wrappers right after the warm-up `synchronize()`; it snapshots
  each scope's record count. `get_stats()` computes the aggregates (`min/max/mean/median/std/count`) over the records
  made after that snapshot, so the existing columns keep their meaning (timed runs only, `count == 50` for scopes
  recorded once per forward), and adds `"warmup_count"` (per scope) and `"samples": json.dumps(all runs, 6 decimals)`
  in run order, warm-ups first. Per-scope counting matters: linear_op's `GPTModel.forward` calls `embed_tokens`
  twice per forward (`linear_op_impl.py:1088,1091`), so `emb` has 6 warm-up records and 100 timed ones; a flat
  "drop the first 3" would have leaked warm-up runs into `emb`'s aggregates (correctness review, iteration 1).
- Row builders (the three attention dicts + the linear_op `stats` dict) add `"warmup_steps": WARMUP_STEPS,
  "active_steps": ACTIVE_STEPS`. `pd.json_normalize` (`attention/main.py:1238`, `linear_op/main.py:761`) turns the
  new key into one `time_stats.<op>.samples` column per op — a JSON string, `json.loads` to read.
- Check: one test in `tests/unit/test_profiling_timing_stats_contract.py`: feed 3 + 4 known floats, assert
  `count == 4`, aggregates ignore the first 3, `samples` decodes to all 7 in order.
- Cost: the timed loop is 10× longer for attention (2.5× for linear_op). Step 3's pilot measures seconds per
  attempt with this loop before the real jobs are sized; fewer samples is not the fallback, splitting TPs across jobs is.
- Not covered, on purpose: cold-start timing. Every shape runs warm-up first in an already-hot process; a cold
  number would need a fresh process per shape. Separate experiment if wanted.

## Step 2 — de-gpt-oss the sweep script (defaults unchanged)

`profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh`:
- `prewarm_aiter()`: replace `NQ_TOTAL, NKV_TOTAL, HD = 64, 8, 64` with env vars
  `PREWARM_NQ="${PREWARM_NQ:-64}" PREWARM_NKV="${PREWARM_NKV:-8}" PREWARM_HD="${PREWARM_HD:-64}"` defined in the
  "What to sweep" block at the top (before `run_in_docker()` runs, so `set -u` is satisfied), passed into the
  embedded python via `os.environ`. Warms every TP × block pair as today.
- `run_in_docker()`: forward, in the script's existing `-e NAME="$NAME"` form, `PREWARM_NQ PREWARM_NKV PREWARM_HD`
  and also `WORK_DIR COLLECT_DIR LOG_DIR` (today the latter three are not forwarded and silently fall back to
  their defaults inside the container). The script's defaults are host-absolute (`$REPO_ROOT/...`), which is fine unforwarded but
  wrong inside the container, so the sbatch passes repo-relative values (`data/profiling/...`); the container's
  working directory is `/workspace/frontier` (= `$REPO_ROOT`), so relative paths resolve there. When a variable is
  unset the `-e` entry is omitted and the in-container default applies, exactly as today. No new mounts.
- `DOCKER_IMAGE` default stays; the job overrides it (`v0.5.9` is not on the nodes, `v0.5.11-rocm700-mi35x` is,
  and its `paged_attention_ragged` / `mha_batch_prefill_func` signatures match the wrapper — probed 2026-09-10).
- Check: `bash -n`, and `./profile_gptoss_attention_full_sweep.sh --dry-run` prints the same commands as before
  when no new env var is set.

## Step 3 — one sbatch file, four submissions: pilot, linear_op, then the two AITER cells

`profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` (`#SBATCH -p XAI -N1 --gres=gpu:8
-o /opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/%x-%j.out`; `-t` is passed per
submission). It reads `STAGE` (`pilot | linear_op | attention`) and, for attention, `AITER_BLOCK_SIZES`,
`AITER_TPS="${AITER_TPS:-1 2 4 8}"`, `WORK_DIR="${WORK_DIR:-data/profiling/sweep_work}"`,
`COLLECT_DIR="${COLLECT_DIR:-data/profiling}"` (all repo-relative, forwarded into the container by Step 2). Body:

1. `cd /opt/shared/frontier-qwen3-profiling/Frontier` (the worktree is rsync'd there from the VM beforehand,
   `.git` and `data/profiling/**/*.csv` excluded; `/opt/shared` is the only NFS path shared across nodes,
   `/data` and `/home/dn` are node-local; everything the job writes lands under this checkout).
   `mkdir -p /mnt/data/aiter-cache`; `docker image inspect lmsysorg/sglang:v0.5.17-rocm720-mi35x >/dev/null || exit 1`.
2. `STAGE=attention`: the Step-2 script with env
   `MODELS=qwen3-a3b-30b-moe BACKENDS=AITER AITER_NUM_GPUS=4 MAX_SEQ_LEN=16384
   PREWARM_NQ=32 PREWARM_NKV=4 PREWARM_HD=128 DOCKER_IMAGE=lmsysorg/sglang:v0.5.17-rocm720-mi35x
   AITER_CACHE_DIR=/mnt/data/aiter-cache` plus the job's `AITER_BLOCK_SIZES AITER_TPS WORK_DIR COLLECT_DIR`, `--docker`. Everything else is
   the template's default: batch list 1…512 (18 values), decode KV 128…16384 (7 values), chunked-prefill grid
   search, true-mixed on (prefill bs {1,2} × chunk {1024,4096,8192} × the decode grids), BF16, cuda_event.
   One cell per job (`__aiter__block16` or `__aiter__block1`), looping TP 1 2 4 8. Expected per cell: 2,772
   standard attempts × 4 TP = 11,088 rows and 360 × 4 = 1,440 true-mixed rows (`TrueMixedBatchInput.is_valid` drops decode KV 16384 and total batches above 128, so the true-mixed grid uses decode KV ≤ 8192 and decode batch ≤ 96), minus whatever
   `is_under_memory_limit` drops at TP 1/2 (96 / 48 KB per token). 4 GPUs as the template does (8 faulted for
   gpt-oss).
3. `STAGE=linear_op`, image `lmsysorg/sglang:v0.5.11-rocm700-mi35x` (`LINEAR_IMAGE`; v0.5.17 dropped vllm, which the linear_op
   profiler imports, and linear_op never calls the prefill kernel that faults in v0.5.11), same docker flags as the sweep's `run_in_docker()`:
   ```
   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G \
     -v "$PWD":/workspace/frontier -w /workspace/frontier -e PYTHONPATH=/workspace/frontier \
     -e FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1 -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
     lmsysorg/sglang:v0.5.17-rocm720-mi35x bash -lc '
       python -m frontier.profiling.linear_op.main --disable_ray --yes --device mi355x --models qwen3-a3b-30b-moe --is_moe \
         --num_gpus 8 --num_tensor_parallel_workers 1 2 4 8 --precision BF16 --profile_method cuda_event \
         --max_tokens 16384 \
         --num_tokens_list $(python -c "from frontier.profiling.utils import get_num_tokens_to_profile as g; print(*[t for t in g(16384) if t != 4000])") \
         --output_dir data/profiling' > data/profiling/sweep_work/logs/linear_op.log 2>&1
   ```
   `--max_tokens 16384` is required: the default is 4096 and `get_num_tokens_to_profile` rejects list values above it.
   Grid (2026-09-15 amendment, user request "3k shapes"): every token count 1…2048, every 8th to 8192, every 16th to 16384
   minus 4000 → 3,327 values × 4 TP = 13,308 rows (`LINEAR_TOKENS_PY` in the sbatch); the first run used the profiler's default
   386-value grid (kept as `linear_op_grid386.csv`). 4000 is excluded because it faults the GPU at TP=1
   (`examples/profiling/profile_mi355x.sh`). Replicated ops (`emb`, layernorms) are NaN on TP>1 rows by
   design (`linear_op/main.py:581-586`). Writes `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv`.
4. `STAGE=pilot`: the attention path with `AITER_TPS=1 MAX_SEQ_LEN=2048 AITER_BLOCK_SIZES=16
   DECODE_KV_CACHE_SIZE_LIST="128 512 1024 2048" TRUE_MIXED_PREFILL_CHUNK_SIZES=1024
   WORK_DIR=data/profiling/sweep_work_pilot COLLECT_DIR=data/profiling_pilot` (own trees, so the pilot's
   block-16 cell can never make the real one be skipped and its CSVs never reach `compute/`). Grid on this
   checkout: 380 standard + 60 true-mixed attempts, each with the new 53-run loop; the profiler
   prints `Profiling: N/N` as it goes, so the log gives wall time and seconds per attempt directly, on the
   same image, kernels and prewarm as the real run.

TP=8 with 4 KV heads is valid for both profilers: `attention_tp_policy.is_attention_tp_supported` and
`model_config.get_num_kv_heads` replicate KV heads when `tp % num_kv_heads == 0` (1 KV head per worker at TP=8),
`linear_op_impl._resolve_num_kv_heads_per_worker` (`:33-52`) does the same, and the upstream a800 dataset for
this exact model already carries TP=8 rows (1,150 attention, 259 linear_op), as does the MI355X SDPA data.

Submission order, from the VM, each after the previous has been looked at:
```
rsync -a --exclude .git --exclude 'data/profiling/**/*.csv' /home/dn/Frontier-qwen3-profiling/ cluster:/opt/shared/frontier-qwen3-profiling/Frontier/
J=/opt/shared/frontier-qwen3-profiling/Frontier/profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch
ssh cluster "sudo -u dn mkdir -p /opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs"   # Slurm needs the -o dir to exist
C=FRONTIER_COMMIT=$(git -C /home/dn/Frontier-qwen3-profiling rev-parse --short HEAD)   # recorded in the job banner (the rsync excludes .git)
ssh cluster "sudo -u dn sbatch -t 00:30:00 -J q3-pilot     --export=ALL,$C,STAGE=pilot     $J"
ssh cluster "sudo -u dn sbatch -t 06:00:00 -J q3-linear    --export=ALL,$C,STAGE=linear_op $J"
ssh cluster "sudo -u dn sbatch -t <sized> -J q3-attn16 --export=ALL,$C,STAGE=attention,AITER_BLOCK_SIZES=16 $J"
ssh cluster "sudo -u dn sbatch -t <sized> -J q3-attn1  --export=ALL,$C,STAGE=attention,AITER_BLOCK_SIZES=1  $J"
```
Sizing rule from the pilot: `seconds_per_attempt × (11088 + 1440) × 1.3` per cell (the pilot is TP=1, the most
expensive TP, so this is an upper bound). If that exceeds the cluster's per-job limit, submit each cell as two
jobs with their own trees, because the sweep names the cell by (model, backend, block) only and would otherwise
skip the second job as "already collected" and overwrite the first job's collected files:
```
ssh cluster "sudo -u dn sbatch -t <sized/2> -J q3-attn16-tp12 --export=ALL,STAGE=attention,AITER_BLOCK_SIZES=16,AITER_TPS='1 2',WORK_DIR=data/profiling/sweep_work_tp12,COLLECT_DIR=data/profiling_tp12 $J"
ssh cluster "sudo -u dn sbatch -t <sized/2> -J q3-attn16-tp48 --export=ALL,STAGE=attention,AITER_BLOCK_SIZES=16,AITER_TPS='4 8',WORK_DIR=data/profiling/sweep_work_tp48,COLLECT_DIR=data/profiling_tp48 $J"
```
(and the same pair for block 1). Step 4 then `pd.concat`s the two half-cell files into `attention*_aiter_block{N}.csv`
before building the canonical trio. The linear_op job goes first so
its 1,544 rows exist regardless of how the attention jobs fare. Check: `sbatch --test-only` for each; `bash -n`.

## Step 4 — bring results home, sanity-check

Commands (run by hand from the worktree):

```
cd data/profiling/compute/mi355x/qwen3-a3b-30b-moe
# rename the legacy files FIRST: the cluster run writes a fresh canonical linear_op.csv that the rsync below would otherwise overwrite
for f in attention attention_true_mixed attention_combined; do git mv $f.csv ${f}_sdpa_block16.csv; done
git mv linear_op.csv linear_op_maxtokens4096.csv        # keeps the superseded 4k-token file of THIS dataset; no other legacy CSV is touched
rsync -a cluster:/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/compute/mi355x/qwen3-a3b-30b-moe/ ./
rsync -a cluster:/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work*/logs/ ../../../../../logs/qwen3_mi355x/   # per-WORK_DIR log trees incl. pilot and split-TP jobs
# only if a cell was split across two TP jobs: first merge the halves (pulled from compute/ under profiling_tp12 / profiling_tp48)
#   pd.concat([read(tp12 file), read(tp48 file)]).to_csv(f"{f}_aiter_block{b}.csv", index=False)
python - <<'PY'
import pandas as pd
for f in ("attention", "attention_true_mixed", "attention_combined"):
    pd.concat([pd.read_csv(f"{f}_aiter_block{b}.csv", low_memory=False, float_precision="round_trip") for b in (1, 16)]).to_csv(f"{f}.csv", index=False)
# float_precision="round_trip": pandas' default parser is not round-trip exact, and the sanity script compares the
# canonical files with the per-cell files by row hash
PY
```

The exact-match `block_size` filter picks the right rows out of the union, and the canonical
`attention.csv` carries the `is_mixed_batch`/`total_tokens` marker columns, so
`attention_dataset_contract.py` is satisfied with the true-mixed/combined siblings present.
The training input for the next phase is `attention_combined.csv` (standard + true-mixed rows; the
predictor splits `attention_df[attention_df["is_true_mixed_batch"]]` out of the same frame,
`sklearn_execution_time_predictor.py:3191`),
plus `linear_op.csv`.

`profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py` (pandas + an `ok()` accumulator so every failure is reported at once; ~80 lines; takes the
dataset dir as its argument, run on the canonical trio and `linear_op.csv`; exits non-zero on any FAIL):
- required dense columns present incl. `head_dim == 128`, `n_q_head 32`, `n_kv_head 4`, `n_embd 2048`;
- `attention_backend == {"AITER"}`, `block_size == {1, 16}`, `num_tensor_parallel_workers == {1,2,4,8}`,
  `max_model_len == {16384}`, `profiling_precision BF16`, `measurement_type CUDA_EVENT`;
- row counts: `attention_combined == attention + attention_true_mixed`; per-cell counts printed against
  the 11,088 / 1,440 expectation;
- grid coverage, per (block_size, TP): build the expected key sets with the profiler's own generators on the
  same inputs — `get_attention_input_combinations(16384, 1, 512, False, False, BATCH, DECODE_KV, True, -1)` →
  `(prefill_chunk_size, kv_cache_size, batch_size, is_prefill)` and `get_true_mixed_attention_input_combinations(...)`
  → `(num_prefill_seqs, prefill chunk, decode_batch_size, decode kv)` — then assert observed keys ⊆ expected
  (nothing unplanned was measured), assert every expected prefill key is present (the memory filter never drops
  prefill), and assert every missing decode / true-mixed key sits at TP ∈ {1, 2} (the only place `is_under_memory_limit`
  can bite at 96 / 48 KB per token); print the missing keys. No `max_num_blocks` recompute: add one only if the missing set is ever not
  explainable by TP 1/2 memory.
- `warmup_steps == 3`, `active_steps == 50`, every `time_stats.<op>.count == 50`, every `time_stats.<op>.samples`
  decodes to 53 floats; print, per op, the histogram of the arg-max position across all rows (a spike at
  positions 0-2 = warm-up effect leaking past 3 warm-ups, flat = noise) and the max/median ratio distribution;
- no NaN in `time_stats.attn_kv_cache_save.median` (all rows), `time_stats.attn_prefill.median` (prefill rows),
  `time_stats.attn_decode.median` (decode rows);
- no symlinks in the dataset dir (`Path.is_symlink()` over `*.csv`; the repo's `deepseek-r1-0528` symlink is elsewhere);
- duplicates: repeated shapes are expected (the chunk-size grid repeats 128/1024/4096/16384 at range seams and a
  full prefill of length L equals chunk L at kv 0), so they are kept as measured and only checked for agreement
  (next bullet); the canonical trio must contain exactly the rows of the two per-cell files (`len` equality);
- the four checks from `GPTOSS_TRUE_MIXED_BATCH_PROFILING.md`: repeated shapes agree (asserted: median relative spread < 5 %, max printed),
  decode median non-decreasing in batch and in KV, true-mixed vs even at a comparable shape is a small
  positive overhead, and the cross-check between the two block-size cells at the same shape is printed;
- linear_op: 1,544 rows, 386 `num_tokens` values, no `4000`, `attn_pre_proj`/`attn_post_proj` medians non-NaN
  everywhere, `use_qk_norm == True`, no `time_stats.add.*` (fused into RMSNorm for this model).

## Files

| Action | Path |
|---|---|
| modify | `frontier/attention/families.py`, `frontier/profiling/attention/attention_wrapper.py`, the nine tests above |
| modify | `frontier/profiling/common/timer_stats_store.py`, `frontier/profiling/linear_op/linear_op_wrapper.py`, `tests/unit/test_profiling_timing_stats_contract.py` (Step 1b) |
| modify | `profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh` (env-driven prewarm dims) |
| create | `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` (new `slurm/` dir), `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py`, `tests/unit/test_qwen3_mi355x_sanity_check.py` (script-contract test on a generator-built dataset, same convention as `test_examples_profiling_contracts.py`; the only check that can run before data exists) |
| data | `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/`: `attention*_aiter_block{1,16}.csv`, canonical trio, `linear_op.csv`; legacy renamed |

## Verification

```
python -m pytest tests/unit/test_attention_family_specs.py tests/unit/test_attention_family_spec_data.py \
  tests/unit/test_attention_main_true_mixed_contract.py tests/unit/test_attention_tp_effective_mapping.py \
  tests/unit/test_execution_time_predictor_max_tokens_budget.py tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py \
  tests/unit/test_spec_decode_predictor_verify_prefill_path.py tests/unit/test_mha_stage2_profile_modeling.py tests/unit/test_mqa_stage2_profile_modeling.py \
  tests/unit/test_profiling_timing_stats_contract.py -q
bash -n profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh && bash -n profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch
bash profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh --dry-run          # unchanged output
for ex in STAGE=pilot STAGE=linear_op STAGE=attention,AITER_BLOCK_SIZES=16 STAGE=attention,AITER_BLOCK_SIZES=1; do ssh cluster "sudo -u dn sbatch --test-only --export=ALL,$ex /opt/shared/frontier-qwen3-profiling/Frontier/profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch"; done
python profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py data/profiling/compute/mi355x/qwen3-a3b-30b-moe
```

## Risks

- Profiler writes CSVs only at the end; a crashed cell loses that cell (one cell per job, rerun skips finished ones).
- 50 timed samples make each cell ~10× longer in the timed loop; the pilot decides `-t` and whether a cell is split across two TP jobs before anything long is submitted.
- `warmup_steps`/`active_steps` are new non-`time_stats.` columns, so `merge_profiling_rows_on_feature_columns`
  treats them as group-by keys: legacy rows (column absent) and new rows never collapse together. Intended.
- MoE/collectives profilers never call `mark_warmup_end()` and keep their own post-warm-up clear, so their
  `warmup_count` is 0 and their `samples` (if ever collected) hold timed runs only. MoE is deferred; noted, not changed.
- `samples` columns enlarge the CSVs (~53 floats × 5 ops per row, roughly 5× today's file size); readers must `json.loads`.
- Largest decode / true-mixed combos are memory-filtered at TP 1/2 — expected, printed by the sanity script.
- 8-GPU AITER fault seen on gpt-oss; running 4 as the template does. Cell wall time is sized from the pilot, not guessed.
- Image is `v0.5.17-rocm720`, not the plan's original `v0.5.11-rocm700`: the pilot (2026-09-15) showed v0.5.11's aiter faults
  (HSA memory aperture violation) in `mha_batch_prefill` for head_dim 128 with paged KV at both block sizes, while v0.5.17 runs the
  same grid clean; the switch was verified by the pilot (440 attempts, 3 CSVs, no faults).

Skipped: dataset validation module, expectations file, prewarm helper module, ingest/submit scripts,
manifest generator, glossary/ADRs, offline grid recomputation, duplicate canonicalisation. Add any of
them only when a second model is profiled with the same pipeline.
