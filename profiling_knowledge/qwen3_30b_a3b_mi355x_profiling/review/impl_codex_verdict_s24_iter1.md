OVERALL: FAIL
BLOCKING_FINDINGS:
  - HIGH: profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py:77 — Standard grid coverage iterates only observed `(block_size, TP)` groups, so an entire missing TP>2 cell can pass as long as marginal block/TP sets remain present — Iterate the explicit `(1,16) x (1,2,4,8)` grid and add a regression deleting a full block16/TP8 cell.
  - HIGH: profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py:70 — True-mixed grid coverage has the same observed-groups-only gap, so a full missing TP>2 true-mixed cell is not checked — Iterate the explicit block/TP grid for true-mixed rows and add a full-cell-missing regression.
  - HIGH: profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py:33 — Canonical-vs-block checks compare only row counts, not that canonical rows equal block1 + block16 rows — Compare normalized row content/hashes and test a same-length stale/corrupt canonical file.
  - HIGH: profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py:34 — `attention_combined == attention + attention_true_mixed` is only a length check, so the training file can be stale or wrong with the same row count — Compare content-equivalence against `pd.concat([attention, attention_true_mixed])` and add a same-length corruption test.
NON_BLOCKING_FINDINGS:
  - none
CHECKLIST_RESULTS:
  - Acceptance criteria met (plan Steps 2, 3, 4): FAIL — Steps 2/3 look aligned, but Step 4 has blocking coverage/equality gaps.
  - gpt-oss behaviour preserved (sweep script defaults unchanged; dry-run identical): PASS — defaults remain gpt-oss and new WORK_DIR/COLLECT_DIR/LOG_DIR forwarding is conditional when unset.
  - sbatch runs the plan's four submissions correctly (pilot isolation, linear_op first with --max_tokens 16384, one AITER cell per job, split-TP override, failure exits non-zero): PASS — the stage bodies match the specified submissions and attention checks missing collected CSV non-zero.
  - sanity_check.py implements the plan's Step 4 checks and the tests would catch their removal: FAIL — missing full-cell regressions and same-row-count union/canonical regressions.
  - Scope discipline (only the plan's four files; minimal-code policy; nothing from the cut list): PASS — the provided diff touches only the four expected files.
FIX_BRIEF:
  - Change grid checks to iterate all expected block/TP pairs, compare canonical/combined content not just lengths, and add exact-message tests for full TP>2 cell removal plus same-length stale canonical/combined corruption.