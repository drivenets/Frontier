# spike_diag_tools — scripts behind the tables in `05_linear_op_spike_root_cause.md`

Input for every script is a diagnostics directory written by `frontier/profiling/linear_op/spike_diag.py`
(one `worker_gpu<k>_pid<pid>.jsonl` per worker process; on the cluster:
`/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/spike_diag/<e>`).
All take the directory as the single argument and print to stdout; Python 3.10 stdlib only.

| script | what it prints | report section |
|---|---|---|
| `parse_diag.py` | per worker: init header, every gen-2 collection with the forward it landed in, every >20× `attn_pre_proj` spike with the host time of that forward and the GC events overlapping it, allocator counters | 05 §3.1, §3.2, §3.3 |
| `xworker.py` | one line per gen-2 collection across all workers: TP, GPU, task, tokens, forward, wall-clock start, duration, `attn_pre_proj` and host time of that forward; spread of start times within a TP pass | 05 §3.1 table, §3.4 |
| `gc_traj.py` | gen-2 counter at task begin over the pass (detects gen-2 collections that fell between tasks), gen-1 collections per task, task durations | 05 §3.5 (E1 vs E2) |
| `gen1_pos.py` | which forward index gen-0 / gen-1 / gen-2 collections land in; net tracked allocations per task | 05 §3.3 (forwards 14 / 36) |
| `q4.py` | duration of gen-0 / gen-1 collections and their effect on the sample they land in | 05 §3.6 (brief Q4) |
| `e8.py` | 200-repetition run: gen-0 landing forwards, every gen-2 with its timed-run index, 100 ms-class spikes | 05 §3.9 |
| `e8b.py` | for each gen-2 collection, which timed scope (if any) absorbed it | 05 §3.9 |

Example: `python3 parse_diag.py /opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/spike_diag/e1b`
