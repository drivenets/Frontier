"""Opt-in diagnostics for the deterministic attn_pre_proj stall at timed run 11
(profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/04_linear_op_spike_specialist_brief.md).

Enabled only when FRONTIER_SPIKE_DIAG=<dir> is set in the worker environment. Per worker
process it records every Python GC collection (start/stop, generation, duration) and the
host-side wall time of every forward call, and writes one JSON line per task to
<dir>/worker_gpu<k>_pid<pid>.jsonl. Recording inside the timed loop uses array.array so it
adds no GC-tracked allocations and cannot shift the GC schedule it is observing.

FRONTIER_SPIKE_GC=freeze|disable changes the interpreter's GC behaviour at worker init
(interventional experiment); unset leaves it untouched.
"""
import array
import gc
import json
import os
import time

DIAG_DIR = os.environ.get("FRONTIER_SPIKE_DIAG")
enabled = bool(DIAG_DIR)

_gc_events = array.array("d")  # 4 doubles per record: perf_counter, phase (0=start, 1=stop), generation, collected
_fwd_times = array.array("d")  # perf_counter at forward begin/end, alternating; reset per task
_task_idx = -1
_task_t0 = 0.0
_task_wall0 = 0.0
_task_gc_count0 = None
_out = None
_gpu_id = None


def _gc_cb(phase, info):
    _gc_events.extend(
        (
            time.perf_counter(),
            0.0 if phase == "start" else 1.0,
            float(info["generation"]),
            float(info.get("collected", -1)),
        )
    )


def worker_init(gpu_id):
    # The GC intervention is applied even when recording is off, so it can be tested without
    # any in-loop instrumentation (FRONTIER_SPIKE_GC=disable|freeze, FRONTIER_SPIKE_DIAG unset).
    mode = os.environ.get("FRONTIER_SPIKE_GC", "")
    if not enabled:
        if mode == "disable":
            gc.disable()
        elif mode == "freeze":
            gc.freeze()
        return
    global _out, _gpu_id
    _gpu_id = gpu_id
    os.makedirs(DIAG_DIR, exist_ok=True)
    _out = open(f"{DIAG_DIR}/worker_gpu{gpu_id}_pid{os.getpid()}.jsonl", "a", buffering=1)
    n_tracked = len(gc.get_objects())
    full_collect_ms = None
    if os.environ.get("FRONTIER_SPIKE_INIT_COLLECT") == "1":
        # NB: a full collection here resets the interpreter's long_lived_pending/long_lived_total
        # and therefore moves the first natural gen-2 collection; off by default.
        t0 = time.perf_counter()
        gc.collect()
        full_collect_ms = (time.perf_counter() - t0) * 1e3
    if mode == "disable":
        gc.disable()
    elif mode == "freeze":
        gc.freeze()
    gc.callbacks.append(_gc_cb)
    _out.write(
        json.dumps(
            {
                "kind": "worker_init",
                "gpu": gpu_id,
                "pid": os.getpid(),
                "wall": time.time(),
                "gc_threshold": gc.get_threshold(),
                "gc_mode": mode or "default",
                "gc_enabled": gc.isenabled(),
                "n_tracked_objects_after_imports": n_tracked,
                "full_collect_ms_at_init": full_collect_ms,
                "init_collect": os.environ.get("FRONTIER_SPIKE_INIT_COLLECT") == "1",
                "frozen": gc.get_freeze_count(),
            }
        )
        + "\n"
    )


def task_begin(num_tokens, tp):
    if not enabled:
        return
    global _task_idx, _task_t0, _task_wall0, _task_gc_count0
    _task_idx += 1
    del _fwd_times[:]
    del _gc_events[:]
    _task_gc_count0 = gc.get_count()
    if _task_idx == 0:
        # heap size after the lazy vllm/model imports that the first task triggers (one list allocation, once)
        _out.write(json.dumps({"kind": "heap", "gpu": _gpu_id, "n_tracked_objects_at_task0": len(gc.get_objects()), "gc_count": list(_task_gc_count0)}) + "\n")
    _task_wall0 = time.time()
    _task_t0 = time.perf_counter()


def forward_begin():
    _fwd_times.append(time.perf_counter())


def forward_end():
    _fwd_times.append(time.perf_counter())


def task_end(num_tokens, tp, result):
    if not enabled:
        return
    t_end = time.perf_counter()
    fwd = _fwd_times.tolist()
    rel = lambda t: round((t - _task_t0) * 1e3, 4)  # ms since task begin
    forwards = [[rel(fwd[i]), rel(fwd[i + 1])] for i in range(0, len(fwd) - 1, 2)]
    ev = _gc_events.tolist()
    gc_recs = [ev[i : i + 4] for i in range(0, len(ev), 4)]
    gc_events = []
    open_start = {}
    for t, phase, gen, collected in gc_recs:
        gen = int(gen)
        if phase == 0.0:
            open_start[gen] = t
        else:
            start = open_start.pop(gen, None)
            gc_events.append(
                {
                    "gen": gen,
                    "start_ms": rel(start) if start is not None else None,
                    "stop_ms": rel(t),
                    "dur_ms": round((t - start) * 1e3, 4) if start is not None else None,
                    "collected": int(collected),
                }
            )
    samples = {
        k.replace("time_stats.", ""): json.loads(v["samples"])
        for k, v in result["time_stats"].items()
        if isinstance(v, dict) and "samples" in v
    }
    rec = {
        "kind": "task",
        "gpu": _gpu_id,
        "task_idx": _task_idx,
        "num_tokens": num_tokens,
        "tp": tp,
        "wall_start": _task_wall0,
        "task_ms": rel(t_end),
        "gc_count_begin": list(_task_gc_count0),
        "gc_count_end": list(gc.get_count()),
        "forwards_ms": forwards,  # host-side [begin, end] of each of the 53 forward calls, ms since task begin
        "gc_events": gc_events,
        "gc_gen2_in_task": sum(1 for e in gc_events if e["gen"] == 2),
        "samples_ms": samples,
    }
    pre = samples.get("attn_pre_proj")
    if pre:
        timed = pre[3:]
        med = sorted(timed)[len(timed) // 2]
        if max(timed) > 20 * med:
            # post-hoc only, so this allocation does not perturb the trigger position
            import torch

            ms = torch.cuda.memory_stats()
            rec["spike"] = {"run": timed.index(max(timed)), "ms": max(timed), "median_ms": med}
            rec["memory_stats_after"] = {
                k: ms[k]
                for k in ms
                if k.startswith(("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms"))
                or k in ("allocated_bytes.all.current", "reserved_bytes.all.current", "reserved_bytes.all.peak", "segment.all.current", "segment.all.peak")
            }
    _out.write(json.dumps(rec, separators=(",", ":")) + "\n")
