"""Occurrence attribution independent of GPU execution wall-clock containment."""
from collections import defaultdict
from b_common import indexed, interval, require
from telemetry_b import ClockJoin


KERNEL_KEY = ["process_id", "gpu_id", "dispatch_id"]


def _forward_request_phases(bundle):
    """process_id/forward_id -> the single request phase (warmup/measurement)
    every sequence in that forward belongs to, from the independent request
    ledger -- not from a request-name convention (code review finding 4's
    acceptance criterion). A forward whose sequences span more than one
    phase is a real ambiguity this cannot resolve at the kernel level (a
    kernel is attributed to a scope, not to one sequence within a batch);
    fail loud rather than silently pick one."""
    requests_by_id = indexed(bundle["requests"], ["request_id"])
    phases = {}
    for f in bundle["forwards"]:
        seen = {requests_by_id[(s["request_id"],)]["phase"] for s in f["sequences"]
                if (s["request_id"],) in requests_by_id}
        require(len(seen) <= 1,
                f"forward {f['process_id']}/{f['forward_id']} mixes request phases: {seen}")
        phases[f["process_id"], f["forward_id"]] = next(iter(seen), None)
    return phases


def extract(bundle):
    manifest = bundle["manifest"]
    processes = indexed(manifest["processes"], ["process_id"])
    kernels = bundle["kernels"]
    indexed(kernels, KERNEL_KEY)
    apis = indexed(bundle["launches"], ["process_id", "correlation_id"])
    scopes = indexed(bundle["scopes"], ["process_id", "scope_id"])
    forward_request_phase = _forward_request_phases(bundle)
    indexed(bundle["ranges"], ["process_id", "marker_id"])
    ranges = defaultdict(list)
    for r in bundle["ranges"]:
        interval(r)
        ranges[r["process_id"], r["thread_id"]].append(r)
    for rows in ranges.values():
        # A stack must be nested, not crossing or duplicated with equal bounds.
        stack = []
        for r in sorted(rows, key=lambda r: (r["begin_ns"], -r["end_ns"])):
            while stack and stack[-1]["end_ns"] <= r["begin_ns"]:
                stack.pop()
            if stack:
                require(r["end_ns"] <= stack[-1]["end_ns"], "crossing host ranges")
                require((r["begin_ns"], r["end_ns"]) != (stack[-1]["begin_ns"], stack[-1]["end_ns"]),
                        "ambiguous equal host ranges")
            stack.append(r)
    clocks = ClockJoin(bundle["telemetry"], manifest["clock_calibrations"], manifest["clock_policy"])
    predecessors, by_queue = {}, defaultdict(list)
    for k in kernels:
        interval(k)
        require((k["process_id"],) in processes, "unknown kernel process")
        require(k["gpu_id"] == processes[k["process_id"],]["gpu_id"], "kernel rank/device mismatch")
        by_queue[k["process_id"], k["gpu_id"], k["queue_id"]].append(k)
    for rows in by_queue.values():
        previous = None
        for k in sorted(rows, key=lambda r: (r["begin_ns"], r["end_ns"], str(r["dispatch_id"]))):
            predecessors[tuple(k[x] for x in KERNEL_KEY)] = previous
            previous = k
    output, groups = [], defaultdict(list)
    for k in kernels:
        key = tuple(k[x] for x in KERNEL_KEY)
        row = dict(k, run_id=manifest["run_id"], scope_id=None, forward_id=None,
                   rank=processes[k["process_id"],]["rank"], request_phase=None,
                   kernel_dur_us=(k["end_ns"] - k["begin_ns"]) / 1000,
                   attribution="unresolved", exclusion_reason=None, attribution_range_id=None)
        predecessor = predecessors[key]
        row.update(predecessor_id=predecessor["dispatch_id"] if predecessor else None,
                   predecessor_name=predecessor["name"] if predecessor else None,
                   queue_interval_us=(k["begin_ns"] - predecessor["end_ns"]) / 1000 if predecessor else None,
                   predecessor_state="observed" if predecessor else "window_boundary")
        api = apis.get((k["process_id"], k["correlation_id"]))
        # Replay intentionally fails closed until capture/node identity is demonstrated.
        if k["execution"] != "eager" or (api and "GraphLaunch" in api["name"]):
            row["attribution"] = "graph_unresolved"
        elif api:
            interval(api)
            require(api["native_pid"] == str(processes[k["process_id"],]["native_pid"]),
                    "launch process identity mismatch")
            candidates = [r for r in ranges[k["process_id"], api["thread_id"]]
                          if r["begin_ns"] <= api["begin_ns"] and api["end_ns"] <= r["end_ns"]]
            targets = [r for r in candidates if r.get("scope_id") is not None]
            exclusions = [r for r in candidates if r.get("exclusion_reason") is not None]
            # Nested target scopes are physically ambiguous; never silently double count.
            if len(targets) == 1 and not exclusions:
                r = targets[0]
                scope = scopes.get((k["process_id"], r["scope_id"]))
                require(scope is not None, "range refers to missing scope metadata")
                forward_ranges = [x for x in candidates if x.get("kind") == "forward"]
                require(len(forward_ranges) == 1 and forward_ranges[0]["forward_id"] == scope["forward_id"],
                        "scope launch lacks matching enclosing forward range")
                row.update(scope_id=r["scope_id"], forward_id=scope["forward_id"],
                           physical_scope=scope["physical_scope"], layer=scope["layer"],
                           variant=scope["variant"], attribution="eager_correlation",
                           attribution_range_id=r["marker_id"], launch_ref=api["raw_ref"],
                           request_phase=forward_request_phase.get((k["process_id"], scope["forward_id"])))
            elif not targets and exclusions:
                r = min(exclusions, key=lambda r: r["end_ns"] - r["begin_ns"])
                row.update(attribution="excluded", exclusion_reason=r["exclusion_reason"],
                           attribution_range_id=r["marker_id"], launch_ref=api["raw_ref"])
            elif targets:
                row["attribution"] = "ambiguous"
        row.update(clocks.join(k))
        output.append(row)
        if row["scope_id"] is not None:
            groups[k["process_id"], row["scope_id"]].append(row)
    summaries = []
    for key, rows in groups.items():
        scope = scopes[key]
        start = min(r["begin_ns"] for r in rows)
        end = max(r["end_ns"] for r in rows)
        # Interval union describes covered device time, distinct from a sum and span.
        union, right = 0, start
        for r in sorted(rows, key=lambda r: r["begin_ns"]):
            union += max(0, r["end_ns"] - max(right, r["begin_ns"]))
            right = max(right, r["end_ns"])
        summaries.append(dict(scope, run_id=manifest["run_id"], n_kernels=len(rows),
                              begin_ns=start, end_ns=end, scope_span_us=(end - start) / 1000,
                              kernel_sum_us=sum(r["kernel_dur_us"] for r in rows),
                              kernel_union_us=union / 1000,
                              # Every kernel in one scope invocation shares one forward, and
                              # therefore one request phase (code review finding 4): downstream
                              # analysis can filter warmup out of measurement statistics by this
                              # field instead of by a request-id naming convention.
                              request_phase=rows[0]["request_phase"]))
    return dict(kernels=output, scopes=summaries)
