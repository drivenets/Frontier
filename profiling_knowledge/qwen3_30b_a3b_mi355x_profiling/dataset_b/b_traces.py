"""Conservative trace adapters. Native records and source row references survive.

CSV dialect is evidenced by the repository's older rocprof integration trace,
not yet verified against the approved serving image. No graph name matching.
"""
import csv
from decimal import Decimal
from b_common import Invalid, interval, require


def rocprof_csv(kernel_path, api_path, process, agents):
    kernels, launches = [], []
    with open(kernel_path, newline="") as f:
        for number, r in enumerate(csv.DictReader(f), 2):
            require(r["Agent_Id"] in agents, "unmapped native GPU agent")
            kernels.append(dict(process_id=process, gpu_id=agents[r["Agent_Id"]],
                                dispatch_id=r["Dispatch_Id"], correlation_id=r["Correlation_Id"],
                                queue_id=r["Queue_Id"], stream_id=r["Stream_Id"],
                                name=r["Kernel_Name"], begin_ns=int(r["Start_Timestamp"]),
                                end_ns=int(r["End_Timestamp"]), execution="eager",
                                raw_ref=f"{kernel_path}:{number}", native=r))
    with open(api_path, newline="") as f:
        for number, r in enumerate(csv.DictReader(f), 2):
            launches.append(dict(process_id=process, native_pid=r["Process_Id"],
                                 thread_id=r["Thread_Id"], correlation_id=r["Correlation_Id"],
                                 name=r["Function"], begin_ns=int(r["Start_Timestamp"]),
                                 end_ns=int(r["End_Timestamp"]),
                                 raw_ref=f"{api_path}:{number}", native=r))
    require(kernels and launches, "empty rocprof input")
    # A graph launch needs information this eager CSV adapter does not retain.
    graph_corr = {a["correlation_id"] for a in launches if "GraphLaunch" in a["name"]}
    for k in kernels:
        if k["correlation_id"] in graph_corr:
            k["execution"] = "graph_unresolved"
    return kernels, launches


def rocprof_marker_csv(path, process):
    """rocprofv3 CSV marker dialect, verified on the approved image/toolchain
    (dataset_b/evidence/rocprofv3-format-probe2-21737): one row IS one complete
    push/pop range (Domain=MARKER_CORE_RANGE_API), not a push/pop event stream.
    pair_markers() below is for a discrete event-stream dialect this tool does
    not emit in CSV mode; do not route rocprofv3 CSV marker rows through it.

    MARKER_CONTROL_API rows (verified: job 21798, exactly one roctxProfilerResume
    and one roctxProfilerPause row, each from marker_runtime.py's own
    start_measurement()/stop_measurement() calls that bracket the measured
    native trace window -- see that module) are a real rocprofv3 marker
    domain, not a push/pop range: no B|scope|/B|fwd| label, no bind_labels
    use. Skipped here, not routed anywhere else.
    """
    ranges = []
    with open(path, newline="") as f:
        for number, r in enumerate(csv.DictReader(f), 2):
            if r["Domain"] == "MARKER_CONTROL_API":
                continue
            require(r["Domain"] == "MARKER_CORE_RANGE_API", f"unexpected marker domain: {r['Domain']}")
            row = dict(process_id=process, thread_id=r["Thread_Id"], marker_id=r["Correlation_Id"],
                       label=r["Function"], begin_ns=int(r["Start_Timestamp"]), end_ns=int(r["End_Timestamp"]),
                       raw_ref=f"{path}:{number}", native=r)
            interval(row)
            ranges.append(row)
    require(ranges, "empty rocprof marker input")
    return ranges


def chrome_ns(value):
    v = Decimal(str(value)) * 1000
    require(v == v.to_integral_value(), "sub-nanosecond Chrome timestamp")
    return int(v)


def kineto(trace, process, devices):
    """Chrome complete events only; ambiguous/incomplete marker forms fail closed.

    Identity fallback is immutable file event index when native dispatch is absent.
    Device mapping is explicit; pid on a GPU event is NOT an OS process id.
    """
    kernels, launches, ranges = [], [], []
    require(isinstance(trace.get("traceEvents"), list), "missing traceEvents")
    for i, e in enumerate(trace["traceEvents"]):
        category = e.get("cat", "")
        marker = e.get("name", "").startswith("B|")
        if category not in ("kernel", "cuda_runtime", "hip_runtime") and not marker:
            continue
        require(e.get("ph") == "X", "unsupported or incomplete trace event")
        base = dict(process_id=process, begin_ns=chrome_ns(e["ts"]),
                    end_ns=chrome_ns(Decimal(str(e["ts"])) + Decimal(str(e["dur"]))),
                    name=e["name"], raw_ref=f"traceEvents[{i}]", native=e)
        args = e.get("args", {})
        if category == "kernel":
            device = str(args["device"])
            require(device in devices, "unmapped Kineto device")
            base.update(gpu_id=devices[device], dispatch_id=f"event:{i}",
                        correlation_id=str(args["correlation"]),
                        queue_id=str(args["stream"]), stream_id=str(args["stream"]),
                        execution="eager")
            kernels.append(base)
        elif marker:
            base.update(thread_id=str(e["tid"]), marker_id=f"event:{i}", label=e["name"])
            ranges.append(base)
        else:
            base.update(thread_id=str(e["tid"]), native_pid=str(e["pid"]),
                        correlation_id=str(args["correlation"]))
            launches.append(base)
    require(kernels and launches, "empty Kineto device/API trace")
    graph_corr = {a["correlation_id"] for a in launches if "GraphLaunch" in a["name"]}
    for k in kernels:
        if k["correlation_id"] in graph_corr:
            k["execution"] = "graph_unresolved"
    return kernels, launches, ranges


def pair_markers(events):
    """Pair normalized roctx push/pop events, separately per process/thread.

    The pinned native roctx dialect must be inspected before an adapter is added.
    Cross-thread range start/stop IDs are deliberately unsupported.
    """
    stacks, ranges = {}, []
    for e in sorted(events, key=lambda r: (r["timestamp_ns"], r["source_index"])):
        key = e["process_id"], e["thread_id"]
        stack = stacks.setdefault(key, [])
        if e["operation"] == "push":
            stack.append(e)
        elif e["operation"] == "pop":
            require(stack, "unmatched marker pop")
            start = stack.pop()
            ranges.append(dict(process_id=key[0], thread_id=key[1],
                               marker_id=start["marker_id"], label=start["label"],
                               begin_ns=start["timestamp_ns"], end_ns=e["timestamp_ns"],
                               raw_ref=[start["raw_ref"], e["raw_ref"]]))
        else:
            raise Invalid("unsupported marker operation")
    require(not any(stacks.values()), "unclosed marker range")
    for r in ranges:
        interval(r)
    return ranges
