"""Assemble one real pilot run's artifacts into normalize_b.py's input bundle.

For each rank, combines the rocprofv3 CSV trace triple (kernel/hip_api/marker,
under <rocprof_trace_dir>/<hostname>/<N>_*_trace.csv, found by matching the
rank's native PID inside hip_api_trace.csv rather than guessing the file
naming) and that rank's atexit-flushed CPU metadata
(<b_metadata_dir>/metadata-pid<native_pid>.json, split by "kind" into
scheduler/forward/scope/calibration_anchor records) -- plus the run-wide
driver requests.json and telemetry sampler JSONL -- into the JSON
normalize_b.py expects as input. Never guesses a rank<->GPU mapping or a
rank<->metadata-file mapping; every rank is given explicitly via --rank.
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

from b_common import digest, read_json, require, write_json
from calibrate_b import calibrate


def load_cpu_metadata(path, expected_capture_id):
    """Only a tier-1 confirmed_drain() file is accepted here: an "emergency"
    (atexit-tier) flush never demonstrated a safe boundary (see
    marker_runtime.py's module docstring) and must not be silently treated as
    equivalent -- the caller gets a clear, actionable error instead of a
    quietly degraded bundle."""
    data = read_json(path)
    require(data.get("drain_mode") == "confirmed",
            f"{path}: not a confirmed drain (drain_mode={data.get('drain_mode')!r}); "
            "an emergency/atexit flush cannot be used to assemble a bundle")
    require(data.get("capture_id") == expected_capture_id,
            f"{path}: capture_id {data.get('capture_id')!r} != expected {expected_capture_id!r}")
    by_kind = {"scheduler": [], "forward": [], "scope": [], "calibration_anchor": []}
    for r in data["records"]:
        require(r["kind"] in by_kind, f"unknown CPU metadata record kind: {r['kind']}")
        by_kind[r["kind"]].append({k: v for k, v in r.items() if k != "kind"})
    require(not data["overflow"], f"marker buffer overflowed during the run ({path})")
    require(data.get("process_id"), f"{path}: missing top-level process_id")
    return by_kind, data["native_pid"], data["backend"], data["process_id"]


def find_rocprof_prefix(trace_dir, native_pid):
    """rocprofv3 -f csv writes <trace_dir>/<hostname>/<N>_{kernel,hip_api,
    marker_api,agent_info}_trace.csv where N is rocprofv3's own internal
    numbering, not the traced process's PID -- identify the right file set by
    matching a hip_api_trace.csv row's Process_Id to native_pid instead.

    This IS this assembler's native-trace-belongs-to-this-capture check (the
    native tracer itself has no notion of capture_id -- it is finalized by
    rocprofv3 as the wrapped process's parent, independently of
    confirmed_drain(), per marker_runtime.py's module docstring). The
    confirmed_drain() metadata file supplies native_pid; require() below
    fails loud if zero or more than one trace file set matches it, rather
    than guessing which trace belongs to this run's capture."""
    candidates = sorted(glob.glob(os.path.join(trace_dir, "*", "*_hip_api_trace.csv")))
    require(candidates, f"no rocprof hip_api_trace.csv under {trace_dir}")
    matches = []
    for api_path in candidates:
        with open(api_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["Process_Id"].strip() == str(native_pid):
                    matches.append(api_path[: -len("hip_api_trace.csv")])
                    break
    require(len(matches) == 1,
            f"expected exactly one rocprof trace set for pid {native_pid}, found {len(matches)}: {matches}")
    return matches[0]


def load_telemetry(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    require(samples, f"empty telemetry sampler output: {path}")
    return samples


def load_requests(driver_output_dirs):
    """drive_b.py's requests.json rows -> b_check.py's request-ledger schema.

    drive_b.py is invoked once per phase present in the request manifest
    (run_b.py: "for phase in (warmup, measurement)"), each writing its own
    requests.json into a separate <output>/<phase> directory. Every phase
    actually driven must be loaded, or the scheduler/trace evidence for a
    real, independently-tracked request (e.g. warmup-0) has no ledger entry
    and b_check.py's reconciliation fails as if it were unexplained traffic."""
    out = []
    seen = set()
    for driver_output_dir in driver_output_dirs:
        rows = read_json(Path(driver_output_dir) / "requests.json")
        for r in rows:
            require(r["status"] == "complete",
                    f"driver request {r['request_id']} did not complete: {r.get('error')}")
            require(r["request_id"] not in seen, f"duplicate request_id across phases: {r['request_id']}")
            seen.add(r["request_id"])
            out.append(dict(
                request_id=r["request_id"], phase=r["phase"],
                input_ids=r["input_ids"], output_ids=r["output_ids"],
                prompt_tokens=r["prompt_tokens"], generated_tokens=r["generated_tokens"],
                requested_new_tokens=r["requested_new_tokens"], status=r["status"],
            ))
    return out


def forward_marker_trace_ns(marker_csv_path):
    """Map forward_id -> trace_ns (this rank's B|fwd|<id> marker row's own
    Start_Timestamp) for every forward marker in this rank's trace. Begin
    timestamp, not the push/pop midpoint: begin_forward() in marker_runtime.py
    records host_begin_ns immediately before the same push call.

    Also resolves the "__drain__" sentinel anchor's trace_ns from rocprofv3's
    MARKER_CONTROL_API roctxProfilerPause row -- marker_runtime.py's
    stop_measurement() emits exactly one such row per capture, bracketing the
    true end of the measured window (code review finding 3)."""
    trace_ns = {}
    with open(marker_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = row["Function"]
            if label.startswith("B|fwd|"):
                # Keep as a string: forward_id is a label/identifier (matching
                # marker_runtime.py's convention), not a number to be sorted
                # or compared arithmetically. calibration_anchor records key
                # on this same string.
                forward_id = label.split("|", 2)[2]
                require(forward_id not in trace_ns, f"duplicate B|fwd|{forward_id} marker row")
                trace_ns[forward_id] = int(row["Start_Timestamp"])
            elif row["Domain"] == "MARKER_CONTROL_API" and label == "roctxProfilerPause":
                require("__drain__" not in trace_ns, "duplicate roctxProfilerPause marker row")
                trace_ns["__drain__"] = int(row["Start_Timestamp"])
    return trace_ns


def kernel_window_and_count(kernel_csv_path):
    """raw_kernel_count and [begin_ns, end_ns] straight from the rocprofv3
    kernel trace -- b_check.py requires this independent of attribution."""
    begin = end = None
    n = 0
    with open(kernel_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            n += 1
            b, e = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            begin = b if begin is None else min(begin, b)
            end = e if end is None else max(end, e)
    require(n > 0, f"empty kernel trace: {kernel_csv_path}")
    return n, begin, end


def build_rank(rank_cfg, capture_id):
    """rank_cfg: {rank, native_pid, gpu_id, b_metadata_json, rocprof_trace_dir,
    complete, drained, metadata_complete, dropped_records} -- the last four
    are the orchestrator's own honest account of this rank's shutdown (did the
    server exit cleanly or get force-killed?), not re-derived here.

    process_id is NOT invented here (e.g. as "rank{N}") -- it must be the
    exact opaque id marker_runtime.py's begin_scope/begin_forward stamped
    onto their own records (now also at the metadata file's top level), or
    normalize_b.bind_labels' per-process matching silently fails to join
    anything (every scope/forward record would carry a different process_id
    than the trace-derived ranges, and bind_labels requires an exact match)."""
    metadata, flushed_pid, backend, process_id = load_cpu_metadata(rank_cfg["b_metadata_json"], capture_id)
    require(flushed_pid == rank_cfg["native_pid"],
            f"metadata file's own native_pid ({flushed_pid}) != declared rank native_pid ({rank_cfg['native_pid']})")
    require(backend == "roctx", f"only the roctx tracer path is wired up in this assembler, got {backend}")
    prefix = find_rocprof_prefix(rank_cfg["rocprof_trace_dir"], rank_cfg["native_pid"])
    agents = {}
    with open(prefix + "agent_info.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["Agent_Type"] == "GPU":
                # rocprofv3's own "Gpu_Id" is an opaque per-boot handle in a
                # different namespace than this run's telemetry gpu_id (the
                # PCI bus id run_b.sbatch resolved via rocm-smi/sysfs). Derive
                # the same PCI bus id here from Location_Id/Domain (standard
                # KFD packed BDF: bus = loc>>8, device = (loc&0xFF)>>3,
                # function = loc&0x7) so kernels and the telemetry-mapped
                # process share one identifier space -- verified against this
                # node's agent_info.csv: Node_Id 2, Location_Id 1280, Domain 0
                # decodes to pci:0000:05:00.0, matching amd-mi355x-3's actual
                # allocated GPU bus id.
                loc = int(row["Location_Id"])
                bus, devfn = loc >> 8, loc & 0xFF
                device, function = devfn >> 3, devfn & 0x7
                pci_id = f"pci:{int(row['Domain']):04x}:{bus:02x}:{device:02x}.{function:x}"
                agents[f"Agent {row['Node_Id']}"] = pci_id
    require(agents, f"no GPU agents found for rank {rank_cfg['rank']}")
    require(rank_cfg["gpu_id"] in agents.values(),
            f"rank {rank_cfg['rank']}'s declared gpu_id {rank_cfg['gpu_id']!r} matches none of this "
            f"trace's GPU agents ({sorted(agents.values())}); telemetry/kernel GPU identity disagree")
    source = dict(
        process_id=process_id, format="rocprof_csv_eager",
        kernels=prefix + "kernel_trace.csv", api=prefix + "hip_api_trace.csv",
        marker_csv=prefix + "marker_api_trace.csv", agents=agents,
    )
    process = dict(
        process_id=process_id, native_pid=rank_cfg["native_pid"], rank=rank_cfg["rank"],
        gpu_id=rank_cfg["gpu_id"],
    )
    trace_ns_by_forward = forward_marker_trace_ns(prefix + "marker_api_trace.csv")
    anchors = []
    for a in metadata["calibration_anchor"]:
        require(a["forward_id"] in trace_ns_by_forward,
                f"calibration anchor for forward {a['forward_id']} has no matching B|fwd| marker row")
        anchors.append(dict(trace_ns=trace_ns_by_forward[a["forward_id"]],
                             host_begin_ns=a["host_begin_ns"], host_end_ns=a["host_end_ns"]))
    raw_kernel_count, begin_ns, end_ns = kernel_window_and_count(prefix + "kernel_trace.csv")
    window = dict(process_id=process_id, begin_ns=begin_ns, end_ns=end_ns,
                  complete=rank_cfg["complete"], drained=rank_cfg["drained"],
                  metadata_complete=rank_cfg["metadata_complete"])
    diagnostics = dict(process_id=process_id, raw_kernel_count=raw_kernel_count,
                        dropped_records=rank_cfg["dropped_records"])
    return source, process, metadata, anchors, window, diagnostics


def assemble(run_dir, driver_output_dirs, telemetry_path, ranks, static, output, evidence_ref,
             capture_id, holdout_fraction=0.2):
    """ranks: list of rank_cfg dicts (see build_rank). static: dict with the
    provenance/manifest fields that do not come from run artifacts (image
    digest, source hashes, patch sha256, model/tokenizer revision, effective
    config, scope_spec_version, data_kind, tp, model dims). evidence_ref
    labels the calibration run for manifest.clock_calibrations[*].evidence_ref
    and manifest.clock_policy.evidence_ref. capture_id must match every
    rank's own confirmed_drain() record (run_b.py's drain_acks.json is the
    source of truth for what capture_id a given run actually used).
    """
    process_sources, processes, scheduler, forwards, scopes = [], [], [], [], []
    windows, trace_diagnostics = [], []
    anchors_by_process, gpu_ids = {}, {}
    for rank_cfg in ranks:
        source, process, metadata, anchors, window, diagnostics = build_rank(rank_cfg, capture_id)
        process_id = process["process_id"]
        process_sources.append(source)
        processes.append(process)
        windows.append(window)
        trace_diagnostics.append(diagnostics)
        gpu_ids[process_id] = process["gpu_id"]
        # scheduler records are cross-rank (keyed by global_forward_id only);
        # take them from rank 0 and sanity-check other ranks agree, rather
        # than silently duplicating or trusting one rank blindly.
        if rank_cfg["rank"] == 0:
            scheduler.extend(metadata["scheduler"])
        else:
            require(metadata["scheduler"] == [], "non-zero rank unexpectedly produced its own scheduler records")
        forwards.extend(dict(f, process_id=process_id) for f in metadata["forward"])
        scopes.extend(metadata["scope"])
        anchors_by_process[process_id] = anchors

    telemetry = load_telemetry(telemetry_path)
    requests = load_requests(driver_output_dirs)

    # One calibration fit per physical GPU (processes can share a GPU only in
    # a misconfigured run, which b_check.py's "ranks share GPU identity"
    # check catches elsewhere); hold out a fraction of anchors per GPU for a
    # residual check this function reports but does not gate on --
    # calibrate_b.calibrate()'s own docstring says to retain held-out
    # evidence, not that this script decides acceptance.
    #
    # Code review finding 3: the two endpoint anchors (chronologically first
    # and last) are never held out. calibrate_b.calibrate() sets
    # valid_trace_begin_ns/end_ns from the fit set's own first/last anchor,
    # so holding out the tail (the previous behavior) shrank the fitted
    # calibration's valid range below the true capture window, wrongly
    # marking later in-window kernels outside_calibration. Holdout is drawn
    # only from the interior, evenly spaced, so the fit's valid range always
    # brackets the complete capture while still exercising a real held-out
    # residual check.
    clock_calibrations, holdout_report = [], {}
    by_gpu = {}
    for pid, anchors in anchors_by_process.items():
        by_gpu.setdefault(gpu_ids[pid], []).extend(anchors)
    for gpu_id, anchors in by_gpu.items():
        anchors = sorted(anchors, key=lambda a: a["trace_ns"])
        n = len(anchors)
        interior = list(range(1, n - 1))
        n_holdout = min(max(0, int(n * holdout_fraction)), len(interior))
        if n_holdout:
            step = len(interior) / n_holdout
            holdout_idx = {interior[int(i * step)] for i in range(n_holdout)}
        else:
            holdout_idx = set()
        fit_set = [a for i, a in enumerate(anchors) if i not in holdout_idx]
        holdout_set = [a for i, a in enumerate(anchors) if i in holdout_idx]
        require(len(fit_set) >= 3, f"fewer than 3 calibration anchors available to fit gpu {gpu_id}")
        cal = calibrate(fit_set, gpu_id, evidence_ref)
        residuals = []
        for a in holdout_set:
            predicted_host = cal["host_anchor_ns"] + round((a["trace_ns"] - cal["trace_anchor_ns"]) * cal["scale"])
            observed_host = (a["host_begin_ns"] + a["host_end_ns"]) // 2
            residuals.append(abs(predicted_host - observed_host))
        holdout_report[gpu_id] = dict(n_fit=len(fit_set), n_holdout=len(holdout_set),
                                       max_holdout_residual_ns=max(residuals) if residuals else None)
        clock_calibrations.append(cal)

    # Freeze clock_policy bounds from this run's own telemetry/calibration
    # evidence (B_SCHEMA.md: "Freeze numeric values from a calibration subset
    # before acceptance") rather than a guessed constant: max_age_ns/max_gap_ns
    # are 2x the sampler's own observed read interval (matching the synthetic
    # fixture's documented "2x interval" convention: tests/fixtures.py uses
    # a 50ms interval -> 100ms bounds), and max_uncertainty_ns covers this
    # run's worst-case calibration uncertainty plus half its slowest observed
    # telemetry read span, with headroom.
    begins_by_gpu = {}
    spreads = []
    for s in telemetry:
        begins_by_gpu.setdefault(s["gpu_id"], []).append(s["read_begin_ns"])
        spreads.append(s["read_end_ns"] - s["read_begin_ns"])
    gaps = []
    for begins in begins_by_gpu.values():
        begins = sorted(begins)
        gaps.extend(b2 - b1 for b1, b2 in zip(begins, begins[1:]))
    require(gaps, "no telemetry sample gaps to derive a clock policy from")
    gaps.sort()
    sampler_interval_ns = gaps[len(gaps) // 2]
    bound_ns = 2 * sampler_interval_ns
    worst_uncertainty_ns = max(
        cal["uncertainty_ns"] + (max(spreads) + 1) // 2 for cal in clock_calibrations
    )
    clock_policy = dict(
        frozen=True, evidence_ref=evidence_ref,
        max_age_ns=bound_ns, max_gap_ns=bound_ns,
        max_uncertainty_ns=worst_uncertainty_ns + sampler_interval_ns,
    )
    manifest = dict(static, processes=processes, raw_files=[], clock_calibrations=clock_calibrations,
                     windows=windows, trace_diagnostics=trace_diagnostics, clock_policy=clock_policy)
    write_json(output, dict(
        manifest=manifest, kernels=[], launches=[], ranges=[],
        scopes=scopes, forwards=forwards, scheduler=scheduler,
        requests=requests, telemetry=telemetry,
        process_sources=process_sources,
    ))
    write_json(str(output) + ".calibration_holdout.json", holdout_report)
    return holdout_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config",
                   help="JSON: {run_dir, driver_output_dirs, telemetry_path, ranks, static, evidence_ref, capture_id}")
    p.add_argument("output")
    a = p.parse_args()
    cfg = read_json(a.config)
    report = assemble(cfg["run_dir"], cfg["driver_output_dirs"], cfg["telemetry_path"],
                       cfg["ranks"], cfg["static"], a.output, cfg["evidence_ref"],
                       cfg["capture_id"], cfg.get("holdout_fraction", 0.2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
