"""Assemble retained native traces and independently collected metadata.

Native roctx CSV pairing is intentionally pending image inspection; supply
normalized paired ranges or normalized push/pop events with raw references.
"""
import argparse
from pathlib import Path
from b_common import digest, read_json, require, write_json
from b_traces import kineto, pair_markers, rocprof_csv, rocprof_marker_csv


def bind_labels(ranges, scopes):
    """Static scope labels acquire invocation IDs from enclosing forward markers."""
    forwards = []
    for r in ranges:
        label = r.get("label", "")
        if label.startswith("B|fwd|"):
            r.update(kind="forward", forward_id=label.split("|", 2)[2])
            forwards.append(r)
    for r in ranges:
        parts = r.get("label", "").split("|")
        if parts[:2] != ["B", "scope"]:
            continue
        require(len(parts) == 5, "invalid static scope marker")
        _, _, physical, layer, variant = parts
        layer = None if layer == "model" else int(layer)
        parents = [f for f in forwards if f["process_id"] == r["process_id"]
                   and f["thread_id"] == r["thread_id"] and f["begin_ns"] <= r["begin_ns"]
                   and r["end_ns"] <= f["end_ns"]]
        require(len(parents) == 1, "scope marker lacks unique forward parent")
        matches = [s for s in scopes if s["process_id"] == r["process_id"]
                   and s["forward_id"] == parents[0]["forward_id"]
                   and (s["physical_scope"], s["layer"], s["variant"]) == (physical, layer, variant)]
        require(len(matches) == 1, "scope marker lacks unique metadata invocation")
        r["scope_id"] = matches[0]["scope_id"]
    return ranges


def normalize(metadata_path):
    b = read_json(metadata_path)
    base = Path(metadata_path).resolve().parent
    b["kernels"], b["launches"] = [], []
    ranges = b.pop("ranges", [])
    files = []

    def path(relative):
        p = base / relative
        files.append(dict(path=str(p), sha256=digest(p)))
        return p

    for source in b.pop("process_sources"):
        pid = source["process_id"]
        if source["format"] == "rocprof_csv_eager":
            kernels, launches = rocprof_csv(path(source["kernels"]), path(source["api"]), pid, source["agents"])
            # Two distinct, non-interchangeable marker dialects: "marker_events" is a discrete
            # normalized push/pop event stream (pair_markers' stack-based pairing); "marker_csv" is
            # rocprofv3's actual CSV marker output, where each row is already one complete paired
            # range (verified: dataset_b/evidence/rocprofv3-format-probe2-21737). Do not route one
            # through the other's adapter.
            if source.get("marker_events"):
                ranges.extend(pair_markers(read_json(path(source["marker_events"]))))
            if source.get("marker_csv"):
                ranges.extend(rocprof_marker_csv(path(source["marker_csv"]), pid))
        elif source["format"] == "kineto_chrome":
            kernels, launches, rs = kineto(read_json(path(source["trace"])), pid, source["devices"])
            ranges.extend(rs)
        else:
            raise ValueError("unsupported native trace format; do not guess a dialect")
        b["kernels"].extend(kernels)
        b["launches"].extend(launches)
    b["ranges"] = bind_labels(ranges, b["scopes"])
    b["manifest"]["raw_files"] = files + [dict(path=str(metadata_path), sha256=digest(metadata_path))]
    # Counts/drop diagnostics MUST come from independent trace diagnostics, not this adapter.
    return b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metadata")
    p.add_argument("output")
    a = p.parse_args()
    write_json(a.output, normalize(a.metadata))


if __name__ == "__main__":
    main()
