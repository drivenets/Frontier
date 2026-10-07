"""Fit trace-to-monotonic anchors; calibration never silently extrapolates."""
import argparse
from b_common import read_json, require, write_json, integer


def calibrate(anchors, gpu_id, evidence_ref):
    """Each native timestamp must be independently bracketed in host time.

    Caller verifies that the tracer's host marker and device dispatch timestamps
    share a clock domain on the pinned toolchain. This function cannot prove it.
    """
    require(len(anchors) >= 3 and evidence_ref, "at least three evidenced clock anchors required")
    rows = sorted(anchors, key=lambda a: a["trace_ns"])
    for r in rows:
        integer(r["trace_ns"], "trace anchor")
        integer(r["host_begin_ns"], "host anchor begin")
        integer(r["host_end_ns"], "host anchor end", r["host_begin_ns"])
    require(len({r["trace_ns"] for r in rows}) == len(rows), "duplicate clock anchors")
    first, last = rows[0], rows[-1]
    x0 = first["trace_ns"]
    y0 = (first["host_begin_ns"] + first["host_end_ns"]) // 2
    yn = (last["host_begin_ns"] + last["host_end_ns"]) // 2
    scale = (yn - y0) / (last["trace_ns"] - x0)
    require(scale > 0, "nonmonotonic clock anchors")
    residual = max(abs((r["host_begin_ns"] + r["host_end_ns"]) // 2 -
                       (y0 + round((r["trace_ns"] - x0) * scale))) +
                   (r["host_end_ns"] - r["host_begin_ns"] + 1) // 2 for r in rows)
    return dict(gpu_id=gpu_id, scale=scale, trace_anchor_ns=x0, host_anchor_ns=y0,
                uncertainty_ns=residual, valid_trace_begin_ns=x0,
                valid_trace_end_ns=last["trace_ns"], evidence_ref=evidence_ref,
                anchor_count=len(rows), method="endpoint-affine/max-observed-residual-plus-read-bracket",
                limitation="Observed alignment bound, not a guarantee between anchors; validate held-out anchors.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("anchors")
    p.add_argument("output")
    p.add_argument("--gpu-id", required=True)
    p.add_argument("--evidence-ref", required=True)
    a = p.parse_args()
    write_json(a.output, calibrate(read_json(a.anchors), a.gpu_id, a.evidence_ref))


if __name__ == "__main__":
    main()
