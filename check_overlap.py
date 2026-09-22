#!/usr/bin/env python3
"""Detect concurrency in a Frontier stage-batch ledger.

Usage:  python3 check_overlap.py <path-to-frontier_stage_batch_ledger.jsonl>

Answers one question: does Frontier's event queue represent operations as
overlapping in simulated time, or does it serialise everything?
"""
import sys, json, collections

path = sys.argv[1] if len(sys.argv) > 1 else "frontier_stage_batch_ledger.jsonl"

recs = []
for ln, line in enumerate(open(path), 1):
    line = line.strip()
    if not line:
        continue
    try:
        recs.append(json.loads(line))
    except json.JSONDecodeError as e:
        print(f"  ! skipped malformed line {ln}: {e}")

print(f"records: {len(recs)}\n")

EPS = 1e-12

# ---- group intervals by cluster, and by (cluster, replica) ----
by_cluster = collections.defaultdict(list)
by_replica = collections.defaultdict(list)
for r in recs:
    iv = (r.get("stage_start_ts"), r.get("stage_end_ts"))
    if iv[0] is None or iv[1] is None:
        continue
    ct = r.get("cluster_type", "?")
    by_cluster[ct].append(iv)
    by_replica[(ct, r.get("replica_id"), r.get("dp_id"))].append(iv)


def max_concurrency(intervals):
    """Peak number of simultaneously-open intervals (sweep line)."""
    ev = []
    for s, e in intervals:
        ev.append((s, 1))
        ev.append((e, -1))
    ev.sort(key=lambda x: (x[0], x[1]))    # closes before opens at equal ts
                                           # (touching intervals are NOT concurrent)
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def self_overlaps(intervals):
    iv = sorted(intervals)
    return sum(1 for i in range(len(iv) - 1) if iv[i + 1][0] < iv[i][1] - EPS)


print("=== per cluster ===")
for ct, iv in sorted(by_cluster.items()):
    span = (min(s for s, _ in iv), max(e for _, e in iv))
    busy = sum(e - s for s, e in iv)
    print(f"{ct:14s} stages={len(iv):6d}  overlapping_pairs={self_overlaps(iv):6d} "
          f" peak_concurrency={max_concurrency(iv):3d}  span={span[0]:.3f}..{span[1]:.3f}"
          f"  busy={busy:.3f}")

print("\n=== per (cluster, replica, dp) ===")
for k, iv in sorted(by_replica.items(), key=lambda x: str(x[0])):
    print(f"{str(k):34s} stages={len(iv):6d} peak_concurrency={max_concurrency(iv):3d}")


def cross(a_list, b_list):
    """Count overlapping pairs across two interval sets, O(n log n)."""
    if not a_list or not b_list:
        return 0
    b = sorted(b_list)
    starts = [s for s, _ in b]
    import bisect
    # sweep: for each a, count b intervals that start before a ends and end after a starts
    b_sorted_by_end = sorted(b, key=lambda x: x[1])
    total = 0
    for s, e in a_list:
        hi = bisect.bisect_left(starts, e)          # b intervals starting before a ends
        cnt = sum(1 for bs, be in b[:hi] if be > s + EPS)
        total += cnt
    return total


print("\n=== cross-cluster overlap (THE AFD QUESTION) ===")
cl = sorted(by_cluster)
for i in range(len(cl)):
    for j in range(i + 1, len(cl)):
        n = cross(by_cluster[cl[i]], by_cluster[cl[j]])
        print(f"{cl[i]:14s} x {cl[j]:14s}  overlapping pairs = {n}")

print("\n=== verdict ===")
any_self = any(self_overlaps(iv) for iv in by_cluster.values())
any_cross = any(cross(by_cluster[cl[i]], by_cluster[cl[j]])
                for i in range(len(cl)) for j in range(i + 1, len(cl)))
if any_cross:
    print("CONCURRENT: stages in different clusters overlap in simulated time.")
    print("-> Frontier's single global event queue represents concurrency. Architecture holds.")
elif any_self:
    print("PARTIAL: overlap within a cluster only, none across clusters.")
    print("-> Investigate: may be a scheduling artefact of this config, not a limitation.")
else:
    print("SERIAL: no overlap anywhere in this run.")
    print("-> Inconclusive on its own. Re-run with micro-batching and higher QPS before concluding.")
