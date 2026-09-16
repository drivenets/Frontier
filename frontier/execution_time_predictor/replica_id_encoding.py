"""Track B Step 48: the one encoding `ATTN_TP`'s own combined
(global replica id, DP lane) identity uses, canonicalised out of the
four-or-five independent inline copies Track B Step 47 counted (one on
`dc-sim`'s own registration side, `comm_groups.py`; the rest inline in
this predictor's own `_predict_comm_operator` and three MTP
synthetic-batch call sites).

Every caller -- registration and resolution alike -- must call this
function rather than reimplement its arithmetic. That duplication is
exactly the condition that produced both replica-identity mismatches
this track has chased (Task 41's original registry collision, and Track
B Step 46's own since-corrected "second mismatch" report): independently
-derived schemes that happen to agree in whatever configuration someone
tested, until one drifts.

`EP`/`MOE_TP`-domain groups never need the DP-lane component -- an FFN
replica is always a separate `Replica` object in this project's real
usage (Task 41's own established fact, re-confirmed directly by Track B
Step 47), so `comm_domain != "ATTN_TP"` returns `replica_id` unchanged.
"""
from __future__ import annotations

from typing import Optional

ATTN_DP_ENCODING_BASE = 1_000_000


def encode_group_replica_id(
    replica_id: int, comm_domain: Optional[str], dp_lane_id: Optional[int] = None
) -> int:
    """The one canonical encoding for a collective group's own
    disambiguating identity.

    `ATTN_TP`: `replica_id * ATTN_DP_ENCODING_BASE + (dp_lane_id or 0)` --
    a single `Replica` with `dp>1` produces multiple same-shaped `TP`
    groups sharing one `replica_id`; the DP lane's own local index is
    what actually varies per group.

    Everything else (`EP`, `MOE_TP`, `DP`, ...): `replica_id` unchanged.
    An FFN replica is always its own `Replica` object here, never an
    internal DP lane, so `replica_id` alone already disambiguates it --
    confirmed directly, Track B Step 47.

    Raises rather than silently wrapping or truncating, matching
    `CommGroupRegistry.register`/`resolve`'s own posture (refuse rather
    than guess): a `dp_lane_id` outside `[0, ATTN_DP_ENCODING_BASE)`, or
    a negative `replica_id`, would corrupt this positional encoding
    against a neighboring `replica_id`'s own lane-0 value silently if
    allowed through.
    """
    if replica_id < 0:
        raise ValueError(f"replica_id must be non-negative, got {replica_id!r}")
    if comm_domain != "ATTN_TP":
        return replica_id
    lane = dp_lane_id or 0
    if not (0 <= lane < ATTN_DP_ENCODING_BASE):
        raise ValueError(
            f"dp_lane_id must be in [0, {ATTN_DP_ENCODING_BASE}), got {lane!r} "
            f"-- a value outside this range would collide with a neighboring "
            f"replica_id's own encoded value rather than raising"
        )
    return replica_id * ATTN_DP_ENCODING_BASE + lane
