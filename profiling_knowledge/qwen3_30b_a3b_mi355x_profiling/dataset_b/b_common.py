"""Dataset B's small, dependency-free record contract (integer nanoseconds)."""
import hashlib
import json
import math
from pathlib import Path

VERSION = "b0.1"


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def read_json(path):
    def reject(value):
        raise Invalid(f"non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def indexed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        require(key not in result, f"duplicate identity {fields}: {key}")
        result[key] = row
    return result


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {name}: {value!r}")
    return value


def interval(row):
    integer(row["begin_ns"], "begin_ns")
    integer(row["end_ns"], "end_ns")
    require(row["end_ns"] >= row["begin_ns"], "reversed timestamps")


def features(forward, model, tp, kv_granularity=None):
    """Vectors stay authoritative; standard uniform aliases are nullable."""
    seqs = forward["sequences"]
    integer(tp, "TP", 1)
    for key in ("n_q_head", "n_kv_head", "head_dim", "n_embd", "n_expanded_embd", "vocab_size", "num_layers", "block_size"):
        integer(model[key], key, 1)
    require(seqs, "empty forward membership")
    indexed(seqs, ["request_id"])
    for seq in seqs:
        integer(seq["query_len"], "query_len", 1)
        integer(seq["prefix_len"], "prefix_len")
        integer(seq["attended_len"], "attended_len", 1)
        require(seq["attended_len"] == seq["prefix_len"] + seq["query_len"],
                "attended length must be prefix + query")
        require(seq["kind"] in ("prefill", "decode"), "unknown sequence kind")
        require(seq["kind"] != "decode" or seq["query_len"] == 1, "decode query != 1")
    n = sum(s["query_len"] for s in seqs)
    bs = len(seqs)
    integer(forward["padded_tokens"], "padded_tokens", n)
    integer(forward["padded_batch_size"], "padded_batch_size", bs)
    pre = [s for s in seqs if s["kind"] == "prefill"]
    dec = [s for s in seqs if s["kind"] == "decode"]
    phase = "mixed" if pre and dec else "prefill" if pre else "decode"
    require(forward["phase"] == phase, "phase disagrees with membership")
    uniform = len({(s["query_len"], s["prefix_len"]) for s in seqs}) == 1
    real_shape = forward["padded_tokens"] == n and forward["padded_batch_size"] == bs
    compatible = uniform and real_shape and phase != "mixed"
    qheads = model["n_q_head"] // tp
    kvheads = max(1, model["n_kv_head"] // tp)
    require(model["n_q_head"] % tp == 0, "nonintegral query head sharding")
    require(model["n_kv_head"] % tp == 0 or tp % model["n_kv_head"] == 0,
            "unsupported KV head sharding")
    result = dict(model)
    result.update(num_tensor_parallel_workers=tp, n_head=model["n_q_head"],
                  local_q_heads=qheads, local_kv_heads=kvheads,
                  qkv_projection_width=(qheads + 2 * kvheads) * model["head_dim"],
                  real_tokens=n, real_batch_size=bs,
                  num_tokens=forward["padded_tokens"], total_tokens=n,
                  batch_size=bs if compatible else None,
                  kv_cache_size=seqs[0]["prefix_len"] if compatible else None,
                  prefill_chunk_size=seqs[0]["query_len"] if compatible and pre else
                  (0 if compatible else None),
                  standard_a_compatible=compatible,
                  num_prefill_seqs=len(pre), decode_batch_size=len(dec),
                  total_prefill_tokens=sum(s["query_len"] for s in pre),
                  total_batch_size=bs,
                  batch_composition_ratio=len(pre) / bs,
                  decode_avg_kv_cache_size=int(sum(s["prefix_len"] for s in dec) / len(dec))
                  if dec else None)
    result["decode_context_mean_exact"] = sum(s["prefix_len"] for s in dec) / len(dec) if dec else None
    if kv_granularity is not None:
        integer(kv_granularity, "kv_cache_prediction_granularity", 1)
    def rounded(value):
        return ((value + kv_granularity - 1) // kv_granularity) * kv_granularity if kv_granularity else None
    result["decode_prediction_kv_cache_size"] = rounded(result["decode_avg_kv_cache_size"]) if dec else None
    result["prefill_chunk_size_squared"] = (result["prefill_chunk_size"] ** 2
                                             if result["prefill_chunk_size"] is not None else None)
    if pre:
        lengths = [s["query_len"] for s in pre]
        mean = sum(lengths) / len(lengths)
        variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        cv = math.sqrt(variance) / mean
        vals = dict(avg_seq_len=mean, min_seq_len=min(lengths), max_seq_len=max(lengths),
                    seq_len_variance=variance, seq_len_cv=cv,
                    seq_len_range=max(lengths) - min(lengths), batch_size=len(pre),
                    batch_cv_interaction=len(pre) * cv,
                    batch_variance_interaction=len(pre) * variance,
                    total_tokens=sum(lengths), total_tokens_squared=sum(lengths) ** 2,
                    kv_cache_size=rounded(int(sum(s["prefix_len"] for s in pre) / len(pre))))
        result.update({"prefill_mixed_" + k: v for k, v in vals.items()})
        result["prefill_context_mean_exact"] = sum(s["prefix_len"] for s in pre) / len(pre)
        result["prefill_mixed_kv_cache_size_raw"] = int(result["prefill_context_mean_exact"])
    return result
