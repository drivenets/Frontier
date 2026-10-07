"""Invented records for mechanism tests; NEVER GPU measurements."""
from copy import deepcopy

MODEL = dict(num_layers=2, n_q_head=32, n_kv_head=4, head_dim=128,
             n_embd=2048, n_expanded_embd=768, vocab_size=151936, block_size=1,
             use_qk_norm=True, use_gated_mlp=True, attn_output_gate=False)


def seq(rid, kind, query, prefix):
    return dict(request_id=rid, kind=kind, query_len=query, prefix_len=prefix,
                attended_len=query + prefix)


def fixture(tp=1, phase="prefill", padded=False, overlap=False, query=8, chunked=False):
    # One decode forward per generated token (not generated-1): with overlap
    # scheduling, prefill never directly yields a sampled token, so a
    # generated_tokens-token request has exactly generated_tokens decode
    # forwards -- see b_check.py's decode/first-token convention comment.
    if phase == "mixed":
        batches = [[seq("d", "prefill", query, 0)],
                   [seq("p", "prefill", query, 0), seq("d", "decode", 1, query)],
                   [seq("p", "decode", 1, query), seq("d", "decode", 1, query + 1)]]
        generations = {"p": 1, "d": 2}
    elif phase == "decode":
        batches = [[seq("p", "prefill", query, 0)], [seq("p", "decode", 1, query)],
                   [seq("p", "decode", 1, query + 1)]]
        generations = {"p": 2}
    elif chunked:
        batches = [[seq("p", "prefill", query, 0)], [seq("p", "prefill", query, query)],
                   [seq("p", "decode", 1, query * 2)]]
        generations = {"p": 1}
    else:
        batches = [[seq("p", "prefill", query, 0)], [seq("p", "decode", 1, query)]]
        generations = {"p": 1}
    procs = [dict(process_id=f"p{rank}", native_pid=str(100 + rank), rank=rank, gpu_id=f"GPU-{rank}") for rank in range(tp)]
    m = dict(schema_version="b0.1", run_id="SYNTHETIC-ONLY", data_kind="synthetic_fixture",
             tp=tp, model=deepcopy(MODEL), scope_spec_version="qwen3-provisional-0.1",
             processes=procs, windows=[], trace_diagnostics=[], clock_calibrations=[],
             clock_policy=dict(frozen=True, evidence_ref="synthetic test policy only",
                               max_age_ns=200000000, max_gap_ns=200000000, max_uncertainty_ns=1000))
    b = dict(manifest=m, kernels=[], launches=[], ranges=[], scopes=[], forwards=[],
             scheduler=[], requests=[], telemetry=[])
    for rid, count in generations.items():
        n = query * (2 if chunked and phase == "prefill" else 1)
        b["requests"].append(dict(request_id=rid, input_ids=[7] * n, output_ids=[9] * count,
                                  prompt_tokens=n, generated_tokens=count, requested_new_tokens=count, status="complete"))
    for i, batch in enumerate(batches):
        kinds = {s["kind"] for s in batch}
        ph = "mixed" if len(kinds) > 1 else next(iter(kinds))
        b["scheduler"].append(dict(global_forward_id=f"g{i}", ordinal=i, sequences=batch,
                                    phase=ph, source_ref=f"independent-scheduler-fixture:{i}"))
    for proc in procs:
        pid, gpu = proc["process_id"], proc["gpu_id"]
        m["windows"].append(dict(process_id=pid, begin_ns=0, end_ns=100000000,
                                 complete=True, drained=True, metadata_complete=True))
        m["clock_calibrations"].append(dict(gpu_id=gpu, scale=1.0, host_anchor_ns=0, trace_anchor_ns=0,
                                            uncertainty_ns=10, valid_trace_begin_ns=0,
                                            valid_trace_end_ns=100000000, evidence_ref="invented anchors"))
        for j, stamp in enumerate((0, 50000000, 100000000)):
            b["telemetry"].append(dict(sample_id=f"{gpu}:{j}", gpu_id=gpu, timebase="host_monotonic_ns",
                                        read_begin_ns=stamp, read_end_ns=stamp + 20,
                                        sensor_resolution_ns=100, sclk_mhz=2300., mclk_mhz=1200., errors=[]))
        dispatch = 0
        for i, obs in enumerate(b["scheduler"]):
            f = dict(process_id=pid, forward_id=f"f{i}", global_forward_id=obs["global_forward_id"],
                     sequences=deepcopy(obs["sequences"]), phase=obs["phase"], execution="eager",
                     padded_tokens=sum(s["query_len"] for s in obs["sequences"]) + (3 if padded else 0),
                     padded_batch_size=len(obs["sequences"]) + (3 if padded else 0))
            b["forwards"].append(f)
            # Independent literal call list, not imported from expected_scopes/spec.
            calls = [(None, "embedding", "tp_native")]
            for layer in range(2):
                calls += [(layer, "input_norm", "norm_only" if layer == 0 else "add_norm"),
                          (layer, "qkv_qk_norm", "compound"), (layer, "rope_kv", "fused"),
                          (layer, "attention_" + obs["phase"], "native"),
                          (layer, "output_projection", "gemm_only"),
                          (layer, "post_norm", "add_norm")]
            calls.insert(0, (None, "excluded", "moe"))
            b["ranges"].append(dict(process_id=pid, thread_id="42", marker_id=f"forward:{i}",
                                    kind="forward", forward_id=f"f{i}", begin_ns=900 + i * 100000,
                                    end_ns=1000 + i * 100000 + len(calls) * 1000, raw_ref=f"forward-range:{i}"))
            for c, (layer, physical, variant) in enumerate(calls):
                host = 1000 + i * 100000 + c * 1000
                sid = f"f{i}:{c}"
                r = dict(process_id=pid, thread_id="42", marker_id=f"m{i}:{c}",
                         begin_ns=host, end_ns=host + 500, raw_ref=f"range:{i}:{c}")
                if physical == "excluded":
                    r["exclusion_reason"] = "moe"
                else:
                    r["scope_id"] = sid
                    b["scopes"].append(dict(process_id=pid, scope_id=sid, forward_id=f"f{i}",
                                             layer=layer, physical_scope=physical, variant=variant))
                b["ranges"].append(r)
                for occurrence in range(2 if overlap and physical != "excluded" else 1):
                    dispatch += 1
                    corr = str(dispatch)
                    b["launches"].append(dict(process_id=pid, native_pid=proc["native_pid"], thread_id="42",
                                               correlation_id=corr, name="hipLaunchKernel", begin_ns=host + 50 + occurrence * 50,
                                               end_ns=host + 60 + occurrence * 50, raw_ref=f"launch:{dispatch}"))
                    device = 1000000 + i * 1000000 + c * 20000
                    b["kernels"].append(dict(process_id=pid, gpu_id=gpu, dispatch_id=corr, correlation_id=corr,
                                              queue_id=str(occurrence), stream_id=str(occurrence), name="repeated_kernel_name",
                                              begin_ns=device, end_ns=device + 10000, execution="eager", raw_ref=f"kernel:{dispatch}"))
        m["trace_diagnostics"].append(dict(process_id=pid, dropped_records=0,
                                            raw_kernel_count=sum(k["process_id"] == pid for k in b["kernels"])))
    return b
