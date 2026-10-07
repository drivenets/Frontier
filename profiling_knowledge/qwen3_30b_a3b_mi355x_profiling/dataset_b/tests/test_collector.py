import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b_check import check
from b_common import Invalid, features, read_json
from b_extract import extract
from check_dataset_a_scope import check_dataset_a_scope
from b_traces import kineto, pair_markers, rocprof_csv, rocprof_marker_csv
from drive_b import drive, validate
from extract_b import run
from telemetry_b import parse_dpm_clock
from fixtures import fixture

SPEC = read_json(ROOT / 'scope_spec.json')


@pytest.mark.parametrize('tp', [1, 4])
@pytest.mark.parametrize('phase', ['prefill', 'decode', 'mixed'])
@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('overlap', [False, True])
@pytest.mark.parametrize('query', [1, 8, 8192])
def test_case_matrix(tp, phase, padded, overlap, query):
    b = fixture(tp=tp, phase=phase, padded=padded, overlap=overlap, query=query)
    result, report = check(b, SPEC)
    assert report['structural_pass'] and not report['dataset_accepted']
    assert report['ranks'] == tp
    assert report['raw_kernels'] == report['target_kernels'] + report['excluded_kernels']
    for s in result['scopes']:
        assert s['scope_span_us'] == 10
        assert s['kernel_sum_us'] == (20 if overlap else 10)
        assert s['kernel_union_us'] == 10
    # All kernel names repeat and device timestamps are outside host scopes.
    assert len({k['name'] for k in b['kernels']}) == 1
    target = next(r for r in result['kernels'] if r['scope_id'])
    assert target['predecessor_id'] is not None
    assert target['predecessor_name'] == 'repeated_kernel_name'
    f = b['forwards'][-1]
    keys = features(f, b['manifest']['model'], tp)
    if padded or phase == 'mixed':
        assert keys['batch_size'] is None and keys['kv_cache_size'] is None


def test_chunked_request_progression():
    _, r = check(fixture(chunked=True, query=4096), SPEC)
    assert r['observed_forwards'] == 3


def mutate(b, case):
    if case == 'empty': b['kernels'] = []
    elif case == 'missing_rank': b['manifest']['processes'] = []
    elif case == 'duplicate_dispatch': b['kernels'].append(copy.deepcopy(b['kernels'][0])); b['manifest']['trace_diagnostics'][0]['raw_kernel_count'] += 1
    elif case == 'lost_record': b['kernels'].pop()
    elif case == 'lost_target_even_with_count': b['kernels'].pop(); b['manifest']['trace_diagnostics'][0]['raw_kernel_count'] -= 1
    elif case == 'missing_marker': b['ranges'].pop()
    elif case == 'unknown_request': b['requests'][0]['request_id'] = 'different'
    elif case == 'missing_scheduler': b['scheduler'] = []
    elif case == 'rank_membership': b['forwards'][0]['sequences'][0]['query_len'] += 1
    elif case == 'bad_shape': b['forwards'][0]['padded_tokens'] = 0
    elif case == 'duplicate_scope': b['scopes'].append(dict(b['scopes'][0], scope_id='new'))
    elif case == 'duplicate_api': b['launches'].append(copy.deepcopy(b['launches'][0]))
    elif case == 'bad_pid': b['launches'][0]['native_pid'] = '9999'
    elif case == 'bad_gpu': b['kernels'][0]['gpu_id'] = 'GPU-wrong'
    elif case == 'unresolved_graph': b['kernels'][1]['execution'] = 'graph_unresolved'
    elif case == 'graph_api': b['launches'][1]['name'] = 'hipGraphLaunch'
    elif case == 'false_eager': b['forwards'][0]['execution'] = 'replay'
    elif case == 'ambiguous': b['ranges'].append(dict(b['ranges'][2], marker_id='nested', begin_ns=b['ranges'][2]['begin_ns']+1))
    elif case == 'reversed_time': b['kernels'][0]['end_ns'] = 0
    elif case == 'window': b['manifest']['windows'][0]['end_ns'] = 2
    elif case == 'truncated': b['manifest']['windows'][0]['drained'] = False
    elif case == 'metadata_loss': b['manifest']['windows'][0]['metadata_complete'] = False
    elif case == 'drops': b['manifest']['trace_diagnostics'][0]['dropped_records'] = 1
    elif case == 'empty_telemetry': b['telemetry'] = []
    elif case == 'wrong_clock_gpu': [r.update(gpu_id='GPU-wrong') for r in b['telemetry']]
    elif case == 'stale': b['manifest']['clock_policy']['max_age_ns'] = 1
    elif case == 'gap': b['manifest']['clock_policy']['max_gap_ns'] = 1
    elif case == 'uncalibrated': b['manifest']['clock_policy']['frozen'] = False
    elif case == 'no_resolution': [r.update(sensor_resolution_ns=None) for r in b['telemetry']]
    elif case == 'uncertain': b['manifest']['clock_policy']['max_uncertainty_ns'] = 1
    elif case == 'outside_calibration': b['manifest']['clock_calibrations'][0]['valid_trace_end_ns'] = 1
    elif case == 'sensor_error': [r.update(sclk_mhz=None) for r in b['telemetry']]
    elif case == 'output_loss': b['requests'][0]['output_ids'] = []
    elif case == 'request_failed': b['requests'][0]['status'] = 'failed'
    elif case == 'unknown_exclusion': b['ranges'][1]['exclusion_reason'] = 'unknown'
    elif case == 'no_evidence': b['manifest']['clock_policy']['evidence_ref'] = ''
    elif case == 'orphan_scope': b['scopes'][0]['forward_id'] = 'missing'
    elif case == 'duplicate_sample': b['telemetry'].append(copy.deepcopy(b['telemetry'][0]))
    else: raise AssertionError(case)


@pytest.mark.parametrize('case', ['empty','missing_rank','duplicate_dispatch','lost_record','lost_target_even_with_count',
    'missing_marker','unknown_request','missing_scheduler','rank_membership','bad_shape','duplicate_scope',
    'duplicate_api','bad_pid','bad_gpu','unresolved_graph','graph_api','false_eager','ambiguous','reversed_time','window','truncated','metadata_loss',
    'drops','empty_telemetry','wrong_clock_gpu','stale','gap','uncalibrated','no_resolution','uncertain',
    'outside_calibration','sensor_error','output_loss','request_failed','unknown_exclusion','no_evidence','orphan_scope','duplicate_sample'])
def test_damaged_records_rejected(case):
    b = fixture()
    mutate(b, case)
    with pytest.raises(Invalid): check(b, SPEC)


def test_cross_rank_forward_loss():
    b = fixture(tp=4)
    b['forwards'].pop()
    with pytest.raises(Invalid): check(b, SPEC)


def test_no_drop_counter_is_limit_not_proof():
    b = fixture()
    b['manifest']['trace_diagnostics'][0]['dropped_records'] = None
    _, r = check(b, SPEC)
    assert any('cannot prove' in s for s in r['limitations'])


def test_shuffled_trace_order_does_not_change_values():
    b = fixture(overlap=True)
    a = extract(b)
    b['kernels'].reverse()
    z = extract(b)
    assert sorted(a['scopes'], key=lambda r:r['scope_id']) == sorted(z['scopes'], key=lambda r:r['scope_id'])
    assert sorted(a['kernels'], key=lambda r:r['dispatch_id']) == sorted(z['kernels'], key=lambda r:r['dispatch_id'])


def test_negative_same_queue_interval_is_preserved():
    b = fixture()
    b['kernels'][1]['begin_ns'] = b['kernels'][0]['begin_ns'] + 5
    r = extract(b)
    assert r['kernels'][1]['queue_interval_us'] < 0


def test_legitimate_dispatch_count_change():
    b = fixture()
    extra = dict(b['kernels'][-1], dispatch_id='extra', begin_ns=b['kernels'][-1]['end_ns'] + 100,
                 end_ns=b['kernels'][-1]['end_ns'] + 500)
    b['kernels'].append(extra)
    b['manifest']['trace_diagnostics'][0]['raw_kernel_count'] += 1
    _, report = check(b, SPEC)
    assert report['structural_pass']


def test_fused_scope_not_duplicated_for_two_a_labels():
    b = fixture()
    r, _ = check(b, SPEC)
    fused = [s for s in r['scopes'] if s['physical_scope'] == 'rope_kv' and s['forward_id'] == 'f0']
    assert len(fused) == 2  # one per layer, not one per A label


def test_strict_json_and_immutable_output(tmp_path):
    src = tmp_path/'input.json'
    src.write_text(json.dumps(fixture()))
    out = tmp_path/'out'
    report = run(src, out, ROOT/'scope_spec.json')
    assert report['structural_pass']
    with pytest.raises(FileExistsError): run(src, out, ROOT/'scope_spec.json')
    src.write_text('{"oops": NaN}')
    with pytest.raises(Invalid): read_json(src)
    src.write_text('{"truncated":')
    with pytest.raises(ValueError): run(src, tmp_path/'fail', ROOT/'scope_spec.json')
    assert (tmp_path/'fail'/'failure.json').exists()


def test_marker_pairing_thread_and_process_identity():
    events=[]
    for pid in ['p0','p1']:
        events += [dict(process_id=pid,thread_id='1',timestamp_ns=1,source_index=0,
                        operation='push',marker_id='r',label='B|scope',raw_ref='start'),
                   dict(process_id=pid,thread_id='1',timestamp_ns=10,source_index=1,
                        operation='pop',raw_ref='end')]
    assert len(pair_markers(events)) == 2
    with pytest.raises(Invalid): pair_markers(events[:-1])
    with pytest.raises(Invalid): pair_markers(events[1:])


def test_kineto_correlation_and_time_units():
    trace={'traceEvents':[
        dict(cat='kernel',ph='X',ts=100.125,dur=10.5,name='same',args={'device':0,'stream':3,'correlation':7}),
        dict(cat='cuda_runtime',ph='X',ts=1,dur=2,pid=100,tid=42,name='hipLaunchKernel',args={'correlation':7}),
        dict(cat='user_annotation',ph='X',ts=0,dur=10,pid=100,tid=42,name='B|f0|scope',args={})]}
    kernels, apis, ranges = kineto(trace,'p0',{'0':'GPU-0'})
    assert kernels[0]['begin_ns'] == 100125
    assert kernels[0]['end_ns'] == 110625
    assert kernels[0]['correlation_id'] == apis[0]['correlation_id']
    assert ranges[0]['thread_id'] == '42'
    trace['traceEvents'][1]['name']='hipGraphLaunch'
    assert kineto(trace,'p0',{'0':'GPU-0'})[0][0]['execution']=='graph_unresolved'
    trace['traceEvents'][2]['ph']='B'
    with pytest.raises(Invalid): kineto(trace,'p0',{'0':'GPU-0'})


def test_rocprof_adapter_retains_native_identity(tmp_path):
    k=tmp_path/'kernel.csv'; a=tmp_path/'api.csv'
    k.write_text('Agent_Id,Queue_Id,Stream_Id,Dispatch_Id,Correlation_Id,Kernel_Name,Start_Timestamp,End_Timestamp\nAgent 5,2,3,1,7,same,100,200\n')
    a.write_text('Process_Id,Thread_Id,Correlation_Id,Function,Start_Timestamp,End_Timestamp\n100,42,7,hipLaunchKernel,1,2\n')
    kernels, apis=rocprof_csv(k,a,'p0',{'Agent 5':'GPU-0'})
    assert kernels[0]['dispatch_id']=='1' and kernels[0]['native']['Agent_Id']=='Agent 5'
    assert apis[0]['native_pid']=='100'
    with pytest.raises(Invalid): rocprof_csv(k,a,'p0',{})


def test_rocprof_marker_csv_schema(tmp_path):
    """Dialect verified on real GPU hardware: dataset_b/evidence/rocprofv3-format-probe2-21737.

    One CSV row IS one complete push/pop range (Domain=MARKER_CORE_RANGE_API), with the exact
    roctxRangePushA label text in Function -- not a push/pop event stream needing pair_markers.
    """
    m = tmp_path / "marker.csv"
    m.write_text(
        'Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp\n'
        'MARKER_CORE_RANGE_API,B|scope|probe_sdk|0|test,152,152,59,1003815454899983,1003817417630766\n'
    )
    ranges = rocprof_marker_csv(m, "p0")
    assert len(ranges) == 1
    r = ranges[0]
    assert r["process_id"] == "p0" and r["thread_id"] == "152" and r["marker_id"] == "59"
    assert r["label"] == "B|scope|probe_sdk|0|test"
    assert r["begin_ns"] == 1003815454899983 and r["end_ns"] == 1003817417630766
    bad = tmp_path / "bad.csv"
    bad.write_text('Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp\n'
                   'HIP_API,foo,1,1,1,1,2\n')
    with pytest.raises(Invalid): rocprof_marker_csv(bad, "p0")
    empty = tmp_path / "empty.csv"
    empty.write_text('Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp\n')
    with pytest.raises(Invalid): rocprof_marker_csv(empty, "p0")


def test_rocprof_marker_csv_skips_control_api_rows(tmp_path):
    """marker_runtime.py's roctxProfilerResume/Pause calls (start_measurement/
    stop_measurement) show up in the real marker CSV as their own domain --
    verified: job 21798, exactly one Resume and one Pause row. These carry no
    B|scope|/B|fwd| label and must not be treated as a push/pop range."""
    m = tmp_path / "marker.csv"
    m.write_text(
        'Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp\n'
        'MARKER_CONTROL_API,roctxProfilerResume,248,248,1,100,200\n'
        'MARKER_CORE_RANGE_API,B|scope|probe_sdk|0|test,152,152,59,1003815454899983,1003817417630766\n'
        'MARKER_CONTROL_API,roctxProfilerPause,248,248,2,300,400\n'
    )
    ranges = rocprof_marker_csv(m, "p0")
    assert len(ranges) == 1
    assert ranges[0]["label"] == "B|scope|probe_sdk|0|test"


def test_marker_runtime_scope_forward_helpers(monkeypatch):
    """begin_scope/end_scope/begin_forward/end_forward: the real call-site API added to
    marker_runtime.py, exercised with markers enabled against the torch.profiler/record_function
    backend (no ROCTx/GPU needed for this unit test; torch itself is only in the GPU image)."""
    pytest.importorskip("torch")
    monkeypatch.setenv("FRONTIER_B_MARKERS", "1")
    monkeypatch.setenv("FRONTIER_B_TRACER", "kineto")
    import importlib
    import marker_runtime
    importlib.reload(marker_runtime)
    try:
        h_fwd = marker_runtime.begin_forward()
        assert h_fwd is not None
        h_scope = marker_runtime.begin_scope("qkv_qk_norm", 0, "compound")
        marker_runtime.end_scope(h_scope)
        h_model = marker_runtime.begin_scope("embedding", None, "tp_native")
        marker_runtime.end_scope(h_model)
        marker_runtime.end_forward(h_fwd)
    finally:
        monkeypatch.delenv("FRONTIER_B_MARKERS", raising=False)
        importlib.reload(marker_runtime)
    # Disabled path: every call must no-op and never raise, matching call sites'
    # unconditional "pass handle to end_scope regardless" contract.
    assert marker_runtime._BUFFER is None
    assert marker_runtime.begin_scope("qkv_qk_norm", 0, "compound") is None
    marker_runtime.end_scope(None)
    assert marker_runtime.begin_forward() is None
    marker_runtime.end_forward(None)


class _FakeForwardMode:
    def __init__(self, mode):
        self._mode = mode

    def is_decode(self):
        return self._mode == "decode"

    def is_mixed(self):
        return self._mode == "mixed"

    def is_extend(self):
        return self._mode == "extend"


class _FakeForwardBatch:
    def __init__(self, mode, batch_size, rids, seq_lens_cpu=None,
                 extend_prefix_lens_cpu=None, extend_seq_lens_cpu=None, num_tokens=None):
        self.forward_mode = _FakeForwardMode(mode)
        self.batch_size = batch_size
        self.rids = rids
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_prefix_lens_cpu = extend_prefix_lens_cpu
        self.extend_seq_lens_cpu = extend_seq_lens_cpu

        class _Shape:
            def __init__(self, n):
                self.shape = (n,)
        self.input_ids = _Shape(num_tokens)


def test_scheduler_forward_metadata_cross_check(monkeypatch):
    """record_scheduler_batch + begin_forward(forward_batch): the two independent
    CPU-metadata sources (schedule_batch.py's Req fields vs. qwen3_moe.py's
    ForwardBatch fields) share one forward-id counter and their sequence
    vectors must agree for check_b to pass -- exercised directly here without
    any GPU/SGLang dependency."""
    monkeypatch.setenv("FRONTIER_B_MARKERS", "1")
    monkeypatch.setenv("FRONTIER_B_TRACER", "kineto")
    import importlib
    import marker_runtime
    importlib.reload(marker_runtime)
    try:
        pytest.importorskip("torch")
        records = marker_runtime._BUFFER.records

        # Prefill: scheduler observes it independently (as schedule_batch.py's
        # patch does), then the model-level hook reads it back from a fake
        # ForwardBatch built the way qwen3_moe.py's real one would be.
        fwd_id = marker_runtime.record_scheduler_batch(
            "prefill",
            [dict(request_id="r1", kind="prefill", query_len=5, prefix_len=0, attended_len=5)],
            "schedule_batch.py:prepare_for_extend",
        )
        assert fwd_id is not None
        fb = _FakeForwardBatch("extend", 1, ["r1"], extend_prefix_lens_cpu=[0],
                                extend_seq_lens_cpu=[5], num_tokens=5)
        h = marker_runtime.begin_forward(fb)
        marker_runtime.end_forward(h)
        sched = [r for r in records if r["kind"] == "scheduler"][-1]
        fwd = [r for r in records if r["kind"] == "forward"][-1]
        assert sched["global_forward_id"] == fwd["global_forward_id"] == fwd_id
        assert sched["sequences"] == fwd["sequences"]
        assert fwd["padded_tokens"] == 5 and fwd["padded_batch_size"] == 1
        assert fwd["process_id"] == marker_runtime._PROCESS_ID

        # Decode: same cross-check, with the decode-specific derivation
        # (prefix_len = attended_len - 1, query_len fixed at 1).
        fwd_id2 = marker_runtime.record_scheduler_batch(
            "decode",
            [dict(request_id="r1", kind="decode", query_len=1, prefix_len=5, attended_len=6)],
            "schedule_batch.py:prepare_for_decode",
        )
        fb2 = _FakeForwardBatch("decode", 1, ["r1"], seq_lens_cpu=[6], num_tokens=1)
        h2 = marker_runtime.begin_forward(fb2)
        marker_runtime.end_forward(h2)
        sched2 = [r for r in records if r["kind"] == "scheduler"][-1]
        fwd2 = [r for r in records if r["kind"] == "forward"][-1]
        assert sched2["global_forward_id"] == fwd2["global_forward_id"] == fwd_id2 != fwd_id
        assert sched2["sequences"] == fwd2["sequences"] == [
            dict(request_id="r1", kind="decode", query_len=1, prefix_len=5, attended_len=6)
        ]

        # A model forward with no matching scheduler call mints its own id
        # instead of reusing a stale pending one.
        fb3 = _FakeForwardBatch("decode", 1, ["r1"], seq_lens_cpu=[7], num_tokens=1)
        h3 = marker_runtime.begin_forward(fb3)
        marker_runtime.end_forward(h3)
        fwd3 = [r for r in records if r["kind"] == "forward"][-1]
        assert fwd3["global_forward_id"] not in (fwd_id, fwd_id2)

        # Mixed forward_mode: not yet implemented, must not fabricate a
        # forward/scheduler record (a calibration_anchor still fires -- that
        # doesn't depend on sequence-vector derivation, only on timing).
        fb4 = _FakeForwardBatch("mixed", 1, ["r1"])
        before = len([r for r in records if r["kind"] in ("forward", "scheduler")])
        h4 = marker_runtime.begin_forward(fb4)
        marker_runtime.end_forward(h4)
        after = len([r for r in records if r["kind"] in ("forward", "scheduler")])
        assert after == before
        assert [r for r in records if r["kind"] == "calibration_anchor"]
    finally:
        monkeypatch.delenv("FRONTIER_B_MARKERS", raising=False)
        importlib.reload(marker_runtime)


def test_begin_scope_emits_metadata_only_inside_a_forward(monkeypatch):
    """bind_labels resolves scope_id by (process_id, forward_id, physical_scope,
    layer, variant); begin_scope must only emit that metadata while nested
    inside a begin_forward()/end_forward() pair, and never outside one."""
    monkeypatch.setenv("FRONTIER_B_MARKERS", "1")
    monkeypatch.setenv("FRONTIER_B_TRACER", "kineto")
    import importlib
    import marker_runtime
    importlib.reload(marker_runtime)
    try:
        pytest.importorskip("torch")
        # Outside any forward: no metadata record, even though the range still fires.
        h = marker_runtime.begin_scope("embedding", None, "tp_native")
        marker_runtime.end_scope(h)
        assert not [r for r in marker_runtime._BUFFER.records if r["kind"] == "scope"]

        fb = _FakeForwardBatch("decode", 1, ["r1"], seq_lens_cpu=[3], num_tokens=1)
        h_fwd = marker_runtime.begin_forward(fb)
        h_scope = marker_runtime.begin_scope("qkv_qk_norm", 2, "compound")
        marker_runtime.end_scope(h_scope)
        marker_runtime.end_forward(h_fwd)
        scope_records = [r for r in marker_runtime._BUFFER.records if r["kind"] == "scope"]
        assert len(scope_records) == 1
        s = scope_records[0]
        fwd_records = [r for r in marker_runtime._BUFFER.records if r["kind"] == "forward"]
        assert s["forward_id"] == fwd_records[-1]["forward_id"]
        assert s["process_id"] == marker_runtime._PROCESS_ID
        assert (s["physical_scope"], s["layer"], s["variant"]) == ("qkv_qk_norm", 2, "compound")

        # Nested begin_forward is a real bug this must catch, not silently allow.
        h_fwd2 = marker_runtime.begin_forward(fb)
        with pytest.raises(RuntimeError):
            marker_runtime.begin_forward(fb)
        marker_runtime.end_forward(h_fwd2)
    finally:
        monkeypatch.delenv("FRONTIER_B_MARKERS", raising=False)
        importlib.reload(marker_runtime)


def test_module_confirmed_drain_contract(monkeypatch, tmp_path):
    """The module-level confirmed_drain() the scheduler control-request
    handler calls: requires FRONTIER_B_OUTPUT, refuses mid-forward, refuses a
    second drain, and a subsequent atexit-tier flush must not overwrite it."""
    monkeypatch.setenv("FRONTIER_B_MARKERS", "1")
    monkeypatch.setenv("FRONTIER_B_TRACER", "kineto")
    monkeypatch.setenv("FRONTIER_B_OUTPUT", str(tmp_path))
    monkeypatch.setenv("FRONTIER_B_CAPTURE_ID", "cap-xyz")
    import importlib
    import marker_runtime
    importlib.reload(marker_runtime)
    try:
        pytest.importorskip("torch")
        assert marker_runtime._CAPTURE_ID == "cap-xyz"
        fb = _FakeForwardBatch("decode", 1, ["r1"], seq_lens_cpu=[3], num_tokens=1)

        # Mid-forward: not a safe boundary.
        h = marker_runtime.begin_forward(fb)
        with pytest.raises(RuntimeError):
            marker_runtime.confirmed_drain("cap-xyz")
        marker_runtime.end_forward(h)

        ack = marker_runtime.confirmed_drain("cap-xyz")
        assert ack["process_id"] == marker_runtime._PROCESS_ID
        on_disk = read_json(marker_runtime._FLUSH_PATH)
        assert on_disk["drain_mode"] == "confirmed" and on_disk["capture_id"] == "cap-xyz"

        # Second drain on the same process: refuse rather than duplicate/overwrite.
        with pytest.raises(RuntimeError):
            marker_runtime.confirmed_drain("cap-xyz")

        # The atexit tier must see _DRAINED and skip entirely, not overwrite
        # the confirmed file with an "emergency" one.
        marker_runtime._atexit_flush()
        assert read_json(marker_runtime._FLUSH_PATH)["drain_mode"] == "confirmed"
    finally:
        monkeypatch.delenv("FRONTIER_B_MARKERS", raising=False)
        importlib.reload(marker_runtime)


def test_module_confirmed_drain_requires_output_dir(monkeypatch):
    monkeypatch.setenv("FRONTIER_B_MARKERS", "1")
    monkeypatch.setenv("FRONTIER_B_TRACER", "kineto")
    monkeypatch.delenv("FRONTIER_B_OUTPUT", raising=False)
    import importlib
    import marker_runtime
    importlib.reload(marker_runtime)
    try:
        pytest.importorskip("torch")
        with pytest.raises(RuntimeError):
            marker_runtime.confirmed_drain("cap-1")
    finally:
        monkeypatch.delenv("FRONTIER_B_MARKERS", raising=False)
        importlib.reload(marker_runtime)


def request_manifest():
    return dict(schema_version='b-requests-0.1',concurrency=2,timeout_s=1,vocab_size=151936,
                requests=[dict(request_id='r1',input_ids=[7,8],max_new_tokens=2,cohort=0,phase='measurement',offset_s=0),
                          dict(request_id='r2',input_ids=[7],max_new_tokens=2,cohort=0,phase='measurement',offset_s=0)])


def test_driver_preserves_failures_and_exact_payload():
    seen=[]
    def transport(url,payload,timeout):
        seen.append(payload)
        if payload['rid']=='r2': raise TimeoutError('fixture')
        return dict(output_ids=[9,10],meta_info=dict(completion_tokens=2,id='r1'))
    rows=drive(request_manifest(),'http://fixture',transport)
    assert [r['status'] for r in rows]==['complete','failed']
    assert seen[0]['sampling_params']==dict(max_new_tokens=2,ignore_eos=True,temperature=0)
    assert rows[1]['error'].startswith('TimeoutError')


def test_driver_missing_output_tokens_is_not_success():
    rows=drive(request_manifest(),'http://fixture',lambda *args:dict(text='hello',meta_info={'completion_tokens':2}))
    assert all(r['status']=='failed' for r in rows)


def test_driver_logprob_token_evidence():
    rows=drive(request_manifest(),'http://fixture',lambda *args:dict(meta_info={'completion_tokens':2,'output_token_logprobs':[[0,5,None],[0,6,None]]}))
    assert all(r['output_ids']==[5,6] and r['status']=='complete' for r in rows)


def test_driver_manifest_errors():
    m=request_manifest(); m['requests'][0]['input_ids']=[151936]
    with pytest.raises(Invalid): validate(m)
    m=request_manifest(); m['requests'][1]['request_id']='r1'
    with pytest.raises(Invalid): validate(m)


def test_dpm_parser():
    assert parse_dpm_clock('0: 500Mhz\n1: 2300Mhz *\n')==2300
    with pytest.raises(Invalid): parse_dpm_clock('0: 500Mhz')


def test_marker_disabled_needs_no_torch_or_roctx(monkeypatch):
    monkeypatch.delenv('FRONTIER_B_MARKERS',raising=False)
    import marker_runtime
    assert marker_runtime.enabled_buffer() is None


def test_enclosing_forward_range_required():
    b=fixture()
    b['ranges']=[r for r in b['ranges'] if r.get('kind')!='forward']
    with pytest.raises(Invalid,match='enclosing forward'): check(b,SPEC)


def test_static_marker_labels_bind_to_forward_occurrences():
    from normalize_b import bind_labels
    b=fixture(phase='decode')
    scopes={(s['process_id'],s['scope_id']):s for s in b['scopes']}
    for r in b['ranges']:
        if r.get('kind')=='forward':
            r['label']='B|fwd|'+r.pop('forward_id')
            del r['kind']
        elif r.get('scope_id'):
            s=scopes[r['process_id'],r.pop('scope_id')]
            layer='model' if s['layer'] is None else str(s['layer'])
            r['label']=f"B|scope|{s['physical_scope']}|{layer}|{s['variant']}"
    b['ranges']=bind_labels(b['ranges'],b['scopes'])
    _,report=check(b,SPEC)
    assert report['observed_forwards']==3


def test_clock_calibration_preserves_large_origins_and_bounds():
    from calibrate_b import calibrate
    origin=10**18
    anchors=[dict(trace_ns=origin+n,host_begin_ns=origin+2*n+95,host_end_ns=origin+2*n+105) for n in [0,1000,2000]]
    c=calibrate(anchors,'GPU-0','independent marker clock evidence')
    assert c['scale']==2 and c['uncertainty_ns']==5
    assert c['host_anchor_ns']==origin+100
    with pytest.raises(Invalid): calibrate(anchors[:2],'GPU-0','ref')


def test_fractional_context_aliases_do_not_replace_vectors():
    from fixtures import seq, MODEL
    f=dict(sequences=[seq('a','prefill',3,129),seq('b','prefill',5,130),seq('c','decode',1,15),seq('d','decode',1,16)],
           padded_tokens=10,padded_batch_size=4,phase='mixed')
    keys=features(f,MODEL,4,kv_granularity=64)
    assert keys['prefill_context_mean_exact']==129.5
    assert keys['prefill_mixed_kv_cache_size_raw']==129
    assert keys['prefill_mixed_kv_cache_size']==192
    assert keys['decode_context_mean_exact']==15.5
    assert keys['decode_avg_kv_cache_size']==15
    assert keys['decode_prediction_kv_cache_size']==64
    assert keys['kv_cache_size'] is None
    assert keys['prefill_mixed_seq_len_variance']==1


def test_parquet_missing_dependency_is_explicit(tmp_path):
    if importlib.util.find_spec('pyarrow') is not None:
        pytest.skip('dependency installed; tested by roundtrip test')
    src=tmp_path/'input.json'; src.write_text(json.dumps(fixture()))
    with pytest.raises(ImportError): run(src,tmp_path/'parquet',ROOT/'scope_spec.json',True)
    assert 'pyarrow' in read_json(tmp_path/'parquet'/'failure.json')['error']


def test_parquet_roundtrip_when_available(tmp_path):
    pa=pytest.importorskip('pyarrow.parquet')
    src=tmp_path/'input.json'; src.write_text(json.dumps(fixture()))
    report=run(src,tmp_path/'parquet',ROOT/'scope_spec.json',True)
    assert pa.read_table(tmp_path/'parquet'/'kernels.parquet').num_rows==report['raw_kernels']


def test_unverified_launch_plan_refuses_before_any_job():
    from run_b import preflight
    with pytest.raises(Invalid,match='image digest'): preflight(read_json(ROOT/'launch_plan.example.json'))


def test_marker_buffer_flush_and_overflow(monkeypatch,tmp_path):
    import marker_runtime
    class Function:
        def __call__(self,*args): return 0
    class Library:
        roctxRangePushA=Function()
        roctxRangePop=Function()
        roctxProfilerPause=Function()
        roctxProfilerResume=Function()
    monkeypatch.setattr(marker_runtime.ctypes,'CDLL',lambda _:Library())
    b=marker_runtime.MarkerBuffer('roctx',max_records=1)
    b.append_cpu_metadata({'row':1}); b.append_cpu_metadata({'row':2})
    assert b.overflow
    h=b.begin(b.label('B|test'))
    # confirmed_drain demands a demonstrated safe boundary (depth==0); an open
    # marker must raise rather than flush a torn state.
    with pytest.raises(RuntimeError): b.confirmed_drain(tmp_path/'bad.json','cap-1')
    # emergency_flush is the opposite contract: never raises, honestly records
    # the unsafe state instead of asserting safety.
    b.emergency_flush(tmp_path/'emergency.json','mid_test')
    emergency = read_json(tmp_path/'emergency.json')
    assert emergency['drain_mode']=='emergency' and emergency['open_marker_depth']==1
    assert emergency['overflow'] is True
    b.end(h)
    ack=b.confirmed_drain(tmp_path/'metadata.json','cap-1')
    # max_records=1: the second append_cpu_metadata call above set overflow
    # and was dropped, not appended -- one record on disk, not two.
    assert ack['record_count']==1 and ack['overflow'] is True
    on_disk=read_json(tmp_path/'metadata.json')
    assert on_disk['overflow'] is True
    assert on_disk['drain_mode']=='confirmed' and on_disk['capture_id']=='cap-1'
    # One drain per process: a second confirmed_drain call on the same buffer
    # must fail rather than silently overwrite or duplicate the capture.
    with pytest.raises(RuntimeError): b.confirmed_drain(tmp_path/'metadata.json','cap-2')


def test_corrupt_native_csv_is_rejected(tmp_path):
    k=tmp_path/'kernel.csv'; a=tmp_path/'api.csv'
    k.write_text('Agent_Id,Queue_Id,Stream_Id,Dispatch_Id,Correlation_Id,Kernel_Name,Start_Timestamp,End_Timestamp\nAgent 5,2,3,1,7,same,100,1004519HIP_RUNTIME_API_EXT\n')
    a.write_text('Process_Id,Thread_Id,Correlation_Id,Function,Start_Timestamp,End_Timestamp\n100,42,7,hipLaunchKernel,1,2\n')
    with pytest.raises(ValueError): rocprof_csv(k,a,'p0',{'Agent 5':'GPU-0'})


def test_dataset_a_scope_canary_passes_on_clean_fixture():
    """Only the ops scope_spec.json already maps to a Dataset A column must be
    fully resolved; this is deliberately narrower than check()'s full gate."""
    report = check_dataset_a_scope(fixture(), SPEC)
    assert report['structural_pass']
    assert report['in_scope_kernels'] > 0
    assert report['in_scope_scope_invocations'] == report['in_scope_kernels']
    assert set(report['out_of_scope_attribution']) <= {'excluded'}


def test_dataset_a_scope_canary_ignores_unresolved_out_of_scope_kernel():
    """A real out-of-scope kernel with no scope AND no exclusion_reason (like
    the MoE kernels this canary was added for) must not block the canary --
    only Dataset-A-mapped scopes are required to be fully resolved."""
    b = fixture()  # one synthetic "excluded"/"moe" range per forward
    excluded = [r for r in b['ranges'] if r.get('exclusion_reason') == 'moe']
    assert excluded
    for r in excluded:
        del r['exclusion_reason']
    report = check_dataset_a_scope(b, SPEC)
    assert report['structural_pass']
    assert report['out_of_scope_attribution'] == {'unresolved': len(excluded)}


def test_dataset_a_scope_canary_fails_on_ambiguous_dataset_a_kernel():
    b = fixture()
    mutate(b, 'ambiguous')  # ambiguity on the embedding scope, a Dataset-A scope
    with pytest.raises(Invalid, match='structurally broken'):
        check_dataset_a_scope(b, SPEC)
