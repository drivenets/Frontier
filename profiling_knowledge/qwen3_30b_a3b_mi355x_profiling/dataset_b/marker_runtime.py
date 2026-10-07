"""SGLang-side support module, now with real call-site integration (see
sglang_b_markers.patch: qwen3_moe.py, qwen2_moe.py, communicator.py,
schedule_batch.py).

Use only when FRONTIER_B_MARKERS=1 at import. No events, GPU synchronization,
or tensor reads on the measured path.

Metadata persistence has two tiers, and they are not interchangeable:

1. **confirmed_drain()** -- the only tier that is allowed to claim a demonstrated
   safe boundary. Called exactly once per process, only from the
   FrontierBDrainReqInput control-request handler
   (sglang_b_markers.patch: scheduler.py's handle_frontier_b_drain), itself
   only ever invoked from Scheduler.process_input_requests -- which runs at
   the top of event_loop_normal/event_loop_overlap, strictly before the next
   batch's run_batch() call and strictly after the previous batch's forward
   call has returned on the host side (confirmed by reading both loops: the
   loop structure guarantees no open marker/forward at that point regardless
   of overlap scheduling, since CPU-side overlap is about the *next* batch's
   scheduling running ahead of GPU completion of the *current* one, not about
   this module's host-side push/pop outliving the synchronous Python call
   that emits it). confirmed_drain() re-verifies depth==0 and no open forward
   itself rather than trusting the caller, and raises instead of flushing a
   torn state. This is the only path the driver (run_b.py) is allowed to
   treat as "the window is closed, it is safe to ask the native tracer to
   finalize and to send termination signals."
2. **emergency_flush()** -- best-effort only, invoked by the atexit hook and
   by nothing else. It never claims a demonstrated safe boundary (a signal or
   abnormal-exit context cannot prove one), is explicitly tagged
   `drain_mode="emergency"` with the actual (possibly nonzero) marker depth
   and the actual (possibly open) forward id recorded honestly, and does not
   raise on bad state -- the goal is to preserve whatever was collected, not
   to assert it is complete. atexit only fires on a clean process exit at
   all (never on SIGKILL), so this is a fallback for the case where
   confirmed_drain() was never reached, not a replacement for it.

Native-trace finalization (rocprofv3 actually finishing its own CSVs) is a
*third*, separate concern this module cannot observe or control from inside
the traced process; verifying a native trace belongs to the same capture is
done later, after the fact, by assemble_bundle.py matching native_pid/
capture_id -- never asserted here.

begin_scope/end_scope/begin_forward/end_forward below are the call-site API.
Forward-scope nesting is resolved later, by normalize_b.bind_labels, purely
from timestamp containment (a scope range must fall inside a B|fwd| range on
the same process/thread) -- so these helpers never need to thread a forward
ID down into instrumented call sites, only emit correctly nested push/pop
pairs.

record_scheduler_batch/begin_forward together produce the CPU-side
"scheduler"/"forward" records b_check.py cross-validates against each other.
record_scheduler_batch is called from schedule_batch.py's prepare_for_extend/
prepare_for_decode, built from Req.prefix_indices/extend_range/rid -- NOT from
ForwardBatch's derived tensors -- so the b_check.py comparison between it and
begin_forward's forward-level record (read from ForwardBatch.seq_lens_cpu/
extend_*_cpu, the tensors the model actually consumes) is a real independent
cross-check, not a value compared against itself. The two share one forward-id
counter: record_scheduler_batch mints the id (SGLang's scheduler always runs
synchronously before the model forward it prepared, within one rank/process),
begin_forward consumes it once. A model forward with no matching scheduler
record (a schedule_batch.py call path this module doesn't yet hook, e.g.
speculative decoding's early-return in prepare_for_decode) mints its own id
instead of silently reusing a stale one -- and b_check.py then correctly fails
it as "forward lacks scheduler evidence" rather than passing spuriously.
"""
import atexit
import ctypes
import itertools
import json
import os
import threading
import time


class MarkerBuffer:
    def __init__(self, backend, max_records=1000000):
        if backend not in ("roctx", "kineto"):
            raise ValueError("select one marker backend: roctx or kineto")
        self.backend = backend
        self.max_records = max_records
        self.records = []
        self.overflow = False
        self.drained = False
        self.local = threading.local()
        if backend == "roctx":
            # rocprofv3 1.1.0 / rocprofiler-sdk 1.1.0 only instruments calls that go through
            # librocprofiler-sdk-roctx.so; libroctx64.so is ABI-identical but is never hooked by
            # this tool (verified empirically: dataset_b/evidence/rocprofv3-format-probe2-21737,
            # both RTLD_LOCAL and RTLD_GLOBAL dlopen of libroctx64.so produce zero marker events
            # and no "MARKER (ROCTx) ... initialized" log line; librocprofiler-sdk-roctx.so does).
            lib = ctypes.CDLL("librocprofiler-sdk-roctx.so")
            self.push = lib.roctxRangePushA
            self.push.argtypes = [ctypes.c_char_p]
            self.push.restype = ctypes.c_int
            self.pop = lib.roctxRangePop
            self.pop.argtypes = []
            self.pop.restype = ctypes.c_int
            # roctx.h: tid=0 means "global pause/resume of the profiler in the
            # current process" -- paired with the server launch's rocprofv3
            # --selected-regions flag, this makes rocprofv3 record kernels
            # only between the Resume/Pause calls below, so raw_kernel_count
            # is the measured window by construction (no post-hoc filtering
            # of model-load/JIT-compile kernels that happen before the first
            # measured forward -- B_SCHEMA.md: "measured-window selection and
            # boundary provenance must be explicit"). --selected-regions
            # already starts every thread unrecorded by default (confirmed:
            # job 21797's server.log logs "rocprofv3 is only recording ...
            # surrounded by roctxProfilerResume(0)/roctxProfilerPause" per
            # thread with no Pause() call made yet) -- do NOT also call
            # roctxProfilerPause(0) here at buffer construction (module
            # import time, before SGLang's own HIP/ROCm runtime context
            # exists): doing so crashed with SIGABRT during Python import in
            # that same job, since the roctx library's pause/resume path
            # apparently assumes an already-initialized ROCm runtime context.
            self.profiler_pause = lib.roctxProfilerPause
            self.profiler_pause.argtypes = [ctypes.c_uint64]
            self.profiler_pause.restype = ctypes.c_int
            self.profiler_resume = lib.roctxProfilerResume
            self.profiler_resume.argtypes = [ctypes.c_uint64]
            self.profiler_resume.restype = ctypes.c_int
        else:
            from torch.profiler import record_function
            self.record_function = record_function

    def label(self, text):
        """Prepare static labels once during model initialization."""
        return text.encode() if self.backend == "roctx" else text

    def begin(self, label):
        depth = getattr(self.local, "depth", 0)
        if self.backend == "roctx":
            if self.push(label) < 0:
                raise RuntimeError("roctx push failed")
            handle = None
        else:
            handle = self.record_function(label)
            handle.__enter__()
        self.local.depth = depth + 1
        return handle

    def end(self, handle):
        if getattr(self.local, "depth", 0) <= 0:
            raise RuntimeError("unmatched marker end")
        if self.backend == "roctx":
            if self.pop() < 0:
                raise RuntimeError("roctx pop failed")
        else:
            handle.__exit__(None, None, None)
        self.local.depth -= 1

    def start_measurement(self):
        """Opens the native trace window (see roctxProfilerPause/Resume note
        in __init__); a no-op for the kineto backend, which has no equivalent
        selected-regions control."""
        if self.backend == "roctx" and self.profiler_resume(0) != 0:
            raise RuntimeError("roctx profiler resume failed")

    def stop_measurement(self):
        """Closes the native trace window at a confirmed_drain() boundary."""
        if self.backend == "roctx" and self.profiler_pause(0) != 0:
            raise RuntimeError("roctx profiler pause failed")

    def append_cpu_metadata(self, record):
        # Caller must supply immutable CPU scalars/tuples copied at scheduling time.
        # No serialization or disk writes here; overflow is fatal to acceptance.
        if len(self.records) >= self.max_records:
            self.overflow = True
            return
        self.records.append(record)

    def _write_atomic(self, path, extra):
        # Same-directory temp file + os.replace: on a POSIX filesystem (the
        # bind-mounted b_metadata dir is a plain directory, not NFS-direct in
        # the container's view) this is atomic -- a reader never observes a
        # partially-written file, and a crash mid-write leaves only the .tmp
        # behind, never a corrupt final path.
        # process_id at the top level (not just nested inside individual scope/
        # forward records) so assemble_bundle.py can use the one authoritative
        # identity throughout, instead of inventing a second "rankN" label
        # that would never match what begin_scope/begin_forward actually
        # stamped onto their own records.
        payload = dict(records=self.records, overflow=self.overflow,
                       native_pid=os.getpid(), process_id=_PROCESS_ID,
                       backend=self.backend, **extra)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, "x") as f:
            json.dump(payload, f, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def confirmed_drain(self, path, capture_id):
        """Raises rather than flushing a torn state; see module docstring
        tier 1. One drain per buffer instance: os.replace would otherwise
        silently overwrite a prior confirmed drain on a second call, which
        must fail loud instead. Returns the ack payload for the
        control-request handler to relay back to the driver."""
        if self.drained:
            raise RuntimeError(f"this buffer already completed a confirmed_drain (path={path!r} requested again)")
        depth = getattr(self.local, "depth", 0)
        if depth != 0:
            raise RuntimeError(f"confirmed_drain called with an open marker (depth={depth}); not a safe boundary")
        self._write_atomic(path, dict(drain_mode="confirmed", capture_id=capture_id,
                                       open_marker_depth=0, forward_open=False))
        self.drained = True
        return dict(process_id=_PROCESS_ID, record_count=len(self.records),
                    overflow=self.overflow, flush_path=path)

    def emergency_flush(self, path, reason):
        """Best-effort only; see module docstring tier 2. Never raises on bad
        state and never claims a demonstrated safe boundary."""
        depth = getattr(self.local, "depth", 0)
        self._write_atomic(path, dict(drain_mode="emergency", reason=reason,
                                       open_marker_depth=depth,
                                       forward_open=_CURRENT_FORWARD_ID is not None))


def enabled_buffer():
    if os.environ.get("FRONTIER_B_MARKERS") != "1":
        return None
    return MarkerBuffer(os.environ["FRONTIER_B_TRACER"])


# Process-wide singleton + call-site API. Imported once at worker process start
# (module import time), per b_common's process-instance identity contract: an
# opaque ID unique to this process instance, not merely the OS PID (a restart
# must get a new identity). The manifest layer (run_b.py / a future manifest
# builder) still owns the full process record (native PID, TP rank, GPU
# identity) -- this is only the id this module stamps onto its own records.
_BUFFER = enabled_buffer()
_PROCESS_ID = f"{os.getpid()}:{time.time_ns()}"
_CAPTURE_ID = os.environ.get("FRONTIER_B_CAPTURE_ID")
_LABELS = {}
_FORWARD_IDS = itertools.count()
_SCOPE_IDS = itertools.count()
_PENDING_SCHEDULER_FORWARD_ID = None
_CURRENT_FORWARD_ID = None
_MEASUREMENT_STARTED = False
_DRAINED = False
_FLUSH_PATH = None
if _BUFFER is not None and os.environ.get("FRONTIER_B_OUTPUT"):
    _FLUSH_PATH = os.path.join(os.environ["FRONTIER_B_OUTPUT"], f"metadata-pid{os.getpid()}.json")

    def _atexit_flush():
        # Tier 2 only (see module docstring): skip entirely if confirmed_drain()
        # already ran -- an atexit flush after a confirmed drain would either
        # collide on the same path or, worse, silently look like a second
        # legitimate capture. Never overwrites a confirmed drain's file.
        # Re-checks current globals (not just the definition-time snapshot)
        # since tests reload this module repeatedly without deregistering
        # prior atexit callbacks; a stale callback must no-op cleanly rather
        # than raise out of its own exception handler.
        if _DRAINED or _BUFFER is None or _FLUSH_PATH is None:
            return
        try:
            _BUFFER.emergency_flush(_FLUSH_PATH, reason="atexit")
        except Exception as e:
            # Nothing downstream can see an exception raised from inside
            # atexit; record the failure instead of losing it silently.
            try:
                with open(_FLUSH_PATH + ".atexit_error", "x") as f:
                    f.write(f"{type(e).__name__}: {e}\n")
            except OSError:
                pass

    atexit.register(_atexit_flush)


def confirmed_drain(capture_id):
    """Call only from the FrontierBDrainReqInput handler (scheduler.py), only
    reachable from Scheduler.process_input_requests -- never from a signal
    handler or atexit, neither of which can prove a safe boundary. Raises if
    FRONTIER_B_OUTPUT was not set (nowhere to flush), if markers are disabled,
    if a marker is still open, or if this process already drained once (one
    capture window per process; a second call means something unexpected
    wanted a second drain and should fail loud, not silently succeed)."""
    global _DRAINED
    if _BUFFER is None:
        raise RuntimeError("markers are disabled; nothing to drain")
    if _FLUSH_PATH is None:
        raise RuntimeError("FRONTIER_B_OUTPUT not set; nowhere to flush to")
    if _DRAINED:
        raise RuntimeError(f"process already drained once (capture_id={_DRAINED!r} requested again as {capture_id!r})")
    if _CAPTURE_ID is not None and capture_id != _CAPTURE_ID:
        # Defense in depth: the driver sets FRONTIER_B_CAPTURE_ID on this same
        # process's environment and is expected to pass the identical value to
        # the drain endpoint. A mismatch means a wiring bug (e.g. the driver
        # reused a stale id, or two runs' env/HTTP calls got crossed) -- fail
        # loud rather than drain under the wrong label.
        raise RuntimeError(f"capture_id mismatch: process env says {_CAPTURE_ID!r}, drain request says {capture_id!r}")
    if _CURRENT_FORWARD_ID is not None:
        raise RuntimeError("drain requested while a forward is still open; not a safe boundary")
    # Close the native trace window at this same safe boundary, before the
    # CPU-metadata snapshot: nothing dispatched after this point belongs to
    # the capture this drain is finalizing.
    _BUFFER.stop_measurement()
    result = _BUFFER.confirmed_drain(_FLUSH_PATH, capture_id)
    _DRAINED = capture_id
    return result


def _label(key):
    """Memoized so hot-path call sites never format/encode a string per call."""
    cached = _LABELS.get(key)
    if cached is None:
        cached = _BUFFER.label(key)
        _LABELS[key] = cached
    return cached


def begin_scope(physical_scope, layer, variant):
    """No-op (returns None) when markers are disabled; call sites must still
    pass the returned handle to end_scope unconditionally.

    Also appends the scope-invocation metadata record normalize_b.bind_labels
    needs to resolve this marker's scope_id (the label itself only carries
    physical_scope/layer/variant, not an id -- bind_labels finds the unique
    metadata entry matching (process_id, forward_id, physical_scope, layer,
    variant) within the enclosing B|fwd| range). Only emitted while inside an
    active begin_forward()/end_forward() pair; a scope marker pushed outside
    one (should not happen on any instrumented call site) still fires the
    ROCTx range but produces no metadata, so bind_labels will correctly fail
    it as lacking a unique forward parent/metadata match rather than silently
    mis-attributing it.
    """
    if _BUFFER is None:
        return None
    layer_token = "model" if layer is None else str(layer)
    if _CURRENT_FORWARD_ID is not None:
        _BUFFER.append_cpu_metadata(dict(
            kind="scope", process_id=_PROCESS_ID, scope_id=str(next(_SCOPE_IDS)),
            forward_id=_CURRENT_FORWARD_ID, physical_scope=physical_scope,
            layer=layer, variant=variant,
        ))
    return _BUFFER.begin(_label(f"B|scope|{physical_scope}|{layer_token}|{variant}"))


def end_scope(handle):
    if _BUFFER is None:
        return
    _BUFFER.end(handle)


def record_scheduler_batch(phase, sequences, source_ref):
    """Call from schedule_batch.py once per prepared batch, with a plain-dict
    `sequences` vector built from Req fields (never from ForwardBatch). Mints
    and returns the forward id the upcoming model forward must consume.

    forward_id/global_forward_id are strings: the B|fwd|<id> marker label is
    text, and normalize_b.bind_labels parses it back out of the label as a
    string to match against scope/forward metadata -- an int here would
    silently fail every match ("0" != 0). `ordinal` is kept a plain int
    specifically because check_b.py sorts scheduler records by it
    numerically; a string ordinal would sort "10" before "2"."""
    global _PENDING_SCHEDULER_FORWARD_ID
    if _BUFFER is None:
        return None
    ordinal = next(_FORWARD_IDS)
    forward_id = str(ordinal)
    _BUFFER.append_cpu_metadata(dict(
        kind="scheduler", global_forward_id=forward_id, ordinal=ordinal,
        phase=phase, sequences=sequences, source_ref=source_ref,
    ))
    _PENDING_SCHEDULER_FORWARD_ID = forward_id
    return forward_id


def _forward_sequences(forward_batch):
    """Built only from ForwardBatch's already-CPU-resident fields
    (seq_lens_cpu, extend_seq_lens_cpu, extend_prefix_lens_cpu, rids) -- no
    device sync, no .cpu() call, matching the no-synchronization-on-the-
    measured-path requirement. Returns (phase, sequences) or (None, None) for
    a forward_mode this hook does not yet cover (e.g. true MIXED batches)."""
    fm = forward_batch.forward_mode
    rids = forward_batch.rids
    # Duck-typed on ForwardMode's own helper methods (is_decode/is_mixed/
    # is_extend) rather than importing the enum, so this module stays
    # dependency-light and importable outside any SGLang context (tests).
    if fm.is_mixed():
        # True MIXED batches need per-sequence kind disambiguation (the
        # 1-token rows riding inside the extend kernel are decodes); not yet
        # implemented.
        return None, None
    if fm.is_decode():
        seq_lens_cpu = forward_batch.seq_lens_cpu
        sequences = [
            dict(request_id=rids[i], kind="decode", query_len=1,
                 prefix_len=int(seq_lens_cpu[i]) - 1, attended_len=int(seq_lens_cpu[i]))
            for i in range(forward_batch.batch_size)
        ]
        return "decode", sequences
    if fm.is_extend():
        # Covers EXTEND and (out of this recipe's scope) TARGET_VERIFY /
        # SPLIT_PREFILL / DLLM_EXTEND, which is_extend() also reports true for.
        prefix = forward_batch.extend_prefix_lens_cpu
        qlen = forward_batch.extend_seq_lens_cpu
        sequences = [
            dict(request_id=rids[i], kind="prefill", query_len=int(qlen[i]),
                 prefix_len=int(prefix[i]), attended_len=int(prefix[i]) + int(qlen[i]))
            for i in range(forward_batch.batch_size)
        ]
        return "prefill", sequences
    return None, None


def begin_forward(forward_batch=None):
    global _PENDING_SCHEDULER_FORWARD_ID, _CURRENT_FORWARD_ID, _MEASUREMENT_STARTED
    if _BUFFER is None:
        return None
    if not _MEASUREMENT_STARTED:
        # Opens the native trace window on the very first forward this
        # process measures -- everything before this point (weight loading,
        # AITER/Triton first-dispatch JIT compilation) is never recorded by
        # rocprofv3 at all (see MarkerBuffer's roctxProfilerPause/Resume
        # note), so it cannot show up as unattributed kernels downstream.
        _BUFFER.start_measurement()
        _MEASUREMENT_STARTED = True
    forward_id = _PENDING_SCHEDULER_FORWARD_ID
    if forward_id is None:
        # Same string requirement as record_scheduler_batch: forward_id must
        # match the B|fwd|<id> label text's parsed type.
        forward_id = str(next(_FORWARD_IDS))
    else:
        _PENDING_SCHEDULER_FORWARD_ID = None
    if forward_batch is not None:
        phase, sequences = _forward_sequences(forward_batch)
        if phase is not None:
            _BUFFER.append_cpu_metadata(dict(
                kind="forward", process_id=_PROCESS_ID,
                forward_id=forward_id, global_forward_id=forward_id,
                phase=phase, execution="eager", sequences=sequences,
                padded_tokens=int(forward_batch.input_ids.shape[0]),
                padded_batch_size=forward_batch.batch_size,
            ))
    if _CURRENT_FORWARD_ID is not None:
        raise RuntimeError("nested begin_forward: forwards do not nest")
    _CURRENT_FORWARD_ID = forward_id
    # Free clock-calibration anchor: every B|fwd| push is a host-bracketed
    # native-trace event with zero extra overhead (no separate calibration
    # kernel/marker). trace_ns is recovered later by matching this forward_id
    # to the B|fwd|<id> row's Start_Timestamp in the marker CSV; calibrate_b.py
    # needs >=3 of these, and a real run has one per forward (dozens+), enough
    # to fit on most and hold out the rest for validation.
    host_begin_ns = time.monotonic_ns()
    handle = _BUFFER.begin(_label(f"B|fwd|{forward_id}"))
    host_end_ns = time.monotonic_ns()
    _BUFFER.append_cpu_metadata(dict(
        kind="calibration_anchor", forward_id=forward_id,
        host_begin_ns=host_begin_ns, host_end_ns=host_end_ns,
    ))
    return handle


def end_forward(handle):
    global _CURRENT_FORWARD_ID
    _CURRENT_FORWARD_ID = None
    end_scope(handle)
