import gc
import os
import time
from contextlib import nullcontext

import torch

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_utils.tensor_parallel_utils import (
    get_padded_vocab_size,
)
from frontier.profiling.common.utils import (
    configure_quantization_manager_for_model_name,
    initialize_dummy_weights,
)
from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.linear_op.linear_op_impl import GPTModel
from frontier.profiling.linear_op import spike_diag
from frontier.profiling.linear_op.profiling_plan import _share_expert_profiling_names
from frontier.profiling.utils import (
    ProfileMethod,
    build_profile_position_indices,
    normalize_profile_method,
)
from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 3
ACTIVE_STEPS = int(os.environ.get("FRONTIER_LINEAR_ACTIVE_STEPS", "50"))  # override only for the spike experiments

# Two-column op timing (profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/07_post_proj_rope_dip_root_cause.md, validated in 08_).
# A CUDA-event scope in the un-synchronised 50-forward loop measures the host's launch span, not the kernel, whenever the
# GPU has drained everything queued before the scope (host ~0.45-0.68 ms/forward vs GPU 0.07-0.47 ms at TP>1 below ~6k tokens).
# Every shape is therefore timed twice: the legacy pass as before (-> time_stats_hostbound) and a GPU-bound pass in which a
# GPU spin is enqueued right before the timed loop so the device runs behind the host for the whole loop (-> time_stats).
# FRONTIER_GPU_BACKLOG_MS=<ms> overrides the spin length (default: BACKLOG_REQUEST_FACTOR x the legacy pass's loop wall).
_GPU_BACKLOG_MS = float(os.environ.get("FRONTIER_GPU_BACKLOG_MS", "0") or 0)
BACKLOG_REQUEST_FACTOR = 4.0   # requested spin = 4x the legacy loop wall: margin over the 3x gate for a cold or drifting clock
BACKLOG_COVERAGE_FACTOR = 3.0  # the delivered (event-measured) spin must cover 3x the legacy loop wall (sanity gate, test_backlog_covers_host_time)
LEGACY_HOST_BOUND_RATIO_THRESHOLD = 1.15  # legacy/GPU-bound: 0.98-1.01 where the idle gap before the kernel is <= 6 us, >= 1.60 where >= 40 us
FORWARD_SPAN_SCOPE = "forward_gpu_span"   # one event pair around each whole forward of the GPU-bound pass (F6 closure column)
PROBE_CYCLES = 2_000_000  # shader-clock probe: a fixed-cycle spin timed by an event pair on the idle device (~1 ms at 2 GHz)
_SLEEP_CYCLES_PER_MS = None


def _sclk_probe_mhz() -> float:
    """Concurrent shader-clock estimate: torch.cuda._sleep counts shader cycles, so PROBE_CYCLES / elapsed_us = MHz.

    Called on an idle device (after a synchronize), so the start event stamps at enqueue and the pair brackets only the spin.
    Cold-vs-warm calibrations measured ~2.0 vs ~2.4 GHz on MI355X (jobs 21385/21389), which is why every pass records it.
    """
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(PROBE_CYCLES)
    end.record()
    torch.cuda.synchronize()
    elapsed_us = start.elapsed_time(end) * 1e3
    return PROBE_CYCLES / elapsed_us if elapsed_us > 0 else float("nan")


def _empty_two_pass_fields() -> dict:
    """Fresh (unshared) values for profile methods without a second pass (record_function; kineto/perf_counter)."""
    nan = float("nan")
    return {
        "time_stats_hostbound": {}, "host_wall_per_forward_ms": nan, "host_wall_per_forward_ms_backlog": nan,
        "host_enqueue_per_forward_ms": nan, "host_enqueue_per_forward_ms_backlog": nan,
        "gpu_backlog_ms": nan, "gpu_backlog_ms_actual": nan, "legacy_host_bound_ratio": {}, "legacy_host_bound": {},
        "sclk_mhz_legacy_start": nan, "sclk_mhz_legacy_end": nan, "sclk_mhz_backlog_start": nan, "sclk_mhz_backlog_end": nan,
    }


def _enqueue_gpu_backlog(ms: float) -> "tuple[torch.cuda.Event, torch.cuda.Event, int]":
    """Enqueue a GPU spin of ~`ms` on the current stream (caller has synchronised, so the device is idle).

    Returns (start_event, end_event, cycles); the caller reads start.elapsed_time(end) after its trailing synchronize
    and re-fits _SLEEP_CYCLES_PER_MS from it. torch.cuda._sleep counts device cycles, so the delivered length depends on
    the clock: a cold 20M-cycle calibration under-delivered the first 80 ms spin by 13-23 % (jobs 21376/21385). The
    calibration therefore spins twice and fits from the second (warmer-clock) spin, and every later spin re-fits it.
    """
    global _SLEEP_CYCLES_PER_MS
    if _SLEEP_CYCLES_PER_MS is None:
        torch.cuda.synchronize()
        for _ in range(2):  # first spin ramps the clock, second is the one we fit
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            torch.cuda._sleep(20_000_000)
            end.record()
            torch.cuda.synchronize()
        _SLEEP_CYCLES_PER_MS = 20_000_000 / start.elapsed_time(end)
        print(f"[gpu_backlog] pid={os.getpid()} cycles_per_ms={_SLEEP_CYCLES_PER_MS:.0f}", flush=True)
    cycles = int(ms * _SLEEP_CYCLES_PER_MS)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()  # the GPU is idle here (caller synchronised), so this stamps at enqueue time
    torch.cuda._sleep(cycles)
    end.record()    # stamps when the spin finishes; read after the caller's trailing synchronize
    return start, end, cycles


def compute_legacy_host_bound(time_stats_hostbound, time_stats, threshold=LEGACY_HOST_BOUND_RATIO_THRESHOLD):
    """Per-op legacy/GPU-bound median ratio and the > threshold flag, for ops present in both passes.

    Marks the LEGACY column as host-bound; informational (the sanity gates are on the GPU-bound column). 0/0 -> 1.0.
    """
    ratio, flag = {}, {}
    for op, legacy in time_stats_hostbound.items():
        gpu_bound = time_stats.get(op)
        if gpu_bound is None:
            continue
        num, den = float(legacy["median"]), float(gpu_bound["median"])
        r = 1.0 if num == den == 0.0 else (float("inf") if den == 0.0 else num / den)
        ratio[op] = r
        flag[op] = bool(r > threshold)
    return ratio, flag


class LinearOpWrapper:
    """
    Wrapper for profiling linear operations in LLM models.
    
    This class profiles all linear-complexity operations including:
    - MLP layers: mlp_up_proj, mlp_down_proj, mlp_act
    - Normalization: input_layernorm, post_attention_layernorm
    - Attention projections: attn_pre_proj, attn_post_proj, attn_rope
    - Residual: add
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        num_tensor_parallel_workers: int,
        profile_method: str,
        rank: int,
        output_dir: str,
        profiling_plan: dict | None = None,
    ):
        super().__init__()

        self.profile_method = normalize_profile_method(profile_method)
        self.timer_stats_store = TimerStatsStore(profile_method=self.profile_method)

        self.model_config = model_config
        configure_quantization_manager_for_model_name(self.model_config.name)
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.rank = rank
        self.output_dir = output_dir
        self.profiling_plan = profiling_plan
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        self.pad_vocab_size = (
            self.model_config.vocab_size % self.num_tensor_parallel_workers != 0
        )
        self.padded_vocab_size = self.model_config.vocab_size
        if self.pad_vocab_size:
            self.padded_vocab_size = get_padded_vocab_size(
                self.model_config.vocab_size, self.num_tensor_parallel_workers
            )
            print(
                f"[WARNING] vocab_size {self.model_config.vocab_size} is not divisible by "
                f"TP={self.num_tensor_parallel_workers}. Padding to {self.padded_vocab_size} for profiling."
            )

        # Initialize a complete GPT model to profile linear operations in context
        self.model = GPTModel(
            model_config,
            num_tensor_parallel_workers,
            (
                ACTIVE_STEPS
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value
                else 1
            ),
            pad_vocab_size=self.pad_vocab_size,
            profiling_plan=self.profiling_plan,
        )
        initialize_dummy_weights(self.model)
        self.model = self.model.to(dtype=self.model_config.dtype).cuda().eval()

    def _get_expected_keys(self) -> list[str]:
        if self.profiling_plan is not None and "enabled_ops" in self.profiling_plan:
            return list(self.profiling_plan["enabled_ops"])

        expected_keys = [
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "attn_pre_proj",
            "attn_post_proj",
            "attn_rope",
        ]
        architecture_profile = self.model_config.get_model_architecture_profile()
        expected_keys.extend(
            op_name
            for op_name in architecture_profile.linear_attention.sharded_ops
            if op_name not in expected_keys
        )
        if (
            getattr(self.model_config, "is_moe", False)
            and hasattr(self.model_config, "supports_share_expert")
            and self.model_config.supports_share_expert()
        ):
            expected_keys.extend(_share_expert_profiling_names())
        return expected_keys

    def _timed_pass(self, input_ids, positions, *, backlog_ms: float, record_forward_span: bool):
        """3 warm-up + ACTIVE_STEPS timed forwards; returns time_stats, loop_wall_ms, enqueue_ms, the measured spin and
        the shader-clock probes.

        Order: synchronize -> warm-up -> synchronize -> mark_warmup_end -> clock probe (sclk_mhz_start: the clock the timed
        loop starts at, after warm-up) -> [spin enqueued on the stream] -> timed forwards -> host enqueue timestamp ->
        synchronize -> clock probe (sclk_mhz_end). The probes use bare events, never CudaTimer, and sit outside every scope;
        the spin precedes every timed start event. TimerStatsStore arithmetic is untouched: get_stats() takes the median
        over the same 50 timed runs.
        """
        global _SLEEP_CYCLES_PER_MS
        self.timer_stats_store.clear_stats()
        diag = spike_diag.enabled
        span_timer = CudaTimer(FORWARD_SPAN_SCOPE) if record_forward_span else nullcontext()
        backlog_events = None
        torch.cuda.synchronize()  # drain the input-construction kernels so warm-up starts on an idle device
        # Keep CPython's cyclic GC out of the timed region: a gen-2 collection on this thread lands between a
        # CudaTimer start event and the kernel launch and shows up as a 150-290 ms sample
        # (profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/05_linear_op_spike_root_cause.md).
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(WARMUP_STEPS):
                if diag:
                    spike_diag.forward_begin()
                with span_timer:
                    self.model(input_ids, positions)
                if diag:
                    spike_diag.forward_end()

            torch.cuda.synchronize()
            self.timer_stats_store.mark_warmup_end()
            sclk_start = _sclk_probe_mhz()  # immediately before the timed loop, device idle after the synchronize
            if backlog_ms > 0:
                backlog_events = _enqueue_gpu_backlog(backlog_ms)
            loop_t0 = time.perf_counter()

            for _ in range(ACTIVE_STEPS):
                if diag:
                    spike_diag.forward_begin()
                with span_timer:
                    self.model(input_ids, positions)
                if diag:
                    spike_diag.forward_end()

            enqueue_ms = (time.perf_counter() - loop_t0) * 1e3  # host wall until the last forward's launch call returned (HIP may still buffer before queue insertion)
            torch.cuda.synchronize()
            loop_wall_ms = (time.perf_counter() - loop_t0) * 1e3
        finally:
            if gc_was_enabled:
                gc.enable()
        sclk_end = _sclk_probe_mhz()

        backlog_actual_ms = 0.0
        if backlog_events is not None:
            start, end, cycles = backlog_events
            backlog_actual_ms = float(start.elapsed_time(end))
            if backlog_actual_ms > 0:
                _SLEEP_CYCLES_PER_MS = cycles / backlog_actual_ms  # re-fit for the next task (clock drifts, see helper)
        return {
            "time_stats": self.timer_stats_store.get_stats(),
            "loop_wall_ms": loop_wall_ms,
            "enqueue_ms": enqueue_ms,
            "backlog_actual_ms": backlog_actual_ms,
            "sclk_mhz_start": sclk_start,
            "sclk_mhz_end": sclk_end,
        }

    @torch.inference_mode()  # disable gradient calculation
    def profile(self, num_tokens: int):
        vocab_range = self.padded_vocab_size // self.num_tensor_parallel_workers
        input_ids = torch.randint(
            low=0,
            high=vocab_range,
            size=(num_tokens,),
            device="cuda",
            dtype=torch.long,
        )
        positions = torch.tensor(
            build_profile_position_indices(
                num_tokens=num_tokens,
                max_position_embeddings=self.model_config.max_position_embeddings,
            ),
            device="cuda",
            dtype=torch.long,
        )

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Run the model once without capturing the graph.
            # This is to make sure that the captured graph does not include the
            # kernel launches for initial benchmarking (e.g., Triton autotune).
            self.model(
                input_ids,
                positions,
            )
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)

            with record_function_tracer:
                self.model(
                    input_ids,
                    positions,
                )

            time_stats = record_function_tracer.get_operation_time_stats(debug=True)

            # Check for missing expected operations
            expected_keys = self._get_expected_keys()
            missing_keys = [k for k in expected_keys if k not in time_stats]
            if missing_keys:
                print(f"[WARNING] num_tokens={num_tokens}: Missing operations: {missing_keys}")
            two_pass_fields = _empty_two_pass_fields()
        else:
            # Pass 1 (legacy): the loop exactly as before -> host-bound for short ops.
            legacy = self._timed_pass(input_ids, positions, backlog_ms=0.0, record_forward_span=False)
            time_stats = legacy["time_stats"]
            two_pass_fields = {**_empty_two_pass_fields(), "host_wall_per_forward_ms": legacy["loop_wall_ms"] / ACTIVE_STEPS,
                               "host_enqueue_per_forward_ms": legacy["enqueue_ms"] / ACTIVE_STEPS,
                               "sclk_mhz_legacy_start": legacy["sclk_mhz_start"], "sclk_mhz_legacy_end": legacy["sclk_mhz_end"]}
            if self.profile_method == ProfileMethod.CUDA_EVENT.value:
                # Pass 2 (GPU-bound, CUDA_EVENT only - the other methods synchronise per scope): same loop with the device
                # held behind the host by a spin enqueued before the timed forwards, so every event pair can only measure
                # device time. This becomes the primary column (time_stats); the legacy pass is kept as time_stats_hostbound.
                requested_backlog_ms = _GPU_BACKLOG_MS if _GPU_BACKLOG_MS > 0 else BACKLOG_REQUEST_FACTOR * legacy["loop_wall_ms"]
                gpu_bound = self._timed_pass(input_ids, positions, backlog_ms=requested_backlog_ms, record_forward_span=True)
                time_stats = gpu_bound["time_stats"]
                legacy_ratio, legacy_flag = compute_legacy_host_bound(legacy["time_stats"], time_stats)
                two_pass_fields.update({
                    "time_stats_hostbound": legacy["time_stats"],
                    "host_wall_per_forward_ms_backlog": gpu_bound["loop_wall_ms"] / ACTIVE_STEPS,
                    "host_enqueue_per_forward_ms_backlog": gpu_bound["enqueue_ms"] / ACTIVE_STEPS,
                    "sclk_mhz_backlog_start": gpu_bound["sclk_mhz_start"], "sclk_mhz_backlog_end": gpu_bound["sclk_mhz_end"],
                    "gpu_backlog_ms": requested_backlog_ms,
                    "gpu_backlog_ms_actual": gpu_bound["backlog_actual_ms"],
                    "legacy_host_bound_ratio": legacy_ratio,
                    "legacy_host_bound": legacy_flag,
                })

        stats = {
            "time_stats": time_stats,
            # two-column timing fields (CUDA_EVENT): time_stats_hostbound = legacy pass; host_wall_per_forward_ms = legacy
            # loop wall / 50 incl. the trailing sync ("W"); *_backlog = GPU-bound pass (contains the spin drain);
            # gpu_backlog_ms = requested spin, gpu_backlog_ms_actual = event-measured spin; legacy_host_bound[_ratio] per op;
            # host_enqueue_per_forward_ms{,_backlog} = host wall to the last launch / 50 (== wall when host-bound or queue-throttled);
            # sclk_mhz_{legacy,backlog}_{start,end} = shader-clock probe immediately before and after each timed loop.
            **two_pass_fields,
            "n_head": self.model_config.num_q_heads,
            "n_kv_head": self.model_config.num_kv_heads,
            "n_embd": self.model_config.embedding_dim,
            "n_expanded_embd": self.model_config.mlp_hidden_dim,
            "vocab_size": self.model_config.vocab_size,
            "use_gated_mlp": self.model_config.use_gated_mlp,
            "use_qk_norm": getattr(self.model_config, "use_qk_norm", False),
            "attn_output_gate": getattr(self.model_config, "attn_output_gate", False),
            "num_tokens": num_tokens,
            "warmup_steps": WARMUP_STEPS,
            "active_steps": ACTIVE_STEPS,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
            "padded_n_embd": (
                self.profiling_plan.get("padded_n_embd", self.model_config.embedding_dim)
                if self.profiling_plan is not None
                else self.model_config.embedding_dim
            ),
            "padded_n_expanded_embd": (
                self.profiling_plan.get(
                    "padded_n_expanded_embd", self.model_config.mlp_hidden_dim
                )
                if self.profiling_plan is not None
                else self.model_config.mlp_hidden_dim
            ),
            "model_arch": self.model_config.model_arch,
            "model_architecture_profile": (
                self.model_config.get_model_architecture_profile().profile_id
            ),
            "share_expert_dim": self.model_config.share_expert_dim,
            "share_q_dim": self.model_config.share_q_dim,
        }
        self.timer_stats_store.clear_stats()

        return stats
