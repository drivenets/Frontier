"""Positional-anomaly probe: run one linear_op worker's task loop in a single process (same LinearOpWrapper.profile()
path as the dense collections) and dump EVERY per-run sample of both passes, with optional interventions.

  --mode default        exactly the collection code path
  --mode busy_backlog   replace the torch.cuda._sleep spin with a chain of 8192^2 bf16 GEMMs of the same length
  --mode warmup6        6 warm-up forwards instead of 3 (shifts the command count before the timed loop)
  --mode hostsleep      time.sleep(0.5 ms) in a forward pre-hook (adds host time per forward, no extra commands)
  --marks               roctx range "fwd<k>" around every forward (for rocprofv3 correlation)
Env knobs of the wrapper still apply (FRONTIER_LINEAR_ACTIVE_STEPS, FRONTIER_GPU_BACKLOG_MS, FRONTIER_LINEAR_BACKLOG_BLOCK_STEPS).
"""
import argparse, json, os, sys, time
import torch
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op import linear_op_wrapper as low
from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan

p = argparse.ArgumentParser()
p.add_argument("--model", default="qwen3-a3b-30b-moe")
p.add_argument("--tp", type=int, required=True)
p.add_argument("--tokens", type=int, nargs="+", required=True)
p.add_argument("--mode", default="default", choices=["default", "busy_backlog", "warmup6", "hostsleep", "clockprobe"])
p.add_argument("--marks", action="store_true")
p.add_argument("--out", required=True)
p.add_argument("--output_dir", default="/tmp/posprobe_out")
a = p.parse_args()
torch.cuda.set_device(0)

if a.mode == "warmup6":
    low.WARMUP_STEPS = 6

if a.mode == "busy_backlog":
    _busy = {}
    def busy_backlog(ms):
        if not _busy:
            n = 4096
            _busy["a"] = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            _busy["b"] = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            for _ in range(20):  # first calls include hipBLASLt solution selection and a cold clock: warm up first
                torch.matmul(_busy["a"], _busy["b"])
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(50):
                torch.matmul(_busy["a"], _busy["b"])
            e.record(); torch.cuda.synchronize()
            _busy["per_call_ms"] = s.elapsed_time(e) / 50
            print(f"[busy_backlog] per GEMM ms={_busy['per_call_ms']:.4f}", flush=True)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, int(round(ms / _busy["per_call_ms"])))):
            torch.matmul(_busy["a"], _busy["b"])
        end.record()
        return start, end, 1  # cycles unused: the wrapper's re-fit of _SLEEP_CYCLES_PER_MS is irrelevant in this mode
    low._enqueue_gpu_backlog = busy_backlog

cfg = ModelConfig.from_model_name(a.model)
plan = build_profiling_plan(model_config=cfg, tp_size=a.tp, attn_tp=[1, 2, 4, 8], ffn_tp=[1, 2, 4, 8], is_moe=True)
print("enabled_ops:", plan["enabled_ops"], "mode:", a.mode, file=sys.stderr)
w = LinearOpWrapper(cfg, a.tp, "cuda_event", rank=0, output_dir=a.output_dir, profiling_plan=plan)

_fwd = {"k": 0}
def pre_hook(mod, args):
    if a.mode == "hostsleep":
        time.sleep(0.0005)
    if a.marks:
        torch.cuda.nvtx.range_push(f"fwd{_fwd['k']}")
PROBE_CYCLES = 200_000
_clk = []  # (forward index, start_event, end_event)
def post_hook(mod, args, out):
    if a.marks:
        torch.cuda.nvtx.range_pop()
    if a.mode == "clockprobe":
        s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_.record(); torch.cuda._sleep(PROBE_CYCLES); e_.record()
        _clk.append((_fwd["k"], s_, e_))
    _fwd["k"] += 1
if a.mode in ("hostsleep", "clockprobe") or a.marks:
    w.model.register_forward_pre_hook(pre_hook)
    w.model.register_forward_hook(post_hook)

env = {k: v for k, v in os.environ.items() if k.startswith("FRONTIER_") or k in ("AMD_LOG_LEVEL", "ROC_AQL_QUEUE_SIZE", "GPU_MAX_COMMAND_BUFFERS")}
with open(a.out, "a") as f:
    for tok in sorted(a.tokens, reverse=True):
        _fwd["k"] = 0; _clk.clear()
        torch.cuda.nvtx.range_push(f"task_tp{a.tp}_tok{tok}")
        t0 = time.time()
        r = w.profile(tok)
        torch.cuda.nvtx.range_pop()
        rec = {"tp": a.tp, "tokens": tok, "mode": a.mode, "marks": a.marks, "t_wall": round(t0, 3), "env": env,
               "warmup_steps": low.WARMUP_STEPS, "active_steps": low.ACTIVE_STEPS, "block_steps": low.BACKLOG_BLOCK_STEPS, "settle_steps": getattr(low, "SETTLE_STEPS", 0), "backlog_kind": getattr(low, "BACKLOG_KIND", "sleep")}
        for k in ("host_wall_per_forward_ms", "host_wall_per_forward_ms_backlog", "host_enqueue_per_forward_ms",
                  "host_enqueue_per_forward_ms_backlog", "gpu_backlog_ms", "gpu_backlog_ms_actual",
                  "sclk_mhz_legacy_start", "sclk_mhz_legacy_end", "sclk_mhz_backlog_start", "sclk_mhz_backlog_end"):
            rec[k] = float(r[k])
        rec["gpu"] = {op: {"warmup_count": int(v["warmup_count"]), "samples": json.loads(v["samples"])} for op, v in r["time_stats"].items()}
        if _clk:
            torch.cuda.synchronize()
            rec["sclk_mhz_by_forward"] = [[k, round(PROBE_CYCLES / (s_.elapsed_time(e_) * 1e3), 1)] for k, s_, e_ in _clk]
        rec["legacy"] = {op: {"warmup_count": int(v["warmup_count"]), "samples": json.loads(v["samples"])} for op, v in r["time_stats_hostbound"].items()}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"task tp{a.tp} tok{tok} mode={a.mode} done in {time.time()-t0:.2f}s", flush=True)
