import inspect, json, traceback, torch
HEAD, BASE, MAXP = 128, 10_000_000.0, 16384
def reference(x, positions, heads):
    inv = 1.0 / (BASE ** (torch.arange(0, HEAD, 2, dtype=torch.float64) / HEAD))
    f = positions.double()[:, None] * inv[None, :]; c, s = f.cos()[:, None, :], f.sin()[:, None, :]
    xh = x.double().view(x.shape[0], heads, HEAD); x1, x2 = xh[..., :64], xh[..., 64:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).view(x.shape)
import sglang.srt.layers.rotary_embedding.factory as fac
src = inspect.getsource(fac.get_rope)
print("=== factory.get_rope aiter branch:"); print("\n".join(l for l in src.splitlines() if "aiter" in l.lower() or "_use_" in l or "is_hip" in l)[:1500])
print("=== module-level flags:", {k: getattr(fac, k) for k in dir(fac) if k.startswith("_use") or k.startswith("_is")})
def bench(rope, label):
    for tp in (1, 8):
        heads, kvh = 32 // tp, max(4 // tp, 1)
        for n in (8, 512, 4096):
            torch.manual_seed(0); pos = torch.randint(0, MAXP, (n,), device="cuda")
            q = torch.randn(n, heads * HEAD, device="cuda", dtype=torch.bfloat16); k = torch.randn(n, kvh * HEAD, device="cuda", dtype=torch.bfloat16)
            q0, k0 = q.clone(), k.clone(); qo, ko = rope(pos, q.clone(), k.clone())
            err = max((qo.float().cpu().double() - reference(q0.float().cpu(), pos.cpu(), heads)).abs().max().item(), (ko.float().cpu().double() - reference(k0.float().cpu(), pos.cpu(), kvh)).abs().max().item())
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                rope(pos, q.clone(), k.clone()); torch.cuda.synchronize()
            kern = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            qs = [q.clone() for _ in range(50)]; ks = [k.clone() for _ in range(50)]; torch.cuda.synchronize(); torch.cuda._sleep(200_000_000)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record()
            for i in range(50): rope(pos, qs[i], ks[i])
            e.record(); torch.cuda.synchronize()
            print(json.dumps({label: f"tp{tp}_n{n}", "max_abs_err": round(err, 4), "kernels": len(kern), "names": sorted({x.name[:45] for x in kern}), "us": round(s.elapsed_time(e) * 1000 / 50, 1)}), flush=True)
try:
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    set_global_server_args_for_scheduler(ServerArgs(model_path="Qwen/Qwen3-30B-A3B", skip_server_warmup=True, load_format="dummy"))
    rope = fac.get_rope(head_size=HEAD, rotary_dim=HEAD, max_position=MAXP, base=BASE, is_neox_style=True, rope_scaling=None, dtype=torch.bfloat16).cuda()
    print("=== sglang factory object:", type(rope).__module__, type(rope).__name__, getattr(rope, "_forward_method", None))
    bench(rope, "sglang_factory")
except Exception:
    traceback.print_exc()
try:
    from aiter.rotary_embedding import get_rope as aiter_get_rope
    print("=== aiter.get_rope signature:", inspect.signature(aiter_get_rope))
    rope = aiter_get_rope(head_size=HEAD, rotary_dim=HEAD, max_position=MAXP, base=BASE, is_neox_style=True, rope_scaling=None, dtype=torch.bfloat16).cuda()
    print("=== aiter object:", type(rope).__module__, type(rope).__name__)
    bench(rope, "aiter_direct")
except Exception:
    traceback.print_exc()
