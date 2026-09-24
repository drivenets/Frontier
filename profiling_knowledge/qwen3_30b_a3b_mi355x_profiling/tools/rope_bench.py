"""Time and validate three RoPE implementations available in the linear_op image on one MI355X."""
import os, sys, json, torch
which = sys.argv[1]  # frontier_customop | vllm_native | vllm_aiter
if which == "vllm_aiter":
    os.environ["VLLM_ROCM_USE_AITER"] = "1"; os.environ["VLLM_ROCM_USE_AITER_ROPE"] = "1"
os.environ.pop("FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK", None)
from frontier.profiling.common.layers import rotary_embedding as fre
HEAD, BASE, MAXP = 128, 10_000_000.0, 16384
def reference(x, positions, heads):
    inv = 1.0 / (BASE ** (torch.arange(0, HEAD, 2, dtype=torch.float64) / HEAD))
    f = positions.double()[:, None] * inv[None, :]; c, s = f.cos()[:, None, :], f.sin()[:, None, :]
    xh = x.double().view(x.shape[0], heads, HEAD); x1, x2 = xh[..., :64], xh[..., 64:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).view(x.shape)
if which == "frontier_customop":
    rope = fre.RotaryEmbedding(HEAD, HEAD, MAXP, BASE, True).cuda()  # forward -> vllm._custom_ops.rotary_embedding (fused HIP kernel)
else:
    rope = fre.get_rope(HEAD, rotary_dim=HEAD, max_position=MAXP, base=BASE, is_neox_style=True, rope_scaling=None, dtype=torch.bfloat16).cuda()
print(which, "type", type(rope).__module__, getattr(rope, "_forward_method", None), "aiter", getattr(rope, "is_rocm_aiter_enabled", None), flush=True)
out = {}
for tp in (1, 8):
    heads, kvh = 32 // tp, max(4 // tp, 1)
    for n in (8, 512, 4096):
        torch.manual_seed(0)
        pos = torch.randint(0, MAXP, (n,), device="cuda")
        q = torch.randn(n, heads * HEAD, device="cuda", dtype=torch.bfloat16); k = torch.randn(n, kvh * HEAD, device="cuda", dtype=torch.bfloat16)
        q0, k0 = q.clone(), k.clone()
        qo, ko = rope(pos, q.clone(), k.clone())
        err = max((qo.float().cpu().double() - reference(q0.float().cpu(), pos.cpu(), heads)).abs().max().item(),
                  (ko.float().cpu().double() - reference(k0.float().cpu(), pos.cpu(), kvh)).abs().max().item())
        for _ in range(5): rope(pos, q.clone(), k.clone())
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            rope(pos, q.clone(), k.clone()); torch.cuda.synchronize()
        kern = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        names = sorted({e.name[:50] for e in kern}); kus = sum(e.device_time for e in kern)  # incl. the 2 clones
        # event timing of 50 calls with the device held behind the host (spin) so it is device time
        qs = [q.clone() for _ in range(50)]; ks = [k.clone() for _ in range(50)]
        torch.cuda.synchronize(); torch.cuda._sleep(200_000_000)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(50): rope(pos, qs[i], ks[i])
        e.record(); torch.cuda.synchronize()
        out[f"tp{tp}_n{n}"] = {"max_abs_err": round(err, 4), "kernels_per_call": len(kern), "kernel_names": names, "device_us_per_call_event": round(s.elapsed_time(e) * 1000 / 50, 1)}
        print(json.dumps({f"tp{tp}_n{n}": out[f"tp{tp}_n{n}"]}), flush=True)
