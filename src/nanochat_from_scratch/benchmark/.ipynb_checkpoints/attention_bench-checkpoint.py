import math
import torch
import torch.nn.functional as F
from einops import einsum, repeat

def manual_gqa(q, k, v):
    group_size = q.shape[1] // k.shape[1]

    k = repeat(
        k, "b g t d -> b (g r) t d", r=group_size
    )
    v = repeat(
        v, "b g t d -> b (g r) t d", r=group_size
    )

    scores = einsum(
        q, k, "b h t d, b h s d -> b h t s"
    ) / math.sqrt(q.shape[-1])

    T = q.shape[-2]
    mask = torch.ones(
        T, T, device=q.device, dtype=torch.bool
    ).tril()

    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)

    return einsum(
        weights, v, "b h t s, b h s d -> b h t d"
    )


def benchmark(fn, warmup=10, repeats=50):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()

        # Wait until all previously submitted CUDA operations on the device have completed. 
        torch.cuda.synchronize()
        # Reset PyTorch's peak CUDA memory statistics before measuring a new operation.
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(repeats):
            output = fn()
        end.record()

        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / repeats
        # max_memory_allocated() => Maximum allocated tensor memory observed since the last reset
        peak = torch.cuda.max_memory_allocated() - baseline

    return ms, peak / (1024 ** 2)


for T in [256, 512, 1024, 2048]:
    B, Hq, Hkv, D = 2, 8, 2, 64

    q = torch.randn(B, Hq, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16)

    manual_ms, manual_mb = benchmark(
        lambda: manual_gqa(q, k, v)
    )

    sdpa_ms, sdpa_mb = benchmark(
        lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )
    )

    print(
        f"T={T:4d} | manual={manual_ms:.3f} ms "
        f"| SDPA={sdpa_ms:.3f} ms "
        f"| speedup={manual_ms/sdpa_ms:.2f}x "
        f"| peak delta={manual_mb:.1f}/{sdpa_mb:.1f} MiB"
    )
