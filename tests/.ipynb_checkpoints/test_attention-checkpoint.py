
import torch
import torch.nn.functional as F
from einops import rearrange

from nanochat_from_scratch.attention import CausalSelfAttention


def test_attention_shape_and_gradients():
    x = torch.randn(2, 4, 8, requires_grad=True)
    attention = CausalSelfAttention(
        hidden_size=8, num_heads=2
    )

    y = attention(x)

    assert y.shape == (2, 4, 8)

    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_causal_mask():
    torch.manual_seed(42)

    attention = CausalSelfAttention(8, 2)
    x = torch.randn(1, 5, 8)

    original = attention(x)

    modified = x.clone()
    modified[:, 3:, :] += 100

    updated = attention(modified)

    # Changing future tokens must not change
    # outputs for earlier token positions.
    torch.testing.assert_close(
        original[:, :3, :],
        updated[:, :3, :],
        atol=1e-6,
        rtol=1e-5,
    )


def test_matches_pytorch_sdpa():
    torch.manual_seed(42)

    attention = CausalSelfAttention(8, 2)
    x = torch.randn(2, 4, 8)

    actual = attention(x)

    q = rearrange(
        attention.q_proj(x),
        "b t (h d) -> b h t d",
        h=2,
    )
    k = rearrange(
        attention.k_proj(x),
        "b t (h d) -> b h t d",
        h=2,
    )
    v = rearrange(
        attention.v_proj(x),
        "b t (h d) -> b h t d",
        h=2,
    )

    context = F.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )
    context = rearrange(
        context, "b h t d -> b t (h d)"
    )

    expected = attention.out_proj(context)

    torch.testing.assert_close(
        actual, expected, atol=1e-5, rtol=1e-4
    )
