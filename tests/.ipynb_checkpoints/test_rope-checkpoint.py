import torch

from nanochat_from_scratch.rope import (
    rotary_cos_sin,
    apply_rope,
)


def test_rope_shape():
    x = torch.randn(2, 4, 8, 64)

    cos, sin = rotary_cos_sin(
        seq_len=8,
        head_dim=64,
        device=x.device,
    )

    assert cos.shape == (1, 1, 8, 32)
    assert sin.shape == (1, 1, 8, 32)

    y = apply_rope(x, cos, sin)

    assert y.shape == x.shape


def test_rope_preserves_norm():
    x = torch.randn(2, 4, 8, 64)

    cos, sin = rotary_cos_sin(8, 64, x.device)
    y = apply_rope(x, cos, sin)

    torch.testing.assert_close(
        x.norm(dim=-1),
        y.norm(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_rope_position_zero():
    x = torch.randn(2, 4, 8, 64)

    cos, sin = rotary_cos_sin(8, 64, x.device)
    y = apply_rope(x, cos, sin)

    # Rotation at position 0 is the identity.
    torch.testing.assert_close(
        x[:, :, 0, :],
        y[:, :, 0, :],
    )