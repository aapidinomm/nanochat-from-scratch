import torch
import torch.nn.functional as F

from nanochat_from_scratch.model import RMSNorm

def test_rmsnorm_reference():
    x = torch.randn(2, 4, 8)

    norm = RMSNorm(eps=1e-6)
    actual = norm(x)

    expected = F.rms_norm(
        x,
        normalized_shape=(8,),
        weight=None,
        eps=1e-6,
    )

    torch.testing.assert_close(
        actual, expected, atol=1e-6, rtol=1e-5
    )

def test_token_independence():
    x = torch.randn(2, 4, 8)
    norm = RMSNorm()

    original = norm(x)

    modified = x.clone()
    modified[1, 0] *= 100

    updated = norm(modified)

    # Other batch elements are unchanged
    torch.testing.assert_close(
        original[0], updated[0]
    )

    # Other tokens in the SAME batch are unchanged
    torch.testing.assert_close(
        original[1, 1:], updated[1, 1:]
    )

def test_rmsnorm_zero_input():
    x = torch.zeros(2, 4, 8)

    y = RMSNorm()(x)

    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros_like(y))