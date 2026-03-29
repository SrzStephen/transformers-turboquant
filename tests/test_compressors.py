"""
Integration tests for TurboQuantCompressorV2 and TurboQuantCompressorMSE.

When use_triton=True but Triton/CUDA is not available, the compressor silently
falls back to the PyTorch path. On CPU test environments both compressors will
use the same PyTorch path and should produce identical results.
"""

import pytest
import torch
from turboquant_pytorch.compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE

B = 1
H = 2

BITS = [2, 3, 4]
DIMS = [64, 128, 256]


@pytest.fixture(params=[(b, d) for b in BITS for d in DIMS], ids=lambda x: f"b{x[0]}_d{x[1]}")
def bd(request):
    return request.param


def make_states(d: int) -> torch.Tensor:
    """Generate random float16 input of shape [B=1, H=2, S=16, d]."""
    torch.manual_seed(0)
    return torch.randn(B, H, 16, d, dtype=torch.float16)


class TestTurboQuantCompressorV2:
    def test_asymmetric_scores_triton_vs_pytorch(self, bd):
        """use_triton=True and use_triton=False should produce equivalent scores."""
        b, d = bd
        states = make_states(d)

        comp_triton = TurboQuantCompressorV2(head_dim=d, bits=b, seed=42, device="cpu", use_triton=True)
        comp_pytorch = TurboQuantCompressorV2(head_dim=d, bits=b, seed=42, device="cpu", use_triton=False)

        compressed_triton = comp_triton.compress(states)
        compressed_pytorch = comp_pytorch.compress(states)

        torch.manual_seed(1)
        queries = torch.randn(B, H, 8, d, dtype=torch.float16)

        scores_triton = comp_triton.asymmetric_attention_scores(queries, compressed_triton)
        scores_pytorch = comp_pytorch.asymmetric_attention_scores(queries, compressed_pytorch)

        assert scores_triton.shape == scores_pytorch.shape, (
            f"Shape mismatch: {scores_triton.shape} vs {scores_pytorch.shape}"
        )
        torch.testing.assert_close(
            scores_triton.float(),
            scores_pytorch.float(),
            atol=1e-2,
            rtol=0.0,
            msg=f"Scores differ beyond atol=1e-2 for b={b}, d={d}",
        )


class TestTurboQuantCompressorMSE:
    def test_decompress_triton_vs_pytorch(self, bd):
        """use_triton=True and use_triton=False should decompress to equivalent tensors."""
        b, d = bd
        states = make_states(d)

        comp_triton = TurboQuantCompressorMSE(head_dim=d, bits=b, seed=42, device="cpu", use_triton=True)
        comp_pytorch = TurboQuantCompressorMSE(head_dim=d, bits=b, seed=42, device="cpu", use_triton=False)

        compressed_triton = comp_triton.compress(states)
        compressed_pytorch = comp_pytorch.compress(states)

        recon_triton = comp_triton.decompress(compressed_triton)
        recon_pytorch = comp_pytorch.decompress(compressed_pytorch)

        assert recon_triton.shape == recon_pytorch.shape, (
            f"Shape mismatch: {recon_triton.shape} vs {recon_pytorch.shape}"
        )
        torch.testing.assert_close(
            recon_triton.float(),
            recon_pytorch.float(),
            atol=1e-2,
            rtol=0.0,
            msg=f"Decompressions differ beyond atol=1e-2 for b={b}, d={d}",
        )
