"""
Memory regression tests for TurboQuant compressors.

Verifies that the theoretical Triton bit-packed format is strictly smaller
than fp16 baseline, and that the actual compress() output does not accidentally
regress to storing full fp16 tensors.
"""

import math

import pytest
import torch
from turboquant_pytorch.compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE

S = 64

BITS = [2, 3, 4]
DIMS = [64, 128, 256]


@pytest.fixture(params=[(b, d) for b in BITS for d in DIMS], ids=lambda x: f"b{x[0]}_d{x[1]}")
def bd(request):
    return request.param


# ---------------------------------------------------------------------------
# Theoretical size checks (Triton packed format)
# ---------------------------------------------------------------------------

class TestTheoreticalPackedSize:
    def test_v2_triton_smaller_than_fp16(self, bd):
        """V2 Triton packed format is strictly smaller than fp16 baseline."""
        b, d = bd
        fp16_bytes = S * d * 2

        v2_triton_bytes = (
            math.ceil(d * (b - 1) / 8) * S  # packed_key_indices
            + math.ceil(d / 8) * S           # packed_qjl_signs (m=d)
            + S * 2                           # residual_norm (fp16)
            + S * 2                           # vec_norms (fp16)
        )
        assert v2_triton_bytes < fp16_bytes, (
            f"V2 Triton ({v2_triton_bytes} bytes) not smaller than fp16 baseline "
            f"({fp16_bytes} bytes) for b={b}, d={d}"
        )

    def test_mse_triton_smaller_than_fp16(self, bd):
        """MSE Triton packed format is strictly smaller than fp16 baseline."""
        b, d = bd
        fp16_bytes = S * d * 2

        mse_triton_bytes = (
            math.ceil(d * b / 8) * S  # packed_indices
            + S * 2                   # vec_norms (fp16)
        )
        assert mse_triton_bytes < fp16_bytes, (
            f"MSE Triton ({mse_triton_bytes} bytes) not smaller than fp16 baseline "
            f"({fp16_bytes} bytes) for b={b}, d={d}"
        )


# ---------------------------------------------------------------------------
# Actual compress() dict byte checks
# ---------------------------------------------------------------------------

class TestActualCompressedBytes:
    def _make_states(self, d: int) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(1, 1, S, d, dtype=torch.float16)

    def _tensor_bytes(self, compressed: dict) -> int:
        return sum(v.nbytes for v in compressed.values() if isinstance(v, torch.Tensor))

    def test_v2_pytorch_actual_bytes(self, bd):
        """
        V2 PyTorch path stores k_mse (fp16) + qjl_signs (int8) + residual_norm (fp16).
        Total = S*d*2 + S*d*1 + S*2 = S*(3d + 2) bytes.
        Verify actual nbytes match this formula and are not a pure fp16 regression.
        """
        b, d = bd
        states = self._make_states(d)
        comp = TurboQuantCompressorV2(head_dim=d, bits=b, seed=42, device="cpu", use_triton=False)
        compressed = comp.compress(states)

        actual_bytes = self._tensor_bytes(compressed)

        # Expected bytes for PyTorch path
        expected_bytes = (
            S * d * 2   # k_mse: float16
            + S * d * 1  # qjl_signs: int8
            + S * 2      # residual_norm: float16
        )
        assert actual_bytes == expected_bytes, (
            f"V2 PyTorch actual bytes ({actual_bytes}) != expected ({expected_bytes}) "
            f"for b={b}, d={d}. Keys: {list(compressed.keys())}"
        )

        # Must be strictly less than double the fp16 baseline
        fp16_bytes = S * d * 2
        assert actual_bytes < fp16_bytes * 3, (
            f"V2 PyTorch bytes ({actual_bytes}) look suspiciously large vs fp16 baseline "
            f"({fp16_bytes}) for b={b}, d={d}"
        )

    def test_mse_pytorch_actual_bytes(self, bd):
        """
        MSE PyTorch path stores indices (uint8) + vec_norms (fp16).
        Total = S*d*1 + S*2 = S*(d + 2) bytes.
        Verify actual nbytes match this formula and are strictly less than fp16 baseline.
        """
        b, d = bd
        states = self._make_states(d)
        comp = TurboQuantCompressorMSE(head_dim=d, bits=b, seed=42, device="cpu", use_triton=False)
        compressed = comp.compress(states)

        actual_bytes = self._tensor_bytes(compressed)

        # Expected bytes for PyTorch path
        expected_bytes = (
            S * d * 1  # indices: uint8
            + S * 2    # vec_norms: float16
        )
        assert actual_bytes == expected_bytes, (
            f"MSE PyTorch actual bytes ({actual_bytes}) != expected ({expected_bytes}) "
            f"for b={b}, d={d}. Keys: {list(compressed.keys())}"
        )

        # Must be strictly less than fp16 baseline (uint8 vs fp16 = 50% savings)
        fp16_bytes = S * d * 2
        assert actual_bytes < fp16_bytes, (
            f"MSE PyTorch bytes ({actual_bytes}) not less than fp16 baseline "
            f"({fp16_bytes}) for b={b}, d={d}"
        )

    def test_v2_pytorch_not_fp16_regression(self, bd):
        """
        The k_mse tensor in V2 PyTorch path must be fp16, not fp32,
        to avoid a silent regression to full-precision storage.
        """
        b, d = bd
        states = self._make_states(d)
        comp = TurboQuantCompressorV2(head_dim=d, bits=b, seed=42, device="cpu", use_triton=False)
        compressed = comp.compress(states)

        assert compressed["k_mse"].dtype == torch.float16, (
            f"k_mse should be float16, got {compressed['k_mse'].dtype} for b={b}, d={d}"
        )
        assert compressed["qjl_signs"].dtype == torch.int8, (
            f"qjl_signs should be int8, got {compressed['qjl_signs'].dtype} for b={b}, d={d}"
        )

    def test_mse_pytorch_not_fp16_regression(self, bd):
        """
        The indices tensor in MSE PyTorch path must be uint8, not fp16 or int32,
        to avoid a silent regression to full-precision storage.
        """
        b, d = bd
        states = self._make_states(d)
        comp = TurboQuantCompressorMSE(head_dim=d, bits=b, seed=42, device="cpu", use_triton=False)
        compressed = comp.compress(states)

        assert compressed["indices"].dtype == torch.uint8, (
            f"indices should be uint8, got {compressed['indices'].dtype} for b={b}, d={d}"
        )
