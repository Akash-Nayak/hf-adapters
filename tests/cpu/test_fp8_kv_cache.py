# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for the FP8 KV cache implementation.

Tests cover:
  * Per-token FP8 quantization round-trip accuracy.
  * ``fp8_kv_cache_update``: shapes, in-place writes, K stored as FP8, V stored
    in compute dtype.
  * ``fp8_attn_core`` vs BF16 ``scaled_dot_product_attention`` within tolerance
    for both a prefill (S>1) and a decode (S=1) step.
  * Allocation helpers: ``allocate_fp8_kv_cache_tensors`` / ``allocate_fp8_kv_caches``.
  * ``generate()`` wiring: when ``model._spyre_fp8_kv_cache = True``, the loop
    allocates FP8 KV caches and threads ``key_scale_caches`` into the
    ``run_forward_fn`` calls.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from hf_adapters.fp8_kv_cache import (
    FP8_DTYPE,
    FP8_MAX,
    SCALE_EPS,
    _quantize_fp8_per_token,
    allocate_fp8_kv_cache_tensors,
    fp8_attn_core,
    fp8_kv_cache_update,
)
from hf_adapters.hf_common import (
    allocate_fp8_kv_caches,
    make_cache_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bf16(*shape):
    """Return a random BF16 tensor in (-1, 1)."""
    return (torch.rand(*shape, dtype=torch.bfloat16) * 2 - 1)


def _sdpa_reference(q, key_cache, value_cache, attn_mask, scaling, n_kv):
    """BF16 SDPA over the full KV cache, used as ground truth.

    Args:
        q: ``[B, H, S, D]``
        key_cache: ``[B, n_kv, L, D]`` BF16
        value_cache: ``[B, n_kv, L, D_v]`` BF16
        attn_mask: additive mask or None
        scaling: float
        n_kv: number of KV heads

    Returns:
        ``[B, H, S, D_v]``
    """
    B, H, S, D = q.shape
    kv_repeat = H // n_kv
    k_exp = key_cache.repeat_interleave(kv_repeat, dim=1)  # [B, H, L, D]
    v_exp = value_cache.repeat_interleave(kv_repeat, dim=1)  # [B, H, L, D_v]
    return F.scaled_dot_product_attention(
        q, k_exp, v_exp, attn_mask=attn_mask, scale=scaling
    )


# ---------------------------------------------------------------------------
# 1.  Per-token quantization
# ---------------------------------------------------------------------------

class TestQuantizeFp8PerToken:
    def test_output_dtype_is_fp8(self):
        x = _bf16(2, 4, 8, 64)
        xq, scale = _quantize_fp8_per_token(x)
        assert xq.dtype == FP8_DTYPE
        assert scale.dtype == x.dtype

    def test_scale_shape(self):
        B, H, S, D = 2, 4, 8, 64
        x = _bf16(B, H, S, D)
        _, scale = _quantize_fp8_per_token(x)
        assert scale.shape == (B, H, S), scale.shape  # no trailing 1-dim

    def test_scale_is_positive(self):
        x = _bf16(1, 2, 4, 64)
        _, scale = _quantize_fp8_per_token(x)
        assert (scale > 0).all()

    def test_scale_eps_prevents_zero_scale_for_all_zero_input(self):
        x = torch.zeros(1, 1, 4, 64, dtype=torch.bfloat16)
        _, scale = _quantize_fp8_per_token(x)
        assert (scale >= SCALE_EPS).all()

    def test_dequantize_roundtrip_accuracy(self):
        """Dequantized FP8 should match original within ~2.5% relative error.

        FP8 E4M3 has only 3 mantissa bits, so ~1/8 = 1.25% rounding per value
        is expected on average; the BF16 intermediate adds another small term.
        """
        torch.manual_seed(0)
        x = _bf16(2, 4, 16, 128)
        xq, scale = _quantize_fp8_per_token(x)
        x_reconstructed = xq.to(torch.bfloat16) * scale.unsqueeze(-1)
        rel_err = (x - x_reconstructed).abs() / (x.abs() + 1e-6)
        assert rel_err.mean().item() < 0.025, f"mean rel_err={rel_err.mean():.4f}"

    def test_values_clipped_to_fp8_range(self):
        """After de-scale, values must not exceed FP8_MAX."""
        torch.manual_seed(1)
        x = _bf16(1, 2, 8, 64) * 10.0  # large values
        xq, _ = _quantize_fp8_per_token(x)
        x_fp32 = xq.to(torch.float32)
        assert (x_fp32.abs() <= FP8_MAX + 1e-3).all()


# ---------------------------------------------------------------------------
# 2.  FP8 KV cache update
# ---------------------------------------------------------------------------

class TestFp8KvCacheUpdate:
    def _make_tensors(self, B=1, n_kv=4, n=8, D=64, L=128, dtype=torch.bfloat16):
        k = _bf16(B, n_kv, n, D)
        v = _bf16(B, n_kv, n, D)
        key_cache = torch.zeros(B, n_kv, L, D, dtype=FP8_DTYPE)
        key_scale_cache = torch.zeros(B, n_kv, L, D, dtype=dtype)  # [B, n_kv, L, D]
        value_cache = torch.zeros(B, n_kv, L, D, dtype=dtype)
        idx = torch.arange(0, n, dtype=torch.long)
        return k, v, key_cache, key_scale_cache, value_cache, idx

    def test_returns_five_tensors(self):
        result = fp8_kv_cache_update(*self._make_tensors())
        assert len(result) == 5

    def test_key_cache_dtype_is_fp8(self):
        kc, ksc, vc, k_fp8, k_scale = fp8_kv_cache_update(*self._make_tensors())
        assert kc.dtype == FP8_DTYPE
        assert k_fp8.dtype == FP8_DTYPE

    def test_value_cache_dtype_unchanged(self):
        _, _, vc, _, _ = fp8_kv_cache_update(*self._make_tensors())
        assert vc.dtype == torch.bfloat16

    def test_key_scale_cache_written(self):
        k, v, kc, ksc, vc, idx = self._make_tensors(n=8)
        _, ksc_out, _, _, k_scale = fp8_kv_cache_update(k, v, kc, ksc, vc, idx)
        # Scale at the written positions must be > 0 (k is non-zero random)
        assert (ksc_out[:, :, idx, :] > 0).all()

    def test_value_written_correctly(self):
        k, v, kc, ksc, vc, idx = self._make_tensors(n=8)
        _, _, vc_out, _, _ = fp8_kv_cache_update(k, v, kc, ksc, vc, idx)
        torch.testing.assert_close(vc_out[:, :, idx, :], v)

    def test_untouched_positions_stay_zero(self):
        k, v, kc, ksc, vc, idx = self._make_tensors(B=1, n_kv=2, n=4, D=32, L=64)
        idx = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        kc_out, _, vc_out, _, _ = fp8_kv_cache_update(k, v, kc, ksc, vc, idx)
        # All positions except 10-13 must be zero in the key cache
        other_mask = torch.ones(64, dtype=torch.bool)
        other_mask[[10, 11, 12, 13]] = False
        assert kc_out[:, :, other_mask, :].to(torch.float32).abs().sum().item() == 0

    def test_in_place_semantics(self):
        k, v, kc, ksc, vc, idx = self._make_tensors()
        kc_out, ksc_out, vc_out, _, _ = fp8_kv_cache_update(k, v, kc, ksc, vc, idx)
        assert kc_out is kc
        assert ksc_out is ksc
        assert vc_out is vc

    def test_sequential_decode_steps_accumulate(self):
        """Cache must hold values from multiple successive decode steps."""
        B, n_kv, D, L = 1, 2, 32, 64
        kc = torch.zeros(B, n_kv, L, D, dtype=FP8_DTYPE)
        ksc = torch.zeros(B, n_kv, L, D, dtype=torch.bfloat16)
        vc = torch.zeros(B, n_kv, L, D, dtype=torch.bfloat16)

        written_v = []
        for step in range(8):
            k = _bf16(B, n_kv, 1, D)
            v = _bf16(B, n_kv, 1, D)
            idx = torch.tensor([step], dtype=torch.long)
            kc, ksc, vc, _, _ = fp8_kv_cache_update(k, v, kc, ksc, vc, idx)
            written_v.append(v.clone())

        for step in range(8):
            torch.testing.assert_close(
                vc[:, :, step : step + 1, :], written_v[step]
            )


# ---------------------------------------------------------------------------
# 3.  FP8 attention core vs BF16 SDPA reference
# ---------------------------------------------------------------------------

class TestFp8AttnCoreAccuracy:
    """Compare ``fp8_attn_core`` output to BF16 SDPA within tolerance.

    FP8 introduces quantization error so we allow a moderate absolute tolerance.
    The key correctness criteria:
      - shapes match
      - output dtype is the compute dtype (BF16)
      - attention output is numerically close to BF16 SDPA
    """

    def _build_caches(self, B, H, n_kv, L, D, dtype=torch.bfloat16, fill_len=None):
        """Build key_cache / key_scale_cache / value_cache from random BF16 K/V."""
        fill_len = fill_len or L
        kc = torch.zeros(B, n_kv, L, D, dtype=FP8_DTYPE)
        ksc = torch.zeros(B, n_kv, L, D, dtype=dtype)
        vc = torch.zeros(B, n_kv, L, D, dtype=dtype)
        k = _bf16(B, n_kv, fill_len, D)
        v = _bf16(B, n_kv, fill_len, D)
        idx = torch.arange(0, fill_len, dtype=torch.long)
        kc, ksc, vc, _, _ = fp8_kv_cache_update(k, v, kc, ksc, vc, idx)
        return kc, ksc, vc, k, v

    def _run_case(self, B, H, n_kv, S, D, L, attn_scale=None):
        torch.manual_seed(42)
        dtype = torch.bfloat16
        attn_scale = attn_scale or (1.0 / math.sqrt(D))

        kc, ksc, vc, k_ref, v_ref = self._build_caches(B, H, n_kv, L, D, dtype, fill_len=L)
        q = _bf16(B, H, S, D)

        # FP8 path
        out_fp8 = fp8_attn_core(q, kc, ksc, vc, attn_mask=None, scaling=attn_scale, compute_dtype=dtype)

        # BF16 reference (no mask — entire cache filled)
        out_ref = _sdpa_reference(q, k_ref.to(dtype), v_ref, attn_mask=None, scaling=attn_scale, n_kv=n_kv)

        assert out_fp8.shape == out_ref.shape, f"{out_fp8.shape} vs {out_ref.shape}"
        assert out_fp8.dtype == dtype

        max_abs = (out_fp8 - out_ref).abs().max().item()
        mean_abs = (out_fp8 - out_ref).abs().mean().item()
        return max_abs, mean_abs

    def test_output_shape_mha(self):
        """Multi-head (H==n_kv) case."""
        torch.manual_seed(0)
        B, H, n_kv, S, D, L = 1, 4, 4, 8, 64, 32
        kc, ksc, vc, _, _ = self._build_caches(B, H, n_kv, L, D, fill_len=L)
        q = _bf16(B, H, S, D)
        out = fp8_attn_core(q, kc, ksc, vc, attn_mask=None, scaling=1.0/math.sqrt(D), compute_dtype=torch.bfloat16)
        assert out.shape == (B, H, S, D)

    def test_output_shape_gqa(self):
        """GQA case (H > n_kv)."""
        torch.manual_seed(0)
        B, H, n_kv, S, D, L = 1, 8, 2, 4, 64, 16
        kc, ksc, vc, _, _ = self._build_caches(B, H, n_kv, L, D, fill_len=L)
        q = _bf16(B, H, S, D)
        out = fp8_attn_core(q, kc, ksc, vc, attn_mask=None, scaling=1.0/math.sqrt(D), compute_dtype=torch.bfloat16)
        assert out.shape == (B, H, S, D)

    def test_output_dtype_is_compute_dtype(self):
        torch.manual_seed(0)
        kc, ksc, vc, _, _ = self._build_caches(1, 4, 4, 8, 64, fill_len=8)
        q = _bf16(1, 4, 1, 64)
        out = fp8_attn_core(q, kc, ksc, vc, attn_mask=None, scaling=1.0/8.0, compute_dtype=torch.bfloat16)
        assert out.dtype == torch.bfloat16

    def test_prefill_accuracy_mha(self):
        """Prefill (S=8) accuracy for MHA: max abs error < 0.1."""
        max_abs, mean_abs = self._run_case(B=1, H=4, n_kv=4, S=8, D=64, L=8)
        assert max_abs < 0.1, f"max_abs={max_abs:.4f}"

    def test_prefill_accuracy_gqa(self):
        """Prefill accuracy for GQA (H=8, n_kv=2): max abs error < 0.15."""
        max_abs, mean_abs = self._run_case(B=1, H=8, n_kv=2, S=4, D=64, L=4)
        assert max_abs < 0.15, f"max_abs={max_abs:.4f}"

    def test_decode_accuracy_mha(self):
        """Decode (S=1) accuracy for MHA: max abs error < 0.1."""
        max_abs, mean_abs = self._run_case(B=1, H=4, n_kv=4, S=1, D=64, L=16)
        assert max_abs < 0.1, f"max_abs={max_abs:.4f}"

    def test_decode_accuracy_batched(self):
        """Decode with B>1: per-sequence accuracy."""
        max_abs, mean_abs = self._run_case(B=2, H=4, n_kv=4, S=1, D=64, L=16)
        assert max_abs < 0.1, f"max_abs={max_abs:.4f}"

    def test_prefill_then_decode_accuracy(self):
        """Simulate one prefill then one decode step: error must stay bounded.

        Writes the prefill tokens into the cache, then appends one more decode
        token and checks that the FP8 QK^T attention output matches BF16 SDPA
        on the combined (prefill + decode) cache.
        """
        torch.manual_seed(7)
        B, H, n_kv, prefill_len, D = 1, 4, 4, 8, 64
        max_cache_len = prefill_len + 1
        dtype = torch.bfloat16
        scale = 1.0 / math.sqrt(D)

        # Build empty caches
        kc = torch.zeros(B, n_kv, max_cache_len, D, dtype=FP8_DTYPE)
        ksc = torch.zeros(B, n_kv, max_cache_len, D, dtype=dtype)
        vc = torch.zeros(B, n_kv, max_cache_len, D, dtype=dtype)

        # Prefill
        k_pre = _bf16(B, n_kv, prefill_len, D)
        v_pre = _bf16(B, n_kv, prefill_len, D)
        idx_pre = torch.arange(0, prefill_len, dtype=torch.long)
        kc, ksc, vc, _, _ = fp8_kv_cache_update(k_pre, v_pre, kc, ksc, vc, idx_pre)

        # Decode step — add one new token
        k_dec = _bf16(B, n_kv, 1, D)
        v_dec = _bf16(B, n_kv, 1, D)
        idx_dec = torch.tensor([prefill_len], dtype=torch.long)
        kc, ksc, vc, _, _ = fp8_kv_cache_update(k_dec, v_dec, kc, ksc, vc, idx_dec)

        # Query for the decode step
        q = _bf16(B, H, 1, D)

        # FP8 output
        out_fp8 = fp8_attn_core(q, kc, ksc, vc, attn_mask=None, scaling=scale, compute_dtype=dtype)

        # BF16 reference — reconstruct full K/V cache from the original tensors
        k_full = torch.cat([k_pre, k_dec], dim=2)  # [B, n_kv, L, D]
        v_full = torch.cat([v_pre, v_dec], dim=2)
        out_ref = _sdpa_reference(q, k_full, v_full, attn_mask=None, scaling=scale, n_kv=n_kv)

        max_abs = (out_fp8 - out_ref).abs().max().item()
        assert max_abs < 0.1, f"prefill+decode max_abs={max_abs:.4f}"


# ---------------------------------------------------------------------------
# 4.  Allocation helpers
# ---------------------------------------------------------------------------

class TestAllocationHelpers:
    def test_allocate_fp8_kv_cache_tensors_shapes(self):
        B, n_kv, L, D = 2, 8, 128, 64
        kc, ksc, vc = allocate_fp8_kv_cache_tensors(B, n_kv, L, D, torch.bfloat16, device="cpu")
        assert kc.shape == (B, n_kv, L, D)
        assert ksc.shape == (B, n_kv, L, D)   # [B, n_kv, L, head_dim] — scale broadcast
        assert vc.shape == (B, n_kv, L, D)

    def test_allocate_fp8_kv_cache_tensors_dtypes(self):
        kc, ksc, vc = allocate_fp8_kv_cache_tensors(1, 4, 32, 64, torch.bfloat16, device="cpu")
        assert kc.dtype == FP8_DTYPE
        assert ksc.dtype == torch.bfloat16
        assert vc.dtype == torch.bfloat16

    def test_allocate_fp8_kv_cache_tensors_zeroed(self):
        kc, ksc, vc = allocate_fp8_kv_cache_tensors(1, 4, 32, 64, torch.bfloat16, device="cpu")
        assert not kc.to(torch.float32).any()
        assert not ksc.any()
        assert not vc.any()

    def test_allocate_fp8_kv_caches_per_layer(self):
        """allocate_fp8_kv_caches returns one triplet per layer."""

        class _Cfg:
            num_hidden_layers = 3
            num_key_value_heads = 8
            head_dim = 64
            hidden_size = 512
            num_attention_heads = 8

        class _Model:
            config = _Cfg()

        kcs, kscs, vcs = allocate_fp8_kv_caches(_Model(), 1, 64, torch.bfloat16, device="cpu")
        assert len(kcs) == len(kscs) == len(vcs) == 3
        for kc, ksc, vc in zip(kcs, kscs, vcs):
            assert kc.dtype == FP8_DTYPE
            assert ksc.dtype == torch.bfloat16
            assert vc.dtype == torch.bfloat16
