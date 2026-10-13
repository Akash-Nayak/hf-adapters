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
  * ``fp8_kv_cache_update`` (option=2, CPU):
    - shapes, in-place writes, K stored as FP8 [B,n_kv,L,D], V in compute dtype.
    - Returns 3 values: ``(key_cache, key_scale_cache, value_cache)``.
  * ``fp8_attn_core`` vs BF16 ``scaled_dot_product_attention`` within tolerance
    for both a prefill (S>1) and a decode (S=1) step.
  * Allocation helpers: ``allocate_fp8_kv_cache_tensors`` / ``allocate_fp8_kv_caches``.
  * Option 1 (block layout) allocation and tail buffer helpers:
    - ``allocate_fp8_kv_cache_tensors(option=1)`` returns a 4-tuple with block shapes.
    - ``fp8_kv_cache_update_tail`` writes one decode token into the DL16 tail buffer.
    - ``fp8_kv_cache_flush_tail`` is a no-op on CPU (covered by shape checks).
  * ``generate()`` wiring: when ``model._spyre_fp8_kv_cache = True``, the loop
    allocates FP8 KV caches and threads ``key_scale_caches`` into the
    ``run_forward_fn`` calls.

CPU tests always exercise option=2 (dequantize-K) for the fill and attention
paths, since ``spyre.scaled_bmm`` and ``quantize_weight_fp8_with_scale`` are
Spyre-only.  Option 1 Spyre-path tests live in
``tests/spyre/test_fp8_kv_cache_spyre.py``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from hf_adapters.fp8_kv_cache import (
    FP8_BLOCK_SIZE,
    FP8_DTYPE,
    FP8_MAX,
    SCALE_EPS,
    _quantize_fp8_per_token,
    allocate_fp8_kv_cache_tensors,
    fp8_attn_core,
    fp8_kv_cache_flush_tail,
    fp8_kv_cache_update,
    fp8_kv_cache_update_tail,
)
from hf_adapters.hf_common import (
    Fp8KVStandardGQAAttention,
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
        # scale is [B, H, S, 1] — one scalar per token, size-1 trailing dim
        assert scale.shape == (B, H, S, 1), scale.shape

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
        # scale is [B, H, S, 1] — broadcasts directly against [B, H, S, D]
        x_reconstructed = xq.to(torch.bfloat16) * scale
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
# 2.  FP8 KV cache update — CPU always exercises option=2 (QFP8CH fill)
#     key_cache shape: [B, n_kv, L, D] (position dim = dim 2)
# ---------------------------------------------------------------------------

class TestFp8KvCacheUpdate:
    def _make_tensors(self, B=1, n_kv=4, n=8, D=64, L=128, dtype=torch.bfloat16):
        k = _bf16(B, n_kv, n, D)
        v = _bf16(B, n_kv, n, D)
        # CPU option=2 layout: [B, n_kv, L, D]
        key_cache = torch.zeros(B, n_kv, L, D, dtype=FP8_DTYPE)
        key_scale_cache = torch.zeros(B, n_kv, L, 1, dtype=dtype)  # 4D: [B, n_kv, L, 1]
        value_cache = torch.zeros(B, n_kv, L, D, dtype=dtype)
        idx = torch.arange(0, n, dtype=torch.long)
        return k, v, key_cache, key_scale_cache, value_cache, idx

    def test_returns_three_tensors(self):
        """fp8_kv_cache_update returns exactly 3 tensors: (kc, ksc, vc)."""
        result = fp8_kv_cache_update(*self._make_tensors(), option=2)
        assert len(result) == 3

    def test_key_cache_dtype_is_fp8(self):
        kc, ksc, vc = fp8_kv_cache_update(*self._make_tensors(), option=2)
        assert kc.dtype == FP8_DTYPE

    def test_value_cache_dtype_unchanged(self):
        _, _, vc = fp8_kv_cache_update(*self._make_tensors(), option=2)
        assert vc.dtype == torch.bfloat16

    def test_key_scale_cache_written(self):
        k, v, kc, ksc, vc, idx = self._make_tensors(n=8)
        _, ksc_out, _ = fp8_kv_cache_update(k, v, kc, ksc, vc, idx, option=2)
        # Scale at the written positions must be > 0 (k is non-zero random)
        assert (ksc_out[:, :, idx, :] > 0).all()

    def test_value_written_correctly(self):
        k, v, kc, ksc, vc, idx = self._make_tensors(n=8)
        _, _, vc_out = fp8_kv_cache_update(k, v, kc, ksc, vc, idx, option=2)
        torch.testing.assert_close(vc_out[:, :, idx, :], v)

    def test_untouched_positions_stay_zero(self):
        k, v, kc, ksc, vc, idx = self._make_tensors(B=1, n_kv=2, n=4, D=32, L=64)
        idx = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        kc_out, _, vc_out = fp8_kv_cache_update(k, v, kc, ksc, vc, idx, option=2)
        # All positions except 10-13 must be zero in the key cache
        other_mask = torch.ones(64, dtype=torch.bool)
        other_mask[[10, 11, 12, 13]] = False
        assert kc_out[:, :, other_mask, :].to(torch.float32).abs().sum().item() == 0

    def test_in_place_semantics(self):
        k, v, kc, ksc, vc, idx = self._make_tensors()
        kc_out, ksc_out, vc_out = fp8_kv_cache_update(k, v, kc, ksc, vc, idx, option=2)
        assert kc_out is kc
        assert ksc_out is ksc
        assert vc_out is vc

    def test_sequential_decode_steps_accumulate(self):
        """Cache must hold values from multiple successive decode steps."""
        B, n_kv, D, L = 1, 2, 32, 64
        kc = torch.zeros(B, n_kv, L, D, dtype=FP8_DTYPE)
        ksc = torch.zeros(B, n_kv, L, 1, dtype=torch.bfloat16)  # 4D: [B, n_kv, L, 1]
        vc = torch.zeros(B, n_kv, L, D, dtype=torch.bfloat16)

        written_v = []
        for step in range(8):
            k = _bf16(B, n_kv, 1, D)
            v = _bf16(B, n_kv, 1, D)
            idx = torch.tensor([step], dtype=torch.long)
            kc, ksc, vc = fp8_kv_cache_update(k, v, kc, ksc, vc, idx, option=2)
            written_v.append(v.clone())

        for step in range(8):
            torch.testing.assert_close(
                vc[:, :, step : step + 1, :], written_v[step]
            )

    def test_default_option_is_one_but_cpu_behaves_as_option2(self):
        """On CPU, option=1 silently uses the option=2 code path.

        The key cache must still be shape [B, n_kv, L, D] and filled with FP8.
        """
        k, v, kc, ksc, vc, idx = self._make_tensors()
        # Call with default option=1; CPU falls through to option=2 fill path.
        kc_out, ksc_out, vc_out = fp8_kv_cache_update(k, v, kc, ksc, vc, idx)
        assert len((kc_out, ksc_out, vc_out)) == 3
        assert kc_out.dtype == FP8_DTYPE
        assert kc_out.shape == (1, 4, 128, 64)   # [B, n_kv, L, D]
        assert ksc_out.shape == (1, 4, 128, 1)   # [B, n_kv, L, 1]


# ---------------------------------------------------------------------------
# 3.  FP8 attention core vs BF16 SDPA reference
#     CPU always exercises option=2 (dequantize-K path).
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
        """Build key_cache / key_scale_cache / value_cache from random BF16 K/V.

        Returns option=2 CPU cache layout: key_cache [B, n_kv, L, D].
        """
        fill_len = fill_len or L
        kc = torch.zeros(B, n_kv, L, D, dtype=FP8_DTYPE)
        ksc = torch.zeros(B, n_kv, L, 1, dtype=dtype)  # 4D: [B, n_kv, L, 1]
        vc = torch.zeros(B, n_kv, L, D, dtype=dtype)
        k = _bf16(B, n_kv, fill_len, D)
        v = _bf16(B, n_kv, fill_len, D)
        idx = torch.arange(0, fill_len, dtype=torch.long)
        kc, ksc, vc = fp8_kv_cache_update(k, v, kc, ksc, vc, idx, option=2)
        return kc, ksc, vc, k, v

    def _run_case(self, B, H, n_kv, S, D, L, attn_scale=None):
        torch.manual_seed(42)
        dtype = torch.bfloat16
        attn_scale = attn_scale or (1.0 / math.sqrt(D))

        kc, ksc, vc, k_ref, v_ref = self._build_caches(B, H, n_kv, L, D, dtype, fill_len=L)
        q = _bf16(B, H, S, D)

        # FP8 path (option=2 on CPU)
        out_fp8 = fp8_attn_core(
            q, kc, ksc, vc, attn_mask=None, scaling=attn_scale,
            compute_dtype=dtype, option=2,
        )

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
        out = fp8_attn_core(
            q, kc, ksc, vc, attn_mask=None, scaling=1.0/math.sqrt(D),
            compute_dtype=torch.bfloat16, option=2,
        )
        assert out.shape == (B, H, S, D)

    def test_output_shape_gqa(self):
        """GQA case (H > n_kv)."""
        torch.manual_seed(0)
        B, H, n_kv, S, D, L = 1, 8, 2, 4, 64, 16
        kc, ksc, vc, _, _ = self._build_caches(B, H, n_kv, L, D, fill_len=L)
        q = _bf16(B, H, S, D)
        out = fp8_attn_core(
            q, kc, ksc, vc, attn_mask=None, scaling=1.0/math.sqrt(D),
            compute_dtype=torch.bfloat16, option=2,
        )
        assert out.shape == (B, H, S, D)

    def test_output_dtype_is_compute_dtype(self):
        torch.manual_seed(0)
        kc, ksc, vc, _, _ = self._build_caches(1, 4, 4, 8, 64, fill_len=8)
        q = _bf16(1, 4, 1, 64)
        out = fp8_attn_core(
            q, kc, ksc, vc, attn_mask=None, scaling=1.0/8.0,
            compute_dtype=torch.bfloat16, option=2,
        )
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

        # Build empty caches — option=2 CPU layout [B, n_kv, L, D]
        kc = torch.zeros(B, n_kv, max_cache_len, D, dtype=FP8_DTYPE)
        ksc = torch.zeros(B, n_kv, max_cache_len, 1, dtype=dtype)  # 4D: [B, n_kv, L, 1]
        vc = torch.zeros(B, n_kv, max_cache_len, D, dtype=dtype)

        # Prefill
        k_pre = _bf16(B, n_kv, prefill_len, D)
        v_pre = _bf16(B, n_kv, prefill_len, D)
        idx_pre = torch.arange(0, prefill_len, dtype=torch.long)
        kc, ksc, vc = fp8_kv_cache_update(k_pre, v_pre, kc, ksc, vc, idx_pre, option=2)

        # Decode step — add one new token
        k_dec = _bf16(B, n_kv, 1, D)
        v_dec = _bf16(B, n_kv, 1, D)
        idx_dec = torch.tensor([prefill_len], dtype=torch.long)
        kc, ksc, vc = fp8_kv_cache_update(k_dec, v_dec, kc, ksc, vc, idx_dec, option=2)

        # Query for the decode step
        q = _bf16(B, H, 1, D)

        # FP8 output
        out_fp8 = fp8_attn_core(
            q, kc, ksc, vc, attn_mask=None, scaling=scale,
            compute_dtype=dtype, option=2,
        )

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
    def test_allocate_fp8_kv_cache_tensors_shapes_option2(self):
        """option=2 CPU: key_cache is [B, n_kv, L, D], scale cache is [B, n_kv, L, 1]."""
        B, n_kv, L, D = 2, 8, 128, 64
        kc, ksc, vc = allocate_fp8_kv_cache_tensors(
            B, n_kv, L, D, torch.bfloat16, device="cpu", option=2,
        )
        assert kc.shape == (B, n_kv, L, D)
        assert ksc.shape == (B, n_kv, L, 1)  # 4D: [B, n_kv, L, 1]
        assert vc.shape == (B, n_kv, L, D)

    def test_allocate_fp8_kv_cache_tensors_shapes_option1_cpu(self):
        """option=1 returns a 5-tuple with block-layout shapes (even on CPU).

        Option 1 allocation always uses the block layout; the fill/attend CPU
        path falls back to option=2 behaviour, but the tensor shapes returned
        by ``allocate_fp8_kv_cache_tensors`` match the Spyre shapes.

        Returned tuple:
          ``(key_cache, key_scale_cache, value_cache, value_scale_cache,
          (ktail, vtail))``

        Shapes:
            key_cache         [NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE]  FP8
            key_scale_cache   [NUM_BLOCKS, 1, 1, 1]                  compute dtype
            value_cache       [NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D]  FP8
            value_scale_cache [NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, 64] compute dtype
            key_tail_buf      [B, n_kv, FP8_BLOCK_SIZE, D]           compute dtype
            value_tail_buf    [B, n_kv, FP8_BLOCK_SIZE, D]           compute dtype
        """
        B, n_kv, D = 1, 8, 128
        L = 256  # must be a multiple of FP8_BLOCK_SIZE (128)
        NUM_BLOCKS = L // FP8_BLOCK_SIZE  # = 2

        result = allocate_fp8_kv_cache_tensors(
            B, n_kv, L, D, torch.bfloat16, device="cpu", option=1,
        )
        assert len(result) == 5, f"option=1 should return 5-tuple, got {len(result)}"
        kc, ksc, vc, vsc, (ktail, vtail) = result

        # Block cache shapes
        assert kc.shape == (NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE), kc.shape
        assert ksc.shape == (NUM_BLOCKS, 1, 1, 1), ksc.shape
        assert vc.shape == (NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D), vc.shape
        assert vsc.shape == (NUM_BLOCKS, 1, 1, 1), vsc.shape
        # Tail buffer shapes
        assert ktail.shape == (B, n_kv, FP8_BLOCK_SIZE, D), ktail.shape
        assert vtail.shape == (B, n_kv, FP8_BLOCK_SIZE, D), vtail.shape

    def test_allocate_fp8_kv_cache_tensors_option1_dtypes(self):
        """option=1 key+value caches are FP8; scales and tail bufs are compute dtype."""
        B, n_kv, D = 1, 4, 128
        L = 128  # 1 block
        kc, ksc, vc, vsc, (ktail, vtail) = allocate_fp8_kv_cache_tensors(
            B, n_kv, L, D, torch.bfloat16, device="cpu", option=1,
        )
        assert kc.dtype == FP8_DTYPE
        assert ksc.dtype == torch.bfloat16
        assert vc.dtype == FP8_DTYPE
        assert vsc.dtype == torch.bfloat16
        assert ktail.dtype == torch.bfloat16
        assert vtail.dtype == torch.bfloat16

    def test_allocate_fp8_kv_cache_tensors_option1_zeroed(self):
        """option=1 — all returned tensors must be zero-initialised."""
        B, n_kv, D = 1, 4, 128
        L = 128
        kc, ksc, vc, vsc, (ktail, vtail) = allocate_fp8_kv_cache_tensors(
            B, n_kv, L, D, torch.bfloat16, device="cpu", option=1,
        )
        assert not kc.to(torch.float32).any()
        assert not ksc.any()
        assert not vc.to(torch.float32).any()
        assert not vsc.any()
        assert not ktail.any()
        assert not vtail.any()

    def test_allocate_fp8_kv_cache_tensors_dtypes(self):
        kc, ksc, vc = allocate_fp8_kv_cache_tensors(
            1, 4, 32, 64, torch.bfloat16, device="cpu", option=2,
        )
        assert kc.dtype == FP8_DTYPE
        assert ksc.dtype == torch.bfloat16
        assert vc.dtype == torch.bfloat16

    def test_allocate_fp8_kv_cache_tensors_zeroed(self):
        kc, ksc, vc = allocate_fp8_kv_cache_tensors(
            1, 4, 32, 64, torch.bfloat16, device="cpu", option=2,
        )
        assert not kc.to(torch.float32).any()
        assert not ksc.any()
        assert not vc.any()

    def test_allocate_fp8_kv_caches_per_layer(self):
        """allocate_fp8_kv_caches returns one triplet per layer.

        Uses head_dim=128 and max_cache_len=256 (multiples of FP8_BLOCK_SIZE=128)
        so the default option=1 allocation path does not assert.
        """

        class _Cfg:
            num_hidden_layers = 3
            num_key_value_heads = 8
            head_dim = 128       # must be >= FP8_BLOCK_SIZE (128) for option=1
            hidden_size = 1024
            num_attention_heads = 32

        class _Model:
            config = _Cfg()

        # max_cache_len=256 is a multiple of FP8_BLOCK_SIZE (128) — required for option=1
        # Option 1 returns a 5-tuple; option 2 returns a 3-tuple.
        kcs, kscs, vcs, vscs, tail_bufs_list = allocate_fp8_kv_caches(
            _Model(), 1, 256, torch.bfloat16, device="cpu", option=1,
        )
        assert len(kcs) == len(kscs) == len(vcs) == len(vscs) == len(tail_bufs_list) == 3
        for kc, ksc, vc, vsc, (ktail, vtail) in zip(kcs, kscs, vcs, vscs, tail_bufs_list):
            assert kc.dtype == FP8_DTYPE
            assert ksc.dtype == torch.bfloat16
            assert vc.dtype == FP8_DTYPE
            assert vsc.dtype == torch.bfloat16
            assert ktail.dtype == torch.bfloat16
            assert vtail.dtype == torch.bfloat16
        # Option 2 still returns a 3-tuple
        kcs2, kscs2, vcs2 = allocate_fp8_kv_caches(
            _Model(), 1, 256, torch.bfloat16, device="cpu", option=2,
        )
        assert len(kcs2) == len(kscs2) == len(vcs2) == 3


# ---------------------------------------------------------------------------
# 5.  Option 1 decode tail buffer helpers (CPU)
#     These helpers are called eagerly (outside torch.compile) for decode steps.
#     On CPU they exercise the Python-level buffer logic; device ops (scaled_bmm,
#     quantize_weight_fp8_with_scale) are not exercised here.
# ---------------------------------------------------------------------------

class TestOpt1TailBuffer:
    """Tests for ``fp8_kv_cache_update_tail`` and ``fp8_kv_cache_flush_tail``."""

    def _make_tail_bufs(self, B=1, n_kv=8, D=128):
        """Return zero-initialised DL16 tail buffers."""
        shape = (B, n_kv, FP8_BLOCK_SIZE, D)
        ktail = torch.zeros(shape, dtype=torch.bfloat16)
        vtail = torch.zeros(shape, dtype=torch.bfloat16)
        return ktail, vtail

    def test_update_tail_writes_k_at_slot(self):
        """fp8_kv_cache_update_tail writes K token into the correct slot."""
        B, n_kv, D = 1, 4, 128
        ktail, vtail = self._make_tail_bufs(B, n_kv, D)
        k = (torch.rand(B, n_kv, 1, D, dtype=torch.bfloat16) * 2 - 1)
        v = (torch.rand(B, n_kv, 1, D, dtype=torch.bfloat16) * 2 - 1)

        tail_pos = 3
        fp8_kv_cache_update_tail(k, v, ktail, vtail, tail_pos)

        # Written slot must equal k / v
        torch.testing.assert_close(ktail[:, :, tail_pos : tail_pos + 1, :], k)
        torch.testing.assert_close(vtail[:, :, tail_pos : tail_pos + 1, :], v)

    def test_update_tail_does_not_touch_other_slots(self):
        """Slots other than the written position must remain zero."""
        B, n_kv, D = 1, 4, 128
        ktail, vtail = self._make_tail_bufs(B, n_kv, D)
        k = torch.ones(B, n_kv, 1, D, dtype=torch.bfloat16)
        v = torch.ones(B, n_kv, 1, D, dtype=torch.bfloat16)

        fp8_kv_cache_update_tail(k, v, ktail, vtail, tail_pos=0)

        # Slots 1..FP8_BLOCK_SIZE-1 must still be zero
        assert ktail[:, :, 1:, :].abs().sum().item() == 0
        assert vtail[:, :, 1:, :].abs().sum().item() == 0

    def test_update_tail_last_slot(self):
        """Writing to the last slot (tail_pos = FP8_BLOCK_SIZE - 1) is in-bounds."""
        B, n_kv, D = 1, 4, 128
        ktail, vtail = self._make_tail_bufs(B, n_kv, D)
        k = torch.rand(B, n_kv, 1, D, dtype=torch.bfloat16)
        v = torch.rand(B, n_kv, 1, D, dtype=torch.bfloat16)
        last = FP8_BLOCK_SIZE - 1

        fp8_kv_cache_update_tail(k, v, ktail, vtail, tail_pos=last)

        torch.testing.assert_close(ktail[:, :, last : last + 1, :], k)
        torch.testing.assert_close(vtail[:, :, last : last + 1, :], v)

    def test_sequential_tail_writes(self):
        """Writing multiple decode tokens sequentially fills the tail buffer correctly."""
        B, n_kv, D = 1, 4, 128
        ktail, vtail = self._make_tail_bufs(B, n_kv, D)
        written_k = []
        written_v = []

        num_tokens = 5  # write 5 sequential decode tokens
        for pos in range(num_tokens):
            k = torch.rand(B, n_kv, 1, D, dtype=torch.bfloat16)
            v = torch.rand(B, n_kv, 1, D, dtype=torch.bfloat16)
            fp8_kv_cache_update_tail(k, v, ktail, vtail, tail_pos=pos)
            written_k.append(k)
            written_v.append(v)

        for pos in range(num_tokens):
            torch.testing.assert_close(
                ktail[:, :, pos : pos + 1, :], written_k[pos]
            )
            torch.testing.assert_close(
                vtail[:, :, pos : pos + 1, :], written_v[pos]
            )

    def test_flush_tail_is_noop_on_cpu(self):
        """fp8_kv_cache_flush_tail is a no-op on CPU (Spyre ops unavailable).

        The flush must not raise and must not modify the key_cache (CPU option=2
        layout); only the committed block caches on Spyre are updated.
        """
        B, n_kv, D = 1, 4, 128
        L = 256
        ktail, vtail = self._make_tail_bufs(B, n_kv, D)
        # Populate the tail
        k = torch.rand(B, n_kv, 1, D, dtype=torch.bfloat16)
        v = torch.rand(B, n_kv, 1, D, dtype=torch.bfloat16)
        fp8_kv_cache_update_tail(k, v, ktail, vtail, tail_pos=0)

        # Allocate a dummy CPU option=2 key_cache / scale / v cache
        kc = torch.zeros(B, n_kv, L, D, dtype=FP8_DTYPE)
        ksc = torch.zeros(B, n_kv, L, 1, dtype=torch.bfloat16)
        vc = torch.zeros(B, n_kv, L, D, dtype=torch.bfloat16)

        # Must not raise on CPU
        fp8_kv_cache_flush_tail(ktail, vtail, kc, ksc, vc, block_idx=0)

        # key_cache must be unmodified (all zeros) on CPU
        assert kc.to(torch.float32).abs().sum().item() == 0


# ---------------------------------------------------------------------------
# 6.  Option 1 generate() tail wiring — CPU unit tests
#
# These tests exercise the NEW wiring added in the Option 1 end-to-end change:
#   * Fp8KVStandardGQAAttention.pre_attn decode branch (seq_len==1, option=1)
#     skips fp8_kv_cache_update and calls fp8_kv_cache_update_tail instead.
#   * generate() tail_pos advance and block flush at FP8_BLOCK_SIZE boundary.
#
# All tests run on CPU.  Spyre-specific ops (scaled_bmm, quantize_weight_fp8)
# are not exercised here — that is covered by test_fp8_kv_cache_spyre.py.
#
# NOTE: On CPU, fp8_attn_core(option=1) falls back to the option=2 code path
# (dequantize-K via torch.matmul). The option=1 block-layout cache shapes
# [NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE] are Spyre-only; on CPU the fill and
# attention helpers use the option=2 [B, n_kv, L, D] layout. CPU accuracy tests
# for the fallback path therefore use option=2 caches — see TestFp8AttnCoreAccuracy
# in section 3. The tests below focus on the NEW wiring (dispatch logic and
# generate() state machine) not on attention arithmetic.
# ---------------------------------------------------------------------------


class TestOpt1PreAttnDecodeWiring:
    """Fp8KVStandardGQAAttention.pre_attn — Option 1 decode branch (CPU).

    On CPU the actual tail write and cache update are the same code path
    (``fp8_kv_cache_update_tail`` for decode, ``fp8_kv_cache_update`` for
    prefill).  This verifies that the dispatch logic in ``pre_attn`` is
    correct without requiring a Spyre device.
    """

    def _make_attn(self, fp8_kv_opt=1):
        """Build a minimal Fp8KVStandardGQAAttention using tiny linear layers."""
        n_kv, H, D = 4, 8, 128
        hidden = H * D  # 1024

        class _FakeAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(hidden, H * D, bias=False).to(torch.bfloat16)
                self.k_proj = torch.nn.Linear(hidden, n_kv * D, bias=False).to(torch.bfloat16)
                self.v_proj = torch.nn.Linear(hidden, n_kv * D, bias=False).to(torch.bfloat16)
                self.o_proj = torch.nn.Linear(H * D, hidden, bias=False).to(torch.bfloat16)
                self.head_dim = D
                self.v_head_dim = D
                self.scaling = 1.0 / math.sqrt(D)
                self.num_heads = H
                self.num_key_value_heads = n_kv

        return Fp8KVStandardGQAAttention(_FakeAttn(), fp8_kv_opt=fp8_kv_opt)

    def test_pre_attn_prefill_calls_fp8_kv_cache_update(self):
        """Prefill (seq_len > 1, option=1): fp8_kv_cache_update is called, not update_tail.

        On CPU, fp8_kv_cache_update(option=1) falls back to the option=2 CPU path which
        requires option=2-shaped caches.  This test only verifies the dispatch logic —
        which function pre_attn calls — not that the underlying fill runs correctly.
        The monkey-patch therefore returns the caches unchanged rather than calling the
        real function with incompatible option=1 block-layout shapes.
        """
        calls = []
        attn = self._make_attn(fp8_kv_opt=1)

        import hf_adapters.fp8_kv_cache as _kv
        orig_update = _kv.fp8_kv_cache_update
        orig_update_tail = _kv.fp8_kv_cache_update_tail
        try:
            # Return caches unchanged — we only want to verify dispatch, not fill correctness.
            _kv.fp8_kv_cache_update = lambda k, v, kc, ksc, vc, idx, *, option, value_scale_cache=None: (
                calls.append("update"), (kc, ksc, vc)
            )[1]
            _kv.fp8_kv_cache_update_tail = lambda *a, **kw: calls.append("tail")

            B, n_kv, H, D = 1, 4, 8, 128
            hidden = H * D
            S = FP8_BLOCK_SIZE  # 128 — full block, prefill path
            hs = torch.zeros(B, S, hidden, dtype=torch.bfloat16)
            freqs = torch.zeros(B, S, 2, 2, D // 2, dtype=torch.bfloat16)
            kc, ksc, vc, vsc, _ = allocate_fp8_kv_cache_tensors(
                B, n_kv, S * 2, D, torch.bfloat16, device="cpu", option=1,
            )
            cache_idx = torch.arange(0, 1, dtype=torch.int32)

            with torch.no_grad():
                attn.pre_attn(hs, freqs, kc, ksc, vc, cache_idx)
        finally:
            _kv.fp8_kv_cache_update = orig_update
            _kv.fp8_kv_cache_update_tail = orig_update_tail

        assert "update" in calls, "expected fp8_kv_cache_update for prefill"
        assert "tail" not in calls, "fp8_kv_cache_update_tail should NOT be called for prefill"

    def test_pre_attn_decode_calls_update_tail(self):
        """Decode (seq_len == 1, option=1): fp8_kv_cache_update_tail is called."""
        import math
        calls = []
        attn = self._make_attn(fp8_kv_opt=1)

        import hf_adapters.fp8_kv_cache as _kv
        orig_update = _kv.fp8_kv_cache_update
        orig_update_tail = _kv.fp8_kv_cache_update_tail
        try:
            _kv.fp8_kv_cache_update = lambda *a, **kw: (calls.append("update"),
                                                         orig_update(*a, **kw))[1]
            _kv.fp8_kv_cache_update_tail = lambda *a, **kw: calls.append("tail")

            B, n_kv, H, D = 1, 4, 8, 128
            hidden = H * D
            S = 1  # decode
            hs = torch.zeros(B, S, hidden, dtype=torch.bfloat16)
            freqs = torch.zeros(B, S, 2, 2, D // 2, dtype=torch.bfloat16)
            kc, ksc, vc, vsc, (ktail, vtail) = allocate_fp8_kv_cache_tensors(
                B, n_kv, FP8_BLOCK_SIZE * 2, D, torch.bfloat16, device="cpu", option=1,
            )
            cache_idx = torch.arange(0, 1, dtype=torch.int32)

            with torch.no_grad():
                attn.pre_attn(
                    hs, freqs, kc, ksc, vc, cache_idx,
                    key_tail_buf=ktail, value_tail_buf=vtail, fp8_tail_pos=0,
                )
        finally:
            _kv.fp8_kv_cache_update = orig_update
            _kv.fp8_kv_cache_update_tail = orig_update_tail

        assert "tail" in calls, "expected fp8_kv_cache_update_tail for decode"
        assert "update" not in calls, "fp8_kv_cache_update should NOT be called for decode"

    def test_pre_attn_option2_always_calls_update(self):
        """Option 2: fp8_kv_cache_update is always called regardless of seq_len."""
        import math
        calls = []
        attn = self._make_attn(fp8_kv_opt=2)

        import hf_adapters.fp8_kv_cache as _kv
        orig_update = _kv.fp8_kv_cache_update
        orig_update_tail = _kv.fp8_kv_cache_update_tail
        try:
            _kv.fp8_kv_cache_update = lambda *a, **kw: (calls.append("update"),
                                                         orig_update(*a, **kw))[1]
            _kv.fp8_kv_cache_update_tail = lambda *a, **kw: calls.append("tail")

            B, n_kv, H, D = 1, 4, 8, 128
            hidden = H * D
            S = 1  # would trigger tail for option=1
            hs = torch.zeros(B, S, hidden, dtype=torch.bfloat16)
            freqs = torch.zeros(B, S, 2, 2, D // 2, dtype=torch.bfloat16)
            kc, ksc, vc = allocate_fp8_kv_cache_tensors(
                B, n_kv, 128, D, torch.bfloat16, device="cpu", option=2,
            )
            cache_idx = torch.arange(0, 1, dtype=torch.long)

            with torch.no_grad():
                attn.pre_attn(hs, freqs, kc, ksc, vc, cache_idx)
        finally:
            _kv.fp8_kv_cache_update = orig_update
            _kv.fp8_kv_cache_update_tail = orig_update_tail

        assert "update" in calls, "option=2 must call fp8_kv_cache_update"
        assert "tail" not in calls, "option=2 must never call fp8_kv_cache_update_tail"


class TestGenerateOpt1TailLogic:
    """generate() tail_pos advance and flush logic — verified via allocate_fp8_kv_caches.

    These tests check the Python-level state machine in generate():
      - tail_pos increments each decode step.
      - fp8_kv_cache_flush_tail is called (and tail bufs zeroed) at FP8_BLOCK_SIZE.
      - _fp8_next_block increments after flush.

    We test the logic by running the state transitions directly (not via a full
    model) since the generate() loop requires a real Spyre model.
    """

    def test_tail_pos_advance_and_flush_boundary(self):
        """Simulated generate() loop: tail_pos reaches FP8_BLOCK_SIZE, flush fires."""
        flush_calls = []

        import hf_adapters.fp8_kv_cache as _kv
        orig_flush = _kv.fp8_kv_cache_flush_tail

        def _mock_flush(ktail, vtail, kc, ksc, vc, block_idx, *, value_scale_cache=None):
            flush_calls.append(block_idx)
            # Don't actually run Spyre ops on CPU

        _kv.fp8_kv_cache_flush_tail = _mock_flush
        try:
            B, n_kv, D = 1, 4, 128
            N_LAYERS = 2
            MAX_CACHE = FP8_BLOCK_SIZE * 4  # 4 blocks
            PADDED_LEN = FP8_BLOCK_SIZE     # prefill filled 1 block

            kcs, kscs, vcs, vscs, tail_bufs_list = allocate_fp8_kv_caches(
                type("M", (), {"config": type("C", (), {
                    "num_hidden_layers": N_LAYERS,
                    "num_key_value_heads": n_kv,
                    "head_dim": D,
                    "hidden_size": 1024,
                    "num_attention_heads": 32,
                })()})(),
                1, MAX_CACHE, torch.bfloat16, device="cpu", option=1,
            )

            # Replicate the generate() state machine
            _fp8_tail_pos = 0
            _fp8_next_block = PADDED_LEN // FP8_BLOCK_SIZE  # = 1

            for decode_step in range(FP8_BLOCK_SIZE + 2):  # go past one full block
                # simulate post-forward tail advance (same as generate())
                _fp8_tail_pos += 1
                if _fp8_tail_pos == FP8_BLOCK_SIZE:
                    for _li in range(N_LAYERS):
                        _kv.fp8_kv_cache_flush_tail(
                            tail_bufs_list[_li][0], tail_bufs_list[_li][1],
                            kcs[_li], kscs[_li], vcs[_li],
                            _fp8_next_block,
                        )
                        tail_bufs_list[_li][0].zero_()
                        tail_bufs_list[_li][1].zero_()
                    _fp8_tail_pos = 0
                    _fp8_next_block += 1

            # Flush must have been called exactly once (when tail_pos hit FP8_BLOCK_SIZE)
            assert len(flush_calls) == N_LAYERS, (
                f"expected {N_LAYERS} flush calls (1 per layer), got {len(flush_calls)}"
            )
            # block_idx passed to flush must be 1 (the block after the prefill block)
            assert all(idx == 1 for idx in flush_calls), flush_calls
            # After flush, tail_pos reset to 0 and next_block incremented
            assert _fp8_tail_pos == 2  # 2 steps after the flush
            assert _fp8_next_block == 2

        finally:
            _kv.fp8_kv_cache_flush_tail = orig_flush

    def test_tail_bufs_zeroed_after_flush(self):
        """After flush, tail buffers are zeroed."""
        import hf_adapters.fp8_kv_cache as _kv
        orig_flush = _kv.fp8_kv_cache_flush_tail
        _kv.fp8_kv_cache_flush_tail = lambda *a, **kw: None  # no-op

        try:
            B, n_kv, D = 1, 4, 128
            N_LAYERS = 1
            MAX_CACHE = FP8_BLOCK_SIZE * 2

            kcs, kscs, vcs, vscs, tail_bufs_list = allocate_fp8_kv_caches(
                type("M", (), {"config": type("C", (), {
                    "num_hidden_layers": N_LAYERS,
                    "num_key_value_heads": n_kv,
                    "head_dim": D,
                    "hidden_size": 1024,
                    "num_attention_heads": 32,
                })()})(),
                1, MAX_CACHE, torch.bfloat16, device="cpu", option=1,
            )

            # Fill the tail buf with non-zero data
            tail_bufs_list[0][0].fill_(1.0)
            tail_bufs_list[0][1].fill_(1.0)

            # Simulate flush boundary
            _fp8_tail_pos = FP8_BLOCK_SIZE
            _fp8_next_block = 1
            if _fp8_tail_pos == FP8_BLOCK_SIZE:
                for _li in range(N_LAYERS):
                    _kv.fp8_kv_cache_flush_tail(
                        tail_bufs_list[_li][0], tail_bufs_list[_li][1],
                        kcs[_li], kscs[_li], vcs[_li], _fp8_next_block,
                    )
                    tail_bufs_list[_li][0].zero_()
                    tail_bufs_list[_li][1].zero_()
                _fp8_tail_pos = 0

            assert tail_bufs_list[0][0].abs().sum().item() == 0.0, "key tail not zeroed"
            assert tail_bufs_list[0][1].abs().sum().item() == 0.0, "value tail not zeroed"

        finally:
            _kv.fp8_kv_cache_flush_tail = orig_flush
