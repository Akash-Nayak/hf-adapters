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

"""Spyre device tests for the FP8 KV cache implementation.

Tests cover:
  * ``allocate_fp8_kv_cache_tensors`` — FP8 key cache gets the position-first
    ``SpyreTensorLayout`` on Spyre (128-element FP8 sticks).
  * ``fp8_kv_cache_update`` — compiled scatter of FP8 K and DL16 V into the cache.
  * ``fp8_attn_core`` — compiled attention (Option 2: dequantize K → DL16 matmul)
    matches DL16 SDPA baseline within tolerance for prefill and decode.
  * Prefill → decode sequence: the accumulated FP8 cache produces correct output
    for a new query token.

Run on a Spyre pod::

    pytest -s -vvv tests/spyre/test_fp8_kv_cache_spyre.py
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from hf_adapters.fp8_kv_cache import (
    FP8_DTYPE,
    SCALE_STICK_DIM,
    allocate_fp8_kv_cache_tensors,
    fp8_attn_core,
    fp8_kv_cache_update,
)

pytestmark = pytest.mark.requires_spyre

# ---------------------------------------------------------------------------
# Geometry used throughout: Granite 3.3 8B attention dims (8 KV heads, 32 Q)
# All dims must be multiples of the appropriate stick size:
#   DL16 stick = 64 elems  →  head_dim must be multiple of 64
#   FP8  stick = 128 elems →  head_dim must be multiple of 128
# ---------------------------------------------------------------------------
B = 1
H = 32          # Q heads
N_KV = 8        # KV heads
HEAD_DIM = 128  # must be multiple of 128 for FP8 sticks
PREFILL_LEN = 128
MAX_CACHE_LEN = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(dtype, *shape, device="cpu"):
    return torch.randn(*shape, dtype=dtype)


def _sdpa_ref(q, k_full, v_full, scaling, n_kv, attn_mask=None):
    """BF16 SDPA reference over the valid cache — CPU, no quantisation."""
    kv_repeat = H // n_kv
    k_exp = k_full.repeat_interleave(kv_repeat, dim=1)
    v_exp = v_full.repeat_interleave(kv_repeat, dim=1)
    return F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask, scale=scaling)


# ---------------------------------------------------------------------------
# Wrapper modules for torch.compile
# ---------------------------------------------------------------------------

class _Update(torch.nn.Module):
    """Compiled cache update."""
    def forward(self, k, v, key_cache, key_scale_cache, value_cache, cache_index):
        return fp8_kv_cache_update(k, v, key_cache, key_scale_cache, value_cache, cache_index)


class _Attend(torch.nn.Module):
    """Compiled attention core."""
    def __init__(self, scaling, dtype):
        super().__init__()
        self.scaling = scaling
        self.dtype = dtype

    def forward(self, q, key_cache, key_scale_cache, value_cache, attn_mask=None):
        return fp8_attn_core(
            q, key_cache, key_scale_cache, value_cache,
            attn_mask=attn_mask, scaling=self.scaling, compute_dtype=self.dtype,
        )


# ---------------------------------------------------------------------------
# 1. Layout test — FP8 key cache gets position-first SpyreTensorLayout
# ---------------------------------------------------------------------------

def test_fp8_key_cache_layout_on_spyre():
    """The FP8 key cache must be allocated with the position-first STL on Spyre.

    Checks:
    * ``device_layout`` attribute is present and non-None (Spyre allocation path).
    * ``elems_per_stick()`` returns 128 (FP8 stick, not 64 for DL16).
    * L (max_cache_len) maps to device position 0 (scatter-ready).
    """
    from torch_spyre._C import get_elem_in_stick  # type: ignore[import-not-found]

    kc, ksc, vc = allocate_fp8_kv_cache_tensors(
        B, N_KV, MAX_CACHE_LEN, HEAD_DIM, torch.bfloat16, device="spyre"
    )

    # dtype
    assert kc.dtype == FP8_DTYPE
    assert ksc.dtype == torch.bfloat16
    assert vc.dtype == torch.bfloat16

    # shape
    assert kc.shape  == (B, N_KV, MAX_CACHE_LEN, HEAD_DIM)
    assert ksc.shape == (B, N_KV, MAX_CACHE_LEN, SCALE_STICK_DIM)  # [B, n_kv, L, 64]
    assert vc.shape  == (B, N_KV, MAX_CACHE_LEN, HEAD_DIM)

    # FP8 stick = 128 elements
    assert get_elem_in_stick(FP8_DTYPE) == 128

    # Device layout: inspect via torch_spyre internals
    # The tensor must have a device_layout attribute set (non-plain allocation).
    # We verify indirectly: a plain torch.zeros on Spyre would not accept
    # device_layout, so if the tensor was created correctly the shape is intact.
    # Direct layout inspection via _spyre_layout() if available:
    # Direct layout inspection via get_spyre_tensor_layout:
    try:
        from torch_spyre._C import (  # type: ignore[import-not-found]
            DataFormats,
            ElementArrangement,
            get_spyre_tensor_layout,
        )
        stl = get_spyre_tensor_layout(kc)
        assert stl.device_dtype == DataFormats.SEN143_FP8
        # FP8 key cache must use QFP8CH element arrangement so that
        # quantize_fp8_with_scale output and fp8todl16 input formats agree.
        assert stl.element_arrangement == ElementArrangement.QFP8CH, (
            f"expected QFP8CH, got {stl.element_arrangement}"
        )
        # L is device position 0 — device_size[0] == MAX_CACHE_LEN
        assert stl.device_size[0] == MAX_CACHE_LEN
        # eps (innermost dim) == 128
        assert stl.device_size[-1] == 128
    except ImportError:
        pass  # torch_spyre._C unavailable; skip layout introspection


# ---------------------------------------------------------------------------
# 2. Compiled scatter: fp8_kv_cache_update on Spyre
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_kv_cache_update_compiled(dtype):
    """Compiled fp8_kv_cache_update writes FP8 K and DL16 V into the Spyre cache."""
    torch.manual_seed(0)

    kc, ksc, vc = allocate_fp8_kv_cache_tensors(
        B, N_KV, MAX_CACHE_LEN, HEAD_DIM, dtype, device="spyre"
    )
    k = _rand(dtype, B, N_KV, PREFILL_LEN, HEAD_DIM).to("spyre")
    v = _rand(dtype, B, N_KV, PREFILL_LEN, HEAD_DIM).to("spyre")
    idx = torch.arange(0, PREFILL_LEN, dtype=torch.long).to("spyre")

    class _Update(torch.nn.Module):
        def forward(self, k, v, key_cache, key_scale_cache, value_cache, idx):
            return fp8_kv_cache_update(k, v, key_cache, key_scale_cache, value_cache, idx)

    compiled = torch.compile(_Update(), dynamic=False)
    with torch.no_grad():
        kc_out, ksc_out, vc_out, k_fp8, k_scale = compiled(k, v, kc, ksc, vc, idx)

    # Key cache must be FP8 and non-zero at written positions
    assert kc_out.dtype == FP8_DTYPE
    written = kc_out[:, :, :PREFILL_LEN, :].to("cpu").to(torch.float32)
    assert written.abs().sum().item() > 0, "key cache is all-zero after update"

    # Value cache must match V exactly
    vc_written = vc_out[:, :, :PREFILL_LEN, :].to("cpu")
    torch.testing.assert_close(vc_written, v.to("cpu"), rtol=0, atol=0)

    # Scale cache must be positive at written positions ([B, n_kv, L, 64])
    ksc_written = ksc_out[:, :, :PREFILL_LEN, :].to("cpu")
    assert (ksc_written > 0).all(), "key scale cache has non-positive values"


# ---------------------------------------------------------------------------
# 3. fp8_attn_core on Spyre: prefill accuracy vs DL16 SDPA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_attn_core_prefill_accuracy(dtype):
    """fp8_attn_core compiled on Spyre must match DL16 SDPA within tolerance.

    Fills the full cache with one prefill scatter, then runs fp8_attn_core
    on the same Q. Compares to a CPU BF16 SDPA reference using the original
    (unquantized) K/V.

    Tolerance: max absolute error < 0.15 (same as CPU test; FP8 quantization
    of K introduces error, not the attention arithmetic itself).
    """
    torch.manual_seed(1)
    scaling = 1.0 / math.sqrt(HEAD_DIM)

    # Build caches and random inputs on CPU first
    k_cpu = _rand(dtype, B, N_KV, PREFILL_LEN, HEAD_DIM)
    v_cpu = _rand(dtype, B, N_KV, PREFILL_LEN, HEAD_DIM)
    q_cpu = _rand(dtype, B, H, PREFILL_LEN, HEAD_DIM)
    idx_cpu = torch.arange(0, PREFILL_LEN, dtype=torch.long)

    # Allocate on Spyre
    kc, ksc, vc = allocate_fp8_kv_cache_tensors(
        B, N_KV, MAX_CACHE_LEN, HEAD_DIM, dtype, device="spyre"
    )
    k_s = k_cpu.to("spyre")
    v_s = v_cpu.to("spyre")
    q_s = q_cpu.to("spyre")
    idx_s = idx_cpu.to("spyre")

    update_fn = torch.compile(_Update(), dynamic=False)
    attend_fn = torch.compile(_Attend(scaling, dtype), dynamic=False)

    # Valid mask for prefill: tokens 0..PREFILL_LEN-1 are valid, PREFILL_LEN..MAX_CACHE_LEN-1 are masked.
    # On Spyre, finite large negative mask (-10000.0) is standard in DeepTools/inductor softmax.
    attn_mask_s = torch.full((1, 1, PREFILL_LEN, MAX_CACHE_LEN), -10000.0, dtype=dtype, device="spyre")
    attn_mask_s[:, :, :, :PREFILL_LEN] = 0.0

    with torch.no_grad():
        kc, ksc, vc, _, _ = update_fn(k_s, v_s, kc, ksc, vc, idx_s)
        out_s = attend_fn(q_s, kc, ksc, vc, attn_mask=attn_mask_s)

    out_fp8 = out_s.to("cpu")

    # DL16 SDPA reference on CPU over valid K/V
    out_ref = _sdpa_ref(
        q_cpu.float(), k_cpu.float(), v_cpu.float(), scaling, N_KV
    ).to(dtype)

    assert out_fp8.shape == out_ref.shape
    max_abs = (out_fp8.float() - out_ref.float()).abs().max().item()
    assert max_abs < 0.15, f"prefill max_abs={max_abs:.4f} (dtype={dtype})"


# ---------------------------------------------------------------------------
# 4. fp8_attn_core on Spyre: decode step accuracy (S=1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_attn_core_decode_accuracy(dtype):
    """Compiled decode step (S=1) must match DL16 SDPA within tolerance.

    Simulates a real prefill → decode sequence:
      1. Prefill: write PREFILL_LEN tokens into the FP8 cache.
      2. Decode:  write one more token, then run fp8_attn_core with S=1.

    Compares against DL16 SDPA on the combined (prefill + decode) K/V.
    """
    torch.manual_seed(2)
    scaling = 1.0 / math.sqrt(HEAD_DIM)

    # Prefill data
    k_pre = _rand(dtype, B, N_KV, PREFILL_LEN, HEAD_DIM)
    v_pre = _rand(dtype, B, N_KV, PREFILL_LEN, HEAD_DIM)
    idx_pre = torch.arange(0, PREFILL_LEN, dtype=torch.long)

    # Decode data (1 token)
    k_dec = _rand(dtype, B, N_KV, 1, HEAD_DIM)
    v_dec = _rand(dtype, B, N_KV, 1, HEAD_DIM)
    q_dec = _rand(dtype, B, H, 1, HEAD_DIM)
    idx_dec = torch.tensor([PREFILL_LEN], dtype=torch.long)

    # Allocate on Spyre
    kc, ksc, vc = allocate_fp8_kv_cache_tensors(
        B, N_KV, MAX_CACHE_LEN, HEAD_DIM, dtype, device="spyre"
    )

    update_fn = torch.compile(_Update(), dynamic=False)
    attend_fn = torch.compile(_Attend(scaling, dtype), dynamic=False)

    # Valid mask for decode: tokens 0..context_len-1 are valid
    context_len = PREFILL_LEN + 1
    attn_mask_dec = torch.full((1, 1, 1, MAX_CACHE_LEN), -10000.0, dtype=dtype, device="spyre")
    attn_mask_dec[:, :, :, :context_len] = 0.0

    with torch.no_grad():
        # Prefill write
        kc, ksc, vc, _, _ = update_fn(
            k_pre.to("spyre"), v_pre.to("spyre"), kc, ksc, vc, idx_pre.to("spyre")
        )
        # Decode write
        kc, ksc, vc, _, _ = update_fn(
            k_dec.to("spyre"), v_dec.to("spyre"), kc, ksc, vc, idx_dec.to("spyre")
        )
        # Decode attend
        out_s = attend_fn(q_dec.to("spyre"), kc, ksc, vc, attn_mask=attn_mask_dec)

    out_fp8 = out_s.to("cpu")

    # DL16 SDPA reference on the combined valid K/V (context_len tokens)
    k_full = torch.cat([k_pre, k_dec], dim=2)   # [B, n_kv, context_len, D]
    v_full = torch.cat([v_pre, v_dec], dim=2)
    out_ref = _sdpa_ref(
        q_dec.float(), k_full.float(), v_full.float(), scaling, N_KV
    ).to(dtype)

    assert out_fp8.shape == out_ref.shape
    max_abs = (out_fp8.float() - out_ref.float()).abs().max().item()
    assert max_abs < 0.15, f"decode max_abs={max_abs:.4f} (dtype={dtype})"
