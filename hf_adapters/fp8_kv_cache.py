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

"""FP8 KV cache for Spyre: K stored as E4M3, QK^T run in DL16 (Option 2).

Storage strategy:

  1. K projection output is DL16.  Compute a per-token scale for K and store
     K as FP8 E4M3 in the cache; also store the per-token K scale alongside
     the quantized K tensor.  V is stored unchanged in compute dtype.

  2. At attention time (Option 2 — active):
     Dequantize K from FP8 to DL16 (``k_fp8 * k_scale``), then run
     QK^T via ``torch.matmul`` (DL16 × DL16 → DL16).

  Option 1 (FP8 QK^T via ``spyre.scaled_mm``) is preserved in code but
  commented out — it is blocked by ``ReStickifyOpHBM on SEN143_FP8`` in the
  compiler (``spyre_kernel.py``).  Switching to Option 1 is a single swap in
  ``fp8_attn_core`` once that blocker is resolved.

V multiply and softmax always run in DL16 — V is NOT quantized in this
implementation.  Only K is stored FP8; V remains in the compute dtype.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
SCALE_EPS = 1e-4  # prevents zero scale for all-zero tokens


# ---------------------------------------------------------------------------
# Per-token FP8 quantization helpers (CPU + Spyre)
# ---------------------------------------------------------------------------


def _quantize_fp8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to FP8 E4M3 with a per-token scale.

    Args:
        x: ``[B, H, S, D]`` or ``[B*H, S, D]`` in the compute dtype.

    Returns:
        ``(xq, scale)`` where ``xq`` is FP8 E4M3, same shape as ``x``,
        and ``scale`` is ``[..., S]`` in the compute dtype (no trailing 1-dim).
        Callers that need ``[..., S, 1]`` for broadcasting must unsqueeze(-1).

    Note on the Spyre path: ``keepdim=True`` is passed to ``quantize_fp8_with_scale``
    but the trailing 1 is squeezed before returning.  Keeping a trailing size-1 dim
    on tensors that flow into ``index_copy_``/``index_put_`` triggers the Spyre
    inductor ``ranges_from_index_vars`` crash (``d1=absent`` / ``op9`` SchedulerNode
    gets ``index_vars=[[c0],[]]``).  Squeezing here avoids the scatter-write bug;
    the ``unsqueeze(-1)`` for broadcasting is deferred to read time in
    ``fp8_attn_core`` where it does not cause a scatter crash.
    """
    if x.device.type == "spyre":
        # Use the same abs().amax() idiom as FP8Linear to avoid the
        # quantscalepertokenfp8 DDL bug (dbo-opt drops last-dim=1 sticks).
        # keepdim=True is required for quantize_fp8_with_scale; squeeze after.
        scale_4d = (
            (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX))
            .clamp(min=SCALE_EPS)
            .clone()
        )
        xq = torch.ops.spyre.quantize_fp8_with_scale(x, scale_4d)
        scale = scale_4d.squeeze(-1)  # [..., S] — no trailing 1 for scatter safety
    else:
        # CPU reference path
        scale_4d = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(
            min=SCALE_EPS
        )
        xq = (x * scale_4d.reciprocal()).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        scale = scale_4d.squeeze(-1)  # [..., S] — no trailing 1
    return xq, scale


# ---------------------------------------------------------------------------
# FP8 KV cache update
# ---------------------------------------------------------------------------


def fp8_kv_cache_update(
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write K (as FP8) and V (as compute dtype) into their caches.

    K is quantized to FP8 per-token before the scatter, and the per-token
    scale is written into ``key_scale_cache``.  V is stored unquantized in
    the compute dtype.

    Args:
        k: ``[B, n_kv, n, head_dim]`` — K projection for ``n`` positions.
        v: ``[B, n_kv, n, head_dim]`` — V projection for ``n`` positions.
        key_cache: ``[B, n_kv, L, head_dim]`` FP8 E4M3.
        key_scale_cache: ``[B, n_kv, L, head_dim]`` compute dtype — per-token K
            scale broadcast across head_dim.  Storing the full head_dim avoids
            any size-1 trailing dimension which triggers the Spyre inductor
            ``ranges_from_index_vars`` scheduler crash.  The extra memory is
            negligible (head_dim BF16 values per token).
        value_cache: ``[B, n_kv, L, head_dim]`` compute dtype.
        cache_index: ``[n]`` int64 destination positions.

    Returns:
        Updated ``(key_cache, key_scale_cache, value_cache)`` plus the
        quantized ``k_fp8`` that callers need for QK^T (avoids re-quantizing).
    """
    k_fp8, k_scale = _quantize_fp8_per_token(k)  # k_fp8: [B,n_kv,n,D], k_scale: [B,n_kv,n]

    if key_cache.device.type == "cpu":
        # CPU PyTorch does not implement index_copy_ for FP8 dtypes.
        # Work around by reinterpreting as uint8, scattering bytes, then viewing
        # back as FP8.  The bit pattern is preserved exactly.
        key_cache.view(torch.uint8).index_copy_(2, cache_index, k_fp8.view(torch.uint8))
    else:
        key_cache.index_copy_(2, cache_index, k_fp8)
    # Expand scale to [B, n_kv, n, head_dim] so the scatter writes a full
    # DL16 row — same shape as key_cache / value_cache.  This avoids any
    # size-1 last dimension that would crash the Spyre inductor scheduler.
    head_dim = k.shape[-1]
    k_scale_exp = k_scale.unsqueeze(-1).expand(*k_scale.shape, head_dim)
    key_scale_cache.index_copy_(2, cache_index, k_scale_exp)
    value_cache.index_copy_(2, cache_index, v)

    return key_cache, key_scale_cache, value_cache, k_fp8, k_scale


# ---------------------------------------------------------------------------
# FP8 QK^T attention core
# ---------------------------------------------------------------------------


def fp8_attn_core(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scaling: float,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute attention with FP8 K cache; QK^T in DL16 (Option 2).

    Dequantizes K from FP8 to DL16 at read time (k_fp8 * k_scale), then runs
    standard DL16 QK^T matmul, softmax, and AV multiply.

    Args:
        q: ``[B, H, S, D]`` query in compute dtype (DL16).
        key_cache: ``[B, n_kv, L, D]`` FP8 E4M3 key cache.
        key_scale_cache: ``[B, n_kv, L, D]`` per-token K scale, broadcast across
            head_dim — same shape as key_cache.
        value_cache: ``[B, n_kv, L, D]`` compute dtype value cache.
        attn_mask: additive causal mask or None.
        scaling: attention scale (``1 / sqrt(head_dim)``).
        compute_dtype: BF16 or FP16.

    Returns:
        ``[B, H, S, D_v]`` attention output in compute dtype.
    """
    B, H, S, D = q.shape
    _, n_kv, L, _ = key_cache.shape

    # GQA: KV heads are grouped into Q heads.
    kv_repeat = H // n_kv

    # ---------------------------------------------------------------------------
    # Option 2: Dequantize K from FP8 to DL16 and run DL16 matmul.
    # ---------------------------------------------------------------------------
    # On Spyre, FP8 -> FP16 is supported natively by the hardware (fp8todl16).
    # Casting to torch.float16 first ensures on-device fp8todl16 executes without
    # falling back to CPU (which would happen if converting FP8 directly to BF16).
    # Then cast the dequantized result to compute_dtype (BF16 or FP16).
    k_dl16 = (key_cache.to(torch.float16) * key_scale_cache.to(torch.float16)).to(
        compute_dtype
    )
    k_expanded = k_dl16.repeat_interleave(kv_repeat, dim=1)  # [B, H, L, D]

    logits = torch.matmul(q, k_expanded.transpose(-2, -1)) * scaling

    # Apply causal mask (additive, -inf for masked positions)
    if attn_mask is not None:
        logits = logits + attn_mask

    # Softmax in compute dtype (DL16) — no expansion needed
    attn_weights = torch.softmax(logits, dim=-1, dtype=compute_dtype)

    # AV multiply: DL16 matmul.  V is stored unquantized.
    # value_cache: [B, n_kv, L, D_v]
    v_expanded = value_cache.repeat_interleave(kv_repeat, dim=1)  # [B, H, L, D_v]
    attn_out = torch.matmul(attn_weights, v_expanded)             # [B, H, S, D_v]
    return attn_out


# ---------------------------------------------------------------------------
# FP8 KV cache allocation helper
# ---------------------------------------------------------------------------


def allocate_fp8_kv_cache_tensors(
    batch_size: int,
    num_kv_heads: int,
    max_cache_len: int,
    head_dim: int,
    compute_dtype: torch.dtype,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate (key_cache_fp8, key_scale_cache, value_cache) triplet.

    ``key_cache`` is FP8 E4M3 allocated with a scatter-ready
    ``SpyreTensorLayout`` on Spyre using ``ElementArrangement.QFP8CH``:
    L (cache-position) sits at device position 0 (required for
    ``index_copy_`` indirect scatter) and the stick dimension covers
    ``head_dim`` with 128-element FP8 sticks.

    **Why QFP8CH?** ``quantize_fp8_with_scale`` produces FP8 tensors in
    ``QFP8CH`` element arrangement (within-stick element order differs from
    ``STANDARD``), and ``fp8todl16`` (inside ``dequantize_fp8_with_scale``)
    reads FP8 in ``QFP8CH`` arrangement.  Writing QFP8CH-arranged FP8 into a
    ``STANDARD``-arranged cache silently corrupts data (same ``stickSize=128``
    but wrong within-stick order), producing ``max_abs ≈ 0.43`` error on device.
    Allocating with ``QFP8CH`` ensures the scatter write and the fp8todl16 read
    use the same element ordering.

    ``key_scale_cache`` is allocated as a ``[B, n_kv, L, head_dim]`` DL16 tensor
    via ``allocate_kv_cache_tensor`` (scatter-ready position-first layout on Spyre).
    The scale is stored broadcast-expanded across ``head_dim`` so the scatter writes
    a full DL16 row — exactly the same shape as ``value_cache``.  This sidesteps the
    Spyre inductor ``ranges_from_index_vars`` crash that occurs for any tensor with
    a size-1 or absent trailing dimension in a compiled scatter.

    ``value_cache`` uses the same scatter-ready DL16 allocation.
    """
    if device is None:
        from hf_adapters.hf_common import DEVICE
        device = DEVICE

    from hf_adapters.hf_common import _cache_position_first_stl

    shape_k = (batch_size, num_kv_heads, max_cache_len, head_dim)
    shape_v = (batch_size, num_kv_heads, max_cache_len, head_dim)

    # FP8 key cache: scatter-ready position-first layout on Spyre, with
    # ElementArrangement.QFP8CH.
    #
    # ElementArrangement.QFP8CH is required because:
    #   1. ``quantize_fp8_with_scale`` (qfp8ch op) produces FP8 tensors in
    #      QFP8CH element arrangement (within-stick element order differs from
    #      STANDARD FP8).
    #   2. ``fp8todl16`` (used by ``dequantize_fp8_with_scale``) reads FP8
    #      tensors in QFP8CH element arrangement.
    # Writing QFP8CH-arranged FP8 into a STANDARD-arranged cache corrupts
    # the data silently (same stickSize=128, but wrong within-stick order).
    on_spyre = torch.device(device).type == "spyre"
    stl = None
    if on_spyre:
        try:
            torch.empty(1, device="spyre")
            from torch_spyre._C import ElementArrangement  # type: ignore[import-not-found]
            stl = _cache_position_first_stl(
                batch_size, num_kv_heads, max_cache_len, head_dim, FP8_DTYPE,
                element_arrangement=ElementArrangement.QFP8CH,
            )
        except Exception:  # noqa: BLE001
            stl = None
    if stl is None:
        key_cache = torch.zeros(shape_k, dtype=FP8_DTYPE, device=device)
    else:
        key_cache = torch.empty(  # type: ignore[call-overload]
            shape_k,
            device=torch.device(device),
            device_layout=stl,
            dtype=FP8_DTYPE,
        )
        # FillDMA does not support SEN143_FP8 — zero via uint8 reinterpret
        # (bit pattern of FP8 zero == uint8 zero).
        key_cache.view(torch.uint8).zero_()

    # key_scale_cache: scatter-ready DL16 allocation, [B, n_kv, L, head_dim].
    # Using the same allocate_kv_cache_tensor path as value_cache gives the
    # position-first SpyreTensorLayout on Spyre (scatter-safe) with a full
    # head_dim DL16 row — no size-1 dims, no scheduler crash.
    from hf_adapters.hf_common import allocate_kv_cache_tensor
    key_scale_cache = allocate_kv_cache_tensor(
        batch_size, num_kv_heads, max_cache_len, head_dim, compute_dtype, device
    )
    value_cache = allocate_kv_cache_tensor(
        batch_size, num_kv_heads, max_cache_len, head_dim, compute_dtype, device
    )
    return key_cache, key_scale_cache, value_cache
