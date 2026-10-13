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

"""FP8 KV cache for Spyre — two option variants selectable at runtime.

Both options store K as FP8 E4M3.  Option 1 also stores V as FP8 QFP8CH
(dequantized at attention time) giving a full ~2× KV memory saving.
Option 2 stores V unquantized in the compute dtype.
They differ in the QK^T matmul path:

  **Option 1 — direct-FP8-matmul** (target; block-scatter layout)
    K is written in **paged block layout** ``[NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE]``
    (transposed, D-inner) using ``index_copy_(0, block_idx, ...)`` — a full-stick
    scatter.  Consumed by ``spyre.scaled_bmm`` at attention time.  A DL16 tail
    buffer ``[B, n_kv, FP8_BLOCK_SIZE, D]`` accumulates decode tokens (n=1) that do
    not yet fill a complete block; it is attended via standard ``torch.matmul``.
    Q is quantized to FP8 per-token before ``scaled_bmm``.
    V is stored as QFP8CH ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D]`` with a
    parallel scale cache ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, 64]`` (trailing dim
    padded to 64 for stick-aligned scatter).  Dequantized at attention time.

  **Option 2 — dequantize-K** (fallback; works on device today)
    K is written as QFP8CH (standard, L-inner) at fill time.  At attention
    time K is dequantized back to DL16 and the standard ``torch.matmul`` path
    is used.  V is stored unquantized (DL16).

Select via the ``option`` argument to :func:`fp8_kv_cache_update` and
:func:`fp8_attn_core`, or pass ``fp8_kv_opt`` to
``prepare_for_spyre(model, fp8_kv_cache=True, fp8_kv_opt=1)``.

---

### Cache tensor shapes

  Option 1 (Spyre):
    key_cache         ``[NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE]`` float8_e4m3fn  QFP8WT
    key_scale_cache   ``[NUM_BLOCKS, 1, 1, 1]``                  compute dtype  per-block scalar
    value_cache       ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D]``  float8_e4m3fn  QFP8CH
    value_scale_cache ``[NUM_BLOCKS, 1, 1, 1]``                  compute dtype  per-block scalar
    key_tail_buf      ``[B, n_kv, FP8_BLOCK_SIZE, D]``           compute dtype  decode staging
    value_tail_buf    ``[B, n_kv, FP8_BLOCK_SIZE, D]``           compute dtype  decode staging (DL16)

    ``NUM_BLOCKS = max_cache_len // FP8_BLOCK_SIZE``

    Both scale caches use the same ``[NUM_BLOCKS, 1, 1, 1]`` shape — one scalar per
    committed block (amax over all tokens in the block), matching the reference
    ``ScaledPagedAttnStore`` design.  This avoids stick-alignment issues and the
    per-token 64-wide padding required by a per-token scale scatter.

    ``allocate_fp8_kv_cache_tensors(option=1)`` returns a **5-tuple**
    ``(key_cache, key_scale_cache, value_cache, value_scale_cache,
    (key_tail_buf, value_tail_buf))``.

  Option 2 (Spyre + CPU):
    key_cache        ``[B, n_kv, L, D]`` float8_e4m3fn
    key_scale_cache  ``[B, n_kv, L, 1]`` compute dtype  — per-token scale (trailing dim = 1)
    value_cache      ``[B, n_kv, L, D]`` compute dtype

    ``allocate_fp8_kv_cache_tensors(option=2)`` returns a **3-tuple**
    ``(key_cache, key_scale_cache, value_cache)``.

  On CPU (both options fall back to the Option 2 CPU path):
    key_cache        ``[B, n_kv, L, D]`` float8_e4m3fn
    key_scale_cache  ``[B, n_kv, L, 1]`` compute dtype
    value_cache      ``[B, n_kv, L, D]`` compute dtype

---

### Fill path — ``fp8_kv_cache_update``

  Option 1 (Spyre) — **prefill** (``n % FP8_BLOCK_SIZE == 0``):
    ``cache_index`` must be pre-computed **block** indices (int32) outside ``torch.compile``.
    1. Cast K to FP16; permute ``[B, n_kv, n, D]`` → ``[n_kv, D, n]``.
    2. ``quantscalepertokenfp8`` → scalar amax → ``quantize_weight_fp8_with_scale``.
    3. Reshape K FP8 to ``[num_b, n_kv, D, FP8_BLOCK_SIZE]``.
    4. ``key_cache.index_copy_(0, block_idx, ...)`` — full-stick scatter.
    5. ``key_scale_cache.index_copy_(0, block_idx, sk.reshape(num_b, 1, 1, 1))``.
    6. V ``[B, n_kv, n, D]`` → unsqueeze → ``[1, n, n_kv, D]`` cast to FP16.
    7. ``abs().amax() / FP8_MAX`` → per-block scalar ``sv``; clamp to ``SCALE_EPS``.
    8. ``quantize_fp8_with_scale(v_in, sv)`` → V FP8 ``[1, n, n_kv, D]``.
    9. Reshape V FP8 to ``[num_b, FP8_BLOCK_SIZE, n_kv, D]`` → scatter dim 0.
   10. ``value_scale_cache.index_copy_(0, block_idx, sv.reshape(num_b, 1, 1, 1))``.

  Option 1 (Spyre) — **decode** (``n < FP8_BLOCK_SIZE``, typically ``n=1``):
    Writes K and V (DL16) into ``key_tail_buf`` / ``value_tail_buf`` at the tail
    position computed by the caller.  **Called eagerly (outside torch.compile)**;
    the tail buffers are fixed-shape DL16 tensors passed as-is to ``fp8_attn_core``.
    When the tail is full (``tail_pos == FP8_BLOCK_SIZE``), the caller flushes it
    by calling ``fp8_kv_cache_flush_tail``.

  Option 2 (Spyre):
    1. Compute scale: ``quantscalepertokenfp8`` → ``[B, n_kv, n, 1]`` fp16.
    2. Encode to QFP8CH via ``quantize_fp8_with_scale``.
    3. Scatter into ``key_cache[:, :, cache_index, :]`` (dim 2 = position dim).
    4. Scatter scale into ``key_scale_cache[:, :, cache_index, :]`` (dim 2).

  Option 2 (CPU):
    Same scale/quantize logic but using ``_quantize_fp8_per_token`` (CPU only).
    FP8 ``index_copy_`` not supported on CPU — reinterprets as uint8.

  V is always scattered unchanged (DL16) into ``value_cache`` for Option 2 / CPU.

---

### Decode path — ``fp8_attn_core``

  Option 1 (Spyre):
    1. Quantize Q per-token to QFP8CH; reshape to ``[n_kv, kv_repeat, S, D]``.
    2. For each committed block: ``spyre.scaled_bmm(q_fp8, k_block)`` → scores;
       apply per-token scale compensation.
    3. Concatenate committed block scores; add DL16 tail scores (``torch.matmul``).
    4. Apply mask, softmax.
    5. AV (committed): gather V FP8, dequantize with per-block scalar scale broadcast
       over ``[n_kv, 1, total_L, D]``; cat DL16 tail V; matmul.
    6. Reshape output back to ``[B, H, S, D]``.

  Option 2 (Spyre + CPU):
    1. Dequantize K from ``[B, n_kv, L, D]`` to DL16.
    2. Expand from n_kv → H via ``repeat_interleave``.
    3. ``torch.matmul(Q, K^T)`` → standard DL16 QK^T.
    4. Softmax + AV multiply (V is DL16).

---

### Why block scatter (Option 1) avoids partial-stick writes

  Each ``index_copy_(0, block_idx, src)`` call writes ``src`` of shape
  ``[num_b, n_kv, D, FP8_BLOCK_SIZE]``.  The innermost dimension
  ``FP8_BLOCK_SIZE = 128`` is exactly one FP8 stick (128 elements).
  Scattering on dim 0 (the block dimension) copies whole contiguous sticks.

  The previous ``[B, n_kv, D, L]`` + ``index_copy_(3, token_idx)`` wrote
  1 token = 1 position inside a 128-element stick — a partial-stick write
  that the hardware does not support.
"""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0          # torch.finfo(torch.float8_e4m3fn).max
SCALE_EPS = 1e-4         # prevents zero scale for all-zero tokens
FP8_BLOCK_SIZE = 128     # FP8 stick size in elements; must equal head_dim for Granite


# ---------------------------------------------------------------------------
# Per-token FP8 quantization helper (CPU path only)
# ---------------------------------------------------------------------------


def _quantize_fp8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to FP8 E4M3 with a per-token scale (CPU only).

    On Spyre, use ``quantize_fp8_with_scale`` / ``quantize_weight_fp8_with_scale``
    directly so the compiler can fuse and lower them to hardware ops.

    Args:
        x: any shape in compute dtype.

    Returns:
        ``(xq, scale)`` where ``xq`` is FP8 E4M3 and ``scale`` is ``[..., 1]``
        (one scalar per token, keepdim=True on the last dim).
    """
    scale = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(min=SCALE_EPS)
    xq = (x * scale.reciprocal()).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale


# ---------------------------------------------------------------------------
# FP8 KV cache update — fill path
# ---------------------------------------------------------------------------


def fp8_kv_cache_update(
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_index: torch.Tensor,
    *,
    option: int = 1,
    value_scale_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write K (FP8) and V (FP8 for Option 1) into the **committed block caches**.

    For Option 1 this handles the **prefill path only** (``n`` is a multiple of
    ``FP8_BLOCK_SIZE``).  Decode writes (``n=1``) go through
    :func:`fp8_kv_cache_update_tail` instead.

    Args:
        k: ``[B, n_kv, n, D]`` K projection for ``n`` new positions.
        v: ``[B, n_kv, n, D]`` V projection for ``n`` new positions.
        key_cache:
            Option 1 Spyre: ``[NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE]`` FP8 QFP8WT.
            Option 2 / CPU: ``[B, n_kv, L, D]`` FP8 QFP8CH.
        key_scale_cache:
            Option 1: ``[NUM_BLOCKS, 1, 1, 1]`` compute dtype per-block scalar.
            Option 2: ``[B, n_kv, L, 1]`` compute dtype per-token scale.
        value_cache:
            Option 1: ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D]`` FP8 QFP8CH.
            Option 2: ``[B, n_kv, L, D]`` compute dtype.
        cache_index:
            **Option 1**: ``[num_b]`` int32 **block** indices
            (e.g. ``torch.arange(start_block, start_block + num_b, dtype=torch.int32)``).
            Must be pre-computed **outside** ``torch.compile`` — deriving block
            indices inside the graph via ``//`` emits an unsupported ``floordiv`` op.
            **Option 2 / CPU**: ``[n]`` int64 **token** positions
            (``arange(start, start+n)``).
        option: 1 = block-scatter QFP8WT fill + ``scaled_bmm`` at attend.
                2 = QFP8CH fill + dequantize-K at attend.
        value_scale_cache:
            Option 1: ``[NUM_BLOCKS, 1, 1, 1]`` compute dtype per-block scalar.
            One scalar per block (amax over all V tokens in the block).
            Matches ``ScaledPagedAttnStore`` reference design.

    Returns:
        Updated ``(key_cache, key_scale_cache, value_cache)``.
    """
    on_spyre = k.device.type == "spyre"
    B, n_kv, n, D = k.shape

    if option == 1 and on_spyre:
        # ── Option 1 prefill fill: block scatter on dim 0 ─────────────────
        # n must be a whole number of FP8 blocks; decode tokens use
        # fp8_kv_cache_update_tail instead.
        assert n % FP8_BLOCK_SIZE == 0, (
            f"Option 1 fp8_kv_cache_update requires n to be a multiple of "
            f"FP8_BLOCK_SIZE={FP8_BLOCK_SIZE}, got n={n}. "
            "Decode tokens (n=1) should use fp8_kv_cache_update_tail."
        )
        num_b = n // FP8_BLOCK_SIZE

        # cache_index is already block indices for Option 1 — pre-computed by
        # the caller outside torch.compile to avoid an unsupported floordiv op.
        block_idx = cache_index  # [num_b] int32

        # K: [B, n_kv, n, D] → drop B (B=1 always for Granite) → [n_kv, D, n] FP16.
        # .contiguous() ensures the N-inner stride layout required by
        # quantize_weight_fp8_with_scale.
        k_fp16 = k[0].to(torch.float16)                     # [n_kv, n, D]
        k_perm = k_fp16.permute(0, 2, 1).contiguous()       # [n_kv, D, n]

        # Per-tensor conservative scale: max of per-token scales across the block.
        # quantize_weight_fp8_with_scale requires a scalar or broadcastable scale.
        sk = torch.ops.spyre.quantscalepertokenfp8(k_perm).amax()  # scalar fp16
        k_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(k_perm, sk)

        # Reshape into blocks: [n_kv, D, n] → [num_b, n_kv, D, FP8_BLOCK_SIZE]
        k_blocked = (
            k_fp8.reshape(n_kv, D, num_b, FP8_BLOCK_SIZE)
            .permute(2, 0, 1, 3)
            .contiguous()
        )  # [num_b, n_kv, D, FP8_BLOCK_SIZE]

        # Full-stick scatter on dim 0.
        key_cache.index_copy_(0, block_idx, k_blocked)
        key_scale_cache.index_copy_(
            0, block_idx, sk.reshape(1, 1, 1, 1).expand(num_b, 1, 1, 1).contiguous()
        )

        # V: quantize to QFP8CH and scatter.
        # v [B, n_kv, n, D] → [1, n, n_kv, D] to match [NUM_BLOCKS, FP8_BS, n_kv, D] layout.
        v_in = v[0].permute(1, 0, 2).unsqueeze(0).to(torch.float16)  # [1, n, n_kv, D]
        # Per-block scalar scale: amax over all tokens + heads in this set of blocks.
        # Matches the ScaledPagedAttnStore reference design (one scalar per block).
        # Using a scalar avoids all stick-alignment issues with per-token scale scatter.
        sv_scalar = (v_in.float().abs().amax() / FP8_MAX).to(
            torch.float16
        ).clamp(min=SCALE_EPS)                                        # scalar FP16
        v_fp8 = torch.ops.spyre.quantize_fp8_with_scale(v_in, sv_scalar)  # [1, n, n_kv, D] QFP8CH
        # Reshape to [num_b, FP8_BLOCK_SIZE, n_kv, D] for dim-0 block scatter.
        v_blocked = v_fp8.reshape(num_b, FP8_BLOCK_SIZE, n_kv, D)
        value_cache.index_copy_(0, block_idx, v_blocked)
        if value_scale_cache is not None:
            # value_scale_cache [NUM_BLOCKS, 1, 1, 1] — same shape as key_scale_cache.
            # Scatter sv_scalar once per block: expand scalar → [num_b, 1, 1, 1].
            sv_blocks = sv_scalar.to(value_scale_cache.dtype).reshape(1, 1, 1, 1).expand(
                num_b, 1, 1, 1
            ).contiguous()
            value_scale_cache.index_copy_(0, block_idx, sv_blocks)
        return key_cache, key_scale_cache, value_cache

    elif on_spyre:
        # ── Option 2 fill on Spyre: write K as QFP8CH ─────────────────────
        k_scale = (
            k.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)
        ).clamp(min=SCALE_EPS)  # [B, n_kv, n, 1]
        k_fp8 = torch.ops.spyre.quantize_fp8_with_scale(k, k_scale)

        key_cache.index_copy_(2, cache_index, k_fp8)
        key_scale_cache.index_copy_(2, cache_index, k_scale)
        value_cache.index_copy_(2, cache_index, v)

    else:
        # ── Option 2 fill on CPU (or option==1 CPU fallback) ──────────────
        k_fp8, k_scale = _quantize_fp8_per_token(k)

        # CPU: FP8 index_copy_ not implemented — reinterpret as uint8.
        key_cache.view(torch.uint8).index_copy_(
            2, cache_index, k_fp8.view(torch.uint8)
        )
        key_scale_cache.index_copy_(2, cache_index, k_scale)
        value_cache.index_copy_(2, cache_index, v)

    return key_cache, key_scale_cache, value_cache


# ---------------------------------------------------------------------------
# Option 1 decode tail helpers (called eagerly, outside torch.compile)
# ---------------------------------------------------------------------------


def fp8_kv_cache_update_tail(
    k: torch.Tensor,
    v: torch.Tensor,
    key_tail_buf: torch.Tensor,
    value_tail_buf: torch.Tensor,
    tail_pos: int,
) -> None:
    """Write one decode token into the DL16 tail buffers (Option 1, in-place).

    Called **eagerly** (outside ``torch.compile``) once per decode step.  The
    tail buffers are fixed-shape ``[B, n_kv, FP8_BLOCK_SIZE, D]`` tensors; the
    caller passes ``tail_pos = current_cache_len % FP8_BLOCK_SIZE``.

    When the buffer becomes full (``tail_pos + 1 == FP8_BLOCK_SIZE``), the caller
    should follow with :func:`fp8_kv_cache_flush_tail`.

    Args:
        k: ``[B, n_kv, 1, D]`` — single decode K token.
        v: ``[B, n_kv, 1, D]`` — single decode V token.
        key_tail_buf: ``[B, n_kv, FP8_BLOCK_SIZE, D]`` DL16 staging buffer (mutated).
        value_tail_buf: ``[B, n_kv, FP8_BLOCK_SIZE, D]`` DL16 staging buffer (mutated).
        tail_pos: slot index within the tail buffer (0 … FP8_BLOCK_SIZE-1).
    """
    key_tail_buf[:, :, tail_pos : tail_pos + 1, :] = k
    value_tail_buf[:, :, tail_pos : tail_pos + 1, :] = v


def fp8_kv_cache_flush_tail(
    key_tail_buf: torch.Tensor,
    value_tail_buf: torch.Tensor,
    key_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_idx: int,
    *,
    value_scale_cache: torch.Tensor | None = None,
) -> None:
    """Flush a full tail buffer into the committed block caches (Option 1, in-place).

    Called **eagerly** when ``tail_pos + 1 == FP8_BLOCK_SIZE``.  After flushing,
    the caller should zero ``key_tail_buf`` and ``value_tail_buf`` and reset
    ``tail_pos = 0``.

    Args:
        key_tail_buf: ``[B, n_kv, FP8_BLOCK_SIZE, D]`` DL16 — full tail block.
        value_tail_buf: ``[B, n_kv, FP8_BLOCK_SIZE, D]`` DL16 — full tail block.
        key_cache: ``[NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE]`` FP8 QFP8WT.
        key_scale_cache: ``[NUM_BLOCKS, 1, 1, 1]`` compute dtype.
        value_cache: ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D]`` FP8 QFP8CH.
        block_idx: integer block index to write into.
        value_scale_cache: ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, 64]`` compute dtype.
            When provided, the V per-token scale is quantized and written here.
    """
    on_spyre = key_tail_buf.device.type == "spyre"
    B, n_kv, _, D = key_tail_buf.shape
    idx = torch.tensor([block_idx], dtype=torch.int32, device=key_tail_buf.device)

    if on_spyre:
        # K: [B, n_kv, FP8_BLOCK_SIZE, D] → [n_kv, D, FP8_BLOCK_SIZE] FP16
        k_fp16 = key_tail_buf[0].to(torch.float16)          # [n_kv, FP8_BLOCK_SIZE, D]
        k_perm = k_fp16.permute(0, 2, 1).contiguous()       # [n_kv, D, FP8_BLOCK_SIZE]
        sk = torch.ops.spyre.quantscalepertokenfp8(k_perm).amax()
        k_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(k_perm, sk)
        # Unsqueeze to [1, n_kv, D, FP8_BLOCK_SIZE] for dim-0 scatter.
        key_cache.index_copy_(0, idx, k_fp8.unsqueeze(0))
        key_scale_cache.index_copy_(0, idx, sk.reshape(1, 1, 1, 1))
        # V: [B, n_kv, FP8_BLOCK_SIZE, D] → [1, FP8_BLOCK_SIZE, n_kv, D] FP16
        v_in = value_tail_buf[0].permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        # Per-block scalar scale — same design as prefill path and ScaledPagedAttnStore.
        sv_scalar = (v_in.float().abs().amax() / FP8_MAX).to(
            torch.float16
        ).clamp(min=SCALE_EPS)                               # scalar FP16
        v_fp8 = torch.ops.spyre.quantize_fp8_with_scale(v_in, sv_scalar)
        value_cache.index_copy_(0, idx, v_fp8)
        if value_scale_cache is not None:
            sv_block = sv_scalar.to(value_scale_cache.dtype).reshape(1, 1, 1, 1)
            value_scale_cache.index_copy_(0, idx, sv_block)
    else:
        # Flush is a no-op on CPU — the tail buffer is read directly by fp8_attn_core.
        pass


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
    *,
    option: int = 1,
    key_tail_buf: torch.Tensor | None = None,
    value_tail_buf: torch.Tensor | None = None,
    block_indices: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    value_scale_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute attention using the FP8 KV cache.

    Option 1 (Spyre only) — direct-FP8-matmul + DL16 tail:
      Reshape block cache, quantize Q, ``spyre.scaled_bmm`` for committed tokens.
      Concatenate tail scores (DL16 ``torch.matmul``) before softmax.

    Option 2 (Spyre + CPU) — dequantize-K:
      Dequantize K from QFP8CH to DL16, run standard ``torch.matmul``.
      Falls back to Option 2 on CPU even when ``option=1``.

    Args:
        q: ``[B, H, S, D]`` query in compute dtype.
        key_cache:
            Option 1: ``[NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE]`` FP8 QFP8WT.
            Option 2: ``[B, n_kv, L, D]`` FP8 QFP8CH.
        key_scale_cache:
            Option 1: ``[NUM_BLOCKS, 1, 1, 1]`` per-block scale.
            Option 2: ``[B, n_kv, L, 1]`` per-token scale.
        value_cache:
            Option 1: ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D]`` FP8 QFP8CH.
            Option 2: ``[B, n_kv, L, D]`` compute dtype.
        attn_mask: additive causal mask of shape ``[B, 1, S, total_L]``
            where ``total_L = NUM_BLOCKS * FP8_BLOCK_SIZE + FP8_BLOCK_SIZE`` for
            Option 1 (committed + tail), or ``[B, 1, S, L]`` for Option 2.
        scaling: ``1 / sqrt(head_dim)``.
        compute_dtype: BF16 or FP16.
        option: 1 = block-layout ``scaled_bmm`` + DL16 tail.
                2 = dequantize-K ``torch.matmul``.
        key_tail_buf: ``[B, n_kv, FP8_BLOCK_SIZE, D]`` DL16 decode staging
            (Option 1 only; pass zeros when no decode tokens yet).
        value_tail_buf: ``[B, n_kv, FP8_BLOCK_SIZE, D]`` DL16 decode staging
            (Option 1 only; not yet flushed to FP8 committed cache).
        block_indices: ``[NUM_BLOCKS, 1]`` int32 tensor on the same device as
            ``key_cache``, containing ``[[0], [1], ..., [NUM_BLOCKS-1]]``.
            Must be pre-built **outside** ``torch.compile`` and passed in as a
            graph argument to avoid ``copy_from_d2d`` re-entry on Spyre.
            When ``None`` (eager / CPU fallback), built automatically.
        q_scale: ``[n_kv, kv_repeat, S, 1]`` FP16 per-token Q scale, pre-computed
            outside ``torch.compile`` via
            ``quantscalepertokenfp8(q_reshaped.to(float16))``.
            When ``None``, computed eagerly (works for CPU and eager mode; for
            compiled graphs pass this as a pre-built device tensor argument).
        value_scale_cache: ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, 64]`` compute dtype.
            Option 1 only.  Stick-padded V per-token scale; slice ``[:,:,:,:1]``
            inside the graph for dequantize broadcast.  When ``None``, V is not
            dequantized (should not happen in normal Option 1 use; fallback for
            backwards compatibility).

    Returns:
        ``[B, H, S, D_v]`` attention output in compute dtype.
    """
    B, H, S, D = q.shape
    on_spyre = q.device.type == "spyre"

    if option == 1 and on_spyre:
        # ── Option 1: FP8 block cache QK^T + DL16 tail ────────────────────
        # key_cache: [NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE]
        NUM_BLOCKS = key_cache.shape[0]
        n_kv = key_cache.shape[1]
        kv_repeat = H // n_kv
        total_L = NUM_BLOCKS * FP8_BLOCK_SIZE  # committed tokens

        # Gather K one block at a time and run scaled_bmm per block.
        #
        # (a) scaled_bmm requires the K operand's last dim to be exactly one
        #     FP8 stick (128 elements).  Concatenating all blocks would produce
        #     a multi-stick tensor that the hardware cannot lower.
        #     Fix: index_select one block per scaled_bmm call; cat scores after.
        #
        # (b) Permute/reshape on already-encoded FP8 tensors destroys the element
        #     arrangement.  Only index_select + squeeze/unsqueeze (no-layout ops)
        #     are safe on an encoded FP8 weight tensor at attention time.
        #
        # (c) GQA broadcast: scaled_bmm supports K with fewer heads than Q
        #     (n_kv < H).  After index_select → squeeze/unsqueeze, K is
        #     [n_kv, 1, D, FP8_BLOCK_SIZE] while Q is [n_kv, kv_repeat, S, D];
        #     the hardware broadcasts the same K head across kv_repeat Q heads.
        #
        # (d) Constructing index tensors inside torch.compile is not supported
        #     on Spyre.  Block index tensors must be pre-built outside the
        #     compiled graph and passed in as arguments.

        # Quantize Q per-token → QFP8CH [n_kv, kv_repeat, S, D].
        # When compiling, pass q_scale as a pre-built device tensor argument;
        # computing it inside the graph may produce a very small intermediate
        # that the hardware scheduler cannot lower.  None-fallback is fine for
        # eager / CPU mode.
        q_gqa = q.reshape(n_kv, kv_repeat, S, D)            # [n_kv, kv_repeat, S, D]
        q_fp16 = q_gqa.to(torch.float16)
        if q_scale is None:
            q_scale = torch.ops.spyre.quantscalepertokenfp8(q_fp16)
        q_fp8 = torch.ops.spyre.quantize_fp8_with_scale(q_fp16, q_scale)
        # q_fp8: [n_kv, kv_repeat, S, D]

        # Per-block FP8 QK^T via index_select + scaled_bmm.
        # block_indices must be a pre-built device tensor passed in as a graph
        # argument; creating it inside compile (torch.tensor([b]).to(device))
        # is not supported on Spyre.
        if block_indices is None:
            # Eager / CPU path: build on CPU then move to device.
            block_indices = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(1).to(
                key_cache.device
            )  # [NUM_BLOCKS, 1]
        # Static Python loop — torch.compile unrolls at trace time.
        block_scores: list[torch.Tensor] = []
        for b in range(NUM_BLOCKS):
            b_idx = block_indices[b]  # [1] int32 — slice of a pre-built device tensor
            k_blk = key_cache.index_select(0, b_idx)        # [1, n_kv, D, FP8_BLOCK_SIZE]
            k_b   = k_blk.squeeze(0).unsqueeze(1)           # [n_kv, 1, D, FP8_BLOCK_SIZE]
            sk_b  = key_scale_cache.index_select(0, b_idx)  # [1, 1, 1, 1]

            # FP8 QK^T — GQA broadcast: K has n_kv heads, Q has n_kv*kv_repeat heads.
            # mat1 [n_kv, kv_repeat, S, D]
            # mat2 [n_kv, 1,         D, 128]
            sc_b = torch.ops.spyre.scaled_bmm(q_fp8, k_b, out_dtype=torch.float16)
            sc_b = sc_b * q_scale * sk_b * scaling          # [n_kv, kv_repeat, S, 128]
            block_scores.append(sc_b.to(compute_dtype))

        # cat across blocks → [n_kv, kv_repeat, S, total_L]
        scores_committed = torch.cat(block_scores, dim=-1)

        # DL16 tail scores (decode tokens not yet in a full committed block).
        if key_tail_buf is not None:
            # key_tail_buf [B, n_kv, FP8_BLOCK_SIZE, D] → k_tail [n_kv, 1, D, FP8_BLOCK_SIZE]
            k_tail = key_tail_buf[0].permute(0, 2, 1).unsqueeze(1)
            scores_tail = torch.matmul(
                q_gqa.to(compute_dtype), k_tail.to(compute_dtype)
            ) * scaling                                      # [n_kv, kv_repeat, S, FP8_BLOCK_SIZE]
            scores = torch.cat([scores_committed, scores_tail], dim=-1)
        else:
            scores = scores_committed

        if attn_mask is not None:
            # attn_mask from generate() is [B, H_or_1, S, mask_L].
            # Option 1 scores are [n_kv, kv_repeat, S, scores_L] — the head dim
            # differs (n_kv×kv_repeat layout vs H or 1).  Collapse to [1,1,S,L]
            # so the mask broadcasts cleanly over both dims.
            # Also pad mask_L → scores_L with large negatives when the tail buffer
            # extends beyond max_cache_len.
            attn_mask = attn_mask[:, :1]          # [B, 1, S, mask_L]
            mask_L = attn_mask.shape[-1]
            scores_L = scores.shape[-1]
            if scores_L > mask_L:
                pad = attn_mask.new_full(
                    (*attn_mask.shape[:-1], scores_L - mask_L), -10000.0
                )
                attn_mask = torch.cat([attn_mask, pad], dim=-1)
            scores = scores + attn_mask.to(scores.dtype)

        attn_weights = torch.softmax(scores, dim=-1, dtype=compute_dtype)

        # AV multiply — committed V (FP8, dequantized) + DL16 tail V.
        # value_cache [NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D] QFP8CH
        # → permute(2,0,1,3) → [n_kv, NUM_BLOCKS, FP8_BLOCK_SIZE, D]
        # → reshape → [n_kv, total_L, D] → unsqueeze(1) → [n_kv, 1, total_L, D]
        vc_flat = (
            value_cache.permute(2, 0, 1, 3)
            .reshape(n_kv, total_L, D)
            .unsqueeze(1)
        )  # [n_kv, 1, total_L, D]
        if value_scale_cache is not None:
            # value_scale_cache [NUM_BLOCKS, 1, 1, 1] per-block scalar.
            # Reshape to [1, 1, NUM_BLOCKS, 1] then expand to [n_kv, 1, total_L, 1]
            # via repeat on the block dim (each block covers FP8_BLOCK_SIZE positions).
            # dequantize_fp8_with_scale: fp8todl16(vc_flat) * scale
            vsc = value_scale_cache.reshape(1, 1, NUM_BLOCKS, 1)   # [1,1,NB,1]
            vsc_exp = vsc.expand(n_kv, 1, NUM_BLOCKS, 1).reshape(
                n_kv, 1, NUM_BLOCKS, 1
            ).repeat(1, 1, FP8_BLOCK_SIZE, 1).reshape(
                n_kv, 1, total_L, 1
            )  # [n_kv, 1, total_L, 1]
            v_flat = torch.ops.spyre.dequantize_fp8_with_scale(
                vc_flat, vsc_exp
            )  # [n_kv, 1, total_L, D] DL16
        else:
            # Fallback: cast directly (no scale applied — for backwards compat).
            v_flat = vc_flat

        if value_tail_buf is not None:
            # Tail V is DL16 (not yet flushed to FP8 committed cache).
            v_tail = value_tail_buf[0].unsqueeze(1)          # [n_kv, 1, FP8_BLOCK_SIZE, D]
            v_full = torch.cat([v_flat.to(compute_dtype), v_tail.to(compute_dtype)], dim=2)
        else:
            v_full = v_flat

        # attn_weights [n_kv, kv_repeat, S, total_L+tail] @ v_full [n_kv, 1, total_L+tail, D]
        # → [n_kv, kv_repeat, S, D] → reshape → [B, H, S, D]
        attn_out = torch.matmul(attn_weights, v_full.to(compute_dtype))
        return attn_out.reshape(B, H, S, D)

    else:
        # ── Option 2 (Spyre + CPU): dequantize K → DL16, torch.matmul ────
        # key_cache: [B, n_kv, L, D] QFP8CH
        n_kv = key_cache.shape[1]
        kv_repeat = H // n_kv
        L = key_cache.shape[2]
        k_scale = key_scale_cache[:, :, :L, :]  # [B, n_kv, L, 1]

        k_dl16 = (
            key_cache.to(torch.float16)
            * k_scale.to(torch.float16)
        ).to(compute_dtype)
        k_expanded = k_dl16.repeat_interleave(kv_repeat, dim=1)  # [B, H, L, D]

        logits = torch.matmul(q, k_expanded.transpose(-2, -1)) * scaling
        if attn_mask is not None:
            logits = logits + attn_mask

        attn_weights = torch.softmax(logits, dim=-1, dtype=compute_dtype)

        v_expanded = value_cache.repeat_interleave(kv_repeat, dim=1)  # [B, H, L, D_v]
        attn_out = torch.matmul(attn_weights, v_expanded)
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
    *,
    option: int = 1,
):
    """Allocate FP8 KV cache tensors.

    **Option 1 returns a 5-tuple:**
    ``(key_cache, key_scale_cache, value_cache, value_scale_cache,
    (key_tail_buf, value_tail_buf))``

    **Option 2 returns a 3-tuple:**
    ``(key_cache, key_scale_cache, value_cache)``

    Option 1 shapes (Spyre):
      ``key_cache``         ``[NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE]`` FP8 QFP8WT
      ``key_scale_cache``   ``[NUM_BLOCKS, 1, 1, 1]``                  compute dtype
      ``value_cache``       ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D]``  FP8 QFP8CH
      ``value_scale_cache`` ``[NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, 64]`` compute dtype
      ``key_tail_buf``      ``[B, n_kv, FP8_BLOCK_SIZE, D]``           compute dtype
      ``value_tail_buf``    ``[B, n_kv, FP8_BLOCK_SIZE, D]``           compute dtype
      (tail buffers are the 5th element, packed as a 2-tuple)

    Option 2 shapes:
      ``key_cache``       ``[B, n_kv, L, D]`` FP8 QFP8CH
      ``key_scale_cache`` ``[B, n_kv, L, 1]`` compute dtype
      ``value_cache``     ``[B, n_kv, L, D]`` compute dtype

    Args:
        batch_size: B.
        num_kv_heads: n_kv.
        max_cache_len: L (max sequence / cache length).
            For Option 1 must be a multiple of ``FP8_BLOCK_SIZE``.
        head_dim: D.  For Option 1 must equal ``FP8_BLOCK_SIZE`` (128).
        compute_dtype: dtype for scales, V, and tail buffers.
        device: target device.
        option: 1 = block-scatter QFP8WT layout; 2 = QFP8CH standard layout.
    """
    if device is None:
        from hf_adapters.hf_common import DEVICE
        device = DEVICE

    on_spyre = torch.device(device).type == "spyre"

    from hf_adapters.hf_common import allocate_kv_cache_tensor

    if option == 1:
        # ── Option 1: block layout ─────────────────────────────────────────
        assert max_cache_len % FP8_BLOCK_SIZE == 0, (
            f"Option 1: max_cache_len={max_cache_len} must be a multiple of "
            f"FP8_BLOCK_SIZE={FP8_BLOCK_SIZE}"
        )
        NUM_BLOCKS = max_cache_len // FP8_BLOCK_SIZE

        # key_cache: [NUM_BLOCKS, n_kv, D, FP8_BLOCK_SIZE] FP8 QFP8WT.
        # Allocated as CPU zeros then moved — avoids RuntimeOperationFill on QFP8WT.
        shape_k = (NUM_BLOCKS, num_kv_heads, head_dim, FP8_BLOCK_SIZE)
        key_cache = torch.zeros(shape_k, dtype=FP8_DTYPE).to(device)

        # key_scale_cache: [NUM_BLOCKS, 1, 1, 1] — one scalar per block.
        # Allocate as plain zeros (no STL needed for this tiny tensor).
        key_scale_cache = torch.zeros(
            (NUM_BLOCKS, 1, 1, 1), dtype=compute_dtype, device=device
        )

        # value_cache: [NUM_BLOCKS, FP8_BLOCK_SIZE, n_kv, D] FP8 QFP8CH token-major.
        # Allocated as CPU zeros then moved — avoids RuntimeOperationFill on QFP8CH.
        shape_v = (NUM_BLOCKS, FP8_BLOCK_SIZE, num_kv_heads, head_dim)
        value_cache = torch.zeros(shape_v, dtype=FP8_DTYPE).to(device)

        # value_scale_cache: [NUM_BLOCKS, 1, 1, 1] — per-block scalar, same shape
        # as key_scale_cache.  One amax scalar per committed V block.
        # Matches ScaledPagedAttnStore reference design.
        value_scale_cache = torch.zeros(
            (NUM_BLOCKS, 1, 1, 1), dtype=compute_dtype, device=device
        )

        # Tail buffers: [B, n_kv, FP8_BLOCK_SIZE, D] DL16 — decode staging.
        # Fixed-shape: always FP8_BLOCK_SIZE slots regardless of how many are filled.
        # Positions beyond tail_pos are zero and masked by attn_mask.
        shape_tail = (batch_size, num_kv_heads, FP8_BLOCK_SIZE, head_dim)
        key_tail_buf = torch.zeros(shape_tail, dtype=compute_dtype, device=device)
        value_tail_buf = torch.zeros(shape_tail, dtype=compute_dtype, device=device)

        return key_cache, key_scale_cache, value_cache, value_scale_cache, (key_tail_buf, value_tail_buf)

    else:
        # ── Option 2 / CPU: standard [B, n_kv, L, D] layout ──────────────
        if on_spyre:
            from hf_adapters.hf_common import _cache_position_first_stl
            from torch_spyre._C import ElementArrangement  # type: ignore[import-not-found]
            try:
                stl = _cache_position_first_stl(
                    batch_size, num_kv_heads, max_cache_len, head_dim, FP8_DTYPE,
                    element_arrangement=ElementArrangement.QFP8CH,
                )
            except Exception:  # noqa: BLE001
                stl = None

            shape_k = (batch_size, num_kv_heads, max_cache_len, head_dim)
            if stl is None:
                key_cache = torch.zeros(shape_k, dtype=FP8_DTYPE, device=device)
            else:
                key_cache = torch.empty(  # type: ignore[call-overload]
                    shape_k,
                    device=torch.device(device),
                    device_layout=stl,
                    dtype=FP8_DTYPE,
                )
                key_cache.view(torch.uint8).zero_()
        else:
            shape_k = (batch_size, num_kv_heads, max_cache_len, head_dim)
            key_cache = torch.zeros(shape_k, dtype=FP8_DTYPE, device=device)

        # Key scale cache: [B, n_kv, L, 1].
        key_scale_cache = allocate_kv_cache_tensor(
            batch_size, num_kv_heads, max_cache_len, 1, compute_dtype, device
        )
        value_cache = allocate_kv_cache_tensor(
            batch_size, num_kv_heads, max_cache_len, head_dim, compute_dtype, device
        )
        return key_cache, key_scale_cache, value_cache
