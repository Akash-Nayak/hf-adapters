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

"""FP8 (E4M3) linear layer for compressed-tensors checkpoints on Spyre.

Per-output-channel weight scale, per-token dynamic activation scale, no bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
SCALE_EPS = 1e-4  # CPU path: keeps reciprocal(x_scale) finite for all-zero rows

# All projections run FP8 scaled_mm on the eval/fp8-prequant-with-hf-adapter
# torch-spyre branch (feature/prequantized-fp8-weight-support base):
#   - DtException on o_proj: fixed by _project_pointwise_dim_order rank_diff<0
#     sparse-stick marker fix in propagate_layouts.py (commit 8a0b32e8)
#   - ReStickifyOpHBM on SEN143_FP8 for down_proj: fixed by the same commit's
#     _project_pointwise_dim_order and find_stick_compatible_input_layout guards
# No exclusions needed on this branch.
DEFAULT_FP8_EXCLUDE: tuple[str, ...] = ()


class FP8Linear(nn.Module):
    """FP8 replacement for a bias-free ``nn.Linear``.

    ``weight`` is stored ``[in_features, out_features]``: transposing an E4M3
    tensor on device needs a restickify Spyre cannot emit for FP8.

    When ``prequantized=True`` the weight is already E4M3 (either placed there by
    ``prequantize_fp8_weights`` after an fp16 round-trip, or loaded directly from
    the checkpoint via ``from_fp8_checkpoint``).  In that case the Spyre path uses
    it as-is and ``load_fp8_model_to_spyre`` DMAs it straight into QFP8WT layout
    with no on-device re-quantization step.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype
        # Set when weight already holds E4M3 — either by prequantize_fp8_weights
        # or by from_fp8_checkpoint (direct-load path).
        self.prequantized = False

        self.register_buffer(
            "weight", torch.empty(in_features, out_features, dtype=dtype)
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=dtype))

    @classmethod
    def from_fp8_checkpoint(
        cls,
        linear: nn.Module,
        dtype: torch.dtype = torch.float16,
    ) -> "FP8Linear":
        """Build an FP8Linear directly from a compressed-tensors E4M3 checkpoint Linear.

        Unlike ``swap_linears_to_fp8`` (which dequantizes E4M3→fp16 so a later
        ``prequantize_fp8_weights`` call can re-quantize on-device), this factory
        keeps the checkpoint weight in E4M3 and marks ``prequantized=True``
        immediately.  ``load_fp8_model_to_spyre`` (``use_fp8_weights=True``) then
        DMAs the E4M3 weight directly into QFP8WT/KERNEL layout on the device,
        skipping the dequantize→fp16→device→re-quantize round-trip entirely.

        The weight is transposed from checkpoint layout ``[out, in]`` to
        ``[in, out]`` to match ``scaled_mm``'s expected operand orientation.
        """
        out_f, in_f = linear.weight.shape  # checkpoint: [out, in]
        fp8 = cls(in_f, out_f, dtype=dtype)
        # Transpose to [in, out] — QFP8WT DMA ignores CPU strides so no
        # .contiguous() needed; the contiguous copy is done inside _dma_to_spyre_fp8_kernel.
        fp8.weight = linear.weight.t().contiguous().to(FP8_DTYPE)
        fp8.weight_scale = linear.weight_scale.reshape(-1).to(dtype)
        fp8.prequantized = True
        return fp8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.out_features)

        if x.device.type == "spyre":
            # Per-token activation scale via standard ops.
            # quantscalepertokenfp8 (torch-spyre#3457) is avoided: it emits a
            # quantization_double_pad.ddl kernel whose output loop dimension
            # (last-dim=1) is "marked to be dropped" by dbo-opt — firmware bug.
            # abs().amax() decomposes into separate kernels that avoid that DDL.
            x_scale = (
                (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX))
                .clamp(min=SCALE_EPS)
                .clone()
            )
            wq = (
                self.weight
                if self.prequantized
                else torch.ops.spyre.quantize_weight_fp8_with_scale(
                    self.weight, self.weight_scale
                )
            )
            xq = torch.ops.spyre.quantize_fp8_with_scale(x, x_scale)
            y = torch.ops.spyre.scaled_mm(
                xq.reshape(-1, xq.shape[-1]), wq, out_dtype=self.compute_dtype
            )
            # Cast scale tensors to compute_dtype so the multiply stays in
            # f16/bf16.  Mixed-dtype mul (bf16 x_scale × f16 y) would promote
            # to fp32, generating a dead f32[1,1,1] dl16tofp32 sdsc kernel
            # whose single-element output stick causes a dbo-opt DDL crash
            # ("dimension marked to be dropped" in quantization_double_pad.ddl).
            dtype = self.compute_dtype
            y = (
                y.reshape(out_shape)
                * x_scale.to(dtype)
                * self.weight_scale.to(dtype)
            )
            return y.to(x.dtype)

        # CPU reference path; fp32 accumulation avoids fp16 overflow.
        x_scale = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(
            min=SCALE_EPS
        )
        if self.prequantized:
            # Weight is already E4M3 in [in, out] orientation — use directly.
            wq = self.weight
        else:
            wq = (
                (self.weight * torch.reciprocal(self.weight_scale))
                .clamp(-FP8_MAX, FP8_MAX)
                .to(FP8_DTYPE)
            )
        xq = (x * torch.reciprocal(x_scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        acc = xq.reshape(-1, xq.shape[-1]).to(torch.float32) @ wq.to(torch.float32)
        return (acc.reshape(out_shape) * x_scale * self.weight_scale).to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"prequantized={self.prequantized}"
        )


def _dequantize_checkpoint_weight(linear: nn.Module) -> torch.Tensor:
    """E4M3 ``[out, in]`` * ``weight_scale [out, 1]`` -> fp16 ``[out, in]``."""
    scale = linear.weight_scale.to(torch.float32)
    return (linear.weight.to(torch.float32) * scale).to(torch.float16)


def swap_linears_to_fp8(
    model: nn.Module,
    *,
    exclude: tuple[str, ...] | None = None,
    dtype: torch.dtype = torch.float16,
    direct_load: bool = False,
) -> tuple[int, int]:
    """Replace a compressed-tensors checkpoint's E4M3 Linears with FP8Linear.

    Must run before the first forward: compressed-tensors decompresses E4M3
    weights in place on first use. Returns ``(n_swapped, n_excluded)``.

    Args:
        direct_load: When ``True``, use ``FP8Linear.from_fp8_checkpoint`` to keep
            the E4M3 weight as-is and mark ``prequantized=True`` immediately.
            ``load_fp8_model_to_spyre`` (``use_fp8_weights=True``) will then DMA
            the E4M3 weight directly into QFP8WT/KERNEL layout, bypassing the
            dequantize→fp16→device→re-quantize round-trip.
            When ``False`` (default), dequantizes to fp16 so the existing
            ``prequantize_fp8_weights`` on-device path can be used.
    """
    if exclude is None:
        exclude = DEFAULT_FP8_EXCLUDE

    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.weight.dtype == FP8_DTYPE
    ]

    n_swapped = n_excluded = 0
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, attr)

        # Excluded projections: always dequantize to fp16 regardless of direct_load
        # so that untouched E4M3 weights don't silently drop weight_scale on .to().
        if attr in exclude or any(part in exclude for part in name.split(".")):
            weight_fp16 = _dequantize_checkpoint_weight(old)
            plain = nn.Linear(
                old.in_features, old.out_features, bias=False, dtype=dtype
            )
            plain.weight.data.copy_(weight_fp16)
            setattr(parent, attr, plain)
            n_excluded += 1
            continue

        if direct_load:
            # Keep E4M3 weight; mark prequantized so load_fp8_model_to_spyre
            # DMAs it directly into QFP8WT without any on-device re-quantization.
            fp8 = FP8Linear.from_fp8_checkpoint(old, dtype=dtype)
        else:
            weight_fp16 = _dequantize_checkpoint_weight(old)  # [out, in]
            fp8 = FP8Linear(old.in_features, old.out_features, dtype=dtype)
            fp8.weight.copy_(weight_fp16.t().contiguous())  # -> [in, out]
            fp8.weight_scale.copy_(old.weight_scale.reshape(-1).to(dtype))
        setattr(parent, attr, fp8)
        n_swapped += 1

    return n_swapped, n_excluded


def prequantize_fp8_weights(model: nn.Module) -> int:
    """Quantize each FP8Linear weight to E4M3/QFP8WT once, in place.

    Run after the device move and before the first forward (compile is lazy).
    Frees the fp16 copy and removes the per-call weight quantize from forward.
    Returns the number of modules quantized.
    """
    # TODO: transfer E4M3 directly into QFP8WT once supported; drops the CPU
    # dequantize and this on-device pass.
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, FP8Linear) or m.prequantized:
            continue
        if m.weight.device.type != "spyre":  # QFP8WT is a device-only layout
            continue

        wq = torch.ops.spyre.quantize_weight_fp8_with_scale(m.weight, m.weight_scale)
        if not isinstance(wq, torch.Tensor) or wq.dtype != FP8_DTYPE:
            raise RuntimeError(
                f"quantize_weight_fp8_with_scale returned "
                f"{getattr(wq, 'dtype', type(wq).__name__)} for {name!r}: "
                f"FP8 weight prequantization requires eager FP8 op support in "
                f"torch-spyre (torch-spyre#3172)."
            )

        m.weight = wq  # rebinding drops the fp16 buffer
        m.prequantized = True
        n += 1

    if n:
        print(f"FP8: {n} weight(s) prequantized to E4M3/QFP8WT")
    return n


def fp8_status(model: nn.Module) -> dict:
    """Counts used to verify the FP8 swap and prequantization took effect."""
    status = {
        "n_fp8": 0,
        "n_prequantized": 0,
        "n_weight_fp8": 0,
        "n_unswapped_e4m3": 0,
        "orientation_ok": True,
    }
    for _, m in model.named_modules():
        # Use type name check to avoid class identity mismatch when FP8Linear
        # is imported from a different PYTHONPATH entry than the model was built with.
        is_fp8 = type(m).__name__ == "FP8Linear"
        if is_fp8:
            status["n_fp8"] += 1
            status["n_prequantized"] += int(getattr(m, "prequantized", False))
            status["n_weight_fp8"] += int(m.weight.dtype == FP8_DTYPE)
            if tuple(m.weight.shape) != (m.in_features, m.out_features):
                status["orientation_ok"] = False
        elif isinstance(m, nn.Linear) and m.weight.dtype == FP8_DTYPE:
            status["n_unswapped_e4m3"] += 1
    return status
