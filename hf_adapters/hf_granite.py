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

"""
HuggingFace Transformers adapter for Granite 3.3 models on Spyre.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "/path/to/granite-3.3-8b-instruct")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/granite-3.3-8b-instruct")
    encoded = tokenizer(["Hello!"], return_tensors="pt")
    outputs = model.generate(**encoded, max_new_tokens=32)
"""

import types

import torch

from hf_adapters.fp8_linear import swap_linears_to_fp8

from hf_adapters.hf_common import (
    _SDPA_MAX_SEQUENCE_TILE_SIZE,
    get_backbone,
    pad_lm_head,
    prepare_rope_and_heads,
    prepare_standard_gqa_blocks,
    prepare_standard_gqa_blocks_fp8_kv,
    text_config,
)


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
    key_scale_caches=None,
):
    """Granite 3.3 backbone: embedding * multiplier, blocks, norm.

    When ``key_scale_caches`` is provided (FP8 KV cache mode), each compiled
    block receives the extra per-token K scale cache and returns it updated.
    """
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)
    h = h * backbone.embedding_multiplier

    selected_freqs = model._spyre_rope(h, position_ids)

    if key_scale_caches is None:
        # Standard BF16/FP16 KV cache path
        for i, compiled_block in enumerate(model._spyre_compiled_blocks):
            h, key_caches[i], value_caches[i] = compiled_block(
                h,
                selected_freqs,
                attn_mask,
                key_caches[i],
                value_caches[i],
                cache_index,
            )
    else:
        # FP8 KV cache path — block signature carries key_scale_cache
        for i, compiled_block in enumerate(model._spyre_compiled_blocks):
            h, key_caches[i], key_scale_caches[i], value_caches[i] = compiled_block(
                h,
                selected_freqs,
                attn_mask,
                key_caches[i],
                key_scale_caches[i],
                value_caches[i],
                cache_index,
            )

    h = model._spyre_compiled_norm(h)
    return h


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
    key_scale_caches=None,
):
    """Granite 3.3 causal-LM forward: backbone + head / scaling."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
        key_scale_caches=key_scale_caches,
    )
    logits = model.lm_head(h)
    return logits / text_config(model.config).logits_scaling

def _fp16_rmsnorm_forward(self, h):
    """RMSNorm with the variance reduction kept in fp16 on Spyre."""
    if h.device.type != "spyre":
        return type(self).forward(self, h)
    variance = (h * h).mean(-1, keepdim=True)
    return self.weight * (h * torch.rsqrt(variance + self.variance_epsilon))


def prepare_for_spyre(model, fp8_kv_cache: bool = False):
    """Apply Spyre adaptations to Granite 3.3 model in-place.

    Args:
        fp8_kv_cache: When ``True``, use FP8 KV cache — K is stored as E4M3
            and QK^T is computed in FP8 via ``spyre.scaled_mm``.  V remains
            in the compute dtype.  Requires the model to already be prepared
            with FP8 weight loading (``swap_linears_to_fp8(direct_load=True)``).
    """
    backbone = get_backbone(model)
    # FP8 checkpoints only; runs before the blocks are built so they close over
    # FP8Linear.  direct_load=True keeps E4M3 weights as-is (prequantized=True)
    # so load_fp8_model_to_spyre can DMA them straight into QFP8WT/KERNEL layout
    # without any dequantize→fp16→device→re-quantize round-trip.
    # Use the model's config dtype for excluded linears (o_proj, down_proj) so
    # they match the activation dtype — avoids bfloat16/float16 mismatch.
    cfg_dtype = getattr(model.config, "torch_dtype", None) or getattr(
        model.config, "dtype", torch.float16
    )
    n_fp8, n_excluded = swap_linears_to_fp8(model, direct_load=True, dtype=cfg_dtype)
    if n_fp8 or n_excluded:
        print(f"FP8: {n_fp8} module(s) -> FP8Linear (direct E4M3), {n_excluded} -> {cfg_dtype} nn.Linear")
    if n_fp8:
        # Stock RMSNorm promotes to fp32 for variance; keep norms feeding FP8Linear
        # in fp16 so the layout chain to scaled_mm remains feasible on Spyre.
        for layer in backbone.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                norm.forward = types.MethodType(_fp16_rmsnorm_forward, norm)
    prepare_rope_and_heads(model)
    pad_lm_head(model)
    if fp8_kv_cache:
        model._spyre_compiled_blocks = prepare_standard_gqa_blocks_fp8_kv(
            backbone.layers, True
        )
        model._spyre_fp8_kv_cache = True
        print(f"FP8 KV cache: enabled (K stored as E4M3, QK^T in FP8)")
    else:
        model._spyre_compiled_blocks = prepare_standard_gqa_blocks(backbone.layers, True)
        model._spyre_fp8_kv_cache = False
    model._spyre_compiled_norm = torch.compile(backbone.norm, dynamic=False)
    model._spyre_prefill_chunk_size = _SDPA_MAX_SEQUENCE_TILE_SIZE
