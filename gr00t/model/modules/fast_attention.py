# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``gr00t_fast``: a transformers attention implementation for the Qwen3-VL backbone.

HF's ``flash_attention_2`` runs FlashAttention-2 CUDA kernels, which do not use Blackwell's tensor
memory / TMA and, being an opaque custom op, force a ``torch.compile`` graph break in every block.
On B300 PyTorch's cuDNN SDPA backend is ~2x faster than FA2 for GR00T's shapes (207-token causal
GQA decoder layers, 256-patch image segments) and is traceable by Inductor. This implementation:

* **regular batches** (no padding): ``F.scaled_dot_product_attention`` with ``is_causal`` and
  native GQA -- the decoder layers of a single-task batch, where every prompt has the same length;
* **packed image segments** (the vision tower passes ``cu_seq_lens_*``): when all segments have the
  same length (every image the same size, the GR00T case) the packed ``(1, h, total, d)`` tensors
  are viewed as a regular ``(n_images, h, L, d)`` batch and go through SDPA; otherwise FlashAttention
  varlen;
* **padded batches** (multi-task prompts of different lengths): unpad + FlashAttention varlen,
  FlashAttention-4 (``flash_attn.cute``, CuTe DSL) when installed, else FlashAttention-2.

Use it with ``Qwen3Backbone(attn_implementation="gr00t_fast")``. ``set_packed_segment_length`` must
be called once per forward with the tokens-per-image (``Qwen3Backbone.forward`` does), so the
uniform-segment test needs no device sync inside the compiled blocks.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

IMPLEMENTATION = "gr00t_fast"

_SDP_BACKENDS = {
    "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
    "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    "math": torch.nn.attention.SDPBackend.MATH,
}

# Tokens per packed image segment for the current forward (None = unknown / not uniform).
_packed_segment_length: Optional[int] = None


def set_packed_segment_length(length: Optional[int]) -> None:
    global _packed_segment_length
    _packed_segment_length = length


def uniform_segment_length(grid_thw: torch.Tensor) -> Optional[int]:
    """Patch tokens per vision segment if every image in ``grid_thw`` has the same (h, w); one
    small device->host copy per forward."""
    hw = grid_thw[:, 1] * grid_thw[:, 2]
    if hw.numel() == 0:
        return None
    first = int(hw[0])
    return first if bool((hw == first).all()) else None


def set_sdpa_backend_priority(priority: str) -> None:
    """Global SDPA backend order, e.g. ``"cudnn,efficient,flash,math"``. Torch's default puts
    cuDNN last; on Blackwell it is the fastest backend for most of GR00T's shapes (the DiT's
    diffusers ``Attention`` uses SDPA and picks the first supported backend in this order)."""
    order = [_SDP_BACKENDS[name.strip()] for name in priority.split(",") if name.strip()]
    missing = [b for b in _SDP_BACKENDS.values() if b not in order]
    order += missing
    if not hasattr(torch._C, "_set_sdp_priority_order"):
        logger.warning("torch has no _set_sdp_priority_order; SDPA priority unchanged")
        return
    torch._C._set_sdp_priority_order(
        [int(b) for b in order]
    )  # same call sdpa_kernel(set_priority=True) makes
    logger.info(f"SDPA backend priority: {[b.name for b in order]}")


# ----------------------------------------------------------------------------------------------------
# varlen flash attention (FA4 if available, else FA2) -- only for padded / ragged inputs
# ----------------------------------------------------------------------------------------------------
_varlen_impl = None


def _get_varlen():
    global _varlen_impl
    if _varlen_impl is None:
        try:
            from flash_attn.cute import flash_attn_varlen_func as fa4_varlen

            def fa4(q, k, v, cu_q, cu_k, max_q, max_k, causal, scale):
                out = fa4_varlen(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=max_q,
                    max_seqlen_k=max_k,
                    softmax_scale=scale,
                    causal=causal,
                )
                return out[0] if isinstance(out, tuple) else out  # FA4 also returns the lse

            _varlen_impl = ("flash_attention_4", fa4)
        except ImportError:
            from flash_attn import flash_attn_varlen_func as fa2_varlen

            def fa2(q, k, v, cu_q, cu_k, max_q, max_k, causal, scale):
                return fa2_varlen(
                    q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale=scale, causal=causal
                )

            _varlen_impl = ("flash_attention_2", fa2)
        logger.info(f"gr00t_fast attention: varlen fallback = {_varlen_impl[0]}")
    return _varlen_impl[1]


def _flash_dtype(q: torch.Tensor) -> torch.dtype:
    """Flash kernels take fp16/bf16 only. After M-RoPE q/k are fp32 even under bf16 autocast
    (autocast casts SDPA's inputs itself, not a custom op's), so cast like HF's flash wrapper."""
    if q.dtype in (torch.float16, torch.bfloat16):
        return q.dtype
    return torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.bfloat16


@torch._dynamo.disable
def _varlen_attention(q, k, v, cu_q, cu_k, max_q, max_k, causal: bool, scale, dropout: float):
    """q/k/v: (total, h, d). Runs eagerly: the flash custom ops do not trace under dynamo."""
    dtype = _flash_dtype(q)
    q, k, v = (x.to(dtype) for x in (q, k, v))
    max_q = int(max_q) if max_q is not None else int((cu_q[1:] - cu_q[:-1]).max())
    max_k = int(max_k) if max_k is not None else int((cu_k[1:] - cu_k[:-1]).max())
    if dropout > 0.0:
        from flash_attn import flash_attn_varlen_func  # FA2 is the only one with dropout

        return flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=dropout, softmax_scale=scale, causal=causal
        )
    return _get_varlen()(q, k, v, cu_q, cu_k, max_q, max_k, causal, scale)


@torch._dynamo.disable
def _padded_attention(query, key, value, attention_mask, is_causal: bool, scale, dropout: float):
    """query/key/value: (b, s, h, d); attention_mask: (b, s) padding mask. Returns (b, s, h, d)."""
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    batch, seq_len, num_heads, head_dim = query.shape
    kv_heads = key.shape[2]
    key_u, indices_k, cu_k, max_k, _ = unpad_input(key, attention_mask)
    value_u = index_first_axis(value.reshape(batch * seq_len, kv_heads, head_dim), indices_k)
    query_u, indices_q, cu_q, max_q, _ = unpad_input(query, attention_mask)
    out = _varlen_attention(
        query_u, key_u, value_u, cu_q, cu_k, max_q, max_k, is_causal, scale, dropout
    )
    return pad_input(out, indices_q, batch, seq_len)


# ----------------------------------------------------------------------------------------------------
# the attention function registered with transformers
# ----------------------------------------------------------------------------------------------------
def gr00t_fast_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Inputs ``(b, h, s, d)``; returns ``((b, s, h, d), None)`` like the other HF implementations."""
    if sliding_window is not None or softcap:
        raise NotImplementedError(
            "gr00t_fast attention: sliding_window / softcap are not supported"
        )
    dropout = dropout if module.training else 0.0
    cu_q = kwargs.get("cu_seq_lens_q")

    if cu_q is not None:
        # Packed segments from the vision tower: (1, h, total, d).
        cu_k = kwargs.get("cu_seq_lens_k", cu_q)
        num_heads, total, head_dim = query.shape[1], query.shape[2], query.shape[3]
        num_segments = cu_q.numel() - 1
        seg = _packed_segment_length
        if seg is not None and num_segments * seg == total and dropout == 0.0:

            def to_batch(x):  # (1, h, total, d) -> (n, h, seg, d)
                return x[0].view(x.shape[1], num_segments, seg, x.shape[3]).transpose(0, 1)

            out = F.scaled_dot_product_attention(
                to_batch(query), to_batch(key), to_batch(value), scale=scaling, is_causal=False
            )  # (n, h, seg, d)
            out = out.transpose(0, 1).reshape(num_heads, total, head_dim).transpose(0, 1)
            return out.unsqueeze(0).contiguous(), None
        q, k, v = (x[0].transpose(0, 1) for x in (query, key, value))  # (total, h, d)
        out = _varlen_attention(
            q,
            k,
            v,
            cu_q,
            cu_k,
            kwargs.get("max_length_q"),
            kwargs.get("max_length_k"),
            False,
            scaling,
            dropout,
        )
        return out.unsqueeze(0), None

    is_causal = kwargs.get("is_causal")
    if is_causal is None:
        is_causal = query.shape[2] > 1 and getattr(module, "is_causal", True)

    if attention_mask is None:
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            enable_gqa=key.shape[1] != query.shape[1],
        )
        return out.transpose(1, 2).contiguous(), None

    # Padded batch: attention_mask is the (b, s) padding mask (see the registered mask function).
    if attention_mask.ndim != 2:
        raise ValueError(
            f"gr00t_fast attention expects a 2D padding mask, got {tuple(attention_mask.shape)}"
        )
    out = _padded_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attention_mask,
        is_causal,
        scaling,
        dropout,
    )
    return out, None


def register_gr00t_fast_attention() -> None:
    """Register ``gr00t_fast`` with transformers (idempotent). The mask function is the flash one:
    ``None`` without padding, the 2D padding mask otherwise."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask

    if IMPLEMENTATION not in AttentionInterface._global_mapping:
        AttentionInterface.register(IMPLEMENTATION, gr00t_fast_attention_forward)
    if IMPLEMENTATION not in AttentionMaskInterface._global_mapping:
        AttentionMaskInterface.register(IMPLEMENTATION, flash_attention_mask)


# ----------------------------------------------------------------------------------------------------
# Qwen3-VL vision attention: route cu_seqlens to our implementation too
# ----------------------------------------------------------------------------------------------------
def _vision_attention_forward(
    self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs
):
    """``Qwen3VLVisionAttention.forward`` with the packed (cu_seqlens) call extended to gr00t_fast.
    Stock HF only packs for flash_attention_2 and otherwise loops over the segments (~3000 SDPA calls
    per layer for a 1024-sample batch)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    attn_output, _ = gr00t_fast_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=None,  # only needed by the ragged fallback, which derives it lazily
        max_length_k=None,
        is_causal=False,
    )
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return self.proj(attn_output)


def patch_qwen3_vl_vision_attention(vision_model: torch.nn.Module) -> int:
    """Bind the packed forward to every vision attention module of ``vision_model``."""
    import types

    n = 0
    for blk in vision_model.blocks:
        blk.attn.forward = types.MethodType(_vision_attention_forward, blk.attn)
        n += 1
    return n
