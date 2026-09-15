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

"""CPU tests for gr00t_fast attention (SDPA paths) and the linear patch embedding."""

import types

from gr00t.model.modules import fast_attention as fa
import pytest
import torch


def _reference(q, k, v, causal):
    """Plain fp32 softmax attention on (b, h, s, d) with GQA by repeating kv heads."""
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    scores = (q.float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5
    if causal:
        s = q.shape[2]
        scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
    return scores.softmax(-1) @ v.float()


class _Module(torch.nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.is_causal = causal
        self.training = False


class TestGr00tFastAttention:
    def test_regular_causal_gqa_matches_reference(self):
        torch.manual_seed(0)
        b, s, h, hk, d = 2, 9, 4, 2, 16
        q, k, v = torch.randn(b, h, s, d), torch.randn(b, hk, s, d), torch.randn(b, hk, s, d)
        out, w = fa.gr00t_fast_attention_forward(_Module(True), q, k, v, attention_mask=None)
        assert w is None and out.shape == (b, s, h, d)
        ref = _reference(q, k, v, causal=True).transpose(1, 2)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_packed_uniform_segments_match_per_segment_reference(self):
        torch.manual_seed(0)
        n_seg, seg, h, d = 3, 5, 2, 8
        total = n_seg * seg
        q, k, v = (torch.randn(1, h, total, d) for _ in range(3))
        cu = torch.arange(0, total + 1, seg, dtype=torch.int32)
        fa.set_packed_segment_length(seg)
        try:
            out, _ = fa.gr00t_fast_attention_forward(
                _Module(False),
                q,
                k,
                v,
                attention_mask=None,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                is_causal=False,
            )
        finally:
            fa.set_packed_segment_length(None)
        assert out.shape == (1, total, h, d)
        for i in range(n_seg):
            sl = slice(i * seg, (i + 1) * seg)
            ref = _reference(q[:, :, sl], k[:, :, sl], v[:, :, sl], causal=False).transpose(1, 2)
            assert torch.allclose(out[:, sl], ref, atol=1e-5)

    def test_uniform_segment_length(self):
        grid = torch.tensor([[1, 16, 16], [1, 16, 16], [1, 16, 16]])
        assert fa.uniform_segment_length(grid) == 256
        assert fa.uniform_segment_length(torch.tensor([[1, 16, 16], [1, 8, 8]])) is None
        assert fa.uniform_segment_length(torch.zeros(0, 3, dtype=torch.long)) is None

    def test_register_is_idempotent_and_visible_to_transformers(self):
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface

        fa.register_gr00t_fast_attention()
        fa.register_gr00t_fast_attention()
        assert (
            AttentionInterface._global_mapping[fa.IMPLEMENTATION] is fa.gr00t_fast_attention_forward
        )
        assert fa.IMPLEMENTATION in AttentionMaskInterface._global_mapping

    def test_rejects_unsupported_options(self):
        q = torch.randn(1, 1, 4, 8)
        with pytest.raises(NotImplementedError):
            fa.gr00t_fast_attention_forward(_Module(True), q, q, q, None, sliding_window=4)


class TestLinearPatchEmbed:
    def test_matches_conv3d(self):
        from gr00t.model.modules.qwen3_vl_fast_patch_embed import (
            _linear_patch_embed_forward,
            apply_fast_qwen3_vl_patch_embed,
        )

        torch.manual_seed(0)
        c, t, p, e = 3, 2, 4, 6
        proj = torch.nn.Conv3d(c, e, kernel_size=(t, p, p), stride=(t, p, p))
        pe = types.SimpleNamespace(proj=proj)
        x = torch.randn(7, c * t * p * p)
        ref = proj(x.view(-1, c, t, p, p)).view(-1, e)
        out = _linear_patch_embed_forward(pe, x)
        assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-5)

        # apply_* only binds when the convolution is the degenerate one-position case
        class Visual(torch.nn.Module):
            def __init__(self, proj):
                super().__init__()
                self.patch_embed = torch.nn.Module()
                self.patch_embed.proj = proj

        model = torch.nn.Module()
        model.visual = Visual(proj)
        assert apply_fast_qwen3_vl_patch_embed(model) is True
        assert torch.allclose(model.visual.patch_embed.forward(x), ref, atol=1e-5)
        strided = torch.nn.Module()
        strided.visual = Visual(torch.nn.Conv3d(c, e, kernel_size=(t, p, p), stride=(1, p, p)))
        assert apply_fast_qwen3_vl_patch_embed(strided) is False
