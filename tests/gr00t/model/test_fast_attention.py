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

"""CPU tests for the linear patch embedding (and, later, gr00t_fast attention)."""

import types

import torch


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
