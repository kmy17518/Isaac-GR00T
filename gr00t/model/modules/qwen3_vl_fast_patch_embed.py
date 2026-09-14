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

"""Qwen3-VL patch embedding as the linear layer it is.

``Qwen3VLVisionPatchEmbed`` applies a ``Conv3d`` whose kernel and stride equal the whole
``(temporal_patch_size, patch_size, patch_size)`` input, i.e. one output position per patch:
``out[n, o] = sum_{c,t,i,j} x[n,c,t,i,j] * w[o,c,t,i,j] + b[o]`` -- exactly ``F.linear`` on the
flattened patch. cuDNN has no good kernel for that degenerate convolution: on B300 with 786k
patches per step it ran a generic ``sm80`` implicit-GEMM plus a layout transform for ~110 ms, where
the equivalent cuBLAS GEMM takes ~2 ms. Same math, fp32-accumulated either way; results differ
only by bf16 accumulation order.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _linear_patch_embed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    weight = self.proj.weight
    x = hidden_states.to(weight.dtype).reshape(hidden_states.shape[0], -1)  # (N, C*T*P*P)
    return F.linear(x, weight.reshape(weight.shape[0], -1), self.proj.bias)


def apply_fast_qwen3_vl_patch_embed(model: torch.nn.Module) -> bool:
    """Bind the linear forward to the vision patch embedding of a Qwen3-VL model (instance-level).
    Returns False when the module layout is not the expected degenerate convolution."""
    patch_embed = getattr(
        getattr(getattr(model, "model", model), "visual", None), "patch_embed", None
    )
    proj = getattr(patch_embed, "proj", None)
    if not isinstance(proj, torch.nn.Conv3d):
        return False
    if tuple(proj.kernel_size) != tuple(proj.stride) or any(p != 0 for p in proj.padding):
        return False
    patch_embed.forward = types.MethodType(_linear_patch_embed_forward, patch_embed)
    return True
