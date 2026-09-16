# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gr00t.model.modules import fast_attention as fa
import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_cuda_gqa_forward_backward_and_padding(padded, head_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(27)
    batch, length, heads, kv_heads = 3, 32, 4, 2
    values = [
        torch.randn(
            batch, h, length, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for h in (heads, kv_heads, kv_heads)
    ]
    reference = [value.detach().float().requires_grad_() for value in values]
    mask = torch.ones(batch, length, dtype=torch.bool, device="cuda")
    if padded:
        mask[1, 20:] = False
        mask[2, :8] = False
    actual, _ = fa.gr00t_fast_attention_forward(
        SimpleNamespace(training=True, is_causal=True),
        *values,
        attention_mask=mask if padded else None,
    )
    expected = torch.zeros_like(actual, dtype=torch.float32)
    for index in range(batch):
        valid = mask[index]
        query, key, value = [x[index : index + 1, :, valid] for x in reference]
        output = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(2, dim=1),
            value.repeat_interleave(2, dim=1),
            is_causal=True,
        )
        expected[index, valid] = output[0].transpose(0, 1)
    assert (actual.float() - expected).norm() / expected.norm() < 0.015
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream.float())
    for actual_value, expected_value in zip(values, reference):
        assert torch.isfinite(actual_value.grad).all()
        relative_error = (
            actual_value.grad.float() - expected_value.grad
        ).norm() / expected_value.grad.norm()
        assert relative_error < 0.02, relative_error.item()
        assert torch.count_nonzero(actual_value.grad.transpose(1, 2)[~mask]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_patch_normalization_matches_processor(dtype):
    from gr00t.model.modules.qwen3_backbone import PixelPatchNormalizer
    from transformers import Qwen2VLImageProcessorFast

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(71)
    processor = Qwen2VLImageProcessorFast(patch_size=4, temporal_patch_size=2, merge_size=2)
    image = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
    normalized = processor(images=image, do_resize=False, return_tensors="pt")["pixel_values"]
    raw = processor(
        images=image, do_resize=False, do_rescale=False, do_normalize=False, return_tensors="pt"
    )["pixel_values"].to(torch.uint8)
    actual = PixelPatchNormalizer.from_image_processor(processor)(raw.cuda()).to(dtype)
    assert torch.equal(actual.cpu(), normalized.to(dtype))
