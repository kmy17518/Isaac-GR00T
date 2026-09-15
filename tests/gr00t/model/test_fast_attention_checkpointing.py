# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
import copy
from functools import partial

from gr00t.model.modules import fast_attention as fa
import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _varlen_reference(q, k, v, cu_q, cu_k, max_q, max_k, causal, scale, dropout):
    outputs = []
    for qs, qe, ks, ke in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
        tensors = [q[qs:qe], k[ks:ke], v[ks:ke]]
        out = F.scaled_dot_product_attention(
            *(x.transpose(0, 1).unsqueeze(0) for x in tensors),
            is_causal=causal,
            scale=scale,
            dropout_p=dropout,
        )
        outputs.append(out[0].transpose(0, 1))
    return torch.cat(outputs)


@pytest.mark.parametrize(
    "device,compiled",
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=pytest.mark.gpu),
        pytest.param("cuda", True, marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("reentrant", [None, False, True])
def test_segment_metadata_survives_overlapping_checkpointed_forwards(
    monkeypatch, reentrant, device, compiled
):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(18)
    config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=128,
        intermediate_size=256,
        num_heads=2,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=32,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0],
    )
    config._attn_implementation = "sdpa"
    reference = Qwen3VLVisionModel(config).to(device).train()
    actual = copy.deepcopy(reference)
    parameter_ids = {name: id(parameter) for name, parameter in actual.named_parameters()}
    state_keys = set(actual.state_dict())
    assert fa.patch_qwen3_vl_vision_attention(actual) == 2
    assert fa.patch_qwen3_vl_vision_attention(actual) == 2
    assert parameter_ids == {name: id(parameter) for name, parameter in actual.named_parameters()}
    assert state_keys == set(actual.state_dict())
    if device == "cpu":
        monkeypatch.setattr(fa, "_varlen_attention", _varlen_reference)
    if reentrant is not None:
        for model in (reference, actual):
            for block in model.blocks:
                block.gradient_checkpointing = True
                block._gradient_checkpointing_func = partial(checkpoint, use_reentrant=reentrant)
    if compiled:
        for block in actual.blocks:
            block.forward = torch.compile(block.forward, dynamic=False)
    reference_losses, actual_losses = [], []
    for grid in (torch.tensor([[1, 4, 4], [1, 4, 12]]), torch.tensor([[1, 4, 8], [1, 4, 8]])):
        grid = grid.to(device)
        pixels = torch.randn(64, 96, device=device)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
        )
        with context:
            expected, _ = reference(pixels, grid)
            result, _ = actual(pixels, grid)
        torch.testing.assert_close(
            result,
            expected,
            atol=0.004 if device == "cuda" else 2e-6,
            rtol=0.04 if device == "cuda" else 2e-5,
        )
        upstream = torch.randn_like(result)
        reference_losses.append((expected * upstream).sum())
        actual_losses.append((result * upstream).sum())
    sum(reference_losses).backward()
    sum(actual_losses).backward()
    for (name, expected), (_, result) in zip(
        reference.named_parameters(), actual.named_parameters()
    ):
        if expected.grad is None:
            assert result.grad is None, name
        else:
            assert result.grad is not None, name
            assert torch.isfinite(result.grad).all(), name
            if device == "cpu":
                torch.testing.assert_close(
                    result.grad, expected.grad, atol=2e-5, rtol=2e-4, msg=name
                )
    if device == "cuda":
        expected = torch.cat(
            [p.grad.flatten() for p in reference.parameters() if p.grad is not None]
        )
        result = torch.cat([p.grad.flatten() for p in actual.parameters() if p.grad is not None])
        relative_error = (result - expected).norm() / expected.norm()
        assert relative_error < 0.02, relative_error.item()
