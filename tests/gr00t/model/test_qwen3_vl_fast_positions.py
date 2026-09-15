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

"""Bitwise equivalence of gr00t.model.modules.qwen3_vl_fast_positions with the stock Qwen3-VL code.

Tiny randomly initialised Qwen3-VL modules (CPU, no download) are driven with randomised token
layouts -- variable prompt lengths, 0..4 images per sample with mixed grids, video frames,
right/left/no padding -- and the batched implementations must return ``torch.equal`` tensors to
the original methods, including through a full ``Qwen3VLModel`` forward.
"""

import random

from gr00t.model.modules.qwen3_vl_fast_positions import (
    apply_fast_qwen3_vl_positions,
    batched_fast_pos_embed_interpolate,
    batched_get_rope_index,
    batched_rot_pos_emb,
    remove_fast_qwen3_vl_positions,
)
import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLVisionModel


IMAGE, VIDEO, VSTART, VEND, PAD = 151, 152, 153, 154, 155
MERGE = 2

# The stock implementation, kept so tests can compute the reference even while the class attribute
# is monkeypatched to make the batched implementation's fallback path fail loudly.
_ORIGINAL_GET_ROPE_INDEX = Qwen3VLModel.get_rope_index


def _vision_config() -> Qwen3VLVisionConfig:
    return Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=4,
        spatial_merge_size=MERGE,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=64,  # 8 x 8 learned grid -> exercises the bilinear resampling
        deepstack_visual_indexes=[0],
    )


def _config() -> Qwen3VLConfig:
    text = Qwen3VLTextConfig(
        vocab_size=160,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    return Qwen3VLConfig(
        text_config=text.to_dict(),
        vision_config=_vision_config().to_dict(),
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=VSTART,
        vision_end_token_id=VEND,
    )


@pytest.fixture(scope="module")
def vl_model() -> Qwen3VLModel:
    torch.manual_seed(0)
    return Qwen3VLModel(_config()).eval()


@pytest.fixture(scope="module")
def vision_model() -> Qwen3VLVisionModel:
    torch.manual_seed(0)
    return Qwen3VLVisionModel(_vision_config()).eval()


def _text(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(3, 150) for _ in range(n)]


def _random_layout(rng: random.Random, batch_size: int, padding: str):
    """Random batch: per sample text / image / video-frame blocks. Returns (ids, mask, img, vid)."""
    rows, image_grids, video_grids = [], [], []
    for _ in range(batch_size):
        row = _text(rng, rng.randrange(0, 6))
        for _ in range(rng.randrange(0, 4)):
            h, w = rng.choice([4, 6, 8]), rng.choice([4, 6, 8])
            if rng.random() < 0.75:  # image
                image_grids.append([1, h, w])
                row += [VSTART] + [IMAGE] * (h // MERGE * (w // MERGE)) + [VEND]
            else:  # video with t frames; Qwen3-VL emits one block per frame with text between
                t = rng.randrange(1, 4)
                video_grids.append([t, h, w])
                for _ in range(t):
                    row += _text(rng, rng.randrange(0, 3))
                    row += [VSTART] + [VIDEO] * (h // MERGE * (w // MERGE)) + [VEND]
            row += _text(rng, rng.randrange(0, 4))
        if not row:
            row = _text(rng, 1)
        rows.append(row)
    length = max(len(r) for r in rows)
    ids = torch.full((batch_size, length), PAD, dtype=torch.long)
    mask = torch.zeros((batch_size, length), dtype=torch.long)
    for i, row in enumerate(rows):
        if padding == "left":
            ids[i, length - len(row) :] = torch.tensor(row)
            mask[i, length - len(row) :] = 1
        else:
            ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
    img = torch.tensor(image_grids, dtype=torch.long) if image_grids else None
    vid = torch.tensor(video_grids, dtype=torch.long) if video_grids else None
    if padding == "none":
        return ids, None, img, vid
    return ids, mask, img, vid


def _assert_rope_equal(vl_model, ids, mask, img, vid):
    ref_pos, ref_delta = _ORIGINAL_GET_ROPE_INDEX(vl_model, ids, img, vid, mask)
    fast_pos, fast_delta = batched_get_rope_index(vl_model, ids, img, vid, mask)
    assert fast_pos.dtype == ref_pos.dtype and fast_pos.shape == ref_pos.shape
    assert torch.equal(fast_pos, ref_pos)
    assert fast_delta.shape == ref_delta.shape and torch.equal(fast_delta, ref_delta)


def _disable_fallback(monkeypatch):
    """Make the batched implementation's fallback fail, so a test proves the closed form itself."""

    def boom(*_args, **_kwargs):
        raise AssertionError("fallback to the original get_rope_index taken for a standard layout")

    monkeypatch.setattr(Qwen3VLModel, "get_rope_index", boom)


@pytest.mark.parametrize("padding", ["right", "left", "none"])
def test_get_rope_index_random_layouts(vl_model, padding, monkeypatch):
    _disable_fallback(monkeypatch)
    rng = random.Random(1234 + len(padding))
    checked = 0
    for _ in range(60):
        ids, mask, img, vid = _random_layout(rng, rng.randrange(1, 6), padding)
        if img is None and vid is None:
            continue
        _assert_rope_equal(vl_model, ids, mask, img, vid)
        checked += 1
    assert checked >= 40


def test_get_rope_index_unusual_layout_falls_back(vl_model):
    """Two image blocks with no separator between them: not the assumed layout -> original path."""
    h = w = 4
    n = h // MERGE * (w // MERGE)
    ids = torch.tensor([[1, 2, VSTART] + [IMAGE] * n + [IMAGE] * n + [VEND, 3]])
    img = torch.tensor([[1, h, w], [1, h, w]])
    _assert_rope_equal(vl_model, ids, torch.ones_like(ids), img, None)


def test_get_rope_index_text_only_and_mixed_batch(vl_model, monkeypatch):
    _disable_fallback(monkeypatch)
    ids = torch.tensor([[5, 6, 7, 8], [9, 10, PAD, PAD]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    _assert_rope_equal(vl_model, ids, mask, None, None)
    _assert_rope_equal(vl_model, ids, None, None, None)
    # one sample with an image, one without, in the same batch
    n = 4 // MERGE * (6 // MERGE)
    row0 = [1, VSTART] + [IMAGE] * n + [VEND, 2, 3]
    row1 = [4, 5, 6]
    length = len(row0)
    ids = torch.tensor([row0, row1 + [PAD] * (length - len(row1))])
    mask = (ids != PAD).long()
    _assert_rope_equal(vl_model, ids, mask, torch.tensor([[1, 4, 6]]), None)


def _random_grids(rng: random.Random, n: int) -> torch.Tensor:
    return torch.tensor(
        [[rng.choice([1, 1, 2]), rng.choice([4, 6, 8]), rng.choice([4, 6, 8])] for _ in range(n)],
        dtype=torch.long,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vision_position_tables_equal(vision_model, dtype):
    model = vision_model.to(dtype)
    rng = random.Random(99)
    grids = [_random_grids(rng, rng.randrange(1, 8)) for _ in range(25)]
    grids.append(torch.tensor([[1, 8, 8]] * 12))  # many identical images (the GR00T case)
    for grid in grids:
        ref_rot = Qwen3VLVisionModel.rot_pos_emb(model, grid)
        fast_rot = batched_rot_pos_emb(model, grid)
        assert torch.equal(fast_rot, ref_rot)
        ref_pe = Qwen3VLVisionModel.fast_pos_embed_interpolate(model, grid)
        fast_pe = batched_fast_pos_embed_interpolate(model, grid)
        assert fast_pe.dtype == ref_pe.dtype and torch.equal(fast_pe, ref_pe)
    model.to(torch.float32)


def test_full_forward_bitwise_equal_after_patch(vl_model):
    """End to end: patched vs stock Qwen3VLModel forward on a batch with two different prompts."""
    rng = random.Random(2024)
    cfg = _vision_config()
    while True:
        ids, mask, img, vid = _random_layout(rng, 3, "right")
        if img is not None and vid is None:
            break
    patches = int(img.prod(dim=1).sum())
    pixel_values = torch.randn(
        patches, cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2
    )

    remove_fast_qwen3_vl_positions(vl_model)
    with torch.no_grad():
        ref = vl_model(
            input_ids=ids, attention_mask=mask, pixel_values=pixel_values, image_grid_thw=img
        )
    assert apply_fast_qwen3_vl_positions(vl_model)
    assert vl_model.get_rope_index.__func__ is batched_get_rope_index
    with torch.no_grad():
        out = vl_model(
            input_ids=ids, attention_mask=mask, pixel_values=pixel_values, image_grid_thw=img
        )
    remove_fast_qwen3_vl_positions(vl_model)
    assert vl_model.get_rope_index.__func__ is Qwen3VLModel.get_rope_index

    assert torch.equal(out.last_hidden_state, ref.last_hidden_state)
    assert torch.equal(out.rope_deltas, ref.rope_deltas)


def test_apply_is_idempotent_and_rejects_other_models(vl_model):
    assert apply_fast_qwen3_vl_positions(vl_model)
    assert apply_fast_qwen3_vl_positions(vl_model)
    remove_fast_qwen3_vl_positions(vl_model)
    assert not apply_fast_qwen3_vl_positions(torch.nn.Linear(2, 2))
