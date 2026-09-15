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

"""Batched replacements for the per-sample / per-image Python loops in HF Qwen3-VL.

Three methods of the stock ``transformers`` Qwen3-VL implementation loop in Python over every
sample and every image of a batch, each iteration issuing a handful of tiny kernels and
``.item()`` device syncs:

* ``Qwen3VLModel.get_rope_index``               -- M-RoPE (t, h, w) position ids of every token
* ``Qwen3VLVisionModel.rot_pos_emb``            -- 2-D rotary tables of every vision patch
* ``Qwen3VLVisionModel.fast_pos_embed_interpolate`` -- bilinearly resampled learned position
  embeddings of every vision patch

On a GR00T training batch of 1024 samples x 3 images that is ~55k ``.item()`` calls and ~6 s of
CPU time per step while the GPU idles (about 90 % of the forward+backward wall time).

The functions below compute the *same tensors, bit for bit* with batched ops:

* position ids are derived from the token layout with cumulative sums / ``cummax`` instead of a
  per-sample loop, for any mixture of text, images and video frames, any prompt length per
  sample and either padding side. If a batch does not have the layout these formulas assume
  (each image/video block is one contiguous run of placeholder tokens announced by a
  ``vision_start`` token and exactly ``t*h*w/merge^2`` long -- true for anything produced by the
  Qwen3-VL processor), the original implementation is used for that call, so the result is
  identical by construction;
* the two vision tables are computed once per *distinct* ``(t, h, w)`` grid and broadcast to all
  images that share it, instead of once per image.

They are installed on a loaded model instance by :func:`apply_fast_qwen3_vl_positions`
(``Qwen3Backbone`` does this when ``fast_vl_position_ids`` is set) and verified against the
originals in ``tests/gr00t/model/test_qwen3_vl_fast_positions.py``.
"""

from __future__ import annotations

import logging
import types

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Qwen3VLModel.get_rope_index
# --------------------------------------------------------------------------------------------------


def _text_only_rope_index(input_ids, attention_mask):
    """Verbatim ``else`` branch of ``Qwen3VLModel.get_rope_index`` (no vision inputs)."""
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype
        )
    return position_ids, mrope_position_deltas


def _spread_forward(values: torch.Tensor, at: torch.Tensor, fill: int) -> torch.Tensor:
    """Carry ``values`` at positions ``at`` (bool) forward along the last dim.

    ``values`` must be non-decreasing along the last dim at the ``at`` positions (true for the
    monotone quantities used here), so a running max reproduces "value of the most recent marker".
    Positions before the first marker read ``fill``.
    """
    marked = torch.where(at, values, torch.full_like(values, fill))
    return torch.cummax(marked, dim=-1).values


def batched_get_rope_index(
    self,
    input_ids: torch.LongTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for ``Qwen3VLModel.get_rope_index`` without the per-sample Python loop.

    Semantics (identical to the original): text tokens advance all three position components by
    one; the tokens of a vision block with LLM grid ``(t, h, w)`` get ``(base + ti, base + hi,
    base + wi)`` where ``base`` is the position the block's first token would have had as text,
    and the text after the block resumes at ``base + max(t, h, w)``. Padded (masked) positions
    hold 1. ``rope_deltas[b] = max position + 1 - sequence length``.
    """
    if video_grid_thw is not None:
        # Qwen3-VL separates video frames with timestamp text, so each frame is its own block.
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is None or (image_grid_thw is None and video_grid_thw is None):
        return _text_only_rope_index(input_ids, attention_mask)

    config = self.config
    merge = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id

    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    if attention_mask is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        valid = attention_mask.to(device) == 1

    is_img = (input_ids == image_token_id) & valid
    is_vid = (input_ids == video_token_id) & valid
    is_vis = is_img | is_vid

    # One block = one contiguous run of placeholder tokens. Its first token marks the block.
    prev_vis = F.pad(is_vis[:, :-1], (1, 0), value=False)
    block_start = is_vis & ~prev_vis
    flat_start_idx = block_start.reshape(-1).nonzero().squeeze(1)  # row-major == HF's visit order
    n_blocks = int(flat_start_idx.numel())
    start_is_img = is_img.reshape(-1)[flat_start_idx]
    n_img_blocks = int(start_is_img.sum())
    n_vid_blocks = n_blocks - n_img_blocks

    # --- exactness guard: the layout must be the one the closed form assumes -------------------
    # (a) the original counts blocks via ``vision_start`` tokens followed by a placeholder token
    #     and consumes exactly one grid row per block, in order.
    is_start_tok = (input_ids == vision_start_token_id) & valid
    next_is_img = F.pad(is_img[:, 1:], (0, 1), value=False)
    next_is_vid = F.pad(is_vid[:, 1:], (0, 1), value=False)
    announced = ((is_start_tok & next_is_img).sum(-1), (is_start_tok & next_is_vid).sum(-1))
    per_row_img = (block_start & is_img).sum(-1)
    per_row_vid = (block_start & is_vid).sum(-1)
    n_img_grids = 0 if image_grid_thw is None else int(image_grid_thw.shape[0])
    n_vid_grids = 0 if video_grid_thw is None else int(video_grid_thw.shape[0])
    layout_ok = (
        n_img_blocks == n_img_grids
        and n_vid_blocks == n_vid_grids
        and bool(torch.equal(announced[0], per_row_img))
        and bool(torch.equal(announced[1], per_row_vid))
    )
    if layout_ok and n_blocks > 0:
        grids = torch.empty((n_blocks, 3), dtype=torch.long, device=device)
        if n_img_blocks:
            grids[start_is_img] = image_grid_thw.to(device=device, dtype=torch.long)
        if n_vid_blocks:
            grids[~start_is_img] = video_grid_thw.to(device=device, dtype=torch.long)
        llm_t = grids[:, 0]
        llm_h = grids[:, 1] // merge
        llm_w = grids[:, 2] // merge
        block_len = llm_t * llm_h * llm_w

        # (b) every run must be exactly as long as its grid says.
        block_id = torch.full((batch_size * seq_len,), -1, dtype=torch.long, device=device)
        block_id[flat_start_idx] = torch.arange(n_blocks, device=device)
        block_id = _spread_forward(block_id.view(batch_size, seq_len), block_start, -1)
        run_len = torch.bincount(block_id[is_vis], minlength=n_blocks)
        layout_ok = bool(torch.equal(run_len, block_len))

    if not layout_ok:
        logger.debug("Qwen3-VL batch layout not recognised; using the original get_rope_index")
        # ``video_grid_thw`` is already one-row-per-frame with t == 1; the original's own
        # expansion is the identity on it.
        return type(self).get_rope_index(
            self, input_ids, image_grid_thw, video_grid_thw, attention_mask
        )

    # --- closed form ---------------------------------------------------------------------------
    # Each valid text token advances by 1; each block advances (once, booked at its first token)
    # by max(t, h, w). The exclusive cumulative sum is the position a token starts from.
    advance = (valid & ~is_vis).long()
    if n_blocks > 0:
        advance.view(-1)[flat_start_idx] += torch.maximum(torch.maximum(llm_t, llm_h), llm_w)
    excl = advance.cumsum(-1) - advance  # (B, L) position of the token if it were text

    if n_blocks > 0:
        positions = torch.arange(seq_len, device=device).expand(batch_size, seq_len)
        block_base = _spread_forward(excl, block_start, 0)  # base position of the current block
        block_first = _spread_forward(positions, block_start, 0)  # index of its first token
        idx_in_block = positions - block_first
        g = grids[block_id.clamp(min=0)]  # (B, L, 3): grid of the block a token belongs to
        h_ = g[..., 1] // merge
        w_ = g[..., 2] // merge
        # Original ordering: t slowest, then h, then w (arange(t).view(-1,1).expand(-1,h*w), ...).
        ti = idx_in_block // (h_ * w_)
        hi = (idx_in_block // w_) % h_
        wi = idx_in_block % w_
        pos_t = torch.where(is_vis, block_base + ti, excl)
        pos_h = torch.where(is_vis, block_base + hi, excl)
        pos_w = torch.where(is_vis, block_base + wi, excl)
        position_ids = torch.stack([pos_t, pos_h, pos_w], dim=0)
    else:
        position_ids = excl.unsqueeze(0).expand(3, -1, -1).clone()

    position_ids = position_ids.to(input_ids.dtype)
    position_ids.masked_fill_(~valid.unsqueeze(0), 1)  # padded positions read 1, as in the original

    max_pos = position_ids.masked_fill(~valid.unsqueeze(0), -1).amax(dim=(0, 2))
    mrope_position_deltas = (max_pos + 1 - seq_len).to(torch.long).unsqueeze(1)
    return position_ids, mrope_position_deltas


# --------------------------------------------------------------------------------------------------
# Qwen3VLVisionModel.rot_pos_emb / fast_pos_embed_interpolate
# --------------------------------------------------------------------------------------------------


def _unique_grids(grid_thw: torch.Tensor):
    """Distinct (t, h, w) rows, the row->distinct index map, and each image's first token offset.

    Done on the CPU copy of the (tiny) grid tensor: one device sync instead of one per image.
    """
    grid_cpu = grid_thw.detach().to("cpu", torch.long)
    uniq, inverse = torch.unique(grid_cpu, dim=0, return_inverse=True)
    n_tokens = grid_cpu.prod(dim=1)
    offsets = F.pad(n_tokens.cumsum(0), (1, 0))[:-1]
    return uniq.tolist(), inverse, offsets, int(n_tokens.sum())


def _scatter_per_grid(out: torch.Tensor, per_grid, uniq, inverse, offsets) -> torch.Tensor:
    """Fill ``out`` (tokens-first) with ``per_grid(t, h, w)`` for every image of each distinct grid."""
    for u, (t, h, w) in enumerate(uniq):
        chunk = per_grid(t, h, w)  # (t*h*w, ...) exactly what the original computes per image
        images = (inverse == u).nonzero().squeeze(1)
        token_idx = offsets[images][:, None] + torch.arange(chunk.shape[0])[None, :]
        out[token_idx.reshape(-1).to(out.device)] = chunk.repeat(
            images.numel(), *([1] * (chunk.dim() - 1))
        )
    return out


def batched_rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """Drop-in for ``Qwen3VLVisionModel.rot_pos_emb``: per distinct grid instead of per image."""
    merge_size = self.spatial_merge_size
    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
    device = freq_table.device

    uniq, inverse, offsets, total_tokens = _unique_grids(grid_thw)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    def coords_for(num_frames: int, height: int, width: int) -> torch.Tensor:
        # Same ops as the original's loop body.
        merged_h, merged_w = height // merge_size, width // merge_size
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)
        return coords

    pos_ids = _scatter_per_grid(pos_ids, coords_for, uniq, inverse, offsets)
    embeddings = freq_table[pos_ids]
    return embeddings.flatten(1)


def batched_fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """Drop-in for ``Qwen3VLVisionModel.fast_pos_embed_interpolate``: per distinct grid."""
    weight = self.pos_embed.weight
    num_grid_per_side = self.num_grid_per_side
    merge_size = self.config.spatial_merge_size

    uniq, inverse, offsets, total_tokens = _unique_grids(grid_thw)
    out = torch.empty((total_tokens, weight.shape[1]), dtype=weight.dtype, device=weight.device)

    def embeds_for(t: int, h: int, w: int) -> torch.Tensor:
        # Same ops as the original's loop bodies (CPU float32 interpolation math, Python-list
        # round trip into the embedding dtype, gather * weight, sum of the 4 corners, then the
        # frame repeat + merge-window permutation), applied to one grid.
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w)
        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)
        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor
        base_h = h_idxs_floor * num_grid_per_side
        base_h_ceil = h_idxs_ceil * num_grid_per_side
        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        idx_tensor = torch.tensor(
            [i.tolist() for i in indices], dtype=torch.long, device=weight.device
        )
        weight_tensor = torch.tensor(
            [wt.tolist() for wt in weights], dtype=weight.dtype, device=weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        pos_embed = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]  # (h*w, D)
        pos_embed = pos_embed.repeat(t, 1)
        return (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )

    return _scatter_per_grid(out, embeds_for, uniq, inverse, offsets)


# --------------------------------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------------------------------

_PATCHED_FLAG = "_gr00t_fast_positions"


def apply_fast_qwen3_vl_positions(model: torch.nn.Module) -> bool:
    """Install the batched methods on a loaded ``Qwen3VLForConditionalGeneration`` / ``Qwen3VLModel``.

    Instance-level overrides (no subclassing), so ``from_pretrained`` / ``save_pretrained`` and
    the state dict are untouched. Returns ``False`` (and changes nothing) for other models.
    """
    vl_model = model.model if type(model).__name__ == "Qwen3VLForConditionalGeneration" else model
    visual = getattr(vl_model, "visual", None)
    if type(vl_model).__name__ != "Qwen3VLModel" or visual is None:
        return False
    if getattr(vl_model, _PATCHED_FLAG, False):
        return True
    vl_model.get_rope_index = types.MethodType(batched_get_rope_index, vl_model)
    visual.rot_pos_emb = types.MethodType(batched_rot_pos_emb, visual)
    visual.fast_pos_embed_interpolate = types.MethodType(batched_fast_pos_embed_interpolate, visual)
    setattr(vl_model, _PATCHED_FLAG, True)
    return True


def remove_fast_qwen3_vl_positions(model: torch.nn.Module) -> None:
    """Undo :func:`apply_fast_qwen3_vl_positions` (restores the class methods)."""
    vl_model = model.model if type(model).__name__ == "Qwen3VLForConditionalGeneration" else model
    for obj, name in (
        (vl_model, "get_rope_index"),
        (getattr(vl_model, "visual", None), "rot_pos_emb"),
        (getattr(vl_model, "visual", None), "fast_pos_embed_interpolate"),
    ):
        if obj is not None and name in obj.__dict__:
            del obj.__dict__[name]
    if _PATCHED_FLAG in vl_model.__dict__:
        del vl_model.__dict__[_PATCHED_FLAG]
