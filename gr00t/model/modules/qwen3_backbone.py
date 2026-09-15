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

import logging

import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


class PixelPatchNormalizer:
    """Rescale + normalize flattened Qwen-VL image patches on the GPU, bit-identical to the
    HF fast image processor doing it on the CPU.

    The processor fuses rescale and normalize into ``(x.float() - mean/rescale) / (std/rescale)``
    (``_fuse_mean_std_and_rescale_factor`` + torchvision ``normalize``: ``sub`` then ``div_`` in
    fp32) and only afterwards patchifies, a pure view/permute. Those elementwise ops commute with
    the reordering, so applying them per channel to the flattened ``(C * T * P * P)`` patch layout
    gives the same fp32 values element for element. The constants are built with the same
    expressions the processor uses.
    """

    def __init__(
        self,
        image_mean,
        image_std,
        rescale_factor: float,
        patch_size: int,
        temporal_patch_size: int,
    ):
        # Exactly transformers' _fuse_mean_std_and_rescale_factor (fp32 tensor * python float).
        mean = torch.tensor(image_mean) * (1.0 / rescale_factor)
        std = torch.tensor(image_std) * (1.0 / rescale_factor)
        per_channel = temporal_patch_size * patch_size * patch_size
        # Flattened patch order is (channel, temporal, patch_h, patch_w): channel-major.
        self.mean = mean.repeat_interleave(per_channel)
        self.std = std.repeat_interleave(per_channel)
        self._device_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def from_image_processor(cls, image_processor) -> "PixelPatchNormalizer":
        return cls(
            image_mean=image_processor.image_mean,
            image_std=image_processor.image_std,
            rescale_factor=image_processor.rescale_factor,
            patch_size=image_processor.patch_size,
            temporal_patch_size=image_processor.temporal_patch_size,
        )

    def __call__(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.shape[-1] != self.mean.numel():
            raise ValueError(
                f"pixel_values patch dim {patches.shape[-1]} != {self.mean.numel()} expected by the normalizer"
            )
        if patches.device not in self._device_cache:
            self._device_cache[patches.device] = (
                self.mean.to(patches.device),
                self.std.to(patches.device),
            )
        mean, std = self._device_cache[patches.device]
        return patches.to(torch.float32).sub(mean).div_(std)


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
        fast_vl_position_ids: bool = True,
        attn_implementation: str | None = None,
        fast_vl_patch_embed: bool = True,
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
            fast_vl_position_ids: replace Qwen3-VL's per-sample / per-image Python loops for
                M-RoPE position ids and vision position tables with batched, bitwise-identical
                implementations (gr00t.model.modules.qwen3_vl_fast_positions). Large batches
                are otherwise CPU-bound on those loops.
            attn_implementation: transformers attention implementation for the VLM. ``None``
                keeps the historical choice (``flash_attention_2`` if installed, else ``sdpa``).
                ``gr00t_fast`` (gr00t.model.modules.fast_attention) uses cuDNN/SDPA for regular
                batches and packed image segments -- ~2x faster than FA2 on Blackwell and traceable
                by torch.compile -- and FlashAttention varlen (FA4 if installed) for padded batches.
            fast_vl_patch_embed: run the vision patch embedding (a Conv3d whose kernel is the whole
                patch) as the equivalent F.linear; cuDNN's kernel for it is ~50x slower on Blackwell.
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if attn_implementation == "gr00t_fast":
            from gr00t.model.modules.fast_attention import register_gr00t_fast_attention

            register_gr00t_fast_attention()
            extra_kwargs["attn_implementation"] = "gr00t_fast"
        elif attn_implementation:
            extra_kwargs["attn_implementation"] = attn_implementation
        elif use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
            **transformers_loading_kwargs,
        ).eval()

        self.attn_implementation = self.model.config._attn_implementation
        if self.attn_implementation == "gr00t_fast":
            from gr00t.model.modules.fast_attention import patch_qwen3_vl_vision_attention

            n = patch_qwen3_vl_vision_attention(self.model.model.visual)
            logger.info(f"Qwen3-VL: gr00t_fast attention (patched {n} vision attention blocks)")

        if fast_vl_patch_embed:
            from gr00t.model.modules.qwen3_vl_fast_patch_embed import (
                apply_fast_qwen3_vl_patch_embed,
            )

            if apply_fast_qwen3_vl_patch_embed(self.model):
                logger.info("Qwen3-VL: patch embedding as F.linear (degenerate Conv3d)")

        if fast_vl_position_ids:
            from gr00t.model.modules.qwen3_vl_fast_positions import apply_fast_qwen3_vl_positions

            if apply_fast_qwen3_vl_positions(self.model):
                logger.info("Qwen3-VL: using batched position-id / vision position computations")
            else:
                logger.warning(
                    "Qwen3-VL: fast_vl_position_ids requested but model type unsupported"
                )

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        # forward() reads the last kept decoder layer's (pre-norm) output through this hook, so the
        # base Qwen3VLModel can be run without the lm_head (registered after the truncation above).
        self._last_layer_output: torch.Tensor | None = None
        self.model.model.language_model.layers[-1].register_forward_hook(self._capture_last_layer)
        # Set by Gr00tN1d7 from its image processor; used when the collator ships uint8 patches.
        self.pixel_patch_normalizer: PixelPatchNormalizer | None = None
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _capture_last_layer(self, module, inputs, output) -> None:
        self._last_layer_output = output[0] if isinstance(output, tuple) else output

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}
        if vl_input["pixel_values"].dtype == torch.uint8:
            # Collator emitted unnormalized patches (pixel_values_dtype="uint8"); do the
            # processor's fp32 rescale+normalize here instead, bit-identically.
            if self.pixel_patch_normalizer is None:
                raise RuntimeError(
                    "uint8 pixel_values need Qwen3Backbone.pixel_patch_normalizer (set by Gr00tN1d7)"
                )
            vl_input["pixel_values"] = self.pixel_patch_normalizer(vl_input["pixel_values"])
        if self.attn_implementation == "gr00t_fast":
            from gr00t.model.modules.fast_attention import (
                set_packed_segment_length,
                uniform_segment_length,
            )

            set_packed_segment_length(uniform_segment_length(vl_input["image_grid_thw"]))
        # Only the pre-norm output of the last kept decoder layer is used (== the causal-LM wrapper's
        # hidden_states[-1]). Capture it with a hook while running the base Qwen3VLModel, which skips
        # the lm_head entirely: even with logits_to_keep=1 the (B, 151k-vocab) GEMM cost ~90 ms per
        # 1024-sample step on B300 (cuBLAS has no good kernel for that shape).
        self._last_layer_output = None
        self.model.model(**vl_input)
        outputs, self._last_layer_output = self._last_layer_output, None
        if outputs is None:
            raise RuntimeError("Qwen3Backbone: last decoder layer output was not captured")
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
