# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7 import gr00t_n1d7 as gm, processing_gr00t_n1d7 as pm
from gr00t.model.modules.qwen3_backbone import PixelPatchNormalizer, Qwen3Backbone
import pytest
import torch


class TinyPart(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.pixel_patch_normalizer = None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_pretrained_dtype_preserves_normalization(tmp_path, monkeypatch, dtype):
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="left"),
        image_processor=SimpleNamespace(
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            rescale_factor=1 / 255,
            patch_size=4,
            temporal_patch_size=2,
        ),
    )
    monkeypatch.setattr(gm, "get_backbone_cls", lambda _: TinyPart)
    monkeypatch.setattr(gm, "Gr00tN1d7ActionHead", TinyPart)
    monkeypatch.setattr(pm, "build_processor", lambda *a, **kw: processor)
    original = gm.Gr00tN1d7(Gr00tN1d7Config(collate_pixel_values_dtype="uint8"))
    original.save_pretrained(tmp_path)
    loaded = gm.Gr00tN1d7.from_pretrained(tmp_path, dtype=dtype)
    raw = torch.arange(256, dtype=torch.uint8)[:, None].expand(-1, 96)
    expected = original.backbone.pixel_patch_normalizer(raw)
    actual = loaded.backbone.pixel_patch_normalizer(raw)
    assert loaded.backbone.pixel_patch_normalizer.mean.dtype == torch.float32
    assert torch.equal(expected, actual)


class ReachedVision(Exception):
    pass


@pytest.fixture
def trt_module(monkeypatch):
    stub = ModuleType("trt_torch")
    stub.Engine = object
    monkeypatch.setitem(sys.modules, "trt_torch", stub)
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "test_trt_forward_normalization", root / "scripts/deployment/trt_model_forward.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["eager", "vit_trt", "llm_trt", "full_trt"])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.bfloat16])
def test_normalized_vision_inputs_in_every_backbone_mode(trt_module, mode, dtype):
    seen = []
    raw = torch.full((4, 96), 128, dtype=torch.uint8)
    normalizer = PixelPatchNormalizer([0.5] * 3, [0.5] * 3, 1 / 255, 4, 2)
    pixels = raw if dtype == torch.uint8 else normalizer(raw).to(dtype)
    expected = normalizer(raw) if dtype == torch.uint8 else pixels

    def capture(values, *args, **kwargs):
        seen.append(values)
        raise ReachedVision

    class InnerModel:
        get_image_features = staticmethod(capture)

        def __call__(self, **kwargs):
            capture(kwargs["pixel_values"])

    class Engine:
        def dtype_of(self, name):
            return torch.float32

        def set_runtime_tensor_shape(self, *args):
            pass

        def __call__(self, values):
            return capture(values)

    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)
    backbone.model = SimpleNamespace(model=InnerModel())
    backbone.vit_engine = Engine()
    backbone.pixel_patch_normalizer = normalizer
    backbone.set_frozen_modules_to_eval_mode = lambda: None
    backbone.attn_implementation = "sdpa"
    inputs = dict(
        pixel_values=pixels,
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        input_ids=torch.tensor([[151]]),
        attention_mask=torch.ones(1, 1),
    )
    forward = {
        "eager": Qwen3Backbone.forward,
        "vit_trt": trt_module.qwen3_backbone_tensorrt_forward,
        "llm_trt": trt_module.qwen3_backbone_llm_trt_forward,
        "full_trt": trt_module.qwen3_backbone_full_trt_forward,
    }[mode]
    with pytest.raises(ReachedVision):
        forward(backbone, inputs)
    assert torch.equal(seen[0].float(), expected.float())
    assert inputs["pixel_values"] is pixels


def test_normalize_pixel_value_lists_without_double_normalization():
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)
    backbone.pixel_patch_normalizer = PixelPatchNormalizer([0.5] * 3, [0.5] * 3, 1 / 255, 4, 2)
    patches = torch.randint(0, 256, (8, 96), dtype=torch.uint8)
    expected = backbone.pixel_patch_normalizer(patches)
    assert torch.equal(backbone.normalize_pixel_values([patches[:4], patches[4:]]), expected)
    assert backbone.normalize_pixel_values(expected) is expected
    assert torch.equal(backbone.normalize_pixel_values((expected[:4], expected[4:])), expected)


@pytest.mark.parametrize("mode", ["eager", "vit_trt", "llm_trt", "full_trt"])
def test_uint8_without_normalizer_fails_before_vision(trt_module, mode):
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)
    backbone.pixel_patch_normalizer = None
    backbone.set_frozen_modules_to_eval_mode = lambda: None
    inputs = dict(
        pixel_values=torch.ones(4, 96, dtype=torch.uint8),
        image_grid_thw=None,
        input_ids=None,
        attention_mask=None,
    )
    forward = {
        "eager": Qwen3Backbone.forward,
        "vit_trt": trt_module.qwen3_backbone_tensorrt_forward,
        "llm_trt": trt_module.qwen3_backbone_llm_trt_forward,
        "full_trt": trt_module.qwen3_backbone_full_trt_forward,
    }[mode]
    with pytest.raises(RuntimeError, match="normalizer"):
        forward(backbone, inputs)
