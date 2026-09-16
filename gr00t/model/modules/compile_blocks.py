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

"""Training-only compilation of repeated transformer blocks without renaming parameters.

Supported groups are ``vision``, ``llm``, ``dit``, and ``vlsa``. Multi-task batches can change
sequence lengths, and compiled backward correctness depends on the distributed configuration.
The training entry point validates the supported DeepSpeed combinations before model setup.
Serving remains eager; compile settings are not part of the saved model configuration.
"""

from __future__ import annotations

import logging
import os

import torch


logger = logging.getLogger(__name__)

VALID_TARGETS = ("vision", "llm", "dit", "vlsa")


def check_training_compile_compatibility(training) -> None:
    """Reject DeepSpeed compile combinations without production gradient validation."""
    targets = {
        target.strip() for target in (training.compile_blocks or "").split(",") if target.strip()
    }
    unknown = targets - set(VALID_TARGETS)
    if unknown:
        raise ValueError(f"unknown compile targets {sorted(unknown)}; valid: {VALID_TARGETS}")
    if training.num_gpus > 1 and not training.use_ddp and targets not in (set(), {"dit"}, {"vlsa"}):
        raise ValueError(
            "Multi-GPU DeepSpeed supports only eager training, --compile-blocks dit, or "
            "--compile-blocks vlsa. Compiling dit and vlsa together produced corrupted "
            "gradients on real multi-task bf16 batches with torch 2.10/CUDA 13; other "
            "combinations have not passed production gradient validation. Remove "
            "--compile-blocks or select one supported group. Single-GPU and DDP "
            "configurations still require their own gradient validation."
        )


def _block_lists(model) -> dict[str, torch.nn.ModuleList | None]:
    backbone = getattr(
        getattr(model, "backbone", None), "model", None
    )  # Qwen3VLForConditionalGeneration
    vl = getattr(backbone, "model", None)  # Qwen3VLModel
    head = getattr(model, "action_head", None)
    dit = getattr(head, "model", None)
    vlsa = getattr(head, "vl_self_attention", None)
    return {
        "vision": getattr(getattr(vl, "visual", None), "blocks", None),
        "llm": getattr(getattr(vl, "language_model", None), "layers", None),
        "dit": getattr(dit, "transformer_blocks", None),
        "vlsa": getattr(vlsa, "transformer_blocks", None),
    }


def _exclude_hf_flash_attention_from_dynamo() -> None:
    """Make HF's flash-attention integration a graph break instead of a compile failure.

    ``transformers`` hands flash-attn a 0-d tensor for ``max_seqlen`` that the custom op casts to
    ``int`` implicitly in eager mode; under fake-tensor tracing that cast fails. Running just the
    attention call eagerly keeps rotary / norms / MLP of each block inside the compiled graphs.
    Idempotent.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:  # pragma: no cover - older transformers
        return
    for key in ("flash_attention_2", "flash_attention_3"):
        fn = ALL_ATTENTION_FUNCTIONS.get(key) if hasattr(ALL_ATTENTION_FUNCTIONS, "get") else None
        if fn is None or getattr(fn, "_gr00t_dynamo_disabled", False):
            continue
        wrapped = torch._dynamo.disable(fn, recursive=True)
        wrapped._gr00t_dynamo_disabled = True
        ALL_ATTENTION_FUNCTIONS[key] = wrapped


def compile_model_blocks(
    model,
    targets: str | list[str] = "vision,llm,dit,vlsa",
    mode: str | None = None,
    dynamic: bool | None = False,
    fullgraph: bool = False,
    coordinate_descent_tuning: bool = False,
    persistent_reductions: bool | None = None,
) -> dict[str, int]:
    """Wrap the ``forward`` of every block in the selected groups with ``torch.compile``.

    Args:
        model: a ``Gr00tN1d7`` instance (any device / dtype; compilation happens lazily at first call).
        targets: comma-separated subset of ``vision, llm, dit, vlsa``.
        mode: ``torch.compile`` mode (``None`` = default, ``"max-autotune-no-cudagraphs"``, ...).
        dynamic: passed to ``torch.compile``; ``False`` specialises on the training shapes.
        fullgraph: passed to ``torch.compile``.
        coordinate_descent_tuning: Inductor's per-kernel block-size search for the fused
            pointwise/reduction kernels (``torch._inductor.config.coordinate_descent_tuning``);
            ~3-4 % faster steps on B300 for ~1 min more compile time (cached afterwards).
        persistent_reductions: ``torch._inductor.config.triton.persistent_reductions``. Set it
            to ``False`` to compile the ``vlsa`` blocks on Blackwell: their layer-norm backward
            otherwise becomes a persistent-reduction kernel that needs more shared memory than the
            GPU has (``No valid triton configs ... OutOfMemoryError``). ``None`` leaves the default.

    Returns:
        ``{group: number of blocks compiled}``.
    """
    if isinstance(targets, str):
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    unknown = sorted(set(targets) - set(VALID_TARGETS))
    if unknown:
        raise ValueError(f"unknown compile targets {unknown}; valid: {VALID_TARGETS}")

    if {"vision", "llm"} & set(targets):
        _exclude_hf_flash_attention_from_dynamo()
    if coordinate_descent_tuning:
        torch._inductor.config.coordinate_descent_tuning = True
    if persistent_reductions is not None:
        torch._inductor.config.triton.persistent_reductions = persistent_reductions
        # Inductor's compile-worker subprocesses build their config from the environment.
        os.environ["TORCHINDUCTOR_PERSISTENT_REDUCTIONS"] = "1" if persistent_reductions else "0"

    lists = _block_lists(model)
    compiled: dict[str, int] = {}
    for group in targets:
        blocks = lists.get(group)
        if blocks is None:
            logger.warning(
                "compile_model_blocks: no '%s' blocks found on this model; skipping", group
            )
            continue
        n = 0
        for block in blocks:
            if getattr(block, "_gr00t_compiled", False):
                continue
            block.forward = torch.compile(
                block.forward, mode=mode, dynamic=dynamic, fullgraph=fullgraph
            )
            block._gr00t_compiled = True
            n += 1
        compiled[group] = n
    logger.info(
        "torch.compile applied to blocks: %s (mode=%s, dynamic=%s)", compiled, mode, dynamic
    )
    return compiled
