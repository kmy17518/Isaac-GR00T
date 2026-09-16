# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

from dataclasses import dataclass
import os

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.data.b1k_prompts import PromptSource, language_key
from gr00t.data.dataset.lerobot_episode_loader import select_task_subset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.eval.eval_b1k_wrapper import load_modality_config
from gr00t.experiment.experiment import run
import tyro


@dataclass
class B1KFinetuneConfig(FinetuneConfig):
    """FinetuneConfig plus BEHAVIOR-1K specific options."""

    task_names: list[str] | None = None
    """Train only on these BEHAVIOR tasks (snake_case names as in ``meta/tasks.parquet``,
    e.g. ``--task-names turning_on_radio``). Works the same whether ``--dataset-path`` is
    the full 100-task ``2026-challenge-demos`` root or a per-task partial download of it:
    only the selected tasks' episodes are loaded, and their normalization statistics are
    computed over those episodes alone and kept under ``meta/task_subsets/<key>/`` (the
    dataset-wide ``meta/stats.json`` is left untouched). Unknown names, or a dataset that
    holds none of the selected tasks, fail fast. Default: every task under the root."""

    wandb_project: str = "B1K"
    """W&B project the run is logged to (``--wandb-project``); the run name is
    ``--experiment-name``. Defaults to the B1K project this script has always used."""

    collate_pixel_values_dtype: str | None = "uint8"
    """How the data collator emits ``pixel_values``. ``uint8`` (default): the unnormalized uint8
    patches, a quarter of the processor's float32 bytes and no float math in the worker; the
    backbone applies the processor's fp32 rescale+normalize on the GPU, bit-identically.
    ``bfloat16``: normalized patches cast to bf16 (also bit-identical under bf16 compute, half
    the bytes). ``None``: the processor's float32. Saved in the checkpoint's processor config."""
    backbone_attn_implementation: str | None = None
    """transformers attention implementation for the VLM backbone. ``None``: flash_attention_2 if
    installed, else sdpa. ``gr00t_fast``: cuDNN/SDPA for regular batches and packed image segments
    (~2x faster than FlashAttention-2 on Blackwell, traceable by torch.compile) and FlashAttention
    varlen -- FA4 (``pip install flash-attn-4``) if installed, else FA2 -- for padded batches."""
    sdpa_backend_priority: str | None = None
    """Global torch SDPA backend order, e.g. ``cudnn,efficient,flash,math``; torch's default puts
    cuDNN last, on Blackwell it is the fastest for the action head's attention. ``None``: default."""

    dataloader_prefetch_factor: int | None = None
    """Batches each dataloader worker keeps ready (PyTorch default 2). Each one is a full per-GPU
    batch in shared memory (~4.8 MB/sample), so 1 halves the dataloader's host-RAM footprint and
    lets you run more workers under a tight RAM limit."""

    compile_blocks: str | None = None
    """``torch.compile`` the repeated transformer blocks: comma-separated subset of
    ``vision,llm,dit,vlsa`` (see ``gr00t.model.modules.compile_blocks``). Fuses the elementwise
    work around the GEMMs; training-only and not bit-identical to eager. Multi-GPU DeepSpeed
    permits only ``dit`` or ``vlsa`` individually; combining them has a gradient corruption
    regression. Requires a compatible Inductor/Triton (B300: torch >= 2.9 cu130)."""

    compile_mode: str | None = None
    """``torch.compile`` mode for ``--compile-blocks`` (e.g. ``max-autotune-no-cudagraphs``)."""
    compile_coordinate_descent: bool = False
    """Inductor ``coordinate_descent_tuning`` for the compiled blocks: ~3-4 % faster steps on B300
    for about a minute more compile time per process (cached in ``TORCHINDUCTOR_CACHE_DIR``)."""
    compile_persistent_reductions: bool | None = None
    """Inductor ``triton.persistent_reductions``. ``False`` is required to compile the ``vlsa``
    blocks on Blackwell (their layer-norm backward otherwise needs more shared memory than exists)."""

    use_ddp: bool = False
    """Multi-GPU with plain PyTorch DDP instead of the default DeepSpeed ZeRO-2. Use it where
    DeepSpeed is unavailable (e.g. aarch64 hosts: ``pyproject.toml`` only pins it on x86_64).
    Every GPU then holds the full optimizer state, which the trainable action head (a fraction
    of the 3B model) keeps small. Ignored for ``--num-gpus 1``."""

    prompt_source: PromptSource | None = None
    """Which text prompt from the BEHAVIOR dataset (``meta/tasks.jsonl``) conditions the policy:
    ``task_description`` -- natural-language instruction, e.g. "Turn on the radio receiver that's on
    the table in the living room."; ``task_name`` -- snake_case identifier, e.g. "turning_on_radio".
    Overrides the language key declared by the modality config (``--modality-config-path``, e.g.
    ``examples/b1k/r1pro.py``, whose default is ``task_name``). Either way the chosen key is
    saved in the checkpoint's processor config, and ``serve_b1k.py`` reads it back from there."""


def select_prompt_source(modality_configs: dict, embodiment_tag: str, prompt_source: str) -> str:
    """Point ``modality_configs[embodiment_tag]["language"]`` at the ``prompt_source`` key.

    Returns the selected language key. Mutates the (shared) modality config dict in
    place so the dataset loader and the processor both see the same key.
    """
    key = language_key(prompt_source)
    if embodiment_tag not in modality_configs:
        raise ValueError(
            f"No modality config registered for embodiment tag '{embodiment_tag}'; pass "
            "--modality-config-path (e.g. examples/b1k/r1pro.py) so the B1K language keys exist."
        )
    previous = modality_configs[embodiment_tag]["language"]
    modality_configs[embodiment_tag]["language"] = ModalityConfig(
        delta_indices=list(previous.delta_indices),
        modality_keys=[key],
    )
    return key


def resolve_language_key(
    modality_configs: dict, embodiment_tag: str, prompt_source: str | None
) -> str:
    """Language key to train on: ``--prompt-source`` if given, else the modality config's own."""
    if prompt_source is not None:
        return select_prompt_source(modality_configs, embodiment_tag, prompt_source)
    return modality_configs[embodiment_tag]["language"].modality_keys[0]


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(B1KFinetuneConfig, description=__doc__)
    embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag).value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    # Optional task subset (--task-names): resolved eagerly against the dataset's
    # tasks table so a typo fails here, before the model is downloaded / loaded.
    task_names = sorted(set(ft_config.task_names)) if ft_config.task_names else None
    if task_names is not None:
        subset = select_task_subset(ft_config.dataset_path, task_names)
        print(
            f"Task subset {task_names}: {len(subset.episode_records)} episodes of "
            f"{ft_config.dataset_path} (task indices {sorted(subset.task_indices)})"
        )

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                        "task_names": task_names,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # Kind of B1K text prompt to train on (see gr00t.data.b1k_prompts): the modality
    # config's language key unless --prompt-source overrides it. The checkpoint's
    # processor config records the result for serve_b1k.py.
    selected_language_key = resolve_language_key(
        config.data.modality_configs, embodiment_tag, ft_config.prompt_source
    )
    print(f"Language key: {selected_language_key} (--prompt-source {ft_config.prompt_source})")

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params

    config.model.load_bf16 = False
    config.model.collate_pixel_values_dtype = ft_config.collate_pixel_values_dtype
    config.model.backbone_attn_implementation = ft_config.backbone_attn_implementation
    config.model.sdpa_backend_priority = ft_config.sdpa_backend_priority
    config.model.reproject_vision = False
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.dataloader_prefetch_factor = ft_config.dataloader_prefetch_factor
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = True
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project
    config.training.use_ddp = ft_config.use_ddp
    config.training.compile_blocks = ft_config.compile_blocks
    config.training.compile_coordinate_descent = ft_config.compile_coordinate_descent
    config.training.compile_persistent_reductions = ft_config.compile_persistent_reductions
    config.training.compile_mode = ft_config.compile_mode
    config.training.experiment_name = ft_config.experiment_name
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.decode_only_used_frames = ft_config.decode_only_used_frames

    run(config)
