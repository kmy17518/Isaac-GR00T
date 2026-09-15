# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.configs.base_config import get_default_config
from gr00t.configs.training.training_config import TrainingConfig
from gr00t.experiment import experiment
from gr00t.model.modules.compile_blocks import check_training_compile_compatibility
import pytest


@pytest.mark.parametrize(
    "targets", ["vision", "llm", "vision,llm,dit,vlsa", "dit,vision", "llm,vlsa", "dit,vlsa"]
)
def test_multigpu_deepspeed_rejects_unvalidated_backbone_compilation(targets):
    config = TrainingConfig(num_gpus=4, use_ddp=False, bf16=True, compile_blocks=targets)
    with pytest.raises(ValueError, match="DeepSpeed"):
        check_training_compile_compatibility(config)


@pytest.mark.parametrize("targets", [None, "", " ", "dit", "vlsa", " dit,dit "])
def test_production_validated_action_head_compilation_is_allowed(targets):
    config = TrainingConfig(num_gpus=4, use_ddp=False, compile_blocks=targets)
    check_training_compile_compatibility(config)


@pytest.mark.parametrize("num_gpus,use_ddp", [(1, False), (1, True), (4, True)])
def test_other_distributed_modes_keep_existing_compile_support(num_gpus, use_ddp):
    config = TrainingConfig(
        num_gpus=num_gpus, use_ddp=use_ddp, compile_blocks="vision,llm,dit,vlsa"
    )
    check_training_compile_compatibility(config)


@pytest.mark.parametrize("use_ddp", [False, True])
def test_unknown_compile_group_fails_early(use_ddp):
    config = TrainingConfig(num_gpus=4, use_ddp=use_ddp, compile_blocks="dti")
    with pytest.raises(ValueError, match="unknown compile targets"):
        check_training_compile_compatibility(config)


def test_experiment_rejects_unsafe_configuration_before_model_or_output_mutation(
    tmp_path, monkeypatch
):
    config = get_default_config()
    config.training = TrainingConfig(
        num_gpus=4, compile_blocks="vision,llm,dit,vlsa", output_dir=str(tmp_path / "run")
    )
    monkeypatch.setattr(experiment, "warn_configs", lambda config: None)
    monkeypatch.setattr(
        experiment, "setup_logging", lambda **kwargs: pytest.fail("started setup before validation")
    )
    with pytest.raises(ValueError, match="DeepSpeed"):
        experiment.run(config)
    assert not (tmp_path / "run").exists()
