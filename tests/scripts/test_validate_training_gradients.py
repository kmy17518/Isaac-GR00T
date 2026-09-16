# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from scripts.b1k.validate_training_gradients import (
    claim_output_directory,
    input_shapes,
    validate_records,
)
import torch


def records():
    return [
        {"event": "input", "step": 1, "shapes": {".inputs.attention_mask": [1024, 207]}},
        {"event": "metrics", "step": 1, "loss": 1.17, "grad_norm": 0.3},
        {"event": "input", "step": 2, "shapes": {".inputs.attention_mask": [1024, 214]}},
        {"event": "metrics", "step": 2, "loss": 1.16, "grad_norm": 0.4},
    ]


def test_valid_production_records():
    assert validate_records(records(), 2, 5.0, True) == {
        "steps": 2,
        "grad_norm_min": 0.3,
        "grad_norm_max": 0.4,
        "sequence_lengths": [207, 214],
    }


@pytest.mark.parametrize("value", [57.255, float("nan"), float("inf"), -1.0])
def test_corrupted_or_nonfinite_norm_rejected(value):
    rows = records()
    rows[-1]["grad_norm"] = value
    with pytest.raises(RuntimeError):
        validate_records(rows, 2, 5.0, True)


def test_missing_step_and_loss_rejected():
    with pytest.raises(RuntimeError, match="Expected gradient metrics"):
        validate_records(records()[:2], 2, 5.0, True)
    rows = records()
    rows[-1].pop("loss")
    with pytest.raises(RuntimeError, match="missing loss"):
        validate_records(rows, 2, 5.0, True)


def test_variable_lengths_are_required_unless_explicitly_disabled():
    rows = records()
    rows[2]["shapes"] = rows[0]["shapes"]
    with pytest.raises(RuntimeError, match="Variable sequence lengths"):
        validate_records(rows, 2, 5.0, True)
    assert validate_records(rows, 2, 5.0, False)["steps"] == 2


def test_fresh_output_allows_other_ranks_but_not_reuse(tmp_path):
    assert claim_output_directory(tmp_path, 0).exists()
    assert claim_output_directory(tmp_path, 1).exists()
    with pytest.raises(FileExistsError):
        claim_output_directory(tmp_path, 0)


@pytest.mark.parametrize(
    "filename", ["config.json", "model.safetensors", "validation-rank0-summary.json"]
)
def test_existing_output_is_never_overwritten(tmp_path, filename):
    existing = tmp_path / filename
    existing.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="fresh"):
        claim_output_directory(tmp_path, 0)
    assert existing.read_bytes() == b"preserve"


def test_input_shapes_does_not_copy_or_compute_tensor_values():
    batch = {"inputs": {"attention_mask": torch.empty(1024, 214, device="meta")}}
    assert input_shapes(batch) == {".inputs.attention_mask": [1024, 214]}


@pytest.mark.parametrize("failure", [None, "nonfinite", "large_gradient"])
def test_actual_trainer_checks_raw_loss_and_preserves_schedule(tmp_path, monkeypatch, failure):
    import gr00t.experiment.experiment as experiment
    from scripts.b1k import validate_training_gradients as validator
    from transformers import Trainer, TrainingArguments

    for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    seen = {}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(to_dict=lambda: {})

        def forward(self, attention_mask):
            scale = 57.25 if failure == "large_gradient" else 0.1
            return {"loss": self.weight * scale + (float("nan") if failure == "nonfinite" else 0.0)}

    def train(config):
        seen["config"] = config
        args = TrainingArguments(
            output_dir=config.training.output_dir,
            max_steps=config.training.max_steps,
            use_cpu=True,
            bf16=False,
            fp16=False,
            report_to="none",
            logging_steps=1,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=1,
            save_strategy="steps",
            save_steps=1000,
            remove_unused_columns=False,
            disable_tqdm=True,
        )
        trainer = experiment.Gr00tTrainer(
            model=TinyModel(), args=args, train_dataset=[{"attention_mask": torch.ones(3)}] * 10
        )
        trainer.train()
        trainer.save_model()
        seen["steps"] = trainer.state.global_step

    def entry(*a, **kw):
        config = SimpleNamespace(
            training=SimpleNamespace(
                resume_from_checkpoint=False,
                skip_weight_loading=False,
                max_steps=150000,
            )
        )
        experiment.run(config)

    monkeypatch.setattr(experiment, "run", train)
    monkeypatch.setattr(experiment, "Gr00tTrainer", Trainer)
    monkeypatch.setattr(validator.runpy, "run_path", entry)
    argv = ["--validation-dir", str(tmp_path), "--validation-steps", "2", "--allow-fixed-length"]
    if failure:
        message = "Nonfinite raw loss" if failure == "nonfinite" else "Invalid gradient norm"
        with pytest.raises(RuntimeError, match=message):
            validator.main(argv)
    else:
        validator.main(argv)
        assert seen["steps"] == 2
        assert seen["config"].training.max_steps == 150000
        assert not seen["config"].training.use_wandb
        assert not seen["config"].training.upload_checkpoints
    events = [
        json.loads(line) for line in (tmp_path / "validation-rank0.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == ("failed" if failure else "passed")
    assert not list(tmp_path.glob("*.safetensors"))
