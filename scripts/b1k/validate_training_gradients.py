#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a bounded production training check without changing its learning-rate schedule.

Pass the normal train_b1k.py arguments after the validation options. All run outputs
are redirected to --validation-dir; remote reporting/uploads and resume are disabled.
Normalization caches must already exist and remain unchanged. This executes optimizer
updates on a disposable model, not on an existing training run. A passing short check
is not a convergence guarantee or a compiled/eager gradient-equivalence comparison.
"""

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]


def input_shapes(value, prefix="") -> dict:
    if hasattr(value, "shape"):
        return {prefix: list(value.shape)}
    result = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(input_shapes(child, f"{prefix}.{key}"))
    return result


def claim_output_directory(output: Path, rank: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    allowed = re.compile(r"validation-rank\d+\.jsonl$")
    if any(not allowed.fullmatch(entry.name) for entry in output.iterdir()):
        raise ValueError("Use a fresh --validation-dir, not an existing training output")
    record_file = output / f"validation-rank{rank}.jsonl"
    with record_file.open("x"):
        pass
    return record_file


def validate_records(records, steps, max_grad_norm, require_variable_length):
    metrics = [row for row in records if row["event"] == "metrics" and "grad_norm" in row]
    observed = {row["step"] for row in metrics}
    if observed != set(range(1, steps + 1)):
        raise RuntimeError(
            f"Expected gradient metrics for steps 1..{steps}, got {sorted(observed)}"
        )
    for row in metrics:
        for key in ("loss", "grad_norm"):
            if key not in row or not math.isfinite(row[key]):
                raise RuntimeError(f"Nonfinite or missing {key} at step {row['step']}")
        if row["grad_norm"] < 0 or row["grad_norm"] > max_grad_norm:
            raise RuntimeError(
                f"Gradient norm {row['grad_norm']} exceeds allowed range at step {row['step']}"
            )
    lengths = {
        shape[-1]
        for row in records
        if row["event"] == "input"
        for name, shape in row["shapes"].items()
        if name.endswith(".attention_mask")
    }
    if require_variable_length and len(lengths) < 2:
        raise RuntimeError(f"Variable sequence lengths were not exercised: {sorted(lengths)}")
    return {
        "steps": steps,
        "grad_norm_min": min(row["grad_norm"] for row in metrics),
        "grad_norm_max": max(row["grad_norm"] for row in metrics),
        "sequence_lengths": sorted(lengths),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--max-observed-grad-norm", type=float, default=5.0)
    parser.add_argument("--allow-fixed-length", action="store_true")
    parser.add_argument("--save-validation-checkpoint", action="store_true")
    parser.add_argument("--validation-help", action="help")
    args, training_args = parser.parse_known_args(argv)
    if (
        args.validation_steps < 2
        or not math.isfinite(args.max_observed_grad_norm)
        or args.max_observed_grad_norm <= 0
    ):
        parser.error(
            "validation-steps must be >=2 and max-observed-grad-norm must be finite and positive"
        )
    rank = int(os.environ.get("RANK", "0"))
    output = args.validation_dir.resolve()
    record_file = claim_output_directory(output, rank)
    records = []

    def emit(event, **values):
        record = {"event": event, **values}
        records.append(record)
        with record_file.open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")

    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(REPO))
    import gr00t.data.stats as stats
    import gr00t.experiment.experiment as experiment
    import torch
    from transformers import TrainerCallback

    original_run = experiment.run
    trainer_class = experiment.Gr00tTrainer

    class ValidationCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            finished = state.global_step >= validation_steps
            control.should_save = finished and save_checkpoint
            control.should_training_stop = finished
            return control

    validation_steps = args.validation_steps
    save_checkpoint = args.save_validation_checkpoint

    class ValidationTrainer(trainer_class):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.add_callback(ValidationCallback())
            emit(
                "configuration",
                training=self.args.to_dict(),
                model=self.model.config.to_dict(),
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
            )

        def training_step(self, model, inputs, *a, **kw):
            emit("input", step=self.state.global_step + 1, shapes=input_shapes(inputs))
            loss = super().training_step(model, inputs, *a, **kw)
            raw_loss = float(loss.detach())
            emit("raw_loss", step=self.state.global_step + 1, loss=raw_loss)
            if not math.isfinite(raw_loss):
                raise RuntimeError(f"Nonfinite raw loss at step {self.state.global_step + 1}")
            return loss

        def log(self, logs, *a, **kw):
            values = {key: float(logs[key]) for key in ("loss", "grad_norm") if key in logs}
            if values:
                emit("metrics", step=self.state.global_step, **values)
            norm = values.get("grad_norm")
            if norm is not None and (
                not math.isfinite(norm) or not 0 <= norm <= args.max_observed_grad_norm
            ):
                raise RuntimeError(f"Invalid gradient norm {norm} at step {self.state.global_step}")
            return super().log(logs, *a, **kw)

        def save_model(self, *a, **kw):
            if save_checkpoint:
                return super().save_model(*a, **kw)

    def cached_stats_only(path, data, **kwargs):
        path = Path(path)
        if not path.is_file() or json.loads(path.read_text()) != json.loads(json.dumps(data)):
            raise RuntimeError(
                f"Precompute matching normalization statistics before validation: {path}"
            )

    def run(config):
        if config.training.resume_from_checkpoint or config.training.skip_weight_loading:
            raise ValueError(
                "Validation requires pretrained weights and a fresh disposable training run"
            )
        if config.training.max_steps < validation_steps:
            raise ValueError("The production max_steps schedule must cover validation-steps")
        config.training.output_dir = str(output)
        config.training.experiment_name = None
        config.training.use_wandb = False
        config.training.upload_checkpoints = False
        config.training.logging_steps = 1
        emit("training_config", config=config.__dict__)
        return original_run(config)

    try:
        with (
            patch.object(experiment, "run", run),
            patch.object(experiment, "Gr00tTrainer", ValidationTrainer),
            patch.object(stats, "_dump_stats_cache_atomic", cached_stats_only),
            patch.object(sys, "argv", ["train_b1k.py", *training_args]),
        ):
            runpy.run_path(str(REPO / "scripts/b1k/train_b1k.py"), run_name="__main__")
        summary = validate_records(
            records, validation_steps, args.max_observed_grad_norm, not args.allow_fixed_length
        )
        emit("passed", **summary)
        (output / f"validation-rank{rank}-summary.json").write_text(json.dumps(summary, indent=2))
    except BaseException as exc:
        emit("failed", error=str(exc))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
