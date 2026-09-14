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

"""``task_names`` plumbing: CLI flag -> ``SingleDatasetConfig`` -> ``DatasetFactory`` ->
stats generation and ``ShardedSingleStepDataset`` (all three must see the same subset)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gr00t.configs.base_config import get_default_config
from gr00t.configs.data.data_config import SingleDatasetConfig
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.types import ModalityConfig
import numpy as np
import pytest
import tyro


def _config(task_names):
    """The data config exactly as ``scripts/b1k/train_b1k.py`` assembles it."""
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": ["/fake/root"],
                        "mix_ratio": 1.0,
                        "embodiment_tag": "new_embodiment",
                        "task_names": task_names,
                    }
                ],
            }
        }
    )
    config.data.modality_configs = {
        "new_embodiment": {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["cam"]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
            "action": ModalityConfig(delta_indices=list(range(4)), modality_keys=["x"]),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        }
    }
    return config


def _mock_dataset():
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=4)
    dataset.shard_lengths = np.full(4, 100)
    dataset.get_shard_length = MagicMock(return_value=100)
    dataset.embodiment_tag = type("ET", (), {"value": "new_embodiment"})()
    stat = {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.2], "q01": [0.0], "q99": [1.0]}
    dataset.get_dataset_statistics.return_value = {"state": {"x": stat}, "action": {"x": stat}}
    return dataset


class TestConfig:
    def test_load_dict_builds_dataclass_with_task_names(self):
        spec = _config(["turning_on_radio"]).data.datasets[0]
        assert isinstance(spec, SingleDatasetConfig)
        assert spec.task_names == ["turning_on_radio"]

    def test_default_is_none(self):
        assert SingleDatasetConfig(dataset_paths=["/x"]).task_names is None
        assert _config(None).data.datasets[0].task_names is None


class TestFactoryForwarding:
    @pytest.mark.parametrize(
        "task_names, expected",
        [(["turning_on_radio"], ["turning_on_radio"]), (None, None), ([], None)],
    )
    def test_same_subset_reaches_stats_and_dataset(self, task_names, expected):
        factory = DatasetFactory(_config(task_names))
        with (
            patch("gr00t.data.dataset.factory.generate_stats") as gen_stats,
            patch("gr00t.data.dataset.factory.generate_rel_stats") as gen_rel,
            patch(
                "gr00t.data.dataset.factory.ShardedSingleStepDataset", return_value=_mock_dataset()
            ) as dataset_cls,
            patch("torch.distributed.is_initialized", return_value=False),
        ):
            factory.build(MagicMock())

        gen_stats.assert_called_once()
        assert gen_stats.call_args.kwargs["task_names"] == expected
        gen_rel.assert_called_once()
        assert gen_rel.call_args.kwargs["task_names"] == expected
        dataset_cls.assert_called_once()
        assert dataset_cls.call_args.kwargs["task_names"] == expected
        assert dataset_cls.call_args.kwargs["dataset_path"] == "/fake/root"


class TestTrainCli:
    def test_task_names_flag(self):
        from scripts.b1k.train_b1k import B1KFinetuneConfig

        base = [
            "--base-model-path",
            "m",
            "--dataset-path",
            "d",
            "--embodiment-tag",
            "new_embodiment",
        ]
        assert tyro.cli(B1KFinetuneConfig, args=base).task_names is None
        parsed = tyro.cli(B1KFinetuneConfig, args=base + ["--task-names", "turning_on_radio"])
        assert parsed.task_names == ["turning_on_radio"]
        parsed = tyro.cli(
            B1KFinetuneConfig,
            args=base
            + ["--task-names", "turning_on_radio", "picking_up_trash", "--max-steps", "3"],
        )
        assert parsed.task_names == ["turning_on_radio", "picking_up_trash"]
        assert parsed.max_steps == 3
