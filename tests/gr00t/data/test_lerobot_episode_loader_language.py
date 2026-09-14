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

"""Language-annotation resolution in ``LeRobotEpisodeLoader``.

The BEHAVIOR challenge demos store the snake_case task *name* as LeRobot's canonical
task string (``meta/tasks.parquet``) and the natural-language *description* in a
``meta/tasks.jsonl`` sidecar. ``modality.json`` annotation entries may point at a
specific tasks table / text field (``tasks_file`` / ``task_field``); these tests pin
that contract on a tiny synthetic v3.0 dataset (no videos) plus the v2.x fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig
import numpy as np
import pandas as pd
import pytest


TASKS = [
    {"task_index": 0, "task_name": "turning_on_radio", "task": "Turn on the radio."},
    {"task_index": 1, "task_name": "picking_up_trash", "task": "Put the cans in the trash."},
]
EPISODE_LENGTH = 6
STATE_DIM = 2
ACTION_DIM = 2

# Annotation entries exercising every resolution mode.
ANNOTATION = {
    # natural-language description from the jsonl sidecar
    "human.task_description": {
        "original_key": "task_index",
        "tasks_file": "tasks.jsonl",
        "task_field": "task",
    },
    # snake_case name from the jsonl sidecar
    "human.task_name": {
        "original_key": "task_index",
        "tasks_file": "tasks.jsonl",
        "task_field": "task_name",
    },
    # stock LeRobot behavior: canonical table, ``task`` field
    "human.canonical": {"original_key": "task_index"},
    # misconfigured: field that does not exist in the table
    "human.missing": {
        "original_key": "task_index",
        "tasks_file": "tasks.jsonl",
        "task_field": "does_not_exist",
    },
}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _stats() -> dict:
    return {
        key: {
            stat: [0.0] * dim if stat != "std" else [1.0] * dim
            for stat in ("mean", "std", "min", "max", "q01", "q99")
        }
        for key, dim in (("observation.state", STATE_DIM), ("action", ACTION_DIM))
    }


def _modality_json() -> dict:
    return {
        "state": {"x": {"start": 0, "end": STATE_DIM}},
        "action": {"x": {"start": 0, "end": ACTION_DIM}},
        "video": {},
        "annotation": ANNOTATION,
    }


def _frames(episode_index: int, task_index: int, global_offset: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation.state": [np.zeros(STATE_DIM, np.float32)] * EPISODE_LENGTH,
            "action": [np.zeros(ACTION_DIM, np.float32)] * EPISODE_LENGTH,
            "timestamp": np.arange(EPISODE_LENGTH, dtype=np.float32) / 30,
            "frame_index": np.arange(EPISODE_LENGTH),
            "episode_index": np.full(EPISODE_LENGTH, episode_index),
            "index": np.arange(EPISODE_LENGTH) + global_offset,
            "task_index": np.full(EPISODE_LENGTH, task_index),
        }
    )


def _make_v30_dataset(root: Path, *, with_tasks_jsonl: bool = True) -> Path:
    """Two-episode v3.0 dataset: episode 0 -> task 0, episode 1 -> task 1, one data file."""
    meta = root / "meta"
    _write_json(
        meta / "info.json",
        {
            "codebase_version": "v3.0",
            "fps": 30,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": None,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
                "action": {"dtype": "float32", "shape": [ACTION_DIM]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        },
    )
    _write_json(meta / "stats.json", _stats())
    _write_json(meta / "modality.json", _modality_json())

    # Canonical v3.0 tasks table: the task string is the (named) index, as LeRobot writes it.
    tasks_df = pd.DataFrame({"task_index": [t["task_index"] for t in TASKS]})
    tasks_df.index = pd.Index([t["task_name"] for t in TASKS], name="task")
    tasks_df.to_parquet(meta / "tasks.parquet")
    if with_tasks_jsonl:
        _write_jsonl(meta / "tasks.jsonl", TASKS)

    episodes = []
    frames = []
    for episode_index, task in enumerate(TASKS):
        offset = episode_index * EPISODE_LENGTH
        frames.append(_frames(episode_index, task["task_index"], offset))
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [task["task_name"]],
                "length": EPISODE_LENGTH,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": offset,
                "dataset_to_index": offset + EPISODE_LENGTH,
            }
        )
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(episodes).to_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "data" / "chunk-000").mkdir(parents=True)
    pd.concat(frames, ignore_index=True).to_parquet(
        root / "data" / "chunk-000" / "file-000.parquet"
    )
    return root


def _make_v21_dataset(root: Path) -> Path:
    """Same content in the v2.1 layout (one parquet per episode, jsonl metadata)."""
    meta = root / "meta"
    _write_json(
        meta / "info.json",
        {
            "codebase_version": "v2.1",
            "fps": 30,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
                "action": {"dtype": "float32", "shape": [ACTION_DIM]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        },
    )
    _write_json(meta / "stats.json", _stats())
    _write_json(meta / "modality.json", _modality_json())
    _write_jsonl(meta / "tasks.jsonl", TASKS)
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {"episode_index": i, "tasks": [t["task_name"]], "length": EPISODE_LENGTH}
            for i, t in enumerate(TASKS)
        ],
    )
    (root / "data" / "chunk-000").mkdir(parents=True)
    for episode_index, task in enumerate(TASKS):
        _frames(episode_index, task["task_index"], episode_index * EPISODE_LENGTH).to_parquet(
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
    return root


def _modality_configs(language_key: str) -> dict[str, ModalityConfig]:
    return {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
        "action": ModalityConfig(delta_indices=[0, 1], modality_keys=["x"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[language_key]),
    }


def _language_per_episode(dataset: Path, language_key: str) -> list[list[str]]:
    loader = LeRobotEpisodeLoader(dataset, _modality_configs(language_key))
    return [loader.load_episode(i)[f"language.{language_key}"].unique().tolist() for i in range(2)]


@pytest.fixture
def v30_dataset(tmp_path):
    return _make_v30_dataset(tmp_path / "v30")


class TestV30AnnotationResolution:
    def test_task_description_reads_natural_language_from_jsonl(self, v30_dataset):
        langs = _language_per_episode(v30_dataset, "annotation.human.task_description")
        assert langs == [["Turn on the radio."], ["Put the cans in the trash."]]

    def test_task_name_reads_snake_case_from_jsonl(self, v30_dataset):
        langs = _language_per_episode(v30_dataset, "annotation.human.task_name")
        assert langs == [["turning_on_radio"], ["picking_up_trash"]]

    def test_default_entry_keeps_stock_lerobot_behavior(self, v30_dataset):
        """No tasks_file/task_field -> canonical tasks.parquet ``task`` string."""
        langs = _language_per_episode(v30_dataset, "annotation.human.canonical")
        assert langs == [["turning_on_radio"], ["picking_up_trash"]]

    def test_canonical_tasks_map_is_unchanged(self, v30_dataset):
        loader = LeRobotEpisodeLoader(
            v30_dataset, _modality_configs("annotation.human.task_description")
        )
        assert loader.tasks_map == {0: "turning_on_radio", 1: "picking_up_trash"}

    def test_missing_field_fails_loudly(self, v30_dataset):
        loader = LeRobotEpisodeLoader(v30_dataset, _modality_configs("annotation.human.missing"))
        with pytest.raises(KeyError, match="does_not_exist"):
            loader.load_episode(0)

    def test_missing_sidecar_table_fails_loudly(self, tmp_path):
        dataset = _make_v30_dataset(tmp_path / "no_jsonl", with_tasks_jsonl=False)
        loader = LeRobotEpisodeLoader(
            dataset, _modality_configs("annotation.human.task_description")
        )
        with pytest.raises(FileNotFoundError, match="tasks.jsonl"):
            loader.load_episode(0)
        # ...while the canonical table still works without the sidecar.
        assert _language_per_episode(dataset, "annotation.human.canonical") == [
            ["turning_on_radio"],
            ["picking_up_trash"],
        ]

    def test_text_is_resolved_per_frame_through_task_index(self, v30_dataset):
        """Every frame of an episode carries that episode's task text."""
        key = "annotation.human.task_description"
        loader = LeRobotEpisodeLoader(v30_dataset, _modality_configs(key))
        df = loader.load_episode(1)
        assert len(df) == EPISODE_LENGTH
        assert (df[f"language.{key}"] == "Put the cans in the trash.").all()


class TestV21AnnotationResolution:
    def test_jsonl_fields_resolve_in_legacy_layout(self, tmp_path):
        dataset = _make_v21_dataset(tmp_path / "v21")
        assert _language_per_episode(dataset, "annotation.human.task_description") == [
            ["Turn on the radio."],
            ["Put the cans in the trash."],
        ]
        assert _language_per_episode(dataset, "annotation.human.task_name") == [
            ["turning_on_radio"],
            ["picking_up_trash"],
        ]
        # Default entry resolves through tasks.jsonl ``task`` (the canonical v2.x table).
        assert _language_per_episode(dataset, "annotation.human.canonical") == [
            ["Turn on the radio."],
            ["Put the cans in the trash."],
        ]
