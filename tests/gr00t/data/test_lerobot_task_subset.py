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

"""Task subsets (``task_names``) of a LeRobot dataset.

Training on one BEHAVIOR task inside the full 100-task ``2026-challenge-demos`` root must
load exactly that task's episodes and normalize with statistics computed over them
alone, kept apart from the dataset-wide ``meta/stats.json``. These tests pin that on a
tiny synthetic v3.0 dataset whose single data file *interleaves* episodes of three tasks
(so within-file row offsets are exercised), plus the v2.1 layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import (
    LEROBOT_TASK_SUBSETS_DIR_NAME,
    LeRobotEpisodeLoader,
    load_lerobot_tasks_table,
    normalize_task_names,
    select_task_indices,
    select_task_subset,
    task_subset_key,
    task_subset_stats_dir,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import (
    RelativeActionLoader,
    check_stats_validity,
    generate_rel_stats,
    generate_stats,
    rel_stats_file_path,
    stats_file_path,
)
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
import numpy as np
import pandas as pd
import pytest


TASKS = [
    {"task_index": 0, "task_name": "turning_on_radio", "task": "Turn on the radio."},
    {"task_index": 1, "task_name": "picking_up_trash", "task": "Put the cans in the trash."},
    {"task_index": 2, "task_name": "can_meat", "task": "Can the meat."},
]
# episode_index -> task_index; task 1's first episode is *not* the file's first row.
EPISODE_TASKS = {0: 0, 1: 1, 2: 2, 3: 1, 4: 0}
EPISODE_LENGTH = 6
STATE_DIM = 2
ACTION_DIM = 2
LANG_KEY = "annotation.human.task_name"


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
        "annotation": {
            "human.task_name": {
                "original_key": "task_index",
                "tasks_file": "tasks.jsonl",
                "task_field": "task_name",
            }
        },
    }


def _features() -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
        "action": {"dtype": "float32", "shape": [ACTION_DIM]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }


def _frames(episode_index: int, task_index: int, global_offset: int) -> pd.DataFrame:
    """Every state/action value equals the episode index, so slices are attributable."""
    value = float(episode_index)
    return pd.DataFrame(
        {
            "observation.state": [np.full(STATE_DIM, value, np.float32)] * EPISODE_LENGTH,
            "action": [np.full(ACTION_DIM, value, np.float32)] * EPISODE_LENGTH,
            "timestamp": np.arange(EPISODE_LENGTH, dtype=np.float32) / 30,
            "frame_index": np.arange(EPISODE_LENGTH),
            "episode_index": np.full(EPISODE_LENGTH, episode_index),
            "index": np.arange(EPISODE_LENGTH) + global_offset,
            "task_index": np.full(EPISODE_LENGTH, task_index),
        }
    )


def make_v30_dataset(root: Path, *, episode_task_index_column: bool = True) -> Path:
    """Five episodes of three tasks, interleaved in one data file (see EPISODE_TASKS)."""
    meta = root / "meta"
    _write_json(
        meta / "info.json",
        {
            "codebase_version": "v3.0",
            "fps": 30,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": None,
            "features": _features(),
        },
    )
    _write_json(meta / "stats.json", _stats())
    _write_json(meta / "modality.json", _modality_json())
    tasks_df = pd.DataFrame({"task_index": [t["task_index"] for t in TASKS]})
    tasks_df.index = pd.Index([t["task_name"] for t in TASKS], name="task")
    tasks_df.to_parquet(meta / "tasks.parquet")
    _write_jsonl(meta / "tasks.jsonl", TASKS)

    episodes, frames = [], []
    for episode_index, task_index in EPISODE_TASKS.items():
        offset = episode_index * EPISODE_LENGTH
        frames.append(_frames(episode_index, task_index, offset))
        record = {
            "episode_index": episode_index,
            "tasks": [TASKS[task_index]["task_name"]],
            "length": EPISODE_LENGTH,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": offset,
            "dataset_to_index": offset + EPISODE_LENGTH,
        }
        if episode_task_index_column:  # the BEHAVIOR demos carry it; stock LeRobot does not
            record["task_index"] = task_index
        episodes.append(record)
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(episodes).to_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "data" / "chunk-000").mkdir(parents=True)
    pd.concat(frames, ignore_index=True).to_parquet(
        root / "data" / "chunk-000" / "file-000.parquet"
    )
    return root


def make_v21_dataset(root: Path) -> Path:
    """Same episodes in the v2.1 layout, with the sidecar-style tasks.jsonl the
    v3.0 -> v2.1 conversion of the demos produces (``tasks`` = descriptions)."""
    meta = root / "meta"
    _write_json(
        meta / "info.json",
        {
            "codebase_version": "v2.1",
            "fps": 30,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": _features(),
        },
    )
    _write_json(meta / "stats.json", _stats())
    _write_json(meta / "modality.json", _modality_json())
    _write_jsonl(meta / "tasks.jsonl", TASKS)
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {
                "episode_index": i,
                "tasks": [TASKS[t]["task"]],
                "length": EPISODE_LENGTH,
                "task_index": t,
            }
            for i, t in EPISODE_TASKS.items()
        ],
    )
    (root / "data" / "chunk-000").mkdir(parents=True)
    for episode_index, task_index in EPISODE_TASKS.items():
        _frames(episode_index, task_index, episode_index * EPISODE_LENGTH).to_parquet(
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
    return root


def _modality_configs() -> dict[str, ModalityConfig]:
    return {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
        "action": ModalityConfig(delta_indices=[0, 1], modality_keys=["x"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[LANG_KEY]),
    }


def _episodes_of(*task_indices: int) -> list[int]:
    return [ep for ep, t in EPISODE_TASKS.items() if t in task_indices]


@pytest.fixture
def v30(tmp_path):
    return make_v30_dataset(tmp_path / "v30")


@pytest.fixture
def v21(tmp_path):
    return make_v21_dataset(tmp_path / "v21")


def _seed_subset_stats(dataset: Path, task_names) -> Path:
    """Give a subset the stats file the loader requires (as generate_stats would)."""
    path = stats_file_path(dataset, task_names)
    _write_json(path, _stats())
    return path


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_normalize_task_names(self):
        assert normalize_task_names(None) is None
        assert normalize_task_names([]) is None
        assert normalize_task_names("a") == ("a",)
        assert normalize_task_names(["b", "a", "b"]) == ("a", "b")

    def test_subset_key_is_readable_for_identifiers(self):
        assert task_subset_key(["turning_on_radio"]) == "turning_on_radio"
        assert (
            task_subset_key(["turning_on_radio", "picking_up_trash"])
            == "picking_up_trash+turning_on_radio"
        )
        # order-insensitive
        assert task_subset_key(["a", "b"]) == task_subset_key(["b", "a"])

    def test_subset_key_hashes_unsafe_or_long_names(self):
        key = task_subset_key(["Turn on the radio."])
        assert key.startswith("sha256-") and len(key) == len("sha256-") + 16
        assert task_subset_key(["x" * 200]).startswith("sha256-")
        # a name containing the separator must not alias a two-task key
        assert task_subset_key(["a+b"]) != task_subset_key(["a", "b"])

    def test_stats_dir(self, tmp_path):
        assert task_subset_stats_dir(tmp_path, None) == tmp_path / "meta"
        assert (
            task_subset_stats_dir(tmp_path, ["turning_on_radio"])
            == tmp_path / "meta" / LEROBOT_TASK_SUBSETS_DIR_NAME / "turning_on_radio"
        )
        assert stats_file_path(tmp_path, ["t"]).name == "stats.json"
        assert rel_stats_file_path(tmp_path, ["t"]).name == "relative_stats.json"

    def test_select_task_indices_matches_any_text_field(self, v30):
        table = load_lerobot_tasks_table(v30, "tasks.jsonl")
        assert select_task_indices(table, ["turning_on_radio"]) == {0}
        assert select_task_indices(table, ["Put the cans in the trash."]) == {1}
        assert select_task_indices(table, ["can_meat", "turning_on_radio"]) == {0, 2}

    def test_select_task_indices_rejects_unknown(self, v30):
        table = load_lerobot_tasks_table(v30, "tasks.parquet")
        with pytest.raises(ValueError, match=r"Unknown task\(s\) \['nope'\].*turning_on_radio"):
            select_task_indices(table, ["nope"])


# ---------------------------------------------------------------------------
# select_task_subset
# ---------------------------------------------------------------------------


class TestSelectTaskSubset:
    def test_v30_by_task_index_column(self, v30):
        subset = select_task_subset(v30, ["picking_up_trash"])
        assert subset.task_indices == {1}
        assert subset.episode_indices == _episodes_of(1) == [1, 3]

    def test_v30_by_tasks_strings_without_task_index_column(self, tmp_path):
        dataset = make_v30_dataset(tmp_path / "stock", episode_task_index_column=False)
        subset = select_task_subset(dataset, ["picking_up_trash", "can_meat"])
        assert subset.episode_indices == _episodes_of(1, 2) == [1, 2, 3]

    def test_v21_snake_case_selector_on_description_tasks(self, v21):
        """The converted demos list descriptions in ``episodes.jsonl``; the snake_case
        name still resolves through the sidecar's ``task_name`` field."""
        subset = select_task_subset(v21, ["picking_up_trash"])
        assert subset.episode_indices == [1, 3]

    def test_no_matching_episode_fails(self, v30):
        # Drop task 2's episodes from the episode table -> valid task, no data.
        meta = v30 / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        df = pd.read_parquet(meta)
        df[df["task_index"] != 2].to_parquet(meta)
        with pytest.raises(ValueError, match="No episodes of task"):
            select_task_subset(v30, ["can_meat"])


# ---------------------------------------------------------------------------
# LeRobotEpisodeLoader(task_names=...)
# ---------------------------------------------------------------------------


class TestLoaderTaskSubset:
    def test_requires_subset_stats(self, v30):
        with pytest.raises(AssertionError, match="task_subsets/picking_up_trash/stats.json"):
            LeRobotEpisodeLoader(v30, _modality_configs(), task_names=["picking_up_trash"])

    def test_filtered_episodes_and_exact_rows(self, v30):
        _seed_subset_stats(v30, ["picking_up_trash"])
        loader = LeRobotEpisodeLoader(v30, _modality_configs(), task_names=["picking_up_trash"])
        assert len(loader) == 2
        assert loader.task_names == ("picking_up_trash",)
        assert loader.task_indices == {1}
        assert loader.stats_dir == task_subset_stats_dir(v30, ["picking_up_trash"])
        # Loader index 0 is dataset episode 1: its rows carry value 1.0 and its task text,
        # even though row 0 of the data file belongs to (unselected) episode 0.
        for loader_idx, episode_index in enumerate([1, 3]):
            df = loader.load_episode(loader_idx)
            assert len(df) == EPISODE_LENGTH
            assert (df[f"language.{LANG_KEY}"] == "picking_up_trash").all()
            assert all(np.all(v == float(episode_index)) for v in df["state.x"])
            assert all(np.all(v == float(episode_index)) for v in df["action.x"])

    def test_multiple_tasks(self, v30):
        names = ["turning_on_radio", "can_meat"]
        _seed_subset_stats(v30, names)
        loader = LeRobotEpisodeLoader(v30, _modality_configs(), task_names=names)
        assert [ep["episode_index"] for ep in loader.episodes_metadata] == _episodes_of(0, 2)

    def test_none_and_empty_keep_every_episode(self, v30):
        for task_names in (None, []):
            loader = LeRobotEpisodeLoader(v30, _modality_configs(), task_names=task_names)
            assert len(loader) == len(EPISODE_TASKS)
            assert loader.task_names is None
            assert loader.stats_dir == v30 / "meta"

    def test_v21_layout(self, v21):
        _seed_subset_stats(v21, ["turning_on_radio"])
        loader = LeRobotEpisodeLoader(v21, _modality_configs(), task_names=["turning_on_radio"])
        assert [ep["episode_index"] for ep in loader.episodes_metadata] == [0, 4]
        df = loader.load_episode(1)
        assert all(np.all(v == 4.0) for v in df["state.x"])


# ---------------------------------------------------------------------------
# Statistics for a task subset
# ---------------------------------------------------------------------------


class TestSubsetStats:
    def test_generate_stats_uses_only_subset_rows_and_own_file(self, v30):
        before = (v30 / "meta" / "stats.json").read_bytes()
        generate_stats(v30, task_names=["picking_up_trash"])

        subset_path = stats_file_path(v30, ["picking_up_trash"])
        assert subset_path.is_file()
        stats = json.loads(subset_path.read_text())
        # Episodes 1 and 3 -> values {1, 3}: mean 2, min 1, max 3 on every dim.
        assert stats["observation.state"]["mean"] == [2.0] * STATE_DIM
        assert stats["observation.state"]["min"] == [1.0] * STATE_DIM
        assert stats["action"]["max"] == [3.0] * ACTION_DIM
        # The dataset-wide file is untouched by a subset run.
        assert (v30 / "meta" / "stats.json").read_bytes() == before
        assert check_stats_validity(v30, ["observation.state", "action"], ["picking_up_trash"])
        assert not check_stats_validity(v30, ["observation.state", "action"])

    def test_generate_stats_is_cached_per_subset(self, v30):
        generate_stats(v30, task_names=["picking_up_trash"])
        first = stats_file_path(v30, ["picking_up_trash"]).stat().st_mtime_ns
        generate_stats(v30, task_names=["picking_up_trash"])  # fingerprints match -> no rewrite
        assert stats_file_path(v30, ["picking_up_trash"]).stat().st_mtime_ns == first
        generate_stats(v30, task_names=["can_meat"])  # another subset, another file
        assert stats_file_path(v30, ["can_meat"]).is_file()
        assert (
            json.loads(stats_file_path(v30, ["can_meat"]).read_text())["action"]["mean"]
            == [2.0] * ACTION_DIM
        )

    def test_unfiltered_generate_stats_is_unchanged(self, v30):
        (v30 / "meta" / "stats.json").unlink()
        generate_stats(v30)
        stats = json.loads((v30 / "meta" / "stats.json").read_text())
        assert stats["observation.state"]["mean"] == [2.0] * STATE_DIM  # mean of 0..4
        assert stats["observation.state"]["max"] == [4.0] * STATE_DIM
        assert not (v30 / "meta" / LEROBOT_TASK_SUBSETS_DIR_NAME).exists()

    def test_v21_generate_stats_subset(self, v21):
        generate_stats(v21, task_names=["turning_on_radio"])
        stats = json.loads(stats_file_path(v21, ["turning_on_radio"]).read_text())
        assert stats["observation.state"]["mean"] == [2.0] * STATE_DIM  # episodes 0 and 4
        assert stats["observation.state"]["max"] == [4.0] * STATE_DIM

    def test_relative_stats_for_subset(self, v30, monkeypatch):
        tag = EmbodimentTag.NEW_EMBODIMENT
        monkeypatch.setitem(
            MODALITY_CONFIGS,
            tag.value,
            {
                "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
                "action": ModalityConfig(
                    delta_indices=[0, 1],
                    modality_keys=["x"],
                    action_configs=[
                        ActionConfig(
                            rep=ActionRepresentation.RELATIVE,
                            type=ActionType.NON_EEF,
                            format=ActionFormat.DEFAULT,
                            state_key="x",
                        )
                    ],
                ),
            },
        )
        generate_stats(v30, task_names=["picking_up_trash"])
        loader = RelativeActionLoader(v30, tag, "x", task_names=["picking_up_trash"])
        assert len(loader) == 2

        generate_rel_stats(v30, tag, task_names=["picking_up_trash"])
        rel_path = rel_stats_file_path(v30, ["picking_up_trash"])
        assert rel_path.is_file()
        rel = json.loads(rel_path.read_text())
        assert "x" in rel and "__fingerprints__" in rel
        assert not (v30 / "meta" / "relative_stats.json").exists()

        # The loader that trains on the subset picks both files up from the subset dir.
        loader = LeRobotEpisodeLoader(v30, _modality_configs(), task_names=["picking_up_trash"])
        assert "relative_action" in loader.stats
        assert loader.get_dataset_statistics()["state"]["x"]["mean"] == [2.0] * STATE_DIM
