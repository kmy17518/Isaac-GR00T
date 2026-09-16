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

"""``convert_v3_to_v2``: carrying a ``meta/tasks.jsonl`` sidecar through the conversion.

The BEHAVIOR challenge demos store the snake_case task id as the v3.0 parquet task
string and ship the natural-language instruction in a ``tasks.jsonl`` sidecar.
Regenerating the legacy ``tasks.jsonl`` from the parquet alone would drop that
description; these tests pin that the sidecar is carried over verbatim and that
``episodes.jsonl`` is remapped to the same strings.

The converter runs in its own LeRobot subproject environment. So this stays a CPU
test of the main environment, the few ``lerobot`` symbols the script imports are
stubbed when ``lerobot`` is not installed (``load_tasks`` mirrors LeRobot's
DataFrame layout: task strings as the named index, ``task_index`` as a column).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = REPO_ROOT / "scripts" / "lerobot_conversion" / "convert_v3_to_v2.py"

TASKS = [
    {"task_index": 0, "task_name": "turning_on_radio", "task": "Turn on the radio."},
    {"task_index": 1, "task_name": "picking_up_trash", "task": "Put the cans in the trash."},
]


def _lerobot_stub_modules() -> dict[str, types.ModuleType]:
    utils = types.ModuleType("lerobot.datasets.utils")
    utils.DEFAULT_CHUNK_SIZE = 1000
    utils.DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    utils.DEFAULT_VIDEO_PATH = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    utils.EPISODES_DIR = "meta/episodes"
    utils.LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
    utils.LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
    utils.LEGACY_TASKS_PATH = "meta/tasks.jsonl"
    utils.load_info = lambda root: json.loads((Path(root) / "meta" / "info.json").read_text())
    utils.load_tasks = lambda root: pd.read_parquet(Path(root) / "meta" / "tasks.parquet")
    utils.serialize_dict = lambda stats: stats
    utils.unflatten_dict = lambda flat, sep="/": {}
    utils.write_info = lambda info, root: None

    constants = types.ModuleType("lerobot.utils.constants")
    constants.HF_LEROBOT_HOME = Path("/nonexistent")
    lerobot_utils = types.ModuleType("lerobot.utils.utils")
    lerobot_utils.init_logging = lambda: None

    lerobot = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    utils_pkg = types.ModuleType("lerobot.utils")
    return {
        "lerobot": lerobot,
        "lerobot.datasets": datasets,
        "lerobot.datasets.utils": utils,
        "lerobot.utils": utils_pkg,
        "lerobot.utils.constants": constants,
        "lerobot.utils.utils": lerobot_utils,
    }


@pytest.fixture
def converter(monkeypatch):
    if importlib.util.find_spec("lerobot") is None:
        for name, module in _lerobot_stub_modules().items():
            monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("convert_v3_to_v2_under_test", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_v30_root(root: Path, *, with_sidecar: bool, sidecar_rows=None) -> Path:
    meta = root / "meta"
    meta.mkdir(parents=True)
    tasks_df = pd.DataFrame({"task_index": [t["task_index"] for t in TASKS]})
    tasks_df.index = pd.Index([t["task_name"] for t in TASKS], name="task")
    tasks_df.to_parquet(meta / "tasks.parquet")
    if with_sidecar:
        rows = TASKS if sidecar_rows is None else sidecar_rows
        (meta / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def _episode_records() -> list[dict]:
    return [
        {
            "episode_index": i,
            "tasks": np.array([t["task_name"]], dtype=object),
            "length": 10,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 10 * i,
            "dataset_to_index": 10 * (i + 1),
        }
        for i, t in enumerate(TASKS)
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_sidecar_is_carried_over_and_episodes_follow_it(converter, tmp_path):
    root = _make_v30_root(tmp_path / "src", with_sidecar=True)
    new_root = tmp_path / "dst"

    task_strings = converter.convert_tasks(root, new_root)

    # Verbatim copy: both the task id and the natural-language description survive.
    assert _read_jsonl(new_root / "meta" / "tasks.jsonl") == TASKS
    assert task_strings == {
        "turning_on_radio": "Turn on the radio.",
        "picking_up_trash": "Put the cans in the trash.",
    }

    converter.convert_episodes_metadata(new_root, _episode_records(), task_strings)
    episodes = _read_jsonl(new_root / "meta" / "episodes.jsonl")
    # episodes.jsonl "tasks" now uses the same strings as tasks.jsonl "task" (LeRobot v2.1 contract).
    assert [ep["tasks"] for ep in episodes] == [
        ["Turn on the radio."],
        ["Put the cans in the trash."],
    ]
    assert [ep["episode_index"] for ep in episodes] == [0, 1]
    assert all("data/chunk_index" not in ep and "dataset_from_index" not in ep for ep in episodes)


def test_without_sidecar_keeps_legacy_behavior(converter, tmp_path):
    root = _make_v30_root(tmp_path / "src", with_sidecar=False)
    new_root = tmp_path / "dst"

    task_strings = converter.convert_tasks(root, new_root)

    assert _read_jsonl(new_root / "meta" / "tasks.jsonl") == [
        {"task_index": 0, "task": "turning_on_radio"},
        {"task_index": 1, "task": "picking_up_trash"},
    ]
    assert task_strings == {
        "turning_on_radio": "turning_on_radio",
        "picking_up_trash": "picking_up_trash",
    }

    converter.convert_episodes_metadata(new_root, _episode_records(), task_strings)
    episodes = _read_jsonl(new_root / "meta" / "episodes.jsonl")
    assert [ep["tasks"] for ep in episodes] == [["turning_on_radio"], ["picking_up_trash"]]


def test_sidecar_index_mismatch_is_rejected(converter, tmp_path):
    root = _make_v30_root(tmp_path / "src", with_sidecar=True, sidecar_rows=TASKS[:1])
    with pytest.raises(ValueError, match="covers task indices \\[0\\]"):
        converter.convert_tasks(root, tmp_path / "dst")


def test_sidecar_row_without_task_is_rejected(converter, tmp_path):
    rows = [{"task_index": 0, "task_name": "turning_on_radio"}, TASKS[1]]
    root = _make_v30_root(tmp_path / "src", with_sidecar=True, sidecar_rows=rows)
    with pytest.raises(ValueError, match="must have 'task_index' and 'task'"):
        converter.convert_tasks(root, tmp_path / "dst")


@pytest.mark.parametrize("with_sidecar", [False, True])
def test_partial_conversion_then_deploy_resolves_both_prompts(
    converter, tmp_path, monkeypatch, with_sidecar
):
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from scripts.b1k import deploy_modality

    root = _make_v30_root(tmp_path / "dataset", with_sidecar=with_sidecar)
    template = json.loads((REPO_ROOT / "examples/b1k/r1pro.json").read_text())
    features = {
        "observation.state": {"dtype": "float32", "shape": [61]},
        "action": {"dtype": "float32", "shape": [23]},
        "task_index": {"dtype": "int64", "shape": [1]},
        **{meta["original_key"]: {"dtype": "video"} for meta in template["video"].values()},
    }
    (root / "meta/info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": features})
    )
    records = _episode_records()[:1]
    metadata = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata.parent.mkdir(parents=True)
    pd.DataFrame(records).to_parquet(metadata)
    data = root / "data/chunk-000/file-000.parquet"
    data.parent.mkdir(parents=True)
    pd.DataFrame({"task_index": [0] * 10, "frame_index": range(10)}).to_parquet(data)

    def write_info(info, destination):
        (destination / "meta").mkdir(parents=True, exist_ok=True)
        (destination / "meta/info.json").write_text(json.dumps(info))

    monkeypatch.setattr(converter, "write_info", write_info)
    monkeypatch.setattr(
        converter, "snapshot_download", lambda *a, **kw: pytest.fail("network download")
    )
    converter.convert_dataset("dataset", root=tmp_path)
    tasks_before = (root / "meta/tasks.jsonl").read_bytes()
    episodes_before = (root / "meta/episodes.jsonl").read_bytes()
    source_before = (tmp_path / "dataset_v3.0/meta/tasks.parquet").read_bytes()
    reference = tmp_path / "canonical.jsonl"
    reference.write_text("".join(json.dumps(row) + "\n" for row in TASKS))
    monkeypatch.setattr(sys, "argv", ["deploy", str(root), "--tasks-file", str(reference)])
    assert deploy_modality.main() == 0
    assert deploy_modality.main() == 0
    assert (root / "meta/episodes.jsonl").read_bytes() == episodes_before
    assert (tmp_path / "dataset_v3.0/meta/tasks.parquet").read_bytes() == source_before
    if with_sidecar:
        assert (root / "meta/tasks.jsonl").read_bytes() == tasks_before
    tasks = _read_jsonl(root / "meta/tasks.jsonl")
    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    assert episodes[0]["tasks"] == [tasks[0]["task"]]
    assert len(tasks) == 2 and len(episodes) == 1
    loader = LeRobotEpisodeLoader.__new__(LeRobotEpisodeLoader)
    loader.dataset_path = str(root)
    loader.tasks_filename = "tasks.jsonl"
    loader._tasks_tables = {}
    loader._annotation_text_maps = {}
    loader.modality_meta = json.loads((root / "meta/modality.json").read_text())
    assert loader._get_annotation_text_map("human.task_description")[0] == TASKS[0]["task"]
    assert loader._get_annotation_text_map("human.task_name")[0] == TASKS[0]["task_name"]
