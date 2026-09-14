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

"""``gr00t.data.b1k_prompts`` and its contract with ``examples/b1k/r1pro.json``."""

from __future__ import annotations

import json
from pathlib import Path

from gr00t.data.b1k_prompts import (
    DEFAULT_PROMPT_SOURCE,
    DEFAULT_TASKS_FILE,
    PROMPT_SOURCES,
    TASK_FIELDS,
    B1KTask,
    find_b1k_task,
    language_key,
    load_b1k_tasks,
    prompt_source_from_language_key,
)
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
R1PRO_JSON = REPO_ROOT / "examples" / "b1k" / "r1pro.json"


class TestLanguageKeys:
    @pytest.mark.parametrize("source", PROMPT_SOURCES)
    def test_round_trip(self, source):
        key = language_key(source)
        assert key == f"annotation.human.{source}"
        assert prompt_source_from_language_key(key) == source

    def test_default_is_task_name(self):
        assert DEFAULT_PROMPT_SOURCE == "task_name"
        assert TASK_FIELDS["task_description"] == "task"
        assert TASK_FIELDS["task_name"] == "task_name"

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError, match="Unknown B1K prompt source"):
            language_key("coarse_action")

    @pytest.mark.parametrize(
        "key", ["annotation.human.coarse_action", "task", "annotation.task_description"]
    )
    def test_foreign_language_key_rejected(self, key):
        with pytest.raises(ValueError, match="not a B1K prompt key"):
            prompt_source_from_language_key(key)


class TestTasksTable:
    def test_repo_copy_of_tasks_jsonl(self):
        tasks = load_b1k_tasks(DEFAULT_TASKS_FILE)
        assert sorted(tasks) == list(range(100)), "expected the 100 challenge tasks, 0-indexed"
        radio = find_b1k_task(tasks, "turning_on_radio")
        assert radio.task_index == 0
        assert radio.prompt("task_name") == "turning_on_radio"
        assert radio.prompt("task_description").startswith("Turn on the radio")
        assert len({t.task_name for t in tasks.values()}) == 100

    def test_dataset_style_file_and_errors(self, tmp_path):
        good = tmp_path / "tasks.jsonl"
        good.write_text(
            json.dumps({"task_index": 3, "task_name": "a_b", "task": "Do a then b."}) + "\n"
        )
        tasks = load_b1k_tasks(good)
        assert tasks == {3: B1KTask(task_index=3, task_name="a_b", task_description="Do a then b.")}
        with pytest.raises(KeyError, match="Unknown B1K task"):
            find_b1k_task(tasks, "nope")

        missing_field = tmp_path / "missing.jsonl"
        missing_field.write_text(json.dumps({"task_index": 0, "task": "only the name"}) + "\n")
        with pytest.raises(ValueError, match="missing fields \\['task_name'\\]"):
            load_b1k_tasks(missing_field)

        duplicate = tmp_path / "dup.jsonl"
        row = json.dumps({"task_index": 0, "task_name": "x", "task": "X."}) + "\n"
        duplicate.write_text(row + row)
        with pytest.raises(ValueError, match="duplicate task_index"):
            load_b1k_tasks(duplicate)

        with pytest.raises(FileNotFoundError):
            load_b1k_tasks(tmp_path / "absent.jsonl")


class TestModalityJsonContract:
    """r1pro.json must expose every prompt kind under the key ``language_key`` produces,
    resolving through the field of meta/tasks.jsonl that ``TASK_FIELDS`` names."""

    @pytest.mark.parametrize("source", PROMPT_SOURCES)
    def test_annotation_entry_per_prompt_source(self, source):
        annotation = json.loads(R1PRO_JSON.read_text())["annotation"]
        subkey = language_key(source).removeprefix("annotation.")
        assert subkey in annotation, f"r1pro.json lacks annotation entry {subkey!r}"
        entry = annotation[subkey]
        assert entry["original_key"] == "task_index"
        assert entry["tasks_file"] == "tasks.jsonl"
        assert entry["task_field"] == TASK_FIELDS[source]
