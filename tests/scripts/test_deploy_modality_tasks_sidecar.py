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

"""The ``tasks.jsonl`` sidecar install of ``scripts/b1k/deploy_modality.py``.

A *per-task partial download* of the BEHAVIOR demos (``huggingface-cli download
--include data/chunk-XXX/** ...``) holds one task's chunks plus
``meta/{info.json,stats.json,tasks.parquet}`` and nothing else -- in particular no
``meta/tasks.jsonl``, which both ``r1pro.json`` annotation keys read. ``deploy_modality.py``
installs the repo's verbatim copy after checking it against ``meta/tasks.parquet``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_MODALITY = REPO_ROOT / "scripts" / "b1k" / "deploy_modality.py"

TASKS = [
    {"task_index": 0, "task_name": "turning_on_radio", "task": "Turn on the radio."},
    {"task_index": 1, "task_name": "picking_up_trash", "task": "Put the cans in the trash."},
]


@pytest.fixture(scope="module")
def deploy_modality():
    spec = importlib.util.spec_from_file_location("deploy_modality_under_test", DEPLOY_MODALITY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def template():
    return json.loads((REPO_ROOT / "examples" / "b1k" / "r1pro.json").read_text())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def partial(tmp_path):
    """v3.0 meta/ as the documented partial download leaves it: tasks.parquet, no jsonl."""
    meta = tmp_path / "partial" / "meta"
    meta.mkdir(parents=True)
    df = pd.DataFrame({"task_index": [t["task_index"] for t in TASKS]})
    df.index = pd.Index([t["task_name"] for t in TASKS], name="task")
    df.to_parquet(meta / "tasks.parquet")
    return tmp_path / "partial"


@pytest.fixture
def sidecar(tmp_path):
    path = tmp_path / "tasks.jsonl"
    _write_jsonl(path, TASKS)
    return path


class TestEnsureTasksSidecar:
    def test_installs_matching_sidecar(self, deploy_modality, template, partial, sidecar):
        status, errors = deploy_modality.ensure_tasks_sidecar(partial, template, sidecar, False)
        assert (status, errors) == ("installed", [])
        assert (partial / "meta" / "tasks.jsonl").read_bytes() == sidecar.read_bytes()
        # Second call: already there.
        assert deploy_modality.ensure_tasks_sidecar(partial, template, sidecar, False) == (
            "present",
            [],
        )

    def test_dry_run_only_plans(self, deploy_modality, template, partial, sidecar):
        assert deploy_modality.ensure_tasks_sidecar(partial, template, sidecar, True) == (
            "planned",
            [],
        )
        assert not (partial / "meta" / "tasks.jsonl").exists()

    def test_refuses_sidecar_from_another_dataset_revision(
        self, deploy_modality, template, partial, tmp_path
    ):
        renamed = tmp_path / "other.jsonl"
        _write_jsonl(renamed, [TASKS[0], {**TASKS[1], "task_name": "picking_up_garbage"}])
        status, errors = deploy_modality.ensure_tasks_sidecar(partial, template, renamed, False)
        assert status == "skipped"
        assert errors and "picking_up_garbage" in errors[0]
        assert not (partial / "meta" / "tasks.jsonl").exists()

        extra = tmp_path / "extra.jsonl"
        _write_jsonl(extra, TASKS + [{"task_index": 2, "task_name": "can_meat", "task": "x"}])
        status, errors = deploy_modality.ensure_tasks_sidecar(partial, template, extra, False)
        assert status == "skipped" and "task indices differ" in errors[0]

    def test_v21_layout_is_left_to_the_loader(self, deploy_modality, template, tmp_path, sidecar):
        v21 = tmp_path / "v21"
        (v21 / "meta").mkdir(parents=True)  # no tasks.parquet: tasks.jsonl *is* canonical there
        assert deploy_modality.ensure_tasks_sidecar(v21, template, sidecar, False) == (
            "skipped",
            [],
        )

    def test_template_without_jsonl_tables_skips(self, deploy_modality, partial, sidecar):
        template = {"annotation": {"human.task_description": {"original_key": "task_index"}}}
        assert deploy_modality.ensure_tasks_sidecar(partial, template, sidecar, False) == (
            "skipped",
            [],
        )

    def test_repo_copy_is_self_consistent(self, deploy_modality):
        """The default sidecar has the demos' 100 tasks and passes its own consistency check."""
        rows = deploy_modality._load_jsonl(deploy_modality.DEFAULT_TASKS_FILE)
        assert len(rows) == 100
        assert rows[0]["task_name"] == "turning_on_radio"
        assert (
            deploy_modality._check_tasks_sidecar(
                rows, {row["task_index"]: row["task_name"] for row in rows}
            )
            == []
        )
