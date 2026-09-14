"""Text prompts of the BEHAVIOR-1K (B1K) challenge dataset.

The challenge demos (``behavior-1k/2026-challenge-demos``) describe every task in
``meta/tasks.jsonl``::

    {
        "task_index": 0,
        "task_name": "turning_on_radio",
        "task": "Turn on the radio receiver that's on the table in the living room.",
    }

``task`` is the natural-language instruction; ``task_name`` the snake_case task id
(also what LeRobot's canonical ``meta/tasks.parquet`` stores as the task string).
The OmniGibson evaluator identifies the running task by ``task_index`` (sent as
``task_id`` with every observation), so serving resolves the prompt through this
table (``scripts/b1k/serve_b1k.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


# Field of ``meta/tasks.jsonl`` that holds each kind of text.
TASK_FIELDS: dict[str, str] = {"task_description": "task", "task_name": "task_name"}

# Verbatim copy of the dataset's ``meta/tasks.jsonl`` (MIT licensed), so a policy can
# be served without the demos on disk. Regenerate with:
#   cp $DATA_ROOT/meta/tasks.jsonl examples/b1k/tasks.jsonl
DEFAULT_TASKS_FILE = Path(__file__).resolve().parents[2] / "examples" / "b1k" / "tasks.jsonl"


@dataclass(frozen=True)
class B1KTask:
    """One row of ``meta/tasks.jsonl``."""

    task_index: int
    task_name: str
    task_description: str


def load_b1k_tasks(tasks_file: str | Path = DEFAULT_TASKS_FILE) -> dict[int, B1KTask]:
    """Load a B1K ``tasks.jsonl`` as ``task_index -> B1KTask``.

    Accepts either the repo copy (default) or a dataset's ``meta/tasks.jsonl``.
    """
    tasks_file = Path(tasks_file)
    if not tasks_file.is_file():
        raise FileNotFoundError(f"B1K tasks file not found: {tasks_file}")
    tasks: dict[int, B1KTask] = {}
    with open(tasks_file, "r") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [
                field for field in ("task_index", *TASK_FIELDS.values()) if row.get(field) is None
            ]
            if missing:
                raise ValueError(f"{tasks_file}:{line_number} is missing fields {missing}: {row}")
            task = B1KTask(
                task_index=int(row["task_index"]),
                task_name=str(row[TASK_FIELDS["task_name"]]),
                task_description=str(row[TASK_FIELDS["task_description"]]),
            )
            if task.task_index in tasks:
                raise ValueError(f"{tasks_file}: duplicate task_index {task.task_index}")
            tasks[task.task_index] = task
    return tasks


def find_b1k_task(tasks: dict[int, B1KTask], task_name: str) -> B1KTask:
    """Look a task up by its snake_case ``task_name``."""
    for task in tasks.values():
        if task.task_name == task_name:
            return task
    raise KeyError(
        f"Unknown B1K task {task_name!r}; known tasks: {sorted(t.task_name for t in tasks.values())}"
    )
