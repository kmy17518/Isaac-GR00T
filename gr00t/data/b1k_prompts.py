"""Text prompts of the BEHAVIOR-1K (B1K) challenge dataset.

The challenge demos (``behavior-1k/2026-challenge-demos``) ship two kinds of text
per task in ``meta/tasks.jsonl``::

    {
        "task_index": 0,
        "task_name": "turning_on_radio",
        "task": "Turn on the radio receiver that's on the table in the living room.",
    }

* ``task_description`` -- the natural-language instruction (``task`` field).
* ``task_name`` -- the snake_case task identifier (``task_name`` field). This is
  also what LeRobot's canonical ``meta/tasks.parquet`` stores as the task string.

Each kind is exposed to the GR00T data loader as its own language annotation key
(``annotation.human.<kind>``, declared in ``examples/b1k/r1pro.json``). The shared
modality config (``examples/b1k/r1pro.py``, passed to both training and serving)
picks the kind a model is trained on through ``language.modality_keys``;
``scripts/b1k/train_b1k.py --prompt-source`` overrides it per run. The checkpoint's
processor config records the resulting key, and serving (``scripts/b1k/serve_b1k.py``)
reads it back so the policy is prompted with the same kind of text it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, get_args


PromptSource = Literal["task_description", "task_name"]
PROMPT_SOURCES: tuple[str, ...] = get_args(PromptSource)
DEFAULT_PROMPT_SOURCE: PromptSource = "task_name"

# ``annotation.human.<prompt source>`` is the language modality key for each kind.
LANGUAGE_KEY_PREFIX = "annotation.human."

# Field of ``meta/tasks.jsonl`` that holds each kind of prompt.
TASK_FIELDS: dict[str, str] = {"task_description": "task", "task_name": "task_name"}

# Verbatim copy of the dataset's ``meta/tasks.jsonl`` (MIT licensed), so a policy can
# be served without the demos on disk. Regenerate with:
#   cp $DATA_ROOT/meta/tasks.jsonl examples/b1k/tasks.jsonl
DEFAULT_TASKS_FILE = Path(__file__).resolve().parents[2] / "examples" / "b1k" / "tasks.jsonl"


def validate_prompt_source(prompt_source: str) -> PromptSource:
    if prompt_source not in PROMPT_SOURCES:
        raise ValueError(
            f"Unknown B1K prompt source {prompt_source!r}; expected one of {list(PROMPT_SOURCES)}"
        )
    return prompt_source  # type: ignore[return-value]


def language_key(prompt_source: str) -> str:
    """Language modality key that trains on ``prompt_source``.

    >>> language_key("task_name")
    'annotation.human.task_name'
    """
    return f"{LANGUAGE_KEY_PREFIX}{validate_prompt_source(prompt_source)}"


def prompt_source_from_language_key(key: str) -> PromptSource:
    """Inverse of :func:`language_key`.

    Raises ``ValueError`` if ``key`` does not denote one of the B1K prompt kinds, so
    callers can fall back to an explicit prompt instead of guessing.
    """
    if key.startswith(LANGUAGE_KEY_PREFIX) and key[len(LANGUAGE_KEY_PREFIX) :] in PROMPT_SOURCES:
        return key[len(LANGUAGE_KEY_PREFIX) :]  # type: ignore[return-value]
    raise ValueError(
        f"Language key {key!r} is not a B1K prompt key; expected one of "
        f"{[language_key(s) for s in PROMPT_SOURCES]}"
    )


@dataclass(frozen=True)
class B1KTask:
    """One row of ``meta/tasks.jsonl``."""

    task_index: int
    task_name: str
    task_description: str

    def prompt(self, prompt_source: str) -> str:
        """The text of kind ``prompt_source`` for this task."""
        return getattr(self, validate_prompt_source(prompt_source))


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
