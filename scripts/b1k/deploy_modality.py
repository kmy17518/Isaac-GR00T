#!/usr/bin/env python3
"""Deploy the B1K R1Pro ``modality.json`` into every task dataset under a root.

All BEHAVIOR-1K R1Pro tasks share one modality layout (61-dim
``observation.state``, 23-dim ``action``, fixed camera keys), so a single
template (``examples/b1k/r1pro.json``) is copied verbatim into each
``<task>/meta/modality.json``. Before copying, each dataset's ``meta/info.json``
is validated against that layout, so any task that deviates from the expected
format is reported loudly instead of being silently mis-sliced at train time.
The tasks table each language annotation key resolves through (``meta/tasks.jsonl``,
which carries both the natural-language ``task`` description and the snake_case
``task_name``) is checked the same way. The per-task partial download in the challenge
docs does not fetch that sidecar (only the canonical ``meta/tasks.parquet``), so when a
v3.0 dataset lacks it, the repo's verbatim copy (``examples/b1k/tasks.jsonl``) is
installed after checking that it agrees with ``meta/tasks.parquet`` on every
``task_index`` -> task name.

Usage:
    python scripts/b1k/deploy_modality.py <b1k_root> [--template PATH] [--tasks-file PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


# Raw R1Pro layout that examples/b1k/r1pro.json is designed for.
EXPECTED_STATE_DIM = 61  # observation.state, PROPRIOCEPTION_INDICES["R1Pro"]
EXPECTED_ACTION_DIM = 23  # action, ACTION_QPOS_INDICES["R1Pro"]

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "examples" / "b1k" / "r1pro.json"
# Verbatim copy of the demos' ``meta/tasks.jsonl`` (see gr00t.data.b1k_prompts).
DEFAULT_TASKS_FILE = Path(__file__).resolve().parents[2] / "examples" / "b1k" / "tasks.jsonl"
CANONICAL_V30_TASKS_FILE = "tasks.parquet"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_canonical_v30_tasks(meta_dir: Path) -> dict[int, str]:
    """``task_index -> task string`` from ``meta/tasks.parquet`` (LeRobot v3.0)."""
    import pyarrow.parquet as pq

    df = pq.read_table(meta_dir / CANONICAL_V30_TASKS_FILE).to_pandas()
    if df.index.name == "task":  # LeRobot writes the task string as the (named) index
        df = df.reset_index()
    return {int(row["task_index"]): str(row["task"]) for row in df.to_dict(orient="records")}


def _sidecar_tables_needed(template: dict[str, Any]) -> set[str]:
    """``tasks_file`` names the template's annotation keys resolve through (``.jsonl`` only)."""
    return {
        meta["tasks_file"]
        for meta in template["annotation"].values()
        if str(meta.get("tasks_file", "")).endswith(".jsonl")
    }


def _check_tasks_sidecar(
    sidecar_rows: list[dict[str, Any]], canonical: dict[int, str]
) -> list[str]:
    """Errors if ``sidecar_rows`` disagree with the canonical table on task_index -> task_name.

    The canonical v3.0 task string of the BEHAVIOR demos is the snake_case task
    name, so the sidecar's ``task_name`` must match it index by index and cover the
    same indices; otherwise the sidecar is from a different dataset revision.
    """
    errors: list[str] = []
    by_index: dict[int, dict[str, Any]] = {}
    for row in sidecar_rows:
        if "task_index" not in row:
            errors.append(f"row without task_index: {row}")
            continue
        by_index[int(row["task_index"])] = row
    if set(by_index) != set(canonical):
        errors.append(
            f"task indices differ: sidecar has {len(by_index)}, meta/{CANONICAL_V30_TASKS_FILE} "
            f"has {len(canonical)} (missing {sorted(set(canonical) - set(by_index))[:5]}, "
            f"extra {sorted(set(by_index) - set(canonical))[:5]})"
        )
    for task_index, row in sorted(by_index.items()):
        expected = canonical.get(task_index)
        if expected is not None and row.get("task_name") != expected:
            errors.append(
                f"task_index {task_index}: sidecar task_name {row.get('task_name')!r} != "
                f"meta/{CANONICAL_V30_TASKS_FILE} task {expected!r}"
            )
    return errors


def ensure_tasks_sidecar(
    dataset: Path, template: dict[str, Any], tasks_file: Path, dry_run: bool
) -> tuple[str, list[str]]:
    """Install ``meta/tasks.jsonl`` from ``tasks_file`` when a v3.0 dataset lacks it.

    Returns ``(status, errors)`` with status one of ``"present"`` (nothing to do),
    ``"installed"`` / ``"planned"`` (copied, or would be under ``--dry-run``), or
    ``"skipped"`` (template does not need a jsonl sidecar, or dataset is not v3.0 --
    the loader then reports a missing table itself). ``errors`` is non-empty when
    the repo copy does not match the dataset's canonical tasks table.
    """
    meta_dir = dataset / "meta"
    needed = _sidecar_tables_needed(template)
    if not needed:
        return "skipped", []
    if len(needed) > 1:
        return "skipped", [f"template references several jsonl tasks tables: {sorted(needed)}"]
    sidecar = meta_dir / next(iter(needed))
    if sidecar.is_file():
        return "present", []
    if not (meta_dir / CANONICAL_V30_TASKS_FILE).is_file():
        return (
            "skipped",
            [],
        )  # v2.x layout: tasks.jsonl *is* the canonical table; nothing to derive from
    if not tasks_file.is_file():
        return "skipped", [f"meta/{sidecar.name} missing and no sidecar source at {tasks_file}"]

    errors = _check_tasks_sidecar(_load_jsonl(tasks_file), _load_canonical_v30_tasks(meta_dir))
    if errors:
        return "skipped", [
            f"cannot install meta/{sidecar.name} from {tasks_file}: {e}" for e in errors
        ]
    if dry_run:
        return "planned", []
    shutil.copyfile(tasks_file, sidecar)
    return "installed", []


def _validate_template(template: dict[str, Any]) -> None:
    """Sanity-check the template against the expected R1Pro dims.

    Guards against the template and EXPECTED_* constants silently drifting apart.
    """
    for section in ("state", "action", "video", "annotation"):
        if section not in template:
            raise ValueError(f"template missing '{section}' section")

    max_state_end = max(group["end"] for group in template["state"].values())
    if max_state_end > EXPECTED_STATE_DIM:
        raise ValueError(
            f"template state slices reach {max_state_end} > EXPECTED_STATE_DIM={EXPECTED_STATE_DIM}"
        )

    spans = sorted((g["start"], g["end"]) for g in template["action"].values())
    cursor = 0
    for start, end in spans:
        if start != cursor:
            raise ValueError(f"template action slices are not contiguous at index {cursor}")
        cursor = end
    if cursor != EXPECTED_ACTION_DIM:
        raise ValueError(f"template action covers {cursor} dims, expected {EXPECTED_ACTION_DIM}")


def _validate_tasks_table(meta_dir: Path, ann_key: str, meta: dict[str, Any]) -> list[str]:
    """Check that the tasks table an annotation key resolves through exists and has its field.

    Mirrors the ``tasks_file`` / ``task_field`` contract of
    ``gr00t.data.dataset.lerobot_episode_loader`` so a dataset whose
    ``meta/tasks.jsonl`` lacks e.g. the natural-language ``task`` field is
    reported here instead of failing (or silently training on the wrong text) later.
    """
    tasks_file = meta.get("tasks_file")
    if tasks_file is None:
        return []  # canonical LeRobot table; nothing extra to check
    task_field = meta.get("task_field", "task")
    tasks_path = meta_dir / tasks_file
    if not tasks_path.is_file():
        return [f"annotation '{ann_key}' -> missing tasks table 'meta/{tasks_file}'"]
    if tasks_path.suffix != ".jsonl":
        return []  # parquet tables are validated by the loader at train time
    errors: list[str] = []
    with open(tasks_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get(task_field) in (None, ""):
                errors.append(
                    f"annotation '{ann_key}' -> meta/{tasks_file}:{line_number} "
                    f"(task_index {row.get('task_index')}) has no '{task_field}' field"
                )
    return errors


def _validate_dataset(
    info: dict[str, Any], template: dict[str, Any], meta_dir: Path | None = None
) -> list[str]:
    """Return a list of format errors (empty list means the dataset is compatible)."""
    features = info.get("features", {})
    errors: list[str] = []

    for key, expected_dim in (
        ("observation.state", EXPECTED_STATE_DIM),
        ("action", EXPECTED_ACTION_DIM),
    ):
        feature = features.get(key)
        if feature is None:
            errors.append(f"missing feature '{key}'")
            continue
        if "float" not in str(feature.get("dtype", "")):
            errors.append(f"'{key}' dtype {feature.get('dtype')!r} is not float")
        if list(feature.get("shape", [])) != [expected_dim]:
            errors.append(f"'{key}' shape {feature.get('shape')} != [{expected_dim}]")

    for video_key, meta in template["video"].items():
        original_key = meta["original_key"]
        feature = features.get(original_key)
        if feature is None:
            errors.append(f"video '{video_key}' -> missing feature '{original_key}'")
        elif feature.get("dtype") != "video":
            errors.append(
                f"video '{video_key}' -> '{original_key}' dtype {feature.get('dtype')!r} != 'video'"
            )

    for ann_key, meta in template["annotation"].items():
        original_key = meta["original_key"]
        if original_key not in features:
            errors.append(f"annotation '{ann_key}' -> missing feature '{original_key}'")
        if meta_dir is not None:
            errors.extend(_validate_tasks_table(meta_dir, ann_key, meta))

    return errors


def find_datasets(root: Path) -> list[Path]:
    """Return dataset roots (dirs containing meta/info.json) under root, recursively."""
    datasets = []
    for info_path in sorted(root.rglob("info.json")):
        if info_path.parent.name == "meta":
            datasets.append(info_path.parent.parent)
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Root dir to search for task datasets (e.g. .../2026-challenge-demos/b1k).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"modality.json template to deploy (default: {DEFAULT_TEMPLATE}).",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=DEFAULT_TASKS_FILE,
        help=(
            "tasks.jsonl to install into meta/ when a v3.0 dataset lacks the sidecar the "
            f"template's annotation keys read, e.g. a per-task partial download (default: "
            f"{DEFAULT_TASKS_FILE}). Only installed if it matches meta/tasks.parquet."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing; exits non-zero if not in sync.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    template_path = args.template.expanduser().resolve()
    if not template_path.is_file():
        print(f"error: template not found: {template_path}", file=sys.stderr)
        return 2
    template = _load_json(template_path)
    _validate_template(template)
    template_bytes = template_path.read_bytes()
    tasks_file = args.tasks_file.expanduser().resolve()

    root = args.root.expanduser().resolve()
    datasets = find_datasets(root)
    if not datasets:
        print(f"error: no datasets (meta/info.json) found under {root}", file=sys.stderr)
        return 1

    written = unchanged = failed = 0
    for dataset in datasets:
        dst = dataset / "meta" / "modality.json"
        # A partial download has no meta/tasks.jsonl; install the repo copy first so
        # the tasks-table validation below sees it (dry-run only plans the install).
        sidecar_status, errors = ensure_tasks_sidecar(dataset, template, tasks_file, args.dry_run)
        if sidecar_status == "installed":
            print(f"[write] {dataset}/meta/tasks.jsonl (from {tasks_file}, matches tasks.parquet)")
        elif sidecar_status == "planned":
            written += 1
            print(f"[plan] {dataset}/meta/tasks.jsonl (would install from {tasks_file})")
        # Under --dry-run the planned sidecar is not on disk yet, so skip the
        # tasks-table check (it would only re-report the missing file).
        info = _load_json(dataset / "meta" / "info.json")
        errors.extend(
            _validate_dataset(
                info, template, meta_dir=None if sidecar_status == "planned" else dataset / "meta"
            )
        )
        if errors:
            failed += 1
            print(f"[FAIL] {dataset}")
            for error in errors:
                print(f"         - {error}")
            continue

        if dst.exists() and dst.read_bytes() == template_bytes:
            unchanged += 1
            print(f"[ok]   {dataset} (unchanged)")
        elif args.dry_run:
            written += 1
            print(f"[plan] {dataset} (would write modality.json)")
        else:
            shutil.copyfile(template_path, dst)
            written += 1
            print(f"[write] {dataset}")

    verb = "would write" if args.dry_run else "written"
    print(
        f"\nSummary: {len(datasets)} dataset(s) | {verb}: {written} | "
        f"unchanged: {unchanged} | failed: {failed}"
    )

    if failed or (args.dry_run and written):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
