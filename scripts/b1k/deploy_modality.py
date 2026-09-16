#!/usr/bin/env python3
"""Deploy the B1K R1Pro ``modality.json`` into every task dataset under a root.

All BEHAVIOR-1K R1Pro tasks share one modality layout (61-dim
``observation.state``, 23-dim ``action``, fixed camera keys), so a single
template (``examples/b1k/r1pro.json``) is deployed into each
``<task>/meta/modality.json``. Before writing, each dataset's ``meta/info.json``
is validated against that layout, so any task that deviates from the expected
format is reported loudly instead of being silently mis-sliced at train time.
The tasks table each language annotation key resolves through (``meta/tasks.jsonl``,
which carries both the natural-language ``task`` description and the snake_case
``task_name``) is checked the same way. The per-task partial download in the challenge
docs does not fetch that sidecar (only the canonical ``meta/tasks.parquet``), so when a
v3.0 dataset lacks it, matching rows from ``examples/b1k/tasks.jsonl`` are installed
after checking every canonical task ID/name. Converted v2 tables are enriched by
matching IDs and existing text; their canonical ``task`` strings are preserved,
and ``human.task_description`` reads the added ``task_description`` field.

Usage:
    python scripts/b1k/deploy_modality.py <b1k_root> [--template PATH] [--tasks-file PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
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
    rows = _index_tasks(df.to_dict(orient="records"), str(meta_dir / CANONICAL_V30_TASKS_FILE))
    return {index: row["task"] for index, row in rows.items()}


def _sidecar_tables_needed(template: dict[str, Any]) -> set[str]:
    """``tasks_file`` names the template's annotation keys resolve through (``.jsonl`` only)."""
    return {
        meta["tasks_file"]
        for meta in template["annotation"].values()
        if str(meta.get("tasks_file", "")).endswith(".jsonl")
    }


def _index_tasks(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label}: task rows must be JSON objects")
        index = row.get("task_index")
        if type(index) is not int or index < 0:
            raise ValueError(f"{label}: invalid task_index {index!r}")
        if index in indexed:
            raise ValueError(f"{label}: duplicate task_index {index}")
        if not isinstance(row.get("task"), str) or not row["task"].strip():
            raise ValueError(f"{label}: task_index {index} has no nonempty 'task' string")
        indexed[index] = row
    if not indexed:
        raise ValueError(f"{label}: empty tasks table")
    return indexed


def _check_tasks_sidecar(
    sidecar_rows: list[dict[str, Any]], canonical: dict[int, str]
) -> list[str]:
    """Require ID/name agreement for every canonical row; allow a reference superset."""
    try:
        by_index = _index_tasks(sidecar_rows, "sidecar")
    except ValueError as exc:
        return [str(exc)]
    errors = []
    missing = set(canonical) - set(by_index)
    if missing:
        errors.append(f"task indices differ: sidecar missing {sorted(missing)}")
    for index, expected in canonical.items():
        row = by_index.get(index)
        if row is not None and row.get("task_name") != expected:
            errors.append(
                f"task_index {index}: sidecar task_name {row.get('task_name')!r} != "
                f"meta/{CANONICAL_V30_TASKS_FILE} task {expected!r}"
            )
    return errors


def _atomic_write(path: Path, payload: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def ensure_tasks_sidecar(
    dataset: Path, template: dict[str, Any], tasks_file: Path, dry_run: bool
) -> tuple[str, list[str]]:
    """Install a missing v3 sidecar or enrich a converted table by matching IDs and text.

    Converted v2 task strings remain canonical for episodes.jsonl. Their human-readable
    descriptions are added separately and selected by the deployed modality metadata.
    """
    needed = _sidecar_tables_needed(template)
    if not needed:
        return "skipped", []
    if len(needed) > 1:
        return "skipped", [f"template references several jsonl tasks tables: {sorted(needed)}"]
    filename = next(iter(needed))
    if Path(filename).name != filename:
        return "skipped", [f"tasks table must be a filename within meta/: {filename}"]
    meta_dir = dataset / "meta"
    sidecar = meta_dir / filename
    try:
        canonical = (
            _load_canonical_v30_tasks(meta_dir)
            if (meta_dir / CANONICAL_V30_TASKS_FILE).is_file()
            else None
        )
        rows = _load_jsonl(sidecar) if sidecar.is_file() else None
        existing = _index_tasks(rows, str(sidecar)) if rows is not None else None
        if existing is not None and canonical is not None and not set(canonical) <= set(existing):
            raise ValueError("existing sidecar task indices differ from meta/tasks.parquet")
        if existing is not None and all(
            isinstance(row.get("task_name"), str) and row["task_name"].strip() for row in rows
        ):
            errors = _check_tasks_sidecar(rows, canonical) if canonical is not None else []
            return "present", errors
        if existing is None and canonical is None:
            return "skipped", []
        if not tasks_file.is_file():
            raise ValueError(f"no canonical sidecar source at {tasks_file}; supply --tasks-file")
        reference = _index_tasks(_load_jsonl(tasks_file), str(tasks_file))
        if any(
            not isinstance(row.get("task_name"), str) or not row["task_name"].strip()
            for row in reference.values()
        ):
            raise ValueError(f"{tasks_file}: reference rows must have nonempty task_name strings")
        if canonical is not None:
            errors = _check_tasks_sidecar(list(reference.values()), canonical)
            if errors:
                return "skipped", errors
        if existing is None:
            rows = [reference[index] for index in canonical]
            status = "installed"
        else:
            enriched = []
            for index, row in existing.items():
                match = reference.get(index)
                if (
                    match is None
                    or row["task"] not in (match["task_name"], match["task"])
                    or row.get("task_name") not in (None, "", match["task_name"])
                    or row.get("task_description") not in (None, "", match["task"])
                ):
                    raise ValueError(
                        f"task_index {index}: existing ID/text does not match {tasks_file}"
                    )
                enriched.append(
                    {**row, "task_name": match["task_name"], "task_description": match["task"]}
                )
            rows = enriched
            status = "enriched"
        for key, annotation in template["annotation"].items():
            if annotation.get("tasks_file") == filename:
                field = annotation.get("task_field", "task")
                if any(not row.get(field) for row in rows):
                    raise ValueError(f"annotation '{key}' requires missing field '{field}'")
        if dry_run:
            return "planned", []
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        _atomic_write(sidecar, payload)
        return status, []
    except (ValueError, OSError) as exc:
        return "skipped", [f"cannot prepare meta/{filename}: {exc}"]


def _dataset_template(dataset: Path, template: dict[str, Any]) -> dict[str, Any]:
    template = json.loads(json.dumps(template))
    description = template["annotation"].get("human.task_description", {})
    filename = description.get("tasks_file")
    if (
        description.get("task_field", "task") == "task"
        and filename
        and (dataset / "meta" / filename).is_file()
    ):
        rows = _load_jsonl(dataset / "meta" / filename)
        if (
            rows
            and all(row.get("task_description") for row in rows)
            and any(row.get("task") == row.get("task_name") for row in rows)
        ):
            description["task_field"] = "task_description"
    return template


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
            "Canonical tasks.jsonl used to install missing v3 sidecars or enrich converted "
            f"tables after ID/text matching (default: {DEFAULT_TASKS_FILE})."
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
    tasks_file = args.tasks_file.expanduser().resolve()

    root = args.root.expanduser().resolve()
    datasets = find_datasets(root)
    if not datasets:
        print(f"error: no datasets (meta/info.json) found under {root}", file=sys.stderr)
        return 1

    written = unchanged = failed = 0
    for dataset in datasets:
        dst = dataset / "meta" / "modality.json"
        info = _load_json(dataset / "meta" / "info.json")
        errors = _validate_dataset(info, template)
        sidecar_status = "skipped"
        if not errors:
            sidecar_status, errors = ensure_tasks_sidecar(
                dataset, template, tasks_file, args.dry_run
            )
        if sidecar_status in ("installed", "enriched"):
            print(f"[write] {dataset}/meta/tasks.jsonl ({sidecar_status}, matched {tasks_file})")
        elif sidecar_status == "planned":
            written += 1
            print(f"[plan] {dataset}/meta/tasks.jsonl (would install from {tasks_file})")
        # Under --dry-run the planned sidecar is not on disk yet, so skip the
        # tasks-table check (it would only re-report the missing file).
        deployed_template = _dataset_template(dataset, template) if not errors else template
        if not errors:
            errors.extend(
                _validate_dataset(
                    info,
                    deployed_template,
                    meta_dir=None if sidecar_status == "planned" else dataset / "meta",
                )
            )
        template_bytes = (json.dumps(deployed_template, indent=4) + "\n").encode()
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
            _atomic_write(dst, template_bytes)
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
