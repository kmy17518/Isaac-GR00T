#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the latest full resumable checkpoint in the Hub's current tree.

Complements hf_checkpoint_uploader.py (eval-only copies at scheduled steps). Each replacement commit
adds ``<prefix>/resume/checkpoint-<step>/**``, including optimizer and RNG state, and logically removes
the previous checkpoint folders. Historical objects remain stored and continue to count against quota.

Automatic irreversible object deletion is disabled: independent publishers and other refs make live
object scans unsafe. Durable transaction records in ``<status-file>.journal/`` retain replaced LFS
candidates for explicit offline operator review, NOT a deletion allowlist. Collection requires stopping
all writers and checking every ref; this monitor never performs it. Preserve the journal across restarts.
The status reports verified uploads separately from pending offline GC candidates. Legacy status or a
remote folder name alone is not proof of completion: without a journal, keep the latest local checkpoint
for verification/republication, or wait for a newer complete checkpoint. If Trainer prunes a pending local
checkpoint, recovery first checks for a lost commit response, then may supersede it with a newer complete
local checkpoint. Both journals and all candidates are retained; superseded steps are not uploaded steps.

Usage (run it detached, e.g. in tmux, next to the training job):
    python scripts/b1k/hf_resume_checkpoint_uploader.py --run-dir $OUTPUT_DIR/<exp> \
        --staging-dir $STAGING_DIR/<exp> --repo-id <user>/<repo> --path-prefix <exp> \
        --num-gpus 4 --max-steps 150000
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import uuid

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import HfHubHTTPError


log = logging.getLogger("hf-resume-uploader")
CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


class PendingCheckpointUnavailable(RuntimeError):
    """The saved local manifest can no longer be used to retry a prepared upload."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def dir_quiet_for(path: Path, seconds: float) -> bool:
    newest = path.stat().st_mtime
    for root, _dirs, files in os.walk(path):
        newest = max(newest, Path(root).stat().st_mtime)
        for f in files:
            try:
                newest = max(newest, (Path(root) / f).stat().st_mtime)
            except FileNotFoundError:
                return False
    return time.time() - newest >= seconds


def full_checkpoint_complete(ckpt: Path, step: int, num_gpus: int, quiet_seconds: float) -> bool:
    """Everything a DeepSpeed ZeRO resume needs is present, consistent, and no longer being written."""
    state_file = ckpt / "trainer_state.json"
    latest = ckpt / "latest"
    ds_dir = ckpt / f"global_step{step}"
    if not (state_file.exists() and latest.exists() and ds_dir.is_dir()):
        return False
    try:
        if int(json.loads(state_file.read_text()).get("global_step", -1)) != step:
            return False
        if latest.read_text().strip() != f"global_step{step}":
            return False
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if not any(ckpt.glob("model*.safetensors")):
        return False
    if not (ds_dir / "mp_rank_00_model_states.pt").exists():
        return False
    optim = list(ds_dir.glob("*optim_states.pt"))
    rng = list(ckpt.glob("rng_state_*.pth"))
    if len(optim) < num_gpus or len(rng) < num_gpus:
        return False
    if not (ckpt / "scheduler.pt").exists():
        return False
    return dir_quiet_for(ckpt, quiet_seconds)


def local_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in run_dir.iterdir():
        m = CKPT_RE.match(p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)


def write_status(path: Path, status: dict) -> None:
    """Atomically persist state before any remote mutation."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path.parent.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".pending-", delete=False
        ) as f:
            tmp = Path(f.name)
            json.dump(status, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def lfs_sha(f: RepoFile) -> str | None:
    if not f.lfs:
        return None
    return f.lfs["sha256"] if isinstance(f.lfs, dict) else f.lfs.sha256


def remote_folder(
    api: HfApi, repo_id: str, path: str, revision: str | None = None
) -> list[RepoFile]:
    try:
        return [
            f
            for f in api.list_repo_tree(
                repo_id, path, recursive=True, expand=True, revision=revision, repo_type="model"
            )
            if isinstance(f, RepoFile)
        ]
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return []
        raise


def readme(prefix: str, step: int, repo_id: str, dest: str) -> str:
    return f"""# {prefix} -- latest full checkpoint (for resume)

`checkpoint-{step}/` is the most recent complete training checkpoint of `{prefix}`, uploaded as-is from the
training run: model weights (`model-*.safetensors`), processor/config, and the DeepSpeed ZeRO-2 state
(`global_step{step}/`, `latest`, `rng_state_*.pth`, `scheduler.pt`, `training_args.bin`), so training can be
resumed from it. Each new checkpoint logically replaces the previous in a single commit. Historical LFS
objects are retained and still count against storage; automatic irreversible deletion is disabled.
The uploader's local transaction journal retains candidates for offline operator review after all writers
are stopped and every ref is checked. Candidates may still be referenced and are not a deletion allowlist.
For eval-only snapshots at scheduled steps see the sibling `checkpoint-*` folders in `{prefix}/`.

Resume:

```bash
hf download {repo_id} --include "{dest}/checkpoint-{step}/*" --local-dir ckpt
# then point --output-dir at a dir containing checkpoint-{step}/ and pass --resume-from-checkpoint
```

Updated {now()}.
"""


def destination(args) -> str:
    parts = f"{args.path_prefix}/{args.dest_subdir}".split("/")
    if ".." in parts or any("\\" in part for part in parts):
        raise ValueError("Hub paths must not contain '..' or backslashes")
    dest = "/".join(part for part in parts if part not in ("", "."))
    if not dest:
        raise ValueError("the resume destination must be a nonempty folder")
    return dest


def journal_dir(args) -> Path:
    return args.status_file.with_name(args.status_file.name + ".journal")


def write_transaction(path: Path, transaction: dict) -> None:
    payload = json.dumps(transaction, sort_keys=True).encode()
    write_status(
        path,
        {"version": 1, "sha256": hashlib.sha256(payload).hexdigest(), "transaction": transaction},
    )


def load_transactions(args) -> list[tuple[Path, dict]]:
    records = []
    for path in sorted(journal_dir(args).glob("*.json")):
        try:
            envelope = json.loads(path.read_text())
            record = envelope["transaction"]
            checksum = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            if envelope["version"] != 1 or envelope["sha256"] != checksum:
                raise ValueError("invalid checksum or version")
            if (
                record["phase"] not in ("prepared", "committed", "verified", "superseded")
                or not isinstance(record["step"], int)
                or not isinstance(record["repo_id"], str)
                or not isinstance(record["dest"], str)
                or not isinstance(record["files"], dict)
                or not record["files"]
                or not isinstance(record["gc_candidates"], dict)
            ):
                raise ValueError("invalid transaction")
            if record["phase"] == "superseded" and (
                not isinstance(record["superseded_by"], str)
                or not isinstance(record["superseded_by_step"], int)
                or record["superseded_by_step"] <= record["step"]
            ):
                raise ValueError("invalid superseded transaction")
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise RuntimeError(
                f"Unreadable resume journal {path}; restore it from backup before retrying. "
                "Do not discard it: it may contain historical object candidates."
            ) from e
        records.append((path, record))
    by_name = {path.name: record for path, record in records}
    for path, record in records:
        if record["phase"] != "superseded":
            continue
        successor = by_name.get(record["superseded_by"])
        if successor is None or (successor["repo_id"], successor["dest"], successor["step"]) != (
            record["repo_id"],
            record["dest"],
            record["superseded_by_step"],
        ):
            raise RuntimeError(
                f"Missing or mismatched successor journal for {path}; restore it from backup"
            )
    return records


def file_manifest(ckpt: Path) -> dict[str, dict]:
    files = {}
    for src in sorted(ckpt.rglob("*")):
        if not src.is_file():
            continue
        size = src.stat().st_size
        sha = hashlib.sha256()
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with src.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
                blob.update(chunk)
        files[src.relative_to(ckpt).as_posix()] = {
            "size": size,
            "sha256": sha.hexdigest(),
            "blob_id": blob.hexdigest(),
        }
    if not files:
        raise RuntimeError(f"No checkpoint files at {ckpt}; preserve the pending journal")
    return files


def checkpoint_files(files: list[RepoFile], dest: str) -> list[RepoFile]:
    return [
        f
        for f in files
        if f.path.startswith(dest + "/")
        and CKPT_RE.fullmatch(f.path[len(dest) + 1 :].split("/")[0])
    ]


def prepare_replacement(api: HfApi, path: Path, record: dict) -> None:
    # Optimistic concurrency protects the folder replacement, not irreversible object collection.
    head = api.repo_info(record["repo_id"], repo_type="model", revision="main").sha
    old_files = checkpoint_files(
        remote_folder(api, record["repo_id"], record["dest"], head), record["dest"]
    )
    folders = sorted({f.path[len(record["dest"]) + 1 :].split("/")[0] for f in old_files})
    if any(int(CKPT_RE.fullmatch(folder).group(1)) > record["step"] for folder in folders):
        raise RuntimeError("A newer resume checkpoint is already published; refusing to replace it")
    for f in old_files:
        oid = lfs_sha(f)
        if oid:
            candidate = record["gc_candidates"].setdefault(oid, {"size": f.size, "paths": []})
            candidate["paths"] = sorted(set(candidate["paths"]) | {f.path})
    record.update({"parent_commit": head, "replaced": folders})
    write_transaction(path, record)


def verify_remote(api: HfApi, record: dict, revision: str | None) -> bool:
    folder = f"{record['dest']}/checkpoint-{record['step']}"
    remote = {
        f.path: f
        for f in checkpoint_files(
            remote_folder(api, record["repo_id"], record["dest"], revision), record["dest"]
        )
    }
    if set(remote) != {f"{folder}/{rel}" for rel in record["files"]}:
        return False
    for rel, expected in record["files"].items():
        actual = remote.get(f"{folder}/{rel}")
        if actual is None or actual.size != expected["size"]:
            return False
        if lfs_sha(actual):
            if lfs_sha(actual) != expected["sha256"]:
                return False
        elif actual.blob_id != expected["blob_id"]:
            return False
    return True


def update_status(args, status: dict) -> None:
    records = [
        record
        for _, record in load_transactions(args)
        if (record["repo_id"], record["dest"]) == (args.repo_id, destination(args))
    ]
    pending = [record for record in records if record["phase"] in ("prepared", "committed")]
    verified = [record for record in records if record["phase"] == "verified"]
    status["uploaded_step"] = max((record["step"] for record in verified), default=0)
    candidates = {oid for record in records for oid in record["gc_candidates"]}
    for key in ("uploaded_at", "commit", "files", "bytes", "replaced"):
        status.pop(key, None)
    if verified:
        latest = max(verified, key=lambda record: record["step"])
        status.update(
            {
                "uploaded_step": max(status.get("uploaded_step", 0), latest["step"]),
                "uploaded_at": latest["verified_at"],
                "commit": latest.get("commit_url"),
                "files": len(latest["files"]),
                "bytes": sum(f["size"] for f in latest["files"].values()),
                "replaced": latest["replaced"],
            }
        )
    for key in ("repo_lfs_total_bytes", "upload_seconds"):
        status.pop(key, None)
    status.update(
        {
            "repo_id": args.repo_id,
            "repo_path": destination(args),
            "gc_policy": "offline_only",
            "gc_journal": str(journal_dir(args)),
            "pending_gc_candidates": sorted(candidates),
            "superseded_steps": sorted(
                {record["step"] for record in records if record["phase"] == "superseded"}
            ),
            "lfs_objects_deleted": 0,
            "freed_bytes": 0,
            "pending_step": pending[0]["step"] if pending else None,
            "pending_since": pending[0]["created_at"] if pending else None,
            "last_error": "",
        }
    )
    write_status(args.status_file, status)


def finish_transaction(api: HfApi, path: Path, record: dict, recovering: bool = True) -> None:
    if record["phase"] == "prepared":
        # A lost commit response is recoverable without re-uploading or the local checkpoint.
        head = (
            api.repo_info(record["repo_id"], repo_type="model", revision="main").sha
            if recovering
            else record["parent_commit"]
        )
        if recovering and verify_remote(api, record, head):
            record.update({"phase": "committed", "commit_oid": head})
            write_transaction(path, record)
        else:
            ckpt = Path(record["checkpoint"])
            if recovering:
                try:
                    available = file_manifest(ckpt) == record["files"]
                except (OSError, RuntimeError) as e:
                    raise PendingCheckpointUnavailable(
                        f"Pending checkpoint unavailable at {ckpt}; restore it or provide a newer complete checkpoint"
                    ) from e
                if not available:
                    raise PendingCheckpointUnavailable(
                        "Pending checkpoint changed; restore it or provide a newer complete checkpoint"
                    )
                prepare_replacement(api, path, record)
            folder = f"{record['dest']}/checkpoint-{record['step']}"
            ops = [
                CommitOperationDelete(path_in_repo=f"{record['dest']}/{old}/", is_folder=True)
                for old in record["replaced"]
            ]
            ops.extend(
                CommitOperationAdd(path_in_repo=f"{folder}/{rel}", path_or_fileobj=str(ckpt / rel))
                for rel in record["files"]
            )
            ops.append(
                CommitOperationAdd(
                    path_in_repo=f"{record['dest']}/README.md",
                    path_or_fileobj=readme(
                        record["prefix"], record["step"], record["repo_id"], record["dest"]
                    ).encode(),
                )
            )
            info = api.create_commit(
                repo_id=record["repo_id"],
                repo_type="model",
                revision="main",
                operations=ops,
                parent_commit=record["parent_commit"],
                commit_message=f"{record['dest']}: full checkpoint {record['step']} (history retained)",
                commit_description="Logical checkpoint replacement only; automatic irreversible GC is disabled.",
            )
            record.update(
                {"phase": "committed", "commit_oid": info.oid, "commit_url": info.commit_url}
            )
            write_transaction(path, record)
    if not verify_remote(api, record, record["commit_oid"]):
        raise RuntimeError(
            "Checkpoint verification failed; pending transaction retained for recovery"
        )
    record.update({"phase": "verified", "verified_at": now()})
    write_transaction(path, record)


def prepare_upload(api: HfApi, args, step: int, ckpt: Path) -> tuple[Path, dict]:
    record = {
        "repo_id": args.repo_id,
        "dest": destination(args),
        "prefix": "/".join(part for part in args.path_prefix.split("/") if part not in ("", ".")),
        "step": step,
        "checkpoint": str(ckpt.resolve()),
        "files": file_manifest(ckpt),
        "gc_candidates": {},
        "phase": "prepared",
        "created_at": now(),
    }
    path = journal_dir(args) / f"{uuid.uuid4().hex}.json"
    prepare_replacement(api, path, record)
    return path, record


def newer_complete_checkpoint(
    args, step: int, successor: tuple[int, Path] | None
) -> tuple[int, Path] | None:
    candidates = [successor] if successor is not None else []
    run_dir = getattr(args, "run_dir", None)
    if run_dir is not None and Path(run_dir).is_dir():
        candidates.extend(local_checkpoints(Path(run_dir)))
    for candidate_step, checkpoint in sorted(candidates, reverse=True):
        if candidate_step <= step:
            continue
        try:
            if full_checkpoint_complete(
                checkpoint,
                candidate_step,
                getattr(args, "num_gpus", 4),
                getattr(args, "quiet_seconds", 120),
            ):
                return candidate_step, checkpoint
        except OSError:
            continue
    return None


def recover_pending(
    api: HfApi, args, status: dict, successor: tuple[int, Path] | None = None
) -> None:
    while True:
        records = [
            (path, record)
            for path, record in load_transactions(args)
            if (record["repo_id"], record["dest"]) == (args.repo_id, destination(args))
        ]
        pending = [
            (path, record)
            for path, record in records
            if record["phase"] in ("prepared", "committed")
        ]
        if not pending:
            break
        path, record = min(pending, key=lambda item: item[1]["step"])
        try:
            finish_transaction(api, path, record)
        except PendingCheckpointUnavailable:
            # finish_transaction checks the remote manifest before consulting the local source.
            replacement = newer_complete_checkpoint(args, record["step"], successor)
            if replacement is None:
                raise
            replacement_step, checkpoint = replacement
            existing = [
                (p, r)
                for p, r in records
                if r["step"] == replacement_step and r["phase"] != "superseded"
            ]
            if existing:
                replacement_path, _ = existing[0]
            else:
                replacement_path, _ = prepare_upload(api, args, replacement_step, checkpoint)
            # Persist the successor first so a crash cannot leave only a retired transaction.
            record.update(
                {
                    "phase": "superseded",
                    "superseded_at": now(),
                    "superseded_by": replacement_path.name,
                    "superseded_by_step": replacement_step,
                    "superseded_reason": "local checkpoint unavailable; newer complete checkpoint selected",
                }
            )
            write_transaction(path, record)
            update_status(args, status)
            log.warning(
                "superseded unavailable checkpoint %d with complete checkpoint %d; GC candidates retained",
                record["step"],
                replacement_step,
            )
    update_status(args, status)


def upload_checkpoint(api: HfApi, args, step: int, ckpt: Path, status: dict) -> None:
    recover_pending(api, args, status, successor=(step, ckpt))
    if status["uploaded_step"] >= step:
        return
    path, record = prepare_upload(api, args, step, ckpt)
    update_status(args, status)
    finish_transaction(api, path, record, recovering=False)
    update_status(args, status)
    log.info(
        "verified full checkpoint %d; history retained, GC candidates require offline review", step
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--path-prefix", required=True, help="experiment folder inside the repo")
    p.add_argument(
        "--dest-subdir",
        default="resume",
        help="subfolder of --path-prefix holding the checkpoint",
    )
    p.add_argument("--num-gpus", type=int, default=4, help="ranks expected in the DeepSpeed state")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="exit after this step's checkpoint is on the Hub",
    )
    p.add_argument("--poll-seconds", type=float, default=60)
    p.add_argument(
        "--quiet-seconds",
        type=float,
        default=120,
        help="checkpoint dir must be unchanged this long",
    )
    p.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="where resume-status.json and its durable .journal/ are kept (default: the run dir)",
    )
    p.add_argument(
        "--status-file", type=Path, default=None, help="overrides <staging-dir>/resume-status.json"
    )
    args = p.parse_args()
    if args.status_file is None:
        args.status_file = (args.staging_dir or args.run_dir) / "resume-status.json"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    if not os.environ.get("HF_TOKEN"):
        log.error("HF_TOKEN is not set")
        return 2
    api = HfApi()
    dest = destination(args)
    try:
        status = json.loads(args.status_file.read_text())
        if not isinstance(status, dict):
            status = {}
    except (OSError, ValueError):
        status = {}
    if (status.get("repo_id"), status.get("repo_path")) != (args.repo_id, dest):
        status = {}
    status.setdefault("uploaded_step", 0)
    log.info(
        "watching %s -> %s/%s (latest on Hub: %s)",
        args.run_dir,
        args.repo_id,
        dest,
        status["uploaded_step"] or "none",
    )
    backoff = args.poll_seconds
    while True:
        try:
            recover_pending(api, args, status)
            if args.max_steps and status["uploaded_step"] >= args.max_steps:
                log.info(
                    "final checkpoint %d verified; historical objects retained",
                    status["uploaded_step"],
                )
                return 0
            complete = [
                (s, c)
                for s, c in local_checkpoints(args.run_dir)
                if s > status["uploaded_step"]
                and full_checkpoint_complete(c, s, args.num_gpus, args.quiet_seconds)
            ]
            if complete:
                step, ckpt = complete[-1]  # newest complete one; skip intermediate ones
                upload_checkpoint(api, args, step, ckpt, status)
                backoff = args.poll_seconds
                if args.max_steps and step >= args.max_steps:
                    log.info("final checkpoint %d is on the Hub; exiting", step)
                    return 0
        except Exception as e:  # noqa: BLE001 - keep the monitor alive
            status["last_error"] = f"{now()} {type(e).__name__}: {str(e)[:500]}"
            write_status(args.status_file, status)
            log.error("cycle failed: %s; retrying in %.0fs", status["last_error"], backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 900)
            continue
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
