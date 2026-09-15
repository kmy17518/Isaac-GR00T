#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the latest *full* (resumable) checkpoint of a run on the Hub -- and only that one.

Complements hf-checkpoint-uploader.py (eval-only copies at scheduled steps). Every poll it looks for
the newest *complete* ``checkpoint-<step>`` in the run dir that is newer than the one on the Hub and
replaces the Hub copy in a single commit:

  * adds ``<prefix>/resume/checkpoint-<step>/**`` -- everything, including the DeepSpeed
    ``global_step<step>/`` optimizer/model partitions, ``latest``, ``rng_state_*.pth``, ``scheduler.pt``,
    ``training_args.bin`` -- so ``--resume-from-checkpoint`` works on a downloaded copy;
  * deletes the previous ``<prefix>/resume/checkpoint-*`` folder(s) in the same commit.

Deleting files on the Hub does not free storage: the LFS objects stay referenced by history and count
against the quota. So after each swap the LFS objects that the removed checkpoint owned and that no file
in the current tree references any more are removed for good with
``HfApi.permanently_delete_lfs_files(..., rewrite_history=True)``, which also rewrites the history so no
commit points at them. Objects still referenced elsewhere -- e.g. the eval-only copy of the same step
shares byte-identical ``model-*.safetensors`` -- are never touched. The status file records what was
uploaded, what was freed, and the repo's LFS total.

Usage (run it detached, e.g. in tmux, next to the training job):
    python scripts/b1k/hf_resume_checkpoint_uploader.py --run-dir $OUTPUT_DIR/<exp> \
        --staging-dir $STAGING_DIR/<exp> --repo-id <user>/<repo> --path-prefix <exp> \
        --num-gpus 4 --max-steps 150000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import HfHubHTTPError


log = logging.getLogger("hf-resume-uploader")
CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True))
    os.replace(tmp, path)


def lfs_sha(f: RepoFile) -> str | None:
    if not f.lfs:
        return None
    return f.lfs["sha256"] if isinstance(f.lfs, dict) else f.lfs.sha256


def remote_folder(api: HfApi, repo_id: str, path: str) -> list[RepoFile]:
    try:
        return [
            f
            for f in api.list_repo_tree(repo_id, path, recursive=True, expand=True)
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
resumed from it. Only one such checkpoint is kept: each new one replaces the previous in a single commit and
the previous checkpoint's LFS objects are permanently deleted (history rewritten) so they do not count
against storage. For eval-only snapshots at scheduled steps see the sibling `checkpoint-*` folders in
`{prefix}/`.

Resume:

```bash
hf download {repo_id} --include "{dest}/checkpoint-{step}/*" --local-dir ckpt
# then point --output-dir at a dir containing checkpoint-{step}/ and pass --resume-from-checkpoint
```

Updated {now()}.
"""


def upload_checkpoint(api: HfApi, args, step: int, ckpt: Path, status: dict) -> None:
    dest = f"{args.path_prefix.strip('/')}/{args.dest_subdir.strip('/')}"
    new_folder = f"{dest}/checkpoint-{step}"

    # what the Hub holds now (paths + LFS object ids), to delete and later garbage-collect
    old_files = remote_folder(api, args.repo_id, dest)
    old_folders = sorted(
        {
            f.path.split("/")[len(dest.split("/"))]
            for f in old_files
            if f.path.startswith(dest + "/checkpoint-")
        }
    )
    old_oids = {sha for f in old_files if (sha := lfs_sha(f))}

    ops: list = []
    n_bytes = 0
    files = []
    for root, _dirs, names in os.walk(ckpt):
        for name in names:
            src = Path(root) / name
            rel = src.relative_to(ckpt).as_posix()
            files.append((src, rel))
            n_bytes += src.stat().st_size
            ops.append(
                CommitOperationAdd(path_in_repo=f"{new_folder}/{rel}", path_or_fileobj=str(src))
            )
    ops.append(
        CommitOperationAdd(
            path_in_repo=f"{dest}/README.md",
            path_or_fileobj=readme(args.path_prefix, step, args.repo_id, dest).encode(),
        )
    )
    for folder in old_folders:
        if folder != f"checkpoint-{step}":
            ops.append(CommitOperationDelete(path_in_repo=f"{dest}/{folder}/", is_folder=True))
    log.info(
        "uploading %s: %d files, %.1f GB -> %s/%s (replacing %s)",
        ckpt.name,
        len(files),
        n_bytes / 1e9,
        args.repo_id,
        new_folder,
        old_folders or "nothing",
    )
    status.update({"pending_step": step, "pending_since": now()})
    write_status(args.status_file, status)
    t0 = time.time()
    info = api.create_commit(
        repo_id=args.repo_id,
        operations=ops,
        commit_message=f"{args.path_prefix}: full checkpoint {step} for resume"
        + (f" (replaces {', '.join(old_folders)})" if old_folders else ""),
    )
    log.info("committed in %.0fs: %s", time.time() - t0, info.commit_url)

    # verify every file landed with the right size
    remote = {f.path: f for f in remote_folder(api, args.repo_id, new_folder)}
    missing = [rel for src, rel in files if f"{new_folder}/{rel}" not in remote]
    wrong = [
        rel
        for src, rel in files
        if f"{new_folder}/{rel}" in remote
        and remote[f"{new_folder}/{rel}"].size not in (None, src.stat().st_size)
    ]
    if missing or wrong:
        raise RuntimeError(f"verification failed: missing {missing[:5]} size-mismatch {wrong[:5]}")

    # garbage-collect the replaced checkpoint's LFS objects that nothing references any more
    freed = 0
    deleted = 0
    if old_oids:
        tree_oids = {
            sha
            for f in api.list_repo_tree(args.repo_id, recursive=True, expand=True)
            if isinstance(f, RepoFile) and (sha := lfs_sha(f))
        }
        victims = [
            x
            for x in api.list_lfs_files(args.repo_id)
            if x.file_oid in old_oids and x.file_oid not in tree_oids
        ]
        if victims:
            api.permanently_delete_lfs_files(args.repo_id, victims, rewrite_history=True)
            deleted = len(victims)
            freed = sum(x.size for x in victims)
            log.info(
                "permanently deleted %d LFS objects (%.1f GB) of the replaced checkpoint, history rewritten",
                deleted,
                freed / 1e9,
            )
        else:
            log.info("no LFS objects to delete (all still referenced or none existed)")
    lfs_total = sum(x.size for x in api.list_lfs_files(args.repo_id))
    status.update(
        {
            "uploaded_step": step,
            "uploaded_at": now(),
            "commit": info.commit_url,
            "files": len(files),
            "bytes": n_bytes,
            "replaced": old_folders,
            "lfs_objects_deleted": deleted,
            "freed_bytes": freed,
            "repo_lfs_total_bytes": lfs_total,
            "pending_step": None,
            "pending_since": None,
            "last_error": "",
            "upload_seconds": round(time.time() - t0),
        }
    )
    write_status(args.status_file, status)
    log.info(
        "repo LFS total now %.1f GB; latest full checkpoint on the Hub: %d",
        lfs_total / 1e9,
        step,
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
        help="where resume-status.json is written (default: the run dir)",
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
    status = json.loads(args.status_file.read_text()) if args.status_file.exists() else {}
    status.setdefault("uploaded_step", 0)
    # what is on the Hub already (in case the status file was lost)
    dest = f"{args.path_prefix.strip('/')}/{args.dest_subdir.strip('/')}"
    try:
        remote_steps = sorted(
            {
                int(m.group(1))
                for f in remote_folder(api, args.repo_id, dest)
                for m in [CKPT_RE.match(f.path.split("/")[len(dest.split("/"))])]
                if m
            }
        )
        if remote_steps:
            status["uploaded_step"] = max(status["uploaded_step"], max(remote_steps))
    except HfHubHTTPError as e:
        log.warning("could not list %s: %s", dest, e)
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
