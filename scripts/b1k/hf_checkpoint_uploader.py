#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watch a GR00T training run dir, stage eval-only checkpoint copies, upload them to the HF Hub.

Runs forever (in tmux) next to the training job:

  1. Every --poll-seconds, list ``<run-dir>/checkpoint-<step>/``. Steps that match the schedule
     (default: every 10k up to 50k, then every 5k) are staged as soon as the checkpoint is
     complete (trainer_state.json + processor_config.json present, directory quiet for a while).
  2. Staging copies ONLY what ``Gr00tPolicy`` / ``serve_b1k.py`` need to run the model --
     weights, model + processor configs, statistics, experiment_cfg -- and skips the resume
     state (DeepSpeed ``global_step*/``, rng, optimizer/scheduler, training_args.bin, ...).
     Files are hard-linked (same filesystem), so staging is instant and the staged copy survives
     the trainer deleting the checkpoint under ``save_total_limit``.
  3. Each staged checkpoint is uploaded to ``<repo-id>/<path-prefix>/checkpoint-<step>/`` (one repo
     can hold several experiments as folders; ``--path-prefix`` is the experiment folder) and two
     README.md model cards are refreshed: ``<path-prefix>/README.md`` (this experiment, with a
     table of uploaded checkpoints) and the repo-root ``README.md`` (index of experiment folders).
     Upload failures are logged and retried with backoff; the staged copy is never lost. Repo id
     and prefix can be changed at runtime through ``<staging-dir>/upload_config.json``
     (``{"repo_id": "...", "path_prefix": "...", "enabled": true}``).

State lives in ``<staging-dir>/status.json`` (human readable) and destination-scoped ``.uploaded``
markers. Changing destinations uploads the staged copies there; switching back reuses valid records.
Unreadable markers are treated as pending, not as successful uploads.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


LOG = logging.getLogger("uploader")

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

# Resume-only state that eval never reads. Everything else in a checkpoint is kept.
SKIP_DIR_PREFIXES = ("global_step",)
SKIP_FILE_PATTERNS = (
    re.compile(r"^rng_state.*\.pth$"),
    re.compile(r"^optimizer.*\.(pt|bin)$"),
    re.compile(r"^scheduler\.pt$"),
    re.compile(r"^training_args\.bin$"),
    re.compile(r"^latest$"),
    re.compile(r"^zero_to_fp32\.py$"),
    re.compile(r"^.*\.pth$"),
)
# Files eval actually opens; used to sanity check a staged copy before upload.
REQUIRED_FILES = ("config.json", "processor_config.json", "statistics.json")


def scheduled(step: int, switch_step: int, early_every: int, late_every: int) -> bool:
    if step <= 0:
        return False
    if step <= switch_step:
        return step % early_every == 0
    return step % late_every == 0


def scheduled_steps(
    max_steps: int, switch_step: int, early_every: int, late_every: int
) -> list[int]:
    return [
        s for s in range(1, max_steps + 1) if scheduled(s, switch_step, early_every, late_every)
    ]


def dir_quiet_for(path: Path, seconds: float) -> bool:
    """True when nothing under ``path`` was modified in the last ``seconds``."""
    newest = path.stat().st_mtime
    for root, _dirs, files in os.walk(path):
        newest = max(newest, Path(root).stat().st_mtime)
        for f in files:
            try:
                newest = max(newest, (Path(root) / f).stat().st_mtime)
            except FileNotFoundError:
                return False
    return time.time() - newest >= seconds


def checkpoint_complete(ckpt: Path, step: int, quiet_seconds: float) -> bool:
    state_file = ckpt / "trainer_state.json"
    if not state_file.exists() or not (ckpt / "processor_config.json").exists():
        return False
    try:
        state = json.loads(state_file.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if int(state.get("global_step", -1)) != step:
        return False
    has_weights = any(ckpt.glob("model*.safetensors")) or (ckpt / "pytorch_model.bin").exists()
    return has_weights and dir_quiet_for(ckpt, quiet_seconds)


def should_skip(rel: Path) -> bool:
    if any(part.startswith(SKIP_DIR_PREFIXES) for part in rel.parts[:-1]):
        return True
    if rel.parts and rel.parts[0].startswith(SKIP_DIR_PREFIXES):
        return True
    return any(p.match(rel.name) for p in SKIP_FILE_PATTERNS)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_checkpoint(ckpt: Path, dst: Path) -> tuple[int, int]:
    """Hard-link the eval-needed files of ``ckpt`` into ``dst``. Returns (files, bytes)."""
    tmp = dst.with_name(dst.name + ".staging")
    if tmp.exists():
        shutil.rmtree(tmp)
    n_files = n_bytes = 0
    for root, _dirs, files in os.walk(ckpt):
        for f in files:
            src = Path(root) / f
            rel = src.relative_to(ckpt)
            if should_skip(rel):
                continue
            link_or_copy(src, tmp / rel)
            n_files += 1
            n_bytes += src.stat().st_size
    missing = [f for f in REQUIRED_FILES if not (tmp / f).exists()]
    if missing:
        shutil.rmtree(tmp)
        raise RuntimeError(f"{ckpt} is missing eval files {missing}; not staging")
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return n_files, n_bytes


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def read_trainer_loss(staged: Path) -> float | None:
    try:
        state = json.loads((staged / "trainer_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for entry in reversed(state.get("log_history", [])):
        if "loss" in entry:
            return float(entry["loss"])
    return None


def normalize_repo_path(path: str) -> str:
    if not isinstance(path, str) or "\\" in path or ".." in path.split("/"):
        raise ValueError("Hub paths must be relative paths without '..' or backslashes")
    return "/".join(part for part in path.split("/") if part not in ("", "."))


def repo_path(prefix: str, name: str) -> str:
    return normalize_repo_path(f"{prefix}/{name}")


def upload_records(checkpoint: Path) -> list[dict]:
    try:
        data = json.loads((checkpoint / ".uploaded").read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    if "version" in data and data["version"] != 2:
        return []
    records = data.get("destinations") if data.get("version") == 2 else [data]
    if not isinstance(records, list):
        return []
    valid = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("repo_id"), str):
            continue
        try:
            path = normalize_repo_path(record.get("repo_path"))
        except ValueError:
            continue
        if record["repo_id"] and path:
            valid.append({**record, "repo_path": path})
    return valid


def uploaded_record(checkpoint: Path, repo_id: str, prefix: str) -> dict | None:
    dest = repo_path(prefix, checkpoint.name)
    return next(
        (
            record
            for record in upload_records(checkpoint)
            if record["repo_id"] == repo_id and record["repo_path"] == dest
        ),
        None,
    )


def write_upload_record(checkpoint: Path, record: dict) -> None:
    records = [
        old
        for old in upload_records(checkpoint)
        if (old["repo_id"], old["repo_path"]) != (record["repo_id"], record["repo_path"])
    ]
    records.append(record)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=checkpoint, prefix=".uploaded.", delete=False
        ) as f:
            tmp = Path(f.name)
            json.dump({"version": 2, "destinations": records}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, checkpoint / ".uploaded")
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


FRONT_MATTER = """---
license: apache-2.0
base_model: nvidia/GR00T-N1.7-3B
tags:
  - robotics
  - vla
  - gr00t
  - behavior-1k
  - b1k-challenge-2026
datasets:
  - behavior-1k/2026-challenge-demos
---
"""


def render_root_readme(api: HfApi, repo_id: str, prefix: str) -> str:
    """Repo-root card: an index of the experiment folders currently in the repo."""
    folders: dict[str, int] = {}
    try:
        for item in api.list_repo_tree(repo_id, repo_type="model"):
            if item.__class__.__name__ == "RepoFolder":
                n = sum(
                    1
                    for sub in api.list_repo_tree(
                        repo_id, path_in_repo=item.path, repo_type="model"
                    )
                    if sub.__class__.__name__ == "RepoFolder"
                    and CHECKPOINT_RE.match(sub.path.split("/")[-1])
                )
                folders[item.path] = n
    except HfHubHTTPError:
        pass
    if prefix.strip("/"):
        folders.setdefault(prefix.strip("/"), 0)
    rows = (
        "\n".join(f"| [`{f}/`](./{f}) | {n} |" for f, n in sorted(folders.items()))
        or "| (none yet) | |"
    )
    return f"""{FRONT_MATTER}
# {repo_id.split("/")[-1]}

GR00T N1.7 fine-tunes for the BEHAVIOR-1K 2026 challenge, one folder per experiment. Each folder has its own
`README.md` with the training recipe and a table of its `checkpoint-<step>/` subfolders, each of which is a
standalone, directly servable GR00T checkpoint (weights + processor config + normalization statistics).

| experiment folder | uploaded checkpoints |
|---|---|
{rows}

```bash
hf download {repo_id} --include "<experiment>/checkpoint-<step>/**" --local-dir ckpt
```
"""


def render_readme(
    args: argparse.Namespace, uploaded: dict[str, dict], repo_id: str, prefix: str
) -> str:
    rows = []
    for name in sorted(uploaded, key=lambda n: int(CHECKPOINT_RE.match(n).group(1))):
        info = uploaded[name]
        loss = info.get("train_loss")
        rows.append(
            f"| `{name}/` | {int(CHECKPOINT_RE.match(name).group(1)):,} | "
            f"{'' if loss is None else f'{loss:.4f}'} | {info.get('uploaded_at', '')} |"
        )
    table = "\n".join(rows) if rows else "| (none yet) | | | |"
    sched = scheduled_steps(args.max_steps, args.switch_step, args.early_every, args.late_every)
    folder = prefix.strip("/") or "."
    title = prefix.strip("/") or repo_id.split("/")[-1]
    return f"""{FRONT_MATTER}
# {title}

GR00T N1.7 fine-tuned on the single BEHAVIOR-1K 2026 challenge task **`{args.task}`**
(task 0 of [behavior-1k/2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos),
200 demos / ~430k frames), starting from [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B).

- Global batch size **{args.global_bs}** ({args.num_gpus} GPUs), {args.max_steps:,} steps, lr 1e-4 cosine, warmup 5%;
  frozen Qwen3-VL backbone, trained action head (projector + DiT), relative arm/torso actions,
  prompt = `task_name` (`{args.task}`), embodiment tag `NEW_EMBODIMENT`, modality config `examples/b1k/r1pro.py`.
- Trained with the `scripts/b1k/train_b1k.py --task-names {args.task}` recipe of the Isaac-GR00T B1K
  branch (see `getting_started/b1k.md` there); W&B project `{args.wandb_project}`, run `{args.experiment_name}`.

## Checkpoints

Each `{folder}/checkpoint-<step>/` is a standalone, directly servable model directory (weights, `config.json`,
processor config + normalization statistics, `experiment_cfg/`). Optimizer / DeepSpeed resume state is
**not** included. Upload schedule: every {args.early_every:,} steps up to {args.switch_step:,}, then every
{args.late_every:,} steps ({len(sched)} checkpoints total when training finishes).

| checkpoint | step | train loss | uploaded (UTC) |
|---|---|---|---|
{table}

## Serve for evaluation

```bash
hf download {repo_id} --include "{repo_path(prefix, "checkpoint-<step>")}/**" --local-dir ckpt
CUDA_VISIBLE_DEVICES=0 python scripts/b1k/serve_b1k.py \\
    --model-path ckpt/{repo_path(prefix, "checkpoint-<step>")} \\
    --modality-config-path examples/b1k/r1pro.py \\
    --embodiment-tag NEW_EMBODIMENT --host 127.0.0.1 --port 8000
```

Then run the BEHAVIOR-1K evaluator against `127.0.0.1:8000` (see `getting_started/b1k.md`).
"""


def load_runtime_config(staging: Path) -> dict:
    cfg_file = staging / "upload_config.json"
    if cfg_file.exists():
        try:
            return json.loads(cfg_file.read_text())
        except (OSError, json.JSONDecodeError) as e:
            LOG.warning("ignoring unreadable %s: %s", cfg_file, e)
    return {}


def write_status(staging: Path, status: dict) -> None:
    status["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    tmp = staging / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True))
    tmp.replace(staging / "status.json")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-dir", required=True, help="training output dir holding checkpoint-<step>/"
    )
    p.add_argument("--staging-dir", required=True, help="where eval-only copies are kept")
    p.add_argument("--repo-id", required=True, help="HF repo, e.g. user/name (public)")
    p.add_argument(
        "--path-prefix",
        default="",
        help="folder inside the repo that holds this experiment's checkpoint-<step>/ dirs ('' = repo root)",
    )
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--switch-step", type=int, default=50_000)
    p.add_argument("--early-every", type=int, default=10_000)
    p.add_argument("--late-every", type=int, default=5_000)
    p.add_argument("--poll-seconds", type=float, default=60)
    p.add_argument(
        "--quiet-seconds", type=float, default=90, help="checkpoint dir must be idle this long"
    )
    p.add_argument(
        "--retry-seconds", type=float, default=600, help="min gap between failed upload retries"
    )
    # metadata for the model card
    p.add_argument("--task", default="turning_on_radio")
    p.add_argument("--global-bs", type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--experiment-name", default="")
    p.add_argument("--wandb-project", default="b1k-challenge-2026-gr00t")
    p.add_argument("--once", action="store_true", help="single pass (for testing)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_dir = Path(args.run_dir)
    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    if not token:
        LOG.error("HF_TOKEN is not set")
        return 2
    api = HfApi(token=token)

    schedule = scheduled_steps(args.max_steps, args.switch_step, args.early_every, args.late_every)
    LOG.info(
        "watching %s -> staging %s -> %s/%s | schedule: %d checkpoints %s",
        run_dir,
        staging,
        args.repo_id,
        args.path_prefix,
        len(schedule),
        schedule[:6] + (["..."] if len(schedule) > 6 else []),
    )

    last_upload_attempt = 0.0
    last_error = ""
    repo_ready_for: tuple[str, str] | None = None  # (repo id, prefix) whose repo + cards exist

    def ensure_repo(repo_id: str, prefix: str, uploaded: dict[str, dict]) -> None:
        """Create the public repo (idempotent) and publish both README cards so the destination is
        visible before the first checkpoint lands."""
        nonlocal repo_ready_for
        if repo_ready_for == (repo_id, prefix):
            return
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=render_readme(args, uploaded, repo_id, prefix).encode(),
            path_in_repo=repo_path(prefix, "README.md"),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Model card for {prefix or 'repo'} ({len(uploaded)} checkpoints)",
        )
        if prefix.strip("/"):
            api.upload_file(
                path_or_fileobj=render_root_readme(api, repo_id, prefix).encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update experiment index",
            )
        repo_ready_for = (repo_id, prefix)
        LOG.info("repo ready: https://huggingface.co/%s/tree/main/%s", repo_id, prefix.strip("/"))

    while True:
        runtime_cfg = load_runtime_config(staging)
        repo_id = runtime_cfg.get("repo_id", args.repo_id)
        prefix = normalize_repo_path(runtime_cfg.get("path_prefix", args.path_prefix))
        uploads_enabled = bool(runtime_cfg.get("enabled", True))

        # ---- 1. stage completed, scheduled checkpoints ---------------------------------------
        staged_now = []
        if run_dir.exists():
            for entry in sorted(run_dir.iterdir()):
                m = CHECKPOINT_RE.match(entry.name)
                if not m or not entry.is_dir():
                    continue
                step = int(m.group(1))
                if not scheduled(step, args.switch_step, args.early_every, args.late_every):
                    continue
                dst = staging / entry.name
                if dst.exists():
                    continue
                if not checkpoint_complete(entry, step, args.quiet_seconds):
                    LOG.info("%s not complete/quiet yet; waiting", entry.name)
                    continue
                try:
                    n_files, n_bytes = stage_checkpoint(entry, dst)
                except Exception as e:  # noqa: BLE001
                    LOG.error("staging %s failed: %s", entry.name, e)
                    last_error = f"stage {entry.name}: {e}"
                    continue
                LOG.info("staged %s (%d files, %s) -> %s", entry.name, n_files, human(n_bytes), dst)
                staged_now.append(entry.name)

        # ---- 2. upload staged checkpoints pending at this destination ------------------------
        staged = sorted(
            (d for d in staging.iterdir() if d.is_dir() and CHECKPOINT_RE.match(d.name)),
            key=lambda d: int(CHECKPOINT_RE.match(d.name).group(1)),
        )
        uploaded: dict[str, dict] = {}
        pending = []
        for d in staged:
            record = uploaded_record(d, repo_id, prefix)
            if record is not None:
                uploaded[d.name] = record
            else:
                pending.append(d)

        now = time.time()
        need_attempt = (
            pending and (now - last_upload_attempt >= args.retry_seconds or staged_now)
        ) or (
            repo_ready_for != (repo_id, prefix) and now - last_upload_attempt >= args.retry_seconds
        )
        if uploads_enabled and need_attempt:
            last_upload_attempt = now
            try:
                ensure_repo(repo_id, prefix, uploaded)
                for d in pending:
                    t0 = time.time()
                    dest = repo_path(prefix, d.name)
                    LOG.info("uploading %s -> %s/%s ...", d, repo_id, dest)
                    info = api.upload_folder(
                        folder_path=str(d),
                        path_in_repo=dest,
                        repo_id=repo_id,
                        repo_type="model",
                        ignore_patterns=[".uploaded", ".uploaded.*"],
                        commit_message=f"Add {dest} ({args.experiment_name})",
                    )
                    rec = {
                        "repo_id": repo_id,
                        "repo_path": dest,
                        "commit_url": getattr(info, "commit_url", str(info)),
                        "uploaded_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "train_loss": read_trainer_loss(d),
                        "seconds": round(time.time() - t0, 1),
                    }
                    write_upload_record(d, rec)
                    uploaded[d.name] = rec
                    LOG.info("uploaded %s in %.0fs: %s", dest, rec["seconds"], rec["commit_url"])
                    # refresh both model cards after every checkpoint
                    repo_ready_for = None
                    ensure_repo(repo_id, prefix, uploaded)
                last_error = ""
            except HfHubHTTPError as e:
                code = e.response.status_code if e.response is not None else "?"
                detail = " | ".join(
                    line.strip() for line in str(e).splitlines()[:2] if line.strip()
                )
                last_error = f"HTTP {code}: {detail[:400]}"
                if code == 403:
                    LOG.error(
                        "cannot write to %s (403). The namespace must exist and the HF_TOKEN account must be "
                        "a member with write access. Staged checkpoints are kept; retrying every %.0fs. "
                        'To upload elsewhere, write {"repo_id": "<user-or-org>/<name>", '
                        '"path_prefix": "<folder>"} to %s',
                        repo_id,
                        args.retry_seconds,
                        staging / "upload_config.json",
                    )
                else:
                    LOG.error("upload failed: %s (retry in %.0fs)", last_error, args.retry_seconds)
                repo_ready_for = None
            except Exception as e:  # noqa: BLE001
                last_error = f"{type(e).__name__}: {e}"
                LOG.exception("upload failed (retry in %.0fs)", args.retry_seconds)
                repo_ready_for = None
        elif pending and not uploads_enabled:
            LOG.info(
                "uploads disabled via upload_config.json; %d staged checkpoints pending",
                len(pending),
            )

        # ---- 3. status + exit condition ------------------------------------------------------
        remaining = [s for s in schedule if f"checkpoint-{s}" not in uploaded]
        write_status(
            staging,
            {
                "run_dir": str(run_dir),
                "repo_id": repo_id,
                "path_prefix": prefix,
                "repo_url": f"https://huggingface.co/{repo_id}/tree/main/{prefix.strip('/')}".rstrip(
                    "/"
                ),
                "schedule": schedule,
                "uploaded": sorted(uploaded, key=lambda n: int(CHECKPOINT_RE.match(n).group(1))),
                "pending_upload": [d.name for d in pending if d.name not in uploaded],
                "remaining_scheduled_steps": remaining,
                "last_error": last_error,
                "uploads_enabled": uploads_enabled,
            },
        )
        if not remaining:
            LOG.info("all %d scheduled checkpoints uploaded; done", len(schedule))
            return 0
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
