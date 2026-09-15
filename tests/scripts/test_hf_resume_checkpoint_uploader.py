# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline Hub replacement and durable recovery state-machine tests."""

import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace

from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.hf_api import RepoFile
import pytest
from scripts.b1k import hf_resume_checkpoint_uploader as uploader


def repofile(path, payload, lfs=True):
    metadata = None
    if lfs:
        metadata = {
            "size": len(payload),
            "oid": hashlib.sha256(payload).hexdigest(),
            "pointerSize": 1,
        }
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    return RepoFile(path=path, size=len(payload), oid=blob, lfs=metadata)


class MemoryHub:
    def __init__(self):
        old = repofile("exp/resume/checkpoint-100/model.safetensors", b"old")
        self.tree = {old.path: old}
        self.objects = {uploader.lfs_sha(old)}
        self.head = "initial"
        self.revisions = {self.head: dict(self.tree)}
        self.commits = []
        self.listings = []
        self.before_commit = None
        self.after_commit = None
        self.fail_commit = False
        self.lose_response = False
        self.fail_verification = False
        self.deleted = []

    def repo_info(self, repo_id, **kwargs):
        return SimpleNamespace(sha=self.head)

    def snapshot(self):
        self.head = f"revision-{len(self.revisions)}"
        self.revisions[self.head] = dict(self.tree)
        return self.head

    def list_repo_tree(self, repo_id, path_in_repo=None, revision=None, **kwargs):
        self.listings.append((path_in_repo, revision))
        if self.fail_verification and self.commits:
            self.fail_verification = False
            raise RuntimeError("transient list failure")
        tree = self.revisions[revision] if revision not in (None, "main") else self.tree
        return [
            f
            for path, f in tree.items()
            if path_in_repo is None or path.startswith(path_in_repo + "/")
        ]

    def create_commit(self, repo_id, operations, parent_commit, **kwargs):
        if self.before_commit:
            hook, self.before_commit = self.before_commit, None
            hook()
        if self.fail_commit:
            self.fail_commit = False
            raise RuntimeError("commit failure")
        if parent_commit != self.head:
            raise RuntimeError("parent commit conflict")
        for operation in operations:
            if isinstance(operation, CommitOperationDelete):
                prefix = operation.path_in_repo.rstrip("/") + "/"
                self.tree = {p: f for p, f in self.tree.items() if not p.startswith(prefix)}
            else:
                assert isinstance(operation, CommitOperationAdd)
                payload = operation.path_or_fileobj
                if not isinstance(payload, bytes):
                    payload = Path(payload).read_bytes()
                path = operation.path_in_repo
                f = repofile(path, payload, lfs=not path.endswith((".json", ".md")))
                self.tree[path] = f
                if uploader.lfs_sha(f):
                    self.objects.add(uploader.lfs_sha(f))
        oid = self.snapshot()
        self.commits.append({"oid": oid, "repo_id": repo_id, "operations": operations, **kwargs})
        if self.after_commit:
            hook, self.after_commit = self.after_commit, None
            hook()
        if self.lose_response:
            self.lose_response = False
            raise RuntimeError("lost commit response")
        return SimpleNamespace(oid=oid, commit_url=f"https://example.invalid/commit/{oid}")

    def list_lfs_files(self, *args, **kwargs):
        pytest.fail("Online LFS enumeration is not a safe GC policy")

    def permanently_delete_lfs_files(self, *args, **kwargs):
        self.deleted.append(args)
        pytest.fail("Automatic irreversible deletion must never be called")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbid_network(*args, **kwargs):
        pytest.fail("Uploader tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_TOKEN", "offline-placeholder")


@pytest.fixture
def setup_upload(tmp_path):
    ckpt = tmp_path / "run" / "checkpoint-200"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"new")
    (ckpt / "config.json").write_text("{}")
    args = SimpleNamespace(
        path_prefix="exp",
        dest_subdir="resume",
        repo_id="fake/repo",
        status_file=tmp_path / "status.json",
    )
    return args, ckpt, MemoryHub()


def journal(args):
    return uploader.load_transactions(args)


def assert_candidates_retained(args, hub):
    oid = hashlib.sha256(b"old").hexdigest()
    assert any(oid in record["gc_candidates"] for _, record in journal(args))
    assert oid in hub.objects
    assert not hub.deleted
    return oid


@pytest.mark.parametrize("writer", ["eval", "branch", "external"])
def test_no_irreversible_gc_with_concurrent_writers_and_other_refs(setup_upload, writer):
    args, ckpt, hub = setup_upload

    def publish():
        if writer == "eval":
            f = repofile("exp/checkpoint-100/model.safetensors", b"old")
            hub.tree[f.path] = f
            hub.snapshot()
        elif writer == "branch":
            hub.revisions["protected-branch"] = {
                "retained/model.safetensors": repofile("retained/model.safetensors", b"old")
            }
        else:
            f = repofile("external/weights.bin", b"old")
            hub.tree[f.path] = f
            hub.snapshot()

    hub.after_commit = publish
    status = {"uploaded_step": 100}
    uploader.upload_checkpoint(hub, args, 200, ckpt, status)
    oid = assert_candidates_retained(args, hub)
    assert any(uploader.lfs_sha(f) == oid for tree in hub.revisions.values() for f in tree.values())
    assert not any(path.startswith("exp/resume/checkpoint-100/") for path in hub.tree)
    assert status["uploaded_step"] == 200
    assert status["gc_policy"] == "offline_only"
    assert status["pending_gc_candidates"] == [oid]
    assert status["freed_bytes"] == status["lfs_objects_deleted"] == 0
    assert "repo_lfs_total_bytes" not in status
    assert hub.listings[-1][1] == hub.commits[0]["oid"]
    assert "history retained" in hub.commits[0]["commit_message"]


@pytest.mark.parametrize("restart", [False, True])
def test_commit_success_then_list_failure_recovers_candidates(setup_upload, restart):
    args, ckpt, hub = setup_upload
    hub.fail_verification = True
    status = {}
    with pytest.raises(RuntimeError, match="transient list"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, status)
    assert journal(args)[0][1]["phase"] == "committed"
    assert_candidates_retained(args, hub)
    if restart:
        status = {}
        shutil.rmtree(ckpt)
    uploader.recover_pending(hub, args, status)
    assert journal(args)[0][1]["phase"] == "verified"
    assert status["uploaded_step"] == 200
    assert status["pending_step"] is None
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_failed_replace_commit_has_write_ahead_candidates_and_can_retry(setup_upload):
    args, ckpt, hub = setup_upload

    def check_journal():
        record = journal(args)[0][1]
        assert record["phase"] == "prepared"
        assert record["parent_commit"] == "initial"
        assert_candidates_retained(args, hub)

    hub.before_commit = check_journal
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    assert not hub.commits
    uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    assert len(hub.commits) == 1
    assert len(journal(args)) == 1
    assert_candidates_retained(args, hub)


def test_lost_commit_response_recovers_without_local_checkpoint(setup_upload):
    args, ckpt, hub = setup_upload
    hub.lose_response = True
    with pytest.raises(RuntimeError, match="lost commit response"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    assert journal(args)[0][1]["phase"] == "prepared"
    shutil.rmtree(ckpt)
    status = {}
    uploader.recover_pending(hub, args, status)
    assert status["uploaded_step"] == 200
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_failed_commit_phase_write_recovers_from_prepared_journal(setup_upload, monkeypatch):
    args, ckpt, hub = setup_upload
    real_write = uploader.write_transaction

    def fail_committed(path, record):
        if record["phase"] == "committed":
            raise OSError("journal disk failure")
        real_write(path, record)

    monkeypatch.setattr(uploader, "write_transaction", fail_committed)
    with pytest.raises(OSError, match="disk failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    assert journal(args)[0][1]["phase"] == "prepared"
    monkeypatch.setattr(uploader, "write_transaction", real_write)
    shutil.rmtree(ckpt)
    uploader.recover_pending(hub, args, {})
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_failure_before_durable_journal_cannot_commit(setup_upload, monkeypatch):
    args, ckpt, hub = setup_upload
    real_replace = uploader.os.replace

    def fail_journal_replace(src, dst):
        if Path(dst).parent == uploader.journal_dir(args):
            raise OSError("atomic replace failure")
        real_replace(src, dst)

    monkeypatch.setattr(uploader.os, "replace", fail_journal_replace)
    with pytest.raises(OSError, match="replace failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    assert not hub.commits
    assert not journal(args)


@pytest.mark.parametrize("corruption", ["truncated", "checksum", "shape"])
def test_torn_or_corrupted_journal_fails_closed(setup_upload, corruption):
    args, ckpt, hub = setup_upload
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    path, _ = journal(args)[0]
    if corruption == "truncated":
        path.write_text('{"transaction":')
    elif corruption == "checksum":
        data = json.loads(path.read_text())
        data["transaction"]["gc_candidates"] = {}
        path.write_text(json.dumps(data))
    else:
        path.write_text("[]")
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="restore it from backup"):
        uploader.recover_pending(hub, args, {})
    assert path.read_bytes() == original
    assert not hub.commits


def test_orphan_temporary_journal_does_not_hide_durable_candidates(setup_upload):
    args, ckpt, hub = setup_upload
    hub.fail_verification = True
    with pytest.raises(RuntimeError, match="transient list"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    (uploader.journal_dir(args) / ".pending-interrupted").write_text('{"sha256":')
    uploader.recover_pending(hub, args, {})
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_prepared_retry_rejects_changed_same_size_local_content(setup_upload):
    args, ckpt, hub = setup_upload
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    (ckpt / "model.safetensors").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        uploader.recover_pending(hub, args, {})
    assert not hub.commits
    assert_candidates_retained(args, hub)


@pytest.mark.parametrize("lfs", [True, False])
def test_verification_rejects_same_size_wrong_content(setup_upload, lfs):
    args, ckpt, hub = setup_upload

    def corrupt():
        path = "exp/resume/checkpoint-200/" + ("model.safetensors" if lfs else "config.json")
        payload = b"bad" if lfs else b"[]"
        hub.revisions[hub.head][path] = repofile(path, payload, lfs=lfs)

    hub.after_commit = corrupt
    status = {}
    with pytest.raises(RuntimeError, match="verification failed"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, status)
    assert status["uploaded_step"] == 0
    assert journal(args)[0][1]["phase"] == "committed"
    assert_candidates_retained(args, hub)


def test_compare_and_swap_conflict_keeps_candidates_and_retries(setup_upload):
    args, ckpt, hub = setup_upload

    def external_commit():
        f = repofile("exp/checkpoint-100/model.safetensors", b"old")
        hub.tree[f.path] = f
        hub.snapshot()

    hub.before_commit = external_commit
    with pytest.raises(RuntimeError, match="parent commit conflict"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    assert_candidates_retained(args, hub)
    uploader.recover_pending(hub, args, {})
    assert "exp/checkpoint-100/model.safetensors" in hub.tree
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_newer_remote_checkpoint_is_never_replaced_by_older_local_one(setup_upload):
    args, ckpt, hub = setup_upload
    f = repofile("exp/resume/checkpoint-300/model.safetensors", b"future")
    hub.tree[f.path] = f
    hub.snapshot()
    with pytest.raises(RuntimeError, match="newer resume checkpoint"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    assert not hub.commits
    assert f.path in hub.tree


def test_candidates_accumulate_across_replacements_and_stay_destination_scoped(setup_upload):
    args, ckpt, hub = setup_upload
    uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    ckpt300 = ckpt.with_name("checkpoint-300")
    shutil.copytree(ckpt, ckpt300)
    (ckpt300 / "model.safetensors").write_bytes(b"newer")
    status = {}
    uploader.upload_checkpoint(hub, args, 300, ckpt300, status)
    assert len(journal(args)) == 2
    assert set(status["pending_gc_candidates"]) == {
        hashlib.sha256(value).hexdigest() for value in (b"old", b"new")
    }
    args.repo_id = "other/repo"
    uploader.recover_pending(hub, args, status)
    assert status["uploaded_step"] == 0
    assert status["pending_gc_candidates"] == []
    assert "commit" not in status
    assert "uploaded_at" not in status
    assert len(journal(args)) == 2
    assert len(hub.commits) == 2


@pytest.mark.parametrize("status_state", ["stale", "missing", "torn"])
def test_main_restart_recovers_before_max_steps_exit(setup_upload, monkeypatch, status_state):
    args, ckpt, hub = setup_upload
    hub.fail_verification = True
    with pytest.raises(RuntimeError, match="transient list"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    if status_state == "missing":
        args.status_file.unlink()
    elif status_state == "torn":
        args.status_file.write_text("{")
    else:
        args.status_file.write_text(json.dumps({"uploaded_step": 200}))
    shutil.rmtree(ckpt)
    monkeypatch.setattr(uploader, "HfApi", lambda: hub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uploader",
            "--run-dir",
            str(ckpt.parent),
            "--repo-id",
            args.repo_id,
            "--path-prefix",
            args.path_prefix,
            "--status-file",
            str(args.status_file),
            "--max-steps",
            "200",
        ],
    )
    monkeypatch.setattr(uploader.time, "sleep", lambda _: pytest.fail("Unexpected retry or hang"))
    assert uploader.main() == 0
    status = json.loads(args.status_file.read_text())
    assert status["uploaded_step"] == 200
    assert status["pending_step"] is None
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


@pytest.mark.parametrize(
    "prefix,subdir,expected",
    [
        ("", "resume", "resume"),
        ("/exp//./", "/resume/", "exp/resume"),
        ("exp", "nested//resume", "exp/nested/resume"),
    ],
)
def test_destination_normalization(prefix, subdir, expected):
    assert uploader.destination(SimpleNamespace(path_prefix=prefix, dest_subdir=subdir)) == expected


@pytest.mark.parametrize("prefix,subdir", [("", ""), ("exp/..", "resume"), ("exp", "../resume")])
def test_unsafe_destination_rejected(prefix, subdir):
    with pytest.raises(ValueError):
        uploader.destination(SimpleNamespace(path_prefix=prefix, dest_subdir=subdir))


def test_verified_journal_recovers_after_status_write_failure(setup_upload, monkeypatch):
    args, ckpt, hub = setup_upload
    real_write = uploader.write_status

    def fail_status(path, status):
        if path == args.status_file and status.get("uploaded_step") == 200:
            raise OSError("status write failure")
        real_write(path, status)

    monkeypatch.setattr(uploader, "write_status", fail_status)
    with pytest.raises(OSError, match="status write failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    assert journal(args)[0][1]["phase"] == "verified"
    monkeypatch.setattr(uploader, "write_status", real_write)
    status = {}
    uploader.recover_pending(hub, args, status)
    assert status["uploaded_step"] == 200
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_parent_retry_merges_new_candidates_instead_of_losing_old_snapshot(setup_upload):
    args, ckpt, hub = setup_upload

    def replace_old_weights():
        path = "exp/resume/checkpoint-100/model.safetensors"
        f = repofile(path, b"externally-replaced")
        hub.tree[path] = f
        hub.objects.add(uploader.lfs_sha(f))
        hub.snapshot()

    hub.before_commit = replace_old_weights
    with pytest.raises(RuntimeError, match="parent commit conflict"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    uploader.recover_pending(hub, args, {})
    candidates = journal(args)[0][1]["gc_candidates"]
    assert set(candidates) == {
        hashlib.sha256(payload).hexdigest() for payload in (b"old", b"externally-replaced")
    }
    assert_candidates_retained(args, hub)


def test_unrelated_resume_files_are_not_gc_candidates_or_logically_deleted(setup_upload):
    args, ckpt, hub = setup_upload
    f = repofile("exp/resume/checkpoint-not-a-step/data.bin", b"unrelated")
    hub.tree[f.path] = f
    hub.snapshot()
    uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    assert f.path in hub.tree
    assert uploader.lfs_sha(f) not in journal(args)[0][1]["gc_candidates"]


def test_same_step_republication_removes_stale_files_and_retains_candidates(setup_upload):
    args, ckpt, hub = setup_upload
    stale = repofile("exp/resume/checkpoint-200/obsolete.bin", b"stale")
    hub.tree[stale.path] = stale
    hub.objects.add(uploader.lfs_sha(stale))
    hub.snapshot()
    status = {}
    uploader.upload_checkpoint(hub, args, 200, ckpt, status)
    assert stale.path not in hub.tree
    assert uploader.lfs_sha(stale) in status["pending_gc_candidates"]
    assert status["uploaded_step"] == 200
    assert_candidates_retained(args, hub)


def complete_successor(args, ckpt, step=300):
    successor = ckpt.with_name(f"checkpoint-{step}")
    successor.mkdir(exist_ok=True)
    (successor / "model.safetensors").write_bytes(f"weights-{step}".encode())
    (successor / "config.json").write_text("{}")
    (successor / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (successor / "latest").write_text(f"global_step{step}")
    ds = successor / f"global_step{step}"
    ds.mkdir(exist_ok=True)
    (ds / "mp_rank_00_model_states.pt").write_bytes(b"model-state")
    (ds / "rank_0_optim_states.pt").write_bytes(b"optim-state")
    (successor / "rng_state_0.pth").write_bytes(b"rng-state")
    (successor / "scheduler.pt").write_bytes(b"scheduler")
    args.num_gpus = 1
    args.quiet_seconds = 0
    assert uploader.full_checkpoint_complete(successor, step, 1, 0)
    return successor


@pytest.mark.parametrize("source", ["pruned", "partial", "changed"])
def test_outage_then_rotation_allows_newer_complete_upload(setup_upload, source):
    args, ckpt, hub = setup_upload
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    old_path, old_record = journal(args)[0]
    successor = complete_successor(args, ckpt)
    if source == "pruned":
        shutil.rmtree(ckpt)
    elif source == "partial":
        (ckpt / "model.safetensors").unlink()
    else:
        (ckpt / "model.safetensors").write_bytes(b"bad")
    status = {}
    uploader.upload_checkpoint(hub, args, 300, successor, status)
    uploader.upload_checkpoint(hub, args, 300, successor, {})
    records = {record["step"]: (path, record) for path, record in journal(args)}
    assert len(records) == 2
    assert records[200][0] == old_path
    assert records[200][1]["phase"] == "superseded"
    assert records[200][1]["gc_candidates"] == old_record["gc_candidates"]
    assert records[200][1]["superseded_by"] == records[300][0].name
    assert records[200][1]["superseded_by_step"] == 300
    assert records[300][1]["phase"] == "verified"
    assert "verified_at" not in records[200][1]
    assert status["uploaded_step"] == 300
    assert status["superseded_steps"] == [200]
    assert status["pending_step"] is None
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


@pytest.mark.parametrize(
    "successor_kind", ["absent", "incomplete", "not-newer", "wrong-ranks", "not-quiet"]
)
def test_missing_source_without_complete_successor_remains_pending(setup_upload, successor_kind):
    args, ckpt, hub = setup_upload
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    old_path, _ = journal(args)[0]
    original = old_path.read_bytes()
    successor = None
    if successor_kind != "absent":
        step = 100 if successor_kind == "not-newer" else 300
        successor = (step, complete_successor(args, ckpt, step))
        if successor_kind == "incomplete":
            (successor[1] / "scheduler.pt").unlink()
        elif successor_kind == "wrong-ranks":
            args.num_gpus = 2
        elif successor_kind == "not-quiet":
            args.quiet_seconds = 3600
    shutil.rmtree(ckpt)
    with pytest.raises(uploader.PendingCheckpointUnavailable, match="newer complete checkpoint"):
        uploader.recover_pending(hub, args, {}, successor=successor)
    assert old_path.read_bytes() == original
    assert len(journal(args)) == 1
    assert not hub.commits
    assert_candidates_retained(args, hub)


@pytest.mark.parametrize("failure", ["commit", "list", "lost-response"])
def test_superseded_and_successor_records_survive_retry_and_restart(setup_upload, failure):
    args, ckpt, hub = setup_upload
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    successor = complete_successor(args, ckpt)
    args.run_dir = ckpt.parent
    shutil.rmtree(ckpt)
    if failure == "commit":
        hub.fail_commit = True
    elif failure == "list":
        hub.fail_verification = True
    else:
        hub.lose_response = True
    with pytest.raises(RuntimeError):
        uploader.upload_checkpoint(hub, args, 300, successor, {})
    records = {record["step"]: record for _, record in journal(args)}
    assert records[200]["phase"] == "superseded"
    assert records[300]["phase"] in ("prepared", "committed")
    status = json.loads(args.status_file.read_text())
    assert status["uploaded_step"] == 0
    assert status["superseded_steps"] == [200]
    assert status["pending_step"] == 300
    if failure != "commit":
        shutil.rmtree(successor)
    args.status_file.unlink()
    status = {}
    uploader.recover_pending(hub, args, status)
    assert status["uploaded_step"] == 300
    assert status["superseded_steps"] == [200]
    assert status["pending_step"] is None
    assert len(journal(args)) == 2
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_lost_old_commit_response_recovers_before_considering_supersession(setup_upload):
    args, ckpt, hub = setup_upload
    hub.lose_response = True
    with pytest.raises(RuntimeError, match="lost commit response"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    successor = complete_successor(args, ckpt)
    shutil.rmtree(ckpt)
    uploader.upload_checkpoint(hub, args, 300, successor, {})
    records = {record["step"]: record for _, record in journal(args)}
    assert records[200]["phase"] == "verified"
    assert records[300]["phase"] == "verified"
    assert "superseded_at" not in records[200]
    assert len(hub.commits) == 2
    assert_candidates_retained(args, hub)


def test_remote_listing_failure_never_supersedes_unknown_commit(setup_upload):
    args, ckpt, hub = setup_upload
    hub.lose_response = True
    with pytest.raises(RuntimeError, match="lost commit response"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    successor = complete_successor(args, ckpt)
    shutil.rmtree(ckpt)
    original = journal(args)[0][0].read_bytes()
    hub.fail_verification = True
    with pytest.raises(RuntimeError, match="transient list failure"):
        uploader.upload_checkpoint(hub, args, 300, successor, {})
    assert len(journal(args)) == 1
    assert journal(args)[0][0].read_bytes() == original
    assert len(hub.commits) == 1


@pytest.mark.parametrize("failure_phase", ["successor-prepared", "old-superseded"])
def test_supersession_journal_write_failure_is_recoverable(
    setup_upload, monkeypatch, failure_phase
):
    args, ckpt, hub = setup_upload
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    successor = complete_successor(args, ckpt)
    args.run_dir = ckpt.parent
    shutil.rmtree(ckpt)
    write = uploader.write_transaction

    def fail_write(path, record):
        if (failure_phase == "successor-prepared" and record["step"] == 300) or (
            failure_phase == "old-superseded" and record["phase"] == "superseded"
        ):
            raise OSError("journal write interrupted")
        write(path, record)

    monkeypatch.setattr(uploader, "write_transaction", fail_write)
    with pytest.raises(OSError, match="journal write interrupted"):
        uploader.upload_checkpoint(hub, args, 300, successor, {})
    records = {record["step"]: record for _, record in journal(args)}
    assert records[200]["phase"] == "prepared"
    assert len(records) == (1 if failure_phase == "successor-prepared" else 2)
    assert not hub.commits
    monkeypatch.setattr(uploader, "write_transaction", write)
    uploader.recover_pending(hub, args, {})
    assert len(journal(args)) == 2
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_main_discovers_complete_successor_before_pruned_pending_blocks_progress(
    setup_upload, monkeypatch
):
    args, ckpt, hub = setup_upload
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    complete_successor(args, ckpt)
    shutil.rmtree(ckpt)
    args.status_file.write_text("{")
    monkeypatch.setattr(uploader, "HfApi", lambda: hub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uploader",
            "--run-dir",
            str(ckpt.parent),
            "--repo-id",
            args.repo_id,
            "--path-prefix",
            args.path_prefix,
            "--status-file",
            str(args.status_file),
            "--num-gpus",
            "1",
            "--quiet-seconds",
            "0",
            "--max-steps",
            "300",
        ],
    )
    monkeypatch.setattr(uploader.time, "sleep", lambda _: pytest.fail("Unexpected retry or hang"))
    assert uploader.main() == 0
    status = json.loads(args.status_file.read_text())
    assert status["uploaded_step"] == 300
    assert status["superseded_steps"] == [200]
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_repeated_rotation_keeps_all_superseded_candidates(setup_upload):
    args, ckpt, hub = setup_upload
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    successor = complete_successor(args, ckpt)
    shutil.rmtree(ckpt)
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 300, successor, {})
    final = complete_successor(args, successor, 400)
    shutil.rmtree(successor)
    status = {}
    uploader.upload_checkpoint(hub, args, 400, final, status)
    assert sorted((record["step"], record["phase"]) for _, record in journal(args)) == [
        (200, "superseded"),
        (300, "superseded"),
        (400, "verified"),
    ]
    assert status["uploaded_step"] == 400
    assert status["superseded_steps"] == [200, 300]
    assert len(hub.commits) == 1
    assert_candidates_retained(args, hub)


def test_missing_successor_journal_fails_closed_without_losing_old_candidates(setup_upload):
    args, ckpt, hub = setup_upload
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 200, ckpt, {})
    successor = complete_successor(args, ckpt)
    shutil.rmtree(ckpt)
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failure"):
        uploader.upload_checkpoint(hub, args, 300, successor, {})
    records = {record["step"]: (path, record) for path, record in journal(args)}
    old_bytes = records[200][0].read_bytes()
    records[300][0].unlink()
    with pytest.raises(RuntimeError, match="successor journal.*restore it from backup"):
        uploader.recover_pending(hub, args, {})
    assert records[200][0].read_bytes() == old_bytes
    assert not hub.commits


def test_generated_card_describes_retained_history_and_offline_candidates():
    card = uploader.readme("exp", 200, "fake/repo", "exp/resume")
    assert "automatic irreversible deletion is disabled" in card
    assert "still count against storage" in card
    assert "not a deletion allowlist" in card
    assert "folders in `exp/`" in card
