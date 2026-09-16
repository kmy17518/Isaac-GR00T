# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline destination-scoped eval uploader regressions."""

import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest
from scripts.b1k import hf_checkpoint_uploader as uploader


class FakeApi:
    def __init__(self):
        self.uploads = []
        self.cards = []
        self.fail_upload = False

    def create_repo(self, *args, **kwargs):
        pass

    def upload_file(self, **kwargs):
        self.cards.append(kwargs)

    def list_repo_tree(self, *args, **kwargs):
        return []

    def upload_folder(self, **kwargs):
        if self.fail_upload:
            raise RuntimeError("offline upload failure")
        self.uploads.append(kwargs)
        return SimpleNamespace(commit_url="https://example.invalid/commit")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbid_network(*args, **kwargs):
        pytest.fail("Uploader tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_TOKEN", "offline-placeholder")


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    staging = tmp_path / "staging"
    checkpoint = staging / "checkpoint-10000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    api = FakeApi()
    monkeypatch.setattr(uploader, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uploader",
            "--run-dir",
            str(run),
            "--staging-dir",
            str(staging),
            "--repo-id",
            "owner/old",
            "--path-prefix",
            "old",
            "--max-steps",
            "10000",
            "--once",
        ],
    )

    def execute(repo="owner/new", prefix="new", enabled=True):
        (staging / "upload_config.json").write_text(
            json.dumps({"repo_id": repo, "path_prefix": prefix, "enabled": enabled})
        )
        assert uploader.main() == 0
        return json.loads((staging / "status.json").read_text())

    return checkpoint, api, execute


@pytest.mark.parametrize(
    "repo,prefix", [("owner/new", "old"), ("owner/old", "new"), ("owner/new", "new")]
)
def test_runtime_destination_change_reuploads_staged_checkpoint(monitor, repo, prefix):
    checkpoint, api, execute = monitor
    (checkpoint / ".uploaded").write_text(
        json.dumps({"repo_id": "owner/old", "repo_path": "old/checkpoint-10000"})
    )
    status = execute(repo, prefix)
    assert len(api.uploads) == 1
    assert api.uploads[0]["repo_id"] == repo
    assert api.uploads[0]["path_in_repo"] == f"{prefix}/checkpoint-10000"
    assert api.uploads[0]["ignore_patterns"] == [".uploaded", ".uploaded.*"]
    assert status["uploaded"] == [checkpoint.name]
    assert status["remaining_scheduled_steps"] == []
    assert len(uploader.upload_records(checkpoint)) == 2


def test_switch_back_reuses_each_destination_record(monitor):
    checkpoint, api, execute = monitor
    execute("owner/old", "old")
    execute("owner/new", "new")
    execute("owner/old", "/old//./")
    execute("owner/new", "new")
    assert len(api.uploads) == 2
    assert len(uploader.upload_records(checkpoint)) == 2


@pytest.mark.parametrize(
    "marker",
    [
        "{",
        "null",
        "[]",
        '"string"',
        "{}",
        '{"repo_id":"owner/new"}',
        '{"repo_id":"owner/new","repo_path":42}',
        '{"repo_id":"owner/new","repo_path":"new/checkpoint-20000"}',
        '{"version":2,"destinations":null}',
        '{"version":2,"destinations":[null]}',
        '{"version":3,"repo_id":"owner/new","repo_path":"new/checkpoint-10000"}',
    ],
)
def test_corrupt_or_mismatched_marker_does_not_satisfy_schedule(monitor, marker):
    checkpoint, api, execute = monitor
    (checkpoint / ".uploaded").write_text(marker)
    status = execute(enabled=False)
    assert status["remaining_scheduled_steps"] == [10000]
    assert status["pending_upload"] == [checkpoint.name]
    assert status["uploaded"] == []
    assert not api.uploads
    status = execute()
    assert len(api.uploads) == 1
    assert status["remaining_scheduled_steps"] == []
    assert uploader.uploaded_record(checkpoint, "owner/new", "new") is not None


def test_unreadable_marker_is_pending(monitor, monkeypatch):
    checkpoint, _, execute = monitor
    marker = checkpoint / ".uploaded"
    marker.write_text("{}")
    real_read = Path.read_text

    def read(path, *args, **kwargs):
        if path == marker:
            raise PermissionError("unreadable marker")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert execute(enabled=False)["remaining_scheduled_steps"] == [10000]


@pytest.mark.parametrize(
    "prefix,stored",
    [
        ("/new//./", "new/checkpoint-10000"),
        ("new", "/new//./checkpoint-10000/"),
        ("", "checkpoint-10000"),
        ("/./", "/checkpoint-10000"),
    ],
)
def test_normalized_prefix_matches_legacy_marker(monitor, prefix, stored):
    checkpoint, api, execute = monitor
    (checkpoint / ".uploaded").write_text(json.dumps({"repo_id": "owner/new", "repo_path": stored}))
    status = execute(prefix=prefix)
    assert not api.uploads
    assert status["remaining_scheduled_steps"] == []
    assert status["path_prefix"] == uploader.normalize_repo_path(prefix)


def test_destination_upload_failure_does_not_mark_new_destination_complete(monitor):
    checkpoint, api, execute = monitor
    execute("owner/old", "old")
    api.fail_upload = True
    status = execute()
    assert status["remaining_scheduled_steps"] == [10000]
    assert "offline upload failure" in status["last_error"]
    assert uploader.uploaded_record(checkpoint, "owner/old", "old") is not None
    assert uploader.uploaded_record(checkpoint, "owner/new", "new") is None


def test_interrupted_marker_replace_preserves_previous_destination(monitor, monkeypatch):
    checkpoint, api, execute = monitor
    execute("owner/old", "old")
    real_replace = uploader.os.replace

    def fail_marker(src, dst):
        if Path(dst) == checkpoint / ".uploaded":
            raise OSError("marker replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(uploader.os, "replace", fail_marker)
    status = execute()
    assert status["remaining_scheduled_steps"] == [10000]
    assert uploader.uploaded_record(checkpoint, "owner/old", "old") is not None
    assert uploader.uploaded_record(checkpoint, "owner/new", "new") is None
    monkeypatch.setattr(uploader.os, "replace", real_replace)
    assert execute()["remaining_scheduled_steps"] == []
    assert len(api.uploads) == 3


@pytest.mark.parametrize("prefix", ["../exp", "exp/../other", "exp\\other", None])
def test_unsafe_prefix_is_rejected(prefix):
    with pytest.raises(ValueError):
        uploader.normalize_repo_path(prefix)
