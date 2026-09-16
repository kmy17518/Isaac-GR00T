# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RGB views must never mutate source videos, including when a subset grows."""

from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scripts.b1k import make_rgb_video_view as view


class InlineExecutor:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


KEYS = ["observation.rgb.front", "observation.rgb.wrist"]


@pytest.fixture
def dataset(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "view"
    (source / "meta").mkdir(parents=True)
    info = {
        "fps": 30,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {key: {"dtype": "video"} for key in KEYS},
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    for key in KEYS:
        for chunk in range(3):
            for file in range(2):
                path = source / view.video_rel_path(info, key, chunk, file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"source-{key}-{chunk}-{file}".encode())
    return source, dest, info


def hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def run(monkeypatch):
    calls = []

    def fake_transcode(src, dst, shortest_edge, fps, **kwargs):
        calls.append((src, dst, shortest_edge, fps, kwargs))
        tmp = Path(dst + ".tmp.mp4")
        payload = f"resized-{shortest_edge}".encode()
        tmp.write_bytes(payload)
        tmp.replace(dst)
        return {
            "source": src,
            "frames": 1,
            "size": [shortest_edge, shortest_edge],
            "bytes": len(payload),
            "seconds": 0,
            "verified_frames": [0],
        }

    monkeypatch.setattr(view, "transcode_file", fake_transcode)
    monkeypatch.setattr(view, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(
        view,
        "_configuration",
        lambda args, source, fps: {
            "source": str(source),
            "shortest_edge": args.shortest_edge,
            "fps": fps,
            "gop": args.gop,
            "cabac": args.cabac,
            "preset": args.preset,
            "version": 1,
        },
    )

    def invoke(dataset, selected=None, extra=()):
        source, dest, _ = dataset
        selected = [(KEYS[0], 0, 0)] if selected is None else selected
        monkeypatch.setattr(view, "selected_video_files", lambda *args: (selected, len(selected)))
        monkeypatch.setattr(
            sys, "argv", ["view", "--source-root", str(source), "--view-root", str(dest), *extra]
        )
        start = len(calls)
        assert view.main() == 0
        return calls[start:]

    return invoke


def test_expansion_materializes_camera_chunk_and_file_symlinks(dataset, run):
    source, dest, info = dataset
    original = hashes(source)
    run(dataset)
    assert (dest / "videos" / KEYS[1]).is_symlink()
    assert (dest / "videos" / KEYS[0] / "chunk-001").is_symlink()
    assert (dest / view.video_rel_path(info, KEYS[0], 0, 1)).is_symlink()
    selected = [(KEYS[0], 0, 0), (KEYS[0], 0, 1), (KEYS[0], 1, 0), (KEYS[1], 0, 0)]
    assert len(run(dataset, selected)) == 3
    for item in selected:
        dst = dest / view.video_rel_path(info, *item)
        assert not dst.is_symlink() and not dst.parent.is_symlink()
    assert (dest / "videos" / KEYS[1] / "chunk-001").is_symlink()
    assert (dest / view.video_rel_path(info, KEYS[0], 1, 1)).is_symlink()
    assert hashes(source) == original
    assert run(dataset, selected) == []


def test_video_root_symlink_is_materialized_on_expansion(dataset, run):
    source, dest, _ = dataset
    run(dataset, [])
    assert (dest / "videos").is_symlink()
    original = hashes(source)
    run(dataset)
    assert not (dest / "videos").is_symlink()
    assert (dest / "videos" / KEYS[1]).is_symlink()
    assert hashes(source) == original


@pytest.mark.parametrize(
    "extra", [["--shortest-edge", "128"], ["--gop", "1"], ["--cabac"], ["--preset", "fast"]]
)
def test_incompatible_configuration_is_rejected_without_changes(dataset, run, extra):
    run(dataset)
    before = hashes(dataset[1])
    with pytest.raises(ValueError, match="new --view-root"):
        run(dataset, extra=extra)
    assert hashes(dataset[1]) == before


@pytest.mark.parametrize("change", ["source", "fps", "version", "unselected_source"])
def test_incompatible_source_or_version_rejects_even_no_jobs(
    dataset, run, monkeypatch, tmp_path, change
):
    source, dest, info = dataset
    run(dataset)
    if change == "source":
        other = tmp_path / "other"
        shutil.copytree(source, other)
        dataset = other, dest, info
    elif change == "fps":
        (source / "meta/info.json").write_text(json.dumps({**info, "fps": 20}))
    elif change == "version":
        configuration = view._configuration
        monkeypatch.setattr(
            view, "_configuration", lambda *args: {**configuration(*args), "version": 2}
        )
    else:
        path = source / view.video_rel_path(info, KEYS[0], 0, 0)
        stat = path.stat()
        path.write_bytes(b"x" * stat.st_size)
    before = (dest / view.MANIFEST_NAME).read_bytes()
    with pytest.raises(ValueError, match="new --view-root"):
        run(dataset, [])
    assert (dest / view.MANIFEST_NAME).read_bytes() == before


def test_cli_forwards_gop_and_cabac(dataset, run):
    call = run(dataset, extra=["--gop", "1", "--cabac"])[0]
    assert call[-1]["gop"] == 1
    assert call[-1]["cabac"] is True


@pytest.mark.parametrize("kind", ["equal", "child", "parent", "root_link", "ancestor_alias"])
def test_rejects_overlapping_or_symlink_roots(dataset, run, tmp_path, kind):
    source, dest, info = dataset
    if kind == "equal":
        dest = source
    elif kind == "child":
        dest = source / "view"
    elif kind == "parent":
        dest = source.parent
    elif kind == "root_link":
        dest.symlink_to(source)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(source)
        dest = alias / "view"
    original = hashes(source)
    with pytest.raises(ValueError, match="independent"):
        run((source, dest, info))
    assert hashes(source) == original


@pytest.mark.parametrize("rel", ["../outside.mp4", "/tmp/outside.mp4", "videos/../../outside.mp4"])
def test_video_template_cannot_escape_view(dataset, run, rel):
    source, dest, info = dataset
    info["video_path"] = rel
    (source / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="under videos"):
        run(dataset)
    assert not dest.exists()


def test_source_aliases_resolve_and_output_file_alias_is_rejected(dataset, run, tmp_path):
    source, dest, info = dataset
    alias = tmp_path / "source_alias"
    alias.symlink_to(source)
    run((alias, dest, info))
    assert run(dataset) == []
    dst = dest / view.video_rel_path(info, KEYS[0], 0, 0)
    dst.unlink()
    dst.hardlink_to(source / view.video_rel_path(info, KEYS[0], 0, 0))
    original = hashes(source)
    with pytest.raises(ValueError, match="aliases"):
        run(dataset)
    assert hashes(source) == original


@pytest.mark.parametrize("location", ["camera", "file", "temp", "manifest", "source"])
def test_rejects_unexpected_external_symlinks(dataset, run, tmp_path, location):
    source, dest, info = dataset
    run(dataset)
    external = tmp_path / "external"
    external.mkdir()
    victim = external / "victim.mp4"
    victim.write_bytes(b"untouched")
    selected = [(KEYS[0], 0, 0)]
    if location == "camera":
        path = dest / "videos" / KEYS[1]
        path.unlink()
        path.symlink_to(external)
        selected = [(KEYS[1], 0, 0)]
    elif location == "manifest":
        path = dest / view.MANIFEST_NAME
        path.unlink()
        path.symlink_to(victim)
    else:
        path = (source if location == "source" else dest) / view.video_rel_path(info, *selected[0])
        if location == "temp":
            path = Path(str(path) + ".tmp.mp4")
        else:
            path.unlink()
        path.symlink_to(victim)
    with pytest.raises(ValueError):
        run(dataset, selected)
    assert victim.read_bytes() == b"untouched"


def test_interrupted_output_rebuilds_and_atomic_manifest_persists_no_jobs(
    dataset, run, monkeypatch
):
    source, dest, info = dataset
    run(dataset)
    dst = dest / view.video_rel_path(info, KEYS[0], 0, 0)
    dst.write_bytes(b"x" * dst.stat().st_size)
    stale = Path(str(dst) + ".tmp.mp4")
    stale.write_bytes(b"interrupted")
    assert len(run(dataset)) == 1
    assert not stale.exists()
    assert run(dataset, [], ["--task-names", "nothing"]) == []
    manifest = json.loads((dest / view.MANIFEST_NAME).read_text())
    assert manifest["selected_files"] == [] and manifest["task_names"] == ["nothing"]
    before = (dest / view.MANIFEST_NAME).read_bytes()
    replace = view.os.replace

    def fail_manifest(src, dst):
        if Path(dst).name == view.MANIFEST_NAME:
            raise OSError("interrupted manifest update")
        return replace(src, dst)

    monkeypatch.setattr(view.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="interrupted manifest"):
        run(dataset)
    assert (dest / view.MANIFEST_NAME).read_bytes() == before
    assert not list(dest.glob(f".{view.MANIFEST_NAME}.*"))


def test_legacy_manifest_requires_fresh_view(dataset, run):
    _, dest, _ = dataset
    dest.mkdir()
    (dest / view.MANIFEST_NAME).write_text('{"files": {}}')
    with pytest.raises(ValueError, match="new --view-root"):
        run(dataset)


def test_internal_destination_alias_is_rejected(dataset, run):
    _, dest, _ = dataset
    run(dataset)
    alias = dest / "alias"
    alias.symlink_to(dest / "videos")
    with pytest.raises(ValueError, match="without symlinks"):
        view._check_output(alias / "new.mp4", dest, dataset[0])


def test_interrupted_encode_keeps_pending_fingerprint(dataset, run, monkeypatch):
    source, dest, info = dataset

    def fail_encode(*args, **kwargs):
        raise RuntimeError("encoder stopped")

    monkeypatch.setattr(view, "transcode_file", fail_encode)
    with pytest.raises(AssertionError):
        run(dataset)
    manifest = json.loads((dest / view.MANIFEST_NAME).read_text())
    assert manifest["files"] == {} and len(manifest["sources"]) == 1
    (source / view.video_rel_path(info, KEYS[0], 0, 0)).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Source changed"):
        run(dataset, [])


def test_worker_rejects_overlap_and_cleans_temporary_files(dataset, tmp_path, monkeypatch):
    source, dest, info = dataset
    dest.mkdir()
    src = source / view.video_rel_path(info, KEYS[0], 0, 0)
    dst = dest / "out.mp4"
    arguments = {"source_root": str(source), "view_root": str(dest)}
    with pytest.raises(ValueError, match="independent"):
        view.transcode_file(str(src), str(src), 16, 30, **arguments)
    outside = tmp_path / "outside.mp4"
    with pytest.raises(ValueError, match="independent"):
        view.transcode_file(str(src), str(outside), 16, 30, **arguments)
    Path(str(dst) + ".tmp.mp4").symlink_to(src)
    with pytest.raises(ValueError, match="independent"):
        view.transcode_file(str(src), str(dst), 16, 30, **arguments)
    Path(str(dst) + ".tmp.mp4").unlink()

    def fail_encode(src, tmp, *args):
        Path(tmp).write_bytes(b"partial output")
        raise RuntimeError("encoder stopped")

    monkeypatch.setattr(view, "_encode_video", fail_encode)
    with pytest.raises(RuntimeError, match="stopped"):
        view.transcode_file(str(src), str(dst), 16, 30, **arguments)
    assert not list(dest.iterdir())


@pytest.mark.parametrize("gop,cabac", [(1, True), (2, False)])
def test_real_codec_expansion_preserves_source_and_requested_coding(
    dataset, monkeypatch, gop, cabac
):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    pytest.importorskip("torchcodec")
    from torchcodec.decoders import VideoDecoder

    source, dest, info = dataset
    selected = [(KEYS[0], 0, 0), (KEYS[0], 1, 0)]
    for item in selected:
        path = source / view.video_rel_path(info, *item)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=32x32:rate=30",
                "-frames:v",
                "8",
                "-c:v",
                "libx264rgb",
                "-qp",
                "0",
                "-threads",
                "1",
                str(path),
            ],
            check=True,
        )
    original = hashes(source)
    monkeypatch.setattr(view, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "view",
            "--source-root",
            str(source),
            "--view-root",
            str(dest),
            "--shortest-edge",
            "16",
            "--jobs",
            "1",
            "--decode-threads",
            "1",
            "--encode-threads",
            "1",
            "--gop",
            str(gop),
            *(["--cabac"] if cabac else []),
        ],
    )
    for subset in [selected[:1], selected]:
        monkeypatch.setattr(view, "selected_video_files", lambda *args: (subset, len(subset)))
        assert view.main() == 0
    assert hashes(source) == original
    for item in selected:
        src, dst = (root / view.video_rel_path(info, *item) for root in (source, dest))
        frames = (
            VideoDecoder(str(src), dimension_order="NHWC")
            .get_frames_at(indices=list(range(8)))
            .data.numpy()
        )
        actual = (
            VideoDecoder(str(dst), dimension_order="NHWC")
            .get_frames_at(indices=list(range(8)))
            .data.numpy()
        )
        expected = np.stack([view.pipeline_head(16)(image=frame)["image"] for frame in frames])
        np.testing.assert_array_equal(actual, expected)
        headers = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-i",
                str(dst),
                "-c:v",
                "copy",
                "-bsf:v",
                "trace_headers",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stderr
        assert "entropy_coding_mode_flag" in headers
        assert all(
            line.endswith(f"= {int(cabac)}")
            for line in headers.splitlines()
            if "entropy_coding_mode_flag" in line
        )
        frame_types = [
            int(line.rsplit("=", 1)[1])
            for line in headers.splitlines()
            if "nal_unit_type" in line and line.endswith(("= 1", "= 5"))
        ]
        assert frame_types == [5 if index % gop == 0 else 1 for index in range(8)]
