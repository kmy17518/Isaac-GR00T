#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a training "view" of a LeRobot v3.0 dataset whose RGB videos are pre-resized and re-encoded, losslessly.

Why: decoding the BEHAVIOR demos is most of a dataloader worker's CPU, for two reasons. (1) The
sharded dataset samples *strided* steps (every ~10th frame of an episode per shard), and with
inter-frame coding (GOP 8 in the source) the decoder must reconstruct every frame in the span to
output the ones used -- ~10 frames decoded per frame used. (2) The 720x720 camera is decoded at full
resolution although the first thing the GR00T image pipeline does with every frame is deterministic:
``LetterBoxPad`` (no-op on square frames) then ``SmallestMaxSize(shortest_image_edge, INTER_AREA)``;
the random crop / second resize / colour jitter come after. This script applies exactly that head
once, offline, and stores the result **losslessly in RGB** (``libx264rgb -qp 0``, 4:4:4) with a
**short GOP** (``--gop 10``) and CAVLC entropy coding, so a strided read costs ~1 decoded frame per
used frame and each frame is cheap to decode. Measured on the stride-10 pattern, single-threaded:
source 10.4 + 2x5.6 ms CPU per sample (3 cameras) -> 1.8 + 2x1.6 ms; files ~40 GB per task.
Nothing else changes: ``data/`` and ``meta/`` are symlinked (episode/frame indexing, stats caches
and the ``*.mp4`` path template are unchanged), depth streams and unselected chunks/files are
symlinked too, and every downstream random stage sees bit-identical inputs.

Every transcoded file is verified bitwise on sampled frames (decoded view frame == pipeline head
applied to the decoded source frame) and recorded in ``<view>/RGB_VIEW_MANIFEST.json``.

Usage:
    python scripts/b1k/make_rgb_video_view.py \\
        --source-root /data/2026-challenge-demos --view-root /data/2026-challenge-demos-rgb256 \\
        --task-names turning_on_radio --modality-json examples/b1k/r1pro.json --jobs 4
    # then train with --dataset-path /data/2026-challenge-demos-rgb256

``--shortest-edge`` must equal the model's ``shortest_image_edge`` / ``image_target_size`` (256 for
GR00T N1.7); a view is only exact for that resolution. Long-GOP lossless files (x264 default GOP
250 + CABAC) decode *slower* than the source for strided reads -- keep ``--gop`` at or below the
sampling stride. The eval/serving path is unaffected (live frames go through the full pipeline).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import albumentations as A
import cv2
import numpy as np


def pipeline_head(shortest_edge: int) -> A.Compose:
    """The deterministic head of the GR00T training image pipeline (see
    ``build_image_transformations_albumentations``): pad to square, resize shortest edge."""
    from gr00t.model.gr00t_n1d7.image_augmentations import LetterBoxPad

    return A.Compose(
        [LetterBoxPad(), A.SmallestMaxSize(max_size=shortest_edge, interpolation=cv2.INTER_AREA)]
    )


def transcode_file(
    src: str,
    dst: str,
    shortest_edge: int,
    fps: float,
    block: int = 256,
    decode_threads: int = 4,
    encode_threads: int = 4,
    preset: str = "medium",
    gop: int = 10,
    cabac: bool = False,
    verify_samples: int = 24,
    seed: int = 0,
    *,
    source_root: str,
    view_root: str,
) -> dict:
    """Encode and verify inside an independent view, never through an output symlink."""
    source, view = _independent_roots(Path(source_root), Path(view_root))
    src_path, dst_path = Path(src).resolve(strict=True), Path(dst)
    if not src_path.is_relative_to(source) or src_path.is_relative_to(view):
        raise ValueError(f"Source video escapes source-root: {src}")
    _check_output(dst_path, view, source)
    if dst_path.exists() and dst_path.samefile(src_path):
        raise ValueError(f"Source and destination are aliases: {src}, {dst}")
    _check_output(Path(dst + ".tmp.mp4"), view, source)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst_path.stem}.", suffix=".tmp.mp4", dir=dst_path.parent)
    os.close(fd)
    try:
        entry = _encode_video(
            src,
            tmp,
            shortest_edge,
            fps,
            block,
            decode_threads,
            encode_threads,
            preset,
            gop,
            cabac,
            verify_samples,
            seed,
        )
        _check_output(dst_path, view, source)
        _check_output(Path(tmp), view, source)
        os.replace(tmp, dst)
        return entry
    finally:
        Path(tmp).unlink(missing_ok=True)


def _encode_video(
    src: str,
    tmp: str,
    shortest_edge: int,
    fps: float,
    block: int,
    decode_threads: int,
    encode_threads: int,
    preset: str,
    gop: int,
    cabac: bool,
    verify_samples: int,
    seed: int,
) -> dict:
    from torchcodec.decoders import VideoDecoder

    head = pipeline_head(shortest_edge)
    dec = VideoDecoder(src, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=decode_threads)
    n = dec.metadata.num_frames
    if not n:
        raise RuntimeError(f"{src}: unknown frame count")
    first = head(image=dec.get_frames_at(indices=[0]).data[0].numpy())["image"]
    h, w = first.shape[:2]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-c:v",
        "libx264rgb",
        "-qp",
        "0",
        "-preset",
        preset,
        "-pix_fmt",
        "rgb24",
        # fixed short GOP (no scene-cut keyframes) so strided reads decode ~1 frame per used frame;
        # CAVLC decodes ~2.5x cheaper than CABAC on lossless frames for ~10 % more bytes
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-x264-params",
        f"cabac={int(cabac)}",
        "-threads",
        str(encode_threads),
        "-movflags",
        "+faststart",
        tmp,
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for start in range(0, n, block):
            idx = list(range(start, min(start + block, n)))
            frames = dec.get_frames_at(indices=idx).data.numpy()
            out = np.stack([head(image=f)["image"] for f in frames])
            proc.stdin.write(out.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {src}")
    except BaseException:
        proc.kill()
        proc.wait()
        Path(tmp).unlink(missing_ok=True)
        raise
    t_enc = time.time() - t0

    # ---- verification: frame count + bitwise equality on sampled frames -----------------------
    new = VideoDecoder(tmp, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=decode_threads)
    if new.metadata.num_frames != n:
        Path(tmp).unlink(missing_ok=True)
        raise RuntimeError(f"{tmp}: frame count {new.metadata.num_frames} != source {n}")
    rng = np.random.default_rng(seed)
    sample = sorted({0, n - 1, *rng.integers(0, n, size=verify_samples).tolist()})
    got = new.get_frames_at(indices=sample).data.numpy()
    want = np.stack(
        [head(image=f)["image"] for f in dec.get_frames_at(indices=sample).data.numpy()]
    )
    if got.shape != want.shape or not np.array_equal(got, want):
        Path(tmp).unlink(missing_ok=True)
        raise RuntimeError(f"{tmp}: decoded frames differ from the source pipeline head")
    return {
        "source": src,
        "frames": n,
        "size": [h, w],
        "bytes": os.path.getsize(tmp),
        "seconds": round(t_enc, 1),
        "verified_frames": sample,
    }


def selected_video_files(
    source: Path, info: dict, video_keys: list[str], task_names: list[str] | None
):
    """(key, chunk, file) referenced by the selected episodes (all episodes when task_names is None)."""
    from gr00t.data.dataset.lerobot_episode_loader import (
        load_lerobot_episode_records,
        select_task_subset,
    )

    if task_names:
        records = select_task_subset(str(source), task_names).episode_records
    else:
        records = load_lerobot_episode_records(str(source), info)
    files = set()
    for rec in records:
        for key in video_keys:
            files.add(
                (key, int(rec[f"videos/{key}/chunk_index"]), int(rec[f"videos/{key}/file_index"]))
            )
    return sorted(files), len(records)


def video_rel_path(info: dict, key: str, chunk: int, file: int) -> str:
    return info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)


def rgb_video_keys(info: dict, modality_json: str | None) -> list[str]:
    if modality_json:
        with open(modality_json) as f:
            return sorted(v["original_key"] for v in json.load(f)["video"].values())
    return sorted(
        k for k, v in info["features"].items() if v.get("dtype") == "video" and ".rgb." in k
    )


MANIFEST_NAME = "RGB_VIEW_MANIFEST.json"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _independent_roots(source: Path, view: Path) -> tuple[Path, Path]:
    if view.is_symlink():
        raise ValueError("view-root must not be a symlink; use an independent directory")
    source, view = source.resolve(strict=True), view.resolve()
    if source.is_relative_to(view) or view.is_relative_to(source):
        raise ValueError("source-root and view-root must be independent, non-nested directories")
    return source, view


def _relative_video_path(rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "videos":
        raise ValueError(f"Video path must stay under videos/: {rel}")
    return path


def _source_fingerprint(source: Path, view: Path, rel: str) -> dict:
    path = (source / _relative_video_path(rel)).resolve(strict=True)
    if not path.is_relative_to(source) or path.is_relative_to(view) or not path.is_file():
        raise ValueError(f"Source video escapes source-root or overlaps view-root: {path}")
    return {"path": str(path), "sha256": _digest(path)}


def _check_output(path: Path, view: Path, source: Path) -> None:
    resolved = path.resolve()
    if (
        not path.is_relative_to(view)
        or not resolved.is_relative_to(view)
        or resolved.is_relative_to(source)
        or resolved == view
        or any(
            parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(view)
        )
    ):
        raise ValueError(f"Output must stay inside independent view-root without symlinks: {path}")


def _mirror_selected(source: Path, view: Path, selected: set[Path]) -> None:
    """Materialize only selected ancestors; leave every untouched sibling linked."""
    ancestors = {parent for rel in selected for parent in rel.parents}

    def visit(src: Path, dst: Path, rel: Path) -> None:
        if rel in ancestors:
            if dst.is_symlink():
                if dst.resolve() != src.resolve():
                    raise ValueError(f"Unexpected directory symlink in view: {dst}")
                dst.unlink()
            _check_output(dst, view, source)
            dst.mkdir(exist_ok=True)
            for child in sorted(src.iterdir()):
                visit(child, dst / child.name, rel / child.name)
        elif rel in selected:
            if dst.is_symlink():
                if dst.resolve() != src.resolve():
                    raise ValueError(f"Unexpected video symlink in view: {dst}")
                dst.unlink()
            _check_output(dst, view, source)
            if dst.exists() and dst.samefile(src):
                raise ValueError(f"Source and destination are aliases: {src}, {dst}")
        elif not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src.resolve())

    for entry in sorted(source.iterdir()):
        if entry.name != MANIFEST_NAME:
            visit(entry, view / entry.name, Path(entry.name))


def _write_manifest(path: Path, manifest: dict, view: Path, source: Path) -> None:
    _check_output(path, view, source)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=view)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _configuration(args: argparse.Namespace, source: Path, fps: float) -> dict:
    from gr00t.model.gr00t_n1d7 import image_augmentations
    import torch
    import torchcodec

    return {
        "format_version": 1,
        "source_root": str(source),
        "shortest_edge": args.shortest_edge,
        "fps": fps,
        "gop": args.gop,
        "cabac": args.cabac,
        "preset": args.preset,
        "decode_threads": args.decode_threads,
        "encode_threads": args.encode_threads,
        "verify_samples": args.verify_samples,
        "codec": "libx264rgb -qp 0 (lossless RGB 4:4:4)",
        "pipeline_head": "LetterBoxPad + SmallestMaxSize(shortest_edge, INTER_AREA)",
        "versions": {
            "albumentations": A.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchcodec": torchcodec.__version__,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True),
            "script_sha256": _digest(Path(__file__)),
            "pipeline_sha256": _digest(Path(image_augmentations.__file__)),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source-root", required=True)
    p.add_argument("--view-root", required=True)
    p.add_argument(
        "--task-names",
        nargs="*",
        default=None,
        help="restrict to these tasks' episodes (default: all)",
    )
    p.add_argument(
        "--modality-json",
        default=None,
        help="e.g. examples/b1k/r1pro.json: transcode only its video keys",
    )
    p.add_argument(
        "--video-keys",
        nargs="*",
        default=None,
        help="explicit video keys (overrides --modality-json)",
    )
    p.add_argument(
        "--shortest-edge", type=int, default=256, help="must match the model's shortest_image_edge"
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--decode-threads", type=int, default=4)
    p.add_argument("--encode-threads", type=int, default=4)
    p.add_argument("--preset", default="medium")
    p.add_argument(
        "--gop",
        type=int,
        default=10,
        help="keyframe interval; keep <= the sampling stride (every ~10th step)",
    )
    p.add_argument(
        "--cabac",
        action="store_true",
        help="CABAC entropy coding (smaller files, ~2.5x slower decode)",
    )
    p.add_argument("--verify-samples", type=int, default=24)
    args = p.parse_args()

    source, view = _independent_roots(Path(args.source_root), Path(args.view_root))
    info = json.loads((source / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    keys = args.video_keys or rgb_video_keys(info, args.modality_json)
    files, n_episodes = selected_video_files(source, info, keys, args.task_names)
    selected = {_relative_video_path(video_rel_path(info, *file)) for file in files}
    print(
        f"{n_episodes} episodes, {len(keys)} video keys {keys}, {len(files)} files to transcode -> {view}"
    )

    config = _configuration(args, source, fps)
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = view / MANIFEST_NAME
    _check_output(manifest_path, view, source)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and manifest.get("configuration") != config:
        raise ValueError(
            "Incompatible RGB view source/configuration/version; use a new --view-root"
        )
    if not manifest and view.exists() and any(view.iterdir()):
        raise ValueError("Nonempty view has no compatible manifest; use a new --view-root")
    manifest.setdefault("files", {})
    manifest.setdefault("sources", {})
    fingerprints = {
        rel: _source_fingerprint(source, view, rel)
        for rel in sorted(set(manifest["sources"]) | {str(rel) for rel in selected})
    }
    # Pending files also retain provenance across interrupted runs.
    for rel, fingerprint in manifest["sources"].items():
        if fingerprint != fingerprints[rel]:
            raise ValueError(f"Source changed for {rel}; use a new --view-root")
    for rel, entry in manifest["files"].items():
        if (
            entry.get("source_fingerprint") != fingerprints[rel]
            or entry.get("configuration_sha256") != config_digest
        ):
            raise ValueError(f"Source/configuration changed for {rel}; use a new --view-root")
    manifest.update(
        configuration=config,
        source_root=str(source),
        shortest_edge=args.shortest_edge,
        video_keys=keys,
        task_names=args.task_names,
        selected_files=sorted(str(rel) for rel in selected),
        sources=fingerprints,
    )

    view.mkdir(parents=True, exist_ok=True)
    _mirror_selected(source, view, {Path(rel) for rel in fingerprints})
    todo = []
    for rel in sorted(fingerprints):
        dst = view / rel
        _check_output(dst, view, source)
        stale_tmp = Path(str(dst) + ".tmp.mp4")
        _check_output(stale_tmp, view, source)
        stale_tmp.unlink(missing_ok=True)
        entry = manifest["files"].get(rel, {})
        if dst.is_file() and entry.get("output_sha256") == _digest(dst):
            continue
        manifest["files"].pop(rel, None)
        todo.append((rel, str(source / rel), str(dst)))
    print(f"{len(fingerprints) - len(todo)} already done, {len(todo)} to do")
    _write_manifest(manifest_path, manifest, view, source)

    failed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {}
        for rel, src, dst in todo:
            futures[
                ex.submit(
                    transcode_file,
                    src,
                    dst,
                    args.shortest_edge,
                    fps,
                    decode_threads=args.decode_threads,
                    encode_threads=args.encode_threads,
                    preset=args.preset,
                    gop=args.gop,
                    cabac=args.cabac,
                    verify_samples=args.verify_samples,
                    source_root=str(source),
                    view_root=str(view),
                )
            ] = rel
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                entry = fut.result()
                if _source_fingerprint(source, view, rel) != fingerprints[rel]:
                    raise RuntimeError(f"Source changed during encoding: {rel}")
                entry.update(
                    source_fingerprint=fingerprints[rel],
                    configuration_sha256=config_digest,
                    output_sha256=_digest(view / rel),
                )
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAILED {rel}: {e}", file=sys.stderr)
                continue
            manifest["files"][rel] = entry
            _write_manifest(manifest_path, manifest, view, source)
            print(
                f"ok {rel}: {entry['frames']} frames, {entry['bytes'] / 1e6:.0f} MB, {entry['seconds']}s, "
                f"verified {len(entry['verified_frames'])} frames bitwise"
            )
    total = sum(e["bytes"] for e in manifest["files"].values())
    print(
        f"done: {len(manifest['files'])} files, {total / 1e9:.1f} GB, {failed} failed -> {manifest_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
