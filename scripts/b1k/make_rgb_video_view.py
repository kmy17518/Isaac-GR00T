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

"""Build a training "view" of a LeRobot v3.0 dataset whose RGB videos are pre-resized, losslessly.

Why: decoding the BEHAVIOR demos (720x720 / 480x480 HEVC) is more than half of a dataloader
worker's CPU, and the first thing the GR00T image pipeline does with every frame is deterministic:
``LetterBoxPad`` (no-op on square frames) then ``SmallestMaxSize(shortest_image_edge,
INTER_AREA)`` -- the random crop / second resize / colour jitter come after. Applying exactly that
head once, offline, and storing the result **losslessly in RGB** (``libx264rgb -qp 0``, 4:4:4
predictive) gives the random stages bit-identical inputs while the decoder handles ~8x fewer pixels
(measured: 214 -> 1590 frames/s on the 720p camera). Nothing else changes: ``data/`` and ``meta/``
are symlinked (episode/frame indexing, stats caches and the ``*.mp4`` path template are unchanged),
depth streams and unselected chunks/files are symlinked too.

Every transcoded file is verified bitwise on sampled frames (decoded view frame == pipeline head
applied to the decoded source frame) and recorded in ``<view>/RGB_VIEW_MANIFEST.json``.

Usage:
    python scripts/b1k/make_rgb_video_view.py \\
        --source-root /data/2026-challenge-demos --view-root /data/2026-challenge-demos-rgb256 \\
        --task-names turning_on_radio --modality-json examples/b1k/r1pro.json --jobs 4
    # then train with --dataset-path /data/2026-challenge-demos-rgb256

``--shortest-edge`` must equal the model's ``shortest_image_edge`` / ``image_target_size`` (256 for
GR00T N1.7); a view is only exact for that resolution. The eval/serving path is unaffected (live
frames go through the full pipeline).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
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
    verify_samples: int = 24,
    seed: int = 0,
) -> dict:
    """Decode ``src`` with torchcodec (the training decoder), apply the pipeline head, encode
    losslessly to ``dst``; then verify sampled frames bitwise. Returns a manifest entry."""
    from torchcodec.decoders import VideoDecoder

    head = pipeline_head(shortest_edge)
    dec = VideoDecoder(src, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=decode_threads)
    n = dec.metadata.num_frames
    if not n:
        raise RuntimeError(f"{src}: unknown frame count")
    first = head(image=dec.get_frames_at(indices=[0]).data[0].numpy())["image"]
    h, w = first.shape[:2]
    tmp = dst + ".tmp.mp4"
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
        Path(tmp).unlink(missing_ok=True)
        raise
    t_enc = time.time() - t0

    # ---- verification: frame count + bitwise equality on sampled frames -----------------------
    new = VideoDecoder(tmp, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=decode_threads)
    if new.metadata.num_frames != n:
        Path(tmp).unlink(missing_ok=True)
        raise RuntimeError(f"{dst}: frame count {new.metadata.num_frames} != source {n}")
    rng = np.random.default_rng(seed)
    sample = sorted({0, n - 1, *rng.integers(0, n, size=verify_samples).tolist()})
    got = new.get_frames_at(indices=sample).data.numpy()
    want = np.stack(
        [head(image=f)["image"] for f in dec.get_frames_at(indices=sample).data.numpy()]
    )
    if got.shape != want.shape or not np.array_equal(got, want):
        Path(tmp).unlink(missing_ok=True)
        raise RuntimeError(f"{dst}: decoded frames differ from the source pipeline head")
    os.replace(tmp, dst)
    return {
        "source": src,
        "frames": n,
        "size": [h, w],
        "bytes": os.path.getsize(dst),
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


def link(target: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        return
    path.symlink_to(target.resolve())


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
    p.add_argument("--verify-samples", type=int, default=24)
    args = p.parse_args()

    source, view = Path(args.source_root), Path(args.view_root)
    info = json.loads((source / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    keys = args.video_keys or rgb_video_keys(info, args.modality_json)
    files, n_episodes = selected_video_files(source, info, keys, args.task_names)
    print(
        f"{n_episodes} episodes, {len(keys)} video keys {keys}, {len(files)} files to transcode -> {view}"
    )

    view.mkdir(parents=True, exist_ok=True)
    # Everything except videos/ is shared with the source (data, meta, README, ...).
    for entry in source.iterdir():
        if entry.name != "videos":
            link(entry, view / entry.name)
    # videos/: symlink whole key dirs we do not touch, whole chunk dirs we do not touch, and the
    # untouched files inside touched chunks.
    selected = {(k, c) for k, c, _ in files}
    for key_dir in sorted((source / "videos").iterdir()):
        key = key_dir.name
        if key not in keys or not any(k == key for k, _ in selected):
            link(key_dir, view / "videos" / key)
            continue
        for chunk_dir in sorted(key_dir.iterdir()):
            chunk = (
                int(chunk_dir.name.split("-")[-1]) if chunk_dir.name.startswith("chunk-") else None
            )
            if chunk is None or (key, chunk) not in selected:
                link(chunk_dir, view / "videos" / key / chunk_dir.name)
                continue
            wanted = {
                video_rel_path(info, key, chunk, f) for k, c, f in files if k == key and c == chunk
            }
            for f in sorted(chunk_dir.iterdir()):
                rel = f"videos/{key}/{chunk_dir.name}/{f.name}"
                if rel not in wanted:
                    link(f, view / rel)

    manifest_path = view / "RGB_VIEW_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        {
            "source_root": str(source.resolve()),
            "shortest_edge": args.shortest_edge,
            "codec": "libx264rgb -qp 0 (lossless RGB 4:4:4)",
            "pipeline_head": "LetterBoxPad + SmallestMaxSize(shortest_edge, INTER_AREA)",
            "versions": {
                "albumentations": A.__version__,
                "opencv": cv2.__version__,
                "torchcodec": __import__("torchcodec").__version__,
            },
            "video_keys": keys,
            "task_names": args.task_names,
        }
    )
    manifest.setdefault("files", {})

    todo = []
    for key, chunk, file in files:
        rel = video_rel_path(info, key, chunk, file)
        dst = view / rel
        if (
            rel in manifest["files"]
            and dst.exists()
            and dst.stat().st_size == manifest["files"][rel]["bytes"]
        ):
            continue
        todo.append((rel, str(source / rel), str(dst)))
    print(f"{len(files) - len(todo)} already done, {len(todo)} to do")

    failed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {}
        for rel, src, dst in todo:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
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
                    verify_samples=args.verify_samples,
                )
            ] = rel
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                entry = fut.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAILED {rel}: {e}", file=sys.stderr)
                continue
            manifest["files"][rel] = entry
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
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
