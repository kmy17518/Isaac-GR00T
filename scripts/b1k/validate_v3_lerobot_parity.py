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

"""Cross-check GR00T's native v3.0 reader against the **real lerobot** reader.

This is the optional, high-value parity check described in the Experiment 2 plan:
it loads the v3.0 dataset with the actual ``lerobot`` library and asserts the
native ``LeRobotEpisodeLoader`` returns identical frames/states. Because GR00T
and lerobot have conflicting pins (datasets / av / wandb) they cannot share a
venv, so this runs as a two-interpreter handshake:

  - ``--mode native`` runs in the **GR00T venv** and dumps the native reader's
    state/action groups and decoded video frames for sampled (episode, frame)s.
  - ``--mode compare`` (default) runs in the **conversion venv** (which has
    ``lerobot``), invokes the native dump via ``--groot-python``, then loads the
    same samples through ``lerobot.LeRobotDataset`` and compares.

Usage (from the conversion venv, which has lerobot installed):
    cd scripts/lerobot_conversion && source .venv/bin/activate
    python ../b1k/validate_v3_lerobot_parity.py \
        --v3-path .../b1k/turning_on_radio_v3.0 \
        --groot-python ../../.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODALITY_CONFIG = REPO_ROOT / "examples" / "b1k" / "r1pro.py"


def _sample_frames(length: int, k: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, num=min(k, length)).astype(int))


# --------------------------------------------------------------------------- #
# Native mode: runs in the GR00T venv, writes an npz of native-reader outputs.
# --------------------------------------------------------------------------- #
def run_native(args: argparse.Namespace) -> int:
    import importlib

    sys.path.insert(0, str(REPO_ROOT))
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    cfg_path = Path(args.modality_config_path)
    sys.path.append(str(cfg_path.parent))
    importlib.import_module(cfg_path.stem)
    modality_configs = MODALITY_CONFIGS[args.embodiment_tag]

    loader = LeRobotEpisodeLoader(args.v3_path, modality_configs, video_backend=args.video_backend)
    state_groups = modality_configs["state"].modality_keys
    action_groups = modality_configs["action"].modality_keys
    video_keys = modality_configs["video"].modality_keys

    out: dict[str, np.ndarray] = {}
    for ep in args.episodes:
        length = loader.get_episode_length(ep)
        frames = _sample_frames(length, args.frames_per_episode)
        out[f"ep{ep}__frames"] = frames

        df = loader._load_parquet_data(ep)
        for g in state_groups:
            out[f"ep{ep}__state__{g}"] = np.vstack(
                [np.asarray(df[f"state.{g}"].iloc[f], dtype=np.float32) for f in frames]
            )
        for g in action_groups:
            out[f"ep{ep}__action__{g}"] = np.vstack(
                [np.asarray(df[f"action.{g}"].iloc[f], dtype=np.float32) for f in frames]
            )
        vid = loader._load_video_data(ep, frames)
        for cam in video_keys:
            out[f"ep{ep}__video__{cam}"] = vid[cam].astype(np.uint8)

    np.savez_compressed(args.out, **out)
    print(f"[native] wrote {args.out} for episodes {list(args.episodes)}")
    return 0


# --------------------------------------------------------------------------- #
# Compare mode: runs in the conversion venv (lerobot), diffs vs the native npz.
# --------------------------------------------------------------------------- #
def run_compare(args: argparse.Namespace) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    modality = json.loads((Path(args.v3_path) / "meta" / "modality.json").read_text())
    state_slices = {g: (s["start"], s["end"]) for g, s in modality["state"].items()}
    action_slices = {g: (s["start"], s["end"]) for g, s in modality["action"].items()}
    cam_to_original = {g: s["original_key"] for g, s in modality["video"].items()}

    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "native.npz"
        cmd = [
            args.groot_python,
            str(Path(__file__).resolve()),
            "--mode",
            "native",
            "--v3-path",
            str(args.v3_path),
            "--modality-config-path",
            str(args.modality_config_path),
            "--embodiment-tag",
            args.embodiment_tag,
            "--video-backend",
            args.video_backend,
            "--frames-per-episode",
            str(args.frames_per_episode),
            "--out",
            str(npz_path),
            "--episodes",
            *[str(e) for e in args.episodes],
        ]
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
        print(f"[compare] running native dump via {args.groot_python}")
        subprocess.run(cmd, check=True, env=env)
        native = dict(np.load(npz_path))

    print(f"[compare] loading v3.0 via lerobot: {args.v3_path}")
    ds = LeRobotDataset(repo_id="parity_check", root=str(args.v3_path))
    episodes_meta = ds.meta.episodes

    failures: list[str] = []
    checks = 0

    def record(ok: bool, label: str, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    for ep in args.episodes:
        frames = native[f"ep{ep}__frames"]
        from_index = int(episodes_meta[ep]["dataset_from_index"])
        lr = [ds[from_index + int(f)] for f in frames]
        lr_state = np.vstack([s["observation.state"].numpy() for s in lr])
        lr_action = np.vstack([s["action"].numpy() for s in lr])

        worst, where = 0.0, ""
        for g, (lo, hi) in state_slices.items():
            d = float(np.abs(native[f"ep{ep}__state__{g}"] - lr_state[:, lo:hi]).max())
            if d >= worst:
                worst, where = d, f"state.{g}"
        record(
            worst <= args.lowdim_tol,
            f"ep{ep}: state matches lerobot",
            f"max|Δ|={worst:.3e} @ {where}",
        )

        worst, where = 0.0, ""
        for g, (lo, hi) in action_slices.items():
            d = float(np.abs(native[f"ep{ep}__action__{g}"] - lr_action[:, lo:hi]).max())
            if d >= worst:
                worst, where = d, f"action.{g}"
        record(
            worst <= args.lowdim_tol,
            f"ep{ep}: action matches lerobot",
            f"max|Δ|={worst:.3e} @ {where}",
        )

        for cam, original_key in cam_to_original.items():
            # lerobot returns CHW float [0,1]; native returns HWC uint8.
            lr_frames = np.stack(
                [(np.clip(s[original_key].numpy(), 0, 1) * 255.0).round() for s in lr]
            ).transpose(0, 2, 3, 1)
            nat = native[f"ep{ep}__video__{cam}"].astype(np.float32)
            if lr_frames.shape != nat.shape:
                record(False, f"ep{ep}/{cam}: frame shape", f"{nat.shape} vs {lr_frames.shape}")
                continue
            mae = float(np.abs(lr_frames - nat).mean())
            record(
                mae <= args.video_mae_tol, f"ep{ep}/{cam}: video matches lerobot", f"MAE={mae:.4f}"
            )

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED: {len(failures)}/{checks} checks failed.")
        return 1
    print(f"ALL {checks} CHECKS PASSED — native v3.0 reader matches lerobot.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["compare", "native"], default="compare")
    p.add_argument("--v3-path", required=True)
    p.add_argument("--modality-config-path", default=str(DEFAULT_MODALITY_CONFIG))
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchcodec")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 50, 199])
    p.add_argument("--frames-per-episode", type=int, default=6)
    p.add_argument("--lowdim-tol", type=float, default=1e-5)
    p.add_argument("--video-mae-tol", type=float, default=2.0)
    p.add_argument("--out", default=None, help="npz output path (native mode)")
    p.add_argument(
        "--groot-python",
        default=str(REPO_ROOT / ".venv" / "bin" / "python"),
        help="GR00T venv interpreter used to produce the native dump (compare mode).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "native":
        if not args.out:
            raise SystemExit("--out is required in native mode")
        return run_native(args)
    return run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
