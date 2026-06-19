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

"""Validate that GR00T's native LeRobot **v3.0** reader is equivalent to the
**v2.1** reader used by Experiment 1.

Experiment 1 trains on a v2.1 dataset that was produced by converting the very
same v3.0 dataset (``scripts/lerobot_conversion/convert_v3_to_v2.py``), so the
two datasets describe identical underlying trajectories. This script loads both
with one ``LeRobotEpisodeLoader`` each (same modality config) and asserts:

  - identical episode counts and per-episode lengths
  - identical dataset statistics (the slices read from ``meta/stats.json``)
  - identical ``state.*`` / ``action.*`` arrays per step (exact)
  - identical ``language.*`` strings per step (exact)
  - near-identical decoded video frames (pixel MAE under a small tolerance;
    the only expected difference is the v2.1 conversion re-segmenting the v3.0
    concatenated MP4s with ``ffmpeg -c copy``)

Usage:
    python scripts/b1k/validate_v3_parity.py \
        --v3-path  .../b1k/turning_on_radio_v3.0 \
        --v21-path .../b1k/turning_on_radio \
        --modality-config-path examples/b1k/r1pro.py \
        --embodiment-tag new_embodiment
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np


def load_modality_configs(modality_config_path: str, embodiment_tag_value: str) -> dict:
    """Import a .py modality config (which self-registers) and return its configs."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    config_path = Path(modality_config_path)
    if not (config_path.exists() and config_path.suffix == ".py"):
        raise FileNotFoundError(f"modality config not found or not a .py file: {config_path}")
    sys.path.append(str(config_path.parent))
    importlib.import_module(config_path.stem)
    if embodiment_tag_value not in MODALITY_CONFIGS:
        raise KeyError(
            f"embodiment tag {embodiment_tag_value!r} not registered by {config_path}; "
            f"available: {sorted(MODALITY_CONFIGS)}"
        )
    return MODALITY_CONFIGS[embodiment_tag_value]


def _stat_tree_max_abs_diff(a: Any, b: Any, path: str = "") -> tuple[float, str]:
    """Recursively diff two nested stats dicts, returning (max_abs_diff, where)."""
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict), f"stats shape mismatch at {path}"
        assert set(a) == set(b), (
            f"stats keys differ at {path or '<root>'}: {sorted(set(a) ^ set(b))}"
        )
        worst, where = 0.0, path
        for key in a:
            diff, sub = _stat_tree_max_abs_diff(a[key], b[key], f"{path}.{key}" if path else key)
            if diff >= worst:
                worst, where = diff, sub
        return worst, where
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    assert arr_a.shape == arr_b.shape, f"stats array shape mismatch at {path}"
    if arr_a.size == 0:
        return 0.0, path
    return float(np.abs(arr_a - arr_b).max()), path


class ParityReport:
    """Accumulates check results and prints a readable PASS/FAIL summary."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        self.checks += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)

    def done(self) -> bool:
        print("\n" + "=" * 70)
        if self.failures:
            print(f"FAILED: {len(self.failures)}/{self.checks} checks failed:")
            for f in self.failures:
                print(f"  - {f}")
            return False
        print(f"ALL {self.checks} CHECKS PASSED — v3.0 reader matches v2.1 (Experiment 1).")
        return True


def _episode_low_dim(loader, episode_id: int) -> dict[str, np.ndarray]:
    """Load only the state/action/language columns for an episode (no video)."""
    df = loader._load_parquet_data(episode_id)
    out: dict[str, np.ndarray] = {}
    for col in df.columns:
        if col.startswith("state.") or col.startswith("action."):
            out[col] = np.vstack([np.asarray(x, dtype=np.float32) for x in df[col]])
        elif col.startswith("language."):
            out[col] = np.asarray(list(df[col]), dtype=object)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-path", required=True, help="Path to the v3.0 dataset root.")
    parser.add_argument("--v21-path", required=True, help="Path to the v2.1 dataset root.")
    parser.add_argument("--modality-config-path", default="examples/b1k/r1pro.py")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to compare in detail (default: a spread across the dataset).",
    )
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--max-video-frames", type=int, default=8)
    parser.add_argument(
        "--video-mae-tol",
        type=float,
        default=2.0,
        help="Max allowed mean abs pixel diff (0-255) from v2.1 ffmpeg re-segmenting.",
    )
    parser.add_argument("--stats-tol", type=float, default=1e-6)
    args = parser.parse_args()

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    modality_configs = load_modality_configs(args.modality_config_path, args.embodiment_tag)

    print(f"Loading v3.0 reader: {args.v3_path}")
    v3 = LeRobotEpisodeLoader(args.v3_path, modality_configs, video_backend=args.video_backend)
    print(f"Loading v2.1 reader: {args.v21_path}")
    v21 = LeRobotEpisodeLoader(args.v21_path, modality_configs, video_backend=args.video_backend)

    report = ParityReport()

    print("\n[1] Version + structure")
    report.check(v3.is_v30, "v3 dataset detected as v3.0", v3.codebase_version)
    report.check(not v21.is_v30, "v21 dataset detected as legacy", v21.codebase_version)
    report.check(len(v3) == len(v21), "episode counts equal", f"{len(v3)} vs {len(v21)}")

    n = min(len(v3), len(v21))
    lengths_equal = all(v3.get_episode_length(i) == v21.get_episode_length(i) for i in range(n))
    report.check(lengths_equal, "per-episode lengths equal")

    print("\n[2] Dataset statistics (from meta/stats.json slices)")
    diff, where = _stat_tree_max_abs_diff(v3.get_dataset_statistics(), v21.get_dataset_statistics())
    report.check(diff <= args.stats_tol, "dataset statistics equal", f"max|Δ|={diff:.3e} @ {where}")

    if args.episodes is not None:
        episodes = args.episodes
    else:
        episodes = sorted(set(np.linspace(0, n - 1, num=min(args.num_episodes, n)).astype(int)))

    print(f"\n[3] Per-episode state/action/language ({len(episodes)} episodes: {episodes})")
    for ep in episodes:
        a = _episode_low_dim(v3, ep)
        b = _episode_low_dim(v21, ep)
        if set(a) != set(b):
            report.check(False, f"ep{ep}: column sets equal", f"{sorted(set(a) ^ set(b))}")
            continue
        worst = 0.0
        worst_key = ""
        lang_ok = True
        for key in a:
            if key.startswith("language."):
                lang_ok = lang_ok and bool(np.array_equal(a[key], b[key]))
            else:
                if a[key].shape != b[key].shape:
                    worst, worst_key = (
                        float("inf"),
                        f"{key}(shape {a[key].shape} vs {b[key].shape})",
                    )
                    break
                d = float(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64)).max())
                if d >= worst:
                    worst, worst_key = d, key
        report.check(
            worst == 0.0, f"ep{ep}: state/action exact", f"max|Δ|={worst:.3e} @ {worst_key}"
        )
        report.check(lang_ok, f"ep{ep}: language exact")

    print(
        f"\n[4] Video frames (pixel MAE, tol={args.video_mae_tol}, "
        f"{args.max_video_frames} frames/episode)"
    )
    for ep in episodes:
        length = v3.get_episode_length(ep)
        steps = np.unique(
            np.linspace(0, length - 1, num=min(args.max_video_frames, length)).astype(int)
        )
        v3_vid = v3._load_video_data(ep, steps)
        v21_vid = v21._load_video_data(ep, steps)
        if set(v3_vid) != set(v21_vid):
            report.check(
                False, f"ep{ep}: camera sets equal", f"{sorted(set(v3_vid) ^ set(v21_vid))}"
            )
            continue
        for cam in sorted(v3_vid):
            fa = v3_vid[cam].astype(np.float32)
            fb = v21_vid[cam].astype(np.float32)
            if fa.shape != fb.shape:
                report.check(False, f"ep{ep}/{cam}: frame shape equal", f"{fa.shape} vs {fb.shape}")
                continue
            mae = float(np.abs(fa - fb).mean())
            report.check(mae <= args.video_mae_tol, f"ep{ep}/{cam}: frames match", f"MAE={mae:.4f}")

    return 0 if report.done() else 1


if __name__ == "__main__":
    raise SystemExit(main())
