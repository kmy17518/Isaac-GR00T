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

"""Parity tests for the native LeRobot v3.0 reader in ``LeRobotEpisodeLoader``.

Two layers:

1. ``TestSyntheticV30VsV21`` (always runs, CPU-only, no external data): builds a
   tiny matched pair of datasets — one in v3.0 layout (multi-episode parquet +
   parquet metadata), one in v2.1 layout (per-episode parquet + JSONL metadata) —
   from the *same* underlying arrays, and asserts the loader returns identical
   state/action/language and that v3.0 row-slicing selects the correct episode.
   This covers the v3.0-specific metadata parsing and parquet slicing.

2. ``TestRealB1KParity`` (skipped unless the BEHAVIOR-1K demo datasets are present
   locally): full parity incl. decoded video frames, exercised against the real
   ``turning_on_radio`` (v2.1) and ``turning_on_radio_v3.0`` datasets. Point it at
   data via ``B1K_DATA_ROOT`` or rely on the default challenge path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import ModalityConfig
import numpy as np
import pandas as pd
import pytest


STATE_DIM = 6
ACTION_DIM = 4
TASKS = ["pick up the cube", "open the drawer"]

MODALITY_JSON = {
    "state": {"a": {"start": 0, "end": 3}, "b": {"start": 3, "end": 6}},
    "action": {"x": {"start": 0, "end": 2}, "y": {"start": 2, "end": 4}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


def _modality_configs() -> dict[str, ModalityConfig]:
    return {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["a", "b"]),
        "action": ModalityConfig(delta_indices=[0, 1], modality_keys=["x", "y"]),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }


def _episode_frames(ep: int, length: int) -> dict[str, np.ndarray]:
    """Deterministic per-(episode, frame) values so mis-slicing is detectable."""
    base = np.arange(length, dtype=np.float32)[:, None]
    state = (ep * 1000 + base + np.arange(STATE_DIM, dtype=np.float32)[None, :]).astype(np.float32)
    action = (ep * 1000 + base + 0.5 + np.arange(ACTION_DIM, dtype=np.float32)[None, :]).astype(
        np.float32
    )
    return {"observation.state": state, "action": action}


def _stats_json() -> dict:
    return {
        "observation.state": {
            s: list(np.zeros(STATE_DIM) + i)
            for i, s in enumerate(["mean", "std", "min", "max", "q01", "q99"])
        },
        "action": {
            s: list(np.zeros(ACTION_DIM) + i)
            for i, s in enumerate(["mean", "std", "min", "max", "q01", "q99"])
        },
    }


def _features() -> dict:
    return {
        "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": None},
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _write_common_meta(meta: Path) -> None:
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "modality.json").write_text(json.dumps(MODALITY_JSON))
    (meta / "stats.json").write_text(json.dumps(_stats_json()))


def _make_v21(root: Path, ep_lengths: list[int]) -> None:
    meta = root / "meta"
    _write_common_meta(meta)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 1000,
        "features": _features(),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
    }
    (meta / "info.json").write_text(json.dumps(info))
    with open(meta / "tasks.jsonl", "w") as f:
        for i, t in enumerate(TASKS):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    with open(meta / "episodes.jsonl", "w") as f:
        for ep, length in enumerate(ep_lengths):
            f.write(
                json.dumps(
                    {"episode_index": ep, "tasks": [TASKS[ep % len(TASKS)]], "length": length}
                )
                + "\n"
            )
    for ep, length in enumerate(ep_lengths):
        frames = _episode_frames(ep, length)
        df = pd.DataFrame(
            {
                "observation.state": list(frames["observation.state"]),
                "action": list(frames["action"]),
                "episode_index": np.full(length, ep, dtype=np.int64),
                "task_index": np.full(length, ep % len(TASKS), dtype=np.int64),
            }
        )
        out = root / f"data/chunk-000/episode_{ep:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)


def _make_v30(root: Path, ep_lengths: list[int], episodes_per_file: int = 2) -> None:
    meta = root / "meta"
    _write_common_meta(meta)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "chunks_size": 1000,
        "features": _features(),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
    }
    (meta / "info.json").write_text(json.dumps(info))

    # tasks.parquet (task string as the named index, mirroring real v3.0 writers)
    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(TASKS)))}, index=pd.Index(TASKS, name="task")
    )
    (meta / "tasks.parquet").parent.mkdir(parents=True, exist_ok=True)
    tasks_df.to_parquet(meta / "tasks.parquet")

    # Pack episodes into multi-episode data files; track each episode's location.
    episode_rows: list[dict] = []
    global_cursor = 0
    file_buffers: dict[int, list[pd.DataFrame]] = {}
    file_base_index: dict[int, int] = {}
    for ep, length in enumerate(ep_lengths):
        file_idx = ep // episodes_per_file
        frames = _episode_frames(ep, length)
        df = pd.DataFrame(
            {
                "observation.state": list(frames["observation.state"]),
                "action": list(frames["action"]),
                "episode_index": np.full(length, ep, dtype=np.int64),
                "task_index": np.full(length, ep % len(TASKS), dtype=np.int64),
            }
        )
        file_buffers.setdefault(file_idx, [])
        file_base_index.setdefault(file_idx, global_cursor)
        file_buffers[file_idx].append(df)
        episode_rows.append(
            {
                "episode_index": ep,
                "tasks": [TASKS[ep % len(TASKS)]],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": file_idx,
                "dataset_from_index": global_cursor,
                "dataset_to_index": global_cursor + length,
            }
        )
        global_cursor += length

    for file_idx, buffers in file_buffers.items():
        out = root / f"data/chunk-000/file-{file_idx:03d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(buffers, ignore_index=True).to_parquet(out)

    ep_dir = meta / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(ep_dir / "file-000.parquet")


@pytest.fixture(scope="module")
def synthetic_pair(tmp_path_factory) -> tuple[Path, Path, list[int]]:
    ep_lengths = [4, 5, 6, 3, 7]
    base = tmp_path_factory.mktemp("lerobot_v3_parity")
    v30 = base / "ds_v30"
    v21 = base / "ds_v21"
    _make_v30(v30, ep_lengths)
    _make_v21(v21, ep_lengths)
    return v30, v21, ep_lengths


def _episode_low_dim(loader: LeRobotEpisodeLoader, ep: int) -> dict[str, np.ndarray]:
    df = loader._load_parquet_data(ep)
    out: dict[str, object] = {}
    for col in df.columns:
        if col.startswith("state.") or col.startswith("action."):
            out[col] = np.vstack([np.asarray(x, dtype=np.float32) for x in df[col]])
        elif col.startswith("language."):
            out[col] = list(df[col])
    return out


class TestSyntheticV30VsV21:
    def test_version_detection(self, synthetic_pair):
        v30, v21, _ = synthetic_pair
        cfg = _modality_configs()
        assert LeRobotEpisodeLoader(v30, cfg).is_v30 is True
        assert LeRobotEpisodeLoader(v21, cfg).is_v30 is False

    def test_episode_counts_and_lengths(self, synthetic_pair):
        v30, v21, ep_lengths = synthetic_pair
        cfg = _modality_configs()
        l3 = LeRobotEpisodeLoader(v30, cfg)
        l21 = LeRobotEpisodeLoader(v21, cfg)
        assert len(l3) == len(l21) == len(ep_lengths)
        assert l3.episode_lengths == ep_lengths
        assert l3.episode_lengths == l21.episode_lengths

    def test_tasks_map_parity(self, synthetic_pair):
        v30, v21, _ = synthetic_pair
        cfg = _modality_configs()
        assert LeRobotEpisodeLoader(v30, cfg).tasks_map == {0: TASKS[0], 1: TASKS[1]}
        assert LeRobotEpisodeLoader(v30, cfg).tasks_map == LeRobotEpisodeLoader(v21, cfg).tasks_map

    @pytest.mark.parametrize("ep", [0, 1, 2, 3, 4])
    def test_state_action_language_parity(self, synthetic_pair, ep):
        v30, v21, _ = synthetic_pair
        cfg = _modality_configs()
        a = _episode_low_dim(LeRobotEpisodeLoader(v30, cfg), ep)
        b = _episode_low_dim(LeRobotEpisodeLoader(v21, cfg), ep)
        assert set(a) == set(b)
        for key in a:
            if key.startswith("language."):
                assert a[key] == b[key], key
            else:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    @pytest.mark.parametrize("ep", [0, 2, 4])
    def test_v30_slicing_selects_correct_episode(self, synthetic_pair, ep):
        """The packed v3.0 file must return *this* episode's rows, not a neighbor's."""
        v30, _, ep_lengths = synthetic_pair
        cfg = _modality_configs()
        loader = LeRobotEpisodeLoader(v30, cfg)
        df = loader._load_parquet_data(ep)
        assert len(df) == ep_lengths[ep]
        expected = _episode_frames(ep, ep_lengths[ep])["observation.state"]
        got = np.vstack(
            [np.concatenate([df["state.a"].iloc[i], df["state.b"].iloc[i]]) for i in range(len(df))]
        )
        np.testing.assert_array_equal(got, expected)

    def test_dataset_statistics_parity(self, synthetic_pair):
        v30, v21, _ = synthetic_pair
        cfg = _modality_configs()
        s3 = LeRobotEpisodeLoader(v30, cfg).get_dataset_statistics()
        s21 = LeRobotEpisodeLoader(v21, cfg).get_dataset_statistics()
        assert s3.keys() == s21.keys()
        for modality in s3:
            for group in s3[modality]:
                for stat in s3[modality][group]:
                    np.testing.assert_array_equal(
                        np.asarray(s3[modality][group][stat]),
                        np.asarray(s21[modality][group][stat]),
                        err_msg=f"{modality}.{group}.{stat}",
                    )


class TestV30Caching:
    """The v3.0 per-file caches must speed up access *without changing data*:
    each consolidated parquet/mp4 is opened once, not once per episode."""

    def test_cache_preserves_data(self, synthetic_pair):
        """Cached (default) and disabled (size 0) loaders return identical data."""
        v30, _, _ = synthetic_pair
        cfg = _modality_configs()
        cached = LeRobotEpisodeLoader(v30, cfg)
        uncached = LeRobotEpisodeLoader(v30, cfg, data_cache_size=0)
        for ep in range(len(cached)):
            a = _episode_low_dim(cached, ep)
            b = _episode_low_dim(uncached, ep)
            assert set(a) == set(b)
            for key in a:
                if key.startswith("language."):
                    assert a[key] == b[key], key
                else:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_opens_each_file_once(self, synthetic_pair, monkeypatch):
        """With caching on, reading every episode opens each data file exactly once
        (and strictly fewer times than the episode count)."""
        import gr00t.data.dataset.lerobot_episode_loader as loader_mod

        v30, _, ep_lengths = synthetic_pair
        loader = LeRobotEpisodeLoader(v30, _modality_configs())

        opened: list[str] = []
        original = loader_mod.pq.read_table

        def _counting(path, *args, **kwargs):
            opened.append(str(path))
            return original(path, *args, **kwargs)

        monkeypatch.setattr(loader_mod.pq, "read_table", _counting)
        for ep in range(len(loader)):
            loader._load_parquet_data(ep)

        n_files = len(loader._file_row_base)
        assert len(opened) == n_files
        assert len(set(opened)) == n_files
        assert n_files < len(ep_lengths)  # multiple episodes share each file

    def test_disabled_cache_reopens_per_episode(self, synthetic_pair, monkeypatch):
        """Disabling the cache reverts to one open per episode — proving the cache
        (not some other change) is what collapses the reads."""
        import gr00t.data.dataset.lerobot_episode_loader as loader_mod

        v30, _, ep_lengths = synthetic_pair
        loader = LeRobotEpisodeLoader(v30, _modality_configs(), data_cache_size=0)

        opened: list[str] = []
        original = loader_mod.pq.read_table

        def _counting(path, *args, **kwargs):
            opened.append(str(path))
            return original(path, *args, **kwargs)

        monkeypatch.setattr(loader_mod.pq, "read_table", _counting)
        for ep in range(len(loader)):
            loader._load_parquet_data(ep)

        assert len(opened) == len(ep_lengths)


def test_video_reader_pool_reuses_decoder(monkeypatch):
    """VideoReaderPool builds one decoder per file and reuses it across calls."""
    import types

    import gr00t.utils.video_utils as video_utils

    built: list[str] = []

    class _FakeDecoder:
        def __init__(self, path, **kwargs):
            built.append(path)

        def get_frames_at(self, indices):
            arr = np.zeros((len(indices), 2, 2, 3), dtype=np.uint8)
            return types.SimpleNamespace(data=types.SimpleNamespace(numpy=lambda: arr))

    fake_tc = types.SimpleNamespace(decoders=types.SimpleNamespace(VideoDecoder=_FakeDecoder))
    monkeypatch.setattr(video_utils, "_lazy_import_torchcodec", lambda: fake_tc)
    monkeypatch.setattr(video_utils, "resolve_backend", lambda path, backend: "torchcodec")

    pool = video_utils.VideoReaderPool("torchcodec", max_size=4)
    frames = None
    for _ in range(5):
        frames = pool.get_frames_by_indices("/fake/a.mp4", [0, 1])
    pool.get_frames_by_indices("/fake/b.mp4", [0])

    assert frames.shape == (2, 2, 2, 3)
    assert built == ["/fake/a.mp4", "/fake/b.mp4"]  # one construction per distinct file


# --- Real-data parity (skipped unless BEHAVIOR-1K demos are present) ----------


def _b1k_root() -> Path | None:
    candidates = []
    env = os.environ.get("B1K_DATA_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(
        Path("/home/stuart/ThunderPuppies/BEHAVIOR-1K/datasets/2026-challenge-demos/b1k")
    )
    for root in candidates:
        if (root / "turning_on_radio_v3.0" / "meta" / "info.json").exists() and (
            root / "turning_on_radio" / "meta" / "info.json"
        ).exists():
            return root
    return None


B1K_ROOT = _b1k_root()
b1k_required = pytest.mark.skipif(
    B1K_ROOT is None,
    reason="BEHAVIOR-1K turning_on_radio (v2.1) + turning_on_radio_v3.0 not found locally",
)

MODALITY_CONFIG_PY = Path(__file__).resolve().parents[3] / "examples" / "b1k" / "r1pro.py"


def _b1k_modality_configs() -> dict:
    import importlib
    import sys

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    sys.path.append(str(MODALITY_CONFIG_PY.parent))
    importlib.import_module(MODALITY_CONFIG_PY.stem)
    return MODALITY_CONFIGS["new_embodiment"]


@pytest.fixture(scope="module")
def b1k_loaders():
    cfg = _b1k_modality_configs()
    v30 = LeRobotEpisodeLoader(B1K_ROOT / "turning_on_radio_v3.0", cfg)
    v21 = LeRobotEpisodeLoader(B1K_ROOT / "turning_on_radio", cfg)
    return v30, v21


@b1k_required
class TestRealB1KParity:
    def test_versions_and_lengths(self, b1k_loaders):
        v30, v21 = b1k_loaders
        assert v30.is_v30 and not v21.is_v30
        assert len(v30) == len(v21)
        assert v30.episode_lengths == v21.episode_lengths

    @pytest.mark.parametrize("ep", [0, 99, 199])
    def test_state_action_language_exact(self, b1k_loaders, ep):
        v30, v21 = b1k_loaders
        a = _episode_low_dim(v30, ep)
        b = _episode_low_dim(v21, ep)
        assert set(a) == set(b)
        for key in a:
            if key.startswith("language."):
                assert a[key] == b[key], key
            else:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    @pytest.mark.parametrize("ep", [0, 199])
    def test_video_frames_match(self, b1k_loaders, ep):
        v30, v21 = b1k_loaders
        length = v30.get_episode_length(ep)
        steps = np.unique(np.linspace(0, length - 1, num=min(8, length)).astype(int))
        f3 = v30._load_video_data(ep, steps)
        f21 = v21._load_video_data(ep, steps)
        assert set(f3) == set(f21)
        for cam in f3:
            assert f3[cam].shape == f21[cam].shape, cam
            mae = float(np.abs(f3[cam].astype(np.float32) - f21[cam].astype(np.float32)).mean())
            assert mae <= 2.0, f"{cam}: video MAE {mae} exceeds tolerance"
