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

"""Gated wrapper for the cross-venv native-vs-lerobot v3.0 parity check.

This drives ``scripts/b1k/validate_v3_lerobot_parity.py``, which loads the v3.0
dataset with the real ``lerobot`` library (in the conversion venv) and asserts
the native GR00T reader returns identical frames/states. It is skipped unless:

  - the conversion venv interpreter with ``lerobot`` is available
    (``LEROBOT_PYTHON`` env var, or a discovered ``scripts/lerobot_conversion``
    venv), and
  - the BEHAVIOR-1K ``turning_on_radio_v3.0`` dataset is present
    (``B1K_DATA_ROOT`` env var, or the default challenge path).

So it stays green/skipped in CI (which has neither) and runs locally where both
environments and the data exist.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "b1k" / "validate_v3_lerobot_parity.py"


def _find_lerobot_python() -> str | None:
    candidates = []
    if os.environ.get("LEROBOT_PYTHON"):
        candidates.append(Path(os.environ["LEROBOT_PYTHON"]))
    candidates.append(REPO_ROOT / "scripts" / "lerobot_conversion" / ".venv" / "bin" / "python")
    candidates.append(
        REPO_ROOT.parent
        / "Isaac-GR00T"
        / "scripts"
        / "lerobot_conversion"
        / ".venv"
        / "bin"
        / "python"
    )
    for python in candidates:
        if not python.exists():
            continue
        probe = subprocess.run([str(python), "-c", "import lerobot"], capture_output=True)
        if probe.returncode == 0:
            return str(python)
    return None


def _find_v3_dataset() -> Path | None:
    candidates = []
    if os.environ.get("B1K_DATA_ROOT"):
        candidates.append(Path(os.environ["B1K_DATA_ROOT"]) / "turning_on_radio_v3.0")
    candidates.append(
        Path(
            "/home/stuart/ThunderPuppies/BEHAVIOR-1K/datasets/2026-challenge-demos/"
            "b1k/turning_on_radio_v3.0"
        )
    )
    for path in candidates:
        if (path / "meta" / "info.json").exists() and (path / "meta" / "modality.json").exists():
            return path
    return None


LEROBOT_PYTHON = _find_lerobot_python()
V3_DATASET = _find_v3_dataset()


@pytest.mark.skipif(
    LEROBOT_PYTHON is None or V3_DATASET is None,
    reason="requires the lerobot conversion venv and the turning_on_radio_v3.0 dataset",
)
def test_native_reader_matches_lerobot():
    cmd = [
        LEROBOT_PYTHON,
        str(SCRIPT),
        "--v3-path",
        str(V3_DATASET),
        "--groot-python",
        sys.executable,
        "--episodes",
        "0",
        "1",
        "199",
        "--frames-per-episode",
        "4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"native-vs-lerobot parity failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "ALL" in result.stdout and "CHECKS PASSED" in result.stdout, result.stdout
