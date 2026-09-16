# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/activate_b300.sh"


def test_activation_prepends_nvrtc_and_is_idempotent(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.write_text('#!/bin/bash\nprintf "12.9.86|%s/nvidia/cuda_nvrtc/lib\\n" "$VIRTUAL_ENV"\n')
    python.chmod(0o755)
    environment = {**os.environ, "VIRTUAL_ENV": str(venv), "LD_LIBRARY_PATH": "/tmp/existing"}
    result = subprocess.run(
        [
            "bash",
            "-uc",
            'source "$1" && first="$LD_LIBRARY_PATH" && source "$1" '
            '&& test "$first" = "$LD_LIBRARY_PATH" && printf "PATH=%s\\n" "$LD_LIBRARY_PATH"',
            "test",
            str(SCRIPT),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"PATH={venv}/nvidia/cuda_nvrtc/lib:/tmp/existing" in result.stdout


def test_activation_failure_preserves_library_path(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.write_text("#!/bin/bash\nexit 1\n")
    python.chmod(0o755)
    environment = {**os.environ, "VIRTUAL_ENV": str(venv), "LD_LIBRARY_PATH": "/tmp/existing"}
    result = subprocess.run(
        [
            "bash",
            "-uc",
            'source "$1"; status=$?; test "$status" -eq 1 '
            '&& test "$LD_LIBRARY_PATH" = /tmp/existing && ! declare -F _b300_configure_nvrtc',
            "test",
            str(SCRIPT),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "uv pip install" in result.stdout
