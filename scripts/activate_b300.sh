#!/bin/bash
# Source after activating the CUDA 12.8 venv to use NVRTC with sm_103 support.
# Install first: uv pip install --python .venv/bin/python 'nvidia-cuda-nvrtc-cu12>=12.9.86,<13'

_b300_configure_nvrtc() {
    local python_bin="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
    local nvrtc version lib_dir existing_path="${LD_LIBRARY_PATH:-}"
    python_bin="${python_bin:-python3}"
    if ! nvrtc="$("$python_bin" - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

dist = distribution("nvidia-cuda-nvrtc-cu12")
version = dist.version
if not (12, 9) <= tuple(map(int, version.split(".")[:2])) < (13, 0):
    raise RuntimeError(f"NVRTC {version} does not satisfy >=12.9,<13")
lib_dir = Path(dist.locate_file("nvidia/cuda_nvrtc/lib"))
if not (lib_dir / "libnvrtc.so.12").is_file():
    raise FileNotFoundError(lib_dir / "libnvrtc.so.12")
print(f"{version}|{lib_dir}")
PY
    )"; then
        printf '%s\n' 'WARNING: NVRTC >=12.9,<13 is required for sm_103 with the CUDA 12.8 venv.'
        printf '%s\n' 'Install: uv pip install --python .venv/bin/python "nvidia-cuda-nvrtc-cu12>=12.9.86,<13"'
        return 1
    fi
    version="${nvrtc%%|*}"
    lib_dir="${nvrtc#*|}"
    if [ "${existing_path%%:*}" != "$lib_dir" ]; then
        export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    printf 'B300 environment configured: NVRTC %s from %s\n' "$version" "$lib_dir"
}

if _b300_configure_nvrtc; then
    unset -f _b300_configure_nvrtc
else
    unset -f _b300_configure_nvrtc
    return 1 2>/dev/null || exit 1
fi
