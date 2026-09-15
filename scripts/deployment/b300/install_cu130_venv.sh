#!/usr/bin/env bash
# install_cu130_venv.sh -- second GR00T environment on CUDA 13 / torch 2.10 for B300 (Blackwell Ultra,
# compute capability 10.3 = sm_103) hosts, so that `--compile-blocks` (torch.compile / Inductor) works.
#
# Why a second venv: the repo's pinned torch 2.7.1+cu128 ships Triton 3.3.1, whose LLVM has no sm_103
# target, and sm_100a binaries are architecture-locked, so Inductor cannot run on sm_103 at all. The
# aarch64 flash-attn / torchcodec wheels under scripts/deployment/dgpu/wheels are built against torch
# 2.7.1, so both are rebuilt from source here. The repo's own .venv keeps working (eager mode, with
# scripts/activate_b300.sh for the NVRTC fix); this venv is only for training with torch.compile.
#
# Prerequisites
#   * the default venv exists (`uv sync --frozen --python 3.10`): its `uv pip freeze` is the dependency set
#   * a CUDA 13 toolkit (default /usr/local/cuda-13.0) and a driver >= 580
#   * CPython 3.10 headers (python3.10-dev). Without root, unpack the distro's libpython3.10-dev package
#     anywhere and point PYTHON_INCLUDE_DIR / PYTHON_LIBRARY at it (see below); deepspeed's Triton import
#     and both source builds need Python.h
#   * FFmpeg 4.4-7 runtime + development headers and pkg-config for torchcodec. Without root, unpack the
#     libav*-dev packages into a sysroot and set FFMPEG_DEV_SYSROOT
#   * ~15 GB disk, network (PyTorch cu130 index, PyPI, GitHub)
#
# Knobs (environment):
#   VENV=<dir>                    venv to create (default: <repo>/.venv-cu130)
#   WORK=<dir>                    scratch dir for clones, wheels and phase markers (default: $VENV-build)
#   BASE_VENV=<dir>               the working venv to copy the dependency set from (default: <repo>/.venv)
#   CUDA_HOME=<dir>               CUDA 13 toolkit (default: /usr/local/cuda-13.0)
#   PYTHON_INCLUDE_DIR=<dir>      directory holding Python.h (default: the venv interpreter's sysconfig include)
#   PYTHON_LIBRARY=<file>         libpython3.10.so (default: sysconfig LIBDIR/libpython3.10.so)
#   FFMPEG_DEV_SYSROOT=<dir>      sysroot with usr/include/<triplet>/libav* and usr/lib/<triplet>/pkgconfig/*.pc
#                                 (default: none -- the system pkg-config must find libavcodec)
#   INSTALL_FA4=1                 also install flash-attn-4 (CuTe DSL, pure Python; used by gr00t_fast
#                                 attention for padded batches) (default: 1)
#   MAX_JOBS=16 NVCC_THREADS=2    parallelism of the source builds (16 keeps a shared box usable)
#   TORCH_VER=2.10.0 TORCHVISION_VER=0.25.0 TORCHCODEC_VER=0.10.0 FA_TAG=v2.8.3 CUTLASS_SHA=<sha>
#
# Runtime: torch + deps ~1-5 min, torchcodec ~1-10 min, flash-attn 10 min (idle 130-core host) to 1-2 h (next
# to a running training job) at MAX_JOBS=16; ~11 min total measured on an idle host. Re-runnable: each
# phase is skipped once its marker exists in $WORK; delete a marker to redo a phase.
#
# Use afterwards:  source $VENV/bin/activate   (no activate_b300.sh needed: CUDA 13's NVRTC knows sm_103)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv-cu130}"
WORK="${WORK:-$VENV-build}"
BASE_VENV="${BASE_VENV:-$REPO_ROOT/.venv}"
TORCH_VER="${TORCH_VER:-2.10.0}"
TORCHVISION_VER="${TORCHVISION_VER:-0.25.0}"
TORCHCODEC_VER="${TORCHCODEC_VER:-0.10.0}"
FA_TAG="${FA_TAG:-v2.8.3}"
CUTLASS_SHA="${CUTLASS_SHA:-f74fea9ce35868d3ae9f8d1dce1969d7250d3f90}"   # same pin as the Spark recipe
INSTALL_FA4="${INSTALL_FA4:-1}"
export MAX_JOBS="${MAX_JOBS:-16}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

log() { echo "[$(date -Is)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(uname -m)" = "aarch64" ] || log "note: written for aarch64 hosts; on x86_64 the default venv already compiles"
[ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc under CUDA_HOME=$CUDA_HOME (need a CUDA 13 toolkit)"
[ -x "$BASE_VENV/bin/python" ] || die "BASE_VENV=$BASE_VENV has no python; run 'uv sync --frozen --python 3.10' first"
command -v uv >/dev/null || die "uv not found"
TRIPLET="$(uname -m)-linux-gnu"
HOST_LIBDIR="/usr/lib/$TRIPLET"
mkdir -p "$WORK"

# Symlinks copied out of -dev packages point at versioned runtime libraries that live on the host.
# Make any dangling link in $1 resolve by creating its target as a link to the host's copy.
fix_dangling_links() {
    local dir="$1" f target
    for f in "$dir"/*.so*; do
        [ -L "$f" ] || continue
        target="$(readlink "$f")"
        case "$target" in /*) continue ;; esac
        if [ ! -e "$dir/$target" ] && [ -e "$HOST_LIBDIR/$target" ]; then
            ln -s "$HOST_LIBDIR/$target" "$dir/$target"
            log "linked $dir/$target -> $HOST_LIBDIR/$target"
        fi
    done
}

# ---- 1. venv + torch cu130 -----------------------------------------------------------------------
if [ ! -f "$WORK/.torch-done" ]; then
    log "creating venv $VENV (python 3.10)"
    rm -rf "$VENV"; uv venv --python 3.10 "$VENV"
    log "installing torch==$TORCH_VER torchvision==$TORCHVISION_VER from the cu130 index"
    uv pip install --python "$VENV/bin/python" \
        --index-url https://download.pytorch.org/whl/cu130 \
        "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER"
    "$VENV/bin/python" - <<'EOF'
import torch, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__,
      "| device cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
EOF
    touch "$WORK/.torch-done"
fi

# Python headers / library: needed by deepspeed (Triton JIT), torchcodec and flash-attn.
PYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR:-$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')}"
PYTHON_LIBRARY="${PYTHON_LIBRARY:-$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") + "/libpython3.10.so")')}"
[ -f "$PYTHON_INCLUDE_DIR/Python.h" ] || die "no Python.h in PYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR (install python3.10-dev, or unpack libpython3.10-dev and set PYTHON_INCLUDE_DIR/PYTHON_LIBRARY)"
fix_dangling_links "$(dirname "$PYTHON_LIBRARY")"
[ -e "$PYTHON_LIBRARY" ] || die "PYTHON_LIBRARY=$PYTHON_LIBRARY does not resolve"
export CPATH="$CUDA_HOME/include:$PYTHON_INCLUDE_DIR${CPATH:+:$CPATH}"

# ---- 2. the rest of the repo's dependency set, same versions as the working venv ---------------------
if [ ! -f "$WORK/.deps-done" ]; then
    log "installing the remaining dependencies (versions frozen from $BASE_VENV)"
    uv pip freeze --python "$BASE_VENV/bin/python" \
        | grep -v -E '^(torch|torchvision|torchaudio|triton|pytorch-triton|flash_attn|flash-attn|torchcodec|nvidia-|deepspeed|gr00t|-e |isaac-gr00t)' \
        > "$WORK/base-requirements.txt"
    uv pip install --python "$VENV/bin/python" -r "$WORK/base-requirements.txt"
    log "installing deepspeed (pure-python build; its Triton import JIT-compiles a CPython extension)"
    uv pip install --python "$VENV/bin/python" "deepspeed==0.17.6"
    log "installing gr00t editable (--no-deps)"
    uv pip install --python "$VENV/bin/python" --no-deps -e "$REPO_ROOT"
    "$VENV/bin/python" - <<'EOF'
import torch, torchvision, transformers, deepspeed
print("torchvision", torchvision.__version__, "transformers", transformers.__version__,
      "deepspeed", deepspeed.__version__)
EOF
    touch "$WORK/.deps-done"
fi

# ---- 3. torchcodec from source (no aarch64 wheels on PyPI; the repo's wheel is torch-2.7.1-only) -------
if [ ! -f "$WORK/.torchcodec-done" ]; then
    if [ -n "${FFMPEG_DEV_SYSROOT:-}" ]; then
        LIBDIR="$FFMPEG_DEV_SYSROOT/usr/lib/$TRIPLET"
        fix_dangling_links "$LIBDIR"
        export PKG_CONFIG_PATH="$LIBDIR/pkgconfig"
        export PKG_CONFIG_SYSROOT_DIR="$FFMPEG_DEV_SYSROOT"
    fi
    pkg-config --exists libavcodec || die "pkg-config cannot find libavcodec (install libav*-dev or set FFMPEG_DEV_SYSROOT)"
    log "ffmpeg via pkg-config: libavcodec $(pkg-config --modversion libavcodec), cflags $(pkg-config --cflags libavcodec)"
    SRC="$WORK/torchcodec"
    [ -d "$SRC" ] || git clone --depth 1 --branch "v$TORCHCODEC_VER" https://github.com/pytorch/torchcodec.git "$SRC"
    uv pip install --python "$VENV/bin/python" ninja pybind11 setuptools wheel "cmake>=3.24"
    # torchcodec's setup.py forwards no CMake flags, but CMake honours CMAKE_TOOLCHAIN_FILE from the
    # environment: pin pybind11's config dir and the Python headers/library (pybind11's helpers use the
    # unversioned FindPython module, torchcodec the versioned one -- set both).
    cat > "$WORK/toolchain.cmake" <<EOF
set(pybind11_DIR "$("$VENV/bin/python" -m pybind11 --cmakedir)" CACHE PATH "")
set(Python3_EXECUTABLE "$VENV/bin/python" CACHE FILEPATH "")
set(Python3_INCLUDE_DIR "$PYTHON_INCLUDE_DIR" CACHE PATH "")
set(Python3_LIBRARY "$PYTHON_LIBRARY" CACHE FILEPATH "")
set(Python_EXECUTABLE "$VENV/bin/python" CACHE FILEPATH "")
set(Python_INCLUDE_DIR "$PYTHON_INCLUDE_DIR" CACHE PATH "")
set(Python_LIBRARY "$PYTHON_LIBRARY" CACHE FILEPATH "")
EOF
    export CMAKE_TOOLCHAIN_FILE="$WORK/toolchain.cmake"
    log "building torchcodec v$TORCHCODEC_VER (CPU decode)"
    ( cd "$SRC" && rm -rf build && I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 ENABLE_CUDA=0 \
        CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" PATH="$VENV/bin:$PATH" \
        nice -n 10 uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps . )
    unset PKG_CONFIG_PATH PKG_CONFIG_SYSROOT_DIR CMAKE_TOOLCHAIN_FILE
    if command -v ffmpeg >/dev/null; then
        ffmpeg -hide_banner -loglevel error -y -f lavfi -i testsrc=size=64x64:rate=30 -t 1 -pix_fmt yuv420p "$WORK/test.mp4"
        "$VENV/bin/python" - "$WORK/test.mp4" <<'EOF'
import sys, torchcodec
from torchcodec.decoders import VideoDecoder
fr = VideoDecoder(sys.argv[1], device="cpu", dimension_order="NHWC").get_frames_at(indices=[0, 10, 20]).data
print("torchcodec", torchcodec.__version__, "decoded", tuple(fr.shape), fr.dtype)
EOF
    else
        "$VENV/bin/python" -c 'import torchcodec; from torchcodec.decoders import VideoDecoder; print("torchcodec", torchcodec.__version__, "imports")'
    fi
    touch "$WORK/.torchcodec-done"
fi

# ---- 4. flash-attn 2 from source -------------------------------------------------------------------
if [ ! -f "$WORK/.flash-attn-done" ]; then
    SRC="$WORK/flash-attention"
    if [ ! -d "$SRC" ]; then
        log "cloning flash-attention $FA_TAG + cutlass $CUTLASS_SHA"
        git clone --depth 1 --branch "$FA_TAG" https://github.com/Dao-AILab/flash-attention.git "$SRC"
        rm -rf "$SRC/csrc/cutlass"; mkdir -p "$SRC/csrc/cutlass"
        git -C "$SRC/csrc/cutlass" init --quiet
        git -C "$SRC/csrc/cutlass" remote add origin https://github.com/NVIDIA/cutlass.git
        git -C "$SRC/csrc/cutlass" fetch --depth 1 --quiet origin "$CUTLASS_SHA"
        git -C "$SRC/csrc/cutlass" checkout --quiet FETCH_HEAD
    fi
    # sm_100 SASS runs on sm_103 (non-'a' code); compute_100 PTX is emitted too for forward compatibility.
    export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-100}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
    SITE=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
    export LD_LIBRARY_PATH="$SITE/torch/lib:$(find "$SITE/nvidia" -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
    uv pip install --python "$VENV/bin/python" ninja packaging psutil setuptools wheel
    log "building flash-attn $FA_TAG (MAX_JOBS=$MAX_JOBS, archs $FLASH_ATTN_CUDA_ARCHS) -- 10 min to 2 h"
    ( cd "$SRC" && nice -n 10 "$VENV/bin/python" setup.py bdist_wheel --dist-dir "$WORK/wheels" )
    WHL=$(ls -t "$WORK"/wheels/flash_attn-*.whl | head -1)
    log "installing $WHL"
    uv pip install --python "$VENV/bin/python" --no-deps "$WHL"
    touch "$WORK/.flash-attn-done"
fi

# ---- 5. flash-attn-4 (optional, pure Python + CuTe DSL; JIT-compiles for the local GPU at first use) --
if [ "$INSTALL_FA4" = "1" ] && [ ! -f "$WORK/.fa4-done" ]; then
    log "installing flash-attn-4 (pre-release, cu13 extra)"
    uv pip install --python "$VENV/bin/python" --prerelease=allow "flash-attn-4[cu13]"
    touch "$WORK/.fa4-done"
fi

# ---- 6. smoke tests --------------------------------------------------------------------------------
log "smoke tests"
"$VENV/bin/python" - <<'EOF'
import torch, flash_attn, torchcodec
from flash_attn import flash_attn_varlen_func
print("flash_attn", flash_attn.__version__, "torchcodec", torchcodec.__version__)
try:
    from flash_attn.cute import flash_attn_func as fa4  # noqa: F401
    import importlib.metadata as md
    print("flash-attn-4", md.version("flash-attn-4"))
except ImportError:
    print("flash-attn-4 not installed")
if torch.cuda.is_available():
    q = torch.randn(512, 16, 64, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 256, 512], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_func(q, q, q, cu, cu, 256, 256)
    print("flash_attn_varlen_func OK", tuple(out.shape), out.dtype)
    f = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2)
    y = f(torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    print("torch.compile (Inductor) OK on", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
EOF
uv pip check --python "$VENV/bin/python" || true
log "done: $VENV  (activate with: source $VENV/bin/activate)"
