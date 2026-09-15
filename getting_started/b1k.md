## GR00T N1.7

This tutorial provides a simplest version instruction to finetune GR00T N1.7 on the 2026 BEHAVIOR-1K Challenge dataset.

### Repo Clone

```
git clone <Isaac-GR00T repo URL>
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

This finetuning instruction is adapted from the original Isaac-GR00T repo. 

### Installation

GR00T uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```
cd Isaac-GR00T
uv sync --frozen --python 3.10
uv pip install --python .venv/bin/python websockets

source .venv/bin/activate

# Install behavior for eval (creates a separate `behavior` conda env)
cd $PATH_TO_BEHAVIOR_1K
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
```

#### Blackwell GPUs (B300)

- **B300, `sm_103`)** — needs a newer NVRTC. Precompiled kernels are fine (`sm_100` SASS runs on `sm_103`), but torch's bundled CUDA 12.8 NVRTC predates `sm_103`, so every kernel PyTorch compiles *at runtime* (the "jiterator" ops, e.g. `torch.prod` on int64 in Qwen3-VL's `rot_pos_emb`) dies on the very first training step — and again in `serve_b1k.py` — with:

  ```
  nvrtc: error: invalid value for --gpu-architecture (-arch)
  ```

  Fix: install CUDA 12.9's NVRTC (same `libnvrtc.so.12` soname, ABI-compatible) into the venv once, then put it ahead of torch's copy in **every shell** you train or serve from:

  ```
  uv pip install --python .venv/bin/python "nvidia-cuda-nvrtc-cu12>=12.9.86,<13"   # once

  source .venv/bin/activate
  source scripts/activate_b300.sh          # each new shell, after activating the venv
  ```

  `activate_b300.sh` prepends the wheel's `lib/` dir to `LD_LIBRARY_PATH` (which wins over the RUNPATH torch uses to find `libnvrtc.so.12`), prints `B300 environment configured: NVRTC 12.9.x ...` on success, is idempotent, and warns with the install command if the wheel is missing. If you do not have the script, the equivalent one-liner is:

  ```
  export LD_LIBRARY_PATH="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ```

- **`torch.compile` on B300 needs a CUDA 13 build of PyTorch — `scripts/deployment/b300/install_cu130_venv.sh` builds it.** The pinned torch 2.7.1+cu128 ships Triton 3.3.1, whose LLVM has no `sm_103` target (`LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32`), and `sm_100a` binaries are architecture-locked (`no kernel image is available for execution on the device`), so `--compile-blocks` cannot run in the default environment. It does with `torch==2.10.0+cu130` (Triton 3.6). Because the aarch64 `flash-attn` and `torchcodec` wheels in `scripts/deployment/dgpu/wheels/` are built against torch 2.7.1, the script makes a *second* venv rather than editing `uv.lock` (the default `.venv` keeps working for eager training and serving):

  1. `uv venv` (Python 3.10) + `torch==2.10.0` / `torchvision==0.25.0` from `https://download.pytorch.org/whl/cu130` (~5 min);
  2. every other dependency at the versions of the default venv (`uv pip freeze` of `.venv`), `deepspeed==0.17.6` (pure-Python build) and the repo as an editable install;
  3. `torchcodec` 0.10.0 from source against the host FFmpeg (~10 min). CMake gets a toolchain file that pins pybind11's config dir and the Python headers/library, because pybind11 uses the unversioned `FindPython` module and torchcodec the versioned one;
  4. `flash-attn` 2.8.3 from source with the Spark recipe's CUTLASS pin, `FLASH_ATTN_CUDA_ARCHS=100` (`sm_100` SASS runs on `sm_103`) — 10 min on an idle 130-core box, 1–2 h next to a running training job, at `MAX_JOBS=16` (deliberately throttled so a shared box stays usable);
  5. `flash-attn-4` (CuTe DSL, pure Python; used by `gr00t_fast` attention for padded batches — see [Training throughput knobs](#training-throughput-knobs)); `INSTALL_FA4=0` skips it;
  6. smoke tests: flash-attn varlen, torchcodec decode of a generated clip, and a `torch.compile` of a small function on the GPU.

  Prerequisites: the default venv (`uv sync --frozen --python 3.10`), a CUDA 13 toolkit (`CUDA_HOME`, default `/usr/local/cuda-13.0`, driver ≥ 580), CPython 3.10 headers, and FFmpeg development headers with `pkg-config`. Hosts without root and without `python3.10-dev` / `libav*-dev` can unpack those distro packages anywhere and point the script at them — that is how the environment behind this guide was built:

  ```
  VENV=/path/to/venv-cu130 \
  PYTHON_INCLUDE_DIR=/path/to/libpython3.10-dev/usr/include/python3.10 \
  PYTHON_LIBRARY=/path/to/libpython3.10-dev/usr/lib/aarch64-linux-gnu/libpython3.10.so \
  FFMPEG_DEV_SYSROOT=/path/to/ffmpeg-dev \
  bash scripts/deployment/b300/install_cu130_venv.sh
  ```

  The script is re-runnable (each phase leaves a marker in `$WORK`, default `$VENV-build`; delete one to redo that phase) and fixes the dangling `.so` symlinks that unpacked `-dev` packages leave behind. Afterwards `source /path/to/venv-cu130/bin/activate` and train with `--compile-blocks …`; `activate_b300.sh` is not needed there (CUDA 13's NVRTC knows `sm_103`). Everything else in this guide is unchanged; a checkpoint trained in this venv serves fine from the default one. On x86_64 with H100/A100 none of this applies: the default environment compiles fine.

#### DeepSpeed on aarch64 hosts

Multi-GPU training (`--num-gpus > 1`) uses DeepSpeed ZeRO-2 by default, but `pyproject.toml` pins `deepspeed` for x86_64 Linux only (no aarch64 wheels on PyPI), so `uv sync` does not install it on aarch64 and `torchrun … train_b1k.py` fails with `DeepSpeed is not available`. Two options:

- **Install it from source** (pure-Python build, a few seconds; ZeRO-2 with the trainer's `adamw_torch` optimizer needs none of DeepSpeed's compiled ops):

  ```
  uv pip install --python .venv/bin/python "deepspeed==0.17.6"
  ```

  A plain `uv sync` (without `--inexact`) removes it again, so re-run this after syncing. `import deepspeed` also imports Triton's inference kernels, which JIT-build a small CPython extension on first import — the host therefore needs the CPython headers (`python3.10-dev`, or another copy of `Python.h` made visible through `CPATH`); without them the import dies with `fatal error: Python.h: No such file or directory`.
- **Fall back to plain DDP** with `--use-ddp` on `train_b1k.py`. Every GPU then holds the full optimizer state; the trainable action head is ~1.6B of the 3.1B parameters, so this costs roughly 15–20 GB more per GPU at the same batch size than ZeRO-2 and is otherwise equivalent.

The N1.7 backbone `nvidia/Cosmos-Reason2-2B` is gated. Accept the gate at [https://huggingface.co/nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) before training. 

### Finetune GR00T

We provide a GR00T N1.7 checkpoint for:

- turning_on_radio task [here](add checkpoint link).

If you would like to run eval only feel free to skip to the last section.

```
export TASK=turning_on_radio                                            # any challenge task
export DATA_ROOT=$PATH_TO_BEHAVIOR_1K/datasets/2026-challenge-demos/b1k # holds one folder per task
export DATASET_PATH=$DATA_ROOT/$TASK                                    # e.g. .../2026-challenge-demos/b1k/turning_on_radio
export OUTPUT_DIR=outputs/b1k-$TASK
```

#### Which demos are on disk: one task or all 100

The demos are one LeRobot v3.0 dataset on the Hub ([behavior-1k/2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos)); each task is one chunk (`chunk-000` = task 0 = `turning_on_radio`, ...). `--dataset-path` always points at a LeRobot root (`data/`, `meta/`, `videos/`), which is either

- a **per-task partial download** — the challenge docs' `huggingface-cli download --include "data/$CHUNK/**" --include "meta/episodes/$CHUNK/**" --include "videos/*/$CHUNK/**" --include meta/info.json --include meta/stats.json --include meta/tasks.parquet`, holding one task's chunks next to the dataset-wide metadata; or
- the **full 3.3 TB root** with all 100 tasks.

Training reads whatever is under the root. To train on one (or a few) task(s) pass `--task-names` to `train_b1k.py`; it works identically on both layouts:

```
--task-names $TASK                      # e.g. turning_on_radio; several names allowed
```

Only the selected tasks' episodes are loaded, and their normalization statistics are computed over those episodes alone and cached in `meta/task_subsets/<task>/{stats,relative_stats}.json` — the dataset-wide `meta/stats.json` is never overwritten, and other subsets of the same root get their own directory. On the full root this is also what keeps the first run from scanning 3 TB of parquet for statistics. On a partial download of that task the flag is a no-op apart from the stats location; a partial download of *other* tasks fails fast (`No episodes of task ...`), as does a misspelled name. `gr00t/data/stats.py ... --task-names $TASK` precomputes the same files.

#### Dataset version: LeRobot v3.0 (default) or v2.1

The challenge demos ship as **LeRobot v3.0**. The GR00T loader reads both **v3.0** and **v2.1** natively (it auto-detects the version from `meta/info.json`); it only additionally needs the GR00T-specific `meta/modality.json` deployed below. Choose one:

- **v3.0 — default, no conversion.** Train directly on the demos as released; `$DATASET_PATH` already points at them.
- **v2.1 — optional, convert first.** Only if your tooling specifically needs v2.1. The converter builds its own environment and runs **in place**: `$DATA_ROOT/$TASK` becomes v2.1 and the original v3.0 is backed up to `$DATA_ROOT/${TASK}_v3.0`.

To convert to v2.1:

```
cd scripts/lerobot_conversion
uv venv --python 3.11 .venv && source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475" \
  huggingface_hub jsonlines numpy pyarrow tqdm
python convert_v3_to_v2.py --root $DATA_ROOT --repo-id $TASK
cd ../..                       # back to the repo root
source .venv/bin/activate      # re-activate the GR00T venv (conversion used its own)
```

The conversion carries `meta/tasks.jsonl` over verbatim (task ids *and* natural-language descriptions), so both `--prompt-source` options keep working on a converted dataset; `episodes.jsonl` uses the same task strings as `tasks.jsonl`, as LeRobot v2.1 expects.

#### Deploy modality.json

Before we can run training, we need GR00T-specific `meta/modality.json`. Deploy it into each task dataset (point it at the root that holds your task folders — run this **after** any v2.1 conversion, since conversion does not carry it over):

```
python scripts/b1k/deploy_modality.py $DATA_ROOT
```

The per-task partial download does not include `meta/tasks.jsonl`, the table both `--prompt-source` options read (see [Language prompt](#language-prompt)); when it is missing from a v3.0 dataset, `deploy_modality.py` installs the repo's verbatim copy (`examples/b1k/tasks.jsonl`) after checking it against the dataset's `meta/tasks.parquet`, and reports `[write] .../meta/tasks.jsonl`.

Normalization statistics (`meta/stats.json`) are generated automatically on the first training run.

#### (Optional) Pre-cache base models

Training auto-downloads the base model and its gated backbone on the first run (with `HF_TOKEN` set), but you can pre-cache them first to fail fast on access/network issues:

```
export HF_TOKEN=hf_xxx         # the account that accepted the Cosmos-Reason2-2B gate
python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN")
snapshot_download("nvidia/GR00T-N1.7-3B", token=tok)
snapshot_download("nvidia/Cosmos-Reason2-2B", token=tok)  # gated backbone
PY
```

#### Train

Run the following command to finetune GR00T:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=online OMP_NUM_THREADS=4 \
torchrun --nproc_per_node=8 --master_port=29500 scripts/b1k/train_b1k.py \
    --experiment-name b1k-$TASK \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path $DATASET_PATH \
    --task-names $TASK \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/b1k/r1pro.py \
    --num-gpus 8 \
    --global-batch-size 2048 \
    --output-dir $OUTPUT_DIR \
    --save-steps 1500 --save-total-limit 5 --max-steps 150000 \
    --dataloader-num-workers 8 --decode-only-used-frames
```

`--task-names $TASK` restricts training to that task whether `$DATASET_PATH` is a per-task partial download or the full 100-task root (see [above](#which-demos-are-on-disk-one-task-or-all-100)); drop it to train on every task under the root.

Checkpoints land in `$OUTPUT_DIR/b1k-$TASK/checkpoint-<step>/`, each one standalone and directly servable.

**Tune** `OMP_NUM_THREADS` **and** `--dataloader-num-workers` **to your CPU.**

#### Training throughput knobs

What a step costs, measured on 4× B300 at 1024 samples per GPU (global batch 4096) on the `turning_on_radio` demos, and the knobs that changed it. Everything marked *exact* produces bit-identical tensors to the stock code path and is verified by tests (`tests/gr00t/model/test_qwen3_vl_fast_positions.py`, `tests/gr00t/model/test_gr00t_processor.py::TestPixelValuesDtype`).

- **Batched Qwen3-VL position ids** (`Gr00tN1d7Config.fast_vl_position_ids`, default on; *exact*). The stock `transformers` Qwen3-VL code computes M-RoPE position ids and the vision tower's position tables with Python loops over every sample and every image — ~55k `.item()` device syncs and ~6 s of CPU per 1024-sample step, with the GPU idle meanwhile. `gr00t/model/modules/qwen3_vl_fast_positions.py` computes the same tensors with batched ops for any mix of prompt lengths, image counts/sizes, video frames and padding, falling back to the original for layouts it does not recognise. Forward+backward at 1024/GPU: 7.2 s → 3.4 s. Set `"fast_vl_position_ids": false` in the model config to disable.
- **No full-vocabulary logits** (*exact*). The backbone only reads `hidden_states[-1]`; `Qwen3Backbone` now passes `logits_to_keep=1`, skipping a `(B, 207, 151936)` logits tensor that was computed and discarded every step.
- `--collate-pixel-values-dtype uint8` (default; *exact*). The collator ships the VLM processor's *unnormalized* uint8 patches — a quarter of the float32 bytes (1.2 instead of 4.8 MB/sample) and no float math in the worker — and `Qwen3Backbone` applies the processor's fp32 rescale+normalize (`(x − mean·255) / (std·255)`, the same `sub`/`div_` sequence) on the GPU; the resulting tensors are bit-identical to the processor's. `bfloat16` emits normalized patches in bf16 instead (also identical whenever the vision tower computes in bf16, half the bytes); `None` is the stock float32. Recorded in the checkpoint's `processor_config.json` (`pixel_values_dtype`), so serving does the same.
- **Numpy episode indexing in `get_shard`** (*exact*). Per-step extraction used ~100k pandas `.iloc` calls per 1024-step shard (~8 % of a worker's CPU); `get_shard` now hands `extract_step_data` an `EpisodeColumns` view of the episode DataFrame (the same row objects). DataFrame callers are unchanged.
- `--dataloader-num-workers` / `--dataloader-prefetch-factor` (default 2, PyTorch's). Each worker holds `prefetch_factor` complete per-GPU batches in shared memory (2.4 MB/sample in bf16, 4.8 MB in fp32) on top of its own ~4–10 GB of decode/augmentation buffers, so host RAM is `workers × (buffers + prefetch × batch)` **per rank**. On hosts with a hard memory limit, keep `prefetch_factor 1` and spend the RAM on workers instead. A worker's CPU per sample is dominated by video decode (see the video view below; ~22 ms of ~35 ms on the source data), then the albumentations + VLM-processor stage (~9 ms); size workers so that `workers × rate ≥ per-GPU batch / step time`, and check the log: `Wait for shard … in 0.00 seconds` means the data side keeps up. `OMP_NUM_THREADS` only affects the trainer processes (the workers' FFmpeg threads are independent); 4 is plenty.
- `--compile-blocks vision,llm,dit` (+ `--compile-mode`; **not** bit-identical). Runs `torch.compile` on the repeated transformer blocks (Qwen3-VL vision blocks, the LLM decoder layers, the action head's DiT blocks — `vlsa` for the VL self-attention blocks is also accepted) by wrapping each block's `forward`, so parameters, module names and checkpoints are untouched. It fuses the elementwise work around the GEMMs, which is ~45 % of GPU time in eager mode: forward+backward at 1024/GPU 3.1 s → 1.9 s. Fusion changes where intermediate roundings happen: against an fp32 reference the compiled model is as close as the eager bf16 model (cosine similarity 0.99992 for both), and with identical dropout masks the losses agree to six decimals. Training-only (`TrainingConfig.compile_blocks`; nothing is saved into the model, serving stays eager). Needs an Inductor/Triton that targets your GPU — the default torch 2.7.1 environment cannot compile for B300/B300 (`sm_103`), see [Blackwell GPUs (B300)](#blackwell-gpus-b300). On this hardware `vlsa` hits an Inductor shared-memory config gap (`No valid triton configs … out of resource`), hence its exclusion from the example.
- **Lossless re-encoded RGB video view** (`scripts/b1k/make_rgb_video_view.py`; *exact*). Video decoding is most of a worker's CPU, for two reasons. First, each shard samples *strided* steps (every ~10th frame of an episode), and with inter-frame coding (the source has a GOP of 8) the decoder must reconstruct every frame in the span to output the ones used — measured single-threaded on the stride-10 pattern, 10.4 ms CPU per used frame for the 720×720 camera and 5.6 ms for each 480×480 one, i.e. ~22 ms per sample, ~10 decoded frames per frame used. Second, the 720×720 stream is decoded at full resolution although the first thing the image pipeline does with every frame is deterministic — `LetterBoxPad` (a no-op on square frames) then `SmallestMaxSize(256, INTER_AREA)` — and only then come the random crop, second resize and colour jitter. The script applies exactly that head once, offline, and stores the 256×256 frames **losslessly in RGB** (`libx264rgb -qp 0`, 4:4:4) with a **GOP of 10 and CAVLC**, so a strided read decodes ~1 frame per frame used and each frame is cheap: 1.8 + 2×1.6 ≈ 5 ms CPU per sample in isolation, ~17 ms inside a fully loaded worker (vs ~70 for a long-GOP lossless file); a worker's total drops from ~80 to ~28 ms CPU per sample. It writes a *view* of the dataset — a new root whose `data/`, `meta/` (including the stats caches), depth streams and untouched chunks are symlinks and only the selected tasks' RGB files are re-encoded, keeping the `.mp4` path template, frame counts and fps (~40 GB per task). Because the stored frames are bit-identical to what the online head produced and the head is the identity on them, every downstream random stage sees identical inputs. Each file is verified bitwise on sampled frames against the source pipeline and recorded in `RGB_VIEW_MANIFEST.json`; a view is only exact for its `--shortest-edge` (256 for N1.7). Beware long-GOP lossless files (x264's default GOP 250 + CABAC): for strided reads they decode *slower* than the source (63 ms per sample measured). Serving is unaffected (live frames take the full pipeline).

  ```
  python scripts/b1k/make_rgb_video_view.py --source-root $DATA_ROOT/2026-challenge-demos \
      --view-root $DATA_ROOT/2026-challenge-demos-rgb256 --task-names $TASK \
      --modality-json examples/b1k/r1pro.json --shortest-edge 256 --gop 10 --jobs 4
  # then: --dataset-path $DATA_ROOT/2026-challenge-demos-rgb256  (same --task-names, same stats)
  ```
- **GPU-side work, measured on the compiled step** (1024 samples/GPU, fwd+bwd 1.97 s before these changes): the frozen backbone's *inference* is ~58 % of it (vision tower 35 %, LLM 16 %), the trainable action head the rest.
  - **Patch embedding as `F.linear`** (`fast_vl_patch_embed`, default on; same math). Qwen3-VL's patch embedding is a `Conv3d` whose kernel is the whole 2×16×16 patch — a linear layer in disguise — and cuDNN has no good kernel for it: 115 ms per step (a generic `sm80` implicit GEMM plus a layout transform) vs 1.3 ms as a GEMM. The linear result is *closer* to an fp32 reference than the convolution's.
  - **No LM head** (*exact*). Only the pre-norm output of the last kept decoder layer is used; `Qwen3Backbone` now captures it with a forward hook while running the base `Qwen3VLModel`, so the 151k-vocabulary GEMM is never issued — even with `logits_to_keep=1` cuBLAS spent ~90 ms per step on that shape.
  - `--backbone-attn-implementation gr00t_fast` (`gr00t/model/modules/fast_attention.py`). HF's `flash_attention_2` runs FlashAttention‑2 kernels that ignore Blackwell's tensor memory and force a `torch.compile` graph break in every block. `gr00t_fast` uses PyTorch SDPA (cuDNN / flash backends) for regular batches — `is_causal` for the decoder, and the packed image segments viewed as a regular `(n_images, heads, 256, d)` batch for the vision tower (HF otherwise loops over the segments) — and FlashAttention **varlen** for padded, multi-task batches: **FlashAttention‑4** (`pip install --prerelease=allow "flash-attn-4[cu13]"`, CuTe DSL, JIT; works on `sm_103`) when installed, else FA2. Numerics: cos 0.99994 to an fp32 reference for both FA2 and `gr00t_fast`; losses agree to 6 decimals on regular and padded batches. Step effect −4.5 %. In isolation FA4 is 1.1–1.4× faster than FA2 on these shapes but slower than cuDNN SDPA for the unpadded ones, which is why it only serves the padded path. `--sdpa-backend-priority cudnn,efficient,flash,math` reorders torch's SDPA backends (default puts cuDNN last); it made no measurable difference at the step level here.
  - `--compile-coordinate-descent` (Inductor `coordinate_descent_tuning`): −3.5 % step for ~1 min more compile per process (cached afterwards). `--compile-persistent-reductions False` is what lets the `vlsa` blocks compile on Blackwell (their layer-norm backward otherwise becomes a persistent-reduction kernel needing more shared memory than exists) — another −1 %.
  - **What did not pay off:** CUDA graphs (`--compile-mode reduce-overhead`) — kernel time is ~92 % of the step so the ceiling is small, cudagraph-trees trips over block outputs kept alive across replays (deepstack features), and at 1024/GPU the graph pools do not fit next to the 253 GB the step already uses. FP8 for the frozen backbone (~0.9 s of the step is its inference; Blackwell FP8 GEMMs are ~2× bf16) would be the next lever, but it changes the features the policy is trained on — a decision, not a free optimisation.
  - Net: fwd+bwd **1.97 → 1.64 s** at 1024/GPU (eval-mode losses identical to 6 decimals; train-mode differences are dropout RNG under Inductor).
- **`GR00T_FFMPEG_THREADS`** (env, dataloader workers; default FFmpeg auto = up to 16 threads per decoder). With 48 workers on a 130-core quota the auto setting ran ~770 decoder threads: 129 cores busy, of which ~60 were contention. `GR00T_FFMPEG_THREADS=4` → 69 cores busy for *more* throughput; the step went 2.33 → 1.93 s. Same decoded frames.
- `--wandb-project` (default `B1K`), `--use-ddp` (see [DeepSpeed on aarch64 hosts](#deepspeed-on-aarch64-hosts)).

**The command this guide's numbers come from** — 4× B300 (284 GB each), 130-core CPU budget, 900 GiB host RAM limit; global batch 4096 uses ~205 GB per GPU with DeepSpeed ZeRO-2. `$DATASET_PATH` is the GOP-10 video view built above, `$OUTPUT_DIR/$EXP_NAME/checkpoint-<step>/` is where checkpoints land:

```
export OMP_NUM_THREADS=4 GR00T_FFMPEG_THREADS=4 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 \
       WANDB_MODE=online WANDB_PROJECT=b1k-challenge-2026-gr00t WANDB_BASE_URL=https://api.wandb.ai
EXP_NAME=single-task-turning-on-radio-bs4096
torchrun --nproc_per_node=4 --master_port=29500 scripts/b1k/train_b1k.py \
    --experiment-name $EXP_NAME --wandb-project $WANDB_PROJECT \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path $DATASET_PATH --task-names turning_on_radio \
    --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/b1k/r1pro.py \
    --num-gpus 4 --global-batch-size 4096 \
    --output-dir $OUTPUT_DIR --save-steps 2500 --save-total-limit 3 --max-steps 150000 \
    --dataloader-num-workers 12 --dataloader-prefetch-factor 1 --decode-only-used-frames \
    --backbone-attn-implementation gr00t_fast \
    --compile-blocks vision,llm,dit,vlsa --compile-persistent-reductions False --compile-coordinate-descent \
    [--resume-from-checkpoint]
```

`--collate-pixel-values-dtype uint8`, `fast_vl_position_ids` and `fast_vl_patch_embed` are the defaults and need no flag. `--resume-from-checkpoint` continues from the newest `checkpoint-<step>/` in `$OUTPUT_DIR/$EXP_NAME` (the whole DeepSpeed state; the run above was paused and resumed three times this way). `WANDB_BASE_URL` matters only on hosts whose environment points W&B at another server.

With the stock code this configuration ran at 7.3 s/step (GPU idle most of the time on position ids, then CPU-bound on video decode); with everything above — the exact data changes, the re-encoded video view, `--compile-blocks`, `gr00t_fast` attention, the patch-embedding/LM-head fixes and 4 decoder threads per worker — it runs at **1.9 s/step** with the GPUs at 97–99 % utilisation: 80 h for 150k steps instead of 300 h. Watch `Wait for shard … in 0.00 seconds` in the log; non-zero waits mean the dataloader is the limit again (add workers if you have the cores and RAM).

#### Checkpoints on the Hub (two monitors)

`--save-steps 2500 --save-total-limit 3` keeps the three newest checkpoints locally (34 GB each: `model-*.safetensors` plus the DeepSpeed ZeRO-2 partitions in `global_step<step>/`, `latest`, `rng_state_*.pth`, `scheduler.pt`, `training_args.bin`). Two scripts under `scripts/b1k/`, run detached next to the training job (tmux), mirror them to one public repo with a folder per experiment:

```
<user>/<repo>/
  README.md                               # index of experiments, kept by the eval-only uploader
  <exp>/
    README.md                             # per-experiment index and usage
    checkpoint-10000/ checkpoint-20000/ …  # eval-only copies at the scheduled steps (kept forever)
    resume/
      README.md
      checkpoint-<step>/                  # the ONE latest full checkpoint, replaced every 2500 steps
```

- **Eval-only copies** — `hf_checkpoint_uploader.py`. At every scheduled step (default: every 10k up to 50k, then every 5k) it waits until `checkpoint-<step>/` is complete and quiet, hard-links the files needed to *serve* the policy into `--staging-dir` (weights, `config.json`, `processor_config.json`, `statistics.json`, `embodiment_id.json`, `experiment_cfg/`, `trainer_state.json`, `wandb_config.json` — no `global_step*/`, rng, scheduler, `training_args.bin`) and uploads that folder to `<exp>/checkpoint-<step>/`. ~7 GB each; they accumulate. It also writes the repo and experiment READMEs and `status.json` (schedule, pending/uploaded steps, last error) in the staging dir. Uploads that fail (e.g. a missing repo permission) are retried with a back-off and never block training.

  ```
  python scripts/b1k/hf_checkpoint_uploader.py --run-dir $OUTPUT_DIR/$EXP_NAME \
      --staging-dir $STAGING_DIR/$EXP_NAME --repo-id <user>/<repo> --path-prefix $EXP_NAME \
      --max-steps 150000 --task turning_on_radio --global-bs 4096 --num-gpus 4 \
      --experiment-name $EXP_NAME --wandb-project $WANDB_PROJECT
  ```

- **Latest full checkpoint, one at a time** — `hf_resume_checkpoint_uploader.py`. Whenever a newer checkpoint is complete for *resume* (`trainer_state.json` at that step, `latest` → `global_step<step>`, model state and optimizer shards and RNG states for all ranks, `scheduler.pt`, weights, directory unchanged for 2 min) it uploads the whole directory as-is to `<exp>/resume/checkpoint-<step>/` **and deletes the previous `resume/checkpoint-*` in the same commit**, then verifies every file and size. Deleting on the Hub does not free storage — the old LFS objects stay referenced by history and count against the quota — so it then calls `HfApi.permanently_delete_lfs_files(..., rewrite_history=True)` on the replaced checkpoint's objects, which removes them for good and rewrites the history so nothing points at them. Only objects that no file in the repo still references are deleted: the eval-only copy of the same step shares byte-identical `safetensors` with the full checkpoint and keeps them alive. Net effect: the repo always holds exactly one 35 GB resumable checkpoint (35.7 GB, ~50 s per upload at ~1 GB/s) plus the eval-only copies. `resume-status.json` in `--staging-dir` records the step, commit, bytes freed and the repo's LFS total; the replace-and-collect path was exercised on a scratch repo before going live.

  ```
  python scripts/b1k/hf_resume_checkpoint_uploader.py --run-dir $OUTPUT_DIR/$EXP_NAME \
      --staging-dir $STAGING_DIR/$EXP_NAME --repo-id <user>/<repo> --path-prefix $EXP_NAME \
      --num-gpus 4 --max-steps 150000
  ```

  Resume elsewhere from the Hub copy:

  ```
  hf download <user>/<repo> --include "$EXP_NAME/resume/checkpoint-<step>/*" --local-dir ckpt
  mkdir -p $OUTPUT_DIR/$EXP_NAME && mv ckpt/$EXP_NAME/resume/checkpoint-<step> $OUTPUT_DIR/$EXP_NAME/
  # then the training command above with --resume-from-checkpoint
  ```

Both scripts need `HF_TOKEN` in the environment (write access to the repo) and poll every 60 s; they exit on their own once the `--max-steps` checkpoint is on the Hub.

#### Language prompt

The challenge demos carry two kinds of text per task in `meta/tasks.jsonl`. The policy is conditioned on one of them:

| Prompt source          | Text fed to the model (before lower-casing / punctuation stripping)               |
| ---------------------- | --------------------------------------------------------------------------------- |
| `task_name` (default)  | `turning_on_radio` (the snake_case task id, what LeRobot's `tasks.parquet` holds) |
| `task_description`     | `Turn on the radio receiver that's on the table in the living room.`              |

Each kind is an annotation key in `examples/b1k/r1pro.json` (`annotation.human.task_name` / `annotation.human.task_description`, both resolved from `meta/tasks.jsonl`, which `deploy_modality.py` validates). The shared modality config `examples/b1k/r1pro.py` — passed to both `train_b1k.py` and `serve_b1k.py` — sets the default; `--prompt-source task_description|task_name` overrides it for one training run. Whichever wins is saved in the checkpoint (`checkpoint-<step>/processor_config.json`, `modality_configs.new_embodiment.language.modality_keys`), so `serve_b1k.py` automatically prompts with the same kind of text — see [Evaluation](#evaluation).

### Evaluation

After finetuning, you can run evaluation by following the steps below:

1. Deploy finetuned checkpoint:
  ```
    source .venv/bin/activate
    # source scripts/activate_b300.sh     # B300 only, see "Blackwell GPUs" above
    CUDA_VISIBLE_DEVICES=0 python scripts/b1k/serve_b1k.py \
        --model-path $PATH_TO_CKPT \
        --modality-config-path examples/b1k/r1pro.py \
        --embodiment-tag NEW_EMBODIMENT \
        --host 127.0.0.1 --port 8000
  ```
    This opens a connection listening on 127.0.0.1:8000. Health-check it with `curl -s http://127.0.0.1:8000/healthz` (returns `OK`).

    The server prompts the policy with the same kind of text it was trained on (read from the checkpoint's language key) and resolves the task text per request from the `task_id` the evaluator sends, using the task table in `examples/b1k/tasks.jsonl` (a copy of the dataset's `meta/tasks.jsonl`). Overrides: `--task-name turning_on_radio` fixes the prompt to one task, `--prompt-source task_name|task_description` forces the kind of text, `--text-prompt "..."` sets it verbatim. Checkpoints trained before `--prompt-source` existed saw task names under the `task_description` key; serve them with `--prompt-source task_name`.
2. Run the evaluation on BEHAVIOR:
  Assume you have behavior env installed (check [https://github.com/StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) for more details), run the following command within the BEHAVIOR-1K directory:

