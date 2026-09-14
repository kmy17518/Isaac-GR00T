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

