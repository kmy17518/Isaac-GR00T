# Starting B1K training & eval (GR00T N1.7) — from scratch

End-to-end runbook to set up, train, and evaluate a GR00T policy on a BEHAVIOR-1K (b1k)
task (reference: `turning_on_radio`) **assuming a fresh machine** — nothing built or
downloaded yet (no venv, no conda env, no cached models, raw v3.0 demo data).

Two separate environments are used:

- **GR00T env** (training + policy serving): a `uv` venv at `Isaac-GR00T/.venv` (Python 3.10).
- **OmniGibson env** (simulation eval): a conda env (here `behavior_my`).

> **Critical:** this checkout is **GR00T N1.7** (Qwen3 / Cosmos-Reason2-2B backbone).
> Use **`nvidia/GR00T-N1.7-3B`** as the base model. The N1.6 (Eagle) checkpoint that the
> original b1k scripts referenced is a *different architecture* and will not load.

---

## 0. Key paths (adjust to your machine)

| What | Path (this machine) |
|---|---|
| GR00T repo | `/home/ubuntu/minyeong/Isaac-GR00T` |
| BEHAVIOR-1K repo | `/home/ubuntu/minyeong/BEHAVIOR-1K` |
| Secrets file you create | `/home/ubuntu/minyeong/.env` |
| b1k data root | `/home/ubuntu/minyeong/BEHAVIOR-1K/datasets/2026-challenge-demos/b1k` |
| Eval task instances (from setup.sh) | `…/datasets/2025-challenge-task-instances` |

---

## 1. Secrets & Hugging Face access (do first)

1. Create `/home/ubuntu/minyeong/.env`:
   ```
   HF_TOKEN=hf_xxx
   WANDB_API_KEY=xxx          # optional; training enables W&B
   ```
2. The N1.7 backbone **`nvidia/Cosmos-Reason2-2B` is gated** — log into HF as the account that
   owns `HF_TOKEN` and click **"Agree and access repository"** at
   `https://huggingface.co/nvidia/Cosmos-Reason2-2B`. Without this, model load 403s.
3. Always export the token before training/serving:
   ```bash
   set -a && source /home/ubuntu/minyeong/.env && set +a
   ```

---

## 2. Code

```bash
cd /home/ubuntu/minyeong/Isaac-GR00T
git checkout lerobot-v2.1        # branch carrying the b1k modality config + train_b1k fixes
```
(If starting from a clean clone, clone `Isaac-GR00T` and `BEHAVIOR-1K` first.)

---

## 3. GR00T env (training + serving)

Requires `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
cd /home/ubuntu/minyeong/Isaac-GR00T
uv sync --frozen --python 3.10
# ^ MUST use --frozen: a normal sync tries to read the aarch64 git-LFS wheel pointers
#   (flash-attn/torchcodec) and fails on x86_64.
uv pip install --python .venv/bin/python websockets
# ^ serve_b1k needs `websockets`; it is not yet declared in pyproject.
```

Quick check: `.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`.

---

## 4. OmniGibson env + assets (eval)

```bash
cd /home/ubuntu/minyeong/BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --dataset --eval --joylo
```
This creates the conda env (e.g. `behavior_my`), installs OmniGibson + Isaac Sim + BDDL + JoyLo +
eval support, and downloads the BEHAVIOR assets + `2025-challenge-task-instances` (which contains
`turning_on_radio`). Note the env's python path, e.g.
`…/miniconda3/envs/behavior_my/bin/python` — used in §8.

Sanity check (also confirms EGL/headless + asset path):
```bash
<behavior_py> -c "from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_INDICES as T; print('turning_on_radio' in T)"
```

---

## 5. Base models (pre-cache; optional but recommended)

Training/serving auto-download with `HF_TOKEN` set, but you can pre-cache:
```bash
cd /home/ubuntu/minyeong/Isaac-GR00T
set -a && source /home/ubuntu/minyeong/.env && set +a
.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
snapshot_download("nvidia/GR00T-N1.7-3B", token=tok)
snapshot_download("nvidia/Cosmos-Reason2-2B", token=tok)  # gated backbone
PY
```

---

## 6. Data prep (per task) — convert v3.0 → v2.1, then add modality.json

The b1k LeRobot **demo** datasets are released in **LeRobot v3.0**, but the GR00T loader needs
**v2.1** plus a GR00T-specific `meta/modality.json`. (These demos are separate from the eval
task-instances in §4 — obtain them from the challenge demo release and place under the b1k data root,
or pass an HF `--repo-id` to the converter, which will download it.)

**6a. One-time: build the conversion venv** (its own subproject env):
```bash
cd /home/ubuntu/minyeong/Isaac-GR00T/scripts/lerobot_conversion
uv venv --python 3.11 .venv
source .venv/bin/activate
# install explicit deps (an editable `-e .` install trips on the LFS-pointer wheels):
uv --no-config pip install \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475" \
  huggingface_hub jsonlines numpy pyarrow tqdm
```

**6b. Convert each task** (in place; original is backed up to `<task>_v3.0`):
```bash
cd /home/ubuntu/minyeong/Isaac-GR00T/scripts/lerobot_conversion
source .venv/bin/activate
python convert_v3_to_v2.py \
  --root /home/ubuntu/minyeong/BEHAVIOR-1K/datasets/2026-challenge-demos/b1k \
  --repo-id turning_on_radio
```

**6c. Deploy `modality.json`** into each task's `meta/` (run after conversion):
```bash
cd /home/ubuntu/minyeong/Isaac-GR00T
python scripts/b1k/deploy_modality.py /home/ubuntu/minyeong/BEHAVIOR-1K/datasets/2026-challenge-demos/b1k
```
This validates each dataset is the R1Pro format (`observation.state`=61, `action`=23, the 3 RGB cams)
and copies `examples/b1k/r1pro.json` → `<task>/meta/modality.json`.

After this, `<b1k_root>/turning_on_radio` is v2.1 with `meta/modality.json` present.

---

## 7. Training

```bash
cd /home/ubuntu/minyeong/Isaac-GR00T
set -a && source /home/ubuntu/minyeong/.env && set +a
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=online \
.venv/bin/torchrun --nproc_per_node=2 --master_port=29500 scripts/b1k/train_b1k.py \
  --experiment-name b1k-turning_on_radio \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /home/ubuntu/minyeong/BEHAVIOR-1K/datasets/2026-challenge-demos/b1k/turning_on_radio \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/b1k/r1pro.py \
  --num-gpus 2 \
  --global-batch-size 256 \
  --output-dir outputs/b1k-turning_on_radio \
  --save-steps 1500 --save-total-limit 5 --max-steps 150000 \
  --dataloader-num-workers 4
```

- `--global-batch-size` is **global** (per-GPU = global ÷ num_gpus). **128/GPU** (global 256 on 2 GPUs)
  is the H100 throughput sweet spot; ≤256/GPU fits 80 GB (~61 GB), **512/GPU OOMs**.
- Tunes projector + diffusion head only (LLM/visual frozen), `use_relative_action=True`.
- Use `WANDB_MODE=offline` to avoid network. **Resume:** add `--resume-from-checkpoint`.
- Checkpoints: `outputs/<exp>/<exp>/checkpoint-<step>/`.

---

## 8. Eval (two processes, two envs, two GPUs)

**Terminal 1 — GR00T policy server** (GR00T venv):
```bash
cd /home/ubuntu/minyeong/Isaac-GR00T
set -a && source /home/ubuntu/minyeong/.env && set +a
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/b1k/serve_b1k.py \
  --model-path outputs/b1k-turning_on_radio/b1k-turning_on_radio/checkpoint-XXXX \
  --modality-config-path examples/b1k/r1pro.py \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 --port 8000
```

**Terminal 2 — OmniGibson websocket eval** (conda env from §4):
```bash
cd /home/ubuntu/minyeong/BEHAVIOR-1K/OmniGibson
CUDA_VISIBLE_DEVICES=1 <behavior_py> -m omnigibson.eval.run_eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 1 2 3 4 5 6 7 8 9 \
  --output-dir /tmp/b1k_eval/standard.public.<team>.<affiliation>.<date>
```
- Omit `--max-steps` for the challenge default (2× mean human-demo length); pass `--max-steps 50`
  for a fast pipeline smoke.
- Writes per-rollout `q_score`/`time`/`agent_distance` JSONs under `<output-dir>/json/`.

**Score:**
```bash
<behavior_py> OmniGibson/omnigibson/eval/utils/score_utils.py -i <parent_of_submission_dir> -o <out_dir>
```

---

## 9. Gotchas / troubleshooting

- **`KeyError: 'Gr00tN1d6'` / Eagle errors** → you pointed at the N1.6 base; use `nvidia/GR00T-N1.7-3B`.
- **403 on `nvidia/Cosmos-Reason2-2B`** → accept the HF gate for that account **and** `source .env` so `HF_TOKEN` is in the process env.
- **`No module named 'websockets'`** → `uv pip install --python .venv/bin/python websockets` (§3).
- **`uv sync` fails on a wheel / TOML** → use `uv sync --frozen`; the duplicate `[tool.uv.sources]` was already fixed, and aarch64 LFS wheels are irrelevant on x86_64.
- **Loader can't read the dataset** → it must be v2.1 with `meta/modality.json`; do §6 (convert + deploy).
- **No eval runner** → use `omnigibson/eval/run_eval.py` (this repo's driver); upstream `omnigibson/learning/eval.py` is absent here.
- **GPUs full** → shared box; check `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv` for other users' jobs before launching.
