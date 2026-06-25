# Training, checkpoint preservation & resuming — GR00T N1.7 on b1k (LeRobot v3.0)

How to finetune GR00T N1.7 on a BEHAVIOR-1K (b1k) task (`turning_on_radio`, read natively from
**LeRobot v3.0**), preserve servable checkpoints before they rotate away, and resume to train
longer. For serving + evaluating the resulting checkpoints, see `docs/EVAL.md`; for first-time
machine setup, see `docs/START.md`.

> **All paths below are relative to the repo root** — run every command from the repo root (the
> directory that contains `scripts/`, `examples/`, `.venv/`). Things that live next to the repo
> are referenced as siblings: `../.env`, `../BEHAVIOR-1K/…`, `../served_checkpoints`.

---

## 0. Prerequisites

- GR00T `.venv` built (`uv sync --frozen --python 3.10`) — see `docs/START.md` §3.
- `../.env` containing `HF_TOKEN` (the gated `nvidia/Cosmos-Reason2-2B` backbone is resolved at
startup) and `WANDB_API_KEY`.
- **Blackwell GPUs (B200/B300) only:** upgrade the JIT nvrtc or training crashes in the Qwen3-VL
backbone — see `docs/START.md` §3a
(`uv pip install --python .venv/bin/python nvidia-cuda-nvrtc-cu12==12.9.86`).
- Dataset present at `../BEHAVIOR-1K/datasets/2026-challenge-demos/b1k/turning_on_radio_v3.0`
with `meta/modality.json` (see `docs/START.md` §6).

---

## 1. Train (finetune from the base model)

Run under `tmux` so it survives disconnects. From the repo root:

```bash
tmux new -s train
# inside the session, from the repo root:
set -a && source ../.env && set +a            # HF_TOKEN + WANDB_API_KEY into the env
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=4,5,6,7            # the GPUs to train on
export OMP_NUM_THREADS=4                       # keep below #workers (avoid CPU oversubscription)
mkdir -p logs

.venv/bin/torchrun --nproc_per_node=4 --master_port=29533 scripts/b1k/train_b1k.py \
  --experiment-name b1k-turning_on_radio-v3 \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ../BEHAVIOR-1K/datasets/2026-challenge-demos/b1k/turning_on_radio_v3.0 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/b1k/r1pro.py \
  --num-gpus 8 \
  --global-batch-size 2048 \
  --output-dir outputs/b1k-turning_on_radio_v3.0 \
  --save-steps 1500 --save-total-limit 5 --max-steps 150000 \
  --dataloader-num-workers 8 --decode-only-used-frames 2>&1 | tee logs/train.log
```


| arg                                   | meaning                                                                                                                                                                                                         |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--nproc_per_node` / `--num-gpus`     | number of GPUs — must match the count in `CUDA_VISIBLE_DEVICES`                                                                                                                                                 |
| `--global-batch-size`                 | **global** batch; per-GPU = global ÷ num_gpus (512 ÷ 4 = **128/GPU**)                                                                                                                                           |
| `--dataloader-num-workers`            | dataloader (video-decode) workers **per rank**. `8` is the sweet spot — the HEVC pipeline is decode-throughput-bound, so too few starves the GPUs and too many exhausts RAM (see §6)                            |
| `--decode-only-used-frames`           | decode only the video frames each shard actually uses instead of whole episodes — **identical training data**, ~1.4× faster, ~10× less decode/CPU/RAM. Fastest known setting; enabled above by default (see §6) |
| `--save-steps` / `--save-total-limit` | checkpoint every N steps, keep only the most recent M (older ones are **deleted** — see §2)                                                                                                                     |
| `--max-steps`                         | total training steps                                                                                                                                                                                            |
| `--base-model-path`                   | finetune base; LLM + vision backbone stay frozen, projector + diffusion action head are tuned                                                                                                                   |
| `--output-dir` + `--experiment-name`  | checkpoints land in `outputs/<output-dir-name>/<experiment-name>/checkpoint-<step>/`                                                                                                                            |


- **W&B:** logging is always on; project is `B1K`, run name = `--experiment-name`. Metrics:
`train/loss` (diffusion / flow-matching loss), `train/grad_norm`, `train/learning_rate`
(cosine schedule with warmup).
- `--master_port` only needs to be unique if another distributed job is running on the box.
- Keep `CUDA_VISIBLE_DEVICES` **disjoint** from any GPU used for serving (see `docs/EVAL.md`).
- **The values above are the fastest validated settings** (FA2 backbone via the default `.venv`,
`--decode-only-used-frames`, `--dataloader-num-workers 8`, `OMP_NUM_THREADS=4`, and the default
`episode_sampling_rate=0.1`). Training is **data-bound on HEVC video decode**, not GPU compute —
see §6 for the full set of optimizations we tried and what happened.

---

## 2. Preserve checkpoints (monitor + copy)

`--save-total-limit 5` keeps only the **last 5** checkpoints — older ones are **deleted** as new
ones are written. To keep checkpoints for later evaluation, copy them out before they rotate away,
copying only the **servable** files (~6.5 GB: config, weights, processor, stats) and skipping the
large DeepSpeed **optimizer state** (`global_step`*, ~27 GB) and other training-only files.

Save this as `ckpt_watcher.sh` and run it **from the repo root** under `tmux`:

```bash
#!/usr/bin/env bash
# Preserve every-INTERVAL-step checkpoint (servable files only) before rotation deletes it.
# Run from the repo root.
set -uo pipefail

EXP=b1k-turning_on_radio-v3
SRC_DIR=outputs/b1k-turning_on_radio_v3.0/$EXP   # where training writes checkpoint-<step>/
DEST_DIR=../served_checkpoints                    # where to preserve servable copies
INTERVAL_STEPS=4500       # keep one checkpoint every N steps ...
START_STEP=4500           # ... starting from this step
POLL_SECONDS=60
STABLE_SECONDS=90         # require model shards unmodified this long (avoid copying mid-save)

REQUIRED=(config.json model.safetensors.index.json processor_config.json statistics.json embodiment_id.json)
log(){ echo "[$(date -Is)] $*"; }
is_target(){ local n=$1; (( n >= START_STEP )) && (( n % INTERVAL_STEPS == 0 )); }

is_complete(){   # servable files present and model shards stable
  local d=$1 f newest now age
  for f in "${REQUIRED[@]}"; do [[ -f "$d/$f" ]] || return 1; done
  compgen -G "$d/model-*.safetensors" >/dev/null || return 1
  newest=$(find "$d" -maxdepth 1 -type f -name 'model-*.safetensors' -printf '%T@\n' | sort -nr | head -1)
  [[ -n "$newest" ]] || return 1
  now=$(date +%s); age=$(awk -v a="$newest" -v b="$now" 'BEGIN{print int(b-a)}')
  (( age >= STABLE_SECONDS ))
}

copy_ckpt(){     # copy to a .partial dir, then atomically rename (consumers never see a half-copy)
  local src=$1 name=$2 dest="$DEST_DIR/$name" tmp="$DEST_DIR/.$name.partial"
  rm -rf "$tmp"
  rsync -a \
    --exclude='global_step*' --exclude='rng_state_*.pth' --exclude='optimizer.pt' \
    --exclude='scheduler.pt' --exclude='trainer_state.json' --exclude='training_args.bin' \
    --exclude='latest' --exclude='zero_to_fp32.py' \
    "$src/" "$tmp/" && mv "$tmp" "$dest" && log "COPIED $name -> $dest"
}

mkdir -p "$DEST_DIR"
log "watcher: $SRC_DIR -> $DEST_DIR; every ${INTERVAL_STEPS} steps from ${START_STEP}"
while true; do
  shopt -s nullglob
  for d in "$SRC_DIR"/checkpoint-*; do
    [[ -d "$d" ]] || continue
    base=$(basename "$d"); step=${base#checkpoint-}
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    is_target "$step" || continue
    name="${EXP}-checkpoint-${step}"
    [[ -e "$DEST_DIR/$name" ]] && continue      # already preserved
    is_complete "$d" && copy_ckpt "$d" "$name"
  done
  sleep "$POLL_SECONDS"
done
```

```bash
mkdir -p logs
tmux new -d -s ckpt_watcher 'bash ckpt_watcher.sh 2>&1 | tee logs/ckpt_watcher.log'
```

- The preserved `b1k-turning_on_radio-v3-checkpoint-<step>/` dirs are directly servable
(`docs/EVAL.md` §2).
- To keep **every** saved checkpoint, set `INTERVAL_STEPS` equal to `--save-steps`. Raise it to
keep fewer (e.g. `4500` keeps one in three when `--save-steps 1500`).
- The copy is safe to run while training writes/rotates: it only reads, skips `global_step`*, and
publishes each copy via an atomic rename.

---

## 3. Resume / train longer (from a checkpoint)

To continue an existing run for more steps, relaunch with the **same** `--output-dir` **and**
`--experiment-name` (so the effective dir `outputs/b1k-turning_on_radio_v3.0/b1k-turning_on_radio-v3`
still holds `checkpoint-<latest>`) plus `--resume-from-checkpoint`. This is a **true resume**:
model weights, optimizer, LR scheduler, RNG, and the step counter all continue from the latest
checkpoint — its `global_step`* optimizer state must still be present in the output dir.

Example — resume from step 150000 for **+100k** steps, saving every 2500:

```bash
tmux new -s train_resume
set -a && source ../.env && set +a
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=4
mkdir -p logs

.venv/bin/torchrun --nproc_per_node=4 --master_port=29534 scripts/b1k/train_b1k.py \
  --experiment-name b1k-turning_on_radio-v3 \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ../BEHAVIOR-1K/datasets/2026-challenge-demos/b1k/turning_on_radio_v3.0 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/b1k/r1pro.py \
  --num-gpus 4 \
  --global-batch-size 512 \
  --output-dir outputs/b1k-turning_on_radio_v3.0 \
  --save-steps 2500 --save-total-limit 5 --max-steps 250000 \
  --dataloader-num-workers 8 --decode-only-used-frames \
  --resume-from-checkpoint 2>&1 | tee logs/train_resume.log
```

- `--max-steps` is the **total** (original + extra): 150000 done + 100000 more → `250000`.
- `--resume-from-checkpoint` resumes from the **latest** `checkpoint-*` in the effective output
dir. Keep `--experiment-name` identical or the resume won't find it.
- W&B starts a **new run** with the same name (two runs share the name — tell them apart by run id
/ created time). For a distinct W&B name without breaking the resume path, set `WANDB_NAME=<name>`
in the env and leave `--experiment-name` unchanged.

**⚠ LR / grad_norm at the resume boundary.** The LR is a **cosine over `--max-steps`**, rebuilt for
the new horizon. Extending `max_steps` "stretches" the cosine, so at the resume step the LR jumps
from the original run's end-of-schedule value (≈ 0) back up to a mid-curve value, and `grad_norm`
rises as the model starts moving again. This is expected — effectively a warm restart that lets the
model keep learning (continuing the *original* 150k schedule would hold LR ≈ 0 and learn nothing).
`train/loss` stays continuous across the boundary, which confirms the weights + optimizer resumed
correctly. To shape the LR: lower the peak with `--learning-rate`, or choose `--max-steps` to
control where the resume step lands on the curve.

**Preserve the resume run's checkpoints into a *separate* dir** so an in-progress eval sweep over
the original run (which serves the newest un-evaluated checkpoint from `../served_checkpoints`, see
`docs/EVAL.md` §6) isn't hijacked by freshly-resumed checkpoints. Reuse the §2 `ckpt_watcher.sh`
with:

```bash
# edit these in ckpt_watcher.sh for the resume run:
DEST_DIR=../served_checkpoints_resume
INTERVAL_STEPS=2500       # every saved checkpoint
START_STEP=150000         # from the resume point onward
```

```bash
tmux new -d -s ckpt_watcher_resume 'bash ckpt_watcher.sh 2>&1 | tee logs/ckpt_watcher_resume.log'
```

---

## 4. Monitoring a run

```bash
tail -f logs/train.log                                          # or logs/train_resume.log
tail -f logs/ckpt_watcher.log                                   # checkpoint copies
nvidia-smi                                                      # GPU memory / utilization
ls outputs/b1k-turning_on_radio_v3.0/b1k-turning_on_radio-v3/   # current checkpoints (last M)
ls ../served_checkpoints/                                       # preserved servable checkpoints
```

- **W&B:** project `B1K`, run name = `--experiment-name`. Watch `train/loss` (down),
`train/grad_norm`, `train/learning_rate`.
- **Stop a run:** `tmux kill-session -t train` (checkpoints persist; resume later via §3).

---

## 5. Gotchas

- **Checkpoint deleted before you copied it** → that's `--save-total-limit` rotation; keep the §2
watcher running for the whole job, and poll often enough relative to `--save-steps`.
- **Resume can't find the checkpoint** → `--output-dir` and/or `--experiment-name` differ from the
original, or the latest `checkpoint-*` lost its `global_step*` optimizer state (a servable-only
copy from §2 is **not** resumable — it intentionally omits the optimizer state).
- **LR jumps up on resume** → expected cosine re-stretch (see §3); not a broken resume.
- **Two GPUs jobs collide** → training and serving must use disjoint `CUDA_VISIBLE_DEVICES`.
- **Blackwell crash in Qwen3-VL `rot_pos_emb`** → nvrtc too old; see `docs/START.md` §3a.

---

## 6. Performance: what we tried (and what failed)

**TL;DR.** At the default batch this training is **data-bound on HEVC video decode**, *not*
GPU-compute-bound (GPUs sit idle waiting for frames). The only change that helped was
`--decode-only-used-frames`; it's the default above. Reference numbers below are 8×H100,
`--global-batch-size 2048`, `episode_sampling_rate=0.1`, FA2:


| setting                                   | s/it     | note                                               |
| ----------------------------------------- | -------- | -------------------------------------------------- |
| compute ceiling (no data loading)         | ~2.35    | measured with a "replay one shard" mock; the floor |
| `**--decode-only-used-frames` (default)** | **~2.8** | ~1.4× faster, identical data                       |
| decode-all (old default)                  | ~3.5–4.0 | jittery; GPUs ~30–50% utilized (starved)           |
| FlashAttention-3                          | ~4.7–5.5 | *slower* — see below                               |


### What worked

- **Decode only the frames each shard uses (`--decode-only-used-frames`).** The loader used to
decode **whole episodes**, but with `episode_sampling_rate=0.1` each shard only reads ~~10% of
those frames — ~90% of the (expensive HEVC) decode was wasted. Decoding only the used frames is
**~~1.4× faster**, uses ~10× less decode/CPU/RAM, and produces **byte-identical** training data
(verified across 4,000+ datapoints incl. boundaries and padding). Keeps `episode_sampling_rate=0.1`.

### What failed or didn't help

- **FlashAttention-3 (Hopper).** Built FA3 from source and wired it into the Qwen3-VL backbone. It
is **~2× slower** for this model: the **vision ViT** runs ~3× slower under FA3's var-len kernels
(text attention is on par). The backbone is frozen (forward-only), so there's nothing to recover.
→ **Stay on FA2** (the default `.venv`). A separate `.venv-fa3` exists but is **not** recommended
for training.
- **GPU/NVDEC video decode.** Made it work end-to-end (installed the driver's `libnvcuvid`, switched
torchcodec to the `+cu128` CUDA wheel; ~2.5× raw decode). But at the production batch the **model
already uses ~75/80 GB**, so decoding on the same GPU **OOMs** (whole-episode decode, and even the
per-worker CUDA contexts, don't fit). Only viable at a much smaller batch or with GPUs dedicated
to decode. → not used. (Moving the CPU transforms to GPU to avoid the copy would only add *more*
GPU-memory pressure — same blocker.)
- **Fewer dataloader workers (4).** Counterintuitively **slower** and jittery: the pipeline is
decode-*throughput*-bound, so too few decoders starve the GPUs. **16 workers** is worse — whole
episodes × 128 workers exhaust RAM/CPU (run-queue >200, `%sys` ~97%, page cache evicted, no
progress). **8 is the sweet spot.**
- `**OMP_NUM_THREADS=8`.** ~5% slower than `4` (OMP threads oversubscribe the CPU against the decode
workers). → default `4`.
- `**episode_sampling_rate=1.0`.** Actually the *fastest* (~2.37 s/it, at the compute ceiling)
because whole episodes then decode **sequentially** with zero waste — but each shard becomes a
single episode, which **lowers per-batch episode diversity**. We kept `0.1` for diversity;
`--decode-only-used-frames` recovers most of the speed without that trade-off.
- **Re-encoding the dataset (HEVC→H.264 / downscale).** Would cut decode cost substantially (HEVC is
the expensive part and the model only needs 256²), but we chose **not** to alter the dataset
encoding.

