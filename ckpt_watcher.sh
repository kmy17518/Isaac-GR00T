#!/usr/bin/env bash
# Preserve every-INTERVAL-step checkpoint (servable files only) before rotation deletes it.
# v3.0 run — preserves into a DEDICATED dir so it never collides with the v2.1 run.
set -uo pipefail

EXP=turning_on_radio_v3.0
SRC_DIR=/home/ubuntu/minyeong/Isaac-GR00T-lerobot-v3.0/outputs/b1k-turning_on_radio_v3.0/$EXP
DEST_DIR=/home/ubuntu/minyeong/served_checkpoints_v3.0   # distinct from the v2.1 run's dir
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
