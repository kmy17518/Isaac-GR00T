# B1K × GR00T — progress summary

**Goal:** make it possible to **train & eval** a GR00T policy on BEHAVIOR-1K (b1k)
LeRobot demo datasets (reference task: `turning_on_radio`), via two experiments:

- **Experiment 1** — keep GR00T on **LeRobot v2.1**; convert the v3.0 demo dataset → v2.1; train + eval.
- **Experiment 2** — upgrade GR00T to read **LeRobot v3.0** natively (no conversion); train + eval.

See `START.md` for how to run things. This file is the status / handoff.

---

## Shared jobs (needed by both experiments)

| Job | Status | Notes |
|---|---|---|
| **A. Modality contract** | ✅ done | `examples/b1k/r1pro.py` (+`.json`) maps R1Pro 61-dim `observation.state` and 23-dim `action` to GR00T modality groups; `scripts/b1k/deploy_modality.py` deploys+validates per task. Deployed to `turning_on_radio`. |
| **B. Train/eval consistency** | ✅ done | Proprio order confirmed authoritative (`replay_obs.py` and eval both use `PROPRIOCEPTION_INDICES["R1Pro"]`); action 23-dim (`ACTION_QPOS_INDICES`); cameras (zed→head, realsense→wrists); language via `task_index`. Loader-validated on the converted dataset. |
| **C. Envs + base model** | ✅ done | GR00T `.venv` built (torch cu128, 8×H100, flash-attn); `nvidia/GR00T-N1.7-3B` + gated `nvidia/Cosmos-Reason2-2B` cached; OmniGibson `behavior_my` imports OK (`gm.DATA_PATH=…/datasets`). |
| **D. Eval assets** | ✅ done | `2025-challenge-task-instances` present; `turning_on_radio` = task **0** with 20 public test instances. |
| **E. 2025/2026 wiring** | ✅ no-op | `turning_on_radio` genuinely lives under `2025-challenge-task-instances`, so `adapter.py`'s hardcoded 2025 paths are correct. Only the *LeRobot demos* are under `2026-challenge-demos`. |
| **F. Eval harness** | ⚠️ partial | Serving code done + validated; eval **driver written** (`omnigibson/eval/run_eval.py`); the **actual eval rollout has not been run** (GPU-blocked — see below). |

---

## Experiment 1 (lerobot-v2.1) — mostly done, not yet a real run

| Step | Status |
|---|---|
| v3.0 → v2.1 conversion of `turning_on_radio` | ✅ done (orig backed up as `turning_on_radio_v3.0`) |
| `meta/modality.json` deployed | ✅ done |
| Loader smoke (200 eps, correct state/action dims) | ✅ done |
| 2-GPU training smoke (N1.7) | ✅ done (`loss≈1.12`, checkpoints saved) |
| Resume-from-checkpoint | ✅ verified (resumed at step 15) |
| Max per-GPU batch / VRAM | ✅ 256/GPU fits (~61 GB), 512/GPU OOM (~79 GB) |
| Forward/step-time vs batch | ✅ measured (8→0.26s, 32→0.45s, 128→1.21s, 256→2.52s; throughput plateaus ~210 samp/s ≥128/GPU) |
| **Full training run** | ❌ not started (only smokes) |
| **Eval rollout (Shared F)** | ❌ not run (GPU-blocked) |

---

## Experiment 2 (lerobot-v3.0) — native reader done & validated equal to Exp 1

Worktree `Isaac-GR00T-lerobot-v3.0` on branch `lerobot-v3.0` (at `712fd3a`, == `lerobot-v2.1`).
The GR00T loader now reads **LeRobot v3.0 natively, no conversion**.

| Step | Status |
|---|---|
| Version-aware `LeRobotEpisodeLoader` (v2.x **and** v3.0) | ✅ done |
| v3.0 metadata: `meta/episodes/*.parquet`, `meta/tasks.parquet` | ✅ done |
| v3.0 data: multi-episode parquet sliced by `episode_index` | ✅ done |
| v3.0 video: concatenated per-key mp4, frame offset `round(from_timestamp*fps)` | ✅ done |
| `modality.json` deployed into `turning_on_radio_v3.0` | ✅ done |
| **Loader parity vs v2.1** (state/action/lang/stats exact; video MAE=0) | ✅ 40/40 exact |
| **Stats paths on v3.0** (`generate_stats`, `generate_rel_stats`) == v2.1 | ✅ exact |
| **Cross-check vs real `lerobot` reader** (state/action exact, video MAE=0) | ✅ 15/15 exact |
| CPU pytest (`tests/gr00t/data/test_lerobot_v3_parity.py`) | ✅ 18 passed |
| No regression on legacy v2.x tests | ✅ 100 passed / 12 skip |
| **GPU training (real weights, 4×B300, 10 steps)** | ✅ v3.0 `train_loss=1.1146` == v2.1 `1.1145` (≈Exp 1's 1.12) |

**Design decision — no `lerobot` dependency in the training env.** Adding `lerobot` (0.4.x, the
v3.0-capable lib) hard-conflicts with GR00T's pins (`datasets==3.6.0` vs `>=4.0.0`, `av==16.1.0`
vs `<16.0.0`, `wandb==0.23.0` vs `<0.22.0`) and drags in teleop/viz deps. The native reader is
self-contained (`pyarrow`/`pandas` only). `lerobot` stays isolated in `scripts/lerobot_conversion`
and is used only as an **optional parity oracle** (`scripts/b1k/validate_v3_lerobot_parity.py`).

**Why "equal to Exp 1" is provable:** Exp 1's `turning_on_radio` (v2.1) was converted **from**
`turning_on_radio_v3.0`; both are on disk, so the v3.0 reader is diffed directly against the v2.1
reader on identical underlying trajectories (and independently against `lerobot`).

### New / changed files (uncommitted)
- `gr00t/data/dataset/lerobot_episode_loader.py` — version-aware v3.0 reading (the only core change).
- `scripts/b1k/validate_v3_parity.py` — v3.0-vs-v2.1 parity harness (state/action/lang/stats/video).
- `scripts/b1k/validate_v3_lerobot_parity.py` — cross-venv native-vs-`lerobot` parity check.
- `tests/gr00t/data/test_lerobot_v3_parity.py` — synthetic (CI-safe) + real-data parity tests.
- `tests/scripts/test_v3_lerobot_parity.py` — gated wrapper for the `lerobot` cross-check.

### GPU training: nvrtc B300 fix (RESOLVED) + end-to-end loss parity
The box is **8× NVIDIA B300 (sm_103)**. The originally-pinned `torch==2.7.1+cu128` ships an nvrtc
that rejects sm_103 → `nvrtc: error: invalid value for --gpu-architecture` inside Qwen3-VL
`rot_pos_emb` (`torch.prod(grid_thw)`), failing **both** v3.0 and v2.1 identically at the model
forward (so it was always infra, never the LeRobot format).

**Fix (already applied in the shared `.venv`):** upgrade the nvrtc wheel that torch JIT-loads to a
B300-aware build —
```bash
uv pip install --python /home/stuart/ThunderPuppies/Isaac-GR00T/.venv/bin/python nvidia-cuda-nvrtc-cu12==12.9.86
```
(torch 2.7.1 stays; it loads `nvidia/cuda_nvrtc/lib/libnvrtc.so.12` from this wheel, now 12.9 which
knows sm_103). NB: a fresh `uv sync` reverts to the 12.8 nvrtc, so re-apply this after any resync,
or pin it. The v3.0 worktree reuses this `.venv` (via `PYTHONPATH`), so the fix is active here too.

**Result (real weights, 4×B300 (GPUs 4–7), global-batch 64, 10 steps, identical config):**

| Run | nvrtc error | `train_loss` | grad_norm |
|---|---|---|---|
| v3.0 (native) | none | **1.1146484** | 0.225364 |
| v2.1 (Exp 1)  | none | **1.1144531** | 0.226072 |

Δloss ≈ 2e-4 (GPU reduction nondeterminism; data is provably identical), both ≈ Exp 1's `loss≈1.12`.
This closes the end-to-end equivalence: **native v3.0 training == v2.1 (Experiment 1) training.**
Logs: `/tmp/b1k_v3_real.log`, `/tmp/b1k_v21_real.log`.

---

## Bugs found & fixed (committed on `lerobot-v2.1`, author/committer `kmy17518`, **not pushed**)

| Commit | What |
|---|---|
| `caa33a7` | add b1k r1pro modality config |
| `f38659b` | add b1k modality.json deploy helper |
| `fe5af34` | preserve robot name/observation for eval serving (`ROBOT_OBS_CONFIGS`; `register_modality_config` was dropping `name`/`observation`) |
| `c2aaaf7` | remove duplicate `[tool.uv.sources]` table in `pyproject.toml` (broke `uv`) |
| `a6d22a8` | fix `train_b1k.py`: `EmbodimentTag.resolve` (was `.value` on a str), **N1.7 migration** (dropped Eagle `model_name`/`eagle_collator`), wired `--resume-from-checkpoint`/`--save-only-model`/`--skip-weight-loading` |

(The Cursor co-author trailer was stripped from all five via a local `filter-branch` rewrite.)

**Not yet committed / ad-hoc:**
- `omnigibson/eval/run_eval.py` — new eval driver (written, **untested** — GPU-blocked).
- `websockets` installed into `.venv` ad-hoc (serve dep; should be added to `pyproject`).

---

## Key findings worth remembering

- **N1.7, not N1.6:** repo code is `Gr00tN1d7` (Qwen3/Cosmos backbone). The b1k scripts originally pointed at the N1.6 (Eagle) base — incompatible. Migrated to `nvidia/GR00T-N1.7-3B`.
- **Cosmos-Reason2-2B is gated** — needs HF gate access + `HF_TOKEN` exported into the train/serve process.
- **modality.json is per-dataset but identical content** across all R1Pro b1k tasks; deploy via the helper.
- **No upstream eval runner** in this checkout — `omnigibson/eval/` has the `Evaluator`/`WebsocketPolicy`/metrics + post-hoc `score_utils.py`, so we wrote `run_eval.py`.
- **LeRobot v3.0 layout (vs v2.1):** `data_path`/`video_path` switch from `episode_{chunk,index}` to
  `{chunk_index,file_index}`; `episodes.jsonl`/`tasks.jsonl` → `meta/episodes/*.parquet` + `meta/tasks.parquet`;
  **many episodes per data parquet** (slice by the `episode_index` column) and **per-key concatenated mp4**
  (this episode's frames start at `round(from_timestamp*fps)` — verified exact). Global `stats.json` is the
  same and `data/*/*.parquet` glob already matches v3.0, so `generate_stats` is format-agnostic.
- **B300 (sm_103) + nvrtc:** `torch 2.7.1+cu128`'s bundled nvrtc rejects sm_103; **fixed** by
  `uv pip install nvidia-cuda-nvrtc-cu12==12.9.86` into the `.venv` (torch JIT-loads the newer
  nvrtc). Independent of dataset format — it broke Exp 1 identically. Re-apply after any `uv sync`.

---

## Where we stopped / next steps

1. **Shared F eval rollout is blocked** — all 8 GPUs were taken by another user's `openpi` job
   (~74 GB each). The serve+eval smoke is fully staged and will run once a GPU frees:
   `serve_b1k` (`checkpoint-50`, `r1pro.py`, `HF_TOKEN`) + `run_eval.py` on `turning_on_radio`.
2. **Validate `run_eval.py`** end-to-end (it's untested), then commit it + add `websockets` to `pyproject`.
3. **Run a real Experiment-1 training** (e.g. global 256 / 128-per-GPU) → eval that checkpoint for a real Q-score.
4. **Experiment 2 reader: ✅ done & validated end-to-end** (see section above) — incl. a real v3.0
   training run whose loss matches v2.1/Exp 1 (nvrtc B300 fix applied). Remaining: a full-length
   training run + eval rollout (Shared F) for a real Q-score.
5. **Commit** the v3.0 reader + parity tests on `lerobot-v3.0` (nothing committed yet). Consider
   pinning `nvidia-cuda-nvrtc-cu12==12.9.86` so the B300 fix survives `uv sync`.
