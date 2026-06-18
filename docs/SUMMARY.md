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

## Experiment 2 (lerobot-v3.0) — not started

- The v3 worktree (`Issac-GR00T-lerobot-v3.0`) was **removed** mid-session; the `lerobot-v3.0` branch
  still has the earlier shared commits cherry-picked, **but not** the later `train_b1k.py` fix.
- No v3.0 reader work has been done (the GR00T loader still only reads v2.x).

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

---

## Where we stopped / next steps

1. **Shared F eval rollout is blocked** — all 8 GPUs were taken by another user's `openpi` job
   (~74 GB each). The serve+eval smoke is fully staged and will run once a GPU frees:
   `serve_b1k` (`checkpoint-50`, `r1pro.py`, `HF_TOKEN`) + `run_eval.py` on `turning_on_radio`.
2. **Validate `run_eval.py`** end-to-end (it's untested), then commit it + add `websockets` to `pyproject`.
3. **Run a real Experiment-1 training** (e.g. global 256 / 128-per-GPU) → eval that checkpoint for a real Q-score.
4. **Experiment 2:** re-create the v3.0 worktree, replicate the `train_b1k.py` fix there, then build the v3.0 reader.
