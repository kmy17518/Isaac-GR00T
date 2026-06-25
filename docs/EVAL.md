# Remote B1K eval over an SSH tunnel (GR00T policy server ↔ OmniGibson)

How to evaluate a trained GR00T policy when the **GPU box that serves the policy** and the
**machine that runs the OmniGibson/Isaac-Sim simulation** are two *different* computers on
*different* networks (e.g. behind NAT, no direct route between them).

This is the normal arrangement when the GPU box can run the policy but **cannot render**
(e.g. a compute-only NVIDIA driver with no Vulkan ICD — Isaac Sim needs Vulkan), so the
simulation has to run elsewhere. The two halves talk over a websocket; we bridge the two
machines with an **SSH tunnel through the GPU box's public SSH port** (port 22), so no extra
inbound ports need to be opened.

See `START.md` §8 for the single-machine (two-GPU, same host) version.

---

## Topology

```
  ┌─────────────────────────────┐         SSH (port 22, public)         ┌──────────────────────────────┐
  │  EVAL BOX (e.g. Lambda)      │  ───────── ssh -L 8000:… ──────────▶  │  GPU BOX (e.g. nebius)        │
  │  OmniGibson + Isaac Sim      │                                       │  GR00T policy server          │
  │  run_eval.py  ──ws──▶ :8000  │   localhost:8000 ─┐         ┌─ localhost:8000                          │
  │  (--host 127.0.0.1)         │                   └── tunnel ┘   serve_b1k.py (Gr00tPolicy on a GPU)   │
  └─────────────────────────────┘                                       └──────────────────────────────┘
```

The eval client connects to `127.0.0.1:8000` on the **eval box**; SSH forwards that to
`127.0.0.1:8000` on the **GPU box**, where the policy server listens. Because traffic arrives
via the SSH login, the server only needs to bind loopback (`127.0.0.1`) — nothing is exposed
on the public network.

---

## 0. Setup-specific values (adjust to your deployment)

| What | Value (this deployment) |
|---|---|
| GPU box public SSH host | `89.124.37.170` (SSH alias `nebius`) |
| GPU box SSH user | `stuart` |
| Policy server port | `8000` |
| GR00T repo (GPU box) | `/home/stuart/ThunderPuppies/Isaac-GR00T` |
| Checkpoint to serve | `outputs/<exp>/<exp>/checkpoint-<step>/` (a real run), or a smoke ckpt under `/tmp` |
| Eval box OmniGibson | `…/BEHAVIOR-1K/OmniGibson` |
| Eval box conda python | `…/miniconda3/envs/behavior/bin/python` (`<behavior_py>`) |

> The GPU box's `eth0` is a **private** IP (`10.160.0.x/32`); it is *not* reachable from
> another network. Only the **public SSH endpoint** (`89.124.37.170:22`) is, which is exactly
> what the tunnel uses.

---

## 1. Prerequisites

**GPU box (serves the policy):**
- GR00T `.venv` working (`source .venv/bin/activate`); on Blackwell B300 / CUDA-13 nodes also
  apply the two driver-stack fixes (else training/serving crash): `nvidia-nccl-cu12>=2.27.7`
  and `nvidia-cuda-nvrtc-cu12>=12.9.86`.
- `websockets` installed in the venv (serving dep).
- A **servable checkpoint** — i.e. a `checkpoint-<step>/` produced by training (it contains
  `config.json`, `model-*.safetensors`, `processor_config.json`, `statistics.json`,
  `experiment_cfg/`). `CheckpointFormatCallback` makes every saved checkpoint standalone.
- `HF_TOKEN` exported (`set -a && source ThunderPuppies/.env && set +a`) — the processor build
  still resolves the gated Cosmos backbone.

**Eval box (runs the simulation):**
- The OmniGibson `behavior` conda env with a **working Isaac Sim / Vulkan** (`run_eval` renders
  via Vulkan — confirm the NVIDIA Vulkan ICD exists: `ls /usr/share/vulkan/icd.d/nvidia_icd.json`).
- The BEHAVIOR datasets at `gm.DATA_PATH`, including
  `2025-challenge-task-instances/` with `metadata/available_tasks.yaml` (maps each task to its
  `scene_model`, e.g. `turning_on_radio → house_double_floor_lower`) and the scene instances.
- `websockets` in the env (the eval client uses it).

---

## 2. Start the policy server (on the GPU box)

```bash
cd /home/stuart/ThunderPuppies/Isaac-GR00T
set -a && source /home/stuart/ThunderPuppies/.env && set +a
CUDA_VISIBLE_DEVICES=<gpu_id> .venv/bin/python scripts/b1k/serve_b1k.py \
  --model-path outputs/b1k-turning_on_radio/b1k-turning_on_radio/checkpoint-XXXX \
  --modality-config-path examples/b1k/r1pro.py \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 --port 8000
```

- Keep `--host 127.0.0.1` (the default): with the SSH tunnel you do **not** need `0.0.0.0`,
  and loopback-only is safer — the server has **no authentication**, so the SSH login is the
  security boundary.
- `CUDA_VISIBLE_DEVICES=<gpu_id>` pins the policy to one GPU.
- Wait until it logs `Starting websocket server on 127.0.0.1:8000...` and
  `Loading checkpoint shards: 100%`. Health check (on the GPU box): `curl -s localhost:8000/healthz` → `OK`.
- Leave it running (it serves forever). Run it in `tmux`/`nohup` if you want to detach.

---

## 3. Open the SSH tunnel (on the eval box)

```bash
# Forward eval-box localhost:8000 -> GPU-box localhost:8000, via the public SSH port.
ssh -fN -L 8000:localhost:8000 stuart@89.124.37.170
```

- `-L 8000:localhost:8000` = "listen on my :8000, forward to the GPU box's :8000".
- `-N` = no remote command (tunnel only). `-f` = background it (without `-f` the command shows
  no output and *looks* hung — that is normal `-N` behavior, not a failure).
- **Auth:** the eval box needs the GPU box's SSH key. Easiest: SSH into the eval box from your
  laptop with agent forwarding (`ssh -A …` / `ForwardAgent yes`), so the key already in your
  agent is used for this onward hop. Otherwise copy the key to the eval box.
- Verify (on the eval box): `curl -s -m 5 http://127.0.0.1:8000/healthz` → should print `OK`.
- Close it later with: `pkill -f "ssh -fN -L 8000:localhost:8000"`.

---

## 4. Run the eval (on the eval box)

```bash
cd …/BEHAVIOR-1K/OmniGibson
OMNIGIBSON_GPU_ID=<gpu_id> <behavior_py> -m omnigibson.eval.run_eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 1 2 3 4 5 6 7 8 9 \
  --output-dir /tmp/b1k_eval/standard.public.<team>.<affiliation>.<date>
```

- Point the eval at **`127.0.0.1:8000`** (the tunnel's local end), *not* the GPU box's IP.
- Select the sim GPU with **`OMNIGIBSON_GPU_ID`**, *not* `CUDA_VISIBLE_DEVICES`: Omniverse uses
  its own (Vulkan) device enumeration, and setting `CUDA_VISIBLE_DEVICES` breaks it
  ("No device could be created" → segfault).
- Omit `--max-steps` for the challenge default (2× mean human-demo length); pass `--max-steps 50`
  for a fast pipeline smoke.
- Writes per-rollout `q_score` / `time` / `agent_distance` JSONs under `<output-dir>/json/`,
  and prints an `EVAL SUMMARY: N_success/N | mean q_score=…` line at the end.

**Score:**
```bash
<behavior_py> OmniGibson/omnigibson/eval/utils/score_utils.py -i <parent_of_submission_dir> -o <out_dir>
```

---

## 5. Teardown

```bash
# eval box: close the tunnel
pkill -f "ssh -fN -L 8000:localhost:8000"
# GPU box: stop the server (frees the GPU + port 8000)
pkill -f "scripts/b1k/serve_b1k.py"     # or: kill <serve_pid>
```

---

## 6. Automated multi-checkpoint eval (orchestrator ↔ client)

Sections 2–5 cover **one** checkpoint at a time, by hand. To sweep **many** checkpoints
(e.g. every preserved checkpoint, newest → oldest) without babysitting, use the two-machine
handshake below. The GPU box serves one checkpoint at a time; the eval box evaluates each as
it becomes ready. They coordinate through a **file "mailbox" on the GPU box** that the eval
box reads/writes over the *same* SSH access the tunnel already uses — so still no inbound
ports on the eval box, and the policy server still binds loopback only.

**Flow:** GPU box serves ckpt N → publishes `ready` → eval box evaluates N → writes `done/N`
→ GPU box stops N and serves the next-newest un-acked checkpoint → … → exits (`alldone`) once
every checkpoint has been evaluated. The `done/` markers make the whole sweep **resumable**
and **idempotent** (re-running either side picks up where it left off).

Scripts (operational, on the GPU box — outside the repo):
- `/home/stuart/ThunderPuppies/eval_orchestration/orchestrator.sh` — Computer 1 (GPU box) loop.
- `/home/stuart/ThunderPuppies/eval_orchestration/eval_client.sh` — Computer 2 (eval box) loop.

Mailbox (`/home/stuart/ThunderPuppies/eval_queue/` on the GPU box):

| File | Writer | Meaning |
|---|---|---|
| `serving.env` | GPU box | current state: `state=ready\|switching\|down\|alldone`, plus `step`, `name`, `port` |
| `done/<step>` | eval box | ack — checkpoint `<step>` has been evaluated |
| `results/<step>.txt` | eval box | the run's `EVAL SUMMARY` line, shipped back for at-a-glance progress |
| `orchestrator.log` | GPU box | serve / advance history |

### Computer 1 — this (GPU) box: serve the sweep

```bash
# Knobs live at the top of the script: SERVED dir, PREFIX, GPU (default 7), PORT (8000),
# MODALITY, EMB, HEALTH_TIMEOUT, ACK_POLL.
tmux new -d -s eval_orch 'bash /home/stuart/ThunderPuppies/eval_orchestration/orchestrator.sh'
```

- Serves `…/served_checkpoints/<PREFIX><step>/` **newest step → oldest**, one at a time, on
  `GPU` / `PORT` (loopback), waiting for `/healthz == OK` before it publishes `ready`.
- Skips any `<step>` that already has a `done/<step>` marker, then exits once all are acked.

Monitor / control (GPU box):
```bash
tail -f /home/stuart/ThunderPuppies/eval_queue/orchestrator.log   # progress
cat     /home/stuart/ThunderPuppies/eval_queue/serving.env        # what's served right now
ls      /home/stuart/ThunderPuppies/eval_queue/done/              # acked steps
cat     /home/stuart/ThunderPuppies/eval_queue/results/*.txt      # eval summaries
tmux kill-session -t eval_orch                                    # stop the sweep (frees the GPU)
```
- **Re-eval a step:** `rm /home/stuart/ThunderPuppies/eval_queue/done/<step>` (re-served next pass; restart `eval_orch` if it already exited).
- **Skip a step:** pre-create its ack, e.g. `touch /home/stuart/ThunderPuppies/eval_queue/done/25500`.
- **Change serving GPU:** edit `GPU=` at the top of `orchestrator.sh`, then restart the `eval_orch` session.

### Computer 2 — eval box: evaluate as checkpoints go ready

```bash
# 1. copy the client from the GPU box (uses your existing SSH access)
scp stuart@89.124.37.170:/home/stuart/ThunderPuppies/eval_orchestration/eval_client.sh ~/

# 2. edit the marked vars at the top: OMNIGIBSON_DIR, BEHAVIOR_PY, SIM_GPU_ID, INSTANCES,
#    OUT_BASE  (GPU_HOST / QUEUE already point at this deployment)
nano ~/eval_client.sh

# 3. run it under tmux so it survives disconnects
tmux new -s eval_client 'bash ~/eval_client.sh'
```

- Polls the GPU box's `serving.env` over SSH; when `state=ready` for a step it hasn't done, it
  auto-(re)opens the `:8000` tunnel (§3), runs `run_eval` (task `turning_on_radio`, §4) for that
  checkpoint into `OUT_BASE/step_<N>/`, then acks `done/<N>` back to the GPU box.
- Stops itself when the GPU box reports `state=alldone`.
- Requires passwordless SSH (key/agent) to `stuart@89.124.37.170` — same auth as the tunnel in §3.

**Start order:** bring up Computer 1 first (so a checkpoint is `ready`), then Computer 2. Either
side may be stopped/restarted independently — the sweep resumes from the existing `done/` markers.

---

## 7. Troubleshooting

- **`curl …/healthz` returns nothing / hangs from the other box (no `OK`)** → the two machines
  can't route to each other (private IPs / different networks). Use the tunnel via the public
  SSH port (§3); don't target the GPU box's private IP or `:8000` directly.
- **`ssh -N` "hangs"** → expected. `-N` opens the tunnel and stays silent. Use `-f` to
  background, and drive the eval from a second terminal. (No password prompt after the host-key
  prompt = your key/agent auth already succeeded.)
- **Direct `--host <public-ip> --port 8000` refused/times out** → cloud firewalls typically
  expose only port 22, not 8000. The SSH tunnel avoids needing to open 8000 (and avoids exposing
  the no-auth server). Only open 8000 in the cloud console if you really want direct access.
- **`AssertionError: Got invalid task name` / `available_tasks.yaml not found`** → the eval box
  is missing `2025-challenge-task-instances/metadata/available_tasks.yaml`. Provide it (per task:
  `scene_model`, `robot_start_position`, `robot_start_orientation`).
- **Isaac Sim segfault / "No device could be created"** → Vulkan can't see the GPU. Check
  `/usr/share/vulkan/icd.d/nvidia_icd.json` exists and the NVIDIA graphics/Vulkan userspace is
  installed (a compute-only driver won't render), and select the GPU via `OMNIGIBSON_GPU_ID`.
- **Connection drops mid-eval** → keep both the server and the `ssh` tunnel alive for the whole
  run (use `tmux`); consider `autossh` for a long eval.
- **Throughput / latency** → each step ships the observation uncompressed (~3 MB: head 720×720×3
  + two 480×480×3 wrists + proprio) up and the action down. Fine on a fast link; adds per-step
  latency on slow ones.
- **(Automated flow, §6) `eval_client` not advancing** → read `eval_queue/serving.env` on the GPU
  box: `state=switching`/`down` means the orchestrator is between checkpoints (wait); `state=alldone`
  means the sweep finished. If one step is stuck, check `/tmp/eval_orch_serve.log` (server startup)
  and `eval_queue/orchestrator.log`. Also ensure the eval box's SSH to `stuart@89.124.37.170` is
  passwordless (key/agent) — otherwise the per-poll `ssh`/`scp` calls block on a prompt.
