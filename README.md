# Cutting DEV

UR5e robotic cutting with SliceIt/DiSECt, SAC and a SuperMimic-style learned
world model. The robot and material execute physical transitions; policy
gradients pass through the learned world model.

| Component | Location | Responsibility |
|---|---|---|
| Learning | `cutting_world_model/` | WM, policy, critic, replay and task adapters |
| Physics | `backends/mujoco/` | MuJoCo Warp, DiSECt, compliance and parallel worlds |
| SliceIt baseline | `integrations/sliceit/` | Pinned upstream sources and integration patches |
| Commands | `scripts/` | Collection, training and frozen evaluation |
| Experiments | `runs/` | Local data, checkpoints and videos; excluded from Git |

## Run

GPU training uses Linux, Python 3.12, NVIDIA CUDA and prepared robot/material
assets. Gazebo collection and SAC use the separate ROS Noetic environment.
Installation and asset preparation: [RUNNING](docs/RUNNING.md).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
# After preparing the assets described in RUNNING:
.venv/bin/python scripts/train_gpu.py --worlds 1024 --wm-only-updates 500 \
  --max-hours 24 --output runs/training
.venv/bin/python scripts/evaluate_gpu.py --checkpoint runs/training/last.pt \
  --output runs/evaluation
```

W&B is enabled by default; authenticate with `.venv/bin/wandb login`.
`--wandb-mode offline` records locally. SIGUSR1 requests an atomic checkpoint
after the current iteration; normal completion and SIGTERM also save.

## Current result

2026-11-04, revision `2495171`: frozen policy62 completes **8/8 cuts**,
**1.00 s simulated time** per cut, **39.03 s wall time** for eight worlds,
peak force **30.66 N**. Critic return MAE remains **153.42**.
Short terminal replay is repaired: actual WM updates use **2433 collision
terminal labels** from one/two-step failures. The next focus is accumulated
WM joint/pose error, which predicts a workspace failure on successful cuts.

[Results](docs/RESULTS.md) · [Current work](docs/PLAN.md) ·
[Methods and differences](docs/METHOD.md) · [Source versions](docs/SOURCES.md)

The two original Git histories are retained. Third-party licenses are listed
in [NOTICE](NOTICE.md). Data, credentials and environments stay outside Git.
