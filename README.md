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

2026-11-04, source revision `57286e4`: frozen policy52 completes **8/8 cuts**,
**0.90 s simulated time** per cut, **35.12 s wall time** for the eight-world
evaluation, peak force **42.60 N**. Critic return MAE is **141.63**.
The replay adapter now admits short terminal sequences: **1121 collision
episodes** of 1–2 steps were excluded by the previous WM sampler. The next
physical-return evaluation checks learning after this repair.

[Results](docs/RESULTS.md) · [Current work](docs/PLAN.md) ·
[Methods and differences](docs/METHOD.md) · [Source versions](docs/SOURCES.md)

The two original Git histories are retained. Third-party licenses are listed
in [NOTICE](NOTICE.md). Data, credentials and environments stay outside Git.
