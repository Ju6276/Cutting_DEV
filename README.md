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
.venv/bin/python scripts/evaluate_benchmark.py --method supermimic \
  --checkpoint runs/training/last.pt \
  --output runs/evaluation
```

W&B is enabled by default; authenticate with `.venv/bin/wandb login`.
`--wandb-mode offline` records locally. SIGUSR1 requests an atomic checkpoint
after the current iteration; normal completion and SIGTERM also save.

## Current result

2026-11-05: the shared evaluator executes policy62 successfully in **20 steps /
1.00 s**, peak material force **30.66 N**. The source SAC adapter completes
real physical transitions, learning updates and checkpoint/replay reload.

With policy/critic frozen, 100 additional WM rounds reduce held-out full-trajectory
position RMSE **19.42 → 7.15 mm**, but miss all **32** successful terminal events.
Predicted final height remains **6.52–16.39 mm** from the target. The next model
check separates the physical-state and terminal-classification gradients.
The last measured critic return MAE is **153.42**; critic weights stayed frozen in this probe.

[Fixed benchmark](docs/BENCHMARK.md) · [Results](docs/RESULTS.md) · [Current work](docs/PLAN.md) ·
[Methods and differences](docs/METHOD.md) · [Source versions](docs/SOURCES.md)

The two original Git histories are retained. Third-party licenses are listed
in [NOTICE](NOTICE.md). Data, credentials and environments stay outside Git.
