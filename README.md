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
assets; Gazebo collection and the original SAC use the separate ROS Noetic
environment. Installation and asset preparation: [RUNNING](docs/RUNNING.md).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
```

**Current working baseline** (`real` critic target, actor horizon 8, BC 0.25).
It is not trained from scratch: it resumes policy62/WM562 (iteration 1561 of the
source-configuration run `critic_mask_validation_20261104`), discards the critic,
pre-fits a fresh critic for 100 rounds on the refilled replay, then trains:

```bash
.venv/bin/python scripts/train_gpu.py --worlds 1024 \
  --resume runs/critic_mask_validation_20261104/training/last.pt \
  --critic-target real --actor-horizon 8 --actor-bc-alpha 0.25 \
  --critic-reset --critic-prefit-rounds 100 \
  --microbatch 64 --critic-microbatch 8192 --max-iterations 860 --max-hours 12 \
  --output runs/current_baseline/training
```

**`source` configuration** (SuperMimic-Reproduced settings: imagined critic
targets, actor horizon 32, no BC; the script defaults), from scratch:

```bash
.venv/bin/python scripts/train_gpu.py --worlds 1024 --wm-only-updates 500 \
  --max-hours 24 --output runs/source/training
```

Frozen evaluation of either checkpoint (definitions in [BENCHMARK](docs/BENCHMARK.md)):

```bash
.venv/bin/python scripts/evaluate_benchmark.py --method supermimic \
  --checkpoint RUN/training/last.pt --output RUN/evaluation
```

Frozen WM + MPPI uses the same evaluation entry with `--method wm-mppi` after
installing `requirements-mpc.txt`. Settings and the complete command are in
[BENCHMARK](docs/BENCHMARK.md#world-model-mppi).

`scripts/record_episode.py` records the same physical benchmark;
`scripts/render_episode.py` produces a two-view video with measured force traces.
[Recording commands](docs/RUNNING.md#record-and-render-a-physical-episode).

W&B is enabled by default; `--wandb-mode offline` records locally. SIGUSR1
requests an atomic checkpoint after the current iteration; normal completion and
SIGTERM also save.

## Current result

Single prism (`public-prism-ur5e-v1`), frozen 8 episodes, one seed
([RESULTS](docs/RESULTS.md) 2026-11-05/06 and 2026-11-08):

| Policy | Success | Steps | Return | Peak material force |
|---|---:|---:|---:|---:|
| policy62 (source configuration, start point) | 8/8 | 20 | 90.81 | 30.66 N |
| Current working baseline, updates 150–300 | 8/8 | 17–18 | 92.2–92.7 | 35.0–46.4 N |
| Current working baseline, last | 8/8 | 19 | 91.37 | 26.10 N |
| SAC in the unified environment (source settings, one environment), step 85000 | 8/8 | 18 | 92.03 | 49.6 N |

The selected-controller evaluations are complete (2026-11-09). On 256
executed-action perturbation cases, policy62 / working policy / unified SAC85k
succeed in 74.6% / 83.2% / 89.1%, with mean material peaks 32.9 / 29.7 / 45.4 N.
The working policy's force-limit failure difference from SAC is inconclusive;
lower material peak does not establish lower failure risk. Training budgets,
input histories and checkpoint selection differ.

Variant A improved fixed-data prediction but did not show a clear benefit on
its pre-stated final-success criterion; it remains analysis. The working-baseline
repeat's temporary collapse and SAC's later collapse are retained in RESULTS.
The existing SAC85k is the current source-derived controller comparison, not a
full reproduction of SliceIt's paper. No SAC retraining is required for this
comparison. Next: consolidate saved evidence and the supported argument, then
scope one material-variation condition. [PLAN](docs/PLAN.md) records the decision;
[Paper direction and algorithm](docs/PAPER.md) describes the connected method.

[Current plan](docs/PLAN.md) · [Benchmark and definitions](docs/BENCHMARK.md) ·
[Results](docs/RESULTS.md) · [Methods and variants](docs/METHOD.md) ·
[Literature](docs/NOVELTY.md) · [Reading list](docs/READING_LIST.md) ·
[Real robot (draft)](docs/REAL2SIM.md) · [Source versions](docs/SOURCES.md)

The two original Git histories are retained. Third-party licenses are listed
in [NOTICE](NOTICE.md). Data, credentials and environments stay outside Git.
