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

[Documentation map](docs/README.md) · [Script entry points](scripts/README.md) ·
[Data and checkpoint locations](docs/DATA.md)

## Run the existing fixed-prism task

These commands reproduce the existing task. Reset-time material sampling and
shared multi-geometry training are proposed in PLAN, not implemented by these
commands.

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
  --resume runs/mpc_assets/policy62_wm562_original.pt \
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

These are historical fixed-task checkpoints, not a matched-budget algorithm
comparison. Variant A improved fixed-data prediction but did not establish its
pre-stated final control benefit. The working-baseline repeat's temporary
collapse and SAC's later collapse remain in RESULTS. The existing SAC85k is
a source-derived controller comparison, not a full reproduction of SliceIt's
paper.

The source learning chain began with fresh networks, 500 WM-only rounds and
later joint updates, reaching policy62/WM562. The working baseline above
continues that policy; it does not imply the entire approach only fine-tunes
an external expert. Resumes and terminal/masking fixes are part of its lineage.

## Current research direction

Train **one shared WM, policy and critic** across declared material conditions
and prepared geometries: prism variation, then grouped prism/jello simulation.
The hypothesis is lower contact force at comparable completion to SAC trained
on the same task distribution and physical-interaction budget. It is not a
measured multi-condition result yet.

[PLAN](docs/PLAN.md) specifies S0–S4: define the distribution and splits,
implement episode-reset sampling, connect geometry groups to one learner, train
matched comparisons, then evaluate predefined conditions. No numerical
randomization ranges or new long runs are selected by this documentation review.
Fresh SAC training is justified for the new distribution; the old fixed-task
comparison remains valid within its stated limits.

Action-perturbation evaluations are retired and excluded from the paper. Their
unchanged data and figures remain in the [historical evidence archive](docs/archive/EVIDENCE_20261110.md).
The unexecuted ten-episode frozen-material pilot is withdrawn. Training
exploration is separate and is not disabled.
[Paper argument and algorithm](docs/PAPER.md) records the proposed claim, existing
evidence and concrete scope decisions for review.

[Current plan](docs/PLAN.md) · [Benchmark and definitions](docs/BENCHMARK.md) ·
[Results](docs/RESULTS.md) · [Methods and variants](docs/METHOD.md) ·
[Literature](docs/NOVELTY.md) · [Reading list](docs/READING_LIST.md) ·
[Real robot (draft)](docs/REAL2SIM.md) · [Source versions](docs/SOURCES.md)

The two original Git histories are retained. Third-party licenses are listed
in [NOTICE](NOTICE.md). Data, credentials and environments stay outside Git.
