# Cutting DEV

Force-conditioned robotic cutting with a learned world model and BPTT policy
optimization. UR5e + FDCC is shared by two material backends: DiSECt
(FEM, prepared cutting interface) and native BladeMaster (MPM, damage).
The current task is **pose_depth_v2**: prescribed final pose without failure,
with a board execution boundary. Pose success is not proof of material separation.

## Current entry points — 2026-11-17

| Purpose | Source |
|---|---|
| Decisions and next work | [PLAN](docs/PLAN.md) |
| Scene, robot, knife, reset distribution and cut locations | [SHARED_SCENE](docs/SHARED_SCENE.md) |
| Completion, rewards and constraints | [TASK_FREE_CUT](docs/TASK_FREE_CUT.md) |
| Learning implementation | [METHOD](docs/METHOD.md) |
| GPU7 / A6000 frozen launch protocol | [Shared training](docs/TRAINING_SHARED_20261116.md) |
| GPU9 SAC launch protocol | [SAC training](docs/TRAINING_SAC_20261116.md) |
| SAC, WM+MPPI and nonlearned MPC coverage | [CURRENT_BASELINES](docs/CURRENT_BASELINES.md) |
| Vision implementation and fixed-heading method-2 pilot | [VISION_ROUTE1](docs/VISION_ROUTE1.md) |
| Measured outcomes and limits | [RESULTS](docs/RESULTS.md) |
| Installation, tests, recording and shared renderer | [RUNNING](docs/RUNNING.md) |
| Server data and preserved run provenance | [DATA](docs/DATA.md) |
| Full documentation index | [docs](docs/README.md) |

Development is consolidated on GitHub main. Running jobs retain their frozen
checkouts and recorded revisions; updating main does not update those jobs.
All long runs require online W&B and a recorded budget.

The selected scene uses freely placed cucumber003, half potato and prism, full
six-axis wrench observations, fixed knife heading, and final-position-only
constraints. Material parameters and simulated limits are not real-robot
calibration. See SHARED_SCENE for exact values instead of duplicated launch recipes.

Visual selection currently executes only a frozen WM+BPTT policy. SAC and the
two planners have independent current-task evaluators; their visual handoffs
are not implemented. Method 2 now has a trained fixed-heading center predictor and a frozen-policy
physical trial; neither visual route's latest trial achieved pose completion.

## Repository map

- cutting_world_model/: learning, task adapters and perception handoff.
- backends/mujoco/: robot, material coupling and parallel simulation.
- integrations/sliceit/: pinned upstream integration and provenance.
- scripts/: [execution and diagnostic entry points](scripts/README.md).
- configs/: [versioned configuration index](configs/README.md).
- runs/: server-only data, checkpoints and evidence, excluded from Git.

## Historical reproduction

The original fixed-prism contract and commands are in
[BENCHMARK](docs/BENCHMARK.md); v1-v3 and old pose-v1 results retain their original
task definitions. [Archived records](docs/README.md#historical-records) are
evidence, not the current training recipe. Historical README recipes remain at
[4653d10](https://github.com/Ju6276/Cutting_DEV/blob/4653d10/README.md).
Never bypass physics/task metadata to present old weights as current-task results.

[Paper scope](docs/PAPER.md) · [Literature](docs/NOVELTY.md) ·
[Real robot draft](docs/REAL2SIM.md) · [Sources](docs/SOURCES.md) ·
[Licenses](NOTICE.md)
