# Cutting DEV

Force-conditioned robotic cutting with a learned world model and BPTT policy
optimization. UR5e + FDCC is shared by two material backends: DiSECt
(FEM, prepared cutting interface) and native BladeMaster (MPM, damage).
The current task version is **pose_depth_v3** (= the pose_depth_v2 task plus a
software board-stop consistency fix and analytic imagined survival; v2 is kept
only for the 2026-11-16 runs): prescribed final pose without failure,
with a board execution boundary. Pose success is not proof of material separation.

## Current entry points — 2026-11-18

| Purpose | Source |
|---|---|
| Decisions and next work | [PLAN](docs/PLAN.md) |
| Scene, robot, knife, reset distribution and cut locations | [SHARED_SCENE](docs/SHARED_SCENE.md) |
| Completion, rewards and constraints | [TASK_FREE_CUT](docs/TASK_FREE_CUT.md) |
| Learning implementation | [METHOD](docs/METHOD.md) |
| Ended 2026-11-16 launches (pose_depth_v2): WM+BPTT and GPU9 SAC | [Shared training](docs/archive/TRAINING_SHARED_20261116.md), [SAC training](docs/archive/TRAINING_SAC_20261116.md) |
| SAC, WM+MPPI and nonlearned MPC coverage | [CURRENT_BASELINES](docs/methods/CURRENT_BASELINES.md) |
| Vision implementation and fixed-heading method-2 pilot | [VISION_ROUTE1](docs/methods/VISION_ROUTE1.md) |
| Measured outcomes and limits | [RESULTS](docs/RESULTS.md) |
| Installation, tests, training/evaluation commands, recording and shared renderer | [RUNNING](docs/RUNNING.md) |
| Server data and preserved run provenance | [DATA](docs/DATA.md) |
| UR10e + ATI: first FDCC hardware setup | [Chinese setup handoff](docs/hardware/UR10E_FDCC_SETUP_ZH.md) |
| Full documentation index | [docs](docs/README.md) |

Development is consolidated on GitHub main. Training runs retain their frozen
checkouts and recorded revisions; updating main does not update those runs.
All long runs require online W&B and a recorded budget.

The selected scene uses freely placed cucumber003, half potato and prism, full
six-axis wrench observations, fixed knife heading, and final-position-only
constraints. Material parameters and simulated limits are not real-robot
calibration. See SHARED_SCENE for exact values instead of duplicated launch recipes.

DP3 midpoint selection now executes frozen WM+BPTT, paired frozen WM+MPPI,
and nonlearned FDCC+MPPI in a one-world native-BladeMaster prototype. In one
nominal cucumber case each, the old policy66 and WM566 planner timed out;
nonlearned MPC reached the pose goal. These are interface illustrations, not
an algorithm ranking or validated slice separation. SAC visual handoff remains
unimplemented; see docs/methods/VISION_ROUTE1.md for exact scope and commands.

## Repository map

- cutting_world_model/: learning, task adapters and perception handoff.
- backends/mujoco/: robot, material coupling and parallel simulation.
- integrations/sliceit/: pinned upstream integration and provenance.
- scripts/: [execution and diagnostic entry points](scripts/README.md).
- configs/: [versioned configuration index](configs/README.md).
- runs/: server-only data, checkpoints and evidence, excluded from Git.

## Historical reproduction

The original fixed-prism contract and commands are in
[BENCHMARK](docs/archive/BENCHMARK.md); v1-v3 and old pose-v1 results retain their original
task definitions. [Archived records](docs/README.md#archive) are
evidence, not the current training recipe. Historical README recipes remain at
[4653d10](https://github.com/Ju6276/Cutting_DEV/blob/4653d10/README.md).
Never bypass physics/task metadata to present old weights as current-task results.

[Paper scope](docs/PAPER.md) · [Literature](docs/literature/README.md) ·
[Real robot draft](docs/hardware/REAL2SIM.md) · [Sources](docs/reference/SOURCES.md) ·
[Licenses](NOTICE.md)
