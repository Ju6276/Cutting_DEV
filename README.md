# Cutting DEV

Force-conditioned robotic cutting with a learned world model and BPTT policy
optimization. A robot with FDCC compliance control is shared by two material
backends: DiSECt (FEM, prepared cutting interface) and native BladeMaster (MPM,
damage). The current task is **pose_depth_v3** with `project_continue_v2` board
handling: prescribed final pose without failure (cumulative arrival >10, 2 mm
tolerance, +100/-100, 500 steps at 0.05 s), geometric nonpenetration and
continuing material/sensor updates at the board. Pose success is not proof of
material separation.

## Entry points

| Purpose | Source |
|---|---|
| Decisions and next work | [PLAN](docs/PLAN.md) |
| Scene, robot, knife, reset distribution and cut locations | [SHARED_SCENE](docs/SHARED_SCENE.md) |
| Completion, rewards and constraints | [TASK_FREE_CUT](docs/TASK_FREE_CUT.md) |
| Learning implementation | [METHOD](docs/METHOD.md) |
| All four controllers: code, task versions and tested coverage | [CURRENT_BASELINES](docs/methods/CURRENT_BASELINES.md) |
| Vision goal selection | [VISION_ROUTE1](docs/methods/VISION_ROUTE1.md) |
| Measured outcomes and limits | [RESULTS](docs/RESULTS.md) |
| Installation, tests, training/evaluation commands, recording and rendering | [RUNNING](docs/RUNNING.md) |
| Main requirements and isolated baseline/render/CAD environments | [Dependencies](scripts/env/README.md) |
| Server data and run provenance | [DATA](docs/DATA.md) |
| UR10e + ATI + CAD knife integration | [Integration handoff](docs/hardware/UR10E_INTEGRATION_REVIEW.md), [Chinese setup handoff](docs/hardware/UR10E_FDCC_SETUP_ZH.md) |
| Full documentation index | [docs](docs/README.md) |

**One current task, not one task per algorithm:** WM+BPTT, SB3 SAC, frozen
WM+MPPI and nonlearned FDCC+MPPI all use pose_depth_v3. DiSECt/BladeMaster
select material physics, not an RL algorithm. Older task versions are
evaluate-only; never bypass physics/task metadata to present old weights as
current-task results.

Every server runs from /dataHDD/dong/Projects/Cutting_DEV with
`source scripts/env/cutting_env.sh` (SAC/planners: `CUTTING_ENV=sac` / `planner`;
see [RUNNING](docs/RUNNING.md#remote-only-development-and-storage)). Frozen
training checkouts keep their recorded revisions. All long runs require an
approved plan, a recorded budget and online W&B.

The `scene_v1` scene uses freely placed cucumber003, half potato and prism on a
UR5e, full six-axis wrench observations, fixed knife heading and final-position-
only constraints; the UR10e/ATI/CAD-knife profile is under integration. Material
parameters and simulated limits are not real-robot calibration.

## Repository map

- cutting_world_model/: learning, task adapters and perception handoff.
- backends/mujoco/: robot, material coupling and parallel simulation.
- integrations/sliceit/: pinned upstream integration and provenance.
- scripts/: [execution and diagnostic entry points](scripts/README.md).
- configs/: [versioned configuration index](configs/README.md).
- runs/: server-only data, checkpoints and evidence, excluded from Git.

[Paper scope](docs/PAPER.md) · [Literature](docs/literature/README.md) ·
[Real robot plan](docs/hardware/REAL2SIM.md) · [Sources](docs/reference/SOURCES.md) ·
[Licenses](NOTICE.md)
