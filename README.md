# Cutting DEV

UR5e single-cut control using MuJoCo Warp, a SliceIt-derived compliant controller
and DiSECt. The learning method predicts transitions with a world model (WM)
and trains a policy by backpropagating through that model.

## Current work — 2026-11-13

The current task is **free_cut_v3**: one cut of a freely placed object, complete
material separation, commanded feed speed and explicit sensor-wrench/workspace
limits. Small object sliding is allowed and logged. The current experiment uses
one scanned cucumber scene with randomized simulation materials; it is not yet
the multi-object paper comparison or a real-robot validation.

An authorized GPU7 run continues joint WM/policy/critic training from 43 policy
updates. Budget: 1250 additional vector steps, at most 9 hours; checkpoints are
measured on the same 64 development conditions. No pass/fail calibration gates,
new reward, uncertainty penalty or action limiter are introduced in this run.
See [PLAN](docs/PLAN.md) for the run manifest and ownership.

## Start here

- [Current plan and run](docs/PLAN.md)
- [Task and reward contract](docs/TASK_FREE_CUT.md)
- [Implemented learning method](docs/METHOD.md)
- [Execution and resume instructions](docs/RUNNING.md)
- [Current results](docs/RESULTS.md)
- [Document map](docs/README.md), [script map](scripts/README.md), [server data](docs/DATA.md)

| Component | Location |
|---|---|
| WM, policy, critic, replay and task adapters | `cutting_world_model/` |
| MuJoCo Warp, DiSECt and compliant control | `backends/mujoco/` |
| Independent SliceIt/Gazebo sources and patches | `integrations/sliceit/` |
| Training, recording and evaluation | `scripts/` |
| Prepared assets, weights, recordings and logs | Server-only paths in [DATA](docs/DATA.md); excluded from Git |

Source, environments and outputs stay on the servers; use Remote SSH through
`tams61`. Active processes keep their pinned checkout. The historical branch name
`codex/free-cut-v1` contains v3 work; it is not the task-version identifier.
Training currently uses the separately reviewed infrastructure commit `b1bcb92`.
Do not assume `main` or a branch name identifies a running experiment.

## Research and reproduction

[Paper scope](docs/PAPER.md) distinguishes the task from unproven method claims.
[Literature evidence](docs/NOVELTY.md) and [paper/code notes](docs/literature/README.md)
retain their reading scope and pinned sources. [REAL2SIM](docs/REAL2SIM.md) is a
conditional draft.

[Historical fixed-prism benchmark](docs/BENCHMARK.md) uses a pose-goal success rule,
not current separation success. Its SAC/WM/MPPI results cannot be pooled with v3.
[Historical records](docs/archive/README.md) retain adverse results and immutable
links to retired plans/reviews.

Physics revision `edge_coordinate_consistent_v2` is checked in task metadata.
Neither scalar nor net feedback is a calibrated Mini45 measurement. Preserve
compatibility checks and reproduce old runs from their own manifests.

Dependencies and licenses: [SOURCES](docs/SOURCES.md), [NOTICE](NOTICE.md).
