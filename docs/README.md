# Documentation map

Start with PLAN for decisions and next work, TASK_FREE_CUT and SHARED_SCENE for
the current task (pose_depth_v3), and DATA for server locations. A dated result
is not a training instruction.

| Document | Purpose |
|---|---|
| [PLAN](PLAN.md) | Current decisions, fixed task/algorithm settings and next work |
| [PAPER](PAPER.md) | Research scope and claims that still require evidence |
| [TASK_FREE_CUT](TASK_FREE_CUT.md) | pose_depth_v3 task: completion, board boundary, rewards, survival and diagnostic labels |
| [SHARED_SCENE](SHARED_SCENE.md) | Shared scene, reset distribution, physical knife/mount and backend differences |
| [METHOD](METHOD.md) | Implementation: learner configuration and variants, physics interfaces, GPU memory policy |
| [RESULTS](RESULTS.md) | Measurements that still hold for the current code and task |
| [RUNNING](RUNNING.md) | Environment, tests, training/evaluation/diagnostic commands, resume, W&B and rendering |
| [DATA](DATA.md) | Server paths, checkpoints and preserved evidence |

- [methods/](methods/): [CURRENT_BASELINES](methods/CURRENT_BASELINES.md) (SAC, frozen
  WM+MPPI, nonlearned FDCC+MPPI and WM+BPTT: code, inputs, coverage);
  [VISION_ROUTE1](methods/VISION_ROUTE1.md) (pre-cut visual goal selection, DP3
  midpoint predictor and physical handoffs).
- [hardware/](hardware/): [UR10E_INTEGRATION_REVIEW](hardware/UR10E_INTEGRATION_REVIEW.md)
  (UR10e/ATI/CAD-knife simulation integration); [UR10E_FDCC_SETUP_ZH](hardware/UR10E_FDCC_SETUP_ZH.md)
  (Chinese UR10e + ATI FDCC bring-up handoff); [REAL2SIM](hardware/REAL2SIM.md)
  (real-robot interface and calibration plan).
- [reference/](reference/): [CUTTING_MECHANISMS](reference/CUTTING_MECHANISMS.md)
  (DiSECt/BladeMaster cutting mechanisms and task signals);
  [SOURCES](reference/SOURCES.md) (upstream revisions, patches and provenance).
- [literature/](literature/README.md): per-paper evidence records and the reading
  list; [candidate topic search](literature/force_adaptive_search_20261112.md).

[Script map](../scripts/README.md) and [configuration index](../configs/README.md)
identify execution entry points and versioned configs. Superseded documents are
not kept in the working tree; Git history retains them.
- [Machines and access](MACHINES.md): hosts, SSH paths, GitHub and W&B credential locations.
