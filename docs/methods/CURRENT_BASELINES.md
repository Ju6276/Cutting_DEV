# Current controllers and baseline status

This is the single controller/version/coverage ledger, including WM+BPTT.
[PLAN](../PLAN.md) selects work and budgets; [SHARED_SCENE](../SHARED_SCENE.md)
defines the scene; [TASK_FREE_CUT](../TASK_FREE_CUT.md) defines the task.
Do not infer a completed experiment from an available CLI or current config.

## One code line, one current task, two material backends

- Current code for ALL four controllers is on main in Cutting_DEV. No controller
  requires switching to a method-specific development branch.
- Current task is pose_depth_v3 (`board.motion=project_continue_v2`). Older task
  versions are retained only to evaluate old checkpoints; their training is refused.
- DiSECt and BladeMaster are material physics backends, not two RL algorithms.
  Either backend is selectable through the same current interfaces. Interface
  availability is not evidence of measured performance on both backends.
- Frozen worktrees retain old run source/data. They are not alternative current
  launch entries.
- No old checkpoint, replay or result is relabelled as v3.

| Controller / CLI name | What is learned? | Execution | Current source / entry |
|---|---|---|---|
| WM+BPTT / wm_policy | WM, critic and policy | Policy inference then FDCC | cutting_world_model/trainer.py; scripts/train_gpu.py |
| SB3 SAC / sac | Policy and Q functions; no learned dynamics | Policy inference then FDCC | cutting_world_model/sac_vector.py; scripts/train_sac_multi.py |
| Frozen WM+MPPI / wm_mppi | Reuses a compatible trained WM; optional paired critic; no optimization of weights at evaluation | Online MPPI over the frozen WM then FDCC | cutting_world_model/shared_baselines.py and mpc.py |
| Nonlearned FDCC+MPPI / analytic_mpc | None | Online MPPI over an approximate FDCC/kinematic model then FDCC | cutting_world_model/shared_baselines.py |

MPC is the receding-horizon control framework; MPPI is the optimizer used by
both of our planners. Nonlearned does not mean model-free or calibration-free.
The original SliceIt TensorFlow/tf2rl runner is historical reproduction, not the
current SB3 baseline.

## Task version

All four controllers use pose_depth_v3: pose +100 with failure precedence,
geometric-depth shaping, analytic imagined survival, fresh-sample force checks
and the `project_continue_v2` board boundary (TASK_FREE_CUT). WM state includes
`board_contact` and `last_wrench`; do not transplant a v2 WM/replay into v3. The
policy's observable input format is the same as in v2. Use the full task name in
logs and comparison tables (free_cut_v3 is an unrelated older separation task).

## Measured coverage by task version

This table distinguishes recorded physical tests from implemented interfaces.
It is not a performance ranking. Operational run status changes separately;
consult each run's progress/manifest rather than this table to see whether it
is still running.

| Controller | Recorded v3 evidence (latch-era, through 11-18) | Historical physical outcomes (v2 only) | Remaining current-task comparison work |
|---|---|---|---|
| WM+BPTT | Each backend: short collect/update/episode/reset/save/resume pipeline at 4dfbc0c; UR10e 9-world learning runs (integration handoff) | Two 11-16 long runs; unequal achieved interaction budgets | No v3 long run yet; no mature v3 result claimed |
| SB3 SAC | GPU9, both main b2b3ea6 and 8f0a517: 6 worlds, 240 transitions; latest smoke saved model/replay with 216 updates | Separate bounded v2 SAC run; early 264-transition smoke scored 1/3 in the old comparison | v3 smoke is not a trained baseline or matched-budget comparison |
| WM+MPPI | Compatible v3 state/analytic survival implemented and regression-covered | DiSECt 3/3 pose successes, 0/3 separated; one native-BladeMaster visual example timed out | No completed v3 physical evaluation established in the audited records; needs compatible v3 WM |
| Nonlearned FDCC+MPPI | v3 task/board-state branch implemented | DiSECt 2/3 pose successes, 0/3 separated; one native-BladeMaster visual example reached pose goal | No completed v3 physical evaluation established in the audited records |

Old comparison source:
GPU9 /dataHDD/dong/Projects/Cutting_DEV_baselines_20261116/runs/baseline_validation_20261116/.
Its analytic_mpc_eval/manifest.json explicitly records pose_depth_v2.
It completed half potato in 151 steps, prism in 137, and timed out on cucumber
at 180. Planner: horizon 4, 32 candidates, one refinement, no terminal critic.
Mean planning cost ~0.134 s/world exceeds the 0.05 s control interval.

The old WM comparison used an early checkpoint with WM105/policy0/critic0,
not a mature trained WM. Native-BladeMaster visual examples are under the same
checkout's runs/dp3_controllers_20261117; their manifests also record v2.
The analytic example completed one cucumber pose case at step 105.
The current header/default config must never imply these records were v3 tests.

## Inputs, outputs and model limitations

Policy and SAC receive 10 observable frames x 34 values plus the 8D commanded
goal (pose + speed). All four controllers use the shared six-action mapper:
two stiffness channels, two control-gain channels, blade-direction offset and
optional pitch. Pitch is disabled in the selected scene (five effective channels).
FDCC translates these actions into robot commands. No controller directly
outputs the predicted sensor wrench as its actuator command.

Planners receive q, qd, signed six-axis wrench history and controller/task history,
known target, step/arrival counters and board state. They have more state access
than the policy observation; disclose this information difference. Geometric
depth comes from kinematics, not privileged spring-break progress.

Frozen WM+MPPI uses the pinned upstream pytorch_mppi optimizer and a compatible
frozen checkpoint. Optional critic must be paired with that checkpoint. Metadata
checks cover task, robot/tool/observation/model layout. No learned weights change.

Nonlearned FDCC+MPPI uses robot kinematics, the shared mapper and FDCC, coarse
ideal joint-command tracking and constant wrench within the planning horizon.
It holds the last provided history sample, which may lag the newest sensor
sample; do not describe it as zero-delay sensing or a calibrated contact model.
It reconstructs controller state from q/qd at each surrogate step and replans
from new measurements each actual control step. It predicts neither changing
food forces nor fracture; measured physical force limits still apply. It cannot
claim predictive force safety or faithful reproduction of a cutting-MPC paper.

## Commands

All commands run from main with an explicitly generated CURRENT v3 config.
Generate both backend configs from configs/shared/scene_v1.json using the
[shared-scene procedure](../SHARED_SCENE.md). Environment count comes from the
generated config; historical launch recipes are not the current config.

    CUTTING_ENV=sac source scripts/env/cutting_env.sh
    python scripts/evaluate_pose_policy.py --method wm_policy --checkpoint V3.pt --conditions V3_DISECT.json --output OUT
    python scripts/evaluate_pose_policy.py --method sac --checkpoint SAC_V3/model.zip --conditions V3_DISECT.json --output OUT
    python scripts/evaluate_pose_policy.py --method wm_mppi --checkpoint V3.pt --conditions V3_DISECT.json --output OUT
    python scripts/evaluate_pose_policy.py --method analytic_mpc --conditions V3_DISECT.json --output OUT

For native BladeMaster replace --conditions with --blademaster-config V3_BM.json.
Each OUT is a different run directory. Analytic MPC has no checkpoint.
Training: [WM+BPTT recipe](../RUNNING.md#current-wmbptt-training-command-pose_depth_v3);
SAC: scripts/train_sac_multi.py with an explicit plan/budget and online W&B.
These templates do not launch or authorize an additional experiment.

## Cutting MPC literature and our provenance

Primary sources checked 2026-11-19:
- [DeepMPC, RSS 2015](https://www.roboticsproceedings.org/rss11/p12.html):
  learned dynamics with online predictive control, evaluated in real robotic
  food cutting. Not a no-learning controller.
- [Mitsioni et al., Data-Driven MPC for Food-Cutting, 2019](https://arxiv.org/abs/1903.03831):
  a recurrent learned dynamics model, force/torque feedback and MPC for
  velocity-controlled robots, including unseen-object evaluation.
- [Modelling and Learning Dynamics for Robotic Food-Cutting](https://arxiv.org/abs/2003.09179):
  learned dynamics with multistep training inside MPC (2020 preprint, 2021
  conference version; see the [reading note](../literature/mitsioni_2021.md)).
- Our optimizer is [UM-ARM-Lab/pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi).
  It is a general-purpose optimizer, not those papers' cutting-controller code.
  No author implementation of those three papers was verified in this audit;
  do not claim that a same-named third-party repository is their release.

No-learning MPC can use an analytical/identified model; neural-network training
is optional, but the model, parameters, objective and constraints are still
needed. Online optimization of actions is not the same as learning model weights.

Real deployment is possible in principle: retain the planner/control structure,
replace simulated feedback by measured robot/sensor state and match robot/tool/
board frames, filtering and timing. That is a deployment plan, not an already
validated result. Our current nonlearned baseline still needs model/FDCC tracking
validation and acceptable planning latency; a simulated frozen board wrench must
not replace live real-sensor readings. Learned WM+MPPI additionally needs its
predictions checked on real transitions. Replanning does not eliminate model error.

## Pre-cut vision

The visual module selects a goal before control, then motion/wrench feedback
closes the execution loop. Historical DP3 examples connect wm_policy, wm_mppi
and analytic_mpc on native BladeMaster. SAC visual handoff is not implemented.
This is separate from task-version coverage and is not a core learning claim.
See [VISION_ROUTE1](VISION_ROUTE1.md) for exact source identities and scope.
