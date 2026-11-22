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

## Paper comparison scope

Main comparisons: WM+BPTT, SB3 SAC and frozen WM+MPPI. The analytic planner
remains an engineering tool for offline/real-controller interface checks; it is
not a required paper baseline or a reproduction of cutting-MPC literature.

WM+MPPI is itself MPC with a learned dynamics model. Use the same compatible WM
checkpoint when isolating online planning versus learned-policy execution.
Disclose planner state access, optional terminal critic, sample/horizon budget,
latency and actual training interactions. Equal wall time is not equal data or
equal optimizer updates. SAC provides the model-free RL comparison.

The analytic model holds a supplied historical wrench during each candidate
rollout. A comparison against it would isolate that particular approximation,
not establish superiority over all analytical or adaptive MPC.

## Verified execution and remaining experiments

Both material backends have passed actual five-food CUDA SAC/WM+MPPI short
execution, learning/save/load and selected-reset checks. The main algorithms
have also entered joint training; do not describe current V3 as untrained.
Recorded scope, paths and fixed run counts are in
[RESULTS](../RESULTS.md#ur10e-baseline-verification).
Current long-run snapshots are in
[RESULTS](../RESULTS.md#ur10e-long-run-snapshot-2026-11-22).

The analytic MPC tare bug is fixed. Both backends passed the post-fix short
physical planning/observe/reset check. Its older full-episode results predate
that fix and are not post-fix performance evidence. No held-out controller
ranking or real-time hardware acceptance follows from an interface pass.

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
For the current UR10e five-food task, generate both backend configs from
configs/shared/ur10e_five_food_v1.json using the
[shared-scene procedure](../SHARED_SCENE.md). Environment count comes from the
generated config; historical launch recipes are not the current config.

    export CUTTING_ENV=sac
    source scripts/env/cutting_env.sh
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
validation and acceptable planning latency; live real-sensor readings must continue after stopping; the current simulation
has no board reaction model. Learned WM+MPPI additionally needs its
predictions checked on real transitions. Replanning does not eliminate model error.

## Pre-cut vision

The visual module selects a goal before control, then motion/wrench feedback
closes the execution loop. Historical DP3 examples connect wm_policy, wm_mppi
and analytic_mpc on native BladeMaster. SAC visual handoff is not implemented.
This is separate from task-version coverage and is not a core learning claim.
See [VISION_ROUTE1](VISION_ROUTE1.md) for exact source identities and scope.
