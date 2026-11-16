# Free-object single cut — shared geometric-progress pose task

Updated 2026-11-16. New opt-in task **pose_depth_v2** is shared by
DiSECt and BladeMaster. The user requested removing spring-release shaping and
using BladeMaster-style downward progress. Previously trained checkpoints retain
free_cut_pose_v1 / blademaster_pose_v1; their results have not been relabelled.

## Current reward and state semantics

Let g_t be the lowest point of the recorded finite-blade rigid proxy above
the board, obtained from robot joint state and blade mounting. Let s be the
positive board stopping clearance and D = g_reset - s > 0 the per-world travel
recorded at episode reset. Define

    rho_t = max(rho_(t-1), clamp(1 - (g_t - s)/D, 0, 1))

Reset rho is zero.

- Progress reward is 10 times the increase in rho.
- Both parts of the speed penalty remain: 20 times relative speed error squared
  times (positive delta-rho + 1/horizon). Its progress factor is now **depth**,
  never released springs. Load and jerk terms retain their previous coefficients.
- Success remains +100 for prescribed pose arrival without force/workspace
  failure, with the existing arrival counter and board boundary. It does not
  require rho = 1; reaching the depth alone does not imply success.
- Force/workspace failure is -100 once; timeout ends without bootstrapping.
  Gamma remains 1; these check configs retain 180 steps.
- Retraction/re-entry into a depth already visited cannot earn progress again.
  Descending through free space does earn geometric progress: this is motion
  shaping, not a material-cutting or contact detector.
- Physical separation is independent: DiSECt reports actual residual springs;
  BladeMaster reports its native topology diagnostics. Do not report geometric
  rho = 1 as material separation.

The two backends and WM call the same depth helper at control-step endpoints.
Depth span and accumulated progress are known history fields, not learned
physical outputs; the WM derives progress analytically from predicted joints.
The dynamics inputs no longer contain spring progress in this version.
The actor observation/action interfaces are unchanged. The per-episode span
supports different objects and start heights without per-object reward weights.

This changes WM state/input/output dimensions relative to the old spring-progress
model. Old replay/WM/critic cannot be silently resumed as this task. New metadata
records task identity, backend identity and geometric progress semantics. No
long training is started by the verification configurations:
configs/free_cut/depth_three_object_check.json and
configs/blademaster/depth_three_object_check.json.

This reward change does **not** unify physical knife geometry or repair residual
material connections. In DiSECt the geometry above remains a rigid proxy for
the separately integrated material knife, as noted below. Knife geometry/mounting
unification is a separate unfinished change; neither existing backend checkpoint
is an equal-knife comparison.

## Historical pose-v1 definition (previous completed runs)

Updated 2026-11-14. The implemented task identity is `free_cut_pose_v1`.
The user authorized changing the +100 success event to prescribed pose arrival and
adding a nonpenetrating cutting-board boundary. Existing v3 running rewards are
retained in this change; it is not a replacement of the entire reward with SliceIt's.

## Completion and failure

- The prescribed goal is the captured robot-base end-effector pose, translated
  along the board normal until the lowest point of the actual finite knife is at
  the board stop height. The blade-to-tool offset and orientation are included.
  It is not a target chosen by the policy, and not the old below-board goal.
- The existing SliceIt interface checks configured axes `[0,2,4]` (x, z, pitch),
  with thresholds `[0.04 m, 0.002 m, 0.04]`; the last is the selected quaternion-vector pose-error component, not radians. Its accumulated arrival counter
  must exceed 10 while the current observation is inside those thresholds.
  This is not a new consecutive-hold requirement.
- Pose completion earns +100 only without force or workspace failure.
  Force/workspace failure takes precedence and costs -100 once, including when
  arrival happens on the same step. The environment requires reset after ending.
- Timeout is an unsuccessful finite-horizon terminal, with no completion bonus
  and no value bootstrap. Current development configuration: 180 steps, 9 s.
- All springs need NOT be released to earn pose completion. Material separation,
  residual springs, geometric coverage and sliding remain separately reported.
  Pose-success rate must not be described as a physical-separation rate.

## Board requirement

The board is the DiSECt plane y=0. The existing finite-blade swept-motion stop
prevents downward and rotational penetration; it is a software interlock, not a
calibrated compliant board-contact force model. Ordinary board arrival is not a
failure and has no added penalty. Geometric coverage is not a success gate.
Invalid penetration below -2 micrometres aborts the simulator transition; it is
not scored as an RL failure. The stop aims for +1 micrometre clearance.

The WM uses the recorded finite-blade geometry with forward kinematics to project
predicted endpoints to the board boundary. This is a rigid kinematic proxy;
physical execution checks the integrated DiSECt knife, including the swept arc.
The boundary projection runs eagerly (including its implicit BPTT derivative); the rest of the WM can use the existing compilation path. The physical/proxy gap discrepancy is measured by `scripts/check_pose_cut.py`.
This proxy must not be described as a learned board contact-force model.

## Running rewards and world model

Unchanged from v3: progress weight 10; speed weight 20 (progress and finite-time
components); load weight 1; jerk weight 0.1; gamma=1. Six-axis sensor limits remain
explicit configuration values, currently [50,50,50,5,5,5] for simulated development,
not calibrated Mini45 safety limits. See [historical equations](archive/TASK_FREE_CUT_V1_V3.md).

Existing release-progress prediction remains for running reward shaping, not
completion. No new coverage head is added. The +100 settlement is computed from
predicted pose/counter and failure conditions; the learned terminal head cannot
issue a probabilistic +100 completion reward. Its labels now mean task termination,
not separation. Hard pose/force/workspace/deadline termination is enforced in
imagination so terminal rewards and bootstrap are not repeated.

This changes the task identity and terminal semantics. Old v3 replay, critic and
completion-head weights must not silently resume as the new task. Old checkpoints
remain usable under their own version.

## Entry points and retained history

- Development configuration: `configs/free_cut/pose_cucumber.json`.
- Physical/update/checkpoint check: `scripts/check_pose_cut.py`.
- Training entry: `scripts/train_gpu.py --conditions configs/free_cut/pose_cucumber.json`;
  this file has four worlds and is not an approved long-training budget.
- Source SliceIt and v1-v3 remain for old checkpoints.
- v4 is the retired coverage-task evaluation, not a training entry.
- [PLAN](PLAN.md) lists remaining work and actual object readiness.
- [RESULTS](RESULTS.md) records physical measurements and their limits.

[SliceIt III-C.4](https://arxiv.org/html/2404.02569v1#S3.SS3.SSS4) uses target-pose
completion. The complete-spring-release criterion was our later task choice.
The new software board stop is our integration safeguard, not a claim to
reproduce SliceIt's compliant board-contact dynamics.
