# Current plan

## Numbered runs

- **1 — SAC + Gazebo + DiSECt**: fresh training in SliceIt
  `runs/sac_long_20261103`, source endpoint 100100 environment steps and the
  existing 72-hour execution deadline. W&B [xqmkhjbc](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/xqmkhjbc).
- **2 — SuperMimic + Gazebo + DiSECt**: fresh connected `control_history_v2`
  training in `runs/control_history_online_20261103`. The source loop collects
  physical transitions, then alternates world-model and BPTT policy/critic
  updates after 40 warmup steps, at 0.4 update groups per environment step.
  W&B [9lptr72s](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/9lptr72s).
  The source endpoint is 320000 iterations; this execution has a six-hour budget
  and saves `data/last.pt` at budget completion, interruption or a caught failure.
  The six-hour budget bounds runtime, rather than defining convergence.
- Both runs share the RTX 4090, with separate ROS/Gazebo masters
  (1: 11321/11355; 2: 11322/11356). MuJoCo remains a separate backend.

## Current implementation

The world model predicts 36 coordinates: joint position/velocity, observed wrench
history and interval peak wrench. Ten frames supply 480 physical/command inputs.
The 860-coordinate rollout state also retains analytic observation/reward history
and success count. Particles remain in DiSECt and recorded data. Source widths,
effective batches 2048/4096, eight-step model fitting, 32-step BPTT, learning rates,
update ratio and task thresholds are unchanged. Microbatch 64 controls storage.

Both runs use the repaired post-step and zero-time reset joint snapshots.
The completed SAC task check had 31 transitions, 30 updates, return 84.00210,
reset pose/history errors 0 and force-read errors 0. Old failed SAC checkpoint
files were deleted; diagnostic logs and replay remain. New world-model metadata
uses `synchronous_paused_joint_snapshot_v1` and rejects incompatible recordings
or resumed checkpoints.

## Next work

1. Continue the unchanged online loop, recording physical step/episode rewards,
   success/collision outcomes, world-model losses and BPTT/critic metrics. A few
   weak episodes or short-term oscillations do not trigger model or threshold changes.
2. Inspect sustained prediction and physical-policy curves. Use separate complete
   episodes for 1/8/32-step error and terminal prediction evaluation; these checks
   explain learning behavior and do not impose an added training admission gate.
3. Evaluate saved policies on complete physical cuts and compare with SAC under
   the same task. Report actual task returns and success separately from imagined
   returns. Diagnose concrete non-finite/state-alignment failures if they occur.

Implementation differences and input scales are in METHOD; experiment commands,
commits and numbers are in RESULTS. All remote runs stay under `/dataHDD/dong`.
