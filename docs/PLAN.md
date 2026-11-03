# Current plan

## Numbered runs

- **1 — SAC + Gazebo + DiSECt**: resumed training in SliceIt
  `runs/sac_resume_20261103`, loading the retained TensorFlow checkpoint and
  replay buffer. The source run endpoint remains 100100 environment steps with
  the existing 72-hour execution deadline.
  W&B [xofe4ps8](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/xofe4ps8).
- **2 — SuperMimic + Gazebo + DiSECt**: stopped at iteration 872/update 316
  after critic divergence. Frozen episode return and bootstrap probes identify
  large value bias and terminal-head errors. The relative causal contributions
  remain to be established; single-world data reuse is a measured regime difference.
- **3 — SuperMimic + MuJoCo Warp + DiSECt**: stopped at vector step 122,
  124928 physical transitions, zero learner updates and zero completed episodes.
  The transition-rescaled schedule is removed. The implementation now uses the
  pinned source vector-step cadence and 1000 replay rows per environment.
- Run 1 training stays on ROS 11321/Gazebo 11355. Frozen SAC evaluation uses
  separate ROS 11323/Gazebo 11357 and the source ten-episode deterministic evaluator.

## Current implementation

The world model predicts 36 coordinates: joint position/velocity, observed wrench
history and interval peak wrench. Ten frames supply 480 physical/command inputs.
The 860-coordinate rollout state also retains analytic observation/reward history
and success count. Particles remain in DiSECt and recorded data. Source widths,
effective batches 2048/4096, eight-step model fitting, 32-step BPTT, learning rates,
update ratio and task thresholds are unchanged. Microbatch 64 controls storage.

Both runs use the repaired post-step and zero-time reset joint snapshots.
Shutdown independently saves artifacts, closes the environment within wall-time
bounds and finishes W&B; a run-specific guard cleans up remaining child processes.
The completed SAC task check had 31 transitions, 30 updates, return 84.00210,
reset pose/history errors 0 and force-read errors 0. Old failed SAC checkpoint
files were deleted; diagnostic logs and replay remain. New world-model metadata
uses `synchronous_paused_joint_snapshot_v1` and rejects incompatible recordings
or resumed checkpoints.

## Next work

1. Source vector-step cadence and critic sample ordering are restored. Full/split
   update comparison and independent review pass; the frozen SAC exit and optional
   failure-priority reward branch are repaired.
2. Isolate force prediction and terminal-head inconsistency before changing the
   learning method. On 735 recorded transitions, substituting measured force
   features reduces missed terminal labels from 14 to 3; the remaining error
   includes predicted pose/counter features and classifier consistency.
3. Check the repaired connected model in physical episodes, then evaluate sustained
   learning with a frozen policy and the original task metrics. Short numerical
   checks establish implementation behavior; completed physical episodes establish
   task outcomes.
4. Frozen SAC evaluation completed ten episodes: 10 successes, mean return
   91.02107, zero optimizer updates. The evaluation fallthrough is repaired;
   dispatch/lifecycle tests cover normal exit and exceptions.

Implementation differences are in METHOD; commands and numbers are in RESULTS.

## External implementation evidence

[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym/tree/0171f5eadee50492d6ff30fa04b55e16cc11b61e)
provides full-update equivalence checks and independent physical evaluation;
both methods are applied here. Its training loop has 500 world-model-only
iterations, its critic uses RunningNorm, and imagined termination is an analytic
sigmoid around task thresholds. These algorithm changes are not in this cutting
candidate. Compare them separately against the reproduced force/terminal failure;
keep SliceIt's physical thresholds and reward fixed. Analytic termination alone
cannot restore a force-limit event that the dynamics model failed to predict.
