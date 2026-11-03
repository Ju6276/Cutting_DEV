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
- **3 — SuperMimic + MuJoCo Warp + DiSECt**: `gpu_wm_first_20261103` runs
  on A6000 GPU 1 with 1024 environments, fresh weights, 500 world-model-only
  rounds and a 24-hour budget. The source vector-step schedule and 1000 replay
  rows per environment are retained. Joint learning starts on WM round 501.
  W&B [dm6e5jan](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/dm6e5jan).
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
   failure-priority reward branch are repaired. WM-first phase counters, partial
   optimizer progress, checkpoint compatibility and both levels of Gazebo
   exception cleanup are implemented and checked.
2. Isolate force prediction and terminal-head inconsistency before changing the
   learning method. On 735 recorded transitions, substituting measured force
   features reduces missed terminal labels from 14 to 3; the remaining error
   includes predicted pose/counter features and classifier consistency.
3. Run the WM-first candidate through 500 model rounds and joint learning. Compare
   force/termination prediction and critic targets, then evaluate a frozen policy
   in physical episodes with the original task metrics. Startup has passed a full
   source-size WM update on recorded transitions; task learning is the running experiment.
4. Frozen SAC evaluation completed ten episodes: 10 successes, mean return
   91.02107, zero optimizer updates. The evaluation fallthrough is repaired;
   dispatch/lifecycle tests cover normal exit and exceptions.

Implementation differences are in METHOD; commands and numbers are in RESULTS.

## External implementation evidence

[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym/tree/0171f5eadee50492d6ff30fa04b55e16cc11b61e)
provides full-update equivalence checks, independent physical evaluation and
500 world-model-only iterations; all three are now applied here. Its RunningNorm
critic and analytic soft termination are separate method choices. The current
candidate keeps the Cuboi learned head and static scales. SliceIt physical
thresholds and reward remain fixed. Analytic termination alone cannot restore
a force-limit event that the dynamics model failed to predict.
