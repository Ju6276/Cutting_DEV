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
- **3 — SuperMimic + MuJoCo Warp + DiSECt**: bounded validation finished on
  A6000 GPU 0 at policy62/WM562. `critic_mask_validation_20261104` collects
  3409 complete episodes and samples 2433 collision terminal labels, including
  the previously excluded one/two-step failures. Frozen evaluation succeeds
  8/8 in 20 steps; value MAE remains high at 153.418925. No process remains
  from this run.
  W&B [5h7xrk7c](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/5h7xrk7c).
- Run 1 training stays on ROS 11321/Gazebo 11355. Frozen SAC evaluation uses
  separate ROS 11323/Gazebo 11357 and the source ten-episode deterministic evaluator.

## Current implementation

The world model predicts 36 coordinates: joint position/velocity, observed wrench
history and interval peak wrench. Ten frames supply 480 physical/command inputs.
The 860-coordinate rollout state also retains analytic observation/reward history
and success count. Particles remain in DiSECt and recorded data. Source widths,
effective batches 2048/4096, maximum eight-step model horizon, 32-step BPTT, learning rates,
update ratio and task thresholds are unchanged. Microbatch 64 controls storage.

Gazebo runs 1 and 2 use the repaired post-step and zero-time reset joint snapshots.
Shutdown independently saves artifacts, closes the environment within wall-time
bounds and finishes W&B; a run-specific guard cleans up remaining child processes.
The completed SAC task check had 31 transitions, 30 updates, return 84.00210,
reset pose/history errors 0 and force-read errors 0. Old failed SAC checkpoint
files were deleted; diagnostic logs and replay remain. New world-model metadata
uses `synchronous_paused_joint_snapshot_v1` and rejects incompatible recordings
or resumed checkpoints.

## Next work

1. The shared run2/run3 learner applies the entering terminal reward once,
   then freezes state and zeros later rewards/bootstrap. Same-checkpoint
   imagination confirms this behavior.
2. Short-sequence replay is repaired and exercised by real learning. Full H8
   loss/gradients and mixed-length updates match their references; the new run
   uses 2433 collision terminal labels from real one/two-step failures. Shared
   code covers run2/run3; the completed physical validation uses run3.
3. Resolve multi-step WM joint/pose error. On the same physical policy62 actions,
   step15 one-step position error is below 0.31 mm per coordinate, but a rollout
   from the initial state reaches roughly 24–26 mm error in x/z and falsely
   crosses the workspace boundary. Audit joint integration, normalization and
   history inputs against the source using the saved action/state records.
   Preserve critic equations, reward, thresholds and optimizer settings; verify
   the next repair with fixed-policy model rollouts and fresh complete physical
   returns. Current complete-return MAE is 153.418925 after ten new updates.
4. Run1's completed frozen SAC evaluation has ten successes, mean return
   91.02107 and zero optimizer updates. Keep its training independent of run2/3
   model and critic changes.

Implementation differences are in METHOD; commands and numbers are in RESULTS.

## External implementation evidence

[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym/tree/0171f5eadee50492d6ff30fa04b55e16cc11b61e)
provides full-update equivalence checks, independent physical evaluation and
500 world-model-only iterations; all three are now applied here. Its RunningNorm
critic and analytic soft termination are separate method choices. The current
candidate keeps the Cuboi learned head and static scales. SliceIt physical
thresholds and reward remain fixed. Analytic termination alone cannot restore
a force-limit event that the dynamics model failed to predict.
