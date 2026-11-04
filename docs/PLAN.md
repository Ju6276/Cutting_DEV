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

## Active work — fixed environment

1. **Frozen WM fit, A6000 GPU 0.** `runs/wm_frozen_fit_20261105` starts from
   policy62/WM562. Freeze policy, critic and their optimizer state. Collect four
   complete episodes in each of 32 environments using the existing 0.1 action
   noise; groups 0–23 supply 96 training episodes and groups 24–31 supply 32
   held-out episodes. The collected split has 93/96 training successes and three
   workspace failures; all 32 held-out episodes succeed. Fit the WM for at most
   100 rounds or 1500 seconds, using the existing H8 objective, effective batch
   2048, ten batches per round and Adam state. The total hard budget is 2700
   seconds. Compare H1/H8/full recorded-action error and first terminal event
   at rounds 0/10/25/50/100; verify policy/critic tensors stay identical.
   W&B [7jnyx9n4](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/7jnyx9n4).
2. **One evaluation environment.** `public-prism-ur5e-v1` fixes the existing
   UR5e + MuJoCo Warp + FDCC + DiSECt assets, reset and task. Constant control,
   SAC and the current WM policy use `scripts/evaluate_benchmark.py`. The old
   single-policy evaluator is replaced by this shared runner. Commands, metric
   definitions and the fixed configuration are in [BENCHMARK](BENCHMARK.md).
3. **Original SAC adapter.** Keep SliceIt's tf2rl implementation, Wave actor,
   batch 1024, learning rate 0.01, auto-alpha, seed 123456, warmup 1000,
   endpoint 100100 and checkpoint interval 5000. A local process bridge joins
   its Python 3.8 learner to the Python 3.12 GPU environment. First validate
   the transition/terminal/reset chain and frozen complete episodes; then train
   SAC from scratch in this same environment for the algorithm comparison.
4. **Critic diagnosis after model fit.** The last full-return MAE is 153.42.
   The old WM predicts a workspace failure on a physically successful cut.
   Check whether sustained fixed-data fitting corrects that error on both groups.
   If both groups remain poor, inspect the loss/representation with the exact
   failing trajectories; if only held-out predictions fail, examine data coverage.
   Next compare a fresh critic to actual complete physical returns before joint
   policy/critic updates. Preserve task rewards, thresholds and source optimizer
   settings throughout.
5. **Model-based comparison.** Connect DiSECt's knife trajectory optimizer to
   the same robot/controller evaluation. Its free-knife optimization results
   remain a separate experiment until that adapter executes the same task.

No new long training is launched during these bounded checks. Run1 keeps its
existing Gazebo process, endpoint and wall budget. Current machine routes and
local environment paths are in `runs/operations/MACHINES.md` on the Mac;
connection details and credentials stay outside Git.

## External implementation evidence

[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym/tree/0171f5eadee50492d6ff30fa04b55e16cc11b61e)
provides full-update equivalence checks, independent physical evaluation and
500 world-model-only iterations; all three are now applied here. Its RunningNorm
critic and analytic soft termination are separate method choices. The current
candidate keeps the Cuboi learned head and static scales. SliceIt physical
thresholds and reward remain fixed. Analytic termination alone cannot restore
a force-limit event that the dynamics model failed to predict.
