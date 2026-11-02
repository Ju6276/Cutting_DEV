# Current plan

## Numbered runs

- **1 — SAC + Gazebo + DiSECt**: stopped at 3257 steps / 3124 updates /
  133 episodes after waiting 1 s for a joint message newer than simulation time
  747.073. The failed run's checkpoint files are deleted; logs and replay remain
  for diagnosis. StepControl now returns the post-integration joint snapshot;
  complete-episode validation precedes a fresh baseline run.
  W&B `064z1a85` retains physical step and episode rewards.
- **2 — SuperMimic + Gazebo + DiSECt**: the active implementation is the
  `control_history_v2` predictor. Latest offline validation completed its 600 s
  budget (32 groups / 324 Adam steps); no connected policy training is running.
  W&B `8j3k5uuo` contains prediction metrics, separate from physical policy reward.
- MuJoCo remains a separate backend with its native state representation.

## Implemented and checked

The world model predicts 36 coordinates: joint position/velocity, observed wrench
history and interval peak wrench. Ten frames supply 480 physical/command inputs.
The 860-coordinate rollout state also retains analytic observation/reward history
and success count. Particles remain in DiSECt and recorded data. Physical/task
formulas, policy observations, source widths, optimizer, batches and horizons are
unchanged. Model metadata rejects incompatible checkpoint layouts.

Independent review and 50 passing tests cover causal histories, resets, task heads
and gradient flow (2 GPU integration checks skipped). Original task equations
match all 182 recorded transitions: reward error <=1.60e-14, terminal mismatches 0.
Final trained 8/32-step gradients match central finite differences with unchanged
branches. The repeated 600-second experiment removes the v1 explosive feedback.

## Remaining work

The current data has 137 training transitions and one 45-transition held-out
cut. At 32 steps, held-out FK error is 15.69 mm and peak-force RMSE 8.97 N;
the single force-limit event is missed. These physical errors still exceed the
constant-velocity/force-persistence prior. No physical policy improvement has been
measured for the compact model.

1. Keep v2 fixed while expanding complete-episode data with the existing source
   action/noise setup; retain whole held-out episodes including success and force
   termination. Report terminal counts/probabilities and physical-unit errors.
2. Continue source world-model learning and compare 1/8/32-step prediction curves
   against persistence. Use training/held-out differences to separate fitting from
   coverage problems before another architecture or optimizer change.
3. Once predictions support the policy horizon, run bounded connected BPTT learning
   and held-out physical policy evaluation, logging actual reward and terminal
   bootstrap separately. Check new cut histories before claiming material-memory
   generalization.

Implementation differences and input scales are in METHOD; experiment commands,
commits and numbers are in RESULTS. All runs stay under `/dataHDD/dong`.
