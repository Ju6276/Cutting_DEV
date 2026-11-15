> Historical record retained for audit; not the current launch plan. See [PLAN](../PLAN.md).

# Six-axis sensor-wrench observation and replay audit

## Interface

The SliceIt-derived default remains unchanged: `target_dims=[0,2,4]` selects
three pose/velocity/jerk axes and, historically, three wrench axes. This was an
inherited interface, not a numerical bug. New runs can explicitly select:

```json
"force_observation": "sensor_wrench6_v1"
```

This is a top-level multi-condition or BladeMaster configuration key, independent
of `task`, `target_dims`, action mapping, force limits and reward. It gives the
policy all six sensor-frame wrench components (Fx,Fy,Fz,Tx,Ty,Tz), normalized by
the corresponding six force/torque limits. It does not add camera, material,
object identity or cutting-progress observations.

Per-frame layout is pose error 3, velocity 3, jerk 3, previous action 6, wrench
history 18 (six axes times three samples), and remaining time 1: 34 values rather
than 25. Policy history remains ten frames, term-major. The source task without
time can use 33 rather than 24 values. Original modes retain identical layouts.

The world model already predicts all six wrench components; their dynamics
fields and loss remain unchanged. Its analytically reconstructed observation now
includes all six axes. The compact state and critic gain the additional observed
history values, and the policy decoder input grows accordingly. Goal stays
8-dimensional (pose plus speed), and output stays six actions.

## Compatibility and semantic checks

New task metadata saves `policy_observation_revision=sensor_wrench6_v1` and
`environment.force_observation_dims=[0,1,2,3,4,5]`. Existing exact task comparison
rejects resuming or pairing old three-axis checkpoints with this interface.
The state constructor also rejects missing revision tags or mismatched saved
observation width. This change is for fresh, explicitly versioned runs, not
implicit weight padding or silently resuming the old policy.

Pose success, velocity/jerk indices, workspace and force-limit checks do not
change. The old force reward selects its original target axes even when the
policy sees six; free-cut load cost already uses the six-component physical
peak. Force extraction uses an explicit slice before optional remaining time,
so a trailing time scalar cannot be mistaken for a wrench-history sample.
The original SliceIt SAC bridge explicitly refuses the new interface rather than
feeding a 33-value observation into its 24-value actor.

Tests check normalized order and sensitivity to the newly visible components,
identical legacy/new rewards and reward gradients, preserved time and pose
outcomes, WM reconstruction, policy shape, metadata rejection and config
forwarding. Source plus focused tests: 47 passed, 5 skipped (GPU/external-env
checks not run in this CPU-only pass). Three additional replay audits pass.
No material simulation or training was launched by these tests. Subsequent integration on GPU7 passed root425/skip5 and backend39;3072-world DiSECt and256-world BM full-learning probes also passed. See TRAINING_SCALE_20261114 for dated scope.

## Independent replay and collection review

`OnlineSchedule.replay_capacity=1000` means 1000 stored rows per world, not
1000 total transitions. `BatchedOnlineReplay.add` writes a transition-start row
per world and an extra endpoint row when that world terminates. Thus terminal
rows consume capacity too. At 768 worlds the ring holds at most 768,000 rows;
at one world it holds only 1000. Resetting one world does not reset the others.

WM batches sample 2048 eight-step windows, ten batches per update; actor updates
sample 4096 starts. Sampling is with replacement from valid rows globally.
It therefore works with fewer distinct windows but reuses data heavily in small
world-count runs. Equal batch size does not imply equal fresh-data consumption.
The unmodified source schedule is 0.4 updates per vector step regardless of
world count, so these configurations have very different update-to-interaction
ratios. A separately versioned transition-based schedule is being implemented by
the parent task; this observation branch does not alter schedule defaults.

Terminal endpoints are excluded as starts; WM windows cannot cross episodes or
the ring write pointer, and short terminal windows are padded with the final
recorded state plus a valid-step mask. Actor history is backfilled within its
own episode. New CPU tests exercise 1/3/7 worlds, different episode lengths,
multiple ring wraps and 4096-sample requests exceeding stored row counts. World,
episode and goal identities remain aligned and terminal labels occur once.
No replay-indexing bug was found in this audit. Uniform valid-row sampling is
not equal-per-episode or equal-per-object sampling; the weighting should be
reported rather than assumed.


### Reward indexing review

The current pose task calls FreeCutTask.step_reward: load uses peak_wrench and
limits directly; speed/jerk use unchanged leading observation components.
The 25/34-dimensional regression compares every returned reward component.
SliceItHeads.force_norm now uses the explicit force slice, which also corrects
its legacy behavior if passed an observation ending with time. The historical
24-dimensional SliceIt reward and its gradient remain unchanged.
