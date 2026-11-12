> Archived 2026-11-12 from docs/METHOD.md: variants that were tested and not adopted
> (BC-only actor, linear force penalty, WM terminal-gradient variant A). The code flags still exist.

## Ablation `actor_objective=bc`

`--actor-objective bc` (requires `--actor-bc-alpha`) keeps the actor's TD3+BC
term at the same scale `mean|J| / (a h)` but drops the gradient of the imagined
objective: the H-step imagination still runs, without autograd, to supply
`mean|J|` and the imagination diagnostics. WM and critic training are
unchanged. It measures how much of a `model` run's behavior comes from the
model-based gradient rather than from behavior cloning toward replay actions.
Default: `model`. Checkpoints record the objective; checkpoints without the
field were `model`. A resume with a different objective needs `--critic-reset`.

## Variant `force_penalty=linear`

SliceIt's force term `-sigmoid(s/2 - 3)` (`s` = norm of the history-mean
`[Fx, Fz, Ty]`, mixed N and N·m units) saturates near 13–20 N: 6 N → −0.125,
10 N → −0.220, 20 N → −0.250. Above about 15 N the reward barely distinguishes
forces up to the 50 N collision limit. `--force-penalty linear
--force-penalty-scale 50` replaces only this term by `-s/50`, keeping the same
three readings, components, weights and every other reward and termination
term. DiSECt penalizes mean knife force directly; normalizing by the existing
50 N limit is our choice, not a literature value.

The setting is stored in the task metadata's `environment`
(`force_penalty`, `force_penalty_scale`); absent keys mean the source reward,
so existing checkpoints and metadata are unchanged. The real-environment reward
and the WM's analytic imagined reward read the same setting. A resume may change
only these reward keys, and only with `--critic-reset` (the critic estimates
returns of the reward it was fitted on). Frozen evaluation accepts such
checkpoints, records `policy_training_reward`, and scores every method with the
benchmark's SliceIt reward and the force/work/safety metrics.
`scripts/legacy/value_eval.py` compares a critic with returns of its own reward,
recomputing the force term of the recorded SliceIt rewards from the recorded
next observations.

Not adopted (RESULTS 2026-11-08); kept as a named variant for the record.

## Variant `wm_terminal_gradient=head` (variant A, completed analysis)

Status updated 2026-11-10: implementation retained as a named variant; its
paired control experiment is complete (RESULTS, 2026-11-09). The pre-stated
final-success criterion did not detect a clear benefit. A is not the selected
main method; the description below records its implementation.

Source route (`dynamics`): the terminal head reads `terminal_features` of the
predicted next state, so the terminal BCE of the WM fitting loss also trains the
dynamics MLP through that state (as in SuperMimic-Reproduced). With `head`, the
fitting loss (`window_loss`, `masked_window_loss`, through `fit_predict`) feeds the
terminal head the detached predicted state: the BCE trains only the head and the
state loss alone trains the dynamics. Forward values are unchanged. Policy
imagination uses `predict`/`step`, which is identical for both routes, so the
actor keeps the path action → predicted state → reward and terminal probability
(`tests/test_wm_terminal_gradient.py`). `wm_diagnostics.window_loss_components`
follows the configured route; set it to `dynamics` to measure the source BCE
gradient.

`--wm-terminal-gradient {dynamics,head}`; default `dynamics`. Checkpoints record
the route; checkpoints without the field used `dynamics`. `train_gpu.py` resumes
only with the same route; `OnlineLearner.restore(...,
change_wm_terminal_gradient=True)` starts the other route from a saved WM, used by
the fixed-data comparison in PLAN.
