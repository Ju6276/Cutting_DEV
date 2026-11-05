# Critic failure analysis — 2026-11-05

Frozen state: tag `freeze-20261105-pre-critic` (code `bc9302c` plus this
document). Read-only review of the code against SuperMimic-Reproduced
`ba61053` (`world_model.py`, `state_spec.py`, `wm_trainer.py`,
`imagination/rollout.py`, `imagination/policy_trainer.py`, `policy/nets.py`)
and of the recorded results in RESULTS. No new training was run for this analysis.

## Implementation check

The H8 autoregressive WM loss, residual integration
(`qd' = qd + dqd`, `q' = q + dt qd' + dq`), GAE recursion, SHAC-style policy
objective and critic update order match the source. No implementation error
was found in `cutting_world_model/`. Both the source and the local model pass
the termination BCE gradient through the predicted state into the dynamics.

One error-path bug was fixed in the diagnostic script
`runs/wm_frozen_fit_20261105/run.py` (outside Git): `frozen` and
`learner.optimizer_steps` were undefined in the `TimeoutError` branch when the
learner was not yet built. The branch was never reached; results are unchanged.
The executed version's md5 is recorded in that run's `PATCHES.txt`.

## Task/source mismatch

| Aspect | SuperMimic source | Cutting task | Consequence |
|---|---|---|---|
| Reward | Dense, smooth tracking terms | Step reward in [-1, -0.25]; terminal +100 / -100 | Return is dominated by terminal events |
| Terminal criterion | Height 0.25 m, tilt 0.8 | goal-z < 2 mm for > 10 cumulative steps; workspace-x 5 cm; force limits | Predicted events depend on errors below WM accuracy |
| Terminal-head input | Root height/tilt, state components | FK pose errors of predicted joints; success counter (step function) | BCE gradient acts on the predicted pose |
| Imagination length | 32 x 0.02 s = 0.64 s inside long episodes | 32 x 0.05 s = 1.6 s; a successful cut lasts 1.0 s (20 steps) | Rollouts extend past the real episode end |

## Causal chain for critic failure

Measured (RESULTS):

1. Full-episode WM position error is 5–7 mm after 100 extra WM rounds,
   larger than the 2 mm goal-z threshold; all 32 held-out successful terminals
   are missed at round 100.
2. On nonterminal imagined paths the H32 endpoint value reaches 22031; with zero
   value inputs the target maximum is 100.6 (bootstrap feedback).
3. A time-matched 13-step horizon gives target maximum 99.72 (no divergence)
   but value MAE 82 (bias remains).
4. Same WM, policy and data: critic trained on recorded physical futures has
   held-out MAE 5.17–10.70; on model futures 46–2800.

Inferred, not yet measured:

- Missed success terminals let imagined trajectories continue beyond the real
  cut into states absent from replay; critic bootstrap there feeds back.
- Between extra WM rounds 50 and 100 on fixed data, state loss rises
  (0.0156 → 0.0233) while BCE falls, and predicted goal-z drifts to
  −6.5…−16 mm. The BCE gradient through the FK terminal features may bias the
  predicted pose. The planned same-batch gradient decomposition tests this.

## Implications

The critic learns accurately from physical futures; its failure comes from
model-generated targets. Expert trajectories as additional WM data do not address
this: replay already contains 3072 successful episodes and the limiting factor
is millimetre accuracy of terminal events. Expert trajectories as tracking
references would change the reward and therefore the task.

First candidate (requires a rule change and an approved plan): fit the critic
on real transitions only and keep WM BPTT for the actor, with the critic as the
H-step bootstrap. Model-based value-expansion literature (MVE, STEVE, MAAC,
SHAC and related) must be reviewed before claiming novelty.
