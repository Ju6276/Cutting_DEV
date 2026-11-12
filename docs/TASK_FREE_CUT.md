# Free-object single cut: reward and world-model implementation (proposal)

Status 2026-11-12: **proposal, not approved, not implemented.** Task decisions are in
[PLAN](PLAN.md); evidence in [RESULTS](RESULTS.md). Weights below are placeholders to be fixed by
the preference check in section 4, not tuned values.

## 1. Quantities

Per control step t (0.05 s):
- rho_t in [0, 1]: fraction of cut springs released (simulation-internal; 1 = separated on scenes
  that passed the topology check).
- v_t: downward speed of the knife; v*: commanded feed speed (task input, 1-5 cm/s).
- u_t in [0, inf): limit utilisation, max over the six sensor components of |w_i| / limit_i, using
  the within-step peak (u_t > 1 is the force-limit event).
- a_t: policy action.

## 2. Episode settlement

| Event | Episode ends | Terminal reward | Counted as |
|---|---|---|---|
| Separated (rho = 1), no limit event in the episode | yes | +R_s | task success |
| Force-limit event (u > 1), including the step in which rho reaches 1 | yes | -R_f | failure; separation recorded separately |
| Workspace bound crossed (explicit geometric bounds of the knife pose) | yes | -R_f | failure |
| Time limit T | yes; **terminal, no bootstrap** | 0 | incomplete (not a success) |

The task is one cut within a time limit, so nothing is earned after T: the time limit is a true
terminal state for the critic and for imagination (the remaining time is part of the critic input or
the episode is short enough for the step count in the state to identify it; to be fixed at
implementation). A step that both separates and violates the limit is a failure (the limit takes precedence, as in
the existing `success & ~collision`). Object displacement is logged and has no reward term.

## 3. Step reward

r_t = w_p * (rho_t - rho_{t-1})                                   progress; sums to w_p over a full cut
    - w_v * [ over_t + (1 - min(u_t, 1)) * under_t ]              feed-speed tracking
    - w_f * min(u_t, 1)                                           load
    - w_j * jerk_t                                                existing smoothness term

with over_t = tanh(max(v_t - v*, 0) / v*) and under_t = tanh(max(v* - v_t, 0) / v*).
Moving faster than commanded is always penalised; moving slower is penalised less as the load
approaches the limit, so slowing down under resistance does not conflict with the limit. There is
no separate step cost and no penalty on lateral (sawing) motion beyond the smoothness term.

Time scales (design conditions, **not guarantees**): with discount gamma, the worst discounted sum of
step penalties over the remaining time, plus what an unfinished episode forgoes, has to stay below
the discounted failure penalty, otherwise failing early can pay; and a completed cut has to beat
stalling until the time limit. These depend on gamma, R_s, R_f, T and the critic's terminal handling
together, so they are evaluated numerically in section 4 for the chosen values instead of being
asserted from one inequality.

The load-dependent relief of the under-speed penalty is a **candidate**: it could reward holding a
high load to cancel the speed penalty. The preference check compares it with the plain form
(w_v * [over_t + under_t]) and with a trajectory that deliberately presses hard and slowly.

## 4. Reward preference check before any training

Fixed trajectories on the representative scene, with the force limit active, total discounted
return computed by the task code: press at v*; press + sawing at v*; press at several times v*;
stall above the object; stall in the material; retract mid-cut; deliberate limit violation; time-out.
Required ordering: completed cuts that respect the speed command and the limit rank above the
others; fast impact, stalling, retraction and deliberate failure have no advantage. The check can
exclude gross errors; it cannot prove the absence of other reward exploits.

## 5. World-model implementation

The policy is trained on imagined rollouts, so every reward term must be computable from the
predicted state:
- v_t (from predicted joint velocity through the kinematics) and u_t (from the predicted
  within-step peak wrench) already are.
- **rho is added to the world-model state as one predicted scalar.** The model predicts a
  non-negative increment that cannot exceed the remaining fraction: rho' = rho + (1 - rho) * s with
  s in [0, 1], so rho stays in [0, 1] and is monotone (damage is irreversible). It is supervised by
  the simulator value in recorded transitions. The imagined progress reward is the realised increase
  rho' - rho, so the total imagined progress reward of a rollout is bounded by w_p * (1 - rho_0) and
  the model cannot keep paying out progress; it is differentiable with respect to the actions.
- Separation in imagination uses the predicted rho through the existing termination mechanism
  (learned terminal head, fitted on the recorded separation label); the force-limit event uses the
  predicted peak wrench as now.
- The policy input is unchanged: it does not receive rho, the object identity or material.
  rho is a training signal available in simulation only; deployment needs the policy alone.

Checks before training with it: prediction error of rho (one step and over the actor horizon) on
held-out rollouts, and that the imagined and physical returns of the section 4 trajectories agree
in ordering.

## 6. Open risks stated up front

- Simulated forces are unvalidated for free and held objects alike. If real free cutting turns out
  to need little force, that is a fact about the task; the paper's emphasis on the force limit is
  then reconsidered, and the simulation is not tuned to be harder.
- The first training on this task is an engineering check of the pipeline (reset, reward, WM
  progress, termination), not evidence of an algorithmic advantage.
- rho is a simulator-internal label; a real-robot stop signal is a separate later question.
