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
| Time limit T | yes (truncation, value bootstrapped) | 0 | incomplete |

A step that both separates and violates the limit is a failure (the limit takes precedence, as in
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

Time scales that must hold (checked in section 4, not assumed): the largest per-step penalty times
the time limit must stay below R_f, so that ending an episode by failing is never better than
continuing; w_p and R_s must make a completed cut better than stalling until the time limit.

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
  non-negative increment (monotone by construction, since damage is irreversible), supervised by
  the simulator value in recorded transitions. The imagined progress reward is the predicted
  increment and is differentiable with respect to the actions.
- Separation in imagination uses the predicted rho through the existing termination mechanism
  (learned terminal head, fitted on the recorded separation label); the force-limit event uses the
  predicted peak wrench as now.
- The policy input is unchanged: it does not receive rho, the object identity or material.
  rho is a training signal available in simulation only; deployment needs the policy alone.

Checks before training with it: prediction error of rho (one step and over the actor horizon) on
held-out rollouts, and that the imagined and physical returns of the section 4 trajectories agree
in ordering.

## 6. Open risks stated up front

- On the settled free cucumber a plain press separates the object at 1.5-2.5 N; if every training
  object behaves like this the task has no force difficulty and cannot show a method.
- A sideways offset of about 1 mm between knife plane and cut plane changes the simulated force
  several-fold (platform limit). Start poses must keep the two aligned, or this artefact becomes
  the main difficulty.
- rho is a simulator-internal label; a real-robot stop signal is a separate later question.
