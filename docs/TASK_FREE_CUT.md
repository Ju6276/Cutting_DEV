# Free-object single cut: free_cut_v1 implementation

Status 2026-11-12: **implementation authorized; training not authorized.** Task decisions are in
[PLAN](PLAN.md); evidence in [RESULTS](RESULTS.md). Explicit provisional weights for interface checks live in configs/free_cut/interface_check.json.
They are not approved training weights; the preference check can reject them.

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
terminal state for the critic and for imagination (the step count is included in the WM/critic state; implemented analytically). A step that both separates and violates the limit is a failure (the limit takes precedence, as in
the existing `success & ~collision`). Object displacement is logged and has no reward term.

## 3. Step reward

r_t = w_p * (rho_t - rho_{t-1})                                   progress; sums to w_p over a full cut
    - dt * w_v * tanh(abs(v_t - v*) / v*)                        feed-speed tracking
    - dt * w_f * min(u_t, 1)                                      load
    - dt * w_j * min(jerk_t, 1)                                   existing normalized jerk

The initial implementation uses the **uncoupled** speed error. All running penalties are rates
multiplied by the 0.05 s control duration. Progress and terminal rewards are event quantities.
No position-goal reward, per-step constant, sliding penalty or explicit sawing penalty is added.
The original observation's jerk norm (/500) is retained and capped at one for this reward.

Time scales (design conditions, **not guarantees**): with discount gamma, the worst discounted sum of
step penalties over the remaining time, plus what an unfinished episode forgoes, has to stay below
the discounted failure penalty, otherwise failing early can pay; and a completed cut has to beat
stalling until the time limit. These depend on gamma, R_s, R_f, T and the critic's terminal handling
together, so they are evaluated numerically in section 4 for the chosen values instead of being
asserted from one inequality.

Load-dependent relief of the under-speed penalty is deferred. It is not implemented or selected
by these checks. No preference for sawing over pressing is required.

## 4. Reward preference check before a training plan

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
- Separation in imagination uses a completion head fitted on recorded **separation** labels,
  not the union of failures/timeouts. Features: six force margins, three workspace margins,
  predicted rho and remaining time (11 features, concatenated with their absolute values).
  Expected completion reward is R_s * sigmoid(logit) unless a hard failure occurs. Survival
  uses the same probability, so a hypothetical completion is not paid in full repeatedly.
  There is no exact predicted-rho==1 check. Failure/deadline hard absorption overrides learned
  survival; the last transition keeps its reward once, then future reward/value is zero.
- The policy input is unchanged: it does not receive rho, the object identity or material.
  rho is a training signal available in simulation only; deployment needs the policy alone.

Checks before training with it: prediction error of rho (one step and over the actor horizon) on
held-out rollouts, and that the imagined and physical returns of the section 4 trajectories agree
in ordering.

## 6. Open risks stated up front

- First interface-check weights failed the speed preference: faster-than-commanded
  pressing earned more return. These weights are not a training recommendation;
  see FREE_CUT_REVIEW and RESULTS.


- Simulated forces are unvalidated for free and held objects alike. If real free cutting turns out
  to need little force, that is a fact about the task; the paper's emphasis on the force limit is
  then reconsidered, and the simulation is not tuned to be harder.
- The first training on this task is an engineering check of the pipeline (reset, reward, WM
  progress, termination), not evidence of an algorithmic advantage.
- rho is a simulator-internal label; a real-robot stop signal is a separate later question.

## 7. Reset and entry points

- GPUEnvironment(..., task=...) selects free_cut_v1. MultiConditionEnvironment.from_config
  reads top-level task. Absent task keeps the historical SliceIt task unchanged.
- A scene is accepted only if it has no fixed particles, two FEM components, nonempty cut
  springs joining those components, and zero residual spring damping. This validates the
  supported separation label, not physical calibration.
- LocalGroup.reset randomizes material first, restores the immutable export for selected worlds,
  settles those worlds, saves their reset snapshots and resets their controller/history clocks.
  Other worlds and their saved snapshots remain unchanged. This costs physical settling time
  on each randomized reset; it is not counted as a policy transition.
- No material or geometry is tuned. Material state after settling retains its velocities.
- WM physical outputs add bounded monotone cut_progress. State history also includes step_count;
  it increments analytically and is scaled by horizon_steps for the critic. Policy observations
  remain 24 dimensions and its goal remains pose + commanded speed (8 dimensions).
- Historical checkpoints have incompatible task metadata and network shapes. Do not bypass
  compatibility checks or label them as trained on free_cut_v1.

Review entry: [FREE_CUT_REVIEW](FREE_CUT_REVIEW.md). No new policy checkpoint exists yet.
