# Free-object single cut: versioned task implementation

Status 2026-11-12: **implementation and bounded update checks authorized; long training not authorized.** Task decisions are in
[PLAN](PLAN.md); evidence in [RESULTS](RESULTS.md). The historical v1 check is configs/free_cut/interface_check.json (gamma=.99, rejected
speed preference). v2 is preserved for reproduction but its speed penalty was too weak.
The CURRENT review candidate is configs/free_cut/review_v3.json (gamma=1); section 9
defines its reward and completion bound. All review configs have explicit provisional
weights and are not approved long-training configurations.
Review evidence and bounded validation: [FREE_CUT_REPAIR](FREE_CUT_REPAIR.md).

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
| Workspace bound crossed (v2/v3: explicit robot-base XYZ bounds of the end-effector pose) | yes | -R_f | failure |
| Time limit T | yes; **terminal, no bootstrap** | 0 | incomplete (not a success) |

The task is one cut within a time limit, so nothing is earned after T: the time limit is a true
terminal state for the critic and for imagination (the step count is included in the WM/critic state; v2/v3 also give the policy normalized remaining time). A step that both separates and violates the limit is a failure (the limit takes precedence, as in
the existing `success & ~collision`). Object displacement is logged and has no reward term.

## 3. Historical v1/v2 step reward (v3 uses section 9)

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

## 5. Shared world-model implementation (v3 completion coupling in section 9)

The policy is trained on imagined rollouts, so every reward term must be computable from the
predicted state:
- v_t (from consecutive predicted joint poses through the kinematics) and u_t (from the predicted
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
- The v2 policy receives the source 24 observation values plus normalized remaining time. It does not receive rho, the object identity or material.
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

- GPUEnvironment(..., task=...) selects the explicit version in the task config. MultiConditionEnvironment.from_config
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
  are 25 dimensions in v2 (24 in v1); the goal remains pose + commanded speed (8 dimensions).
- Historical checkpoints have incompatible task metadata and network shapes. Do not bypass
  compatibility checks or label them as trained on the new free-cut version.

Review entry: [FREE_CUT_REVIEW](FREE_CUT_REVIEW.md). No new policy checkpoint exists yet.


## 8. v2 review corrections

- gamma=1 is explicit in the task. OnlineLearner uses it for the actor and all critic
  targets; explicit mismatched trainer configs are rejected. ModelRollout rejects an MPPI
  discount mismatch; the reference evaluation checks its discount against the task.
  Source/v1 defaults remain .99. Removing temporal discount does **not** remove all incentive
  to finish sooner: load/speed/jerk costs still accumulate until termination.
- The progress fraction at initialization is .001, represented by a logit bias, retaining
  a nonzero derivative. This is an initialization choice, not a physical rate or a success
  threshold. All other physical residuals retain source zero initialization.
- Completion remains a learned head, not hard-gated by an invented rho threshold.
  WM-fitting metrics compare its probability to recorded physical separation, and report
  probability exceeding predicted progress. completion_progress_report also checks
  autoregressive predictions. An architectural consistency guarantee is **not** claimed.
  Imagined event summaries are labelled as predictions, never physical calibration.
- v2 workspace uses the robot-base EE XYZ box explicitly supplied in the config.
  The review box is x=[.038,.238], y=[.405,.605], z=[.096,.269] metres. It covers the source
  initial EE (.138,.505,.189), 3 cm lifted starts and upward withdrawal; z lower is 3 cm
  below the historical endpoint (.126). This is an explicit **engineering review envelope**,
  not a calibrated robot-safety boundary or a universal box for every future asset.
  It does not impose an object sliding threshold or an orientation limit.
- Reset uses material-specific scratch batches padded to powers of two and a CUDA graph
  of 32 unchanged integrator substeps. Every substep contributes to the no-contact check.
  The knife is stationary. Duration/material draws are unchanged, velocities are not cleared,
  and only selected rows are copied back. GPU parallel sums can produce floating differences;
  numerical comparisons and cost are reported in FREE_CUT_REPAIR.


## 9. Current review candidate: free_cut_v3 (supersedes v2 for new checks)

v2's observed ranking was insufficient: 20x-speed pressing lost only 0.08 return.
Old v1/v2 configs still reproduce their equations; v3 changes task metadata and cannot
silently resume an old task checkpoint. Physical separation, six-component simulation
limits, workspace, friction and settling duration are unchanged.

Let rho be cut progress, v the signed downward speed, v* > 0 the command, H the
finite episode horizon and dt the control interval. Define

    e = ((v - v*) / (abs(v) + v*))^2                   in [0,1]
    c_speed = w_speed * e * (max(rho_next - rho, 0) + 1/H)
    r_step = w_progress * (rho_next - rho) - c_speed
             - dt * (w_load * min(utilization,1) + w_jerk * min(jerk,1))

Success/failure settlement and gamma=1 are unchanged. This is a task reward design,
not a claimed new RL method or a physically measured cost. The progress part charges
for speed mismatch on each released fraction regardless of how few steps the cut takes.
Reciprocal forward speeds have equal mismatch (3x and 1/3x); the finite-time term also
penalizes waiting/retraction. No additional sliding or sawing penalty is introduced.
The review chooses w_speed=20 explicitly: a full cut at extreme mismatch can lose up
to 20 from the progress part, plus at most 20 over the full time budget. Other review
weights remain progress=10, load=1, jerk=.1, success=failure=100. These are engineering
reward units, not calibrated physical parameters or a promise that a critic is accurate.

The running-penalty upper bound is P = 2*w_speed + H*dt*(w_load+w_jerk).
The constructor requires BOTH terminal magnitudes > w_progress + P. For the review
H=100 and dt=.05, P=45.5: successful return >=54.5, unfinished nonfailure return >=-45.5,
unfinished return <=10, force/workspace failure return <=-90. Thus a deliberate failure
cannot beat an otherwise unfinished nonfailure, and a completed nonfailure beats any
unfinished episode, under these bounded signals. This is a reward bound, not a guarantee
that a learned model/critic predicts the correct return. A longer production horizon is
validated against the same condition and must not inherit this config unquestioningly.

Completion/progress coupling in v3:

    f = clamp((rho_next-rho)/(1-rho), 0, 1)
    p_complete_this_step = f * sigmoid(raw_completion_head)

The old state is required. With no new release the completion hazard is zero up to dtype
tiny. Across an imagined rollout, cumulative completion probability is bounded by
(rho_final-rho_initial)/(1-rho_initial), apart from floating error. A weaker bound using
rho_next alone would allow repeated hazards to accumulate success at constant partial
progress; that candidate was discarded during this review. At already-complete inputs
f=0 (there is no new completion event). Physical complete rows are terminal replay targets,
not new episode starts.

WM BCE, completion reward, survival and MPPI receive the corresponding stable logit.
For differentiable prediction, f is taken directly from the network sigmoid that generated
the progress increment, NOT reconstructed by subtracting rounded float32 states near 1.
State-only reconstruction is allowed for diagnostics without state gradients. Imagination
metrics retain the actual emitted probability instead of reconstructing it afterward. This necessary consistency bound follows because a new
completion releases all remaining material. It is not a 98% threshold, nor an assumption
that mean progress equals completion probability. The head still learns physical separation
labels. It can satisfy the bound and remain inaccurate; calibration is empirical.
Head-only fitting detaches BOTH predicted and previous progress states. Policy BPTT keeps
both paths through progress, the head, reward and survival. Only dtype tiny is used for
numerically finite endpoint logits, not an experiment-defined threshold.

Reset cache: LRU eviction limits the sum of cached scratch-world capacities to the
next power of two of the parent batch size. Eviction breaks the cached-step closure
cycle so tensor/graph memory is not left waiting for Python GC. This bounds scratch
capacity, not total device memory: replay, networks, allocator reserves and graphs are
measured separately. Evictions can cause later recapture; do not claim zero reset cost.
