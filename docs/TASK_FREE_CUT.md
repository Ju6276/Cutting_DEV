# Free-object single-cut task — free_cut_v3

Current implementation and the authorized 2026-11-13 continuation use
[short_train_cucumber.json](../configs/free_cut/short_train_cucumber.json).
This is the task contract; [PLAN](PLAN.md) owns run budgets and [METHOD](METHOD.md)
owns the learner. No task/reward changes are introduced by the documentation cleanup.

## Scene, signals and inputs

- One cut of a freely placed object; passive sliding is logged, not thresholded or
  penalized. Pressing and motion along the blade are allowed, without a preference
  for sawing. There is no multi-cut or active-rolling objective.
- Accepted assets have exactly two FEM components joined only by nonempty cut
  springs, no fixed particles and no residual cut-spring damping. On these assets,
  all cut-spring stiffnesses zero means no remaining material connection.
- rho in [0,1] is the released-cut-spring fraction. Only rho=1 is separation;
  98% remains a historical diagnostic, not task success.
- Each policy step is dt=0.05 s. Commanded feed speed v* is positive, 1–5 cm/s.
- Wrench utilization u=max_i |w_i|/limit_i uses within-step sensor-component peaks
  at the 2 ms coupling samples. The current simulation limits are
  [50,50,50 N, 5,5,5 N m], not measured Mini45 limits.
- Material net-force norm, sensor wrench and limit utilization are different
  recorded quantities; do not compare the force norm directly to an axis limit.
- Current policy input: ten frames of 25 observations (legacy 24 plus remaining
  time), target pose plus speed (goal dimension 8), output six actions.
  Only wrench components [0,2,4] enter the policy observation; six-axis expansion
  is deferred. The WM predicts all six wrench components and also receives rho.
  Object/material identity and simulator rho are not policy inputs.

## Episode settlement

Precedence is **force/workspace failure > separation success > timeout**.

| Event | Terminal contribution | Episode label |
|---|---:|---|
| All springs released, no force/workspace failure | +100 | Success |
| Sensor-component limit or EE workspace crossed, including a separating step | -100 once | Failure; separation also recorded |
| Deadline without success/failure | 0 | Incomplete; true termination, no bootstrap |

The entering terminal transition retains its running reward. Later rewards and
value bootstrap are zero. Success at the deadline is permitted; simultaneous
force/workspace failure suppresses success. Each world ends/resets independently.

The current robot-base **EE** workspace is x=[0.038,0.238], y=[0.405,0.605],
z=[0.096,0.269] m. It is a configured robot workspace, not an object-sliding bound
or a certified real-robot safety envelope.

## Reward and horizon

Let v be signed downward knife speed, H the task deadline, and j the existing
normalized jerk measure capped at 1. Define:

    e = ((v - v*) / (abs(v) + v*))^2
    delta_rho = rho_next - rho
    speed_cost = w_speed * e * (max(delta_rho, 0) + 1/H)
    r_step = w_progress * delta_rho - speed_cost
             - dt * (w_load * min(u,1) + w_jerk * min(j,1))

Add the terminal contribution above. Current weights are progress=10, speed=20,
load=1, jerk=0.1, success=failure=100; gamma=1. These are reward design units,
not fitted material quantities. No pose-goal bonus, constant step cost, sliding
penalty or explicit sawing penalty is added.

Speed mismatch is charged per released fraction as well as per elapsed fraction
of the deadline. This prevents fast cuts avoiding the speed cost solely by taking
fewer steps. Reciprocal forward speed ratios have equal mismatch. This does not
guarantee that a learned predictor or policy optimizes the physical reward correctly.

The current training/evaluation horizon is **H=180 steps (9 simulated seconds)**;
its geometric/command-speed derivation is in the training config. Historical
`review_v3.json` uses H=100 for reference checks and must not silently replace
the production task or be used with its checkpoints.

The code requires success/failure magnitudes to exceed w_progress + P, where

    P = 2*w_speed + H*dt*(w_load+w_jerk).

For H=180, P=49.9 and 100>59.9. Under the bounded task signals, running penalties
cannot overturn the intended ordering of successful, incomplete and deliberately
failed episodes. Total physical episode return is at most 110; a tighter
remaining-return upper bound is 100+10*(1-rho). These are reward bounds, not critic
calibration or force-safety guarantees.

## World-model progress and completion

The WM predicts a fraction f=sigmoid(z) of the remaining material:

    rho_next = rho + (1-rho)*f
    p_complete = f * sigmoid(raw_completion_head)

The completed-cut label is physical separation, not the union of failure/timeouts.
For an already-complete input, f=0. Completion reward, survival and completion BCE
use the corresponding stable logit. No invented near-completion threshold is used.

For differentiable rollouts, f is read directly from the network sigmoid, not
reconstructed by subtracting rounded states near rho=1. Progress is monotone and
bounded; total progress reward cannot exceed the remaining progress budget.
The fraction output initializes to 0.001; other physical residuals initialize
to zero. This is network initialization, not a physical damage rate.

The completion head reads predicted force/workspace margins, progress and
remaining time. An optional head-only fitting variant detaches the progress path
for BCE; the current continuation uses `wm_terminal_gradient=dynamics`.
BPTT through the predicted state/reward/survival remains enabled during policy
learning. The probability bound can hold while completion prediction is wrong;
current diagnostics measure that error explicitly.

Predicted force/workspace failures and deadlines absorb imagined rollouts.
Same-step force/workspace failure contributes zero completion reward. On the
deadline step, completion reward is still allowed if no failure occurs, then the
trajectory ends. The diagnostic must follow this same priority.

## Reset and evidence

Material is sampled before each selected world's settling. Those worlds restore
the immutable export, settle for 0.75 s with their own material and a stationary
non-contacting knife, and retain the resulting velocities. Every material
substep contributes to the contact check. Controller/history clocks then reset;
other worlds stay unchanged. Settling work is not counted as policy transitions.

Scratch batches are padded to powers of two, with total cached capacity limited
to the next power of two of the parent batch; a 32-substep CUDA graph reduces
dispatch overhead. This bounds scratch capacity, not all GPU memory or reset cost.

Measured reference rankings, reset cost, numerical repeatability limits and the
isolated low-limit abort probe are in [2026-11-12 results](archive/RESULTS_20261111_20261112.md).
Current learned-policy results are in [RESULTS](RESULTS.md). The references do not
prove an absence of all reward exploits or real-world material accuracy.

## Historical versions

v1 used gamma=.99 and a time-integrated speed penalty; its preference check failed.
v2 added gamma=1/time/workspace but the speed difference remained too small.
Their configs/code stay for checkpoint reproduction; the old equations and review
handoffs are available from [the archive index](archive/README.md). Do not bypass
metadata/network-shape checks to resume them as v3.

Real-robot stopping/separation detection and material/friction calibration remain
separate work. Simulator rho is privileged training information; a deployed
policy does not require it, but a planner that reads the WM state needs a declared
progress initialization/estimation route.
