# Implemented cutting method

Current candidate: free_cut_v3 with **WM + BPTT, real-replay critic targets,
actor horizon 8 and BC 0.25**. Task equations are defined only in
[TASK_FREE_CUT](TASK_FREE_CUT.md). Run settings and code ownership are in
[PLAN](PLAN.md); this file describes code, not an established method contribution.

## Source and module map

The active learner adapts SuperMimic-Reproduced `ba610534` and SliceIt's UR5e
compliant-control task. It is distinct from the separately adapted official
SuperMimic implementation. Pins, licenses and that comparison mapping:
[SOURCES](SOURCES.md).

| Component | Current implementation |
|---|---|
| State/history, FK and analytic task features | `control_state.py`, `free_cut_state.py` |
| Residual dynamics and completion head | `world_model.py` |
| Policy and critic | `policy.py` |
| WM fitting, imagination and actor/critic optimization | `trainer.py` |
| Episode-aware replay | `replay.py`, `online_replay.py`, `sequence_replay.py` |
| Collection schedule | `online_schedule.py` |
| Physical task and grouped resets | `gpu_env.py`, `multi_env.py`, `free_cut.py` |
| Robot/material coupling and kernels | `backends/mujoco/` |

All module paths in the table's first seven rows are under `cutting_world_model/`.
For previous representations, Gazebo native-state wiring, value variants and
source-only mappings, see the pinned [historical method specification](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/METHOD.md).
Those alternatives are not additions to the current learner.

## Collection and representation

MuJoCo Warp executes the UR5e; DiSECt advances the deformable material. Material
reaction is fed to the sensor/filter/compliant-control path; it is not an
additional direct external load on the MuJoCo robot. The robot responds through
the controller. The six SliceIt-derived actions set compliance/reference commands;
execution clips to the existing action bounds. The lateral command integrates
over time. This continuation adds no lateral limiter.

The policy uses ten observation frames and a goal encoder; current observations
are 25-D and goals 8-D. Decoder output has six dimensions. Encoder/decoder widths
remain 2048/1024/512/256 with a 32-D goal latent.

The WM uses control history to approximate unobserved material/controller memory.
Its physical outputs include joints, the observed six-axis wrench history,
interval peak wrench and released fraction. FK, commanded motion and derived
observation features are updated analytically. Step count/remaining time advance
deterministically. The model does not predict every particle or spring.

The WM currently reads simulator cut progress. Policy input excludes progress,
material labels and object identity. The policy sees wrench components [0,2,4],
while the limit check uses six components; six-axis policy observations are a
separate deferred interface change. Neither history nor privileged training
signals guarantee an accurate or fully observed dynamics model.

## World-model fitting

The source residual MLP uses three hidden layers of width 2048. The maximum
training window is eight steps; each WM update fits 10 minibatches of 2048
windows. Loss is normalized state error plus completion BCE on recorded physical
labels. Current fitting keeps BCE gradients into dynamics (`dynamics` option);
the head-only variant is retained separately, not silently enabled.

Replay stores clean/executed actions, reached states, rewards and episode labels,
1000 rows per world. Short completed windows are padded and masked; padding and
cross-reset transitions do not enter losses. Microbatches are weighted by valid
transition counts to preserve effective-batch gradients.

The fresh-run startup performs 500 WM-only updates while the initial policy with
execution noise continues collecting. Then each scheduled update fits WM and
updates the policy/critic. The source collection cadence is warmup 40 vector steps,
UTD 0.4; the first update is at vector step 42. Environment count changes physical
transition counts, not this cadence. The current run resumes an already-joint
checkpoint; it does not repeat WM pretraining.

## Policy objective and real critic

For horizon h=8, the actor optimizes the survival-weighted predicted rewards plus
the discounted surviving endpoint value. Gradients traverse the learned WM and
analytic task features, not MuJoCo or DiSECt. WM/critic parameters are frozen
during the actor gradient calculation; gradients with respect to their inputs remain.

The current task sets gamma=1. Source alternatives default to horizon 32/gamma=.99.
The real-critic configuration still uses 32-step recorded windows and lambda=.95,
not complete-episode Monte Carlo targets. A nonterminal window bootstraps from
the critic's own endpoint value; a real terminal (including this task's deadline)
removes bootstrap. Thus `real` means recorded transitions/rewards, not freedom
from value-estimation bias. Each critic update samples 4096*32 valid states.

BC adds, using the existing implementation,

    -J/h + (mean(abs(J)) / (alpha*h)) * MSE(policy(history, goal), recorded_clean_action)

with alpha=0.25 in the current continuation. The regularizer is sampled from
replay and is attributed to TD3+BC; this online procedure is not the original
offline-RL algorithm. The source configuration has BC off. Static feature scales,
rather than a newly added RunningNorm critic, are retained.

Completion probability weights the success reward and survival consistently.
Force/workspace/deadline endings absorb rollouts; their entering reward is counted
once, later rewards/bootstrap are zero. The v3 coupling and numerical endpoint
rules are specified in TASK_FREE_CUT. A bounded completion formula is not evidence
of calibrated predictions, and an unconstrained critic can still exceed reward bounds.

## Sensor convention and physical revision

Current free-cut groups select net feedback (`net_contact_sensor_v1`). Sum contact
forces on the knife and their moments, transform to the sensor frame with the
rigid wrench transform, then apply the existing filter/controller interface.
The historical scalar bridge instead sums contact magnitudes on one axis with
zero material moment; keep its results separate.

The wrench moment uses contact positions cached before the final force-producing
material substep is integrated. Net-mode limits include the final coupling sample.
Coupling samples are simultaneous at 2 ms; a component-wise temporal peak envelope
must not be reported as a simultaneous net-force magnitude.

Contact revision `edge_coordinate_consistent_v2` corrects the upstream segment
coordinate mismatch: search, position/velocity interpolation and force scattering
all use p=(1-u)*x1+u*x2. At a fixed contact this preserves summed force, moment and
power transfer. These identities are not global energy conservation, a full
adjoint proof, or a real-material calibration. Metadata retains the revision for
scalar and net modes alike; old runs require their recorded code/config/cache.

One test was narrowed from 4 ms per-spring equality to same-input one-substep
equality plus finite/irreversible damage and long-position checks. The original
long-horizon stiffness discrepancy remains unresolved; do not describe the
narrower test as equivalent coverage. Detailed derivation and failures remain
in [historical results](archive/RESULTS_20261111_20261112.md).

DFlex execution changes in `1b287d5` launch forward/backward kernels on PyTorch's
current CUDA stream, key compiled caches by CPU and CUDA source, and use
graph-compatible scalar initialization. They affect legacy paths too; they do
not change contact/fracture equations. Per-material settling/scratch cache is in
TASK_FREE_CUT. Physical calibration and bitwise long-run determinism are not claimed.

## Runtime repair used by the continuation

The running training checkout is `b1bcb92`, a separate reviewed infrastructure
branch; these notes do not merge its code into every branch.

- Schedule state v2 saves warmup origin, fractional UTD phase and full ScheduleCfg.
  Restored replay continues that cadence; intentional empty replay refills first.
  Old files warn and record defaults/origin-0/phase-0 migration. Physics restarts.
- Evaluation cleanup closes the environment on normal return, exceptions, timeout
  and catchable termination.
- Classification metrics accumulate int64 event counts, then divide once across
  valid transitions. These are pooled rates, not averages of minibatch rates.
  Missed/false-positive fractions have all valid transitions as denominator;
  derive conditional rates from counts when reporting recall/false-alarm rates.

The source loss, gradients and two Adam steps were checked unchanged by the metric
repair in float32/64. Evidence and tests: [RESULTS](RESULTS.md). No uncertainty
penalty, critic clamp, new observation interface or new reward was added.

## Other implementations

- `source`, `real_wm`, BC-only, head-only and risk variants remain separately named.
- The official SuperMimic adaptation is a separate code path ([SOURCES](SOURCES.md)).
- SliceIt/Gazebo SAC and the unified fixed-prism SAC are historical baselines,
  with task/selection limitations in [BENCHMARK](BENCHMARK.md).
- Frozen WM+MPPI and traditional references are on `codex/free-cut-mppi` (`3635dab`);
  that branch's progress-estimation/interface work is not a current long-run result.

Same-model planning comparisons must pair the actual network tensors and disclose
planner state information and computation time. Do not merge checkpoints/tasks by
filename or reuse old pose-completion rates as separation rates.
