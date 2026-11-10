# Current plan

Updated 2026-11-10 after the multi-condition research review.
[PAPER](PAPER.md) holds the proposed research claim and actual algorithm;
[METHOD](METHOD.md) maps code; [BENCHMARK](BENCHMARK.md) defines existing task
metrics; [RESULTS](RESULTS.md) retains historical experiments.

## Direction: one learned controller across cutting conditions

Develop one shared WM, policy and critic across a declared training distribution
of materials and geometries. Start with prism material variation, then incorporate
prepared prism geometries and the jello cylinder through grouped simulation.
The old fixed-prism results are feasibility and development evidence, not the
intended final experimental scope.

The proposed claim is that model-based policy learning can produce a shared
compliant-cutting controller which, at comparable task completion, uses lower
contact force than a matched-budget SAC. This is a hypothesis, not an existing
result. Multiple objects alone do not establish novelty.

**No more action-perturbation evaluations, and no such results in the paper.**
Preserve their files as historical diagnostics; stop using the 256-case rates,
noise confidence intervals and figures as manuscript evidence. Normal training
exploration is separate and is not disabled by this decision.
The unexecuted ten-episode frozen elasticity pilot is withdrawn. The next work
is a training task distribution, not zero-shot testing of an untrained range.

## Correct training provenance

The source-configuration learning chain starts from fresh weights and a
500-round WM-only stage, then proceeds through resumed joint-training stages
to policy62/WM562 (62 policy updates cumulatively). policy62 completes 8/8
nominal evaluations. It is not a supplied expert or externally pretrained
cutting policy. The intervening terminal/masking fixes and resumes are part of
the historical lineage; do not describe it as one uninterrupted, unchanged run.

The later `real + H8 + BC 0.25` working configuration was evaluated by
continuation from policy62. Thus from-scratch learning exists for the source
chain, while the working variant's clean from-scratch performance on the new
distribution is still to be measured. Both facts belong in the paper.

## What exists and what must be connected

| Capability | Evidence/current code | Remaining integration |
|---|---|---|
| Per-world material parameters | P0 checked E, density and cutting-spring stiffness scaling; `BatchedDiSECt.set_material_scales` | Reset-time sampler, condition IDs, persistent RNG, material-specific reset/metadata and replay provenance |
| Same-shape GPU batching | `pack_model` duplicates one scene; fixed particles/springs per world | Preserve this path for each object group; it is not already a heterogeneous batch |
| Different object geometry | Exported prism and free jello have reference rollouts | Group dispatch/collection, compact-state merge, per-group reset and recording |
| Shared learning representation | `ControlStateSpec` predicts fixed-size robot/wrench quantities, independent of particle count | One learner consumes all groups; preserve sequence boundaries and object/condition provenance |
| Position variation | Fixed initial frame and fixed coupling transforms today; knife is constrained to its prepared plane | Physically consistent scene/cut-plane/robot/goal transformations; not an observation-only or visual offset |
| SAC in unified environment | Existing fixed-task SAC85k | Parallel collection/training with the same new distribution and declared interaction budget |
| Same-WM MPPI and reference | Frozen evaluators already work on the fixed task | Use the task distribution, matched model identity, per-condition goals/reference frames |

These are missing features for a new task, not newly discovered failures of the
already tested fixed-task implementation.

## Route B viability check (proposal, awaiting user/Codex approval, 2026-11-10)

**Relation to S0-S4.** S0-S4 build the experimental setting (one shared
controller across conditions, matched-budget SAC). Route B is the candidate
method contribution inside that setting; B1 and B2 use existing data and models
and can run while S0/S1 are implemented. If B is supported, its bound enters the
S3 training comparison as the method's force-limit constraint.

**Why.** The method contribution is open: A had no clear control benefit, and the
working design (`real` critic + short horizon + BC, designed in this project and
supported by its ablations) is the method's base but has precedents per
component. Route B targets a measured failure: every WM missed the contact-onset
force spike at force-limit events (19-23 of 23 missed, one-step under-prediction
11-20 N, while nominal trajectories average +0.08 N), and model-based
optimization exploits it (same-WM MPPI chose predicted limit ratios 0.71-0.99 and
ended by the force limit in 4 of 6 runs). Literature (NOVELTY section 9):
pessimistic safety costs in world models (LAMBDA, CAP) and GP force chance
constraints in MPC (Matschek, Haninger) exist; not found: the measured one-sided
onset under-prediction and a calibrated onset-aware peak-force bound inside WM
policy optimization. The decisive question for novelty: does ordinary ensemble
(epistemic) pessimism already flag the onset under-prediction?

**Hypothesis.** The onset error is a bias shared by independently trained WMs,
so ensemble disagreement does not flag it, while a one-sided bound calibrated on
real onset events does, and using that bound as the force-limit constraint stops
the planner from exploiting it.

**Data.** No new collection.
- Fit: the working-baseline run's replay (`critic_abl_a250_20261106/training/replay.pt`,
  A6000; training episodes with 7-14% force-limit endings) plus the
  `wm_frozen_fit_20261105` training split.
- Calibration: reliability-supplement dev split (23 force-limit endings, 16 at
  contact onset).
- Test: reliability-supplement final split (13 force-limit endings), sealed so
  far; opened only for this check's final numbers.

**B1 — offline prediction check (no policy training).** Predictors of the next
control step's peak sensor wrench given state and action (one step before the
event):
1. Mean: the working-baseline WM.
2. Ensemble: 5 WMs trained from independent initializations on the fit data
   (same architecture, losses and budget as the WM-only stage, capped below),
   bound = max over members and mean + 2 std.
3. Quantile head: a tau = 0.95 pinball-loss head on the frozen working-baseline WM
   features (head only).
4. Conformal residual: working-baseline prediction plus the 95% quantile of
   calibration residuals (true - predicted peak), computed separately for the
   contact-onset phase (no spring released yet and the knife within the
   object's top band) and the cutting phase. No training.
Metrics on the test split, per predictor: share of force-limit events flagged
(bound above the limit one step before), false-alarm rate on nominal steps
(bound above the limit while the true peak stays below), mean slack of the bound
on nominal steps, and for the ensemble the member spread at onset vs nominal.
Decision rule (fixed now):
- B supported: a calibrated bound (3 or 4) flags at least 80% of test force-limit
  events at no more than 10% nominal false alarms, while the ensemble bounds flag
  at most half as many events at matched false-alarm rate.
- B not supported: the ensemble bounds flag within 20 percentage points of the
  calibrated bound (pessimism already solves it), or no bound reaches 80% at
  10% false alarms (the onset spike is not predictable from the WM state; the
  route is reconsidered, not patched).

**B2 — planner check (evaluation only, only if B1 supports B).** Same-WM MPPI on
the working-baseline WM, settings as before, no terminal critic (its imagined-
state values are out of range and would confound the force effect). The
predicted force-limit predicate uses (i) the mean, (ii) the ensemble bound,
(iii) the calibrated bound; 8 planner seeds each on the nominal task. Metric:
force-limit endings, success, steps, material peak force. Supported if (iii) has
fewer force-limit endings than (i) and (ii) without losing completion.

**After B1/B2.** Only if both support B: a training plan that uses the calibrated
bound as the force-limit constraint in BPTT (Lagrangian or adaptive penalty, as
CAP), evaluated in the S3 setting against the working baseline, SAC and same-WM
MPPI. If B fails, the paper positioning is rediscussed before any
further GPU time.

**Implementation.** Ensemble training from fresh initializations on a replay
file; quantile head on frozen WM features; conformal residual tables; flag
metrics; a bound option for MPPI's force-limit predicate. Tests for each.

**Budget.** B1: ensemble about 5 x 1 h WM fitting plus minutes for the rest
(GPU7 when free, or A6000 when its other user's jobs end); B2: 24 MPPI episodes,
about 2 h. Single GPU. No automatic extension.

## Ordered implementation and experimental stages

### S0 — freeze a task-distribution specification

Deliver one versioned specification, shared by training and evaluation, before
launching new training. It must include:

- Prepared scene identities and hashes; initially prism, then eligible prism
  variants and the jello cylinder. Apple is deferred because the current
  reference fails there, not declared impossible for every controller.
- Explicit material values, units, sampling rule, parameter correlations and
  source provenance. P0 scale checks are implementation evidence, not a chosen
  training distribution or real-food calibration.
- Elastic modulus, density and cutting-spring stiffness are separate parameters.
  Spring stiffness does not equal fracture energy or a complete damage law;
  softness/rest lengths/contact parameters remain explicit.
- Admissible placement transforms and matched start/goal/cut-plane definitions.
  New geometry must be exported with its cut interface; no mesh-only scaling
  that leaves masses, rest geometry or cut preparation inconsistent.
- Separate training/development/test condition manifests. Test new draws within
  the training range separately from genuinely held-out material ranges or
  geometries. Fix them before seeing controller results. Hyperparameter choice
  uses development data, not the held-out test.
- One common observable-information contract: robot/proprioception, sensor force
  history and known cutting goal. Do not give one learner hidden material labels.
  Specify history access for SAC and WM policies before claiming an
  algorithm-controlled comparison. MPPI's model state access remains disclosed.

Cutting speed stays an action-dependent behavior. It becomes a task condition
only if an externally specified deadline/desired-speed task is deliberately
introduced; that is not this plan.

The parameter intervals, condition counts and compute budget are not selected
by this document. They belong in S0, not invented later inside a collector.

### S1 — prism material randomization at episode reset

Use the existing scale operator, sample only reset worlds, and keep a condition
fixed until that episode ends. Start each new episode from undamaged material
with its selected parameters; refresh any reference arrays or force/recording
metadata which describe that state. Preserve worlds which are still running.

Save condition parameters, episode IDs, reset RNG state and physical-transition
counts in recording/checkpoint provenance. Prevent replay windows and histories
from crossing resets. Finite histories must not accidentally carry the previous
material into a new episode.

Use a small functional integration check to verify actual material arrays,
selective reset and one genuine learner update from mixed-condition data.
This checks implementation, not learning quality; do not redesign the network
from a few update losses. Placement variation joins only when its physical
transform and prepared cut interface have been implemented consistently.

Deliverable: one policy/WM training pipeline with changing episode materials.
Any intermediate training is development; preserve its full data/compute counts.

### S2 — grouped geometries, shared learner

Create one homogeneous simulation group per prepared geometry/topology.
Keep particle/spring arrays and low-level reset/force state within their groups.
Concatenate fixed-size control histories, goals, actions and episode transitions
for **one shared WM, policy and critic**. Do not train a separate policy per object.

Choose a declared group/condition sampling weight so fast-ending scenes do not
silently dominate the replay. Preserve original per-environment provenance and
independent RNG streams. The compact state makes a shared network feasible;
it does not prove that ten-frame history is sufficient to identify every material.

Complete one joint collection/update cycle across both prism and jello groups,
then freeze the executable distribution. These functional checks replace
re-running a broad physics validation campaign.

### S3 — matched-distribution learning comparison

Train WM + BPTT and SAC on the same final training distribution and physical
interaction budget. The recommended clean comparison starts both learning
procedures from fresh networks. If a staged curriculum or existing checkpoint
is reused instead, declare it and count all pretraining/continuation exposure
for both sides; do not compare a warm-started method with a fresh SAC while
ignoring its history.

For WM + BPTT, the current working recipe is the starting method:
physical-trajectory critic, H8 and replay BC 0.25. Do not silently add A, change
the reward or attach a new force safety mechanism. Existing code supports a
source configuration, but the selected new-task recipe must be recorded once.

Budget counts actual individual environment transitions, including WM-only
collection, refills and other warmup interaction. Vector steps, update counts
and wall time are not substitutes. Record model/critic/actor updates and total
wall/GPU cost separately. Pure optimizer passes do not add physical samples.

Use equal observable information where the claim compares learning algorithms;
otherwise explicitly label a system-level comparison. Same interaction budget
does not require identical learning rates or parallel counts. Record optimizer
and update-to-data settings in advance, including any permitted development
tuning. Do not automatically reuse the old unstable SAC settings or silently
replace them with unreported settings.

SAC85k remains a historical fixed-task comparator. Fresh SAC training is needed
here because the task distribution changes, not because its old result is invalid.

First complete a development run of each method. Multi-seed confirmation follows
a fixed method and protocol, as the user requested. A wall cap is a resource
limit; report an incomplete interaction budget rather than calling it complete.

### S4 — frozen comparison on predefined conditions

Evaluate the shared policy, matched-distribution SAC, paired same-WM MPPI and
a physically aligned reference trajectory. MPPI/reference need no policy
training, but their task interfaces and online computation still need accounting.
Do not let MPC query hidden true material parameters absent from its paired WM.

Report each material/geometry group and an explicitly weighted aggregate:
goal completion, force-limit/workspace/time-limit endings, duration, material
peak/mean/impulse, sensor-limit utilization, released-spring fraction and online
cost. Keep outcomes separate from physical-separation evidence.

“Comparable completion with lower force” needs a predeclared completion criterion
and force summary. A confidence interval containing zero does not prove equal
completion. Include all outcomes and a clearly labeled paired both-success
analysis; do not obtain lower force by dropping failures or unfinished episodes.

Within-distribution conditions, held-out material ranges and held-out shapes
are separate claims. A policy trained jointly on prism and cylinder has not
demonstrated unseen-shape generalization merely by succeeding on both.
Model prediction errors support interpretation, not replace physical task results.

## Stop expansion, not scientific reporting

No action-noise evaluation, ten-episode zero-shot pilot, automatic A repeat,
new simulator, arbitrary 3-D cutting plane or multi-cut topology task is queued.
No generic “real critic versus imagined critic” experiment precedes building the
new task distribution by default. A targeted ablation is selected only when it
tests the final method claim.

Current work is this scoped plan and implementation assessment. No new long run
is launched until a concrete S0 specification and run settings/budget exist.
Low early reward does not automatically justify changing the task or model.

## Storage and execution

Remote SSH only. GPU7 canonical checkout:
`/dataHDD/dong/Projects/Cutting_DEV`; machine routes and protected dependencies:
`runs/operations/MACHINES.md`. GPU9 is retired. Recheck actual GPU occupancy
before execution; no current occupancy or running-job claim is made here.
All source, data, checkpoints, figures and caches stay on servers.
