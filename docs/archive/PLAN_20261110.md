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
result. Multiple objects alone do not establish novelty. Method contribution depends
on B or another justified mechanism; empirical success does not settle that
question. See PAPER and the per-paper records in [literature](literature/README.md).

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

## Route B — bounded method screening, separate from new-policy training

Scope revised after user/Codex review, 2026-11-10. This is a candidate method
plan, not implemented functionality or authorization to launch its GPU jobs.

**Role in S0–S4.** B1 uses historical simulation transitions for predictor
fitting/evaluation; B2 uses a frozen historical WM to plan new physical
rollouts. Neither trains a new policy. They screen a force-risk formulation,
not validate multi-material learning or the final algorithm. S0–S2 continue
independently. B failing or remaining inconclusive ends this candidate screen;
it does not stop task integration or force a restart of the entire paper.

The main learning experiment is S3: new shared WM/policy/critic weights and
new online interaction across the declared material/geometry distribution,
compared with fresh SAC under the same physical-data budget. policy62 and its
continuations are historical references, not the default initialization.
If B is selected, it must enter and be evaluated in this new-policy training.

**Question and evidence.** Existing checks report that the tested WMs missed
19–23 of 23 recorded force-limit events; MPPI predicted sub-limit ratios and
hit the physical limit in 4/6 recorded runs. This motivates testing whether
calibrated upper predictions help control. It does not prove a universal
contact bias, intrinsic unpredictability, or that ensemble uncertainty must
fail. Residual/quantile calibration and uncertainty-aware control have
precedents (NOVELTY); neither an ensemble's failure nor adding a calibration
table alone establishes a new method.

**Quantity and available information.** Predict the next control interval's
sensor-limit utilization R = max_i(peak_wrench_i / force_limit_i), using the
state/history at t and candidate action a_t. Components and limits follow
BENCHMARK, including torque; R > 1 is the existing physical limit event.
Material-contact force remains a separate metric. Calibrate the scalar maximum,
not six unrelated 95% bounds interpreted as a joint 95% guarantee.

A phase-dependent bound may use only declared causal robot/force history and
goal information; any known scene geometry must be explicitly shared under S0.
Spring release, true damage, future contact labels and future measured wrench
are unavailable to this controller. They may stratify offline reports, but must
not select the deployed bound. Freeze an observable phase rule before test
access; absent such a rule, use the global bound and make no phase-aware claim.

**Historical B1 data; no new collection.**

- Fit: working-baseline replay
  (`critic_abl_a250_20261106/training/replay.pt`, A6000) and
  `wm_frozen_fit_20261105` training split. Recorded run-level failure percentages
  are not a verified composition of the retained replay file.
- Calibration/development: reliability-supplement dev split (reported 23
  force-limit endings, including 16 onset events).
- Held-out screen: reliability-supplement final split (reported 13 force-limit
  endings), kept sealed until predictors and operating settings are frozen.
  It is this historical pilot's held-out screen, not the future multi-condition
  paper test. Do not reuse it to select or revise the method after opening.

Record disjoint episode provenance across fit/calibration/test. Adjacent frames
are not independent trials. Stratified diagnostic collection does not represent
deployment event prevalence. A raw residual percentile or pinball head does not
by itself guarantee coverage: finite-sample conformal ranks and assumptions
about calibration/test sampling must be specified. Otherwise report empirical
coverage only. Old-policy calibration cannot certify new policies, imagined
candidate states, new materials or a whole rollout.
References: [conformal prediction](https://arxiv.org/pdf/2107.07511),
[CQR](https://arxiv.org/pdf/1905.03222).

**B1 — one offline screen, no policy updates.** Retain four predictor families:

1. Existing mean WM as the historical reference.
2. Five independently initialized WMs on the declared fit data. Record each
   member's fitting budget, data sampling and uncertainty construction.
   A deterministic-member spread is not all forms of predictive uncertainty.
3. A 0.95 pinball-loss head on frozen causal WM state/action features.
4. A one-sided residual correction, global or using the predeclared observable
   phase, on the existing WM.

Give ensemble and single-model predictors the same held-out calibration
opportunity. Max-member and mean-plus-two-standard-deviations remain heuristic
references, not guaranteed bounds. Choose variants/operating settings on dev
only, with the proposed 10% nominal false-alarm budget treated as a development
choice rather than a task threshold or safety guarantee. Freeze one setting
per family; do not tune them to match false alarms on the test set.

Report event counts/recall, achieved false-alarm rate and upper-bound slack;
show contact-relevant nominal steps separately so easy free-motion steps do
not hide false alarms. Uncertainty summaries use episodes as the sampling unit,
not independent adjacent frames. Report empirical bound coverage separately
from event recall. With 13 held-out failures, one event is about 7.7 percentage
points: this is screening evidence, not proof of a reliable deployment rate.

**Frozen B1 advancement rule (development pilot, not a safety standard).**

Before opening the held-out split, choose **one** B candidate and **one**
ensemble comparator on development data, giving both the same calibration
opportunity. Record the exact checkpoints, causal features, score, threshold,
episode IDs and fitting/calibration provenance. Other families remain
descriptive references; the test cannot select a winning B family.

The unit is an episode. On a force-limit episode, count a detection only when
the prediction made *before executing the first violating control interval*
flags that interval. Do not count a warning inferred from its measured force,
post-event frames or an absorbed state. On each non-force-limit episode,
count one false alarm if any valid pre-terminal interval is flagged; exclude
padding. This episode false-alarm rate is distinct from per-step rates,
which are also reported with the declared contact-relevant strata.

1. On development, both selected settings must have episode false-alarm rate
   at most 10%. If no eligible setting exists, end this candidate as
   inconclusive without opening the held-out split.
2. On the reported 13 held-out force-limit episodes, let `b` count episodes
   detected only by B and `c` those detected only by the ensemble.
   Require **net additional detections `b - c >= 4`**.
3. On the same held-out non-force-limit episodes, B must produce **no more
   false-alarm episodes than the ensemble**. This prevents the +4 rule from
   rewarding a simple shift toward always warning. Report achieved rates and
   denominators rather than assuming the development 10% rate transfers.

Only if all three hold does B advance to its already bounded B2 screen.
Otherwise label this pilot inconclusive for advancement, or explicitly worse
if the paired outcomes show a loss, and end B; no threshold revision or fresh
test set follows automatically. This is a conservative project choice, not a
literature-derived performance threshold.

Report the full paired detection table, counts and uncertainty. Even the
case `b=4, c=0` has two-sided exact McNemar p=0.125; the +4 gate does **not**
establish statistical superiority. Thirteen events cannot certify deployment
safety, multi-material transfer or a new method. Development selection and
calibration on reused data support empirical screening only, unless the
additional split/assumptions required for a coverage claim are supplied.
If a calibrated standard ensemble suffices, use that evidence to narrow the
method claim rather than requiring ensemble failure.

**B2 — optional planner screen, no policy training.** Use the same frozen
working WM for motion prediction, the same nominal task and no terminal critic
in all arms. Compare the mean-only planner, calibrated ensemble risk estimate
and selected single-model bound, with eight paired planning seeds per arm.

Retain the physical task reward, success and termination definitions. Specify
the upper bound as an explicit planning risk constraint or cost, including its
strength and all-candidates-infeasible behavior, before execution. Do not merely
replace the absorption predicate while retaining an inconsistent reward:
`mpc.py` obtains reward from `wm.predict()` before computing absorption.
Keep risk cost, task reward and physical endings separately logged. The
constraint/cost formula is still an implementation choice to be fixed; it is
not silently selected by this document.

Report every ending, completion, duration, sensor-limit exceedance, material
peak/impulse and planning time. Reduced force through stalled/incomplete cuts
is not a control benefit. Eight planner seeds are a pilot, not training seeds.
B2 can support a planning mechanism; it cannot establish BPTT learning benefit.

**Bridge to S3.** If retained, B is an explicitly named variant alongside the
working objective. Before its S3 run, define its differentiable risk objective,
action-gradient path and how calibration is refreshed/invalidated as the WM
and interaction distribution change. A hard Boolean stop alone supplies no
useful pathwise force gradient. Reuse the procedure on S3 training/development
data, not old prism residual constants as if they transfer automatically.
Keep the new multi-condition test split unused for fitting/calibration.
An additive bound does not itself repair the mean dynamics or its gradients.

**Bounded allocation.** Original proposed caps remain: B1 at most five GPU
hours for fitting, B2 at most two hours/24 physical rollouts, on one available
GPU. Log all costs; incomplete fitting is incomplete, not model failure. No
automatic extensions or broader diagnostic queue. This revision launches
nothing and does not make B a prerequisite for S0–S2.

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
interaction budget. The planned main comparison starts both from fresh networks,
including a fresh WM and critic for the WM method. Its WM-only collection and
later joint updates sample the declared distribution. policy62 is not the
default warm start. This trains a new shared controller; it is not another
evaluation or continuation of the old fixed-prism policy.

Any later proposal to reuse a checkpoint/curriculum must be an explicit
protocol revision with all pretraining exposure counted for both sides. It
must not silently replace this fresh-training comparison.

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

The immediate implementation track is S0–S2, independent of B's outcome.
Its S3 training requires a concrete S0 distribution and run settings/budget.
Any historical B pilot requires its separately frozen configuration within
the cap above; this document edit launches neither track. Low early reward
does not automatically justify changing the task or model.

## Storage and execution

Remote SSH only. GPU7 canonical checkout:
`/dataHDD/dong/Projects/Cutting_DEV`; machine routes and protected dependencies:
`runs/operations/MACHINES.md`. GPU9 is retired. Recheck actual GPU occupancy
before execution; no current occupancy or running-job claim is made here.
All source, data, checkpoints, figures and caches stay on servers.
