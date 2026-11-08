# Current plan

The single current task list. The research thesis, connected algorithm and paper
experiments are in [PAPER](PAPER.md). Measurements live in [RESULTS](RESULTS.md),
task and metric definitions in [BENCHMARK](BENCHMARK.md), implementation and
sources in [METHOD](METHOD.md), literature evidence in [NOVELTY](NOVELTY.md).

## Destination and current stage

Develop a world-model-based method for force-limited compliant cutting whose
policy improvements carry over to physical execution. The paper follows one
chain: **robot/material interaction data → motion/force model and task events →
BPTT policy optimization → executed cutting performance**.

Current stage: test whether the implemented terminal-supervision intervention A
improves actual control, on the existing single-cut task. Position prediction
already improved in the fixed-data comparison. This control comparison is the
next missing link. Multi-cut tasks, loss reweighting B, feature rescaling C and
new contact-model architectures are outside this stage.

## Decisions in force

- Benchmark: free-bottom `public-prism-ur5e-v1`. Fixed-bottom results remain in
  RESULTS; they do not change the main task.
- Current working baseline: `real` critic + h8 + BC 0.25, continued from
  policy62/WM562 with a fresh critic and 100 pre-fit rounds. Candidate: the same
  configuration with `--wm-terminal-gradient head` (A).
- Source, working baseline, V5, Gazebo SAC and unified-environment SAC keep their
  separate BENCHMARK names. Task, reward, material, action bounds and force-limit
  definitions stay fixed.
- Physics advances the robot and material; actor gradients pass through the
  learned model. Successful goal-pose completion and material separation retain
  their separate BENCHMARK measurements.
- Method development uses one training seed. Multi-seed confirmation follows a
  fixed method and comparison protocol.

## Completed evidence — WM diagnosis closed 2026-11-08

- True-state task predicates reproduce every checked recorded label.
- Isolating terminal BCE from dynamics fitting changes held-out H8 tip error
  from 7.88 to 0.28 mm after 100 rounds. The fixed training split has no
  force-limit and only 3 workspace endings; those predictions remain poor.
- On five checked successful policy trajectories, the WM captures the higher
  force of faster actions. The reward accepts this trade-off.
- Missed endings put critic evaluations beyond the physical episode. Their
  causal effect on actor learning remains to be tested by the control pair.

These results motivate the method experiment in PAPER. They are not a queue of
local metrics that must all improve before control can be evaluated.

## Next experiment — baseline vs baseline + A (running since 2026-11-08 21:43)

This draft replaces the next-stage draft at `5678c97`. No new run is authorized
by this document update.

**Question.** Does isolating terminal supervision during WM fitting improve the
cutting policy learned through the WM, with the task and actor objective fixed?

**Hypothesis.** Better physical prediction near task endings improves executed
control. A changes the fitting gradient route; it is not a new force objective.

**Pair.** Both arms resume policy62/WM562, reset the critic and pre-fit it for
100 rounds; use `real` + h8 + BC 0.25, 1024 worlds, the same seed and update
budget. A changes only `--wm-terminal-gradient head`. Both retain the normal
online collection and WM update loop. The extra offline 100-round refit from
previous PLAN is removed: its outputs were not inputs to this joint comparison,
and fixed-data evidence already exists. The final supplement set remains sealed.

**Budget.** Retain the previous proposed budget: A6000 GPU 0 and 1, one arm per
GPU, 12 h wall each; `--max-iterations 860` (end iteration 2421). Existing
checkpoint/progress/timeout handling applies. No automatic extension.

**Measurements.** (Revised 2026-11-08 after Codex review of `f316f74`.)

From the fixed start the frozen task is near its ceiling (policy62 8/8 in 20
steps; working baseline 8/8 in 17–19), so the main outcome is robustness of the
frozen policies under predefined action perturbations. Two layers of uncertainty
are kept apart: case-paired intervals describe the difference between these two
trained policies on this perturbation distribution; they do not include the
variation from another training seed. Method development stays at one seed; more
training episodes do not replace seeds.

1. **Perturbed frozen evaluation (main outcome).** `scripts/evaluate_perturbed.py`:
   256 cases fixed before any policy runs (index, sigma 0.3/0.6/1.0, first
   perturbed step 0/5/10, i.i.d. or persistent direction, own noise seeds; seed
   base 3000000, disjoint from the reliability supplement). Every arm and
   checkpoint executes the same cases, paired by index; each case runs to its own
   end and all four outcomes are kept (success, force limit, workspace,
   time limit); no stopping by outcome counts. Reported: outcome rates; material
   peak/mean force and safety margin over all cases; steps and force also on the
   cases both arms complete successfully. Results are called robustness under
   action perturbations.
2. **Training episodes (auxiliary).** Terminal-reason rates per 20 vector steps;
   on-policy exposure, not held-out rates.
3. **Unperturbed frozen 8 episodes (reported).** The fixed-start task result.
4. **Reference run (shown, not a threshold).** `critic_abl_a250_20261106` has the
   baseline configuration on an older commit; measurements 1–3 are shown for it.
   One baseline − replicate difference is one repeat, not a bound on training
   noise. A bitwise match would only show that this resume setup is reproducible.
5. WM prediction errors explain results; they do not decide them.

**Budget addition.** Perturbed evaluation of 15 checkpoints (A, baseline,
reference) × 256 cases, about 2 h on one GPU after training.

**Decision.** Main quantity: the case-paired difference A − baseline of the
perturbed success rate at the final checkpoint; force-limit, workspace and
time-limit rates reported separately; other checkpoints show the trend.
- Interval supports an improvement → development evidence; A advances to the
  main comparisons.
- Interval includes zero → "no clear benefit detected" (not "no benefit"); A is
  not advanced as a control method.
- Clearly worse → the candidate is rejected.
- Mixed (e.g. success up, force-limit rate up) → this round closes with a mixed
  conclusion.
"Not significantly worse" is not evidence that the arms are equally safe.
No claim of simultaneous faster and lighter cutting is assumed.

**Expectation, stated before the run.** The most direct effect of A is on
prediction and control near task endings. All WMs missed the contact-onset force
spikes of force-limit episodes; whether A changes force-limit risk through
changed predictions and policy behaviour is an open experimental question.

**If A is not a control method.** The comparisons (SAC, same-WM MPC, fixed
compliance) proceed with the working baseline. `real` critic + h8 + BC is an
existing combination, not a contribution by itself; the paper's contribution
must then come from mechanism evidence and comparison results. A further model
change is proposed only against a measured failure that limits control, with its
own plan and literature basis.

An implementation error that invalidates the pair is repaired and its affected
arm rerun within a separately recorded budget. Another imperfect local metric
alone does not create a new blocking stage.

## Paper evidence after this decision

### WM–MPPI configuration selection completed (2026-11-08)

Current planner: frozen WM562, H16, 512 samples, four refinements, temperature
0.1, noise std 0.5, discount 0.99, no terminal critic. The physics, task reward,
action bounds and force limits retain the benchmark settings.

GPU7 planner seeds 0/1/2 completed the original goal-pose task in 42/46/36 steps;
material peaks 13.42/15.75/16.26 N, with no force-limit or workspace ending.
The previous H8 configuration exhausted its 600 s budget. Full measurements,
selection evidence and the rejected frozen-critic candidate are in RESULTS.
The executable command is in BENCHMARK.

The planner is ready for the common algorithm comparison. That stage fixes
information access, development cases and compute/data budgets for SAC,
WM–MPPI, BPTT policy and fixed control; it records planning time separately from
physical stepping. The A control pair retains its own decision rule above.

### Four-week delivery target (2026-11-08 to 2026-12-08)

1. Week 1: close the A control decision; establish MPPI, SAC and fixed-controller
   comparisons on the same task. Freeze the method claim after control evidence.
2. Week 2: selected single-cut material variations and held-out conditions;
   matched mechanism ablations. Finish the method and experimental-protocol draft.
3. Week 3: independent seeds after method/protocol freeze; conduct the scoped
   robot study if hardware, sensing and safety procedures are ready.
4. Week 4: finish plots, manuscript, reproducibility commands and demonstration;
   independent review of claims against recorded results.

Target: a complete manuscript and reproducible evidence. Robot availability and
an end-to-end advantage are the schedule's dependencies. The end of week 1 is a
decision point: if the candidate has no control benefit, choose a supported
empirical claim or revise the method and delivery target. Repeated local fixes
do not extend the same candidate indefinitely. Multi-cut tasks, new simulators
and visual-policy migration remain outside this four-week scope.

### Comparative evidence

PAPER specifies the purpose of these comparisons; each needs a concrete run plan.

1. Select the method supported by control evidence and finish its mechanism
   ablations, reusing completed source/critic/BC-only results where applicable.
2. Same-WM MPC requirements (review 2026-11-09): pair each planner run with the
   WM and critic of the policy it is compared with (WM562 pairs with policy62;
   the working baseline and A use their own final WM/critic); fix the terminal
   value option once for that comparison with the paired critic (the rejected
   policy62 critic is the imagined-target one with value MAE 153; the working
   baseline's critic uses `real` targets); state the information difference
   (the planner uses the full physical history state, the policy its
   24-component observation history); report planning time per control step
   (0.55-0.64 s against the 50 ms period) and the lateral goal error for every
   method. Compare it with SAC, same-WM MPC and fixed compliance control in the unified
   task. Match information and action access; account for warm-start data and
   model fitting when comparing cost. Prescribed trajectories remain separately
   labeled because they bypass the policy action mapper.
3. Test material variation and then held-out shapes/materials. Multi-object
   per-world parameters passed P0. Free prism, jello cylinder and apple sphere
   placement checks are complete. Reference control completes prism/jello;
   apple ends at the force limit. Choose material ranges and object access in
   that stage's plan, keeping the main task definitions.
4. Once method and protocol are fixed, run independent seeds and the real-robot
   study in [REAL2SIM](REAL2SIM.md). The robot plan still needs user review.

Paper drafting can already cover task, platform, source method, connected
algorithm and completed mechanism evidence. Main comparative claims follow the
control experiments. DiSECt trajectory optimization stays parked; it is not a
prerequisite for the same-WM MPC comparison.

## Last recorded machine snapshot (2026-11-09)

- A6000 GPU 0: a_pair_base_20261108; GPU 1: a_pair_head_20261108.
  Both run frozen code 7c1ab92 in Cutting_DEV_wmterm and retain all active paths.
- GPU7: canonical Cutting_DEV main; inactive experiment checkouts and data
  consolidated. No new planner/collection/training job is started during cleanup.
- GPU9 is retired from the working plan.
- Machine routes, storage and protected dependencies:
  server runs/operations/MACHINES.md. Development now uses Remote SSH.

### WM–MPPI episode video completed (2026-11-08)

The seed-0 frozen WM562 H16 episode reached the task goal in 43 steps / 2.15 s,
with material peak 13.25 N. The recorder shares the benchmark control loop;
video uses original UR5e visual meshes, the DiSECt surface including virtual cut
vertices, and measured force traces. Start/contact/end frames and raw-data
alignment checked. Output: `runs/videos_20261108/wm_mppi/`; commands in RUNNING,
numbers in RESULTS. Physical task and controller settings unchanged.


## Current operation — remote repository consolidation (2026-11-09)

The user requested cleaning the repositories and GPU storage first, moving
source and experiment files off the Mac, and using Remote SSH afterward.
GPU7 main is the development entry. Archive inactive experiment checkouts and
preserve their uncommitted changes. Keep the A6000 A pair and every path it uses
in place while running. GPU9 is retired from the working plan.

After consolidation, the next MPC question is task feasibility across different
objects/materials or cutting conditions, with additional interaction data where
the WM lacks coverage. The current WM562/MPPI result establishes completion of
the existing prism goal-pose task. It does not supply apple or multi-material
training coverage. Choose the next task and dataset scope before launching new
collection or fitting. A broad SAC/policy/reference comparison is not the next
GPU7 job. No new MPC, collection or training batch is started during cleanup.
