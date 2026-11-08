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

## Next experiment — baseline vs baseline + A (proposed; awaiting approval)

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

**Measurements.** Frozen 8-episode evaluation at updates 150/200/250/300/last,
plus training curves and terminal-reason counts per 20 vector steps. Report all
checkpoints, with the fixed-budget final checkpoint as the primary comparison:

- Physical success and each failure reason first; completion steps and return.
- Sensor limit metrics, material peak/mean force and cut fraction separately.
- WM prediction evidence explains the mechanism; it is not the outcome gate.

The frozen task measures execution from the fixed start. Noisy training episodes
show on-policy failure exposure; their rates are not held-out generalization
rates. Repeated frozen episodes are not independent training seeds.

**Decision.** Evaluate task performance and force risk together under the existing
reward. A consistent control benefit supports advancing the candidate to the
main comparisons. Equal control outcomes with better prediction support only a
model-fitting improvement. Worse outcomes reject A for this learning setup.
Mixed outcomes are reported as mixed; close this candidate round and reconsider
the method claim before requesting another run. The former “win at 3 of 5
checkpoints” rule is removed; correlated checkpoints do not establish a general
advantage. No claim of simultaneous faster and lighter cutting is assumed.

An implementation error that invalidates the pair is repaired and its affected
arm rerun within a separately recorded budget. Another imperfect local metric
alone does not create a new blocking stage.

## Paper evidence after this decision

PAPER specifies the purpose of these comparisons; each needs a concrete run plan.

1. Select the method supported by control evidence and finish its mechanism
   ablations, reusing completed source/critic/BC-only results where applicable.
2. Compare it with SAC, same-WM MPC and fixed compliance control in the unified
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

## Last recorded machine snapshot (2026-11-08)

- A6000 GPU 0/1 and GPU7: recorded idle. Refresh before scheduling approved work.
- Run 1 (SliceIt SAC, Gazebo, tamsGPU9): unreachable ("no route to host") in the
  last check. Last recorded snapshot: 46044 environment steps, 43747 updates,
  2297 episodes; the 72 h deadline from 2026-11-03 has passed.
- Routes: `runs/operations/MACHINES.md` on the Mac (outside Git).
