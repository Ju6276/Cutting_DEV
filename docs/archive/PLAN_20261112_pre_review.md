> Archived copy of docs/PLAN.md at a8b5083 (2026-11-12, before the task/platform review rewrite in 0ca52c3).
> Proposals in it are not approved decisions unless the current PLAN says so.

# Plan

Updated 2026-11-12 (after the repository clean-up). The only active plan. Topic and prior
art: [PAPER](../PAPER.md); measurements: [RESULTS](../RESULTS.md); code: [METHOD](../METHOD.md);
data: [DATA](../DATA.md). Previous plans: [PLAN_20261111](../archive/PLAN_20261111.md),
[PLAN_20261110](../archive/PLAN_20261110.md).

## Where we are

- No job of ours is running (GPU7 and both A6000 GPUs free of our jobs).
- The first multi-object runs are stopped and deleted. The risk-penalty and official
  SuperMimic runs failed; the base run exposed four defects in the task definition
  (below). The problem statement "the WM under-predicts the contact-onset force spike"
  is retired (PAPER §1).
- **Topic: open.** Candidate: force-adaptive cutting of unknown objects under one hard
  force limit (PAPER §2). The user's assessment (2026-11-12): motivation too simple and
  too few modalities for ICRA/IROS as it stands. The prior-art search agrees it is
  incremental as a task claim (PAPER §3). Options to strengthen it are listed in
  "Topic options"; nothing is decided.

## Fixed decisions (user)

- Method paper for ICRA/IROS/RA-L. Platform, objects and baselines are experiments.
- One shared WM/policy/critic over scanned objects, randomized materials, knife start
  height, target cutting speed as a task input.
- One force limit for every object; no object identity, material label or per-object
  reward/success setting reaches the learner; it adapts from force feedback. Simulation
  limit used so far: `[50, 50, 50 N, 5, 5, 5 N·m]` (not a real-robot threshold; the real
  robot has an external ATI Mini45).
- The paper's core must follow from the task, not from a failure seen in our own runs.
- One change at a time; a written plan is approved before a paid run.

## Task defects (RESULTS 2026-11-12)

| # | Defect | Evidence |
|---|---|---|
| 1 | Success = knife at one fixed pose, not "object cut" | Held potato: 245/245 springs cut, not success. Unheld potato: pushed 32 mm, 57/245 cut, success |
| 2 | The bottom hold pinches the knife and inflates force | cucumber_003: 37 N held, 5.6 N unheld (and then reaches the goal); the unheld potato is pushed away |
| 3 | Reward makes "as fast as possible" optimal; lateral drift is free | Policy 20-27 cm/s against a 1-5 cm/s command; force-limit hits at that speed |
| 4 | Reference trajectory is a fixed displacement | Reaches the goal only with start lift ~0 |

## Next steps, in order

1. **Object constraint per scene (first; everything else depends on the forces it gives).**
   For each object decide how it is kept in place so that it neither slides nor pinches
   the knife (candidates: no hold for long objects that stay put; side/end hold away from
   the cut plane; friction increase). Check: reference cut at 3 cm/s cuts >= 98% of the
   springs, object shift < 3 mm, no residual force after the cut. Scene preparation only;
   nothing reaches the policy. About half a day on GPU7.
2. **Task definition (user approval of the written formulas before coding).**
   - Success: released cut-spring fraction >= 0.98 without a force-limit event (real robot:
     knife at board height).
   - Reward, identical for every object: progress = released fraction; speed tracking as
     the main shaped term; no step cost that rewards speed; lateral-drift penalty.
   - Feasibility of every training/held-out condition within the limit, from step 1 forces.
3. **Decision experiment: hand-tuned force-feedback speed governor** on the corrected task
   (never implemented so far). If it cuts every object at the commanded speed within the
   limit, the candidate topic is void as stated and one of the topic options is needed.
   If it fails in identifiable conditions, those conditions are the motivation. 1-2 days.
4. **Topic decision (user)** with step 3 and PAPER §3.
5. Base on the corrected task; baselines (FORGE-style recurrent force-penalty policy,
   terminating safe RL or SAC, governor); constraint on predicted peak force; held-out
   evaluation; real robot ([REAL2SIM](../REAL2SIM.md)). Planned in detail after step 4.

## Topic options (discussion, 2026-11-12)

| Option | Idea | Cost |
|---|---|---|
| A. Geometry input | Height profile of the object in the cut plane (simulation: known; real: depth camera). Vision gives where/when contact happens and thickness, force gives hardness | Medium; low-dimensional, no image rendering |
| B. Sawing as a learned action | Use lateral motion to reduce normal force when the limit is near (DiSECt reports large real force reductions from sawing) instead of only slowing down | Medium; lateral action exists, reward changes |
| C. Heterogeneous objects | Skin/core/layers: force changes abruptly, anticipation needed | High; new assets, harder real validation |

Assistant's recommendation: A + B, IROS timeline (deadline usually early March) rather
than a one-month draft; real-robot experiments on a few foods are needed either way.

## Code kept but not results

Risk head + adaptive penalty (`--risk-constraint`), official SuperMimic adaptation
(`cutting_world_model/official`), SAC bridge, WM-MPPI: kept until the topic is decided.
Retired diagnostics are in `scripts/legacy/`. Historical run folders of 2026-11-04..10
are kept on the servers (DATA).

## Storage and execution

Code and data live on GPU7 (`/dataHDD/dong/Projects/Cutting_DEV`, branch `main`) and the
A6000; the Mac keeps nothing. Check `nvidia-smi` before using a GPU.
