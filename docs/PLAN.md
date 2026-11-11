# Plan

Updated 2026-11-12. The only active plan. Topic and prior art: [PAPER](PAPER.md);
measurements: [RESULTS](RESULTS.md); code: [METHOD](METHOD.md); data: [DATA](DATA.md).
Previous plans: [PLAN_20261111](archive/PLAN_20261111.md) (S3 launch, retired
problem statement), [PLAN_20261110](archive/PLAN_20261110.md).

## Where we are

- The first multi-object runs (base, base + risk penalty, official SuperMimic on the
  prism) are stopped. The risk and official runs failed; the base run exposed defects
  in the task definition (RESULTS 2026-11-12). All three shared these defects.
- The problem statement "WM under-predicts the contact-onset force spike" is retired
  (PAPER §1). The candidate topic is force-adaptive cutting of unknown objects under
  one hard force limit (PAPER §2); **the user has not decided it yet**.

## Fixed decisions (user)

- Method paper. Platform, objects and baselines are experiments, not contributions.
- One shared WM/policy/critic over scanned objects, randomized materials, knife start
  height, target cutting speed as a task input.
- One force limit for every object; the learner gets no object identity, material
  label or per-object setting and adapts from force feedback. Simulation limit in the
  first runs: `[50, 50, 50 N, 5, 5, 5 N·m]`, a simulation setting, not a real-robot
  threshold (external ATI Mini45 on the real robot).
- No per-object reward or success parameters (they would hard-code object identity).
- One change at a time; a plan is approved before a paid run.

## Task defects found (RESULTS 2026-11-12)

| # | Defect | Evidence |
|---|---|---|
| 1 | Success = knife at one fixed pose (the prism's board height), not "object cut" | Held potato: 245/245 springs cut, not success. Unheld potato: pushed 32 mm, 57/245 cut, success |
| 2 | Bottom-held halves cannot spread: residual force after the cut keeps the knife above the goal | 14.7 N (potato), 34-47 N (cucumber_003) after every cut spring is released |
| 3 | Reward makes "as fast as possible" optimal; lateral drift is free | Policy 20-27 cm/s against a 1-5 cm/s command; prism force-limit hits at that speed; steady x drift to the workspace limit |
| 4 | The reference trajectory is a fixed displacement | Reaches the goal only when the start lift is ~0 |

## Next steps, in order

1. **Task definition (needs user approval before coding).** Proposal:
   - Success: released cut-spring fraction >= a threshold (e.g. 0.98) without a
     force-limit event; real-robot equivalent: knife at board height.
   - Object stays held by its bottom (unheld objects are pushed away).
   - Reward, identical for every object: progress = released fraction; speed tracking
     as the main shaped term; no step cost that rewards speed; lateral-drift penalty;
     force-limit event = failure.
   - Check every training/held-out condition is feasible within the limit, or define
     "stop without violating the limit" as the correct outcome when it is not.
2. **Topic decision (user)** with the prior-art record (PAPER §3).
3. **Base only on the corrected task** (A6000): does one policy cut all training
   objects at the commanded speed within the limit? Deterministic evaluation, not
   training episodes, decides.
4. **Baselines the reviewer will ask for**: model-free force-penalty recurrent policy
   (FORGE style), terminating safe RL (SRL-VIC/CaT style) or SAC, hand-tuned
   force-feedback speed governor. Official SuperMimic needs a non-negative step reward
   or a failure penalty at soft termination before it is a fair comparator.
5. **Constraint on predicted peak force**: normalised objective or PI-Lagrangian,
   every failure ending counted; one change at a time.
6. Held-out evaluation, then real robot ([REAL2SIM](REAL2SIM.md)).

## Storage and execution

Code and data live on GPU7 (`/dataHDD/dong/Projects/Cutting_DEV`, branch `main`) and
the A6000; the Mac keeps nothing. Both A6000 GPUs and GPU7 are free of our jobs since
2026-11-12. Check `nvidia-smi` before using a GPU.
