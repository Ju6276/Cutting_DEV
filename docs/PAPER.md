# Paper scope

Updated 2026-11-13. This describes the research scope, **not an established novelty
claim**. [PLAN](PLAN.md) owns experiments; [TASK_FREE_CUT](TASK_FREE_CUT.md) owns the
simulation task; [RESULTS](RESULTS.md) owns measurements.

## Task and intended application

One knife cut through a freely placed deformable object, with complete separation,
a commanded feed speed and measured wrist-force feedback. Small passive movement
is allowed. Active rolling and multi-cut planning are outside the current scope.
The eventual aim is one controller across materials and geometries, with
sim-to-real validation; the current run uses one cucumber scene.

The simulation stop rule and wrench limits are engineering task conventions.
They do not establish a real-robot stop detector, calibrated material properties
or Mini45 safety guarantees. The real setup must be measured separately.

## Implemented candidate

A learned robot/force/progress world model supplies short imagined trajectories.
BPTT optimizes a policy using predicted task reward and a terminal critic estimate.
The current configuration uses real-replay critic targets, an eight-step actor
horizon and BC regularization at 0.25. The deployed policy does not read simulator
cut progress; the training WM currently does. Architecture and interfaces:
[METHOD](METHOD.md).

These ingredients have precedents. Their combination is an implemented candidate,
not automatically a new generic RL method. The optional risk head, official-source
adaptation, MPPI and SAC remain separate implementations, not adopted additions to
the active run.

## What this experiment answers

The short v3 run improved physical completion while developing optimistic model
and value estimates on failed lateral-motion trajectories. More joint training is
being measured under a fixed budget to see whether incoming data corrects that
mismatch. This is development evidence; it does not establish algorithm superiority,
sample efficiency, multi-object transfer or a new failure mechanism.

Do not derive the paper's motivation from an unvalidated simulator or one weak
checkpoint. The earlier force-spike premise and first S3 comparisons were retired;
their evidence and corrections remain in [history](archive/README.md).

## Evidence still needed for a paper claim

- A literature-grounded cutting problem and a concrete mechanism addressing it.
- Credible task/physics assumptions and independently evaluated control results.
- Comparisons appropriate to the eventual claim, with information access, training
  budget and online computation disclosed.
- Cross-condition/object evidence for generalization claims; real-robot evidence
  for claims about physical cutting or sensor-force safety.

These are research requirements, not instructions to launch additional jobs.
Candidate precedents and their reading limitations are in [NOVELTY](NOVELTY.md)
and [the paper/code ledger](literature/README.md). The current implementation and
diagnostic findings alone do not fix the paper title or method contribution.
