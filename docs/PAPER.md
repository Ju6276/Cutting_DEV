# Paper scope — 2026-11-17

## Research question and implemented system

Study whether a force-conditioned controller, optimized through a learned
world model, can execute a specified single cutting motion across varying
objects and material responses. Pre-cut vision proposes a geometric goal;
motion and wrench history drive the low-level controller during execution.

The implemented low-level system is a shared policy within each backend,
34 observations per frame x10 history plus an8D base-frame goal, six action
outputs (pitch disabled), FDCC, learned dynamics/wrench prediction, real-data
critic,8-step BPTT and BC.25. Exact definitions live in [METHOD](METHOD.md)
and [TASK_FREE_CUT](TASK_FREE_CUT.md); the combination is not by itself an
established novel algorithm.

## Task and claim boundaries

Current pose_depth_v2 rewards reaching the prescribed final pose without
higher-priority failure. Geometric maximum-depth progress replaces the old
spring-release reward. Material separation is reported independently.
The board stop is a software execution boundary; force thresholds and material
parameters have not been calibrated to the real robot.

DiSECt uses prepared material cut interfaces; native BladeMaster does not
require that topology. Shared robot/task settings do not make their contact
or damage laws identical. Cross-backend results require their frozen
configuration and training budgets to be disclosed.

## Vision scope

Route1 selects a visible midpoint from depth. Its frozen-policy physical
handoff ran but timed out; successful general visual cutting is not established.
Method2 has been trained on fixed-heading, small-translation cases of the three
existing shapes and connected to the frozen policy, as a process pilot. Its held-out known-shape localization improved over the geometric reference,
but both RL physical trials timed out. Later DP3 controller clips also include
a paired WM+MPPI timeout and one analytic-MPC pose completion. These are
single-case illustrations, not a controller ranking. This is not evidence of new-object visual
generalization or a paper contribution by itself. See [VISION_ROUTE1](VISION_ROUTE1.md).

## Comparisons and evidence still needed

- WM+BPTT versus SAC and fixed feedback/reference control under explicit common
  task conditions; matched-budget claims need actual matched data accounting.
- Frozen WM+MPPI separates online planning from policy execution; its input
  access and computational cost must be disclosed.
- Nonlearned approximate FDCC+MPPI is an engineering baseline, not a reproduction
  of a published cutting-MPC method or calibrated material dynamics.
- Contact/force/action traces and controlled ablations are needed to support
  adaptation claims. Feature clusters or a successful clip alone do not prove
  material identification or causal use of force history.
- Real-robot validation is needed for claims about physical food separation,
  sensing accuracy or safe force regulation.

No new superiority, safety or novelty claim follows from current integration.
[PLAN](PLAN.md) is the only task list; [RESULTS](RESULTS.md) holds measurements.
Prior-art evidence remains in [NOVELTY](NOVELTY.md) and [READING_LIST](READING_LIST.md).
The task/force-spike discussion from November12 is historical:
[previous paper scope](https://github.com/Ju6276/Cutting_DEV/blob/4653d10/docs/PAPER.md).
The force-spike premise and risk-head method are not the selected main route.
