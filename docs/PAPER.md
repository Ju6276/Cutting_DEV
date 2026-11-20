# Paper scope

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

Task pose_depth_v3 (500 steps, cumulative arrival >10, 2 mm tolerance) rewards
reaching the prescribed final pose without higher-priority failure, with
geometric maximum-depth progress shaping. Material separation is reported
independently. The board is a geometric nonpenetration boundary
(`project_continue_v2`), not a board-force model; force thresholds and material
parameters have not been calibrated to the real robot. The hardware target is
UR10e + ATI Mini45 with the user's CAD knife (integration in progress).

DiSECt uses prepared material cut interfaces; native BladeMaster does not
require that topology. Shared robot/task settings do not make their contact
or damage laws identical. Cross-backend results require their frozen
configuration and training budgets to be disclosed.

## Vision scope

Pre-cut vision selects a visible midpoint from depth (geometric or learned, DP3
encoder). Learned localization on known shapes improved over the geometric
reference, but frozen-policy physical handoffs timed out; one analytic-MPC case
reached the pose goal. These are single-case illustrations, not a controller
ranking. This is not evidence of new-object visual
generalization or a paper contribution by itself. See [VISION_ROUTE1](methods/VISION_ROUTE1.md).

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
Prior-art evidence: [reading list](literature/README.md#reading-list).
The force-spike premise and risk-head method are retired.
