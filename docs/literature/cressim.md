# CRESSim-MPM — surgical cutting/suturing simulator

Reviewed 2026-11-10. [Paper, 2502.18437v3](https://arxiv.org/abs/2502.18437).
Read scope: MPM formulation, cutting/contact construction, coupling,
implementation, demonstrations and limitations. Equations are read for
architecture, not independently re-derived or reproduced.

## Method and evidence

Standard, MLS and position-based MPM solvers share surgical geometry/contact
support. Slicer signed-distance information distinguishes sides for transfer;
blade-spine handling prevents treating the blunt side as a cutting edge.
Rigid contact corrects nodal relative velocities; accumulated momentum change
is returned to an external rigid-body solver. Unity is an integration layer,
not the material engine.

Demonstrations cover scalpel cutting and suturing with arc/segment geometry.
The manuscript explicitly leaves broader performance comparisons and additional
material models to future work. These examples do not validate food calibration,
policy learning or cutting-gradient correctness.

## Author code inspected

[yafei-ou/CRESSim-MPM, 09aa5009b8580351f516b6df7660e87821fc5eb6](https://github.com/yafei-ou/CRESSim-MPM/tree/09aa5009b8580351f516b6df7660e87821fc5eb6).
Read `src/solver/mpm_rigid_contact.h`: rigid velocity includes angular
contribution; momentum and torque accumulate via atomic adds; distance,
normal velocity, friction and spine mode determine collision response.
The optional `UseEffectiveMass` branch is explicitly labeled uncertain in
author comments and defaults false in this template. That comment is a
limitation to check before reuse, not evidence that our project has that bug.

The repository provides CUDA/C++ and C interfaces, with external two-way
coupling requirements. No automatic differentiation path or learned controller
was verified, and no build/run was executed.

## Relevance

Evidence that cutting geometry and reaction feedback can be a standalone module.
It supplies simulation concepts, not a ready replacement for the current
WM+BPTT method. No simulator migration is proposed by this review.
