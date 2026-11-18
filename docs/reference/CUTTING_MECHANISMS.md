# Cutting mechanisms and task signals

## DiSECt

The prepared mesh has virtual nodes on a predefined cutting interface, connected
by cutting springs. In the current source, spring stiffness is reduced by a term
proportional to positive extension, gated by knife contact, and clipped at zero:

`k_next = max(0, k - contact * max(length-rest_length,0) * softness)`.

The spring and damping forces, material deformation and contact are still solved
by the physical backend. Gradual weakening represents progressive separation
and supports differentiation through a piecewise simulation. It does not guarantee
that a geometric knife sweep will sever every connection. Remaining springs
must not be deleted merely to make a trial successful.

This is a backend choice, not a requirement of WM + BPTT. The learned world model
supplies the derivatives used for policy optimization. Removing the connecting
springs while keeping the pre-split mesh would instead change the initial body
into disconnected pieces. Switching to another complete cutting backend is a
different, explicit choice.

The new shared depth reward uses board-normal knife progress, not released
spring fraction. Physical damage remains available as a diagnostic.
Reference: [DiSECt](https://arxiv.org/abs/2203.10263);
local `backends/mujoco/third_party/disect/disect/dflex/sim.py`,
`eval_cut_springs`.

## BladeMaster reproduction

There are three distinct mechanisms:

1. **Deformation and internal stress.** TLMPM advances particle deformation
   gradients. The fixed-corotated constitutive law has elastic strain energy;
   its stress is the derivative with respect to deformation. Optional plasticity
   and viscosity add irreversible response. It is explicit dynamic simulation,
   not a global minimum-energy solve at each control step.
2. **Cutting resistance.** The active forward solver accumulates directional
   contact work near the finite blade edge. For positive relative advance,
   `delta_W = positive_edge_impulse * relative_edge_speed` (joules), and
   `D_next = min(1, D + delta_W / threshold)`.
   The local threshold is `cutting_energy * particle_volume**(2/3) *
   (1-sharpness)/sharpness`. Once D reaches one, the corresponding front-push
   resistance is removed. This is an empirical cutting-work law; accumulated
   work is not the same quantity as stored elastic energy.
3. **Persistent separation.** The swept blade creates side labels online.
   Incompatible sides cannot exchange momentum through the same grid states,
   and the labels persist after the knife withdraws. This is what removes the
   need for a preselected cut plane. The side-label mechanism is distinct from D;
   a topology result cannot be inferred from a damage-threshold count alone.

The current UR5e interface uses our reproduced forward `Solver`, not every
newer cohesive/adjoint experimental module present in the external repository.
It does not establish exact fracture-energy calibration or exact global energy
conservation in the coupled robot simulation.

Reference: [BladeMaster, Section III](https://arxiv.org/html/2609.27342v2).
Code: `blademaster_warp/material.py` (`corotated_stress`),
`contact.py` (`resist_cut`, side assignment), `solver.py` (threshold),
and grid compatibility in `material.py`.

## Arrival counter

The retained SliceIt-derived pose task increments a counter on control samples
inside the configured goal tolerance. Current success requires the current
sample to be inside, counter > 10, and no failure. This is eleven qualifying
50 ms control samples; it is not the total elapsed step count. The inherited
counter is cumulative, not necessarily consecutive. Until formal termination
the controller can continue producing actions. After termination, the
environment requires reset before another step.

Pose completion and physical separation are different recorded outcomes.
Neither the counter nor the shared-knife work changes that distinction.
