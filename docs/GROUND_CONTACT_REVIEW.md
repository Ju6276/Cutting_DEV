# DiSECt ground-contact diagnostic correction

Scope: BladeMaster-derived DiSECt candidates, especially tofu. No physical
kernel, contact parameter, task limit, reward or primary three-object configuration
is changed. The physics remains `edge_coordinate_consistent_v2`.

## Cause

DiSECt's original `meshing/meshing.cpp` states that `contactless_particles` lie
on the "empty" side of a cut tet. `dflex/model.py` exports this as `contact_mask`.
In `dflex/sim.py`, `eval_contacts` applies ground forces only to enabled nodes;
`eval_hard_particle_contacts` also handles actual cut-surface points interpolated
as `(1-t)*q[i] + t*q[j]`, distributing force with those same weights.
The two geometric sets describe contact. A ghost endpoint used for interpolation
is not itself an occupied material point.

The former asset checker used the minimum over every FEM node and described
-4.7 to -10.4 mm as tofu penetrating the board. That interpretation was not
justified. This was a reporting bug: `ground_stats` is used by the asset-check
script, not the learning, reward, termination or physics execution paths.
The diagnosis could mislead platform decisions but did not directly change
policy training. New candidate assets are not in the primary training set.

## Repair

The checker now records per world: all FEM nodes, ghost nodes, contact-enabled
nodes, cut-surface interpolation points, and the union of the two actual contact
sets. Each set records count, minimum y and counts below 0/-1/-5 mm. Ghost
excursions remain visible, but are not labeled physical penetration. Every
reference control step records these groups, so initial/final and transient
sampled minima can be compared. The finite-penalty ground law can still permit
small genuine overlap; those values are reported without claiming a hard
nonpenetration guarantee.

Tests distinguish a ghost 10 mm below ground from an actual interpolated cut
point 4 mm below ground and verify packed-world indexing. Seven focused asset
tests pass. Integrated GPU7 regression: root388passed/5skipped and backend38passed.

## Bounded physical check

Question: does a full nominal tofu cut contain large overlap of actual contact
geometry, or only extrapolated ghost nodes? Use the existing 3 s reset, nominal
material, 30 mm starting gap, 30 mm/s board-ending reference, 180-step ceiling,
50 N/5 N m simulation limits, and 700 s wall cap. No optimizer or parameter
search. Record peak knife force, force utilization, knife-board gap, spring
release and separate physical/ghost ground coordinates. Evidence lives in
`runs/ground_contact_20261114/` on GPU7. Initial snapshot: `inspect.json` and
`settled.pt`; complete reference: `reference/report.json`.

At 3 s reset, the lowest ghost was -4.702 mm; lowest enabled contact node
-0.096 mm; lowest interpolated cut point -0.074 mm. Eleven ghost nodes, and
no actual contact points, lay below -1 mm. Effective ground parameters remain
stiffness 1000, damping 0, contact distance 0.00001 m.

## Completed physical result

The nominal tofu reference completed at step 70 with pose success, return
105.744, and 344/436 released springs (not full separation). Peak material
force was 12.113 N; maximum six-axis limit utilization 0.305; minimum recorded
knife-board gap +0.868 mm. No force/workspace failure or nonfinite state occurred.
The knife's no-through-board interlock remains unchanged.

| Geometry set | After 3 s reset, minimum y (mm) | End of reference (mm) |
|---|---:|---:|
| All FEM nodes / ghost minimum | -4.702 | -9.551 |
| Contact-enabled nodes | -0.096 | -0.296 |
| Interpolated cut-surface points | -0.074 | -0.222 |

Across the recorded control frames, the deepest actual contact-geometry point
was -0.309 mm and deepest ghost was -10.396 mm. No actual contact point in these
frames was below -1 mm. Measurements are at 50 ms control boundaries, so these
are sampled extrema, not a bound on every 10 microsecond material substep.
The current reference uses 3 s settling, whereas the historical -10.36 mm
all-node report used 0.75 s. It is a new physical check, not reconstruction of
the unsaved old final coordinates.

Conclusion: the observed order-of-10-mm excursion in the new check belongs to
extrapolated ghost degrees of freedom, not occupied contact geometry. The
reporting correction is sufficient for this demonstrated issue. A genuine
approximately 0.3 mm penalty-contact overlap remains; it is disclosed rather
than hidden by clamps, material tuning or a success-threshold change. There
is no new evidence here requiring a physical kernel correction. This does not
validate realistic food contact parameters, full material separation, or any
long-training claim.
