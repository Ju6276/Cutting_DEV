# Contact correction: review handoff — 2026-11-12

Scope: re-review of commits `38b85a5` (net-wrench interface) and `4b229df`
(contact coordinates), plus this documentation reconciliation. This is Codex's
self-review for Claude to check, not an independent external review or a new
training result. Current task decisions remain in [PLAN](PLAN.md).

## What was wrong, and why the correction follows

In the bundled `backends/mujoco/third_party/disect/disect/dflex/sim.py`,
`edge_knife_sdf_collision` searches the segment at
`p=(1-u)*x1+u*x2`, but previously returned `c=1-u`.
Its force consumer gathered position and velocity with `(1-c,c)` while
scattering contact force to particles with `(c,1-c)`.

Consequently, the old **SDF search point and particle-force distribution agreed**.
The inconsistent part was the opposite position/velocity used to evaluate contact
force and report the knife moment. Calling this only a reversed-force-weight bug
was incomplete. Merely changing the scatter weights would move the applied force
away from the original search point.

The repair returns `c=u` and uses `(1-c,c)` for position, velocity and force
distribution. At a fixed contact, the following identities then hold:

- `f1+f2 = F`.
- `x1 cross f1 + x2 cross f2 = p cross F`.
- `v1 dot f1 + v2 dot f2 = v_contact dot F`.

These are transfer identities, not global energy conservation or an automatic-
differentiation proof. The lower cut side reverses its endpoints in search,
interpolation, scatter and wrench reconstruction alike. The optional triangle
producer already returns the second-endpoint coordinate; only its scatter needed
the corresponding correction.

`state.knife_f` stores forces **on the material**. The knife wrench uses their
negative sum and negative contact moment. Its contact points use
`knife_contact_q`, cached before the material integrator overwrites positions;
using post-integration positions with the earlier forces would add a timing error.
The preceding interface repair also corrected rigid wrench transforms and checked
the final net-mode coupling sample for force-limit termination.

## Evidence and its boundaries

The CUDA tests launch production kernels, rather than a reimplementation of the
interpolation formula. They cover the forced endpoint, both cut sides, four packed
worlds, an SDF re-query at the reconstructed point, and triangle force scattering.
The four-world case checks isolation/accounting; it is not a heterogeneous-object
or large-scale throughput test. The triangle test supplies contact data; it does
not validate the complete triangle collision-search algorithm.

Before/after kernel evidence is under
`runs/codex_contact_coordinate_20261112/` on GPU7:
`before.json`, `after.json`. The endpoint moment/power residuals change from
0.164082 N m / 1.449986 W to zero; the four-world residuals after correction are
about 2.29e-9 N m / 7.16e-9 W.

The same directory contains `press_saw/report.json` and `comparison.json`:
four real physical rollouts (E scales 2 and 1; press and prescribed saw).
Mean net forces change by less than 0.6% compared with the pre-coordinate repair.
All reach the historical pose goal and the 98% release diagnostic. Neither is
proof of complete physical separation. These runs disable force-limit termination
for measurement; they cannot establish constrained-task success or a learned
sawing policy. No new full cut was needed for this second review because physics
and measurement code have not changed since that rerun.

## Test change requiring explicit review

The original 4 ms packed-vs-single spring-stiffness equality failed at one spring
(about 15.68 vs 16.18). Identical single-world repeats also showed that gap, with
position differences below 2.5e-7 m. A synchronized trace did not reproduce the
split. Accumulation-order/contact-branch sensitivity is a plausible explanation,
not an isolated root cause.

The replacement keeps the original tolerances for same-input **one-substep**
position/stiffness, retains the 4 ms position comparison, and checks finite,
nonnegative, irreversible spring stiffness. It removes the old 4 ms per-spring
equality requirement. This is **reduced long-horizon damage-equivalence coverage**,
even though no tolerance was increased. It does not prove the original difference
harmless or resolve its trigger. Original failure/repeat logs remain available.
The second review corrected comments that prematurely asserted a particular
contact-branch switch.

## Provenance and reproduction

Runtime physics revision: `edge_coordinate_consistent_v2`. It is included for both
scalar and net feedback and propagated to grouped environments. WM-policy/MPPI
metadata comparisons and training-resume comparisons retain this field.
Do not remove it to treat old checkpoints as same-physics results.

`38b85a5` is the immediate pre-coordinate-fix reference. To reproduce an older
experiment exactly, use that experiment's recorded commit/configuration, not
assume all old runs used `38b85a5`. Isolate kernel caches. The separately pinned
Gazebo/SliceIt patches were not changed. Licenses and upstream attribution remain.

## Documentation reconciliation

Current README, PLAN, RESULTS summary, script map, backend METHOD, BENCHMARK,
RUNNING, PAPER and REAL2SIM distinguish historical scalar feedback from optional
net feedback, old peak envelopes from current simultaneous samples, and historical
task completion from unvalidated separation. Sawing is an approved task direction;
the early negligible-effect result is no longer described as the latest universal
finding. REAL2SIM no longer prescribes unclamped specimens against the current
bottom-held simulation decision, or asserts all gradient fitting requires a fixed
bottom. Dated history and literature evidence are retained, not rewritten as new
results.

Fresh verification logs for this review:
`runs/codex_contact_recheck_20261112/backend.log` and `root.log`.
Fresh checks: backend **36 passed**; root **286 passed / 5 skipped**.
The corresponding [RESULTS](RESULTS.md) entry records timing, skips and artifacts.
All eight prior physical-run source hashes still match. The documentation check
repaired 31 relocated archive links; all 219 local file targets exist.
No long training, reward changes, material retuning or new acceptance thresholds.
Claude's untracked `scripts/record_reference_cut.py` is outside these changes.

## Remaining task decisions, not another repair queue

Choose/validate the physical completion criterion, realizable fixture, force
feedback convention and material/friction calibration before new training claims.
This patch resolves the demonstrated coordinate inconsistency; it does not settle
those decisions, validate whole-simulator adjoints, or certify every physical model.
