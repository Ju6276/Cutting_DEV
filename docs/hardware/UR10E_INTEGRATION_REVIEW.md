# UR10e integration review handoff — 2026-11-20

## Review scope and branch
Cutting_DEV: codex/ur10e-platform, based on main f12ca04.
Claude's complete board-contact-fixes series through 9ea91b2 is an ancestor.
Independent review: 8ae1bf8; follow-up reset/measurement corrections: 51b2c0b.
Review record: [RESULTS](../RESULTS.md).

Native backend: [BladeMaster-Lab 166a269](https://github.com/Ju6276/BladeMaster-Lab/commit/166a269),
branch codex/ikea-mesh-tool, based on c51ef733364aa9cbbbceb6ba5b422cabf67ddefc.
The native branch is a forward robot-integration candidate; full differentiation
acceptance is excluded by the user's current scope.

This is a review candidate, not a completed long-training acceptance report.
No formal UR10e long run has been launched. Main has not been merged.

## Fixed task contract
pose_depth_v3 / project_continue_v2; no permanent freezing. Cumulative arrival >10,
2 mm height tolerance, existing +100/-100 rules, 500 action-step episode limit,
50 ms action interval, gamma=1. DiSECt material dt remains 10/8/10 microseconds
(cucumber/half-potato/prism). Final horizontal radius remains 40 mm; process slip
penalty disabled. Pitch policy channel disabled. Board geometry blocks penetration;
there is no knife-board reaction-force model. Material separation is diagnostic.

## Implementation
- Explicit UR5e/UR10e robot profiles, model/controller hashes and ATI sensing frame.
- Actual mounted IKEA CAD mesh and 56-point curved edge, shared by physics/render.
- DiSECt grid SDF with winding-number inside/outside sign; re-exported cut topology.
- BladeMaster mesh contact/curved-edge side updates; no prepared material cut plane.
- Robot-specific baseline controller loading; reject incompatible robot checkpoints.
- Six-axis force observation unchanged. Sensor transforms preserve force/moment origin.
- Reset board-contact state derives from pose. Float32 endpoint rounding respects
  the existing penetration bound without changing tolerances.
- Dependencies consolidated in requirements/ and the root installation entry.

Asset bundle:
`/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_midblade_v1`.
Paired configuration:
`/dataHDD/dong/Assets/Cutting_DEV/ur10e_shared_midblade_20261120/`.
The fixed nominal pose supports the central curved edge with 8.25-degree tip-down
pitch; the policy cannot vary pitch. Knife net mass 0.170 kg is user-provided;
PLA inertial values remain CAD-volume estimates. ATI range/clocking require the
actual hardware certificate/calibration before deployment.

## Available verification
All paths below are under
`/dataHDD/dong/Experiments/Cutting_DEV/runs/ur10e_platform_20261120/`.

| Evidence | Result | Location / host |
|---|---|---|
| Root suite | 778 passed, 21 skipped | root_suite_final1.log / GPU7 |
| Backend suite | 44 passed | backend_suite_current.log / GPU7 |
| UR10e DiSECt learning | 9 worlds; WM20 / actor2 / critic40 optimizer steps; manual partial reset, save/restore passed | disect_learning_midblade9/report.json / GPU7 |
| UR10e BladeMaster learning | Same update counts; reset/save/restore passed | bm_learning_midblade9/report.json / GPU7 |
| Baseline real-record adapters | Each backend: WM+MPPI, analytic MPC planning; SAC2 updates on60 recorded transitions and exact weight reload passed | *_baseline_adapters_cpu/report.json / GPU7 |
| Baseline regressions | 25 passed | baseline_adapter_regression.log / GPU7 |
| Requirements | Original pins retained;10 environment tests passed | dependencies_reorganization.json / GPU7 |
| Mesh distance | 10,000 surface-near samples; max error0.06352mm | sdf_winding.log / GPU7 |

The 9-world learner runs contain zero naturally ended episodes. Natural
termination/reset/next-step checks are running separately:
DiSECt on A6000 GPU1 (`disect_reference_midblade9_gpu1`);
BladeMaster on GPU7 (`bm_reference_midblade9`).
They have a two-hour budget and do not change the500-step task horizon.
At DiSECt step151,0/9 episodes had ended. Reconstructed historical minimum
blade clearance was3.49–4.62mm, using initial FK and monotone depth. This is
not instantaneous target error and does not identify the cause of stalling.
Evidence: `depth_gap_diagnostic.json` in that run (A6000).

Both three-baseline physical smoke jobs are queued after their corresponding
reference check; each reuses one physical initialization and executes short
planner/SAC steps. Offline adapter checks are not closed-loop performance results.
No gradient full-suite or hundreds-world retest blocks this user's main workflow.

## Timing comparison
A6000 GPU0, current code,9 worlds each, unchanged material/action dt.
Two warmup steps, then three timed steps, CUDA synchronization around components:
UR5e mean5.494s/action (material2.570s);
UR10e mean9.354s/action (material6.064s).
Most of the increase is in material advance. Geometry/topology changed along with
the robot, so this is a whole-platform comparison, not isolated robot inertia cost.
Evidence: `paired_platform_timing_gpu0.json` / A6000.
The running reference check also includes WM reward/termination checks.
No11-second wait is configured.

## Review requests and remaining work
1. Check CAD coordinate conventions, curved-edge dispatch and batch material isolation.
2. Check UR10e FDCC/ATI loading across training, SAC, both planners and render paths.
3. Check no historical latch logic applies to project_continue_v2.
4. Inspect natural terminal/reset outputs when available and the final target pose.
5. Check profile/asset mismatch rejection and current entry-point documentation.
6. The unused RobotPlatform.require_cutting_ready helper still rejects custom
   platforms unconditionally; resolve/remove it before publishing a ready status.
7. Separate source-at-start from source-at-end when reading live-probe provenance:
   an earlier probe version rewrites source_commit at successful completion.
   Initial source_status and copied validation sources identify the running version.

Final merge is authorized by the user once the applicable checks complete.
Each material backend can become ready independently. A finite launch budget and
online W&B are required for formal training; baseline performance and physical
knife-board reaction validation are separate from this interface acceptance.
