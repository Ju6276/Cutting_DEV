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

The reviewed board fixes and platform work are consolidated for the authorized main merge.
No formal UR10e long run has been launched. The verification scope and remaining
physical limitations below are not changed by merging.

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
`/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_fdcc_v2` (current; old bundle retained for recorded experiments).
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
termination/reset/next-step checks (`check_reference_episodes.py`, two-hour
budget, 500-step horizon) were not complete at d7e05a2: DiSECt
`disect_reference_midblade9_gpu1` (A6000), BladeMaster `bm_reference_midblade9`
(GPU7). At DiSECt step 151, 0/9 episodes had ended. Reconstructed historical minimum
blade clearance was3.49–4.62mm, using initial FK and monotone depth. This is
not instantaneous target error and does not identify the cause of stalling.
Evidence: `depth_gap_diagnostic.json` in that run (A6000).

Three-baseline physical smoke checks (`check_baseline_adapters.py --live`) follow
the reference checks; each reuses one physical initialization and executes short
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
The reference check also includes WM reward/termination checks.
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


## 2026-11-21 FDCC correction and render consolidation

The nominal 5000 N/m direction incorrectly moved from vertical to horizontal when
UR5e stiffness values were copied into UR10e knife_mount axes. The compiler now
inserts knife_compliance at the same origin and compensates the task-TCP fixed
transform. Physical TCP, knife and ATI poses are unchanged. Reference stiffness
and both policy stiffness channels use the corrected axes.

Real A6000 reference comparison: old cucumber capture ended at step 80 with
5.713 mm blade gap and no success; corrected run succeeded at step 63, gap
1.482 mm, cumulative arrival 11, no force-limit/workspace failure. This is one
reference condition, not a learned-policy result or all-material acceptance.
Evidence: new_knife_fdcc_v2/motion.json and new_knife_video/motion.json.

The current food configs contain cucumber and HALF potato (prism removed by user
decision, 2026-11-21). New formal UR10e
config uses the corrected fdcc_v2 bundle. Source-scene physics is not changed by
the presentation renderer.

One presentation entry: scripts/render_scene.py. Recorded platform identity is
mandatory; pinned scan sources/colors and camera/board settings are in
configs/render/studio_v1.json. Missing assets fail instead of silently changing
colors. Pure-white PLA and metal ATI/IKEA are explicit. Exterior normals no
longer average prepared internal cut faces into the food skin. No physical
vertices are moved to hide a real gap. Batch diagnostic views are separate.

OmniObject3D banana_001 (Object.usd and textures/Scan.jpg) was downloaded from the
AlbertS666 mirror. Its inspected archive has no tofu/luncheon category. Existing
procedural banana/tofu/luncheon geometries have deterministic synthetic appearance,
explicitly not attributed to that scan. The banana scan is a future physical-asset
candidate, not silently substituted for the existing banana contact geometry.
Physics preset provenance remains in configs/blademaster_disect/catalog.json;
transferred elastic values are not real-food calibration and fracture models
are not interchangeable.

## Merge verification (2026-11-21)

GPU7 final root suite: 771 passed, 25 skipped (merge_root_suite.log).
Backend suite from its documented working directory: 44 passed
(merge_backend_correct_cwd.log). An earlier invocation from repository root failed
14 file-relative fixture loads; it was a test invocation error, not a repaired
physics defect, and was rerun from backends/mujoco.

Registered UR10e half-potato/cucumber stills and the wide scene were rendered and
visually inspected with the same appearance/camera/board contract. Both material
backends export the actual CAD knife and pure-white PLA. The MPM presentation maps
scan colors through interpolated particle displacement instead of averaging coarse
particle colors, avoiding a separate blurred appearance. The surface reconstruction
still differs from FEM, as the physical representations differ.

The legacy three-object DiSECt reference check finished 500 steps with status passed
on A6000, covering natural timeout/reset. It used the earlier midblade controller
bundle; it must not be cited as a success-rate test for corrected fdcc_v2.
Corrected-controller evidence remains the same-condition cucumber success at step 63.

Mac delivery location is now /Users/judong/Documents/Cutting_DEV/Media:
current holds current reviewed images/videos; archive preserves earlier experiments
and superseded previews. moved_20261121.json records 24 moves from Downloads.
No experiment data was deleted. tams61 receives code/docs only; run-host assets and
the native BladeMaster CAD dependency are synchronized separately.
