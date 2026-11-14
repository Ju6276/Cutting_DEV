# BladeMaster geometry and elastic parameters in DiSECt

Implemented on 2026-11-14. This path retains DiSECt's pre-generated cutting
interface and the UR5e/FDCC `free_cut_pose_v1` task. It adds four candidate
geometries; it does not switch the physical backend.

## Inputs and mapping

Source checkout: `BladeMaster-Warp-forward-check-20261107`,
`4b2d983daeaf9ba1f6a66bc0bb01712b1eb602c5`. Source parameters and local
geometry choices are recorded in that repo's `docs/PARAMETERS.md`,
`blademaster_warp/presets.py`, `geometry.py` and `scenarios.py`.
These are the user's reproduction assets: analytic shapes, not scanned foods.

| Candidate | Source geometry / dimensions | E (Pa) | Poisson ratio | Density (kg/m3) |
|---|---|---:|---:|---:|
| banana | Curved analytic banana in 120 x 60 x 30 mm domain | 400000 | 0.45 | 1100 |
| cake | Cylinder, diameter 160 mm, height 56 mm | 10000 | 0.40 | 1100 |
| tofu | Block, 80 x 80 x 70 mm | 6000 | 0.35 | 1100 |
| luncheon | Block, 100 x 50 x 42 mm | 20000 | 0.40 | 1200 |

The banana's actual bounds are smaller than its sampling domain.
The z-up source frame is rotated into DiSECt's y-up frame without rescaling.
The cake retains its original diameter. Its centre is 40.37 mm from the
fixed x=0 cutting plane, so the slice cross-section fits the existing 150 mm
blade; the other centres use the existing 0.37 mm mesh/cut-plane offset.

Only E, nu and density transfer directly; Lamé coefficients are recalculated.
Source yield stress, fracture-energy scales and sharpness have no equivalent
mapping to the present DiSECt cut-spring law and are recorded as omitted.
All spring, contact, damping, friction and ground values come from the explicit
`configs/free_scenes/potato_003_half.json` template. Runtime net-wrench feedback
uses the current 0.01 m/s friction transition. No Gc-to-spring-stiffness mapping
or food-specific force calibration is introduced.

Meshing uses fTetWild with the same edge/envelope settings as the current food
pipeline. Flat source support faces are preserved and checked for inverted or
collapsed tetrahedra; no particles are fixed. All exported meshes, scene weights,
mesher intermediates and reports remain under server Assets/runs directories.

## Entry points

`scripts/prepare_blademaster_disect_assets.py` runs in the food-mesh environment
and writes meshes, provenance catalog, DiSECt export configs and pose-task configs.
It checks the source revision and refuses to overwrite mesh files.

```bash
/dataHDD/dong/Research/food_meshes/.venv/bin/python \
  scripts/prepare_blademaster_disect_assets.py \
  --source /dataHDD/dong/Projects/BladeMaster-Warp-forward-check-20261107 \
  --assets /dataHDD/dong/Assets/Cutting_DEV/blademaster_disect_20261114
```

Then export each named JSON using the existing original DiSECt exporter in the
SliceIt Python environment. Set `MPLBACKEND=Agg` for headless plotting imports:

```bash
python backends/mujoco/scripts/export_disect_scene.py \
  --disect-root /dataHDD/dong/Projects/SliceIt/disect \
  --config configs/blademaster_disect/banana.json \
  --output /dataHDD/dong/Assets/Cutting_DEV/prepared/sweep/bm_banana_20261114 \
  --steps 0
```

The export-only trace extends below the board to generate the complete prescribed
cut interface. It is not an executable robot path: physical references stop their
command at the board and use the existing finite-blade execution interlock.

Repeat for cake, tofu and luncheon. Pose configs live beside the exporter configs:
`configs/blademaster_disect/pose_{banana,cake,tofu,luncheon}.json`.
They use the existing task and force limits, nominal material multipliers,
30 mm initial clearance and 10 microsecond material substeps. Banana/luncheon
use 0.75 s settling; cake/tofu now use 3 s, following the trace checks below.
These settings define development checks, not a multi-object long-training budget.

```bash
source runs/sac_env.sh
PYTHONPATH=. .venv/bin/python scripts/check_blademaster_disect_assets.py \
  --output runs/bm_disect_assets_reference --max-steps 180 --wall-seconds 900
```

This checks actual construction, topology, zero fixed particles, settling,
reaction wrench, board stop, and the shared pose-task reward under a prescribed
30 mm/s reference. It records pose completion and spring separation separately.
A full episode may succeed or fail; successful loading does not imply a trained
policy or fully separated material.

## Verification record

Server root: `runs/bm_disect_assets_20261114/`.
CPU meshes: banana 254 vertices / 741 tets; cake 609 / 2638; tofu 1202 / 5660;
luncheon 839 / 3757. Exported cutting springs: 53, 202, 436, 218 respectively.

Six focused tests check elastic-unit conversion, preservation of template
contact/damage parameters, no plasticity/fracture substitution, initial geometry
clearance, explicit candidate identities and ground-overlap counts. Physical runs and full-suite
results are appended below after completion.

The completed one-step construction checks used a uniform 30 mm starting gap.
With 10 mm, the soft tofu touched the held knife during settling and the existing
reset guard rejected it. The change raises the tool only; it does not alter
material stiffness, damping, friction or task thresholds.

At 0.75 s, measured COM speeds were 0.049/0.209/0.514/0.093 mm/s for
banana/cake/tofu/luncheon; particle mass-weighted RMS speeds were
0.165/0.323/1.276/0.173 mm/s. No springs released during settling.
These endpoint observations are not continuous-window stationary checks;
the cake and tofu are not certified stationary under the potato preparation
criteria. None of these candidates is automatically added to long training.

An independent CPU load of each exported model ran `validate_topology`
successfully: exactly two FEM components connected only by the undamped cutting
springs, with no fixed particles. Evidence: `topology.json`.
The checker now invokes this function explicitly before recording validation;
pose-task construction alone does not perform the same structural check.

## Longer settling and physical limitations

`settle_trace/report.json` records cake/tofu at 50 ms intervals through 3 s,
using their own nominal materials and the unchanged 30 mm tool gap. The same
continuous 0.25 s preparation criteria as the potato checker were used:
COM speed <= 0.2 mm/s, particle mass-weighted RMS speed <= 1 mm/s, and COM
travel <= 0.01 mm over the window. Cake satisfied every later window from
1.10 s, tofu from 1.95 s. At 3 s their COM speeds were 0.182/0.139 mm/s and
particle RMS speeds 0.304/0.219 mm/s. Knife contact and spring release were zero.
The final candidate configs and generation function therefore use 3 s for these
two objects. The earlier full references below used 0.75 s; their results are
explicitly retained as observations under the old initialization, not evidence
for the final 3 s configuration.

**Loading and pose success do not certify acceptable material/board contact.**
In the 0.75 s reference, tofu's lowest particle was at -4.84 mm initially
and -10.36 mm at pose completion relative to board y=0. Cake was -0.869 mm
initially and -0.799 mm finally; banana was -0.117 and -0.099 mm.
These values come directly from DiSECt material `particle_q.y`, not a rendered
robot-frame transform. `dflex/sim.py::eval_contacts` uses the same y-up normal,
with penetration `min(y - contact_distance, 0)` driving a finite penalty force;
it has no hard position projection. Inheriting another object's ground/contact
parameters does not prevent soft objects from penetrating the board.

No ground stiffness, material parameter or task tolerance was changed to hide
this. All candidates are marked `candidate_not_training_validated`, especially
tofu with substantial material penetration. The checker now records effective
ground stiffness/damping/distance and counts below y=0, -1 mm and -5 mm separately
from the knife's board gap. The earlier full references did not retain particle
positions, so their final below-board particle counts cannot be reconstructed.
The new 3 s construction check records these counts explicitly.

## Completed nominal UR5e reference episodes

All four use the retained 180-step horizon, 30 mm/s board-ending reference,
same reward/force limits, and **0.75 s settling**. Banana/cake/tofu completed
in the first 900 s wall budget. The first budget expired during luncheon;
that partial run is retained and excluded from outcomes. Only luncheon was
re-run under the additional 1200 s cap. GPU7 was shared with the main task;
these runtimes are not throughput measurements.

| Object | Pose success | Steps | Return | Released/total springs | Peak material force (N) | Max six-axis utilization |
|---|---|---:|---:|---:|---:|---:|
| banana | yes | 66 | 105.52 | 43/53 | 1.18 | 0.024 |
| cake | yes | 68 | 106.56 | 174/202 | 10.20 | 0.312 |
| tofu | yes | 73 | 104.77 | 326/436 | 14.25 | 0.299 |
| luncheon | yes | 66 | 107.33 | 189/218 | 0.64 | 0.017 |

No case triggered force-limit, workspace or knife-through-board failure.
None fully separated. Minimum recorded knife-board gaps were respectively
+0.058, +0.414, +0.979 and +0.075 mm. Luncheon's material minimum y was
-0.705 mm at reset and -0.313 mm finally. Material penetration and pose success
are separate facts; the table is not evidence of fully cut or calibrated foods.

Reports: `reference/report.json`, `reference_remaining/report.json`, and
`reference_summary.json`. Actual per-case configuration is embedded in each
raw report. Different shapes and material parameters both change across these
cases, so the force differences do not isolate a geometry-only effect.

## Final config reset check (3 s)

The actual candidate JSONs were then loaded through `MultiConditionEnvironment`,
with each world reset using its own nominal material for 3 s, followed by one
physical UR5e reference step. Both cake and tofu completed this one-step budget
without exception. The embedded configs confirm `settle_seconds=3.0`.
This is a reset/step check, not another complete reference episode.

| Object | Reset particle RMS (mm/s) | Initial minimum y (mm) | Particles below board | Below -1 mm | Below -5 mm |
|---|---:|---:|---:|---:|---:|
| cake | 0.303 | -0.937 | 120 | 0 | 0 |
| tofu | 0.218 | -4.702 | 121 | 11 | 0 |

After the one step, minima were -0.938/-4.703 mm with unchanged counts.
The effective kernel settings in both cases were `contact_ke=1000`,
`contact_kd=0`, and `contact_distance=0.00001 m`. These effective values are
reported rather than assuming JSON ground-radius aliases set the kernel value.
Longer settling reduces motion; it does not eliminate material penetration.
Evidence: `final_3s_smoke/report.json`. All six focused CPU tests passed;
combined root/backend regression is run by the parent integration task.

Integrated validation in the shared cutting worktree: **366 root tests passed, 5 skipped; 38 backend tests passed**. Logs: `runs/half_potato_integration_verified_root_20261114.log` and `runs/half_potato_integration_backend_20261114.log`. The earlier partial agent suites are not the final acceptance record.
