# Shared cutting scene — 2026-11-16

Authoritative input: [scene_v1.json](../configs/shared/scene_v1.json).
Both backend configurations are generated from this file; do not independently
edit their generated JSONs. Case sampling is identical in checks and eventual
training. Old runs retain their frozen configuration and are not paired runs.

## Starting the cut as on the intended robot

1. Place the object, let it settle using its actual material parameters.
2. Establish the board frame, cut location and object height.
3. Position the actual mounted blade above that individual object with a sampled
   5–30 mm clearance. This is free space, not placement in contact with food.
4. Cut using motion and force history. No camera image is supplied to the policy.

Current implementation uses the settled **whole object's highest physical particle**
as a conservative surface-height proxy. It is not a local surface estimator.
A real robot would obtain placement/height before cutting from perception or
measurement. Camera localization, its errors, and calibration are NOT implemented
or validated by this reset helper. The 5–30 mm distribution is the existing
engineering training range, not a measured best real-robot clearance.

A case is identified by object name, case seed and world index. Independent seeded
sampling generates explicit per-world clearances before either backend starts;
changing backend RNG use or object-group order does not change a case.
The sampled gap stays fixed for that world. After reset/settling the knife is
repositioned to the new surface height with that same gap. Other worlds retain
their state exactly. This follows the former main run's once-per-world sampling;
it does not mean a new gap is sampled every episode.

There is no separate fixed-5-mm debug task. Checks replay selected cases from
this same definition. Small worlds_per_object is a resource setting, not a
different initialization/reward implementation. Material state is not teleported
to fit the knife. If the requested start is outside the robot workspace, setup
fails rather than expanding the workspace silently.

## Complete shared configuration

| Item | Both backends under scene_v1 |
|---|---|
| Objects / allocation | Cucumber 003, flat-bottom half potato, prism; equal world count per object |
| Object sampling | Fixed geometry group per world; one shared learner, not sequential object training |
| Robot and controller | Same UR5e asset, FDCC parameters, action mapping and sensor frame |
| Physical knife / mount | disect_knife_rigid_mount_v1; same finite blade and rigid EE transform |
| Cut location / orientation | Prepared cut location; nominal vertical knife, no automatic cut-site search |
| Start | Per-object settled top + uniform 5–30 mm; paired explicit cases |
| Settling | 1.0 s simulated time with that episode's actual material parameters |
| Goal | Shared board-aligned prescribed pose, final XY radius plus z/pitch tolerances and arrival counter |
| Board | Height 0; finite-blade stop at 1 micrometre; support plane has no finite edge |
| Workspace | EE in robot-base frame: x[.038,.238], y[.405,.605], z[.096,.269] m |
| Force / torque limits | [50,50,50,5,5,5], componentwise; simulation settings, not Mini45 calibration |
| Observations | 34 values per frame, full six-component sensor wrench; history length is learner configuration |
| Outputs | Six network outputs; pitch disabled, five effective channels |
| Pitch option | Default off; when explicitly enabled, target angle ±5 degrees, rate ≤5 degrees/s |
| Final position | Robot EE XY distance from prescribed target <40 mm, plus original height/pitch/arrival conditions |
| Optional path constraint | task.tool_motion.enabled: false = final only (default); true = EE XY distance 25 mm free / 60 mm failure / weight 20 |
| Task version | pose_depth_v3 = pose_depth_v2 task + software board-stop consistency + analytic imagined survival (TASK_FREE_CUT); v2 archived for the 2026-11-16 runs |
| Reward | geometric maximum-depth progress 10, speed 20, load 1, jerk .1; success +100, failure −100 (unchanged from v2) |
| Completion | Pose arrival without higher-priority failures; material separation recorded independently |
| Episode / discount | 500 control steps / 25 s; gamma 1 (2026-11-19) |
| Target speed | Uniform 1–5 cm/s, resampled per world per episode; both reference controllers use it |
| Young's modulus / density | Per-episode log-uniform scales .5–2 / .8–1.25 |
| Cutting resistance | Per-episode log-uniform scale .5–2: DiSECt spring stiffness, BladeMaster cutting energy |
| Friction coefficients | Board .2, knife .2; contact laws still differ; no calibration claimed |
| Nominal cucumber | E=2.6 MPa, nu=.37, density=950 kg/m³ |
| Nominal half potato | E=2 MPa, nu=.45, density=1101 kg/m³ |
| Nominal prism | E=.04 MPa, nu=.45, density=830 kg/m³ |

Workspace and tool-position thresholds are **fixed physical distances**, not scaled
with object size. Rendered board edges are not a physical boundary. The current
configuration has no particle-displacement penalty: fragment opening is not charged.
Tool distance is measured from the prescribed goal in robot-base XY, not from a
moving object or from the rendered board edge. Both modes retain force/workspace/board limits.

## Shared physical knife and mounting

Merged from the former SHARED_KNIFE.md on 2026-11-18. Revision
`disect_knife_rigid_mount_v1`: the knife of this shared scene (introduced with
pose_depth_v2, unchanged in pose_depth_v3).
The reference is the **actual main DiSECt contact blade**, not the URDF appearance mesh.

### Common geometry and installation

The finite blade is 150.000006 mm long, 59.040001 mm high and 3.500000 mm thick at
the spine. Its cross-section has a thin 0.080 mm shoulder 0.040 mm above the edge
and a long taper to the spine. Both backends use the same ten physical vertices.

The BladeMaster blade frame is right-handed, z up and y along the edge.
Its conversion to the DiSECt blade frame is
`(x_D,y_D,z_D)=(x_B,z_B,-y_B)+(0,center_y,0)`.
A canonical source UR5e configuration determines a single EE-to-blade mounting,
then ordinary FK moves the blade. Tests compare the transformed vertices at
several joint configurations, not just one initial image. The along-edge axis
matches the source sawing direction in robot-base X.

- BladeMaster: a new optional two-slope signed-distance profile replaces the
  old vertical broad faces. Contact normals, contact points, broad-face transport
  constraints and board vertices all use this profile. Changing only the mesh
  or bounding-box dimensions would leave inconsistent physical contact.
- DiSECt: the new tool revision follows the rigid EE mount, interpolated over
  material substeps. It removes accumulated tool/robot pose drift from the old
  velocity-only, plane-projected coupling. The finite-tool swept joint-path
  board interlock remains active.
- Force and moment transformations continue to use the actual tool/material and
  sensor origins. There is no display-only translation masquerading as a fix.

The BladeMaster profile implementation is in the separately versioned
`BladeMaster-Warp-shared-knife-20261116` worktree. It is a change to our
reproduction, not an official upstream release.

### What remains intentionally different

FEM cutting springs versus TLMPM side labels/contact work, material
discretization, constitutive and fracture parameters, and material substep
integration remain backend-specific. DiSECt interpolates the tool over 2 ms;
the BladeMaster external-tool port holds its supplied tool pose over the
2 ms material exchange. Those coupling approximations have not been made
identical by this knife change.

The existing frozen policies were trained with different original knives,
task progress signals, conditions and interaction budgets. They are historical
results. This change neither retrains them nor converts their old results into
a controlled backend comparison. New training must use an explicit shared-tool
configuration and new run identity.

The renderer draws recorded physical blade vertices and the retained original
URDF handle. The previously invented cylindrical shank and collar have been
removed: neither had measured CAD provenance or user authorization. No
replacement connector or visual blade displacement is introduced. A measured
adapter mesh remains unavailable; rigid-transform checks do not validate real
hardware mounting. No second blue blade is displayed.

### Knife commands

Use the shared contract preparation and replay commands in
[Preparation and replay](#preparation-and-replay). They generate both backend
inputs from `configs/shared/scene_v1.json` and verify their manifest hashes.
The former `shared_knife_check.json` inputs now live in
`configs/archive/scene_checks_20261116/`; they describe historical probes, not
current launch instructions. Numeric probe results remain in RESULTS.

Material mechanisms and arrival-counter semantics: [CUTTING_MECHANISMS](reference/CUTTING_MECHANISMS.md).

### Historical scene alignment audit (superseded reset settings)

Condensed; the full text is in
[SHARED_KNIFE.md at 4dfbc0c](https://github.com/Ju6276/Cutting_DEV/blob/4dfbc0c/docs/SHARED_KNIFE.md).
Earlier 2026-11-16 captures, before the shared reset contract (nominal materials,
one world/object, requested 5 mm clearance, 30 mm/s feed, no policy, no control
step). They are evidence of the mismatch, not current settings. Actual lowest
blade height above the topmost material sample after settling:

| Object | DiSECt (mm) | BladeMaster (mm) |
|---|---:|---:|
| Cucumber | 5.52 | 17.45 |
| Half potato | 4.47 | 29.45 |
| Prism | 4.71 | 5.59 |

The historical BladeMaster implementation computed the initial height from the
maximum object height across its groups; DiSECt used each group's scene/start
lift. The shared reset contract above supersedes this. Outputs:
runs/shared_scene_views_20261116/{disect,blademaster}/ (close views, cucumber_wide.png,
recorded geometry, camera metadata, render_audit.json).

## Explicit differences and remaining work

- Native BladeMaster preset provenance: all three geometries start from
  `paper-banana`; geometry, E, Poisson ratio and density are overridden, but the
  nominal yield stress (30000 Pa) and cutting energy (700 J/m²) remain preset
  values. These are not calibrated fracture properties for these three objects.
  See [framework audit](archive/FRAMEWORK_AUDIT_20261116.md).
- Material physics: FEM / pre-inserted cutting springs versus MPM / native damage.
  Equal E, density or friction coefficients do not imply equal resulting forces.
- Geometry discretization: each backend samples the same intended object with its
  own particle/discretization scheme; settled heights/deformation can differ.
- Material timesteps differ. Robot/material exchange is 2 ms; DiSECt interpolates
  tool motion within that interval, BladeMaster currently holds the supplied pose.
- Material episodes use the same per-world deterministic sampler in both backends.
  Each reset draws E, density, a cutting-resistance multiplier and target speed.
  Cutting-spring stiffness and cutting energy retain their different meanings;
  matching their scale distributions does not equate fracture physics.
- BladeMaster now applies per-world GPU coefficient buffers, restores selected
  worlds to their raw state and settles them with the new material. It does not
  reuse a nominal-material settled snapshot. The group timestep is chosen for
  maximum E / minimum density so changes do not require new captured graphs.
- Checkpoints include sampler configuration, next-episode counters and values.
  Resume continues the streams and starts fresh physical episodes, following the
  existing trainer's reset-on-resume convention.
- A long run still needs resource/capacity and budget selection, plus online W&B.
- Enabled pitch has mapping/WM gradient tests, but active tilted blade/material/
  board-contact validation remains pending. Existing validation probes vertical reference
  motion with pitch off; the sawing action mapping retains regression coverage.

## Preparation and replay

Run in /dataHDD/dong/Projects/Cutting_DEV on `main` on any server (confirm
`git rev-parse HEAD`; other worktrees hold older code, see [DATA](DATA.md));
keep assets on servers.
The preparer preserves original scenes, exports CUDA-resident variants with shared
friction coefficients, records scene/geometry checksums, and writes the two backend
configs. Hash checks reject edits to prepared assets or a mismatched contract.

~~~bash
source scripts/env/cutting_env.sh
PY=python
$PY scripts/prepare_shared_scenes.py --contract configs/shared/scene_v1.json --output runs/NEW_SHARED_SCENE/assets
$PY scripts/check_pose_depth.py --backend disect --shared-contract configs/shared/scene_v1.json --prepared runs/NEW_SHARED_SCENE/assets/manifest.json --output runs/NEW_SHARED_SCENE/disect.json --seconds 900
$PY scripts/check_pose_depth.py --backend blademaster --shared-contract configs/shared/scene_v1.json --prepared runs/NEW_SHARED_SCENE/assets/manifest.json --output runs/NEW_SHARED_SCENE/blademaster.json --seconds 900
~~~

The contract fingerprint includes task.version: assets prepared for the archived
pose_depth_v2 contract are rejected for the current file; regenerate them (the
DiSECt scene bytes are unchanged by the version, so the per-object scene and
geometry hashes stay equal).

Reports contain requested and actual blade gaps, settled tops, EE/goal poses,
loaded friction, pitch settings, observation width, physical-versus-WM reward
agreement and exact preservation of non-reset worlds. Numeric results belong in RESULTS.


## Episode case replay

The shared episode_conditions block defines the ranges and seed. A draw is keyed
by object name, world index and that world's episode counter. Resetting world A
does not consume world B's random stream. Per-episode parameters, speed and counter
are recorded in episode logs; they are not policy or WM material inputs.
For the same case and episode number, both backends receive the same multipliers
and speed. Native nominal fracture parameters remain backend-specific.

## Two tool-position modes

Set common.task.tool_motion.enabled to false (final only) or true (path and final),
then regenerate both backend inputs. The prepared manifest checks the entire
contract, including this switch. Final XY radius remains 40 mm in both modes;
this extends the former 40 mm single-axis criterion to the horizontal plane.
Height tolerance 2 mm, pitch-error tolerance .04 (quaternion-vector component,
about 0.08 rad; not radians) and >10 qualifying arrival
samples are retained. Qualifying samples are cumulative, not necessarily consecutive.
When path control is off, passing 60 mm alone neither adds cost nor ends the task;
the existing robot workspace still applies. Failure to return to the final region
before the horizon gives timeout, not a successful cut.

With path control on, each 50 ms endpoint subtracts
20/500 * clamp((distance-.025)/(.060-.025),0,1)^2.
Above 60 mm the task ends with failure -100 once; exactly 60 mm remains inside.
These are configurable engineering settings. The 2026-11-16 shared long runs used final-only; their launch records do not establish
that optional path constraints improve learned behavior.

Reward-equation audit: append --tool-reward-audit to either check_pose_depth.py
command above. It evaluates both switches using measured transitions and
counterfactual goal offsets; it does not assert learned motion improvement.

Shared-tool studio and overview renderers now use the existing main-run
24 x 16 cm board exterior for both backends. Original recorded display corners
are preserved in studio exports. This is a visual footprint, not a finite
support boundary; physics remains an infinite support plane.

## Actual prepared DiSECt cut planes — audited 2026-11-17

These are the prepared assets loaded by the GPU7 shared long run, not merely
the source mesh defaults. Reconstructing virtual cut-edge points from
cut_edge_indices and cut_edge_coords gives the following initial interface.
Material axes: y is up, x is the cut normal (object length direction), z follows
the knife edge. Each plane is x=0 within numerical precision.

| Object | Material x bounds (mm) | Plane x (mm) | Offset from bounds midpoint (mm) | Fraction along x span |
|---|---|---|---|---|
| cucumber_003 | -85.9233, 86.3772 | 0 | -0.2270 | 49.8683% |
| potato_half | -43.9809, 44.4819 | 0 | -0.2505 | 49.7168% |
| prism | -50.0000, 50.0000 | 0 | 0 | 50.0000% |

Thus all are middle cross-cuts; only prism is exactly centered in the exported
bounding box. This does not imply equal material volume or mass on either side.
At the initial coupling transform all three planes are approximately base
Y=0.505 m, with normal +base Y. These are initial material interfaces, not
a claim that deforming or sliding material remains in a fixed world plane.

The exporter sweeps the reference knife edge to prepare cutting topology; it
does not autonomously choose a center. Object placement and knife trajectory
determine the plane before export. A visual goal cannot relocate that topology.

Evidence on GPU7:
/dataHDD/dong/Experiments/Cutting_DEV/runs/cut_plane_audit_20261117/current_planes.json.
Loaded files under
/dataHDD/dong/Experiments/Cutting_DEV/runs/long_readiness_20261116/prepared/:

| File | SHA256 |
|---|---|
| cucumber_003.pt | e3311f1a9b64d266ba597582fd096888cc665f2a4370693e1f4a89ab0b6fe0ad |
| potato_half.pt | 46e32d9535b1014e79bc5b76764b6626e1383d2d5007f556436ee230e3384a56 |
| prism.pt | 53b979a7efdaf9be4b41523806be8f2cab5dc3d9d3e04dad6eb116c12482b641 |
