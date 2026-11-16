# Shared physical knife and mounting

Revision: `disect_knife_rigid_mount_v1`, opt-in with `pose_depth_v2`.
The reference is the **actual main DiSECt contact blade**, not the URDF appearance mesh.

## Common geometry and installation

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

## What remains intentionally different

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

## Current commands

Use the shared contract preparation and replay commands in
[SHARED_SCENE](SHARED_SCENE.md#preparation-and-replay). They generate both backend
inputs from `configs/shared/scene_v1.json` and verify their manifest hashes.
The former `shared_knife_check.json` inputs now live in
`configs/archive/scene_checks_20261116/`; they describe historical probes, not
current launch instructions. Numeric probe results remain in RESULTS.

Material mechanisms and arrival-counter semantics: [CUTTING_MECHANISMS](CUTTING_MECHANISMS.md).


## Historical scene alignment audit (superseded reset settings)

These are the earlier 2026-11-16 captures, before the shared reset contract.
They are retained as evidence of the mismatch, not current settings. See
SHARED_SCENE for per-object settled height + sampled clearance in both backends.

The historical captures use nominal materials, one world/object and the same
requested 5 mm clearance/lift setting and 30 mm/s feed command. No policy is
loaded and no control step is taken. Source configurations remain unchanged.

Actual lowest blade height above the topmost material sample after settling:

| Object | DiSECt (mm) | BladeMaster (mm) |
|---|---:|---:|
| Cucumber | 5.52 | 17.45 |
| Half potato | 4.47 | 29.45 |
| Prism | 4.71 | 5.59 |

The historical BladeMaster implementation computed the initial height from the maximum object
height across its groups; DiSECt used each group's scene/start lift. This is
the physical reset-condition difference recorded in these snapshots; the shared
reset contract supersedes it.
Material particle centres and FEM nodes also represent surfaces differently.

The studio board rectangles are 160 x 240 mm and 300 x 300 mm respectively.
These are display footprints; the material support is a plane, and the
robot-base EE workspace is a separate explicit box. Changing the image
footprint would not create physical board-edge collision.

Outputs: runs/shared_scene_views_20261116/{disect,blademaster}/
contains cucumber/potato/prism close views and cucumber_wide.png, along with
recorded geometry, nominal config and camera metadata. Each corresponding
backend pair uses exactly the same camera; render_audit.json checks all four.
The two backends retain different material laws, integration and sampling
distributions. A matched comparison still needs matched actual reset states,
command/material case definitions and budgets in addition to the shared knife.
