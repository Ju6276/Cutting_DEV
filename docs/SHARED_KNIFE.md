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

The retained render adapter draws recorded physical blade vertices; its shank
is display geometry. It does not implement contact or alter the mounting.
No second blue blade is needed for normal presentation.

## Commands

From GPU7 `/dataHDD/dong/Projects/Cutting_DEV_board`, after
`source runs/sac_env.sh`, use the MPPI Python environment (SciPy is required
by the BladeMaster mesh pipeline):

```bash
PYTHONPATH=.:backends/mujoco /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python \
  scripts/check_shared_knife.py --backend disect \
  --config configs/free_cut/shared_knife_check.json \
  --output runs/NEW_DIRECTORY/disect.json
PYTHONPATH=.:backends/mujoco /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python \
  scripts/check_shared_knife.py --backend blademaster \
  --config configs/blademaster/shared_knife_check.json \
  --output runs/NEW_DIRECTORY/blademaster.json
```

The probe measures physical tool vertices relative to robot FK, finite
clearance above the board, finite wrench values and reference task termination.
It has an explicit wall budget, closes environments on failure, and does not
replace successful material separation with pose completion.
Numerical results are recorded in RESULTS (2026-11-16). Both three-object
reference probes completed; material separation is recorded independently.

Material mechanisms and arrival-counter semantics: [CUTTING_MECHANISMS](CUTTING_MECHANISMS.md).
