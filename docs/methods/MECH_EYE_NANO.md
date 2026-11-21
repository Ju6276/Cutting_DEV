# Mech-Eye NANO simulation adapter

Status (2026-11-21): lab hardware identified; simulation remains provisional,
not a calibrated vendor digital twin.

Field collection and editable handoff: [Linux camera checklist](../hardware/MECH_EYE_NANO_CHECKLIST_ZH.md).

## Confirmed lab camera

- Mech-Eye NANO V4, colour, nominal object distance 550 mm. The matching
  catalogue variant is NANO-V4D550C (not a directly captured device-model string).
- ju-ROG API capture on 2026-11-21 confirms hardware 4.0.0, firmware 2.3.4,
  serial TAC5522CA302E001 and live IP 192.168.122.101. API model text is
  `Mech-Eye NANO`; retain it separately from the catalogue variant.
- Camera PC: Ubuntu 22.04.5, MechEyeAPI 2.6.0, Python 3.10.12;
  enp108s0 at 192.168.122.100/24, 1000 Mbps. No robot commands were sent.
- Official recommended working distance for this variant: 450--600 mm.
  See [official V4 model table](https://docs.mech-mind.net/en/eye-3d-camera/2.3.2/hardware/camera-models.html).
- RGB and depth each return 1280 x 1024; API intrinsics separately read as
  fx=1728.8903839489603, fy=1728.8846096722514,
  cx=642.6250682011389, cy=512.2956888349067 px.
  API depth/XYZ unit is millimetres, invalid samples are NaN; convert once by 0.001
  for metre-based planning. SDK XYZ axes are right/down/forward in the optical frame.
- One desk connectivity capture is recorded (0.870 s, not measured frame rate).
  Empty-board/food samples, independent physical scale check, camera-to-base
  calibration and installed pose remain missing. API self-consistency is not
  physical calibration. Full raw values and evidence paths are in the field checklist.

The existing 350 mm simulation profile below does **not** represent this
confirmed 550 mm device. This documentation update does not change its optics,
sensor configuration, or training results.

## Available upstream systems

Mech-Mind's [picking simulation tool](https://docs.mech-mind.net/en/suite-software-manual/2.2.0/viz-tools/picking-simulation-tool-configuration.html)
offers camera deployment, FOV inspection, simulated point-cloud noise, acquisition
and image export in its own software. This is more than CAD, but no drop-in
Nano sensor plugin for this repository's MuJoCo/material backends was found.
The [official ROS interface](https://github.com/MechMindRobotics/mecheye_ros_interface)
is a real-camera driver, not a sensor simulator.

Official CAD: https://insights.mech-mind.de/en/cad_download_thank_you_en .
The NANO (V4) archive provides STEP, DXF/PDF dimensions and an example image.
Its CAD revision is unrelated to our task version. The external asset directory
is /dataHDD/dong/Assets/Cutting_DEV/cameras/mecheye_nano_v4 .
provenance.json records download SHA, the selected housing solid and millimetre
to metre conversion. FOV solids and the mounting bracket are excluded from the
housing mesh. The complete solid includes connectors: approximately
153.916 x 85.040 x 51.003 mm. The drawing specifies the main body as
145 x 85 x 51 mm with an approximately 9 mm side protrusion. Do not shrink the
connector-inclusive mesh to 145 mm. Housing uses a black display material based
on the official example image, not a measured texture or factory optical frame.

## Implemented approximation

configs/vision/mecheye_nano_provisional_v1.json selects ordinary NANO,
nominal 350 mm focus: 1280 x 1024, 0.3--0.45 m depth range, fixed eye-to-hand.
Its pinhole fx/fy are fitted to the published 220 x 150 mm FOV at 0.3 m.
This approximation is NOT factory calibration; the published FOV alone cannot
determine distortion or the exact effective depth ROI.
The optical axis points to the recorded board origin from 0.43 m away,
30 degrees off vertical, in front of the robot along +base X.
CAD-to-optical registration is an approximate assembly fit, not factory extrinsics.
The support structure is not modelled. No fictitious bracket is attached to the arm.

The capture writes optical-axis depth in metres (invalid = 0), the known-robot
mask, an XYZ cloud in robot-base coordinates, intrinsics/extrinsics and provenance.
Depth uses geometric rasterization, real occlusion and the declared range.
RGB uses scripts/render_episode_still.py, the SAME canonical Blender materials,
board and lighting as presentation output; only the perspective sensor camera
changes. The old VTK colour render is no longer exported by capture_visual_studio.
The unreferenced capture_visual_cut_camera.py entry has been retired; there is
only one current capture entry and one canonical Blender appearance path.
The first provisional capture folders without _v2 used the obsolete VTK RGB;
their depth checks remain measurements, but those RGB previews are superseded.

Not simulated: fringe projection/reconstruction, wet/specular material-dependent
dropouts, lens distortion, measured noise and real acquisition timing.
The manufacturer's 0.1 mm repeatability is not used as Gaussian noise: it was
measured on a particular ceramic target. Real captures are needed to fit these.
The advertised 0.6--1.1 s capture interval is metadata, not a 20 Hz sensor claim.
This camera is used before cutting; it does not add an image input to the policy.

## Reproduce

Use the render environment and a canonical studio scene exported with
scripts/export_evaluation_scene.py. Prepare a rig from its source recording:

    CUTTING_ENV=render source scripts/env/cutting_env.sh
    python scripts/prepare_camera_rig.py --recording RECORDING --output RIG
    python scripts/capture_visual_studio.py --capture-only --scene SCENE.npz --camera RIG/camera.json --config RIG/board.json --sensor-profile configs/vision/mecheye_nano_provisional_v1.json --output CAPTURE

The named-profile snapshot adapter requires explicit world_from_base calibration
and verifies world == robot base, board origin, and profile/camera intrinsics.
It refuses missing or inconsistent calibration instead of silently relabelling
world coordinates. prepare_camera_rig.py writes this calibration.
For a wide presentation use render_scene.py with --sensor-rig RIG/rig.json.
Outputs use the same canonical rendering entry; no independent food palette.

## Vision network and the five foods

Current learned midpoint pilot: DP3 official XYZ encoder, 64 feature channels,
plus our bounded one-scalar regression head. It is not the complete DP3 diffusion
policy. Visible object points are randomly sampled to 512 XYZ rows, with
replacement only if fewer than 512 are visible (minimum 50 required).
Input origin: visible cloud mean X/Y and measured board height Z, divided by
0.1 m. Output: a residual along fixed base Y from the observed centroid,
scaled back to metres (bounded to +/-50 mm). The geometry adapter constructs
the cutting plane and end-effector goal. It does not estimate arbitrary
orientation or material parameters.

New five-food geometry does not automatically train this network. A five-food
visual benchmark needs train/held-out captures for all five, preparation-consistent
geometry (half potato, peeled banana), object translations and calibrated
visibility; fixed-orientation pilot scope remains explicit. Actual UR10e target
orientation/frame compatibility must be tested before applying an old frozen
vision checkpoint. The policy interface remains 34 x 10 + 8 = 348 inputs and
6 mapped control outputs; pitch remains disabled. Neither the vision network
nor its material labels enter the policy as hidden inputs.

## Capture validation and remaining visual execution scope

The five existing cutting snapshots include the robot and knife at the cutting
start pose. These are RGB-D **acquisition** checks, not successful visual
target-selection or closed-loop cutting results. In the cucumber view, the blade
occludes the food and splits its foreground into disconnected components; the
existing selector correctly refuses an ambiguous target. We neither hide the
robot nor relax that rejection rule. An actual pre-cut observation pose must put
the tool outside the food ROI before selecting a target and approaching it.
That is a visual execution step, not an additional requirement for training
the existing nonvisual cutting policy.

Capture preserves each renderer subprocess log and requires exit code zero,
the image, metadata and projection self-check. Blender's perspective projection
is checked against the depth camera intrinsics (three noncollinear optical
points; tolerance 0.002 pixel). RGB and depth use identical physical meshes and
board dimensions; differing rasterizers still have distinct antialiasing and
edge coverage. This is geometric alignment, not a calibrated photometric model.

On GPU7 Blender 2.82, two small NumPy matrix products in the new sensor-camera
path triggered a system libblas dgemm_oncopy_BARCELONA segmentation fault before
rendering. The same rigid transforms are now computed with elementwise axis
sign changes and einsum, without BLAS. Failed run logs remain in the run folder;
successful renders must be reported from the new validated_* folders.

2026-11-21 evidence: all five captures in
/dataHDD/dong/Experiments/Cutting_DEV/runs/mecheye_nano_20261121/validated_{cucumber,potato_half,banana,tofu,luncheon}
completed with 1280 x 1024 RGB, metric depth, robot mask, cloud and metadata.
All five Blender projection checks measured 0.000255 pixel maximum error.
The 868,752--928,466 valid depth pixels include the board and robot, not just
food. Related camera/visual geometry tests: 32 passed.

### Housing appearance audit

Reinspection of the official STEP found five solids: the complete camera housing
(3761 faces, 610105 mm3), one mounting bracket, and three FOV solids. There is no
separate omitted lens or panel solid. The housing includes the two front window
frames, cooling fins and side connectors. Its STEP colour declarations are white,
not the finishes in the vendor illustration.

The canonical renderer now assigns silver to the actual protruding connector
geometry (source CAD x < -43.495078 mm). The official 35 x 20 mm warning-label
plane (STEP housing face 984) receives the corresponding rectangle of NANO.png,
with a 0.02 mm display-only offset to prevent z fighting. The profile records
the image rectangle and CAD plane; this is a visual label registration, not
optical calibration. No logo decal, optical lens interior or support stand was
invented. The supplied photograph contains a logo, but there is no isolated
texture or verified mapping for it, so the rendered housing still omits that logo.

The fixed wide view sees mainly the back/side; it cannot be compared directly
with the vendor's front-side product illustration. nano_detail.png is a close
inspection of the same scene and asset with only the observer camera changed.
The physical sensor pose, robot, food, lighting and geometry stay unchanged.

CAD orientation check: the two windows lie near CAD y=-48.44 mm and the supplied
FOV solids extend along CAD -Y. The current rotation maps CAD -Y to optical +Z,
so the viewing direction faces out of the windows. The optical origin remains
an approximate registration, not a measured camera centre or hand-eye transform.
Appearance corrections do not calibrate this unknown.
