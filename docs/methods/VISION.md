# Visual goals — geometric and learned centers

This page owns the visual algorithm and inference/training interface.
Actual Nano device parameters belong in the
[hardware record](../HARDWARE.md#nano-capture);
the provisional simulated sensor is described below.
Historical scores and physical handoffs belong in [RESULTS](../RESULTS.md#vision-2026-11-17).

## Current implementation

Vision selects a goal before cutting. The tool is parked outside the food ROI
for acquisition, then moved to the cut start before motion/wrench feedback
drives execution. The existing simulation reset/IK handoff is not a real-robot
collision-checked approach planner. There is no continuous visual servoing or
image input to the low-level policy.

visible-midpoint uses metric optical-axis depth, camera/board calibration and
mounted blade geometry. It masks known robot pixels, segments the largest
foreground component and takes the 2–98% visible extent midpoint along its
horizontal PCA major axis. Axis ratio below2 rejects ambiguity; the prototype
heading restriction is15 degrees. This finds a visible midpoint, not equal
volume, material identity or an optimal cutting plane. It does not receive a
prescribed target XY. The prescribed mode remains a height/interface diagnostic.

fixed-midpoint and learned-midpoint use a fixed +baseY plane normal and a
horizontal calibrated board. Geometry converts the selected location into
start and final control-point poses through the mounted blade transform.
The low-level goal is [x,y,z,qx,qy,qz,qw,target_speed] in robot-base coordinates.
Quaternion sign must match the frozen policy convention, even though q and -q
represent the same physical rotation. Arbitrary new goals are not guaranteed
to lie in a policy's trained distribution.

## DP3 encoder

New visual training defaults to dp3_xyz_v1. The official PointNetEncoderXYZ is
vendored in
cutting_world_model/vendor/dp3 with its license and SOURCE.json.
XYZ layers 3–64–128–256, ReLU, LayerNorm and max pooling produce 64 features;
our 64–64–1 SiLU/tanh regression head is additional. No diffusion-policy network
or pretrained action weights are used.

cloud_input in visual_center_features.py samples512 visible XYZ points
randomly, with replacement only when fewer than512 exist; at least50 finite
points are required. It subtracts the observed cloud's meanX/Y and the observed
board heightZ, then divides by0.1 m. Axes remain robot-base axes, not a rotated
object frame. Output is one residual along baseY from the observed centroid,
bounded to ±50 mm after unit conversion. It does not output an8D robot goal;
the geometric adapter constructs that goal.

The encoder architecture can pool a different number of points, but current
preprocessing and checkpoint validation explicitly require512. Changing to1024
requires updating/verifying that contract; it is not an already-tested mode.
local_v1 remains only for old checkpoints and paired architecture comparisons.
Unknown architectures or mismatched point/scale metadata fail.

## Training data and scope

The existing pilot used settled cucumber003, half-potato and prism surfaces:
900 ideal-depth cases,300 fixed-heading translations per shape on a20×15 grid
covering ±15 mm. A shared placement-ID split gives600/150/150
train/validation/test cases. Full-surface bounds provide offline midpoint labels;
network inference sees only the visible cloud. No placement's repeated view
crosses splits, but test shapes are known and nearby.

The later DP3/local comparison reused this development test set. Historical
small errors are not unseen-shape or real-depth generalization evidence.
The five-food UR10e physical configuration does not automatically retrain vision;
new data must respect half-potato/peeled-banana preparation, measured visibility
and the actual tool orientation. Nominal desktop-camera captures are not a
real-cutting benchmark.

## Commands

Use explicit calibrated camera/visual config and source snapshots. Historical
front_camera_v1 is a synthetic pilot camera, not the real Nano.
These templates do not authorize another training run.

~~~bash
export CUTTING_ENV=render
source scripts/env/cutting_env.sh
python scripts/generate_visual_center_data.py --scenes SCENES --camera CAMERA.json --config VISUAL.json --output DATASET --wall-seconds 1800
~~~

~~~bash
export CUTTING_ENV=sac
source scripts/env/cutting_env.sh
python scripts/train_visual_center.py --data DATASET --output TRAIN --architecture dp3_xyz_v1 --epochs 120 --wall-seconds 900 --wandb-mode online
python scripts/plan_visual_cut.py --selection learned-midpoint --center-checkpoint TRAIN/center.pt --depth CAPTURE/depth.npz --camera CAMERA.json --config VISUAL.json --output PLAN.json
~~~

For a geometric selector choose visible-midpoint or fixed-midpoint and omit
--center-checkpoint. The optional --checkpoint checks frozen-policy interface
compatibility; plan_visual_cut does not execute the robot.

## Physical integration status

evaluate_visual_cut.py supports wm_policy, wm_mppi and analytic_mpc on one
native BladeMaster world, with matching --policy, --backend-checkpoint,
--render-python and optional --center-checkpoint/--model-checkpoint.
SAC visual execution is not implemented. DiSECt prepared cut topology is not
silently moved when a new visual plane is selected.

Recorded visual closed-loop examples are historical pose_depth_v2/180-step
runs with frozen cross-backend weights. They are not current UR10e five-food
validation or matched-controller comparisons; details and paths are in RESULTS.
Do not copy the old policy66 launch paths as a current deployment recipe.
New checks must match robot/tool/task/goal metadata and preserve material state
during pre-episode placement.

The canonical renderer uses the recorded robot/food/tool identity. Presentation
lines mark the chosen plane only, not verified material separation. Source
meshes, RGB textures and simulated contact parameters have separate provenance.

## Simulated camera

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
provenance.json records the selected housing solid and millimetre
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
