# Visual goals — geometric and learned centers

This page owns the visual algorithm and inference/training interface.
Actual Nano device parameters belong in the
[hardware record](../hardware/MECH_EYE_NANO_CHECKLIST_ZH.md);
the provisional simulated sensor belongs in [MECH_EYE_NANO](MECH_EYE_NANO.md).
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
pinned at47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97, vendored in
cutting_world_model/vendor/dp3 with its license and SOURCE.json.
XYZ layers3–64–128–256, ReLU, LayerNorm and max pooling produce64 features;
our64–64–1 SiLU/tanh regression head is additional. No diffusion-policy network
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
