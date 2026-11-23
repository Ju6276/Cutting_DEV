# 视觉模块：几何选面与 DP3 定位

原 VISION_ROUTE1.md 的算法说明已合并到本页，并未删除 DP3 实现或权重。
本页只维护视觉；主方法与任务在[方法](../METHOD.md)，
真实 Nano 参数和标定在[硬件](../HARDWARE.md#nano-capture)，
历史实测分数在[结果](../RESULTS.md#vision-2026-11-17)。

## 在当前方法中的位置

- **主方法已定：**WM+BPTT 学习低层切割策略，SAC 和冻结 WM+MPPI 为主要对照。
- **视觉方案一：**点云几何选切面，不训练定位网络。
- **视觉方案二：**DP3 官方点云编码器加本项目定位回归头；不是完整的 DP3 扩散策略。
- 两方案都先生成目标，再交给低层控制。切割过程中策略使用运动/力历史，不持续输入图像。
- 现有 DP3 权重来自旧三形状定位实验；当前五食物 UR10e 控制策略训练完成，
  不代表视觉也已在这五种食物上重训或通过闭环验证。

统一流程：刀移出食材视野 → 采集、分割点云 → 几何/DP3 选位置 →
安装变换生成目标 → 移到起刀位置 → 运动与力反馈切割。
仿真的 reset/IK 衔接不能当作真机避碰接近规划。

## Current implementation

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
The historical checkpoint is not trained on the current five-food UR10e configuration;
new data must respect half-potato/peeled-banana preparation, measured visibility
and the actual tool orientation. Nominal desktop-camera captures are not a
real-cutting benchmark.

## Current five-food dataset contract

The current generator takes an explicit scene manifest: a JSON list of
objects with `name` and `scene` (canonical pre-cut scene.npz) fields.
It uses the settled cucumber, half-potato, peeled banana, tofu and luncheon surfaces.
The approved next run fixes the current Nano camera and food heading, with 300
translations per food on the same +/-15 mm grid: 1,500 samples, split into
1,000 training / 250 validation / 250 test cases. Every food/view sharing a
placement ID belongs to one partition. This tests known shapes, not unseen food.

New checkpoints declare `table_aligned_base_v1`: the calibrated board origin
is subtracted before visible-cloud centering; table axes currently stay parallel
to the robot base. The returned Y residual is transformed back to the base frame
before constructing the tool goal. Rotated table axes are rejected. This explicit
frame contract does not create viewpoint invariance: with the current axes it is
numerically equivalent to the old centered XYZ normalization. Old checkpoints
without this field retain their base-frame convention.

Only depth is rendered for dataset generation, with measured Nano intrinsics.
No alternate RGB food textures are introduced. Ideal depth, no robot occlusion,
fixed camera and one settled shape per food do not establish real-camera transfer.
The optional camera-azimuth flag is available but disabled in this run.
Training uses CUDA and online W&B, validation selects the checkpoint, and the test
partition is evaluated only after selection. The frozen low-level RL actor is
not retrained by this visual job.

## Commands

Use explicit calibrated camera/visual config and source snapshots. Historical
front_camera_v1 is a synthetic pilot camera, not the real Nano.
These templates do not authorize another training run.

~~~bash
export CUTTING_ENV=render
source scripts/env/cutting_env.sh
python scripts/generate_visual_center_data.py --scene-manifest SCENES.json --camera CAMERA.json --config VISUAL.json --sensor-profile configs/vision/mecheye_nano_v4_550_lab_v1.json --camera-azimuth-deg 0 --output DATASET --wall-seconds 3600
~~~

~~~bash
export CUTTING_ENV=sac
source scripts/env/cutting_env.sh
python scripts/train_visual_center.py --data DATASET --output TRAIN --architecture dp3_xyz_v1 --epochs 120 --wall-seconds 900 --wandb-mode online
python scripts/plan_visual_cut.py --selection learned-midpoint --center-checkpoint TRAIN/center.pt --depth CAPTURE/depth.npz --camera CAMERA.json --config VISUAL.json --output PLAN.json
~~~

For a geometric selector choose visible-midpoint or fixed-midpoint and omit
--center-checkpoint. Always specify --selection: the CLI default prescribed
is the old target/height diagnostic, not visual center selection. The optional --checkpoint checks frozen-policy interface
compatibility; plan_visual_cut does not execute the robot.

## Physical integration status

evaluate_visual_cut.py supports wm_policy, wm_mppi and analytic_mpc on one
native BladeMaster world, with matching --policy, explicit --group, --blademaster-config (or --backend-checkpoint),
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

The current visual suite uses configs/vision/mecheye_nano_v4_550_lab_v1.json:
confirmed NANO V4 D550C, 1280 x 1024, measured ROG API intrinsics, proposed
0.55 m board distance, 30 degrees off vertical in front of the robot along
+base X. Extrinsics are a simulation placement, not an eye-to-hand calibration.
The 0.45--0.60 m depth clipping window is a nominal operating envelope, not
the camera's configured hardware DepthRange. Real-camera details stay in HARDWARE.

The older mecheye_nano_provisional_v1 profile is retained for replaying historical
350 mm approximation captures; it must not be described as the actual lab camera.
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
The provisional capture-time field is descriptive metadata, not a measured 20 Hz sensor claim.
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

2026-11-21 evidence: all five captures in
/dataHDD/dong/Experiments/Cutting_DEV/runs/mecheye_nano_20261121/validated_{cucumber,potato_half,banana,tofu,luncheon}
completed with 1280 x 1024 RGB, metric depth, robot mask, cloud and metadata.
All five Blender projection checks measured 0.000255 pixel maximum error.
The 868,752--928,466 valid depth pixels include the board and robot, not just
food. Related camera/visual geometry tests: 32 passed.

### 外壳与坐标限制

统一渲染器使用官方 STEP 中的完整外壳，排除支架和视场实体；
实际接头显示为银色，警告标签来自官方图片，未虚构镜头内构或支架。
外壳主色为黑色，未标定外观不等于真实成像模型。
CAD 的 -Y 朝向窗外，当前装配将其映射到光学 +Z；
光学原点与外壳的配准仍是近似值。换观察角度不改变机器人、食品或传感器位姿。

更早的外壳审计与渲染调试过程保留在 Git 历史，日常使用不需阅读。


## 当前 UR10e 五食材视觉接入（2026-11-23，验证中）

运行顺序保持：将实际机器人模型放到当前工作空间的观察位 → 同一场景渲染 RGB-D → 512点归一化输入冻结DP3 → 生成目标 → IK放置到起刀位 → 冻结策略执行。
观察位不再写死旧UR5e坐标；移刀后传感器滤波零点与FDCC重力基准一起重置。材料粒子位置保持不变；这里的IK放置仍不是连续避碰接近轨迹。

`evaluate_visual_suite.py` 对配置中的每种食材各运行一次，单独保存感知、目标、控制轨迹、回合结果和错误，并设置单物体与总时长上限。旧DP3权重只训练过三形状；香蕉、豆腐、午餐肉此次属于冻结视觉模型的新形状流程测试。之后再生成五食材数据训练视觉网络，不能把入口接通写成视觉训练已完成。

相机配置 `configs/vision/mecheye_nano_v4_550_lab_v1.json` 使用ROG记录的 Nano V4内参，0.55 m拟定物距、机器人前方30度俯视。外参仍是仿真设计，不是实测眼在手外标定。理想深度不模拟结构光重建或湿表面缺测。外观继续使用统一Blender场景。

~~~bash
export CUTTING_ENV=planner
source scripts/env/cutting_env.sh
python scripts/evaluate_visual_suite.py --policy FROZEN_ACTOR.pt --blademaster-config FIVE_FOOD_CONFIG.json --center-checkpoint DP3_CENTER.pt --render-python .venv-render/bin/python --output NEW_OUTPUT_DIR --wall-seconds 7200 --per-food-seconds 1200
~~~

当前执行BladeMaster。DiSECt视觉选出的新切面需要重新生成匹配的预切拓扑；不能直接移动终点或用预设切面替换视觉输出来假装完成闭环。
