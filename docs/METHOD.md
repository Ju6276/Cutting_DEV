# 方法、任务与仿真配置

当前算法路线已确定：WM+BPTT 为主方法，SAC 与冻结 WM+MPPI 为主要对照；
解析 MPC 用于控制链工程测试。DP3 视觉只负责切前目标定位，
独立接口与适用范围统一见[视觉模块](methods/VISION.md#在当前方法中的位置)，
不在本文重复相机参数或视觉实验。

当前主线：UR10e + ATI Mini45 + IKEA CAD刀，五食材，
pose_depth_v3 / project_continue_v2。本页统一说明算法、任务和场景；
命令见[运行](OPERATIONS.md)，实机见[硬件](HARDWARE.md)，研究取舍见[研究](RESEARCH.md)。
旧配置只用于读取旧结果，不是另一套当前入口。上游来源见[来源](reference/SOURCES.md)。

## Scene

唯一当前训练分布：[ur10e_five_food_v1.json](../configs/shared/ur10e_five_food_v1.json)。
黄瓜003、半土豆、去皮香蕉、豆腐、午餐肉，每环境固定所属食物组。
一份策略覆盖该后端五种食物，两材料后端分别训练。小配置3×5=15环境，
长训按批准规模重新生成，不因环境数不同改变任务规则。

香蕉是均匀去皮果肉；豆腐/午餐肉来自本地复现块体预设，不是原论文实测标定。
名义E/ν/ρ：黄瓜2.6 MPa/.37/950 kg/m³，土豆2 MPa/.45/1101，
香蕉.4 MPa/.45/1100，豆腐6 kPa/.35/1100，午餐肉20 kPa/.4/1200。
板/刀面摩擦均.2，尚需真机标定。

每回合重新采样E×[.5,2]、ρ×[.8,1.25]、切削阻力尺度×[.5,2]、
目标速度[.01,.05] m/s，再静置1 s仿真时间。
起刀间隙5–30 mm按case/world抽一次，回合间保持；
每次按该物体静置后最高粒子重新放刀，不取本批最高物体。
这是保守高度代理，不是视觉估计误差模型。环境不会reset时更换食物类型。

重置：选中食材恢复/采样材料 → 使用不接触恢复形状的静置刀位 →
静置 → 按新顶部重放刀 → 重建FDCC/滤波/去皮/历史。未选中环境不变。
旧“按上一轮较低顶部放刀，再恢复较高物体”顺序已修。
board_contact从最终FK计算，通用restore_frame不能代替该协议。

UR10e工作空间约x=[.528213,.728213]、y=[-.100017,.099983]、z=[.370,.543] m，
约束基座系控制点，精确值以JSON为准；不是旧UR5e的x=[.038,.238]等范围。
物理支撑面无限延伸，渲染板边缘不是掉落判据。
UR5e [scene_v1](../configs/shared/scene_v1.json)与旧UR10e两食材保留配置
不可和当前五食材checkpoint混用。

## Controllers

- **WM+BPTT**：学习WM、critic和策略；执行为策略→mapper→FDCC。
- **SB3 SAC**：学习策略和Q，无学习动力学；执行同一mapper/FDCC。
  原SliceIt TensorFlow SAC仅为历史复现。
- **冻结WM+MPPI**：已训练WM预测候选动作，在线MPPI优化，评估不改权重；
  可选同一checkpoint的critic终值。
- **解析MPC / analytic_mpc**：FDCC/运动学近似模型加MPPI，无网络训练。
  保留为工程联调/控制链测试工具，不是论文必需基线，更不是FDCC本身。

MPC是滚动规划框架，MPPI是优化器；两个规划器都用它。
四方法同在main，共享物理执行控制器；物理后端和算法选择独立。
“无训练”不代表无模型、无标定或预测准确。

策略/SAC输入10×34观测+8维目标[x,y,z,qx,qy,qz,qw,target_speed]，
位姿在机器人基座系、表示控制点。规划器还用关节及控制/任务历史，
信息访问比策略多，比较时应披露。六动作对应两刚度、两增益、沿刀偏移、可选倾角；
当前最后通道禁用，五个有效控制量，不是六维位姿或关节力矩。
沿刀偏移每50 ms最多累计2 mm，没有独立累计限幅。

解析模型在候选预测内保持已有历史wrench样本，真实执行一步后读新反馈再规划；
不预测下压/拉锯/切穿后的受力变化。重构FDCC递推是近似，
但必须保留物理控制器的本回合去皮基准，不能每次预测重新去皮。
此bug已修，短物理验证见[结果](RESULTS.md#ur10e-baseline-verification)。
它不是任何已发表切割MPC的完整复现。

WM+MPPI优化负的存活加权任务回报，加上游MPPI采样修正；
可选终值gamma^H*survival_H*V(s_H,g)，终止后为0，无actor BC项。
候选动作归一化[-1,1]再映射任务范围。显式记录H、采样数、温度、噪声、
refinements与规划耗时；物理仿真耗时不等于规划耗时。
模型/critic配对、跨后端调用及任务/工具元数据均需核对。

## Completion, failure and timeout

- Goal: the captured robot-base end-effector pose translated along the board
  normal until the lowest point of the actual finite knife is at the board stop
  height (blade-to-tool offset and orientation included). The policy does not
  choose it.
- Arrival sample: final EE distance in robot-base XY strictly below 40 mm
  (`completion_position.radius_m = 0.04`), z error < 2 mm, pitch-error
  component < 0.04 (quaternion-vector component, about 0.08 rad; not radians).
  The counter increases only when all hold; success needs the current sample
  inside and counter > 10 (cumulative, not consecutive).
- Success earns +100 once, only without force or workspace failure. Failure
  takes precedence and costs -100 once, including on the same step as arrival.
  The environment requires reset after an ending.
- Timeout at 500 control steps (25 s at 0.05 s) is an unsuccessful terminal
  without completion bonus or value bootstrap. Gamma is 1. Initialization
  settling is outside this clock.
- Board contact is neither success nor failure. Penetration below
  -2 micrometres aborts the simulator transition (an implementation error, not an
  RL failure).
- Material separation, residual springs, coverage and sliding are reported
  separately. Pose-success rate must not be described as a separation rate.

## Rule provenance

上游规则来自下方固定版本的SliceIt源码链接；本地复现不等于未修改的官方实现。
原始核对在GPU7 runs/sliceit_task_rule_audit_20261119/evidence.json。

| Rule | Official SliceIt source | Implemented pose_depth_v3 | Provenance / reason for difference |
|---|---|---|---|
| Arrival and +100 | Current pose error satisfies selected-axis thresholds and cumulative arrival counter exceeds threshold; no force/torque collision. No material-separation test. | Pose arrival plus cumulative counter >10; no higher-priority failure; +100 once. Also requires the configured final XY region. | Pose-based completion is inherited. Final XY region and explicit failure precedence are our task choices; completion does not certify cut-through. |
| Height tolerance and counter | Ordinary simulation: 0.5 mm, >10. **DiSECt simulation: 2 mm, >10.** Real: 2 mm, >1. | 2 mm, >10. Counter increments on qualifying samples; it is not a consecutive-hold requirement. | The current pair already exists in official DiSECt simulation. Do not remove the counter merely to match upstream. |
| Force/torque limits | Component limits; collision ends the episode and contributes -100. DiSECt simulation uses 50 N / 10 N m; real config uses 50 N / 5 N m. | Six-component limits 50 N / 5 N m, checked on fresh 2 ms filtered samples; -100 once. | Our shared sensor/limit contract uses the latter scale. Sample timing and peak bookkeeping are adapter-specific. |
| Workspace | Out-of-workspace ends the episode; no separate fixed -100. | Explicit robot-base workspace; exit ends the episode with -100. | Our failure settlement; a deviation, not an upstream rule. |
| Time and step cost | 500 steps. Step cost -1 is part of the normalized continuous reward (contribution -0.25 with equal weights). | 500 steps / 25 s; timeout is terminal with no bootstrap. No separate per-step -1. | Horizon aligned by user request (2026-11-19). Continuous rewards and timeout semantics are our design. |
| Continuous rewards | Normalized distance, force and jerk terms, plus normalized step cost. | Maximum geometric-depth increment, target-speed deviation, normalized six-axis load and jerk costs (below). | Our motion and speed objectives, not the full SliceIt reward. |
| Board geometry and reaction | Official scene contains a board collision mesh. No board latch or board-hit failure in task code. | Finite-blade geometric boundary; no calibrated blade-board reaction model. | Added here to prevent penetration; missing board physics is not an upstream requirement. |
| At the board | No permanent freeze in SliceIt task code. | Per-substep nonpenetration projection; material, controller and sensors continue; lateral/lift proposals execute; arrival counter accumulates. | Our geometric boundary, not a physical board reaction. |
| Material separation | Not checked by the pose-success code. | Separate diagnostic only; never a success gate or reward. | Do not relabel pose success as physical separation. |

Official sources:
[DiSECt simulation config](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_rl/config/simulation/slicing_3d_disect.yaml),
[ordinary simulation config](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_rl/config/simulation/slicing_3d.yaml),
[real config](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_rl/config/real/slicing_3d.yaml),
[termination](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_openai/src/ur3e_openai/task_envs/force_control/disect_slicing.py),
[reward](https://github.com/omron-sinicx/ur_python_utilities/blob/7cd7cb8ce491b7953fce61ca891382e893e2ec43/ur3e_openai/src/ur3e_openai/cost_utils.py),
[board collision geometry](https://github.com/omron-sinicx/sliceit/blob/3a6964862f1f135af9d98dd400efeb44ba394c15/catkin_ws/src/o2ac_scene_description/urdf/cooking_environment.xacro).
[SliceIt III-C.4](https://arxiv.org/html/2404.02569v1#S3.SS3.SSS4) uses target-pose completion.

## Rewards

Let g_t be the lowest point of the recorded finite-blade rigid proxy above the
board, from robot joint state and blade mounting; s the positive board stopping
clearance; D = g_reset - s > 0 the per-world travel recorded at reset:

    rho_t = max(rho_(t-1), clamp(1 - (g_t - s)/D, 0, 1)),   rho_reset = 0

- Progress reward: 10 times the increase in rho. Retraction/re-entry cannot earn
  progress again. Descending through free space does earn progress: this is
  motion shaping, not a cutting or contact detector.
- Speed penalty, with positive commanded feed v* and m = ((v-v*)/(|v|+v*))^2:
  20*m*(positive delta-rho + 1/horizon).
- Load/jerk cost at the 50 ms control period: 0.05*(min(u,1)+0.1*min(j,1)), with
  u the maximum absolute normalized six-axis sensor wrench and j the normalized
  jerk measure.
- Success +100, failure -100 (above). Success does not require rho = 1.
- With final-only defaults the running-penalty bound is
  2*20 + 500*0.05*(1+0.1) = 67.5; adding maximum progress 10 gives 77.5 < 100,
  so terminal settlement dominates. With the optional tool-motion cost the bound
  plus progress is 97.5 < 100.

The backends and the WM call the same depth helper at control-step endpoints.
Depth span and accumulated progress are known history fields; the WM derives
progress analytically from predicted joints. The per-episode span supports
different objects and start heights without per-object weights. Physical
separation is independent: DiSECt reports residual springs, BladeMaster its
native topology diagnostics. Do not report rho = 1 as material separation.

## Board and wrench

project_continue_v2将向板内的端点作纯法向末端平移投影，不附加切向/转动分量；
速度只去掉向内法向分量。止挡间隙1 μm，有浮点裕量和扫掠子步检查；
关节限位/奇异性无法满足边界则报错，不改任务规则掩盖。
这是几何防穿透，没有刀—板反力。

接触后材料、控制器和传感器继续推进，允许横移、抬刀，无永久冻结。
board_contact表示当前有限刃FK间隙≤10 μm，抬刀可回0，reset也按最终FK计算。
物理与WM共用边界定义；WM离散步不复现全部物理子步扫掠，是模型近似。
每个2 ms新滤波读数只进入区间峰值/力限一次，不重算上一控制步末读数。
超限停止该世界，其他世界继续。last_wrench是步末六轴读数，不是峰值或均值。
虚拟ATI读数由机器人内部作用和材料反力换算/滤波/去皮，没有真实ATI噪声。

默认只限制最终位置，无粒子滑移罚。
可选过程约束按基座XY控制点距目标：25 mm内自由，25–60 mm每步扣
20/500*clamp((d-.025)/(.060-.025),0,1)^2，>60 mm失败-100。
不是从物体边缘起算的正方形。当前该约束和bounded_pitch_v1均关闭。

## World model
One frame has 107 coordinates: 42 predicted physical ones (joint position/velocity
12, three-sample wrench history 18, interval peak wrench 6, `last_wrench` 6) and
65 derived ones (motion command, previous pose/velocity/acceleration, the
34-wide observation, success counter, step count, depth progress, depth span,
completion XY error and `board_contact`). Ten frames give 1070 state coordinates
(`prediction_layout=control_history_v2`). The dynamics input takes 54
coordinates per frame, 540 in total: joints 12, wrench history/peak/last 30,
motion command 6, previous action 6. The model predicts a 42-wide delta. The
policy sees 34 observations x 10 frames plus an 8D goal. The WM fits 8-step
windows, the actor imagines 8 steps (`--actor-horizon 8`) and the real-data
critic uses 32-step recorded windows.

FK, observation, motion-command, derivative-history, counter, reward and
terminal formulas advance the derived coordinates analytically; derived pose,
velocity, acceleration, observation copies and counter feed the analytic heads
and critic, not the dynamics MLP. This follows the pinned source's separation
of physical dynamics state from analytic observation/reward features. History
length 10 matches the policy history; reset repeats the actual initial frame,
and each transition shifts only past frames. Particles, spring damage and native
controller/filter packets remain in physical recordings but are absent from
model inputs and targets. The SuperMimic residual MLP widths, normalized
squared-error form, terminal BCE, optimizer and effective batches are kept.
No learned latent state or recurrent network is added.

当前无锁存：有效真实转移正常参与动力学拟合；旧锁存mask/独立分母仅供旧元数据兼容。
学习终止头继续BCE训练，其梯度仍可回传动力学；当前策略/MPPI的存活权重
只用解析成功/失败/超时，不用头概率。终止当步奖励保留一次，之后吸收无bootstrap。

## Learning

选定real-data critic H32、actor H8、gamma=1、GAE lambda=.95、BC alpha=.25。
source默认仍为imagined critic/H32/无BC，当前方案必须显式传参。

- WM：8步自回归归一化平方误差+终止BCE，按有效真实转移平均；
  completed短窗口以终状态填充、mask为假，未完短尾不采样，不跨reset/覆盖边界。
- Critic：真实奖励/终止构造H32 GAE，非终止窗口尾用V、真正终止取消bootstrap。
  收集至131072个有效三元组后拟合，保留源网络、Adam和5轮×4 minibatches。
- Actor：通过WM反传J，并用critic尾值；附回放动作BC：
  -J/h + mean(|J|)/(alpha*h)*MSE(pi(o),a)，alpha=.25不是固定MSE系数。
- 微批按有效样本权重累积，最后optimizer step/clip；
  effective WM/actor batch2048/4096，微批512、critic微批8192。
- 热身30720真实交互，每640交互积累一轮学习额度；
  前500轮仅WM，第501轮联合训练。初期冻结策略仍带既有探索噪声采集。
  学习轮不等于Adam步或vector step。
- schedule=physical_transitions_v2，LR时钟按交互/256。
  当前cosine周期在短预算内变化小，不能称完成完整衰减。
- 续训恢复模型/optimizer、配对replay、学习额度、未开始轮与热身起点；
  物理重新初始化、旧未完回合关闭，不是恢复精确粒子瞬态。
  不带replay则重新填充热身，不算原样续训。

real_wm/risk等旧选项不是当前方法。研究选择见[研究](RESEARCH.md)。

## Material backends

**DiSECt**：DFlex FEM、预切虚拟节点/弹簧，回合内拓扑预先制备。
IKEA曲刃已接距离场/接触几何并重新生成场景，不是只换显示。
材料步长黄瓜/新增三食物10 μs、半土豆8 μs，500控制步不改变这些求解间隔。
接触采用edge_coordinate_consistent_v2一致插值/力散射。
net为三维净反力与力矩；旧scalar是另一种信号约定，不能混称六轴测力。

**BladeMaster**：本地Warp复现MPM，接触、切削作用区和侧别更新读取实际刃线；
不需要DiSECt预设切割拓扑。世界间材料缓冲、历史、reset独立。
原生实现见[来源](reference/SOURCES.md)，不要换回旧直楔刀分支。
学习不通过物理求导；原生库梯度测试不是当前长训的必经项。

共享机器人、FDCC、工具安装、任务、采样分布与wrench坐标，
不代表接触/损伤律或反力相等。食物自由摆放，不夹持。
分离仅诊断：DiSECt残余弹簧；BladeMaster记components/side codes等，
separated为null，不以分量数或侧别宣称预定切片脱落。
日志不进入奖励/成功/策略/WM。纹理也不决定材料。

## Implementation locations

cutting_world_model/：模型、任务派生、SAC/规划、replay与学习；
backends/mujoco/cutting_gpu_backend/：机器人、FDCC、DiSECt与测力耦合；
blademaster_backend.py连接外部原生库。视觉为切前选目标，
见[视觉](methods/VISION.md)，不作为低层图像输入。

Torch/Warp共享显存：默认physics/reset前释放空闲Torch缓存，
支持expandable_segments，按Warp高水位+1 GiB检查余量；
不自动减环境数、不吞OOM。日志区分当前阶段峰值与已结束阶段。
实测修复/旧任务结果只在[结果](RESULTS.md)记录，不在方法里累积开发日记。

现场ROS1控制器沿用同一SliceIt来源补丁，新增刚度参数范围修正。
UR内置补偿力采用独立native配置；空中增益与训练参数分开，
来源和运行边界见[真机模块](../hardware/ur10e_ros1/README.md)。
