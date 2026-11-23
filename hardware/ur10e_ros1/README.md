# UR10e 真机模块（ROS1）

这里是 Cutting_DEV 内的真机模块，不是另一个仓库。算法和仿真仍使用原有目录。
现场是 tamsCage / Ubuntu 20.04 / ROS Noetic。设备及标定统一见
[硬件手册](../../docs/HARDWARE.md)，本文只维护当前构建与操作方法。

**已实测：UR内置力反馈、FDCC空中往返、连续FDCC＋GPU解析MPC空中联调、
位置轨迹下移和训练刀具朝向对齐。未验证食材接触切割或学习策略的真机执行。**
本次整理不移动机器人；新入口只做离线核验，运动结果来自原始现场记录。

## 模块内容

- `config/`：native-UR力反馈配置、连接信息、训练朝向及用户认可位姿。
- `controller/`：现场C++控制器的构建脚本与新增补丁。
- `launch/`：UR驱动入口；两个运动控制器均以stopped启动。
- `scripts/fdcc_air.py`：固定朝向空中往返，外层入口为 `run_fdcc_air.py`。
- `scripts/continuous_supervisor.py`：连续FDCC＋GPU解析MPC空中入口。
- `scripts/training_orientation.py`：保持当前控制点与水平航向，匹配训练刀尖下倾。
- `scripts/vertical_reposition.py`：显式指定基座Z位移的位置轨迹，非FDCC/MPC。
- `scripts/capture_pose.py`：只读保存当前关节与各工具位姿。
- `verification/`：离线数值对照，不连接机器人。

不导入gated MPC、失败的全图捕获原型、临时源码副本、旧机器交接文档。
原始日志保留在仓库外；内部worker不是独立操作入口。

## 保存的位置

[accepted_pose.json](config/accepted_pose.json) 是2026-11-23用户认可的现场位姿，
含六关节角和工具坐标。基座系 `knife_control` 约为
`[0.057522, 0.961404, -0.063962]` m，刀尖下倾8.25°、侧倾接近0°。
这不是完全水平，也不是标定后的切割终点。

训练朝向经水平航向对齐后作用于当前控制点，**未复制仿真桌高或名义初始关节角**。
程序启动不会回放这个位置。底座、工具或桌面变更后重新核验。

## 控制器与力反馈

插件是SliceIt/cambel修改版FZI
`position_controllers/CartesianComplianceController`，实例名 `fdcc_shared`。
它在控制电脑ROS进程内运行，经UR驱动送关节目标，不修改控制箱固件。

构建复用[仓库已有来源补丁](../../integrations/sliceit/patches/2-cartesian_controllers.patch)，
再应用刚度范围修正 `controller/compliance_range.patch`。上游许可证留在下载的源树中。
未改变本仓库仿真训练控制律。

当前用**UR内置已补偿 /wrench**，原参考系 `tool0_controller`；
换算到 `ati_sensing_frame` 时同时处理旋转和力矩原点。
ATI仅机械安装，电信号未接通。native配置的 `tool.mass=0` 表示FDCC不重复补偿重力。
UR法兰总负载估计0.555227 kg，ATI下游刀具/刀架约0.352681 kg，两者范围不同。
假刀具尚未称重；脚本不自动改Payload或清零。

默认 `error_scale=0.0168`，成功空中联调用0.2。
MPC预测与执行同时使用0.2，结束恢复原值。
离线控制公式一致不等于两边动态响应一致。

## 构建与环境

从Cutting_DEV根目录运行。新构建用独立目录，不覆盖活动工作区：

```bash
hardware/ur10e_ros1/controller/build.sh "$HOME/ur10e_fdcc_module_ws"
source "$HOME/ur10e_fdcc_module_ws/devel/setup.bash"
```

需已有ROS Noetic、UR ROS1驱动及上游catkin依赖。脚本只构建，不启动驱动或运动。
原已验工作区 `$HOME/ur10e_fdcc_alignment_ws` 保留。

追加工具时保留本机已标定机械臂：

```bash
python3 hardware/ur10e_ros1/scripts/prepare_description.py \
  --arm-urdf "$HOME/ur10e_fdcc_records/alignment_candidate_20261121/robot_description.original.urdf" \
  --tool-urdf "$HOME/ur10e_fdcc_records/alignment_candidate_20261121/platform/robot.urdf" \
  --mesh-root "$HOME/ur10e_fdcc_records/alignment_candidate_20261121/platform" \
  --output "$HOME/ur10e_fdcc_records/robot_description.current.urdf"
```

这个离线命令拒绝覆盖文件或重复追加工具；现场已有描述可以继续使用。
STL、工厂标定、虚拟环境和运行记录不入Git。

ROS用系统Python3.8；GPU规划另用Python3.12 / Torch2.4.1+cu118，
适配GTX1060（sm61），不把GPU环境用于导入ROS。
规划依赖使用[固定baseline依赖](../../requirements/baselines.txt)中的pytorch-mppi版本。

```bash
export CUTTING_PLANNER_PYTHON="$HOME/Cutting_DEV/.venv-mpc-cu118/bin/python"
export CUTTING_REAL_RUNS="$HOME/ur10e_fdcc_records/current"
export CUTTING_KNIFE_MESH="$HOME/ur10e_fdcc_records/alignment_candidate_20261121/platform/knife.stl"
export CUTTING_ROBOT_IP=192.168.0.123
```

上述规划环境是现场已安装环境；其他电脑需指定自己的CUDA环境，不修改训练服务器依赖。

## 启动与只读检查

机器人192.168.0.123；控制电脑直连网口192.168.0.1。
External Control的Host IP为直连电脑地址、端口50002，机器人为Remote。
不要同时启动旧FZI launch或第二个UR驱动。

```bash
export CUTTING_ROBOT_DESCRIPTION="$HOME/ur10e_fdcc_records/alignment_candidate_20261121/robot_description.candidate.urdf"
roslaunch hardware/ur10e_ros1/launch/remote_control.launch \
  calibration:="$HOME/ur10e_fdcc_ws/config/ur10e_calibration.yaml"
```

此命令建立驱动连接，但运动控制器不启动、不自动回放位姿。

```bash
python3 hardware/ur10e_ros1/scripts/capture_pose.py
python3 hardware/ur10e_ros1/scripts/training_orientation.py
python3 hardware/ur10e_ros1/scripts/vertical_reposition.py --delta-z-m -0.02
python3 hardware/ur10e_ros1/scripts/continuous_supervisor.py
```

没有 `--execute` 不发运动。执行检查使用RTDE真实PLAYING状态：
ROS的robot_program_running曾在暂停时仍为true，不能只看这个布尔值。
反馈/规划过期、位移、关节速度及力超界时停止。

## 现场准备好后的执行入口

以下命令会运动，一次只选一个：

```bash
# FDCC往返：临时增益0.2，结束恢复。
python3 hardware/ur10e_ros1/scripts/run_fdcc_air.py --execute
# 持续FDCC＋异步GPU MPC：竖直10 mm目标，最多20 s。
python3 hardware/ur10e_ros1/scripts/continuous_supervisor.py --execute --bounded-stiffness
# 位置控制准备动作，非柔顺切割：
python3 hardware/ur10e_ros1/scripts/training_orientation.py --execute
python3 hardware/ur10e_ros1/scripts/vertical_reposition.py --delta-z-m -0.02 --execute
```

不自动清零或改负载。空中默认合力8 N、合力矩0.5 N·m，
与论文任务50 N / 5 N·m分量限值不同。接触实验需另定参数。
退出停止相应运动控制器。不能用 `python -O` 移除运行检查。

## MPC的实际范围

这是限定竖直方向的空中工程测试，不是论文完整六动作算法验收：

- GPU64候选、4步窗口；用真实q/qd/wrench更新，不拿预测关节作反馈。
- 同一目标插值持续生成参考，MPC调柔顺参数；拉锯和倾角动作固定0。
- FDCC持续运行，目标约20 Hz更新，规划异步约0.6–0.8 s一次，**不是20 Hz MPC**。
- 模型仍是理想关节跟踪、窗口内保持测得力不变，没有食材模型。
- 刀下30 mm虚拟平面仅满足状态定义；不是测量的桌面或碰撞保护。
- 当前上限20 s处于25 s模型时限内；独立监控拒绝超过7 s的规划。
- CUDA捕获仅在独立规划进程使用；ROS进程负责反馈、控制器及停止。
- 空中到位为3D误差1 mm、朝向1°、连续11帧，与论文累计计数不同。

原始证据根目录 `$HOME/ur10e_fdcc_records/`：
`native_ft_commissioning_20261123/air_execute_172833.json`、
`mpc_native_air_20261123/continuous_execute_204946/result.json`、
`lower_table_20261123/execute_210001.json`、
`training_tool_pose_20261123/execute_211709.json`。
实测摘要统一见[结果](../../docs/RESULTS.md#hardware-checks)。
ATI接通、实测负载/TCP/桌面、接触响应和学习策略部署尚待完成。

## 本次整理验证

控制器已在tamsCage的新工作区从固定来源和补丁重新构建通过；launch仅离线解析，未启动驱动。
GPU离线回放中，1/32/64候选的加速预测与原模型状态/奖励一致，终止判定一致；
真实反馈状态推导一致。它们验证代码整理，没有重做实机运动或接触试验。

GPU7提交前回归：主仓库825通过、5跳过；后端48通过。
日志：GPU7仓库runs/ur10e_module_review_20261123/；现场构建/解析记录在
tamsCage的ur10e_fdcc_records/module_staging_20261123/。

切换到本分支后，tamsCage的GTX1060离线GPU回放也通过：
1/32/64候选状态和奖励最大差0，反馈推导一致；本次单次规划约0.594 s。
记录在ur10e_fdcc_records/module_validation_20261123/，没有连接机器人。
