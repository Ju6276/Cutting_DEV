# tamsCage：UR10e 控制电脑与现场控制器

记录日期：2026-11-21（Europe/Berlin）。以下为 SSH、ROS 只读查询和三秒反馈采样结果，
不是运动或切割验收。此次没有激活控制器、发布目标、清零传感器或改动现场参数。

## 电脑与目录

- 主机：tamsCage；地址：134.100.13.143；账号：dong。
- 本次经 tams61 跳板成功连接；认证由用户提供，凭证不进入仓库。
- Ubuntu 20.04.6 LTS；内核 5.15.0-153-lowlatency；ROS Noetic。
- 用户目录：/homeL/dong。
- 项目仓库：/homeL/dong/Cutting_DEV，现场仍在 hardware/ur10e-mpc-dry-run 分支，
  本次未切换该分支，未将它当作已同步到最新 main。
- FDCC 工作区：/homeL/dong/ur10e_fdcc_ws。
- ATI 工具仓库：/homeL/dong/ATI-Mini45-DEV。
- 现场记录：/homeL/dong/ur10e_fdcc_records。
- 本次原始查询：该目录下 codex_readonly_20261121/，
  包含 controller_inventory.json、source_inventory.json、feedback_check.json、
  loaded_libraries.json。早期 inventory 中 XML 解析失败仅因 rosparam CLI 的
  YAML 字符串包装；随后通过 rospy.get_param 正确解析，链接列表见 source_inventory。

## 实际加载的控制器

现场是 FZI cartesian_controllers 的 ROS1 分支，源码目录：
/homeL/dong/ur10e_fdcc_ws/src/cartesian_controllers，查询时工作区无改动。
实际 UR 驱动进程加载了此工作区 devel/lib 下的
libcartesian_compliance_controller.so 和 libcartesian_controller_base.so。
仓库源码版本用于追溯，不据此宣称二进制与当前源码已做可重现构建校验。

ROS 控制器名 fdcc，类型 position_controllers/CartesianComplianceController。
查询时状态是 initialized，**不是 running**。
运行的是 joint_state_controller、speed_scaling_state_controller、
force_torque_sensor_controller 等反馈控制器。

/fdcc/target_frame 无发布者；/fdcc/ft_sensor_wrench 无发布者。
/wrench 由 /ur_hardware_interface 发布，不能视为 ATI 外置传感器已经接入。
三秒只读采样取得关节消息 1501 条（约 500 Hz），UR wrench 1500 条（约 500 Hz），
FDCC 力输入消息 0 条。UR wrench 的 header.frame_id 是 tool0_controller。
这里测到的是反馈发布频率，不是已经验证 FDCC 实际闭环运行在该频率。

关节消息顺序从 elbow_joint 开始，而控制器配置从 shoulder_pan_joint 开始。
桥接必须按 joint name 映射，不可直接假定数组排列一致。

## 与当前 UR10e 仿真控制器的具体差异

两边都基于 FDCC，但不是同一配置，也不是完全相同的控制律。

1. **坐标与工具**：现场 end_effector_link、ft_sensor_ref_link、
   compliance_ref_link 都是 tool0；当前仿真分别为 knife_control、
   ati_sensing_frame、knife_compliance。现场 robot_description 中只有 UR 本体、
   flange 和 tool0，没有 ATI、刀架和刀具链接。不能只更改参数中的链接名字。
2. **力位组合**：现场上游 computeComplianceError 为位置弹簧项加力误差；
   仿真启用 SliceIt 适配的 use_parallel_force_position_control，
   按 selection 对位置和力项加权。现场源码中未实现仿真这组 sel_* 选择参数；
   仅向 ROS 参数服务器加入这些名字不会自动实现相同行为。
3. **刚度**：现场平移三轴均 500 N/m、转动三轴均 20 N·m/rad；
   仿真参考轴下平移为 [5000,2000,2000]、转动为 [80,40,80]。
   坐标系不同，不能把相同的轴编号当作相同的物理方向。
   这些是名义配置；仿真动作映射器还会按动作改变控制参数。
4. **求解及增益**：现场 iterations=10、error_scale=0.5；
   仿真 iterations=1、error_scale=0.0168。
   现场平移 P=[0.05,0.05,0.05]、转动 P=[1.5,1.5,1.5]；
   仿真名义值为 [0.2,0.2,0.05] 和 [2,0.2,2]，D 均为零。
   数字差异不直接等价于实际刚度倍数。
5. **负载与力解释**：现场 /fdcc 未配置 tool 和 gravity，源代码缺省为零补偿；
   仿真包含估算工具质量约 0.352681 kg、传感器坐标系重心 z 约 66.838 mm、
   基座重力 [0,0,-9.8]。此处是 ATI 下游负载，不等于 UR 法兰全部负载。
   现场 hand_frame_control=true，仿真配置为 false，也需在目标力坐标转换中对齐。

仓库仿真来源：平台资产 ur10e_ati_ikea_fdcc_v2/controller.json、
cutting_world_model/controller_profile.py，以及
backends/mujoco/cutting_gpu_backend/controller/fdcc.py。
现场公式来源：cartesian_compliance_controller.hpp 的 computeComplianceError；
负载缺省来源：cartesian_force_controller.hpp 的初始化逻辑。

## MPC 当前已完成的范围

解析 MPC 的去皮基准错误已经修复并进入 main：每个世界的预测固定使用
实际物理控制器导出的基准。DiSECt 和 BladeMaster 各两世界的短物理验证，
覆盖规划、一步物理执行、observe、部分 reset 和再次规划，均通过。
旧代码同一无外力转动回归会产生虚假力；修复后该断言通过。
完整验证范围与日志见 [METHOD](../METHOD.md)。

这没有改变原来的恒定历史力近似、粗略控制器递推和理想关节跟踪；
也不证明上述现场 ROS1 控制器已等价于仿真控制器。
[真机 MPC 交接](UR10E_MPC_HANDOFF_ZH.md) 的仿真评估入口不能直接下发真机指令。

## 接下来的现场工作顺序

先在不发布运动指令的条件下补齐 ATI 的实际采集与 ROS 输入，
核对测量原点/轴向、单位、时间戳和补偿方式；不能拿 UR /wrench 代替 ATI。
再把实际安装几何加入机器人描述，对齐控制点、工具负载和坐标变换。
选定并验证与当前算法相符的控制律、参数与动作映射，然后做只计算不下发的 MPC
影子运行，检查预测、时延和命令映射。完成这些后才制定明确的现场运动测试。

本次没有改动以上现场状态，没有因用户在旁就自动启动 fdcc。
