# tamsCage：UR10e 控制电脑与现场控制器

现场记录：2026-11-21；文档整理：2026-11-22。原工作区保持不变，
独立候选已经通过离线输出对齐，但未接入活动UR驱动。以下查询状态是当日快照，
不是当前正在运行的进程断言；用户随后已关闭机器人。没有发送机器人运动指令。

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

## 原有现场控制器与仿真的差异（尚未切换候选）

以下描述原有 /homeL/dong/ur10e_fdcc_ws，不是后文已经离线对齐的候选。

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

## 接入文档的分工

本页只维护现场盘点、原控制器和独立候选及其对齐证据。
步骤见 [FDCC接入](UR10E_FDCC_SETUP_ZH.md)，上层接口见
[MPC交接](UR10E_MPC_HANDOFF_ZH.md)，标定方法见 [REAL2SIM](REAL2SIM.md)。
仿真解析MPC修复和SAC/WM+MPPI审阅结果统一在
[RESULTS](../RESULTS.md#ur10e-baseline-verification)。

## 2026-11-21：ATI 暂缓后的离线对齐候选

用户要求暂缓 ATI 连接排查，先对齐末端坐标、控制律和动作映射。
以下内容已经在 tamsCage 创建并验证，**没有加载到活动 ROS 驱动**：

- 独立控制器工作区：/homeL/dong/ur10e_fdcc_alignment_ws。
- 配置、脚本、日志与模型：/homeL/dong/ur10e_fdcc_records/alignment_candidate_20261121。
- 从活动 /robot_description 只读导出原机器人描述，保留所有原机械臂节点，
  在 tool0 下增加仿真平台的九个固定工具链接。检查原节点不变、链接唯一、
  三个控制/传感器参考点可达，check_urdf 解析通过。
  文件是 robot_description.original.urdf 和 robot_description.candidate.urdf。
  这里保留的是现场现有机械臂参数，不声称已验证它的机器人序列号或工厂标定来源。
- fdcc.candidate.yaml 从仿真平台 controller.json 生成，反序列化逐值一致。
- 控制器使用项目固定的 SliceIt/cambel cartesian_controllers 源码及现有补丁，
  在新工作区用 catkin_make -j2 -l2 编译通过，没有覆盖旧工作区或热替换驱动库。
  该分支实现 S*K*位姿误差 + (I-S)*力误差，与仿真力位加权的来源一致。
- action_mapping_check.json 使用原样复制的 SlicingActionMapper，未另写一套公式。
  在现场既有 Python 环境进行三个离线动作检查：刚度/增益上下界、
  正向动作累计 2 mm、反向抵消、第六通道禁用及有限值检查均通过。
  映射器每个策略步调用一次（20 Hz）；不能每个 500 Hz 伺服周期重复累计。
  这是映射检查，不是已实现 ROS 命令发送桥，也不代表完整 MPC 已现场运行。

### 离线检查发现的参数范围不匹配

SliceIt C++ 分支 dynamic_reconfigure 将 trans_x 的上限设为 5000 N/m，
而当前仿真动作映射可输出 10000 N/m。直接传相同动作，会在 ROS 配置层被裁掉，
即使两个控制器同源，动作仍不等价。

使用真实生成的 C++ ComplianceControllerConfig::__clamp__ 验证：
原配置将 10000 改成 5000，检查返回失败；只在独立候选工作区把 trans_x
允许上限对齐到 10000，重新生成和编译后，相同检查保留 10000 并通过。
没有改变默认刚度、任务力限、仿真动作或任何活动机器人参数。
证据：check_config_range.cpp、range_before.log、range_after.log、
controller_range.patch、build.log、build_range.log。
候选修改用于排除部署接口失真，**不是已经允许真机按最大刚度运行**。

### 尚不能声称完成的部分

- ATI 轴向和安装角目前仍是仿真几何假设，需要现场核对；质量、重心仍含 CAD 估算。
- 有负载时，C++ 初始化的重力基准不等于已完成去皮。
  恢复 ATI 后必须在同一静止姿态同步传感器清零与 FDCC 的 signal_taring，
  并测零力残差；本次没有调用清零或 taring 服务。
- 同源控制律和相同参数不代表完整数值/时序等价，固定输入的关节输出
  对照已按后文完成；仍需实际输入的影子运行，动态参数传输也需要保证一致的生效时刻。
- 真机闭环、实测工具坐标、ATI 输入、机器人伺服及停止行为尚未验收。
  现有 fdcc 仍保持原状态，未发布目标或关节指令。

## 两种六维力信号的区别

UR10e 自带末端六维力/力矩传感器；现场 UR 驱动发布 /wrench，
本次消息坐标为 tool0_controller。它和外置 ATI Mini45 是不同的传感器，
测量位置、下游负载及标定不同，不能直接交换数据源。
依据：[UR10e 官方数据表](https://www.universal-robots.com/media/1807466/ur10e_e-series_datasheets_web.pdf)。

当前仿真不连接真实 ATI，也不是读取真机 UR /wrench：
在虚拟 ati_sensing_frame 处，由 MuJoCo 机器人内部力和材料后端计算的
接触反力/力矩组成读数，按测量原点和坐标变换，经过源滤波/去皮链路送入 FDCC。
DiSECt 与 BladeMaster 的材料计算不同，但都使用该传感器坐标约定。
当前砧板仍只有几何防穿透边界，没有板面反力模型；不能把仿真 wrench
称为包含真实砧板反力、ATI 噪声或已标定的全部传感器特性。


## 同输入的 ROS C++ / Python 控制器对照（2026-11-21）

离线对照已完成，不再只以“编译通过”作为依据。使用上述候选 URDF 和配置，
在 tamsCage 的隔离 ROS master http://127.0.0.1:11349 上初始化真实
CartesianComplianceController C++ 模板及其已编译 IK solver。硬件接口由
内存中的六个关节 handle 提供，未连接活动 UR 驱动；测试结束关闭隔离 master。

输入包括连续 30 次变化的关节位置、末端目标和六维 sensor wrench；
保留非零工具质量/重心，初始同步去皮，控制器内部速度连续演化。
同一输入送入当前 Python FDCCController，比较六关节位置指令：
最大绝对误差 **1.3322676295501878e-15 rad**，阈值 1e-10，检查通过。
不是重写一个独立 C++ 公式来与 Python 互证，而是调用候选控制器的
init/starting/update 和实际力输入回调；force/target 的单位与原点保持一致。

证据位于 alignment_candidate_20261121：
controller_fixture.cpp、controller_fixture_input.txt、
controller_fixture_expected.json、controller_fixture_actual.log、
controller_output_comparison.json、prepare_controller_fixture.py。
覆盖固定控制参数下的数值计算和递推，不覆盖实际机器人伺服、网络时延、
在线动态参数同步、硬件关节饱和及真实 ATI 采集。实际控制器没有切换。
与仿真同源的 FDCC 将运行在 tamsCage 的 ROS 进程，通过 UR 驱动发送关节目标，
不是写入或替换 UR 控制箱固件。

## UR 内置 F/T 作为可选输入的边界

用户设备是 **UR10e**。不要和无实体末端 F/T 的旧 CB3 UR10 混称。
[UR 官方数据表](https://www.universal-robots.com/media/1807466/ur10e_e-series_datasheets_web.pdf)
列出各轴标称力范围 ±100 N、力矩范围 ±10 N·m；力准确度 ±5.5 N、
力矩准确度 ±0.5 N·m，precision 分别为 ±5 N 和 ±0.2 N·m。
准确度不等于分辨率，也不等于每个采样点都必然有该幅度噪声。
[e-Series/CB3 的官方论坛说明](https://forum.universal-robots.com/t/internal-force-sensor-properties/229/4)
区分 e-Series 工具末端实体 F/T 与 CB3 关节电流估算。

可以把 UR 自带传感器作为独立的输入方案用于链路测试，但本次没有切换：
必须核对驱动给出的 wrench 原点/坐标、TCP 与负载设置、已有重力补偿和清零，
然后变换到控制器约定，避免把已补偿的 UR 输出再当作 ATI 原始读数补偿一次。
ATI 及其上游转接件在 UR 内置传感器下游，而在 ATI 自身测量中不属于同一负载。
即使它们都输出六维数，也不能只改 topic 名称就认为接口等价。

对于很软食材的细小力特征，是否采用 UR 内置传感器，需比较实际静态噪声、
重复性和目标信号幅度；连接方便本身不证明更适合材料辨识。
