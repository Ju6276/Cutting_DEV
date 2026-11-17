# UR10e + ATI Mini45-E：FDCC 真机配置与验证交接

更新：2026-11-17。给另一台 Linux 控制电脑及接手开发者阅读。
**状态：配置方案，尚未在这台 UR10e 上安装或通过真机验收。本文不启动机器人，不授权远程无人值守运动。**

## 1. 本轮目标与已确认硬件

先跑通：预设笛卡尔目标 → FDCC → UR10e，ATI 提供实时六维反馈。
先空载运动，再用无刃测试工具验证柔顺接触；不接 RL、MPC、相机，不切食物。
不新增“接近板面自动切换控制器”的逻辑。

| 项目 | 已确认 / 待确认 |
|---|---|
| 机器人 | UR10e；PolyScope 版本、机器人 IP、实际运动学标定文件待读取 |
| 外置传感器 | ATI Mini45-E，用户确认标定 SI-580-20 |
| 标定量程 | 传感器坐标系 Fx/Fy ±580 N，Fz ±1160 N；Mx/My/Mz ±20 N·m；不是任务停止阈值 |
| ATI 采集 | 接口类型、采集盒固件、IP/协议、有效标定文件与序列号待现场确认；历史记录提到 Net F/T 9105-NETBA，不能据此直接假定当前接线 |
| 连接件、刀具 | 同事制作，CAD、质量、重心和安装方向待用户提供；第一轮不装刀 |
| 相机 | 梅卡曼德，型号待确认；本轮不接 |
| 控制电脑 | Ubuntu/ROS 版本、网卡、管理员权限待确认；不要求训练 GPU |

SI-580-20 的量程不等于可任意同时施加的六轴载荷，更不等于刀架/工具允许载荷。
真机停止阈值在实际工具与测试条件下确定，不照搬仿真的 50 N / 5 N·m。

## 2. FDCC 是哪个项目

- 算法：Forward Dynamics Compliance Control，Scherzinger 等，IROS 2017，
  [论文 DOI](https://doi.org/10.1109/IROS.2017.8206325)。
- 采用的开源实现：[FZI cartesian_controllers](https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers)，ROS 2 分支 `ros2`，BSD-3-Clause。
- 实际柔顺控制插件：`cartesian_compliance_controller/CartesianComplianceController`。
- 同仓库的 `cartesian_motion_controller` 用于先排查纯运动接口；
  `cartesian_force_controller` 是另一个力控制插件，不把三者都叫 FDCC。
- 机器人驱动：[Universal Robots ROS 2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)。
- 官方 FZI README 链接的 UR10e 接入示例：
  [cartesian_controllers_universal_robots](https://github.com/stefanscherzinger/cartesian_controllers_universal_robots/tree/ros2)。

2026-11-17 查阅源码的固定版本：
- FZI：`f8aad77697802d9b9ca08d044b87ce58167e4bf9`。
- UR 接入示例：`2c45b972c39811ae5d7a9b7d0fb81daa0d29a875`。

这些是阅读基线，不代表已经与目标电脑的 UR 驱动版本一起编译通过。
FZI 当前实现与论文原型有所演进，当前 ROS 2 实现也不能自动视为仓库历史 SliceIt ROS1 补丁的逐位等价实现。

### 示例里已经发现、接入时必须修正的地方

上述 UR10e 示例并不是可以原样启动的本项目部署包：
1. Xacro 引用了仓库自己的 `config/test_calibration.yaml`，必须替换为这台 UR10e 的标定。
2. 它把 UR 内置力传感器 broadcaster 的输出重映射到 `ft_sensor_wrench`，**不是 ATI**。
3. 示例 `ft_sensor_ref_link: tool0` 也不能当作 ATI 安装坐标。
4. 旧 launch 使用 `ur_ros2_control_node`；与选定发行版驱动的可执行文件及参数接口核对，不直接套到新驱动。
5. 示例 IP、刚度、增益、周期都不是本机已验证值。

## 3. 系统选择和权限

新装电脑建议 Ubuntu 24.04 + ROS 2 Jazzy；如果已是 Ubuntu 22.04 + Humble，
保留现有组合并使用对应发行版驱动/依赖。不要混用 Noetic、Humble、Jazzy 的环境。
FZI 仓库有 Humble/Jazzy 的 CI 配置，但仍需在本机编译核对。

- 管理员一次性处理：安装 ROS/apt 依赖、网卡配置、必要的设备访问和实时调度权限。
- 普通用户处理：源码编译、参数修改、记录数据、启动程序。
- 不使用 `sudo ros2 ...` 绕过错误。
- 本轮 Linux 用户目录安装，不在 Mac 下载源码/训练数据。
- 仿真训练使用的 Python 3.12/CUDA 虚拟环境不用带到 ROS 工作空间。

先记录（只读）：
```bash
cat /etc/os-release
uname -r
printenv ROS_DISTRO
ip -br address
```
记录 PolyScope 版本、机器人 IP、控制电脑有线 IP、ATI 采集盒接口和 IP。
未安装 ROS 时，先按 [Jazzy Ubuntu 安装说明](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
或对应 Humble 文档完成管理员安装，之后再执行下文。

## 4. 下载文档与编译 FDCC

另一台电脑先获取本项目文档（已有仓库时只做干净工作区的快进更新）：
```bash
git clone https://github.com/Ju6276/Cutting_DEV.git
cd Cutting_DEV
git log -1 --oneline
```
本文不要求安装 Cutting_DEV 的 GPU 训练依赖。

以下为 Ubuntu 24.04/Jazzy，**只安装与编译，不连接或驱动机器人**：
```bash
source /opt/ros/jazzy/setup.bash
sudo apt update
sudo apt install ros-jazzy-ur ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
  python3-colcon-common-extensions python3-rosdep git build-essential

mkdir -p "$HOME/ur10e_fdcc_ws/src"
cd "$HOME/ur10e_fdcc_ws/src"
git clone -b ros2 https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers.git
git -C cartesian_controllers checkout f8aad77697802d9b9ca08d044b87ce58167e4bf9
cd "$HOME/ur10e_fdcc_ws"
```

首次使用 rosdep、且机器尚未初始化时由管理员运行 `sudo rosdep init`；
已初始化不要重复。随后：
```bash
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build --packages-skip cartesian_controller_simulation cartesian_controller_tests \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 pkg prefix cartesian_compliance_controller
ros2 pkg prefix ur_robot_driver
```
如果目标发行版依赖/API 不兼容，保留构建日志，针对该版本修复；
不要悄悄跟随最新分支后仍报告固定版本通过。

记录环境：
```bash
mkdir -p "$HOME/ur10e_fdcc_records"
git -C src/cartesian_controllers rev-parse HEAD > "$HOME/ur10e_fdcc_records/fdcc_commit.txt"
dpkg-query -W 'ros-jazzy-ur*' > "$HOME/ur10e_fdcc_records/ur_packages.txt"
```

## 5. UR10e 驱动与标定：先连接，不自动运动

跟随 [UR Jazzy 机器人设置](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/jazzy/doc/ur_robot_driver/ur_robot_driver/doc/installation/robot_setup.html)
配置匹配版本的 External Control URCap、电脑地址与机器人程序。使用有线网络，
按选定驱动核对远程控制/程序启动要求，不假定只有 ping 通就能控制。

现场确定 IP 后，提取标定（机器人可保持 idle；不是运动命令）：
```bash
read -r -p "UR10e IP: " UR10E_IP
mkdir -p "$HOME/ur10e_fdcc_records"
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:="$UR10E_IP" \
  target_filename:="$HOME/ur10e_fdcc_records/ur10e_calibration.yaml"
```

先查看本机 driver 支持的参数：
```bash
ros2 launch ur_robot_driver ur_control.launch.py --show-args
```
随后按该版本启动文档配置 `ur_type:=ur10e`、robot_ip 和真实标定文件。
这是启动硬件控制通道的步骤，由现场人员执行；本文不提供会自动发运动目标的脚本。
核对标定无 mismatch、模型确为 UR10e、关节状态与示教器一致。

只读检查：
```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
ros2 topic echo /joint_states --once
```
不同时激活多个争用同一关节命令接口的运动控制器。驱动默认轨迹控制器、
FZI 运动控制器和柔顺控制器之间的切换需在集成 launch 中明确管理。

## 6. ATI 采集、坐标与独立接入包

在工作空间中新建本机的 `ur10e_fdcc_bringup` 包；这是接手电脑需要完成的实现，
**目前 Cutting_DEV 没有提供已经验证的该包**。建议交付：
- `urdf/ur10e_ati_tool.urdf.xacro`：本机运动学标定、ATI/转接件/测试 TCP。
- `config/controllers.yaml`：FZI 插件、六关节、坐标系、调度及逐项验证后的参数。
- `config/ati.yaml`：采集接口、SI-580-20 标定、序列号、时间戳/单位，不提交凭据。
- `launch/bringup.launch.py`：UR 驱动、传感器、补偿、日志，FDCC 默认 inactive。
- 目标发送和状态监督：初始化目标为当前实际姿态；启动不自动走向零位。
- 数据过期、目标发布者退出、通信丢失及停止处理；测试完再允许接触。

采集发布原始与补偿后的 `geometry_msgs/msg/WrenchStamped`：
建议 `/ati/wrench_raw`、`/ati/wrench_compensated`，明确 frame_id、N/N·m 与采样时间。
这些是建议接口名，不声称已有驱动会直接发布。若确认 Net F/T，按采集盒官方协议/
标定读取并转换 counts；其他接口用其相应 SDK。UR 内置力可独立记录，不能冒充外置 ATI。

机械链：机器人法兰 → 传感器安装侧 → 工具侧 → 夹具/测试工具。
图纸 [Mini45-E 9230-05-1315](https://www.ati-ia.com/app_content/Documents/9230-05-1315.auto.pdf)
规定轴向、测量原点和螺钉深度；不能以外壳中心或电缆方向替代定义。
[SI-580-20 官方量程](https://www.ati-ia.com/products/ft/ft_models.aspx?id=Mini45)。

配置对应关系（不是可直接启动的 YAML）：
- `robot_base_link`：实际 URDF 的机器人基座 link；注意 UR 的 base 与 base_link 不能混用。
- `end_effector_link`：本次测试 TCP。
- `ft_sensor_ref_link`：ATI 测量原点，原始轴向对应的 link。
- `compliance_ref_link`：选择并记录柔顺参考点，第一轮可用测试 TCP。
- `command_interfaces`：优先核对 position 接口；按本机驱动实际支持配置。
- 六个 joints 与机器人/前缀完全一致。
- 显式将 FDCC 的测量输入 remap 到 ATI 补偿主题；核对该固定版插件的实际主题名和类型。

偏置扣除与重力补偿不是一回事。工具质量/重心需测量；在多姿态下验证无接触残差。
力的旋转与力矩的参考点平移都要正确，确认驱动与 FDCC 谁执行变换，避免重复计算。
补偿后的读数仍保留明确的坐标系。初始化滤波与历史必须来自真实采样。
完整日志包括六轴原始/补偿信号，不只记录 Fx/Fz/My。

## 7. 本轮验证顺序与完成标准

不直接照抄仿真 UR5e 的 controller.json 或上游演示增益到真机。
先选定现场可接受的速度/位移、六轴载荷边界和停止方式，再逐级增加响应。
FZI 的 `error_scale=0` 可用于零响应初始化，但不是硬件安全停止功能。

1. **只读检查**：机器人状态、ATI 时间戳与六轴方向有效；TF/CAD 与实际一致。
2. **运动检查**：无刀、空载或无刃测试工具；当前姿态初始化目标；
   用 CartesianMotionController 做逐轴小位移，确认方向、跟踪、停止和无意外跳变。
3. **FDCC 检查**：ATI 反馈已校正，切换至 CartesianComplianceController；
   低响应起步，逐轴轻触，确认响应方向，检查力矩与力臂；不直接从演示高增益开始。
4. **接触检查**：低速接触选定测试表面，观察持续力反馈、稳定性、释放后的恢复；
   停止后不冻结传感器数值。数据中断时不得无限执行上次命令。
5. **记录与复核**：保存目标、q/qd、TCP、六轴 wrench、控制器状态、时间戳、
   启停/超限事件和参数。实测周期、延迟、停止时间，不承诺未测的 1 kHz 或零停止距离。

每步记录“通过 / 失败 / 未测”和证据；没有现场硬件测量不能标为通过。
本轮不以“切开”作为验收，不需要训练或 GPU。

## 8. 给另一台电脑上的开发者

请依次完成：
1. 回填 OS/ROS、PolyScope、网络和 ATI 采集接口，选定匹配的软件版本。
2. 安装并编译上述 FDCC；保存版本与完整错误日志。
3. 用本机标定和传感器安装建模；适配 UR10e 示例，不运行其未修改 launch。
4. 建立独立 bringup 包，检查默认 inactive、当前姿态初始化、单一命令控制器、
   外置 ATI 数据来源、数据过期和停止处理。
5. 先做 mock/离线接口检查，再由现场人员按第 7 节做无刀真机验证。
6. 提交 bringup 源码、安装说明、参数与结果摘要；标定文件按实验室规定存放，
   网络凭据、bag、大文件不提交 Git。

需要用户补充的是实物安装 CAD/工具参数、电脑与采集接口信息，不是额外训练批准。
不自动启动切割、不接旧策略、不实现额外板面接管。

## 9. 与当前仿真、后续 MPC/RL 的关系

当前仿真正式任务是 pose_depth_v3，v2 只作旧实验复核。本轮不修改仿真任务或训练。
FDCC 真机闭环不复制仿真的 board_contact 锁存/冻结材料/冻结 wrench；
“内部规则一致性修复”不是已完成真机迁移。
真机先确认 FDCC 实际行为，后续再决定任务的停止与结算如何对应。

FDCC 通过后，固定轨迹、解析 MPC、RL 可以共用底层驱动与反馈接口。
UR5e 仿真策略/WM 不能直接认定适用于 UR10e；MPC 的在线时延与真实受力模型
也要单独验证。本轮不接梅卡曼德，不要求切前板面识别或视觉闭环。

关联：[真机总体计划](REAL2SIM.md)、[当前任务](TASK_FREE_CUT.md)、[当前计划](PLAN.md)。

## 10. 本次文档交付的验证范围

- 2026-11-17：核对 FZI/UR 官方资料及上述固定示例源码；本次仅修改文档。
- 五个修改文档的本地链接与本文 bash 代码块语法检查通过。
- 按仓库规则，在空闲 GPU9 对基线源码 58dbb7e 重跑：主仓库 639 passed / 3 skipped；
  后端 41 passed。日志在 GPU9
  /dataHDD/dong/Projects/Cutting_DEV_ur10e_docs_check/runs/ur10e_docs_check/。
- 未在目标控制电脑安装 ROS，未编译该 FDCC 工作空间，未连接 UR10e/ATI，
  未发送运动指令。这些 GPU 回归不构成 FDCC 真机验收。
