# Mech-Eye NANO V4：现场核对与回填

状态：2026-11-21 已回填第一轮可读取项；空板/食材样本及独立尺量待补，最低交付尚未齐全。
本轮只读取相机信息、导出配置和静态样本；不运行机器人、MPC 或 RL，
不升级固件，不恢复出厂设置，不覆盖当前相机参数组。
本页汇总 ju-ROG 的现场回填；[采集脚本](https://github.com/Ju6276/Cutting_DEV/blob/ec37631/scripts/diagnostics/check_mecheye_nano.py) 目前保留在相机工作分支，尚未作为 main 的运行入口合入。下文命令在该现场分支执行，不要求训练服务器安装相机 SDK。
已有硬件记录见 [Nano 仿真适配](../methods/MECH_EYE_NANO.md)。

## 1. 已确认，不必重新猜型号

- 用户通过 Linux 查询确认：NANO V4，硬件 4.0.0，固件 2.3.4。
- 照片铭牌序列号：TAC5522CA302E001。
- 彩色、550 mm 标称物距；对应官方目录型号 NANO-V4D550C。
  目录匹配名称与 API 返回的原始 model 字符串分开保存。
- 官方该型号推荐工作距离 450–600 mm；550 mm 不是镜头焦距。
- IP 贴纸 192.168.122.101 尚不能作为当前连接地址的证据。

当前仓库 `configs/vision/mecheye_nano_provisional_v1.json` 是旧的 350 mm
近似配置。不要只改名称；先用下面的实测信息建立实机相机配置。

## 2. 第一轮：现在从相机和电脑读取

请接手 Codex 按本机已安装 SDK 的示例/API 操作。固件版本不等于 SDK 版本，
不要假定固件 2.3.4 就必须安装同版本 SDK，也不要凭文档猜 API 函数名。
先检查现有环境；如已有采集脚本，复用它并记录版本。

1. **连接与软件**：当前设备 IP、API 原始型号/序列号/硬件/固件；
   Linux 发行版、SDK/API 版本、Python 版本及绑定包版本、采集脚本提交号。
   没有权限或接口不提供的字段写“未获取 + 原因”，不要填推测值。
2. **图像与内参**：分别导出 2D 彩色、深度的实际宽高；
   相应的 fx/fy/cx/cy、畸变模型及系数（若 API 提供），原始参数输出也保存。
   标明参数属于哪一个图像、是否已校正、是否裁剪或缩放。
   不把 RGB 内参直接当作深度内参。
3. **深度/点云约定**：深度数据类型、单位/比例、无效值定义；
   XYZ 的单位、坐标轴方向和参考光学坐标系；
   RGB 是否已对齐深度、点云颜色是怎样注册的。
   若 API 输出已校正或注册的结果，记录其保证范围；不要重复校正。
   单位由对应版本文档及一次已知距离/尺寸检查确认，不能仅看数值大小猜。
4. **采集参数**：导出当前活动参数组及名称；记录 2D/3D 曝光、
   HDR/多曝光、深度范围、ROI、点云滤波/平滑等本机实际提供的选项。
   未提供的参数明确写“不支持/未公开”；先保存原始配置，再考虑调参。
5. **最小样本**：固定相机，空砧板一组、砧板上一件静止食材一组。
   每组至少保存一次原始 RGB、原始深度、XYZ 点云及配套元数据；
   先不下采样。样本先用已有黄瓜或半个土豆，不要求五种食材齐全。
   记录每次采集的时间戳与一次采集耗时；若 RGB/深度分开触发，
   明确它们不是同一时刻。不要把单次耗时称为稳定帧率。
   刀具若挡住食材，记录遮挡；不要让脚本为了拍照自动移动机器人。

本轮最低交付是上述设备信息、参数导出、内参/单位说明和两组原始样本。
若方便，可额外拍 10 次同一静态场景估计重复性；这不是第一轮的硬性要求。
数据量大的图片、点云和原始导出放仓库外；Git 只存回填摘要、路径、SHA256
以及采集脚本（若有）。不要上传环境密钥或机器人凭证。

## 3. 第二轮：安装与标定，不能只靠读取设备获得

以下可以先留空，不妨碍第一轮提交：

- 相机安装照片、镜头大致朝向、到砧板的距离、可见砧板范围。
  标清距离测量起点；壳体到板面的尺量不能直接冒充光学中心距离。
- 砧板长宽、实际安装位置及高度；相机看到的桌面和砧板分别是哪一平面。
- 眼在手外标定：`T_base_camera_optical`，约定
  `p_base = T_base_camera_optical @ p_camera_optical`。
  平移以米计，记录源/目标 frame 名称、标定方法、时间、原始数据路径、
  独立验证点误差和安装是否之后移动过。
  不能用 CAD 外壳坐标系或手工摆放矩阵冒充已标定的光学外参。
- 若以后需要准确展示外壳与视锥的相对位置，另核对 CAD 到光学坐标系变换；
  此项与机器人基座外参不同。
- 食材点云缺测情况：湿表面、边缘、阴影和反光，之后用重复采集评估。
  第一轮不要为了补洞去改原始点云或宣称已标定噪声。

只有外参和板面都验证后，才能把相机里的切口位置作为机器人基座下的目标。
这份清单的完成本身不代表机器人运动链路已验证。

## 4. 在此回填

直接修改本节并提交到你的工作分支。每个数字注明“API / 官方文档 / 实测 /
估计”；无法获取的写原因。保留原始设备字符串，勿由目录型号覆盖。

### 4.1 本轮范围与设备/软件

本轮结论：设备信息、活动配置、内参和一组办公桌连通性 RGB-D/XYZ 已取得；
**第一轮最低交付未完成**。用户明确回复“现场暂不具备条件，记录未完成”，
因此空砧板、静止食材两组样本和已知距离/尺寸的独立核对均未做。
办公桌图像不冒充空板或食材样本；目录型号也不覆盖 API 原始字符串。

下文 `API` 表示设备/SDK 返回，`实测` 表示宿主读取、计时或保存数据的计算，
`官方文档` 表示对应版本接口说明；未使用估计数值。文件哈希均为文件字节的 SHA256。

| 项目 | 结果 | 来源 |
|---|---|---|
| 检查日期、操作者、电脑 | 2026-11-21；用户 ju，Codex 执行；主机 `ju-ROG` | 实测，Europe/Berlin 宿主时间 |
| 读取的 main 基线 | `5a4f61f25d423e96908411c9da931919827287d1` | 实测，Git |
| 采集脚本提交号 | `01fac11b20685ec489757f488ea8002563952451`，采集时脚本无已跟踪差异 | 实测，`metadata.json` |
| 采集入口 | `scripts/diagnostics/check_mecheye_nano.py`；复用本机已有 `capture_point_cloud.py` 的连接、联合采集和 SDK PLY 导出流程 | 实测，仓库脚本 |
| Linux / Python | Ubuntu 22.04.5 LTS；Python 3.10.12；解释器 `/home/ju/Downloads/mechmind-camera/.venv/bin/python` | 实测，系统与运行进程 |
| Python 绑定/依赖 | `MechEyeAPI==2.6.0`，`numpy==2.2.6`，`opencv-python-headless==5.1.0.95` | 实测，包元数据 |
| 实际加载的原生 API | `get_api_version_info()`：`Mech-Eye API V2.6.0 2026-06-17. Copyright 2016-2025 Mech-Mind Robotics Technologies Co., Ltd. All rights reserved.` | API 原文 |
| 独立安装的 SDK 包版本 | 未获取：未找到 `mecheyeapi` 的 dpkg 登记；本轮实际使用绑定包自带 `libMechEyeApi.so`，其路径/哈希另存 `sources.json`，不把固件号当 SDK 版本 | 实测 |
| API model 原文 | `Mech-Eye NANO` | API |
| API serial 原文 | `TAC5522CA302E001` | API |
| API hardware / firmware 原文 | `4.0.0` / `2.3.4` | API，`Version.to_string()` |
| 当前设备 IP / 子网掩码 | `192.168.122.101` / `255.255.255.0`；已用该地址实际连接，非仅抄 IP 贴纸 | API |
| 电脑连接 | 内置 `enp108s0`，主机相机网段地址 `192.168.122.100/24`，链路 `1000 Mbps` | 实测，网卡 sysfs 和路由 |
| 活动参数组 | `default`；可见组为 `default`、`calib`、`Metal welding`、`Charging port locating`，未切换组 | API |

### 4.2 图像、内参和坐标约定

RGB 与深度分别从 `CameraIntrinsics.texture`、`.depth` 读取；虽然本次返回值相同，
没有把 RGB 内参复制给深度。实际数组与 `get_camera_resolutions()` 的结果一致。

| 输出/内参 | RGB（texture） | 深度（depth） | 来源 |
|---|---|---|---|
| 实际宽 × 高 | 1280 × 1024 | 1280 × 1024 | API + 实测数组形状 |
| `fx`（px） | 1728.8903839489603 | 1728.8903839489603 | API |
| `fy`（px） | 1728.8846096722514 | 1728.8846096722514 | API |
| `cx`（px） | 642.6250682011389 | 642.6250682011389 | API |
| `cy`（px） | 512.2956888349067 | 512.2956888349067 | API |
| `k1,k2,p1,p2,k3` | `[0,0,0,0,0]` | `[0,0,0,0,0]` | API 原字段；完整浮点输出见 `intrinsics.json` |
| 数组类型/布局 | `uint8`，H×W×3，SDK 为 BGR；PNG 为标准彩色文件，回读 BGR 与数组逐值一致 | `float32`，H×W；无整型量化或显示色图替代 | API + 实测 |

- **校正/裁剪状态**：脚本未裁剪、缩放、去畸变或重投影；按 SDK 输出原样保存。
  官方 `Intrinsics2DCamera` 将接口描述为针孔模型和畸变参数，但本机接口未明确提供
  图像是否已完成镜头校正的标志；零系数不能证明物理镜头没有畸变。
  原生配置另有 `EnableDistortionCorrection=false` / `DistortionCorrection=0`，
  当前活动参数枚举未公开对应 getter；不据此替代图像校正状态的确认，也不重复校正。
  `Scan3DBinningEnable=false` 为原生配置返回，未自行改动。
- **深度类型、单位/比例、无效值**：API `get_point_cloud_unit()` 返回
  `CoordinateUnit_Millimeter`（枚举原值 `0`）；对应原生 API 文档说明深度为相机参考系的 Z，
  不是径向距离；`float32` 的每个数值按 API 声明以 mm 表达，换算 m 使用比例 `0.001`。
  无效值为 `NaN`。本帧实测没有零值、负有限值或无穷值，不能将零值当作本机已验证的无效编码。
  **尚缺实物已知距离/尺寸检查，因此物理深度标度仍未独立确认。**
- **XYZ 单位、参考系与轴方向**：SDK 组织化 `float32` XYZ，API 声明单位为 mm，
  原始相机光学参考系；非机器人基座系、非 CAD 外壳系。原生配置
  `GenicamScan3dCoordinateTransformEnable=false`，API `CoordinateTransformation` 原值为
  `[0,0,0,0,0,0]`。本帧数据与 `X=(u-cx)Z/fx`、`Y=(v-cy)Z/fy` 一致，
  因而输出的 +X 对应图像向右、+Y 对应图像向下，+Z 为远离相机的光轴方向
  （官方相机参考系说明 + API 数据内部检查）。光学原点到外壳/基座的实物位置未测。
  深度数组与 XYZ 的 Z 逐值相等（含无效掩码；实测最大有限差 `0 mm`）。
  对 `10000` 个有效像素的针孔投影内部核对，最大残差 `0.00008631674575031098 px`
  （实测）；这只检验 SDK 数据与其内参自洽，不是物理精度或标定误差。
- **RGB 与深度配准关系**：API 返回的 `depth_to_texture` 为单位旋转矩阵、零平移，
  两路内参与分辨率相同。彩色点云由 `Frame2DAnd3D.save_textured_point_cloud()` 导出，
  不由脚本自行按索引拼色或二次配准。对应版本文档保证该联合帧的彩色点云为逐像素配准的
  XYZ/BGR 数据；不能把该保证外推为任意其他采集模式或原始传感器图像已对齐。
  配准的独立标靶误差未测，精确曝光同时性也未测。

本段依据：原生 API 2.6.0 的
[Frame3D](https://docs.mech-mind.net/api-reference/eye-api-camera-cpp/2.6.0/classmmind_1_1eye_1_1_frame3_d.html)、
[CameraIntrinsics](https://docs.mech-mind.net/api-reference/eye-api-camera-cpp/2.6.0/structmmind_1_1eye_1_1_camera_intrinsics.html)、
[Intrinsics2DCamera](https://docs.mech-mind.net/api-reference/eye-api-camera-cpp/2.6.0/structmmind_1_1eye_1_1_intrinsics2_d_camera.html)、
[Frame2DAnd3D](https://docs.mech-mind.net/api-reference/eye-api-camera-cpp/2.6.0/classmmind_1_1eye_1_1_frame2_d_and3_d.html)，
以及[官方相机参考系说明](https://docs.mech-mind.net/en/3d-measurement-manual/latest/software-overview/basic-concepts.html)。
前述版本化 API 页面原始 HTML、绑定代码和采集日志在仓库外 `source-evidence/` 保存；索引为 `sources.json`。

### 4.3 活动采集设置与原始导出

全部配置先经 `user_set_manager().save_to_file()` 导出，再读取参数和采集；
之后再次导出。两份原生配置的字节 SHA256 相同，逐参数读取结果也完全相同。

下表的值均为 **API 当前值**；曝光时间的 ms 和增益 dB 来自 API 2.6.0
[Scanning2D](https://docs.mech-mind.net/api-reference/eye-api-camera-cpp/2.6.0/_scanning2_d_8h_source.html) /
[Scanning3D](https://docs.mech-mind.net/api-reference/eye-api-camera-cpp/2.6.0/_scanning3_d_8h_source.html) 官方定义。

| 选项 | 当前值与适用范围 |
|---|---|
| 2D 曝光 | `Scan2DExposureMode=Timed`；`Scan2DExposureTime=30.0 ms` |
| 2D HDR | 当前未启用；保留的序列 `[30.0] ms`，getter 可读，当前模式不可写；不当作正在 HDR 采集 |
| 2D 自动曝光 ROI | `(upper_left_x,upper_left_y,width,height)=(0,0,0,0)`；这是自动曝光 ROI，不能当作 RGB 裁剪 ROI |
| 2D 其他 | `Scan2DSharpenFactor=0.0`；`Scan2DToneMappingEnable=false`；期望灰度 `100` 为保留值，Timed 下不生效 |
| 3D 曝光 | `Scan3DExposureCount=1`，`Scan3DExposureSequence=[7.5] ms`，本次未采用多曝光；`Scan3DGain=0.0 dB` |
| 3D 深度范围 | `DepthRange={min:200,max:4000} mm`；这是当前配置，不是该目录型号推荐工作距离 |
| 3D ROI | `(upper_left_x,upper_left_y,width,height)=(0,0,0,0)`；API 描述为未设置 ROI；数组保留完整尺寸和 NaN |
| 投影 | `ProjectorFringeCodingMode=Fast`；`ProjectorPowerLevel=High`；`AntiFlickerMode=Off` |
| 点云处理 | `PointCloudNoiseRemoval=Weak`；`PointCloudOutlierRemoval=Weak`；`PointCloudSurfaceSmoothing=Off`；`PointCloudEdgePreservation=Normal` |
| 补洞 | `PointCloudGapFilling=Normal` 可读、当前不可写；本机描述限定于 Translucent 模式，不能据保留值认定 Fast 模式正在补洞 |
| 阈值 | `FringeContrastThreshold=3`；`FringeMinThreshold=1` |
| 未公开/不支持项 | 当前 getter 枚举之外的独立帧率、传感器曝光时间戳、镜头校正状态未获取；不能仅因未枚举就断言硬件不支持。完整可见参数、类型、读写性和原文说明见导出 |

“原始”在此指**未经本脚本二次处理的 SDK 输出**；相机已有去噪/离群点设置仍保留，
不宣称取得了未滤波的传感器原始条纹数据。

原始证据根目录（主机 `ju-ROG`，均在 Git 仓库之外）：

```text
/home/ju/Downloads/mechmind-camera/checklist-20261121-round1/
```

下表路径均相对该根目录；哈希为实测 SHA256。
`MANIFEST.sha256` 还列出全部官方页面、绑定源代码、网络信息和日志的校验和。

| 文件 | SHA256 |
|---|---|
| `MANIFEST.sha256` | `72974b365fec1fb5fb30f48c6f15aa9d7f8eeb6597cb9c78cfc7ea13508987b4` |
| `desk-connectivity/user_sets_before.json` | `4b4b0d567066a67ea528a83987d3d5d5191141da79e8dd560fddbb6f8c8067c5` |
| `desk-connectivity/user_sets_after.json` | `4b4b0d567066a67ea528a83987d3d5d5191141da79e8dd560fddbb6f8c8067c5` |
| `desk-connectivity/active_parameters_before.json` | `84b7ccd1344a1a2553372e2a85bfd03ec56c0de0da3e5d5b44023b6f1f3cecff` |
| `desk-connectivity/active_parameters_after.json` | `84b7ccd1344a1a2553372e2a85bfd03ec56c0de0da3e5d5b44023b6f1f3cecff` |
| `desk-connectivity/intrinsics.json` | `946cb7e2085d646907a49b399efea16e6e706dc1f0b7346845b5e608bec66d21` |
| `desk-connectivity/intrinsics_sdk_print.txt` | `3360611a38265868f901c249df834c95e3cc38ca31874489c5704f397a82f888` |
| `desk-connectivity/rgb.png` | `506e8b54d364cd90d95f46713160a11e957ba7d5d89489e470d342843b0ac900` |
| `desk-connectivity/rgb_bgr.npy` | `6a7659df21c2309e27a15a2c298fec47280ef3d13ab0229e85cfb050e70f45db` |
| `desk-connectivity/depth.npy` | `52223e52dde880b4d14cc8374f15812b8ca7d34b154f04c526e67694e091d85d` |
| `desk-connectivity/xyz.npy` | `b9814d84fa771180cb3ce5b9cb4fce0d6965bf4592c9d02670420d5804aad293` |
| `desk-connectivity/xyz_organized.ply` | `397da4aa7ad8d9752e328d09ede14df8968cf77631840f82b1bec297ff435818` |
| `desk-connectivity/xyzrgb_organized.ply` | `839fedf9ba9a844e5b302400a930c47ef349209a126fe6d8e319d364c759cf9d` |
| `desk-connectivity/metadata.json` | `a838368e40679c95bba5a80cfc60af3b7bab1fa37f622c9d569e2e640765c9e1` |
| `desk-connectivity/manifest.json` | `3099631c301b204b1b13d98bd747263ff6785636fd8450f4c6cd77a93e5cd082` |
| `sources.json` | `0fb92f15aa17d39dfc71fe1e2f5486f2fd2b3e551be253f9d8baaeb77a0b89db` |
| `validation.json` | `b6f2973385ca91cd2f777f3046b7543757e46798d8a9db99228e7688ac8d6621` |

### 4.4 样本、缺失项与验证边界

- **空板样本目录、文件清单/校验和**：未获取；用户确认现场暂不具备条件。
  未创建伪样本目录，也没有以办公桌代替砧板。
- **食材样本目录、文件清单/校验和**：未获取；同上，现场没有按要求布置的静止食材样本。
- **额外连通性样本**：`desk-connectivity/` 为办公桌/显示器场景，右侧有人体入镜；
  未确认相机机械固定或全场景静止，不计为两组最低样本，也不用来估计静态重复性。
  `.npy` 保留完整像素数组；两个组织化 PLY 均保存 `1310720` 个位置，含无效点，没有下采样
  （API 导出 + 实测文件行数）。
- **各次时间戳、耗时、触发**：本轮仅一次联合 `capture_2d_and_3d`，不是脚本分开触发。
  开始 `2026-11-21T20:37:16.851675+01:00`，返回 `2026-11-21T20:37:17.721354+01:00`
  （实测，宿主 Europe/Berlin 墙钟）；调用耗时 `0.869643763 s`
  （实测，`perf_counter_ns` 单调时钟）。这是一次采集耗时，**不是稳定帧率**。
  SDK RGB/depth `frame_id` 本次均为原值 `0`（API），不能据此证明曝光同步；
  传感器曝光时间戳/时钟同步状态未获取，本机帧接口未公开相应字段。
- **当前缺测**：深度有效正有限值 `598763`，`NaN=711957`，共 `1310720` 个像素，
  有效占比约 `45.682%`；有限深度 `944.7740478515625–2570.529296875 mm`
  （API 数据的实测统计，单位按 API 声明）。屏幕、桌面和右侧人体不代表食材工作面，
  未对缺测原因作归因。该样本不能用于声明推荐工作距离内的精度或湿表面性能。
  刀具遮挡食材未评估：没有砧板/食材布置，未移动机器人。
- **已知距离/尺寸单位核对**：未完成；没有用户实测尺寸、测距基准或标靶。
  目录的标称物距不能替代现场尺量，API 数值大小和上述内部投影检查也不能替代物理尺度验证。
- **安装照片、距离及测量起点**：未获取；未提供安装照片或现场尺量。
- **眼在手外标定状态**：本轮未做，也未获取已有经验证的外参。
  外参、板面及独立验证数据路径均未获取；没有生成 `T_base_camera_optical`。
- **本轮是否改变设备配置**：否。未调参、切换/覆盖参数组、升级固件、恢复出厂、
  运动机器人或运行 MPC/RL；前后配置原文及参数表一致。
  会话此前为排障曾临时切换 Fast/Accurate 并恢复，且配置过电脑网卡 IP；
  这些不当作本轮首次读取前完全未动设备的证据。本轮以本次导出为基线。
- **验证**：脚本编译、`--help`、真实设备只读导出和采集通过；PNG 无损回读、
  组织化点数、无效掩码、深度/XYZ 一致性、前后配置和全部采集文件哈希核对通过，见 `validation.json`。
  AGENTS 要求的 RTX 4090 根/后端完整测试未运行：本机没有项目 `.venv`，
  GPU7 直连 SSH 被拒绝，跳板机公钥认证失败；未将它们标作通过。
  本次没有修改仿真配置、物理后端、训练或机器人控制代码。

重采命令（使用一个**不存在且位于仓库外**的新目录；准备好真实场景后填写正确 `--scene`）：

```bash
/home/ju/Downloads/mechmind-camera/.venv/bin/python \
  scripts/diagnostics/check_mecheye_nano.py \
  --ip 192.168.122.101 --output /ABS/NEW/EXTERNAL/DIRECTORY --scene ACTUAL_SCENE
```

校验本轮外部原始资料：

```bash
cd /home/ju/Downloads/mechmind-camera/checklist-20261121-round1
sha256sum -c MANIFEST.sha256
```

## 5. 官方参考

- [Ubuntu Python API 使用说明](https://docs.mech-mind.net/en/eye-3d-camera/2.3.4/api/python-ubuntu.html)：
  实际调用以本机安装版本为准。
- [NANO V4 型号与工作距离](https://docs.mech-mind.net/en/eye-3d-camera/2.3.2/hardware/camera-models.html)。
- [眼在手外标定说明](https://docs.mech-mind.net/en/suite-software-manual/latest/vision-calibration/eth-auto-calib.html)：
  供后续选定标定流程时参考，本轮不启动自动机器人标定。
