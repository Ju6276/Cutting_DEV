# 运行：电脑、环境、命令与数据

所有当前方法都在main，不按算法切换旧worktree。
任务/算法见[方法](METHOD.md)，设备见[硬件](HARDWARE.md)。
大写路径/预算是占位符；运行目的、预算按[研究](RESEARCH.md)确定。
文档更新不改变运行中的源码/环境或重启训练。

## Machines

共7台电脑、3台训练服务器，设备IP不算电脑：

- **Mac**：本地编辑/Remote SSH、看结果；交付目录 /Users/judong/Documents/Cutting_DEV，
  Media放视频图片，Reports放报告图库。不是训练主仓库。
- **GPU7=tamsGPU7**：dong@134.100.13.217，1×RTX4090，主要开发/训练。
- **A6000=tamsGPU5**：dong@134.100.13.216，2×RTXA6000；这两个名字是同一机器。
- **GPU9=tamsGPU9**：dong@134.100.13.219，1×RTX4090，训练/评估。
- **tams61**：dong@100.70.97.92，SSH跳板；~/Projects/Cutting_DEV文档副本，不训练。
- **tamsCage**：dong@134.100.13.143，UR10e控制电脑，Ubuntu20.04.6/ROSNoetic。
  地址不是机器人控制箱IP。仓库 /homeL/dong/Cutting_DEV，
  原FDCC /homeL/dong/ur10e_fdcc_ws，候选 /homeL/dong/ur10e_fdcc_alignment_ws，
  ATI工具 /homeL/dong/ATI-Mini45-DEV，记录 /homeL/dong/ur10e_fdcc_records。
  上次盘点的硬件工作分支不代表始终同步main。
- **ju-ROG**：Nano采集电脑，Ubuntu22.04.5，相机网卡enp108s0为192.168.122.100/24，
  相机192.168.122.101；不是已验证的SSH网段。
  Python /home/ju/Downloads/mechmind-camera/.venv，采集目录同父级checklist-20261121-round1。

三服务器主仓库 /dataHDD/dong/Projects/Cutting_DEV。
GPU7通常负责推送，其他服务器既有配置以fetch为主；不再声称全项目只有GPU7可push，
现场电脑可有自己的用户配置。凭证不打印、不复制。
Mac别名cutting-gpu7/cutting-a6000经tams61；GPU9用
ssh -J dong@100.70.97.92 dong@134.100.13.219。
GPU7有a6000/gpu9别名；大文件服务器间直传，运行checkout保留启动源码。
A6000默认GPU1/sm86，4090默认GPU0/sm89；默认值不是预留，启动前检查占用。

## Environment

requirements.txt是主环境，两材料后端共享。
[依赖索引](../scripts/env/README.md)区分主环境、SAC/规划、渲染与CAD；
不因UR5e/UR10e另建重复环境。渲染/训练MuJoCo版本不同，不能混装。
ROS源网格/完整studio导出放GPU7；A6000缺ROS不妨碍已有平台训练。
上次记录Blender为GPU7/9 2.82、A6000 3.4.1未验收，不声称机器环境全相同。

~~~bash
source scripts/env/cutting_env.sh
# Select optional environment; return using CUTTING_ENV=base.
CUTTING_ENV=sac source scripts/env/cutting_env.sh
CUTTING_ENV=planner source scripts/env/cutting_env.sh
CUTTING_ENV=render source scripts/env/cutting_env.sh
~~~

长训必须在线W&B，project=sliceit-supermimic、
entity=ju-dong6276-technical-university-of-munich。
入口设置NETRC=/dataHDD/dong/.config/wandb/netrc；认证失败先解决，不改离线。
确认run页面和远端history可读。本地日志/checkpoint用于恢复，不替代在线记录。

## Scene preparation

当前是UR10e五食物；已有几何不重复导出CAD，只生成运行配置。
输出目录必须不存在。环境数从生成JSON求和，会覆盖--worlds。
默认3×5=15；如512×5=2560，应复制contract只改worlds_per_object，再重新生成配对配置。
热身下限worlds×9，30720可覆盖至3413环境。

~~~bash
source scripts/env/cutting_env.sh
python scripts/prepare_shared_scenes.py --contract configs/shared/ur10e_five_food_v1.json --output runs/NEW_SHARED/assets
python scripts/check_pose_depth.py --backend disect --shared-contract configs/shared/ur10e_five_food_v1.json --prepared runs/NEW_SHARED/assets/manifest.json --output runs/NEW_SHARED/disect_check.json --seconds 900
python scripts/check_pose_depth.py --backend blademaster --shared-contract configs/shared/ur10e_five_food_v1.json --prepared runs/NEW_SHARED/assets/manifest.json --output runs/NEW_SHARED/bm_check.json --seconds 900
~~~

已验证配置不因文档更新/换卡重做全部物理检查；容量变化做有上限容量测试，
物理/接口改变则验证受影响路径。不能只改版本号续训旧checkpoint。

## Training
```bash
source scripts/env/cutting_env.sh
python scripts/train_gpu.py \
  --conditions PREPARED/disect.json --output RUN/disect_training \
  --schedule transitions --warmup-transitions 30720 --transitions-per-update 640 \
  --wm-only-updates 500 --microbatch 512 --critic-microbatch 8192 \
  --critic-target real --actor-horizon 8 --actor-bc-alpha 0.25 \
  --max-transitions BUDGET_TRANSITIONS --max-hours BUDGET_HOURS \
  --checkpoint-every 32 --seed 0 --wandb-mode online --wandb-project sliceit-supermimic
```

PREPARED是上节生成目录。BladeMaster替换--conditions为--blademaster-config PREPARED/blademaster.json。
SAC使用scripts/train_sac_multi.py、CUTTING_ENV=sac、同一生成配置、显式预算及在线W&B；
具体选项以--help为准。裸train_gpu默认是source，不是上述当前real/H8/BC选项。

--max-transitions和--max-hours是本次启动追加预算，续训传剩余而非原总预算。
completed为达到交互预算，budget_complete为达到墙钟上限，不表示策略学好了。

## Resume

--resume RUN/last.pt配replay.pt；命名checkpoint配同stem.replay.pt，不拆开。
恢复模型/optimizer/计数、replay、学习额度、未开始轮与热身起点，不重新热身；
物理重新初始化，旧未完回合关闭。
--resume-without-replay显式放弃replay并重新填充，不是原样恢复。
部分完成optimizer轮的剩余工作记录为丢弃，不整轮重放；
未开始排队轮先执行再采集，旧文件未记录的pending不能恢复。

SIGUSR1在迭代边界保存；正常结束/SIGTERM写last.pt和progress.json。
原生崩溃可绕过Python清理，需核对进程退出状态。
显存保护不自动减环境数、不把OOM记通过。

## Evaluation

统一evaluate_pose_policy.py；配置需匹配checkpoint的机器人/工具/任务。
每世界预定首回合，不探索、不学习；超墙钟/未完不算完整评估。
位姿成功与材料分离分开报告。

~~~bash
CUTTING_ENV=planner source scripts/env/cutting_env.sh
python scripts/evaluate_pose_policy.py --method wm_policy --checkpoint V3.pt --conditions PREPARED/disect.json --output OUT_POLICY
python scripts/evaluate_pose_policy.py --method sac --checkpoint SAC/model.zip --conditions PREPARED/disect.json --output OUT_SAC
python scripts/evaluate_pose_policy.py --method wm_mppi --checkpoint V3.pt --conditions PREPARED/disect.json --output OUT_WM_MPPI
python scripts/evaluate_pose_policy.py --method analytic_mpc --conditions PREPARED/disect.json --output OUT_MPC
~~~

BladeMaster替换为--blademaster-config PREPARED/blademaster.json；输出目录分别指定。
解析MPC无需checkpoint。--record-geometry保留视频几何；
--worlds-per-object/--seed/--max-hours按评估计划给定。
纯MPC保留作工程联调，不是论文必需基线。

## Checks

根/后端分别运行，保持模块解析独立；用有余量的GPU，不中断训练：

~~~bash
CUTTING_ENV=planner source scripts/env/cutting_env.sh
CUTTING_BACKEND_ROOT="$PWD/backends/mujoco" PYTHONPATH=.:backends/mujoco python -m pytest tests -q
(CUTTING_ENV=base source scripts/env/cutting_env.sh; cd backends/mujoco; PYTHONPATH=. ../../.venv/bin/python -m pytest tests -q)
~~~

skip明确记录；单元测试、实际接口/重置、策略效果是不同证据。
不是每次长训都要跑遍全部诊断。

- check_pose_depth.py：任务方程/选中reset；--tool-reward-audit检查两种位置规则。
- check_board_continue.py：下压、贴板横移、抬刀、部分reset。
- diagnose_pose_predictions.py：V3冻结策略真实轨迹上的WM/critic诊断，统一解析终止。
- audit_pose_reward.py：DiSECt十参考轨迹，所有组、10 mm间隙、3 cm/s。
  检查指令、有限值和奖励，不是WM预测或BladeMaster轨迹审计。
- verify_entry/verify_resume：读取已有证据，不是另跑训练；
  resume前缀检查不支持环绕ring，pending=0不证明已验证恢复排队轮。

~~~bash
python scripts/diagnose_pose_predictions.py --checkpoint RUN/last.pt --config PREPARED/disect.json --worlds-per-object 2 --seed 0 --k 8 --output NEW_DIAGNOSTIC
python scripts/audit_pose_reward.py --configs PREPARED/disect.json --output NEW_AUDIT --wall-seconds 5400
python -m scripts.verify_entry RUN_DIR review --backends disect
python -m scripts.verify_resume PRE_DIR POST_DIR OUT_JSON
~~~

## Rendering

统一configs/render/studio_v1.json，使用记录的机器人/工具/食物身份，不逐视频替换。
半土豆、去皮香蕉；PLA纯白、ATI/刀金属，不加蓝刀、虚构杆或高亮预切缝。
扫描颜色映射是近似，不代表材料。CAD安装件已有，不再称“安装模型缺失”。

~~~bash
CUTTING_ENV=render source scripts/env/cutting_env.sh
python scripts/render_scene.py --evaluation RUN --group 0 --backend DiSECt --robot-platform ur10e --source-root /opt/ros/noetic/share --source-root /dataHDD/dong/Projects/SliceIt/catkin_ws/src --output RUN/video.mp4 --video
~~~

远景--view wide；BladeMaster用--backend BladeMaster。UR5e仅用于实际UR5e记录。

~~~bash
CUTTING_ENV=planner source scripts/env/cutting_env.sh
python scripts/compose_wrench_video.py --render RUN/video.mp4 --output RUN/video_wrench.mp4
~~~

曲线读sidecar和原轨迹：20 Hz端点滤波六轴力（N/N·m），不是500 Hz峰值或材料合力。
冻结policy重跑不冒充原训练实时视频。相机/点云算法见[视觉](methods/VISION.md)。

## Assets and records

GPU7主目录runs链接 /dataHDD/dong/Experiments/Cutting_DEV/runs；
A6000/GPU9主目录runs是 /dataHDD/dong/Projects/Cutting_DEV/runs 本地目录，
也有运行直接写Experiments。每个结果的完整路径为准，不套用另一台根目录。

GPU7 /dataHDD/dong/Assets/Cutting_DEV下：
- robots/ur10e_ati_ikea_fdcc_v2：UR10e/ATI/IKEA与FDCC配置。
- ur10e_shared_five_food_v1：配对运行配置；长训可有独立规模副本。
- ur10e_ikea_midblade_scenes_v1：DiSECt黄瓜/半土豆；
  ur10e_ikea_five_food_scenes_v1：新增三食品。
- blademaster_intact_20261114：黄瓜/半土豆完整几何；
  five_food_intact_v1：新增三食品，不是五种都在这里。
- cameras/mecheye_nano_v4：相机CAD/外观，不是实机外参。

Native BladeMaster源 /dataHDD/dong/Projects/BladeMaster-Warp-ur10e-ikea，
不替换旧直楔刀分支；版本/许可证见[来源](reference/SOURCES.md)。
食品扫描在 /dataHDD/dong/Research/food_meshes/；
[候选资产](reference/OMNIOBJECT3D_FOOD_CANDIDATES.md)含本机HTML图库，
候选网格不自动加入训练。

当前长训/验证目录按机器见[结果](RESULTS.md)。
旧任务checkpoint、配对replay、扫描和冻结checkout保留，
不因文档过时删除实验数据。旧目录索引可查Git历史，不重复作当前操作说明。
清理回执位于Experiments/Cutting_DEV/operations/cleanup_20261115/、
maintenance/和/dataHDD/dong/Archive/。

交付媒体归本机Documents/Cutting_DEV，不散放Downloads。
正文不贴长哈希，不要求用户回填；自动兼容性检查仍保留。
