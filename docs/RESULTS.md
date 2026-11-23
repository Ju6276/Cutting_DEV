
# 验证与实验结果

每项只代表记录的平台、任务、样本和日期，不把旧结果视作当前全场景保证。
这是固定实测记录，不是实时训练看板；运行状态需查进程/progress.json。
[方法](METHOD.md)维护规则，[运行](OPERATIONS.md)维护命令，[研究](RESEARCH.md)维护下一步。
详细历史开发过程保留在[原结果记录](https://github.com/Ju6276/Cutting_DEV/blob/7cc9d1d/docs/RESULTS.md)，
原始日志/数据不随文档合并删除。正文不要求人工回填哈希。

## Five-food visual localization (2026-11-23)

GPU9: /dataHDD/dong/Experiments/Cutting_DEV/runs/visual_center_five_foods_20261123/.
The user selected a fixed current Nano camera, fixed food heading and +/-15 mm
translations. Five known settled surfaces: cucumber003, half-potato, peeled
banana, tofu, luncheon. Dataset: 1500 ideal-depth cases, 1000/250/250
train/validation/test, grouped by placement ID across foods. Network input is
512 foreground XYZ points; board pixels support plane estimation but are
excluded from encoder input. No robot/tool is included in training depth.

DP3 XYZ encoder plus the existing midpoint regression head trained for 120
epochs on CUDA. Validation selected epoch 11; weights are at train/center.pt.
Online W&B reports finished and all 120 epoch rows are remotely readable:
[training run](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/j09wwv7c).

On the 250 held-out cases, midpoint-Y MAE / P95 are 0.169 / 0.395 mm.
Per-food MAE: cucumber 0.122, potato 0.227, banana 0.239, tofu 0.147,
luncheon 0.111 mm. On exactly these cases, historical frozen DP3 gives
12.237 / 15.967 mm; visible geometric midpoint gives 0.684 / 2.831 mm.
The geometric selector is better on banana, tofu and luncheon, while the new
network improves the aggregate mainly through cucumber and potato.
These are known-shape, fixed-camera, ideal-depth results, not real-camera,
novel-shape or cutting-success evidence. No test-driven retuning was performed.
Reports: train/report.json, comparison.json, train/test_predictions.npz.

The prior frozen old-DP3/current-actor visual execution at
visual_five_foods_20261122/evaluation_v3 completed cucumber, banana, tofu and
luncheon with pose success at 125/93/102/96 steps. Half-potato exhausted its
1200 s wall budget around step 320; that is neither success nor a 500-step
task timeout. Each completed case still reported one material component.
New visual weights have started a separate frozen-actor execution in
visual_center_five_foods_20261123/evaluation; inspect progress before claiming
it complete. RL weights, task and physical parameters are unchanged.

The data/inference contract changes passed GPU9 root tests (839 passed,
6 skipped) and backend tests (48 passed). Current execution is native
BladeMaster only; DiSECt still needs matching pre-cut topology for a newly
selected plane.

## UR10e long-run snapshot (2026-11-22)

Read from progress.json on the run hosts; these are fixed records, not live
status assertions. Run root: /dataHDD/dong/Experiments/Cutting_DEV/runs/.

- A6000 DiSECt: ur10e_disect_fivefood_20261121_resume_resetfix/train,
  budget_complete, 2560 worlds, 624640 cumulative transitions, 926 WM rounds,
  425 policy rounds; resume wall time 52793.48 s. It resumed the old run after
  the reset-order fix, so roughly 30 h combined is not 30 h continuously on
  repaired code. Remaining queued rounds: 2. This supports the measured
  configuration's run completion, not all GPU capacities.
- GPU7 BladeMaster: ur10e_blademaster_fivefood_20261121_30h/train,
  budget_complete, 320 worlds, 366720 transitions, 524 WM rounds, 24 policy
  rounds, 108037.11 s; one queued round remains.
- Both use 500 initial WM-only learning rounds. Unequal achieved data and policy
  updates prevent a matched-budget comparison; completed budgets do not prove
  converged policies. Paired model/replay files are retained.
- GPU9 SAC uses ur10e_sac_disect_fivefood_20261121_resetfix_30h/train;
  consult its progress and process for current status. Its SB3 gradient-update
  count is not the WM+BPTT learning-round unit.

Final held-out performance of these checkpoints has not been established by
this documentation update.

## Board continuation

当前project_continue_v2，两后端均验证压、保持、贴板滑动、抬刀和部分reset：
材料与传感器继续推进，抬刀后board_contact清零，无永久冻结。
真实10 mm横移按横移阶段起点重新统计为9.985–9.998 mm；
原9.0–9.7 mm报告混入下压漂移，已修。贴板reset标志按FK计算，未选中世界不变。
证据GPU7 Experiments/Cutting_DEV/runs/board_normal_20261119/audited/、
review_reset_contact_20261120/。这不是砧板反力或真机触板验收。

## UR10e platform integration evidence

UR10e/ATI/IKEA、刚度方向与板面修复已合入。
9世界学习流程分别达到WM20/actor2/critic40优化步，含选中reset和保存恢复；
这批未自然结束，后续五食品自然回合/长训提供了对应覆盖。
几何距离场10,000近表面点最大误差0.06352 mm。
FDCC柔顺参考系修正后，一个黄瓜参考案例从第80步仍未到位，
改善为第63步成功、间隙1.482 mm、到位计数11。不是策略对照结果。

A6000同卡9世界、两热身+三计时步：
UR5e平均5.494 s/action（材料2.570 s），UR10e9.354 s（材料6.064 s）。
这是整个平台/刀形比较，未隔离机器人惯性成本；没有配置“等待11秒”。
证据GPU7/A6000 /dataHDD/dong/Experiments/Cutting_DEV/runs/ur10e_platform_20261120/，
disect_learning_midblade9/report.json、bm_learning_midblade9/report.json、
new_knife_fdcc_v2/motion.json、paired_platform_timing_gpu0.json。

## Five-food physical checks

2026-11-21，15世界随机配置，两后端奖励/终止方程与部分reset隔离通过，
最大误差DiSECt1.4833e-8、BladeMaster1.5715e-8。
名义参考下压：每食品1世界、30 mm/s、起刀10 mm，观察120步但任务上限仍500。

- DiSECt黄瓜/半土豆/香蕉/豆腐/午餐肉在51/53/40/115/52步位姿成功；
  共9次自然结束/reset，并有reset后继续步，奖励误差≤3.71e-6。
- BladeMaster黄瓜/香蕉/豆腐/午餐肉50/39/67/51步成功；
  半土豆120步尚未结束，随后定位到下述共用去皮bug，修复后成功。
  当次8次结束/reset，奖励误差≤3.71e-6。
- 豆腐“穿地6.740 mm”原统计含空侧辅助节点，不是材料表面。
  GPU9另一次真实表面逐控制步检查最大穿透0.296627 mm，
  同次所有节点最大11.010555 mm，第141步成功。
  辅助节点参与FEM/切边插值，不可任意删除或激活接触。
  只覆盖50 ms边界，不代表内部子步也满足该界限。

证据GPU7/A6000 /dataHDD/dong/Experiments/Cutting_DEV/runs/ur10e_five_food_20261121/，
GPU9同根tofu_surface_20261121/report.json。
香蕉/豆腐/午餐肉预设来源与标定限制见方法/硬件；
名义参考成功不证明策略泛化或完全材料分离。

## Reset fixes

**FDCC去皮坐标，2026-11-21：**旧reset把基座重力当传感器系并漏重心力矩，
UR10e零传感器读数产生6.8768 N假向上力，预测静态滞后2.7523 mm，
实测2.7584 mm。改为传感器系重力+COM力矩，未减固定常数/删除工具重量。
修复后BladeMaster半土豆第50步成功，到位11、z误差1.8317 mm，
最终sensor Fz=-8.739 N，真实接触反馈仍在。旧UR5e零负载配置不产生同一偏差。
4项回归通过，旧代码其中2项失败。
证据GPU7 runs/fdcc_tare_20261121、blademaster_potato_tare_fixed_20261121。

**DiSECt重置顺序：**SAC曾在46,080交互时静置接触检查异常退出，
原日志无物体/world，不能恢复历史具体样本。
两世界豆腐复现：上一轮顶部65.154 mm，原始形状70.800 mm，
先按旧顶部放刀再恢复物体消耗5/6 mm间隙。修复为恢复物体后准备刀位。
正式代码两世界初始化/分批/全reset通过，未选中观测逐位一致，
最大起刀误差5.72e-8 m，原接触守卫仍启用。
BladeMaster独立五食品/两世界选中reset无刀接触，未选中粒子/观测不变，
起刀误差≤3.80e-7 m，未发现同一顺序缺陷。
证据GPU9 /dataHDD/dong/Experiments/Cutting_DEV/runs/sac_reset_diagnosis_20261121/。
没有改奖励、力限、任务或checkpoint间隔。

## UR10e baseline verification

The following are recorded checks, not newly rerun by documentation cleanup.

- Five-food interfaces (2026-11-21, GPU9): each backend collected 40 parallel
  steps / 200 transitions and fit a small matching WM for two updates, followed
  by two physical steps of each planner (8 samples, horizon 8). SAC ran eight
  vector steps/updates and exact saved-weight reload. DiSECt 814.9 s;
  BladeMaster 588.3 s. The separate production SAC smoke saved/reloaded
  model/replay after 40 transitions and six optimizer updates.
  Partial resets [0,2,4] then [1,3] preserved unselected frame fields exactly.
  Evidence: runs/ur10e_baselines_20261121 on GPU9, *_five_adapters/report.json,
  *_sac_cli_five, sac_cli_restore_verification.json, *_reset_groups.json.
- Post-tare-fix CUDA SAC and learned WM+MPPI (A6000): five foods per backend,
  two actual steps; frozen WM unchanged, candidates/model/actions on CUDA,
  selected reset isolated. SAC completed two CUDA optimizations and exact
  model/replay reload. DiSECt 269.5 s, BladeMaster 116.6 s.
  Evidence: /dataHDD/dong/Experiments/Cutting_DEV/runs/baseline_gpu_audit_20261121/,
  probe.py and {disect_cuda,blademaster_cuda}/report.json.
- Independent code review traced SAC terminal_observation-before-reset,
  task timeouts as terminated, transition-credit/checkpoint pairing,
  per-world MPPI goal/sequence isolation, analytic absorption paying the terminal
  reward once, and optional critic only on surviving states. No new confirmed
  SAC/learned-MPPI implementation bug was found. Learning WM never constructs
  AnalyticFDCCModel; SAC steps the actual environment. Neither uses the faulty
  surrogate retare path. Finite-history WM still approximates hidden controller/
  filter state; this is not an accuracy guarantee for new poses or loads.
- Analytic MPC: physical per-world tare is now reused across candidates/horizon.
  Old-code regression gave up to 1.614 N false force; fixed error <1e-12.
  Two-world physical probes on both backends passed plan/step/observe/partial
  reset/replan (67.9/52.8 s). Evidence:
  A6000 runs/mpc_tare_fix_20261121/{disect,blademaster}_v2/;
  GPU9 runs/analytic_mpc_tare_fix_20261121/. Here '_v2' is a probe-directory
  suffix, not pose_depth_v2.
- Earlier full-episode analytic MPC at 6258561 scored DiSECt 3/5 and
  BladeMaster 4/5 pose completions. Those runs predate the MPC tare fix; they
  are not a post-fix result or a paper benchmark.
  GPU9 runs/ur10e_analytic_mpc_fivefood_20261121/{disect,blademaster}/summary.json.
- Eight-candidate five-food planning measured median 0.205–0.215 s/world
  for WM+MPPI and 0.489–0.511 s for analytic MPC. This exceeds the 50 ms
  task interval; no real-time claim is made.


## Vision (2026-11-17)

- Prescribed-plane depth estimator: visible-top errors 0.34/0.14/0.03 mm
  (cucumber/half potato/prism), ideal single-frame depth. GPU7 `runs/vision_route1_20261117`.
- Visible-major-axis selector on seven OmniObject3D scans: five pass, potato003
  and apple002 reject as ambiguous. GPU7 `runs/vision_variants_20261117`.
- Fixed-heading midpoint learning (900 cases, 600/150/150 split by placement):
  held-out MAE learned 0.210 mm, visible-extent midpoint 5.499 mm, constant
  coordinate 7.685 mm, visible centroid 15.129 mm. Known shapes only.
  GPU7 `runs/visual_center_20261117`.
- Official DP3 encoder vs local network, three paired seeds: mean MAE
  0.2013 -> 0.1370 mm; bootstrap mean difference -0.0643 mm, 95% [-0.0872, -0.0433]
  (reused development test set). GPU7 `runs/dp3_center_20261117`.
- Physical handoffs (native BladeMaster nominal cucumber, frozen DiSECt policy66
  transferred across backends, 180 steps, v2): geometric, learned and DP3
  selectors with RL all timed out (final height error 2.5–2.9 mm vs 2 mm
  tolerance); paired WM566+MPPI timed out (only 5 qualifying samples); nonlearned
  FDCC+MPPI reached pose success at step 105. Planning took ~0.46 s per step, above
  the 50 ms period. Single illustrations, not a ranking. GPU9
  `runs/vision_closedloop_20261117`, `runs/visual_center_closedloop_20261117`,
  `runs/dp3_controllers_20261117`.

## Hardware checks

**tamsCage，2026-11-21，只读盘点：**原FZI ROS1 fdcc状态initialized，不是running；
target_frame和ft_sensor_wrench无发布者。3秒内关节1501条、UR /wrench 1500条、
FDCC力输入0条；UR wrench frame为tool0_controller。这是反馈频率，不是闭环验收。
原始证据 /homeL/dong/ur10e_fdcc_records/codex_readonly_20261121/。

**独立候选：**保留机械臂描述，添加九个工具链接，URDF检查与同源控制器编译通过；
YAML对应仿真配置、动作映射离线检查通过，未加载候选到活动UR驱动。
原FZI使用tool0三参考点、不同力位组合/增益，不视为已对齐。
发现候选dynamic_reconfigure将trans_x裁到5000 N/m；对齐仿真范围后，
真实C++ __clamp__ 保留10000。只修接口范围，不是允许实机用最大刚度。

**输出对照：**真实C++控制器和IK solver通过内存关节handle，
在隔离ROS master处理30次连续变化输入，同步初始去皮并保留递推状态。
与Python FDCC关节输出最大差1.3323e-15 rad（阈值1e-10）。
不覆盖ATI、网络、实机伺服、动态参数同步和硬件饱和。
证据 /homeL/dong/ur10e_fdcc_records/alignment_candidate_20261121/，
文件 controller_output_comparison.json、action_mapping_check.json、
range_before.log、range_after.log、controller_range.patch、候选URDF/YAML。
当前接入和标定见[硬件](HARDWARE.md)。

**Nano，ju-ROG，2026-11-21：**设备/内参/配置读取和一次办公桌联合RGB-D/XYZ采集通过；
采集前后配置相同。1280×1024，保留1,310,720点位（含NaN），未下采样。
本帧有效598,763个（45.682%），深度约944.77–2570.53 mm；
一次约0.870 s，不是稳定帧率或推荐物距内精度试验。
深度与XYZ的Z一致，10,000像素内部投影最大残差约8.63e-5 px。
空板/食材、已知尺寸/距离及眼在手外标定未做。
证据 /home/ju/Downloads/mechmind-camera/checklist-20261121-round1/；
desk-connectivity下rgb.png、depth.npy、xyz.npy、intrinsics.json、
active_parameters_before.json、metadata.json及根目录validation.json。
原始数据保留，不要求用户手工核对哈希。
