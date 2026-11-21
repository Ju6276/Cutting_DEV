# SAC 与 WM+MPPI 仿真路径审阅（2026-11-21）

## 实际 GPU 短检查：两后端均通过

2026-11-21 在 A6000 两张独立显卡上，以当前 main 和 UR10e 五食材配置执行了真实仿真短检查：

- **DiSECt（GPU0）：通过，269.5 秒。**
- **BladeMaster（GPU1）：通过，116.6 秒。**

每个后端均包含黄瓜、半颗土豆、香蕉、豆腐和午餐肉各一个实际物理环境。WM+MPPI 使用匹配的已有 WM 检查点，8 个候选、2 步规划时域，执行两次真实物理 step；模型、规划状态和动作均断言为 CUDA，冻结 WM 权重未变。中途只重置一个环境，其他环境的状态、观测、末次受力、接触标志及规划序列精确不变，随后继续规划。

SAC 每个后端执行两次真实批量 step（10 次交互），完成两次 CUDA 优化；actor、critic、梯度与训练采样 batch 都在 CUDA，确认权重实际更新。保存模型及 replay 后在 CUDA 重新加载，权重逐值一致、恢复的 replay 采样仍在 CUDA。SB3 的 replay 存储使用 CPU NumPy，是其正常数据存储设计，不等于优化在 CPU 上运行。

通过标准来自上述内容断言及逐项 report，不只是进程退出码。此次没有新增算法修复，没有重新采集 40 步 WM 数据，也没有运行纯 MPC。范围是当前真实 GPU 接口与学习/规划执行流程的短验收，不是整回合成功率或长期稳定性评估。两项验收均已结束释放显卡；现有训练未被停止或修改。

A6000 证据目录：
`/dataHDD/dong/Experiments/Cutting_DEV/runs/baseline_gpu_audit_20261121/`

- `probe.py`：实际验收脚本。
- `disect_cuda/report.json`、`disect_cuda.log`：DiSECt 逐项结果。
- `blademaster_cuda/report.json`、`blademaster_cuda.log`：BladeMaster 逐项结果。
- 两个输出目录各保存本轮 SAC 模型与 replay；report 同时记录物理 GPU 映射、CUDA device 和冻结权重检查结果。

最初一次直接调用 Python 缺少统一运行环境的 TORCH_EXTENSIONS_DIR，在物理构造之前退出；改为 source 仓库正式环境入口后完成以上检查。旧的 `disect/report.json` 是这次启动配置失误的记录，不能与最终 `disect_cuda/report.json` 混用。


## 去皮修复后的独立代码复核

复核对象为已包含解析 MPC 去皮修复的当前代码（内部基准 d7392ff）。本节是逐段读代码的结论，不是把上面的两步 CUDA probe 扩大解释成完整算法审阅。本轮未确认新的 SAC 或学习 WM+MPPI 实现错误，没有因此修改算法代码，也没有重复运行全套测试。

### 坐标、控制器状态与去皮基准

- `shared_baselines.py:190–202` 的学习分支加载 `CuttingWorldModel` 和可选 critic，冻结参数；解析 FDCC 模型只在另一分支构造。`world_model.py:63–68,128–129` 的学习预测路径依次执行动作映射、神经动力学预测、物理字段积分和解析任务派生，不构造 FDCCController，也不在预测姿态上调用它的 reset。这是学习 WM+MPPI 不存在本次“重新去皮”错误的调用链依据，不只是输入字段没有变化。
- `sac_vector.py:63–85` 将动作交给真实物理环境，随后记录观测和奖励；SAC 不运行解析 MPC 的预测控制器。物理 FDCC 沿既有 `update` 路径推进，只有物理 reset 时更新初始状态。当前 DiSECt 控制入口见 `environment.py:309–319`，传感器观测选择见 `coupling.py:77–81`；没有在 SAC 适配器里额外旋转受力或再次去皮。
- 限制：当前学习 WM 使用可观测历史与解析历史字段，并未显式输入全部 FDCC 递推量、滤波器内部状态或去皮基准。它是有限历史预测模型，不能由上述调用链保证任意初始姿态和未知负载下的准确性。本轮没有把这种建模近似改成另一种状态设计，也没有把解析 MPC 的“固定未来受力”假设误套到学习 WM 上。

### 部分 reset、终止与 bootstrap

- `sac_vector.py:69–90` 先用最后真实观测和当时目标生成 terminal_observation，再仅重置结束环境的历史与回合统计；非结束环境历史不清空。`TimeLimit.truncated=False` 对应本项目有限时域任务的真正终止，交给 SB3 后终止样本不 bootstrap。当前 500 步时限不是应保留 bootstrap 的外部截断。
- `control_state.py:78–105` 将上一状态的历史向前移动；`reset_batch_history` 只替换 mask 为真的环境。`shared_baselines.py:212–227,257–292` 每个 world 各有 rollout、目标、规划序列和步数；只有模型权重共享。部分 reset 清选中 world 的 U 和步数，不重新加载或改动冻结 WM 权重。
- `mpc.py:69–97` 使用转移前的存活权重支付当前奖励，并用当前解析终止结果吸收后续回报。当前 V3 的 learned termination logits 不参与存活权重；`mpc.py:103–118` 的可选 critic 只在存活状态上求值，并使用 horizon 折扣。critic 的位姿误差方向与 `trainer.py:377–379` 相同。

### 目标、候选与续训

- `shared_baselines.py:181–202` 校验任务折扣与检查点接口；`shared_baselines.py:260–274` 逐 world 绑定目标位姿和速度，候选动作经 `ModelRollout.task_action` 映射后再执行。未发现候选共用可变目标或跨 world 继承规划序列的路径。
- `scripts/train_sac_multi.py:79–94` 恢复模型、优化器和 replay，并检查历史长度、更新频率、环境数及接口一致性。`sac_vector.py:128–143` 保留并随模型保存剩余交互信用；`scripts/train_sac_multi.py:115–122` 在上一轮数据插入和更新完成之后保存，避免在 SB3 插入当前转移之前保存一半状态。
- `baseline_checkpoint.py:18–44` 先写完 payload、最后替换 manifest，读取时检查配对摘要；它能拒绝保存中断留下的混合文件，不保证自动恢复上一代检查点。SAC 续训明确重新开始物理回合，保留的 replay 是独立一步转移加观测历史，不声称逐位恢复现场。学习 WM+MPPI 加载冻结检查点进行评估，规划序列重建，不属于持续训练的 checkpoint 恢复。
- `evaluate_pose_policy.py:226–299` 只累计仍 active 的首个回合，记录完成后再 reset；已完成 world 不重复记入成功率。冻结模型前后检查和提前预算耗尽的失败记录仍保留。

这些审阅没有发现需要新增复现的具体错误场景，因此没有以“再跑更多测试”替代结论。已有两后端 CUDA 证据支持短执行路径；长时域模型准确性、成功率、实时性和长期稳定性仍应由各自实验结果回答。

## 结论与范围

本次独立阅读当前 main 的 SAC 多环境训练/评估和冻结 WM+MPPI 实现，未确认新的算法实现错误，因此没有修改奖励、动作、终止规则或训练代码。这个结论只覆盖下列路径，不代表没有任何 bug，也不是对真机或切割成功率的认证。

解析 MPC 的工具去皮基准错误已在之前修复；它位于 AnalyticFDCCModel，不等于 SAC 或冻结学习 WM 使用同一错误模型。仅检查该上下文没有进入网络输入不足以排除其他错误；后续独立代码审阅沿实际执行调用链核对，见上面的独立代码复核。

内部复核基准：main b7c8e5f；GPU9 空闲开发 checkout Cutting_DEV_reset_settling。只运行 CPU 定向测试，没有变更或暂停 GPU9/GPU7 的现有训练，没有操作机器人。

## 已逐项检查

- SAC：CuttingVecEnv 的 10 帧观测加 8 维目标；末步观测在部分 reset 前复制进 terminal_observation；仅重置选中的历史、计数和回报；500 步任务的 timeout 仍是 terminal，TimeLimit.truncated 为 false。
- SAC：TransitionSAC 根据真实交互数分配更新次数，保留不满一轮的 transition_credit；保存/续训恢复模型和 replay，检查任务与机器人身份、历史长度、更新频率及环境数；配对文件有哈希检查；续训恢复的是学习状态，物理环境重新 reset，不声称逐位恢复仿真。
- SAC：当前声明动作范围为六维 [-1,1]，物理和 WM 调用同一 SliceIt action mapper。前四项负值映射到参数下限是现行 mapper 的规则，本次未当作 SAC 的新增错误去改。
- WM+MPPI：冻结检查点和可选 critic 的权重加载、元数据验证、动作归一化转换；每个环境有独立 MPPI 候选序列、目标、历史与步数；部分 reset 只清对应状态和计划。
- WM+MPPI：奖励使用转移前存活权重，解析终止当步结算一次，后续权重归零；当前 V3 不让学习终止头削减规划回报；terminal critic 只计算存活状态，并应用 horizon 折扣。
- 当前 V3 的 board_contact/last_wrench 通过状态规格打包；当前 project_continue_v2 不按历史锁存机制冻结；历史版本使用自己的规则。
- 共用评估入口：先记录完成回合，再对结束环境 reset；已完成环境不重复计入结果；最终检查模型权重未变；max-hours 提前耗尽不是 completed。

主要代码入口：
[VecEnv 与更新次数](../../cutting_world_model/sac_vector.py)、
[SAC 训练](../../scripts/train_sac_multi.py)、
[检查点配对](../../cutting_world_model/baseline_checkpoint.py)、
[共享规划器](../../cutting_world_model/shared_baselines.py)、
[MPPI rollout](../../cutting_world_model/mpc.py)、
[状态历史](../../cutting_world_model/control_state.py)、
[当前 V3](../../cutting_world_model/pose_depth.py)、
[共同评估](../../scripts/evaluate_pose_policy.py)。

## 实际验证

CPU 测试环境使用现有 .venv-sac，CUDA_VISIBLE_DEVICES 置空、OMP_NUM_THREADS=1，PYTHONPATH=.:backends/mujoco。

第一组 **36 passed**（2.34 秒）：

```sh
python -m pytest -q tests/test_shared_baselines.py tests/test_mpc.py \
  tests/test_sac_refusal.py tests/test_sac_evaluation_dispatch.py \
  tests/test_analytic_mpc_tare.py
```

包括真实 SB3 的小网络更新与保存恢复、安装的 pytorch_mppi 规划器、合成物理环境和小型机器人/WM fixture；不应称为五食材两后端完整真实仿真重新验收。

另一次 **19 passed**（2.48 秒）：

```sh
python -m pytest -q \
  tests/test_board_continue.py::test_resting_state_with_zero_predicted_motion_reports_contact \
  tests/test_pose_depth_current.py tests/test_baseline_probe_groups.py \
  tests/test_gpu_control_history.py
```

额外隔离实验：在同一当前 V3 fixture frame 中加入值全为 12345 的 controller_tare_compensation，比较加入前后：
SAC 展平观测、WM pack_batch 状态逐值相同，frame_metadata 完全相同，状态 base_fields 不包含该字段。实验通过。这是执行上下文，不是新增网络输入。

## 尚未查清的运行环境异常

扩展组合以下命令在 GPU9 的 CPU 上两次发生进程级 Illegal instruction，不能记为通过：

```sh
python -m pytest -q tests/test_board_continue.py tests/test_pose_depth_current.py \
  tests/test_baseline_probe_groups.py tests/test_gpu_control_history.py
```

两次都定位到 kinematics.py 的 quat_exp_second_order 中 torch.cos 调用，经 joints_for/FK 的 Jacobian fixture 进入；第一次在 resting-state 检查，第二次在 physics/WM contact-flag 一致性检查。并非 pytest 断言失败。首次失败的测试独立重跑通过（见上面的 19 项）。本次没有证据把它定性成 SAC/MPPI 的 Python 逻辑错误，也没有证据宣布硬件/依赖完全正常。若要扩大 CPU 运行部署范围，需要隔离该原生异常；禁止把这组写成完整通过。

## 仍未由本轮证明的内容

- 当前已训练检查点在五食材上的成功率、WM 长 horizon 的模型误差、规划延迟。
- GPU 大批量连续训练、真实后端反复 reset 的长期稳定性；本轮不重复已有长训。
- 真机 ATI 输入、坐标标定、控制器法则/频率及动作映射的端到端一致性。本轮没有机器人运动。
- 解析 MPC 固定未来受力的近似依然存在；这是解析模型的假设，不能冒称 WM+MPPI 的预测方式。

## CPU 原生异常的有限复核（同日补充）

进一步把问题从 pytest 组合中拆出：直接重复当前 V3 fixture 的 FK 目标求解（5 个 clearance，15 轮），不启动 SAC/MPPI、不启动物理后端、不用 GPU。设置 TORCH_COMPILE_DISABLE=1，并把 torch.compile 替换为一旦调用就抛错的函数。该 eager 循环第 6 轮仍发生 SIGILL，Python 栈仍在 quat_exp_second_order 的 torch.cos。这排除了“必须由本项目 torch.compile/inductor 编译触发”的解释。

在 gdb 下重复同一小循环（上限 30 轮），第 3 轮捕获 SIGSEGV：
native PC 位于 PyTorch 的 libtorch_cpu.so 中 **VMLSETMODE_**（Intel MKL VML），指令是 mov (%rdi),%edi，后续 native 栈异常。另一轮仅 17 个 board_continue 测试在 gdb 下全部通过；说明问题并非每次都会出现。

环境为 Intel Core i9-14900K、Linux 5.15.0-179-lowlatency、PyTorch 2.4.1+cu121、MKL 2022.2；PyTorch 实际报告 AVX2，CPU flags 包含 AVX2。仅凭这些信息不能认定是 ISA 配错，也不能区分库二进制/ABI 与硬件稳定性等原因。单独 torch.cos 的 100 万次三元素调用，以及各 10 万次标量前向/求导均通过，不能用它们覆盖 FK 循环的故障。

本轮最终定位是 **尚未解决的 CPU 原生运行故障**，不是已确认 SAC/WM+MPPI 算法公式错误，也不构成“代码都没问题”的结论。没有修改算法、替换数学函数或安装依赖；到此停止扩大排查。

证据保存于 GPU9：

`/dataHDD/dong/Experiments/Cutting_DEV/runs/sac_wm_mppi_audit_20261121/`

- cutting_cpu_sigill_repro_20261121.py：最小 eager FK 循环。
- cutting_cpu_sigill_minimal_gdb_20261121.log：gdb 捕获 VMLSETMODE_ 的 SIGSEGV。
- cutting_cpu_sigill_audit_20261121.log：gdb 下 17 项通过的对照。
- cpu.txt、torch_environment.txt：CPU 与 PyTorch/MKL 编译及调度信息。

复现应在开发 checkout 根目录执行，并使用当前 .venv-sac Python：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:backends/mujoco:tests OMP_NUM_THREADS=1 \
TORCH_COMPILE_DISABLE=1 timeout 60 /dataHDD/dong/Projects/Cutting_DEV/.venv-sac/bin/python \
-X faulthandler /dataHDD/dong/Experiments/Cutting_DEV/runs/sac_wm_mppi_audit_20261121/cutting_cpu_sigill_repro_20261121.py
```

## 本轮后续限定：CPU 专项不是当前 GPU 训练/规划验收的阻塞项

用户随后明确当前算法计算使用 GPU，不要求把本次 CPU 专项扩大为新的验收工程。因此停止进一步 CPU 实验，未更换训练依赖，未新增 CPU 环境入口，未保留 MKL 强制派发默认值。

只读核对实际设备路径：
- scripts/train_sac_multi.py 新建和续训 SB3 模型均明确 device='cuda'。VecEnv 会把观测转为 NumPy/replay 主机数据，但 actor/critic 的优化由 SB3 CUDA 模型执行；这条路径不调用测试中的 at_gap/joints_for。
- SharedMPPIPolicy 将学习 WM/critic 放到 env.device；MPPI 候选、控制序列和状态在同一设备。当前仿真 env.device 是 CUDA。
- 解析 MPC 在同一共享入口显式把 env.device 传给 AnalyticFDCCModel；其状态规格、SerialChain 和控制器预测也随该设备运行。
- 出错复现的 at_gap 位于 tests/test_board_continue.py，joints_for 位于 tests/test_pose_depth_v3.py，均是为边界断言构造姿态的测试 fixture。生产 cutting_world_model/ 和 scripts/ 没有调用它们。生产代码也使用 FK/投影数学，但不等于在 GPU 路径调用这段 CPU fixture。

没有证据表明这次 CPU 原生崩溃沿当前 CUDA SAC/WM+MPPI/解析 MPC 路径触发；也不能据此保证所有 CPU 初始化或所有将来的平台均安全。不能把未解决的 CPU fixture 故障冒称当前 GPU 算法失败。

补充定位证据均保存到上述 runs 目录：
- 官方同版本 CUDA wheel 的 libtorch_cpu.so 哈希与当前文件及安装 RECORD 一致；未发现多套 torch/MKL 同时加载，LD_PRELOAD 为空。不能将其定性为已证实的库损坏或硬件损坏。
- 最小 standalone_fk_repro.py 只依赖 torch 与同目录的 kinematics/JSON fixture，不导入 pytest、WM 或材料后端；当前 CUDA wheel 在 CPU 运算时仍复现 SIGSEGV。
- MKL_ENABLE_INSTRUCTIONS=AVX2 曾两次通过 30 轮循环和一次 35 项组合，但第三次独立循环仍崩溃。该候选不成立，环境入口和相应候选测试已撤回。MKL_CBWR=AVX2 也未解决；两者不是同一个设置。
- 隔离环境 cutting-cpu-audit-20261121 使用官方 torch 2.4.1+cpu（仍为 MKL 2022.2），原 FK fixture 30 轮在 3 个独立进程均通过；71 项定向组合通过，另 2 个独立进程原 35 项组合通过。该环境仅作调查对照，没有链接为正式运行入口，也没有替换训练 CUDA 环境；收尾已删除该临时环境和本轮下载的 wheel，保留环境信息、日志及 wheel_comparison.json。
- 原生底层根因仍未完全确定。最终交付没有算法修补、没有数学替代、没有宣称已修复当前 CUDA wheel 的全部 CPU 路径。

Intel 的 [ISA 派发说明](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-0/instruction-set-specific-dispatch-on-intel-archs.html) 说明了 MKL_ENABLE_INSTRUCTIONS 的用途；官方支持该开关并不意味着本次未通过重复验证的设置可作为修复。
