# SAC 与 WM+MPPI 仿真路径审阅（2026-11-21）

## 结论与范围

本次独立阅读当前 main 的 SAC 多环境训练/评估和冻结 WM+MPPI 实现，未确认新的算法实现错误，因此没有修改奖励、动作、终止规则或训练代码。这个结论只覆盖下列路径，不代表没有任何 bug，也不是对真机或切割成功率的认证。

解析 MPC 的工具去皮基准错误已在之前修复；它位于 AnalyticFDCCModel，不等于 SAC 或冻结学习 WM 使用同一错误模型。本次核对该上下文没有进入两者的网络输入。

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
