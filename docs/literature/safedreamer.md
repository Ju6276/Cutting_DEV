# SafeDreamer：Safe Reinforcement Learning with World Model

核对日期：2026-11-10。ICLR 2024，[2307.07176v3](https://arxiv.org/abs/2307.07176v3)。已读主文与附录 A–C（含所有任务、参数和消融）；官方 [PKU-Alignment/SafeDreamer](https://github.com/PKU-Alignment/SafeDreamer)，提交 529221f02b54d5093e304d877eb2aaf6da63a9ff；未运行。

## 逐节与算法

§4 的 OSRP 用同一 RSSM 做约束 CEM 规划，并混合 actor 候选；OSRP-Lag 加入长期成本 critic 与 PID 乘子；BSRP-Lag 在模型内训练安全 actor，省去部署规划。§5 解释 WM/双 critic 学习。§6 和附录比较视觉/低维 Safety Gym、驾驶等任务的回报、累计代价与训练成本率，并讨论约束预算、重建损失和成本权重的影响。成本约束是累计期望量；近零平均代价不是逐动作安全保证。

## 核心源码核对

已读 SafeDreamer/agent.py、behaviors.py 全部，jaxutils.Lagrange 与 configs.yaml 相关项。
- WorldModel.loss 按 grad_heads 决定各头是否回传 latent；默认含 decoder/reward/cont/cost，成本并未默认 detach。
- ImagSafeActorCritic.train 只 detach actor 的状态输入，imagine 保留动作到模型转移链；连续动作默认 actor_grad_cont=backprop。
- loss 的 backprop 分支直接对 return/cost 求导；论文式(7)/(10) 的 stop-gradient 写法不足以完整描述实际 actor 梯度，应以这里的显式分支为准。
- 双 critic 的训练目标 stop_gradient；slow network 作正则。不是“不用目标控制”的简单标量 critic。
- CCEPlanner/PIDPlanner 评估短期成本加长期价值，热启动并执行第一步；无可行候选时选择低成本轨迹。此 fallback 不能称硬安全。
- 低维成本还可由重建几何经阈值计算；不等同学连续接触力峰值。

## 对 B 的意义

安全 WM 同时服务规划和策略、双 critic、BPTT、实际成本更新乘子均有开源先例。B 的价值必须在特定力风险建模与控制机制而非这些组件组合上。适合借用统一 WM 的控制对照与明确的梯度配置；不宜整库迁入来替换当前任务，也不能用其抽象累计成本结论代替传感器分量力限。

资料：contact_risk/2307.07176.*、code/safedreamer/。未完整审计 JAX/TensorFlow 底层库及所有环境包装器，未做数值复现。
