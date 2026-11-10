# CAP：Conservative and Adaptive Penalty（Route B 直接先例）

核对日期：2026-11-10。论文：[arXiv 2112.07701v1](https://arxiv.org/abs/2112.07701v1)，已读主文、理论与附录 A–E。源码：[Redrew/CAP](https://github.com/Redrew/CAP)，提交 d870ccd9665038edd850dbb503d29a7369ec6e2a；未运行训练。

## 逐节与机制

§3–4 用转移误差的积分概率度量给累计成本建立上界，将模型成本加上不确定性罚项。有限状态理论使用计数置信界和联合界；神经网络实验改用集成分歧并依据实际成本自适应调节惩罚系数，不具有同一个已证明的界。§5 与附录在网格、速度受限 HalfCheetah 和 CarRacing 上评估回报、违规及样本量；CarRacing 代价涉及滑移/摩擦，并非完全与接触无关。规划采用约束 CEM：优先可行高回报样本，不足时选较低成本样本。

重要边界：控制目标是期望累计成本；低成本 fallback 不等于确保满足约束。理论中的真实值函数有界、误差界有效等假设必须说明。

## 代码实际做了什么

- cap-pets/ccem.py：ConstrainedCEM 热启动平移动作序列，采样候选，模型粒子展开，筛选满足预测预算的 elites；整个规划使用 no_grad，不是训练一个 BPTT actor。
- _predict_next 使用集成高斯预测的标准差范数构成惩罚。二值成本的阈值与浮点风险并不是同一种量。
- optimize_kappa 根据实际回合成本与预算差，用 Adam 更新罚系数；这属于实际执行反馈的适应，不是 conformal calibration。
- cap-planet/planner.py 对潜变量集成模型执行相似的约束采样规划。
- run_cap_pets.py 连接真实环境成本记录和系数更新。只核对了相关调用与配置，未完整审计其模型训练实现；不据此声称整个仓库无 bug。

## 对本项目的意义

“给模型预测成本加不确定性，执行后适应保守程度”已经存在。B 必须区别于 CAP 的风险控制机制，而不仅把 cost 换成 force。可复用真实执行结果回馈保守性、以及保留不可行率的报告方式；不能把神经网络集成分歧直接写成有覆盖保证的上界。我们若做逐控制区间峰值约束，也应明确与累计期望代价的区别。

资料：服务器 contact_risk/2112.07701.* 和 contact_risk/code/cap/。未阅读范围：库依赖内部实现与引用论文；未复现其 benchmark。
