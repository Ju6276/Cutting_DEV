# LAMBDA：Learning Safe Latent Policies（Route B 直接先例）

核对日期：2026-11-10。论文：[arXiv 2201.09802v4](https://arxiv.org/abs/2201.09802v4)。已读主文与附录 A–J；参考文献用于追踪，未逐篇阅读。源码：[yardenas/la-mbda](https://github.com/yardenas/la-mbda)，固定提交 542df63fb3ab5712185644d07ca2968faa6f76fe。本次未执行其训练。

## 逐节与机制

§3–4：视觉 RSSM 学习状态转移、奖励、终止与约束代价；SWAG 表示权重后验。策略在模型内展开，收益取乐观估计，约束取悲观估计，以增广拉格朗日更新策略。独立的回报和安全 critic 使用 TD(λ)。§5：Safety Gym 六任务，对照无模型方法及同模型 CEM-MPC；另有不使用 critic 和不使用后验不确定性的消融。报告回报、训练/评估代价和环境交互数。附录细化 SWAG、实验参数、梯度与局限；Push 任务的遮挡失败没有被隐藏。

约束是期望累计代价预算，不是每步接触力上界。后验近似及不确定性不能自动给出分布无关覆盖保证。任务没有切割不可逆状态。

## 实际源码与梯度流

- la_mbda/la_mbda.py：LAMBDA.train 先更新 WM，再从后验 rollout 更新 actor/critics；模型样本上的最大安全代价与最大收益分开选择。
- la_mbda/models.py：WorldModel.inference_step 联合训练状态、奖励、终止和成本。成本 BCE 可以回传共享模型；论文并没有通用地隔离安全辅助头梯度。
- WorldModel.generate_sequence 中 actor 的输入 features 使用 stop_gradient，动作之后的转移链仍可微。不能把这个 stop_gradient 误说成“策略不穿过模型”。
- Critic.train 以 delayed model 与 stop_gradient 的 λ-return 为目标，定期复制目标网络；SafetyCritic.penalize 更新乘子，构成增广拉格朗日罚项。
- swag_world_model.py 对模型参数采样后逐模型展开；不是只给预测均值加常数安全余量。

## 对本项目的意义

“学习模型 + 不确定性悲观代价 + BPTT 策略”及“同一模型：策略 vs MPC”都有直接先例，不能作为本项目方法新意。可复用的是严格分开回报/风险、冻结目标与明确梯度路径的设计。若 B 只有校准一个力余量，需要证明它改变了什么控制问题、如何处理持续策略变化与不可逆损伤，且需与更接近的校准接触模型对照。图片 RSSM、SWAG 和累计二值成本不能直接当作我们的传感器分量力限。

资料保存在服务器 contact_risk/2201.09802.* 与 contact_risk/code/lambda/。未阅读范围：其引用论文；未做数值复现。
