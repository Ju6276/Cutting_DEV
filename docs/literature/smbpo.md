# SMBPO：Safe Reinforcement Learning by Imagining the Near Future

核对日期：2026-11-10。论文 [2202.07789v1](https://arxiv.org/abs/2202.07789v1)，已读 §§1–6、附录 A.1–A.3。官方 [gwthomas/Safe-MBPO](https://github.com/gwthomas/Safe-MBPO)，提交 416dd13b6d4eceec7ac89ece161627e166ea54d8；未运行。

## 论文与假设

先区分已经违规与“不可恢复、但尚未违规”的状态。假设所有不可恢复状态会在有限 H* 内暴露违规；用足够大的吸收惩罚让最优策略避险。理论以覆盖真实后继的集合模型和最坏情况 Bellmin 备份建立下界。实际算法改为概率集成和平均目标，论文明确承认不是理论中的最小值备份。实验在 MuJoCo 四类运动任务比较回报、累计失败与交互数，附录给随机系统扩展和实现参数。Ant/Humanoid 移除了难建模的接触力观测，不能当作已解决力预测的证据。

## 核心代码

已读 src/smbpo.py、src/ssac.py、src/dynamics.py 全部。
- SMBPO.rollout 在 no_grad 下从真实 replay 起点做短模型 rollout，依原环境 check_done/check_violation 截断，存入虚拟 replay；不是穿过动力学训练 actor。
- update_solver 混合真实/虚拟转移，调用 SAC。SSAC.compute_target 对违规样本设固定吸收价值；双 Q 与 EMA target，actor 对 Q 求导。
- BatchedGaussianEnsemble 学状态增量和奖励的高斯均值/方差；模型随机抽取，不做 conformal 校准。
- 代码将整个违规转移 target 直接赋为吸收值，不能直接替换本项目只扣一次终止罚分的任务定义。

## 对 B 的意义

短视野预警、学习模型和失败罚项并不新。可借鉴“可恢复性”与“当下是否超限”分开：切割控制中的制动距离、柔顺内部速度可能使尚未超限的动作已经不可恢复。但这只是有依据的研究问题，不能据本文就证明我们的回合存在此机制。它的校准要求是所有状态动作上的集合包含真值，远强于分布边际 conformal 覆盖。

资料：服务器 contact_risk/2202.07789.* 与 code/smbpo/。未阅读其他基线库；未进行实验复现。
