# AHAC：接触刚性引起的梯度问题与自适应展开长度
核对日期：2026-11-10。证据：全文 T + 作者核心代码 C；没有复现实验。
来源：[arXiv 2405.17784v2](https://arxiv.org/abs/2405.17784v2)；[官方项目](https://adaptive-horizon-actor-critic.github.io/)。
已读全文 §§1–7、Appendix A–I（20 页），包括假设、理论推导、单环境与并行版本、完整消融。

## 论文方法与结果
软接触极陡时，有限样本的一阶策略梯度可能错过很窄的有效区间。单环境 AHAC-1 在刚性接触处截断梯度；并行 AHAC 调共同展开长度，并结合双 critic 和迭代价值拟合。理论要求平滑、有界梯度，实验将这些假设放宽；“empirical bias”不能混同于所有一阶梯度统计上有偏。
五种 dflex locomotion 任务，10 seeds，报告 IQM/区间、交互和墙钟时间；不是学到的 WM，不是力限安全实验。contact threshold 限的是接触刚性代理，不是最大允许接触力。

## 代码核对
官方链接指向 imgeorgiev/DiffRL，固定 commit `bb59db5cf65e63740787bf22f91bae3103b30d19`，服务器 code/ahac。
已完整读 `src/shac/algorithms/ahac.py`、`models/critic.py`；另读 `ahac1.py` 的 rollout/接触截断与价值目标部分（其他辅助函数未逐行读）。
- `AHAC.compute_actor_loss`：actor→tanh→dflex.step→reward/critic 可微；终止归零，时限截断可 bootstrap。实际 actor loss 是 reward+critic，论文 Eq.10 的显式接触惩罚未出现在该函数。
- `train`：对 contact_forces/accelerations 的统计调整 lambda/H；`compute_target_values` 在 no_grad 内构造 TD(λ)。
- 当前 `DoubleCriticMLP` 用同一个 modules 列表建立两次 Sequential，模块对象因此共享。这是静态代码发现，未执行验证；不能直接宣称这份当前实现有两个独立 critic。
- 当前并行归一化把负 acceleration 也用 maximum(negative,1)，与 AHAC-1 的保留符号截断不同。需先查版本才能拿它作严格实现模板。

## 对 B 的意义
短视野和接触处分段求导已有清晰先例；“动态缩短 rollout”不能直接当新意。AHAC 处理已知物理梯度的接触刚性，B 处理学到的模型的力误差，二者问题不同。若模型数值校准很好而动作梯度仍错误，应研究决策相关的梯度可靠性，不能仅用误差大小决定任意 detach。
