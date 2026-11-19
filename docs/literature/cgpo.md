# CGPO：有限时域约束的梯度估计

> Project interpretation updated 2026-11-19; original paper/code reading scope is unchanged.
> Superseded project hypotheses are [archived](../archive/LITERATURE_POSITIONING_20261110.md).
核对日期：2026-11-10。证据：全文 T；作者代码未取得。
来源：[arXiv 2412.11138v1](https://arxiv.org/abs/2412.11138v1)；[ICML 2024](https://proceedings.mlr.press/v235/dai24d.html)。
已读：全文 §§1–8、Appendix A–F（31 页），包括 Taylor/KKT 推导、指标定义和 F.4 的 WM 版本。

## 方法与梯度
把有限轨迹的非折扣累计 reward/cost 对策略参数作一阶展开，在欧氏信赖域内求约束步；依据预测与实测变化之比调整域半径。理论上界仍含未知 Hessian 界和非零约束误差，不能读成硬安全。
实现说明采用 SHAC 短展开、reward/cost 双 critic，critic 输入时间，TD(λ) 拟合。**§6.4 与附录 F.4 已明确给出 learned WM+BPTT 的 MB-CGPO**：MLP 联合预测下一状态、reward、cost，真实采集后训练模型，再在模型里生成轨迹与策略梯度。

## 任务、数据、指标
Brax CartPole/Reacher/Ant/HalfCheetah，五种子；主指标是可行收敛的环境交互数，以及靠近预算边界时更新的违规比例。约束是累计位置/速度成本；有简单非光滑函数、非可微 Swimmer 的 WM 消融。不是瞬时接触力上限，也不是切割。

## 代码检索与不足
检查全文链接、PMLR 页面，以及精确题名 + GitHub/CGPO/作者搜索；本轮未获得可归属的作者算法仓库。不能把同名其他 CGPO 或 OmniSafe 的通用代码当成本文实现。
数学推导已读，但没有执行独立定理验证；模型误差、稀有事件和 critic 偏差不在理想 Taylor 界中得到自动解决。

## 与当前研究的关系

短视野世界模型策略梯度与约束优化已有先例。当前代码没有采用 CGPO 信赖域；相关分析用于区分模型预测准确性、动作梯度可靠性及真实执行结果，不构成现有切割算法的安全证明。
