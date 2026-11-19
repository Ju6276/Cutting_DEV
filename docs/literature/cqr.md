# CQR：Conformalized Quantile Regression

> Project interpretation updated 2026-11-19; original paper/code reading scope is unchanged.
> Superseded project hypotheses are [archived](../archive/LITERATURE_POSITIONING_20261110.md).

核对日期：2026-11-10。论文 [1905.03222v1](https://arxiv.org/abs/1905.03222v1)，全文 §§1–7、附录 A/B 与结果图表已读。官方 [yromano/cqr](https://github.com/yromano/cqr)，提交 37eeb0604cc57efc40b84e3539e8cfa8f09e30c1；未运行。

## 逐节与保证

§2–4：用 pinball loss 学上下条件分位数，再在独立校准集计算上下越界残差的最大值，采用有限样本修正的顺序统计量扩展区间。允许负校准修正，因此也能缩短过宽区间。§5 比较局部尺度归一化；§6 在 11 回归数据集报告覆盖率、区间宽度及 20 次划分。附录证明秩/交换性结果与合成异方差实验。

核心保证是对新样本的边际覆盖，须校准与测试可交换；不保证每个给定输入的条件覆盖。单侧控制和双侧控制不是同一个 α。

## 代码核对与实际边界

已读 helper.run_icp、AllQNet_RegressorAdapter，torch_models.py 的训练/损失/预测逻辑，以及 nonconformist/nc.py 和 icp.py 的回归校准路径。
- fit 只使用 proper training；calibrate 独立存储残差；predict 加上排序校准分位。
- AllQuantileLoss 是可微的分位数训练；NumPy 的排序校准及预测导出不构成 BPTT 控制实现。
- QuantileRegErrFunc.apply_inverse 将超范围索引夹到最大残差。若 n 很小且所需覆盖太高，正确有限样本规则可能要求无穷区间；直接照搬这个夹取实现会失去该保证。
- LearnerOptimizedCrossing.predict 的顺序 min/max 赋值在发生分位交叉时有覆盖原数组的风险。这里只是静态代码发现，未执行确认，不据此否定论文方法。

## 与当前研究的关系

本条保留为校准方法背景，当前主线没有使用 CQR。若以后比较校准，必须明确独立样本单位、交换性假设及策略变化；同回合高度相关帧不自动构成独立样本。历史事件筛选规则不属于当前任务或论文成功标准。

资料：contact_risk/1905.03222.*、code/cqr/。未核对全部数据清洗脚本与上游 sklearn 实现。
