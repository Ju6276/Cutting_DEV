# Archived literature project interpretations — 2026-11-10

> HISTORICAL ONLY. Retired project hypotheses; not the current method, task, training gates or contribution claims. Archived on 2026-11-19. Paper/source evidence remains in the current per-paper notes.

Source snapshot: main dbdddf67e26cb15b4ddf64355b3a391652d77611. This archive preserves the superseded interpretation text, including dated assumptions that were not established results.


## mopo

### Relevance

Pessimism against model exploitation is prior art. Route B must compare against
a fairly calibrated ensemble and disclose distribution assumptions; adding
an uncertainty penalty alone is not new. A statistically conservative force
estimate and a return-error bound are different objects.


## smbpo

### 对 B 的意义

短视野预警、学习模型和失败罚项并不新。可借鉴“可恢复性”与“当下是否超限”分开：切割控制中的制动距离、柔顺内部速度可能使尚未超限的动作已经不可恢复。但这只是有依据的研究问题，不能据本文就证明我们的回合存在此机制。它的校准要求是所有状态动作上的集合包含真值，远强于分布边际 conformal 覆盖。

资料：服务器 contact_risk/2202.07789.* 与 code/smbpo/。未阅读其他基线库；未进行实验复现。


## safedreamer

### 对 B 的意义

安全 WM 同时服务规划和策略、双 critic、BPTT、实际成本更新乘子均有开源先例。B 的价值必须在特定力风险建模与控制机制而非这些组件组合上。适合借用统一 WM 的控制对照与明确的梯度配置；不宜整库迁入来替换当前任务，也不能用其抽象累计成本结论代替传感器分量力限。

资料：contact_risk/2307.07176.*、code/safedreamer/。未完整审计 JAX/TensorFlow 底层库及所有环境包装器，未做数值复现。


## cgpo

### 对 B 的意义
“安全约束+短视野+WM 的解析策略梯度”不是空白。其信赖域可以作算法先例，但移植整套 CGPO 会改变优化器，不是给当前 B 加一个名字。对我们更直接的启发是：校准数值上界不证明其动作方向导数可靠；应区分预测保守性与策略更新的真实风险变化。


## srl_vic

### What this changes for our direction

Force-aware policy learning, adaptive stiffness, safety critics and recovery switching are established mechanisms. A new cutting paper cannot use that collection alone as its novelty.

The current cutting task has irreversible internal damage and contact-mode changes. A wrench-only safety critic may alias states with different future force responses; adding history is plausible but is not by itself proof of a new algorithm. Route B should show that a specific model mechanism changes reliable action choice, not merely that a classifier predicts old trajectories better.

Do not copy the paper's 40 N-style limit, risk discount or threshold into the cutting task. Existing task definitions and units must remain explicit. A learned risk score below a threshold is not a hard per-step force guarantee.


## capture

### 对我们 B 的影响
“按接触阶段校准，而非全局固定余量”已有直接先例；不能据此独立声称方法新意。可借用其训练/分组/校准数据严格分离、候选未来模式求并集、按模式单独统计覆盖。不能照搬已知刚性几何层：切割中的损伤与再接触会改变未来动力学，并非固定 C-space 边界。要研究这一差别，必须先证明可观测历史对力风险存在额外信息。当前已有 peak_wrench 输出，因此新增峰值头也不是贡献。


## cqr

### 对本项目

可以用作 B 的标准校准基线；不构成方法新意。以回合最大欠预测分数校准时，样本数是独立回合数，不是该回合所有高度相关的帧数。多材料、策略迭代和以风险筛选采样都须另说明交换性或漂移。13 个事件的比较规则是开发决策，不是该定理给出的安全置信度；事件命中数和全样本覆盖率也不同。

资料：contact_risk/1905.03222.*、code/cqr/。未核对全部数据清洗脚本与上游 sklearn 实现。


## cerl

### Consequence for route B
Pessimistic model-ensemble cost + BPTT + constrained policy update already exists.
Replacing obstacle costs with force ratios is an adaptation, not automatically a new algorithm.
A cutting contribution must address the actual peak-observation/prediction/control issue and demonstrate resulting behavior.
CERL's expected-cost constraint cannot certify that every 2 ms sensor sample stays below our force threshold.

Follow-up code access on 2026-11-10: the official anonymous root, README endpoint and repository API all returned HTTP 403 from GPU7; the web tool could not retrieve the anonymous site. No code-read evidence is claimed. Exact-title/author-code searches did not verify a replacement public repository.


## dalal_safety_layer

### Implications for cutting and Route B

A calibrated force upper bound and an action correction are not sufficient novelty by themselves. The literature already has learned constraint sensitivities, projections and evidence that prediction loss can be misleading.

A cutting-specific opportunity would need a concrete mechanism for reliable action effects across contact changes and irreversible damage history, plus evidence that the chosen actions improve physical outcomes. Merely adding a global conformal residual constant to the mean does not repair its derivative with respect to action.

This paper also motivates an honest comparator: a simple learned local constraint-response model can be an informative baseline against a much larger WM. Whether it is adequate for the current cutting interval is an empirical question, not a reason to silently replace the existing task or force definition.


## lambda

### 对本项目的意义

“学习模型 + 不确定性悲观代价 + BPTT 策略”及“同一模型：策略 vs MPC”都有直接先例，不能作为本项目方法新意。可复用的是严格分开回报/风险、冻结目标与明确梯度路径的设计。若 B 只有校准一个力余量，需要证明它改变了什么控制问题、如何处理持续策略变化与不可逆损伤，且需与更接近的校准接触模型对照。图片 RSSM、SWAG 和累计二值成本不能直接当作我们的传感器分量力限。

资料保存在服务器 contact_risk/2201.09802.* 与 contact_risk/code/lambda/。未阅读范围：其引用论文；未做数值复现。


## ahac

### 对 B 的意义
短视野和接触处分段求导已有清晰先例；“动态缩短 rollout”不能直接当新意。AHAC 处理已知物理梯度的接触刚性，B 处理学到的模型的力误差，二者问题不同。若模型数值校准很好而动作梯度仍错误，应研究决策相关的梯度可靠性，不能仅用误差大小决定任意 detach。


## cap

### 对本项目的意义

“给模型预测成本加不确定性，执行后适应保守程度”已经存在。B 必须区别于 CAP 的风险控制机制，而不仅把 cost 换成 force。可复用真实执行结果回馈保守性、以及保留不可行率的报告方式；不能把神经网络集成分歧直接写成有覆盖保证的上界。我们若做逐控制区间峰值约束，也应明确与累计期望代价的区别。

资料：服务器 contact_risk/2112.07701.* 和 contact_risk/code/cap/。未阅读范围：库依赖内部实现与引用论文；未复现其 benchmark。


## passivity_centric

### Relation to cutting and Route B

This is a strong non-cutting precedent for **structured physical constraints plus learned compliant control**. It disproves a broad claim that prior safe learning ignores contact forces or energy.

A cutting adaptation could use controller/energy structure, but passivity alone cannot replace the project's actual force-limit event definition or a substep peak-force predictor. Conversely, a statistically calibrated force bound cannot establish passivity. These are different claims.

The useful lesson is methodological: a contribution needs an explicit physical or statistical mechanism with assumptions, not just another auxiliary loss. Copying an energy tank into the current controller would be an additional design change requiring its own cutting-specific justification; it is not recommended here as a spontaneous replacement for the approved S0–S4 route.


## cpo

### Implications for Route B

CPO is a relevant constrained-RL comparator or conceptual baseline, not a drop-in WM+MPC module. Constraining average cost is not equivalent to predicting and avoiding a peak force event.

A method that replaces the CPO cost with force still needs to state whether it constrains expected accumulated cost, probability of any violation, a filtered sensor component, or an interval maximum. Conformal coverage does not turn CPO's expected constraint into a hard guarantee.

A worthwhile cutting mechanism must address what the model knows about changing contact/material history and how its action sensitivity affects control. The existence of this paper rules out treating “separate safety cost plus policy optimization” as a new idea.


## chee_conformal_control

### 代码与对 B 的关系

论文、PMLR 页面及精确标题+作者+GitHub 检索未发现该篇独立官方代码；CasADi/IPOPT 是依赖，不能标成方法代码已开源。暂记“未发现可核对实现”，不是永久无代码。

“校准预测 + 约束收紧 + 控制”已有明确先例。本项目若仅给预测力加校准余量再优化，不足以成为新的校准控制机制。能借鉴的是将模型覆盖、参考可行性和闭环结果分开报告；不能照搬边际覆盖到自适应策略下每步 force-limit。

资料：服务器 contact_risk/chee24a.pdf 与文本。未逐篇阅读引用工作；未运行其控制实验。


## conformal_tutorial

### Implications for Route B

A calibrated upper force bound may be a useful component and diagnostic. It cannot alone justify “safe cutting” or a new policy-learning algorithm.

In particular, with a constant calibrated radius `U(s,a)=f_hat(s,a)+q`, the action derivative remains `dU/da=df_hat/da`. Better numerical coverage does not repair an incorrect action-dependent force gradient. An action-dependent uncertainty term can change this derivative, but its reliability still requires evidence under the actions the optimizer actually chooses.

For a small pilot, freeze the operating point on development data, preserve episode-level pairing, report gained and lost detected events and false alarms, and treat any count threshold as a resource-allocation rule. Do not rename it statistical significance or deployment safety.



## parmar_contact_learning_bias

### Relation to Route B

This is strong evidence against treating our force-peak error as merely an insufficient loss weight or an MLP capacity issue. Contact-time uncertainty, partial observations and force filtering can matter before calibration is considered.

Useful borrowing: report contact-onset timing and short recursive rollout behavior separately from aggregate one-step MSE. Do not soften our simulator or lower contact stiffness merely to improve the model fit.

Not supplied by this work: a calibrated force bound, a safe policy update, or a proof that a particular cutting representation fixes the problem. It motivates examining whether the model gives the correct *action-dependent change* in force, rather than only the correct average force.



## hewing_cautious_gp_mpc

### Relation to Route B

Already established: learned dynamics + propagated model uncertainty + online optimization under chance constraints. Therefore simply putting an upper prediction bound into MPPI/MPC is not a new general principle.

Useful borrowing: distinguish observation noise, model error and rollout uncertainty; state exactly which approximation is propagated; budget online compute; account for action-dependent uncertainty.

What remains open for our implementation is not answered here: partially observed, irreversible cutting state; force maxima between control samples; and whether a model's action gradient points toward a physically lower-force action. Calibrating a marginal force bound is only one part of that control question.



## README

# Literature review index — 2026-11-10

This is the per-paper evidence ledger requested by the user, produced by Codex
with one literature subagent. It covers the named method papers in the reading list below
and archived NOVELTY, with separately identified simulator and data references.
[Synthesis](../literature/../archive/NOVELTY.md), [paper argument](../literature/../PAPER.md) and
[current task list](../literature/../PLAN.md) have different roles.

Each record states **what was actually read**, method and data, relevant
experiments/metrics, limitations, and inspected code/version where available.
A scoped method/experiment read is not a complete proof or dependency audit.
A downloaded PDF, an abstract, a README or an advertised Code button is not
counted as an implementation review. None of these external algorithms was
trained or numerically reproduced in this review. No sealed test was opened.

The main conclusion is conditional: the inherited WM+BPTT framework and a
multi-object benchmark do not themselves establish a new method. B also needs
a distinction from existing conservative world-model control, force MPC and
contact-conditioned calibration. The recommended research question concerns
useful action changes under changing cutting interaction; it is a hypothesis,
not a claim that these papers prove an untouched gap.


## blademaster.md — superseded project role

Potential future forward data generator; it can support a learned WM without differentiating through its topology. Replacing DiSECt now is outside S0–S4 and would confound the learning comparison. This review neither revives the old solver-rewrite route nor claims it is a fully differentiable cutting system.
