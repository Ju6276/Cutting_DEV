# 文档入口

当前主线：UR10e / ATI Mini45 / IKEA刀、五食材、pose_depth_v3 /
project_continue_v2。UR5e和旧任务记录不等于当前UR10e配置。
论文主对照为SAC和WM+MPPI；解析MPC保留作工程测试工具。

## 仿真、训练和论文

- [PLAN](PLAN.md)：当前决定和待办，不累积已完成的开发计划。
- [METHOD](METHOD.md)：模型、学习、状态、控制和后端的实现。
- [TASK_FREE_CUT](TASK_FREE_CUT.md)：奖励、成功/失败、砧板边界。
- [SHARED_SCENE](SHARED_SCENE.md)：当前UR10e五食材，以及明确分开的UR5e配置。
- [CURRENT_BASELINES](methods/CURRENT_BASELINES.md)：各方法输入输出、比较用途、模型近似。
- [RUNNING](RUNNING.md)：安装、训练、评估、诊断和渲染命令。
- [RESULTS](RESULTS.md)：带条件和时间的测量证据，包括修复与长训记录。
- [PAPER](PAPER.md)：论文问题、比较范围与尚待证实的结论。
- [DATA](DATA.md)与[MACHINES](MACHINES.md)：数据目录、运行记录、电脑和访问方式。

## 真机：按这个顺序读

1. [UR10E_FDCC_SETUP_ZH](hardware/UR10E_FDCC_SETUP_ZH.md)：
   当前ROS1底层控制路线，以及仿真/真机候选的关系。
2. [TAMSCAGE](hardware/TAMSCAGE.md)：现场原控制器、独立对齐候选和离线实测证据。
3. [REAL2SIM](hardware/REAL2SIM.md)：TCP、传感器、摩擦、材料辨识与验证方法。
4. [UR10E_MPC_HANDOFF_ZH](hardware/UR10E_MPC_HANDOFF_ZH.md)：
   上层目标/反馈/动作、MPC影子运行和接入流程；不是可直接驱动机器人的入口。
5. [MECH_EYE_NANO_CHECKLIST_ZH](hardware/MECH_EYE_NANO_CHECKLIST_ZH.md)：
   ju-ROG相机实机参数、采集记录及待标定项。

离线控制器计算对齐已经通过；真实ATI、在线传输与真机闭环尚未验收。
这些状态只在现场记录更新，不在多个安装文档分别维护。

## 视觉、资产和来源

- [VISION_ROUTE1](methods/VISION_ROUTE1.md)：两种视觉选点方法、DP3输入输出和调用入口。
- [MECH_EYE_NANO](methods/MECH_EYE_NANO.md)：仿真相机与统一渲染。
  暂定350 mm光学配置不是实机550 mm相机标定。
- [食材资产候选](reference/OMNIOBJECT3D_FOOD_CANDIDATES.md)、
  [切割机制](reference/CUTTING_MECHANISMS.md)、
  [代码来源](reference/SOURCES.md)及[论文证据](literature/README.md)。
- [脚本索引](../scripts/README.md)和[配置索引](../configs/README.md)。

过时交接单和重复审阅文档已经删除；仍有效的测量集中在RESULTS。
Git保留文档历史，服务器实验数据不随文档清理删除。
