# Shared scene v1 with task pose_depth_v2 (archived 2026-11-17)

Byte-identical copy of configs/shared/scene_v1.json as used by the 2026-11-16
shared runs (launched from the frozen checkout 5ec9f85, which keeps its own copy).
Kept only for traceability: reading and evaluating those checkpoints. New
training on pose_depth_v2 is refused by scripts/train_gpu.py and
scripts/train_sac_multi.py. The current file configs/shared/scene_v1.json differs
only in task.version (pose_depth_v3); see docs/TASK_FREE_CUT.md.
