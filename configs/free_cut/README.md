# free_cut_v1 interface configuration

interface_check.json is an explicit, **rejected reward candidate**, retained to
reproduce the physical preference check. It is not an approved training config.
Fast pressing earned more return than following the commanded speed. See
docs/FREE_CUT_REVIEW.md and docs/RESULTS.md.

scripts/check_free_cut_task.py consumes this file. It contains the task weights,
horizon, six-dimensional simulation limits and reference settings. No real Mini45
limit or calibrated friction is implied.

For a future MultiConditionEnvironment configuration, the same task object goes
under top-level task; each group must explicitly select net feedback and positive
settle_seconds. Material randomization is followed by per-world re-settling.
Do not copy the rejected weights into a long training launch.
