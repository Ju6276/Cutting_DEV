from pathlib import Path
import pytest
# The pure group-index helper must also be usable without optional SB3 imports.
import ast

def helper(name="representative_worlds"):
    tree = ast.parse((Path(__file__).parents[1]/"scripts/check_baseline_adapters.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<group helper>", "exec"), namespace)
    return namespace[name]

@pytest.mark.parametrize("counts,expected", [([3,3],[0,3]), ([1]*5,[0,1,2,3,4]), ([2,1,4],[0,2,3])])
def test_recorded_baseline_group_indices(counts, expected):
    assert helper()({"groups":[{"worlds": n} for n in counts]}, sum(counts)) == expected

def test_recorded_baseline_rejects_mismatched_worlds():
    with pytest.raises(ValueError):
        helper()({"groups":[{"worlds":2}]}, 3)

@pytest.mark.parametrize("task", [
    {"version": "pose_depth_v2"},
    {"version": "pose_depth_v3", "board": {}},
    {"version": "pose_depth_v3", "board": {"motion": "project_continue_v1"}},
    {"version": "pose_depth_v3"},
])
def test_live_probe_refuses_historical_tasks(task):
    with pytest.raises(ValueError):
        helper("validate_live_config")({"task": task})

def test_live_probe_accepts_current_task():
    helper("validate_live_config")({"task": {
        "version": "pose_depth_v3", "board": {"motion": "project_continue_v2"}}})
