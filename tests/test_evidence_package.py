import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("build_evidence", Path(__file__).parents[1] / "scripts/build_evidence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_mixed_disturbances_and_case_streams_are_rejected():
    a = {"contract": "public-prism-ur5e-v1", "cases": [{"case": 0, "seed": 7}],
         "perturbation_point": "executed_action"}
    for update in ({"cases": [{"case": 0, "seed": 8}]},
                   {"perturbation_point": "raw_policy_output"}, {"contract": "another-scene"}):
        with pytest.raises(ValueError, match="Mixed"):
            module.require_same_cases([a, {**a, **update}])


def test_wall_budget_prefix_is_not_a_failure():
    assert module.classify_episode({"completed": False, "success": False, "collision": False}) == "censored"
    assert module.classify_episode({"completed": True, "success": False, "truncated": True}) == "truncated"
    assert module.classify_episode({"completed": True, "collision": True}) == "collision"


def test_saved_pair_is_verified_against_raw_cases():
    a = [{"id": 0, "outcome": "success", "steps": 20}, {"id": 1, "outcome": "outside", "steps": 5}]
    b = [{"id": 0, "outcome": "success", "steps": 21}, {"id": 1, "outcome": "success", "steps": 20}]
    report = {"perturbation_point": "executed_action",
              "a_minus_b": {"all_cases": {"success": {"mean_difference": -.5, "interval_95": [-1, 0]}},
                           "both_success": {"cases": 1, "steps": {"mean_difference": -1, "interval_95": [-1, -1]}}}}
    module.verify_pair(report, a, b)
    report["a_minus_b"]["both_success"]["steps"]["mean_difference"] = -3
    with pytest.raises(ValueError, match="does not match"):
        module.verify_pair(report, a, b)
