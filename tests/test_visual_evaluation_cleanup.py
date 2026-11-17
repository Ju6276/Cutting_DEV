"""Persistence faults must never strand a live physical environment."""
from types import SimpleNamespace
import json

import pytest
from scripts import evaluate_visual_cut as evaluation


@pytest.mark.parametrize("failure", ["planner", "trajectory", "summary", "none"])
def test_result_persistence_closes_environment(tmp_path, monkeypatch, failure):
    closed = []
    env = SimpleNamespace(close=lambda: closed.append(True))
    planner = SimpleNamespace(plans=[{"step": 1}])
    status = {"state": "completed"}

    def write(name, value):
        if (failure == "planner" and name == "planning.json") or (
            failure == "summary" and name == "summary.json"
        ):
            raise OSError("injected disk failure")
        (tmp_path / name).write_text(json.dumps(value))

    if failure == "trajectory":
        def fail_save(*args, **kwargs):
            raise OSError("injected disk failure")
        monkeypatch.setattr(evaluation.torch, "save", fail_save)
    if failure == "none":
        evaluation.save_results_and_close(env, planner, [], [], status, tmp_path, write)
        assert json.loads((tmp_path / "summary.json").read_text())["status"]["state"] == "completed"
    else:
        with pytest.raises(OSError, match="injected disk failure"):
            evaluation.save_results_and_close(env, planner, [], [], status, tmp_path, write)
        assert status["state"] == "failed"
        assert json.loads((tmp_path / "progress.json").read_text())["state"] == "failed"
    assert closed == [True]


def test_unwritable_failure_log_keeps_original_error_and_closes(tmp_path):
    closed = []
    env = SimpleNamespace(close=lambda: closed.append(True))
    calls = []

    def write(name, value):
        calls.append(name)
        raise OSError(name)

    with pytest.raises(OSError, match="planning.json"):
        evaluation.save_results_and_close(
            env, SimpleNamespace(plans=[]), [], [], {}, tmp_path, write)
    assert calls == ["planning.json", "progress.json"]
    assert closed == [True]
