"""Evidence validators must reject missing or poisoned outputs, not silently pass."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.verify_entry import finite_learner

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("key", ["world_model", "policy", "critic", "wm_optimizer"])
def test_nonfinite_learner_component_is_rejected(key):
    learner = {k: {"weight": torch.ones(2)} for k in ("world_model", "policy", "critic")}
    learner[key] = {"weight": torch.tensor([float("nan")])}
    assert not finite_learner(learner)


def test_missing_network_is_not_a_finite_checkpoint():
    assert not finite_learner({"policy": {"x": torch.ones(1)}, "critic": {"x": torch.ones(1)}})


def run_check(path, check):
    return subprocess.run([sys.executable, "-m", "scripts.verify_entry", str(path), "test",
                           "--backends", "disect", "--checks", check],
                          cwd=ROOT, capture_output=True, text=True)


def test_empty_evidence_directory_fails(tmp_path):
    result = run_check(tmp_path, "train_wm")
    assert result.returncode == 1, result.stderr
    assert not json.loads((tmp_path / "verify_test.json").read_text())["train_wm_disect"]["ok"]


def test_summary_existence_is_not_a_diagnostic_pass(tmp_path):
    directory = tmp_path / "diagnose_disect"
    directory.mkdir()
    (directory / "summary.json").write_text("{}")
    result = run_check(tmp_path, "diagnose")
    assert result.returncode == 1, result.stderr


def test_reported_failed_stage_has_failure_exit_code(tmp_path):
    (tmp_path / "pose_check_disect.json").write_text(json.dumps(
        {"status": "failed", "wm_reward_error_max": 1, "masked_reset_unselected_exact": False}))
    assert run_check(tmp_path, "pose_check").returncode == 1


def test_explicit_stage_selection_accepts_complete_evidence(tmp_path):
    (tmp_path / "pose_check_disect.json").write_text(json.dumps(
        {"status": "passed", "wm_reward_error_max": 1e-8, "masked_reset_unselected_exact": True}))
    assert run_check(tmp_path, "pose_check").returncode == 0
