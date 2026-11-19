"""pose_depth_v3 is the single current task version; pose_depth_v2 is read/evaluate only."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutting_world_model.free_cut import CURRENT_POSE_DEPTH, refuse_retired_training, validate_task

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'configs/shared/scene_v1.json'
ARCHIVE = ROOT / 'configs/archive/shared_scene_v1_pose_depth_v2_20261116/scene_v1.json'


def test_current_shared_scene_preserves_archive_except_version_and_authorized_horizon():
    current, archived = (json.loads(p.read_text()) for p in (CURRENT, ARCHIVE))
    assert current['common']['task']['version'] == CURRENT_POSE_DEPTH == 'pose_depth_v3'
    assert archived['common']['task']['version'] == 'pose_depth_v2'
    assert archived['common']['task']['horizon_steps'] == 180
    assert current['common']['task']['horizon_steps'] == 500
    archived['common']['task']['version'] = 'pose_depth_v3'
    archived['common']['task']['horizon_steps'] = 500
    archived['common']['task']['board']['motion'] = 'project_continue_v1'
    assert archived == current
    validate_task(current['common']['task'])


def test_v2_training_is_refused_and_v3_is_accepted():
    refuse_retired_training({'version': 'pose_depth_v3'})
    refuse_retired_training(None)
    with pytest.raises(ValueError, match='read/evaluate'):
        refuse_retired_training({'version': 'pose_depth_v2'})


@pytest.mark.parametrize('retired',['pose_depth_v2','pose_depth_v3'])
def test_train_gpu_refuses_v2_config_before_building_physics(tmp_path, monkeypatch,retired):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import train_gpu
    task = json.loads(ARCHIVE.read_text())['common']['task']
    task['version']=retired
    config = tmp_path / 'v2.json'
    config.write_text(json.dumps({'task': task, 'groups': []}))
    built = []
    import cutting_world_model.multi_env as multi_env
    monkeypatch.setattr(multi_env.MultiConditionEnvironment, 'from_config',
                        classmethod(lambda cls, *a, **k: built.append(1)))
    args = SimpleNamespace(backend_root=ROOT / 'backends/mujoco', seed=0, force_penalty='sliceit',
                           force_penalty_scale=50., blademaster_config=None, conditions=config)
    with pytest.raises(ValueError, match='kept only|read/evaluate-only'):
        train_gpu.run(args, tmp_path, {}, 0.)
    bm = tmp_path / 'bm.json'
    bm.write_text(json.dumps({'task': task, 'worlds': 1}))
    with pytest.raises(ValueError, match='kept only|read/evaluate-only'):
        train_gpu.run(SimpleNamespace(**{**vars(args), 'conditions': None, 'blademaster_config': bm}),
                      tmp_path, {}, 0.)
    assert not built


def test_training_guard_rejects_legacy_v3_but_accepts_current_contract():
    old=json.loads(ARCHIVE.read_text())['common']['task']
    old['version']='pose_depth_v3'
    with pytest.raises(ValueError,match='Historical pose_depth_v3'):
        refuse_retired_training(old)
    current=json.loads(CURRENT.read_text())['common']['task']
    refuse_retired_training(current)
