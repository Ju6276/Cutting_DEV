"""train_sac_multi.py refuses the read/evaluate-only pose_depth_v2 before creating its output."""
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'configs/archive/shared_scene_v1_pose_depth_v2_20261116/scene_v1.json'


@pytest.mark.parametrize('flag', ['--conditions', '--blademaster-config'])
def test_sac_multi_refuses_v2_before_output_directory(tmp_path, monkeypatch, flag):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    task = json.loads(ARCHIVE.read_text())['common']['task']
    assert task['version'] == 'pose_depth_v2'
    config = tmp_path / 'v2.json'
    config.write_text(json.dumps({'task': task, 'groups': [], 'worlds': 1}))
    output = tmp_path / 'sac_out'
    import cutting_world_model.multi_env as multi_env
    import cutting_world_model.blademaster_backend as blademaster
    built = []
    monkeypatch.setattr(multi_env.MultiConditionEnvironment, 'from_config',
                        classmethod(lambda cls, *a, **k: built.append(1)))
    monkeypatch.setattr(blademaster, 'BladeMasterEnvironment', lambda *a, **k: built.append(1))
    monkeypatch.setattr(sys, 'argv', ['train_sac_multi.py', flag, str(config), '--output', str(output), '--smoke',
                                      '--max-transitions', '16', '--max-hours', '.01'])
    with pytest.raises(ValueError, match='pose_depth_v2 is kept only'):
        runpy.run_path(str(ROOT / 'scripts/train_sac_multi.py'), run_name='__main__')
    assert not output.exists() and not built
