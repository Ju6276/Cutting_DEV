"""train_sac_multi.py refuses read/evaluate-only task semantics before creating its output."""
import json
from pathlib import Path
import runpy
import sys

import pytest

# train_sac_multi imports Stable-Baselines3 at module level (optional SAC environment).
pytest.importorskip("stable_baselines3")

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'configs/archive/shared_scene_v1_pose_depth_v2_20261116/scene_v1.json'


# Each retired case must be refused with its own message (not merely any refusal).
RETIRED = {
    'pose_depth_v2': (None, 'pose_depth_v2 is kept only to read/evaluate the 2026-11-16 runs'),
    'pose_depth_v3_latch': (None, 'Historical pose_depth_v3 board latch is read/evaluate-only'),
    'pose_depth_v3_continue_v1': ('project_continue_v1',
                                  r'board\.motion=project_continue_v1 \(f12ca04 .* is read/evaluate-only'),
}


@pytest.mark.parametrize('retired', sorted(RETIRED))
@pytest.mark.parametrize('flag', ['--conditions', '--blademaster-config'])
def test_sac_multi_refuses_v2_before_output_directory(tmp_path, monkeypatch, flag,retired):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    task = json.loads(ARCHIVE.read_text())['common']['task']
    motion, message = RETIRED[retired]
    task['version'] = retired[:len('pose_depth_vN')]
    if motion is not None:
        task['board']['motion'] = motion
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
    with pytest.raises(ValueError, match=message):
        runpy.run_path(str(ROOT / 'scripts/train_sac_multi.py'), run_name='__main__')
    assert not output.exists() and not built
