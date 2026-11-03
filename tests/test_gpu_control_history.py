import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from test_control_state import compact_fixture


def tensor_batch(frames):
    return {name: torch.as_tensor(np.stack([frame[name] for frame in frames]))
            for name in frames[0]}


def test_batched_control_history_matches_episode_streams_through_partial_reset():
    metadata, initial = compact_fixture()
    spec = ControlStateSpec(metadata)
    frames = [copy.deepcopy(initial), copy.deepcopy(initial)]
    frames[1]['joint_q'] += .2
    state = spec.pack_batch(tensor_batch(frames)).float()
    expected = [spec.pack(frame) for frame in frames]
    for step in range(13):
        for world, frame in enumerate(frames):
            frame['joint_q'] += .001 * (world + 1)
            frame['wrench_history'][:] = step + world
            frame['observation'][9:15] = (step + world) / 20
        state = spec.pack_batch_transition(tensor_batch(frames), state)
        expected = [spec.pack_transition(frame, previous)
                    for frame, previous in zip(frames, expected)]
        torch.testing.assert_close(state, torch.stack(expected), atol=0, rtol=0)
        if step == 6:
            before = state.clone()
            frames[0] = copy.deepcopy(initial)
            state = spec.reset_batch_history(tensor_batch(frames), state,
                                             torch.tensor([True, False]))
            expected[0] = spec.pack(frames[0])
            torch.testing.assert_close(state[0], expected[0], atol=0, rtol=0)
            torch.testing.assert_close(state[1], before[1], atol=0, rtol=0)
    model = CuttingWorldModel(metadata, hidden=(12, 12))
    assert isinstance(model.spec, ControlStateSpec)
    assert (model.spec.state_dim, model.spec.delta_dim, model.spec.dynamics_dim) == (860, 36, 480)
    assert state.dtype == torch.float32


@pytest.mark.parametrize('failure', ['save', 'finish'])
def test_gpu_finalization_records_failure_and_attempts_remaining_stages(monkeypatch, tmp_path, failure):
    script = Path(__file__).resolve().parents[1] / 'scripts/train_gpu.py'
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location('gpu_training_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events, records = [], []
    status = {'state': 'completed'}

    def checkpoint(name):
        events.append('save')
        assert name == 'last.pt'
        if failure == 'save':
            raise OSError('checkpoint write failed')

    def record(**values):
        events.append('record')
        records.append(copy.deepcopy(dict(status, **values)))

    def finish(path, state):
        events.append('finish')
        assert path == tmp_path / 'last.pt'
        if failure == 'finish':
            raise RuntimeError('tracker finish failed')

    errors = module.finalize_gpu_run(checkpoint, record, SimpleNamespace(finish=finish),
                                     tmp_path, status, 12, 3)
    assert events == ['save', 'record', 'finish', 'record']
    assert errors[0]['stage'] == failure
    assert records[-1]['state'] == 'failed'
    assert records[-1]['iteration'] == 12
    assert records[-1]['updates'] == 3
    assert records[-1]['cleanup_errors'] == errors
