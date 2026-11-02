import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cutting_world_model.native_state import decode_controller_snapshot, decode_filter_snapshot
from cutting_world_model.state_spec import CuttingStateSpec
from test_state_spec import fixture


def response(value):
    return SimpleNamespace(success=True, message=json.dumps(value))


def test_native_packet_keeps_measured_and_controller_velocity_distinct():
    values = np.arange(74, dtype=float)
    values[:4] = [1, 23, 4.5, 6]
    packet = decode_controller_snapshot(response(values.tolist()))
    np.testing.assert_array_equal(packet['controller_measured_qd'], values[10:16])
    np.testing.assert_array_equal(packet['controller_velocity'], values[22:28])
    np.testing.assert_array_equal(packet['controller_previous_position_command'], values[28:34])
    np.testing.assert_array_equal(packet['controller_previous_error'], values[34:40])
    np.testing.assert_array_equal(packet['controller_cached_tip_pose'], values[40:47])
    np.testing.assert_array_equal(packet['controller_target_pose'], values[47:54])
    assert packet['hardware_stamp'][0] == values[-1]


@pytest.mark.parametrize('value', [[], [[]], [1, 0, 0, 6], [2, 0, 0, 6], [1, 0, 0, 6.5]])
def test_rejects_malformed_native_packets(value):
    with pytest.raises(ValueError):
        decode_controller_snapshot(response(value))


def native_frame():
    meta, frame = fixture()
    for name in CuttingStateSpec.native_recurrent_fields:
        frame[name] = np.zeros(6)
    for name in ('controller_cached_tip_pose', 'controller_target_pose', 'coupling_previous_pose'):
        frame[name] = np.array([0., 0., 0., 0., 0., 0., 1.])
    frame['filter_history'] = np.zeros((200, 6))
    frame['filter_zi'] = np.zeros((2, 6))
    frame['wrench_queue'] = np.zeros((25, 6))
    for name in ('filter_history_count', 'filter_initialized', 'filter_added_age', 'filter_raw_age',
                 'controller_sample_age', 'filter_sample_age', 'wrench_queue_count', 'coupling_initialized'):
        frame[name] = np.ones(1)
    frame['filter_flags'] = np.ones(2)
    for name in CuttingStateSpec.recurrent_constants:
        frame[name] = np.zeros(6)
    meta['state_layout'] = 'native_ros_recurrent_v1'
    return CuttingStateSpec.metadata_from_frame(meta, frame), frame


def test_native_state_pack_retains_hidden_velocity_and_filter_memory():
    metadata, frame = native_frame()
    spec = CuttingStateSpec(metadata)
    before = spec.pack(frame)
    frame['controller_velocity'][2] = .7
    frame['filter_zi'][1, 3] = .2
    after = spec.pack(frame)
    assert torch.count_nonzero(before != after) == 2
    torch.testing.assert_close(spec.field(before, 'joint_qd'), spec.field(after, 'joint_qd'))
    fields = spec.integrate_physical(after[None], torch.zeros(1, spec.delta_dim))
    torch.testing.assert_close(fields['controller_velocity'][0, 2], torch.tensor(.7))
    torch.testing.assert_close(fields['filter_zi'][0, 1, 3], torch.tensor(.2))
    assert torch.isfinite(spec.canonical(after)).all()
    assert spec.error_vector(after, after).shape == (spec.delta_dim,)


def test_filter_decoder_preserves_delays_and_expiry_timestamp():
    _, frame = native_frame()
    data = {name: frame[name].tolist() for name in (
        'filter_history', 'filter_history_count', 'filter_zi', 'filter_initialized',
        'filter_offset', 'filter_added_wrench', 'filter_flags')}
    data.update(schema_version=[1], sequence=[19], stamp=[2.], raw_stamp=[1.999], topic2_stamp=[1.995])
    packet = decode_filter_snapshot(response(data))
    assert packet['filter_history'].shape == (200, 6)
    assert packet['filter_zi'].shape == (2, 6)
    assert packet['raw_stamp'][0] - packet['topic2_stamp'][0] == pytest.approx(.004)
    del data['filter_zi']
    with pytest.raises(ValueError, match='filter_zi'):
        decode_filter_snapshot(response(data))


def test_native_sampler_reads_source_after_observation_instead_of_latching_earlier():
    from cutting_world_model.native_state import NativeStateRecorder
    _, frame = native_frame()
    sensor = {name: frame[name].tolist() for name in (
        'filter_history', 'filter_history_count', 'filter_zi', 'filter_initialized',
        'filter_offset', 'filter_added_wrench', 'filter_flags')}
    sensor.update(schema_version=[1], sequence=[7], stamp=[2.], raw_stamp=[1.999], topic2_stamp=[1.995])
    controller = np.zeros(74)
    controller[:4] = [1., 9., 2., 6.]
    controller[-2:] = [1., 2.]
    recorder = NativeStateRecorder.__new__(NativeStateRecorder)
    calls = []
    def controller_service():
        calls.append('controller')
        return response(controller.tolist())
    def filter_service():
        calls.append('filter')
        return response(sensor)
    recorder.controller_service = controller_service
    recorder.filter_service = filter_service
    # The final controller tick completes while the observation is obtained.
    controller[22:28] = .7
    sensor['filter_zi'][0][0] = .25
    recorded = recorder.read(boundary=2.)
    assert calls == ['controller', 'filter']
    np.testing.assert_array_equal(recorded['controller_velocity'], np.full(6, .7))
    assert recorded['filter_zi'][0, 0] == .25
    assert recorded['controller_sample_age'][0] == 0.
    assert recorded['filter_added_age'][0] == pytest.approx(.005)
