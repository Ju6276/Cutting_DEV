"""pose_depth_v3: software board-stop boundary consistency (CPU, synthetic fixtures).

Physical adapters are exercised through their real step/reset code on stub physics
with every world already latched (no CUDA robot/material needed); the latch-step
peak of a live physical interval is covered at the rule-function level only.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_pose_cut import config
from test_state_spec import fixture
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.free_cut import FreeCutTask, POSE_DEPTH_VERSIONS
from cutting_world_model.pose_cut import gap_from_pose
from cutting_world_model.pose_depth import (PROGRESS_SEMANTICS, BOARD_LATCH_GAP_M, PoseDepthStateSpec,
                                            PoseDepthV3StateSpec, latched_interval_peak)
from cutting_world_model.world_model import CuttingWorldModel

# Board below the EE along +y; the UR-like test chain (z-axis revolute joints) moves the EE in x/y.
GEOMETRY = dict(vertices_in_ee=[[0, 0, 0]], board_origin_base=[0, -1e-3, 0],
                board_normal_base=[0, 1, 0], stop_gap_m=1e-6)


def model(version='pose_depth_v3', backend='mujoco_warp_disect', hidden=(12, 12)):
    meta, frame = fixture()
    task = config()
    task['version'] = version
    task.pop('initial_progress_fraction')
    # Bent arm (nonsingular in x/y), EE at y=0 with the q=0 orientation.
    frame.update(joint_q=np.array([.5, -1., .5, 0., 0., 0.]))
    frame.update(cut_progress=np.array([0.]), depth_span=np.array([1e-3 - 1e-6]),
                 step_count=np.array([0.]), observation=np.zeros(25))
    if version == 'pose_depth_v3':
        frame.update(last_wrench=np.zeros(6), board_contact=np.zeros(1))
    meta.update(free_cut_task=task, goal_dim=8, physical_backend=backend,
                progress_semantics=PROGRESS_SEMANTICS, board_pose_geometry=copy.deepcopy(GEOMETRY))
    meta = ControlStateSpec.metadata_from_frame(meta, frame)
    torch.manual_seed(0)
    wm = CuttingWorldModel(meta, hidden=hidden).double()
    state = wm.spec.pack(frame, dtype=torch.double)[None]
    pose = wm.spec.fk(wm.spec.field(state, 'joint_q'))
    state = put(wm.spec, state, previous_ee_pose=pose)
    goal = torch.cat((pose, torch.tensor([[.03]], dtype=torch.double)), -1)
    return wm, state, goal


def put(spec, state, **values):
    """Overwrite current-frame fields of a single-row packed state."""
    state = state.clone()
    for name, value in values.items():
        state[..., spec.layout[name]] = torch.as_tensor(value, dtype=state.dtype).reshape(-1)
    return state


def joints_for(spec, q0, dx, dy):
    """Joint angles moving the EE by (dx, dy) in the base frame, joints 0/1 only."""
    target = spec.fk(q0)[0, :2] + q0.new_tensor([dx, dy])
    v = q0[0, :2].clone()
    for _ in range(50):
        f = lambda x: spec.fk(torch.cat((x, q0[0, 2:]))[None])[0, :2]
        jac = torch.autograd.functional.jacobian(f, v)
        v = v - torch.linalg.solve(jac, f(v) - target)
    q = q0.clone()
    q[0, :2] = v
    return q


def joint_delta(spec, state, q):
    delta = torch.zeros(1, spec.delta_dim, dtype=state.dtype)
    delta[:, spec.delta_slices['joint_q']] = (q - spec.field(state, 'joint_q')) / spec.out_scale[spec.delta_slices['joint_q']]
    return delta


def gap(spec, state):
    return gap_from_pose(spec.fk(spec.field(state, 'joint_q')), spec.board_geometry)


def test_family_and_layout_and_v2_artifacts_are_refused():
    assert POSE_DEPTH_VERSIONS == ('pose_depth_v2', 'pose_depth_v3')
    v3, s3, _ = model()
    v2, s2, _ = model('pose_depth_v2')
    sp = v3.spec
    assert type(sp) is PoseDepthV3StateSpec and type(v2.spec) is PoseDepthStateSpec
    assert sp.physical_fields[-1] == 'last_wrench' and sp.history_fields[-2:] == ('board_contact', 'control_history')
    assert 'last_wrench' not in v2.spec.layout and 'board_contact' not in v2.spec.layout
    lw = sp.layout['last_wrench']
    assert set(range(lw.start, lw.stop)) <= set(sp.dynamics_indices.tolist())
    torch.testing.assert_close(sp.out_scale[sp.delta_slices['last_wrench']], sp.force_limits.double())
    assert sp.layout['board_contact'].start not in sp.dynamics_indices.tolist()
    # v2 metadata relabelled as v3 has no latch fields.
    bad = copy.deepcopy(v2.spec.metadata)
    bad['free_cut_task']['version'] = 'pose_depth_v3'
    with pytest.raises(ValueError, match='last_wrench'):
        PoseDepthV3StateSpec(bad)
    with pytest.raises(ValueError, match='pose_depth_v3 task'):
        PoseDepthV3StateSpec(v2.spec.metadata)
    with pytest.raises(RuntimeError):
        v3.load_state_dict(v2.state_dict())
    from cutting_world_model.policy import SliceItCritic
    with pytest.raises(RuntimeError):
        SliceItCritic(sp.state_dim, 8).load_state_dict(SliceItCritic(v2.spec.state_dim, 8).state_dict())
    from cutting_world_model.sequence_replay import SequenceReplay
    old = SequenceReplay(1, 4, v2.spec.state_dim, 6, 8, 'cpu')
    with pytest.raises(ValueError, match='shape differs'):
        SequenceReplay(1, 4, sp.state_dim, 6, 8, 'cpu').load_state_dict(old.state_dict())


def test_first_contact_latches_with_physical_criterion():
    wm, s, g = model()
    sp = wm.spec
    down = joints_for(sp, sp.field(s, 'joint_q'), 0., -2e-3)  # would end 1 mm below the board
    fields = sp.integrate_physical(s, joint_delta(sp, s, down))
    end = gap_from_pose(sp.fk(fields['joint_q']), GEOMETRY)
    assert GEOMETRY['stop_gap_m'] - 1e-9 <= end.item() <= BOARD_LATCH_GAP_M
    assert fields['board_contact'].item() == 1
    assert not fields['joint_qd'].any()
    # Near the board but not limited (ends 15 um above): no latch, motion kept.
    near = joints_for(sp, sp.field(s, 'joint_q'), 0., -(1e-3 - 1.5e-5))
    fields = sp.integrate_physical(s, joint_delta(sp, s, near))
    assert gap_from_pose(sp.fk(fields['joint_q']), GEOMETRY).item() == pytest.approx(1.5e-5, abs=1e-9)
    assert fields['board_contact'].item() == 0
    # Criterion itself: limited AND end gap <= 10 um, inclusive.
    base = sp.integrate_physical(s, torch.zeros(1, sp.delta_dim, dtype=s.dtype))
    for limited, end_gap, expected in ((True, 1e-5, 1.), (True, 1.1e-5, 0.), (False, 5e-6, 0.), (True, 1e-6, 1.)):
        out = sp.apply_board_latch(s, dict(base), torch.tensor([limited]), torch.tensor([end_gap], dtype=s.dtype))
        assert out['board_contact'].item() == expected


def latched_state(sp, s, last=(5., -1., 0., 0., .1, 0.)):
    pose = sp.fk(sp.field(s, 'joint_q'))
    return put(sp, s, board_contact=[1.], last_wrench=list(last), peak_wrench=[20., 0, 0, 0, 0, 0],
               wrench_history=torch.arange(18.).reshape(3, 6) / 10, joint_qd=[.3] * 6,
               cut_progress=[.4], previous_ee_pose=pose)


def test_latched_state_is_frozen_and_lateral_or_upward_motion_has_no_effect():
    wm, s, g = model()
    sp = wm.spec
    q0 = sp.field(s, 'joint_q')
    move = joints_for(sp, q0, -1e-2, 1e-3)  # 10 mm lateral and 1 mm up
    delta = joint_delta(sp, s, move)
    delta[:, sp.delta_slices['joint_qd']] = .7
    delta[:, sp.delta_slices['wrench_history']] = .3
    delta[:, sp.delta_slices['last_wrench']] = .2
    delta[:, sp.delta_slices['peak_wrench']] = .9
    free = sp.integrate_physical(s, delta)
    shift = sp.fk(free['joint_q'])[0, :2] - sp.fk(q0)[0, :2]
    assert shift[0].item() < -9e-3 and shift[1].item() > 9e-4  # the unlatched step does move
    held = latched_state(sp, s)
    out = sp.integrate_physical(held, delta)
    assert torch.equal(out['joint_q'], sp.field(held, 'joint_q'))
    assert not out['joint_qd'].any()
    assert torch.equal(out['wrench_history'], sp.field(held, 'wrench_history'))
    assert torch.equal(out['last_wrench'], sp.field(held, 'last_wrench'))
    assert torch.equal(out['peak_wrench'], sp.field(held, 'last_wrench').abs())
    assert torch.equal(out['cut_progress'], sp.field(held, 'cut_progress'))
    assert out['board_contact'].item() == 1
    # Through the full learned step with a nonzero network output.
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=.1)
    nxt, reward, _ = wm.step(held, torch.full((1, 6), .9, dtype=s.dtype), g)
    assert torch.equal(sp.field(nxt, 'joint_q'), sp.field(held, 'joint_q'))
    assert not sp.field(nxt, 'previous_velocity').any()  # matches physical: frozen pose, zero velocity
    assert torch.equal(sp.field(nxt, 'wrench_history'), sp.field(held, 'wrench_history'))
    assert sp.field(nxt, 'step_count').item() == sp.field(held, 'step_count').item() + 1
    assert torch.isfinite(reward).all()


def test_peak_rule_first_contact_keeps_peak_then_frozen_reading():
    wm, s, g = model()
    sp = wm.spec
    down = joints_for(sp, sp.field(s, 'joint_q'), 0., -2e-3)
    delta = joint_delta(sp, s, down)
    limits = sp.force_limits.double()
    delta[:, sp.delta_slices['peak_wrench']] = torch.tensor([20., 0, 0, 0, 0, 0], dtype=s.dtype) / limits
    delta[:, sp.delta_slices['last_wrench']] = torch.tensor([5., 0, 0, 0, 0, 0], dtype=s.dtype) / limits
    assert (limits[0] > 20).item()
    fields = sp.integrate_physical(s, delta)
    assert fields['board_contact'].item() == 1
    assert fields['peak_wrench'][0, 0].item() == pytest.approx(20)  # in-step peak of the latch step
    state, _, _ = sp.derive(s, fields, torch.zeros(1, 6, dtype=s.dtype), g)
    for _ in range(3):
        fields = sp.integrate_physical(state, delta)
        assert fields['peak_wrench'][0, 0].item() == pytest.approx(5)  # never 20, never 0
        state, _, _ = sp.derive(state, fields, torch.zeros(1, 6, dtype=s.dtype), g)


@pytest.mark.parametrize('arrive', [True, False])
def test_latched_clock_and_counter_continue_to_success_or_timeout(arrive):
    wm, s, g = model()
    sp = wm.spec
    held = latched_state(sp, s)
    if not arrive:
        g = g.clone()
        g[:, 0] += .05  # 50 mm away: never arrives
        held = put(sp, held, step_count=[sp.free_task.horizon - 3.])
    q = sp.field(held, 'joint_q')
    state, ended, steps = held, False, 0
    zero_delta = torch.zeros(1, sp.delta_dim, dtype=s.dtype)
    while not ended:
        state, _, done = sp.derive(state, sp.integrate_physical(state, zero_delta),
                                   torch.zeros(1, 6, dtype=s.dtype), g)
        ended = bool(done)
        steps += 1
        assert torch.equal(sp.field(state, 'joint_q'), q) and steps < 50
    failure, timeout = sp.hard_events(state)
    assert not failure.any()
    assert bool(sp.pose_success(state)) == arrive and bool(timeout) == (not arrive)
    if arrive:
        assert steps <= sp.heads.env['successes_threshold'] + 2


def test_rule_function_latch_interval_and_collision():
    limits = torch.full((3, 6), 30., dtype=torch.double)
    reading = torch.zeros(3, 6, dtype=torch.double)
    reading[:, 0] = torch.tensor([5., -5., 40.])
    latched = torch.tensor([True, True, False])
    # Latch interval: in-interval peak 20 N already contains the frozen 5 N sample.
    peak = torch.zeros(3, 6, dtype=torch.double)
    peak[:, 0] = 20.
    p, c = latched_interval_peak(peak, reading, latched, torch.zeros(3, dtype=torch.bool), limits)
    assert p[:, 0].tolist() == [20., 20., 20.] and not c.any()
    # Later intervals start from zero: |frozen reading|, never zero, never 20.
    p, c = latched_interval_peak(torch.zeros_like(peak), reading, latched, torch.zeros(3, dtype=torch.bool), limits)
    assert p[:, 0].tolist() == [5., 5., 0.] and not c.any()
    over = reading.clone()
    over[0, 1] = -31.
    p, c = latched_interval_peak(torch.zeros_like(peak), over, latched, torch.tensor([False, True, False]), limits)
    assert c.tolist() == [True, True, False] and p[1].abs().sum() == 0  # collided world untouched


class Metrics:
    def zero_(self): pass
    def check_wrench(self, *args): pass
    def accumulate(self, *args, **kwargs): pass
    def as_dict(self): return {}


def disect_stub(reading):
    return SimpleNamespace(device='cpu', dtype=torch.float64, worlds=len(reading), peak=torch.full_like(reading, 99.),
                           metrics=Metrics(), mapper=lambda *a: {}, target=None, frames={'tip': None},
                           chain=SimpleNamespace(tip='tip'), task={'environment': {'agent_control_dt': .05}},
                           feedback={'observation_wrench': reading.clone()},
                           force_limits=torch.full_like(reading, 30.),
                           board=SimpleNamespace(contacted=torch.ones(len(reading), dtype=torch.bool)),
                           evaluation_paused=None, force_model='net', frame=lambda: {})


def blademaster_stub(reading, v3, spec):
    n = len(reading)
    pose = spec.fk(torch.zeros(n, 6, dtype=torch.double))
    return SimpleNamespace(device='cpu', dtype=torch.float64, worlds=n, peak=torch.full_like(reading, 99.),
                           metrics=Metrics(), mapper=lambda *a: {}, target=None, frames={'tip': pose},
                           chain=SimpleNamespace(tip='tip'), task={'environment': {'agent_control_dt': .05}},
                           feedback={'sensor_wrench': reading.clone()}, contacted=torch.ones(n, dtype=torch.bool),
                           force_limits=torch.full_like(reading, 30.), board_latch_v3=v3, depth_task=True,
                           board_geometry=GEOMETRY, reward_depth_span=torch.full((n, 1), 1e-3, dtype=torch.double),
                           progress=torch.zeros(n, 1, dtype=torch.double), frame=lambda: {})


def test_both_backends_report_frozen_reading_after_latch_and_settle_identically():
    from cutting_gpu_backend.environment import CuttingBatch
    from cutting_world_model.blademaster_backend import BladeMasterBatch
    wm, s, g = model()
    sp = wm.spec
    reading = torch.tensor([[5., -2., 1., 0., -.1, 0.], [-12., 0., 0., .2, 0., 0.]], dtype=torch.double)
    action = torch.zeros(2, 6, dtype=torch.double)
    disect = disect_stub(reading)
    _, c_disect, _ = CuttingBatch.step(disect, action)
    bm = blademaster_stub(reading, True, sp)
    _, c_bm, _ = BladeMasterBatch.step(bm, action)
    torch.testing.assert_close(disect.peak, reading.abs(), atol=0, rtol=0)  # existing DiSECt behaviour
    torch.testing.assert_close(bm.peak, disect.peak, atol=0, rtol=0)
    assert torch.equal(c_disect, c_bm) and not c_bm.any()
    # pose_depth_v2 BladeMaster is unchanged: latched worlds contribute nothing.
    old = blademaster_stub(reading, False, sp)
    BladeMasterBatch.step(old, action)
    assert not old.peak.any()
    task = FreeCutTask(sp.free_task.config, sp.heads)
    obs = torch.zeros(2, 25, dtype=torch.double)
    speed = torch.full((2,), .03, dtype=torch.double)
    progress = torch.zeros(2, dtype=torch.double)
    costs = [task.step_reward(progress, progress, obs, p, speed) for p in (disect.peak, bm.peak)]
    torch.testing.assert_close(costs[0], costs[1], atol=0, rtol=0)
    board = dict(knife_min_board_gap=torch.full((2, 1), 1e-6), knife_board_contact=torch.ones(2, 1, dtype=torch.bool))
    events = [task.events(obs, p, None, torch.ones(2, dtype=torch.long), c, pose=torch.zeros(2, 7, dtype=torch.double),
                          board=board, success_counter=torch.zeros(2, dtype=torch.double))
              for p, c in ((disect.peak, c_disect), (bm.peak, c_bm))]
    assert all(torch.equal(events[0][k], events[1][k]) for k in ('collision', 'success', 'timeout', 'terminated'))
    # Same frozen reading above the limit settles as a force failure in both rules.
    high = reading.clone()
    high[0, 0] = 31.
    _, c1, _ = CuttingBatch.step(disect_stub(high), action)
    _, c2, _ = BladeMasterBatch.step(blademaster_stub(high, True, sp), action)
    assert c1.tolist() == c2.tolist() == [True, False]


class StubPhysics:
    def __init__(self, key, initial, q, target):
        self.force_model = 'net'
        self.key, self.initial, self.q, self.target = key, initial, q, target
        self.feedback = {key: torch.zeros_like(initial)}
        self.contacted = torch.zeros(len(initial), dtype=torch.bool)

    def reset(self, mask):
        self.feedback[self.key][mask] = self.initial[mask]
        self.contacted[mask] = False
        return self.frame()

    def frame(self):
        n = len(self.q)
        return dict(joint_q=self.q.clone(), joint_qd=torch.zeros_like(self.q), target_pose=self.target.clone(),
                    wrench_history=torch.zeros(n, 3, 6, dtype=torch.double),
                    peak_wrench=torch.zeros(n, 6, dtype=torch.double), cut_spring_ke=torch.ones(n, 2),
                    knife_board_contact=self.contacted[:, None].clone(), cut_progress=torch.zeros(n, 1),
                    depth_span=torch.ones(n, 1, dtype=torch.double), motion_command=torch.zeros(n, 6))


@pytest.mark.parametrize('backend', ['mujoco_warp_disect', 'mujoco_warp_blademaster'])
def test_partial_reset_clears_only_selected_latches_and_reads_initialised_sensor(backend):
    from cutting_world_model.gpu_env import GPUEnvironment
    from cutting_world_model.blademaster_backend import BladeMasterEnvironment
    wm, s, g = model(backend=backend)
    sp = wm.spec
    n = 3
    key = 'sensor_wrench' if backend == 'mujoco_warp_blademaster' else 'observation_wrench'
    initial = torch.arange(1., 19., dtype=torch.double).reshape(n, 6)  # nonzero settled readings
    q = torch.zeros(n, 6, dtype=torch.double)
    physics = StubPhysics(key, initial, q, sp.fk(q))
    env = object.__new__(BladeMasterEnvironment if backend == 'mujoco_warp_blademaster' else GPUEnvironment)
    meta = copy.deepcopy(sp.metadata)
    env.__dict__.update(physics=physics, metadata=meta, worlds=n, device=torch.device('cpu'),
                        fk=sp.fk.double(), heads=sp.heads, environment=meta['environment'],
                        free_task=FreeCutTask(meta['free_cut_task'], sp.heads), previous_pose=sp.fk(q),
                        previous_velocity=torch.zeros(n, 6, dtype=torch.double),
                        previous_acceleration=torch.zeros(n, 6, dtype=torch.double),
                        success_counter=torch.zeros(n, dtype=torch.double), step_count=torch.zeros(n, dtype=torch.long),
                        needs_reset=torch.zeros(n, dtype=torch.bool), current=None,
                        depth_span=torch.ones(n, 1, dtype=torch.double), start_clearance_m=None,
                        target_speed=torch.full((n,), .03, dtype=torch.double), condition_sampler=None)
    frame = env.reset()
    torch.testing.assert_close(frame['last_wrench'], initial, atol=0, rtol=0)
    assert not frame['board_contact'].any() and frame['board_contact'].shape == (n, 1)
    # Every world latches with a frozen reading during an episode.
    frozen = -3 * initial
    physics.contacted[:] = True
    physics.feedback[key] = frozen.clone()
    env.step_count += 4
    env.current = env._observe(physics.frame(), torch.zeros(n, 6))
    assert env.current['board_contact'].all()
    torch.testing.assert_close(env.current['last_wrench'], frozen, atol=0, rtol=0)
    frame = env.reset(torch.tensor([True, False, True]))
    assert frame['board_contact'][:, 0].tolist() == [0., 1., 0.]
    torch.testing.assert_close(frame['last_wrench'][[0, 2]], initial[[0, 2]], atol=0, rtol=0)
    torch.testing.assert_close(frame['last_wrench'][1], frozen[1], atol=0, rtol=0)
    assert env.step_count.tolist() == [0, 4, 0]
    # The packed WM state reads the same fields.
    state = sp.pack_batch({k: frame[k].double() for k in sp.base_fields})
    assert sp.field(state, 'board_contact')[:, 0].tolist() == [0., 1., 0.]


def transitions(sp, s, contact):
    """(H+1, n, D) recorded states with the given per-step latch flags."""
    rows = []
    for t, flags in enumerate(contact):
        row = torch.cat([put(sp, s, board_contact=[float(f)], joint_q=[.001 * t] * 6) for f in flags])
        rows.append(row)
    return torch.stack(rows)


def test_dynamics_loss_masks_latched_starts_and_keeps_first_contact():
    wm, s, g = model()
    sp = wm.spec
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=.01)
    states = transitions(sp, s, [(0, 0), (1, 0), (1, 1), (1, 1)])
    actions = torch.zeros(3, 2, 6, dtype=s.dtype)
    goal = g.repeat(2, 1)
    done = torch.zeros(3, 2, dtype=s.dtype)
    valid = torch.ones(3, 2, dtype=torch.bool)
    loss, metrics = wm.window_loss(states, actions, goal, done, valid)
    assert metrics['dynamics_fit_count'].item() == 3 and metrics['dynamics_masked_count'].item() == 3
    assert metrics['metric_transition_count'].item() == 6
    # Masked targets (latched starts) do not enter the loss; the first-contact target does.
    changed = states.clone()
    changed[2, 0, sp.layout['joint_q']] += .5   # target of world 0, t=1 (latched start)
    changed[3, :, sp.layout['joint_q']] += .5   # targets of t=2 (latched starts)
    assert torch.equal(wm.window_loss(changed, actions, goal, done, valid)[0], loss)
    changed = states.clone()
    changed[1, 0, sp.layout['joint_q']] += .5   # world 0 first-contact transition target
    assert wm.window_loss(changed, actions, goal, done, valid)[0] != loss
    changed = states.clone()
    changed[2, 1, sp.layout['joint_q']] += .5   # world 1 first contact at t=1
    assert wm.window_loss(changed, actions, goal, done, valid)[0] != loss
    # The terminal BCE term is unchanged by masking (same as with every start unlatched).
    free = transitions(sp, s, [(0, 0)] * 4)
    assert torch.equal(wm.window_loss(free, actions, goal, done, valid)[1]['term_bce'], metrics['term_bce'])


@pytest.mark.parametrize('route', ['dynamics', 'head'])
def test_all_latched_batch_is_finite_and_does_not_update_dynamics(route):
    from cutting_world_model.policy import SliceItCritic
    from cutting_world_model.trainer import OnlineLearner
    from third_party.supermimic.trainer_config import WMTrainerCfg
    wm, s, g = model()
    sp = wm.spec
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=.01)
    valid = torch.ones(2, 2, dtype=torch.bool)

    def batch(flags):
        return SimpleNamespace(states=transitions(sp, s, flags), a_exec=torch.full((2, 2, 6), .3, dtype=s.dtype),
                               xi=g.repeat(2, 1), done=torch.tensor([[0., 0.], [0., 1.]], dtype=s.dtype),
                               valid_steps=valid)
    latched = batch([(1, 1)] * 3)
    wm.terminal_gradient = route
    loss, metrics = wm.window_loss(latched.states, latched.a_exec, latched.xi, latched.done, valid)
    assert torch.isfinite(loss) and metrics['state_loss'].item() == 0 and metrics['dynamics_fit_count'].item() == 0
    loss.backward()
    assert all(p.grad is None or not p.grad.any() for p in wm.dynamics.parameters())
    assert any(p.grad is not None and p.grad.any() for p in wm.term_head.parameters())
    wm.zero_grad(set_to_none=True)
    cfg = WMTrainerCfg(compile=None, lr=1e-3, weight_decay=.1, window=2, batch_size=2, batches_per_update=1)
    for flags, moves in (([(1, 1)] * 3, False), ([(0, 0), (1, 0), (1, 1)], True)):
        learner = OnlineLearner(wm, torch.nn.Linear(1, 1), SliceItCritic(sp.state_dim, 8), microbatch=1,
                                wm_cfg=cfg, wm_terminal_gradient=route)
        before = [p.detach().clone() for p in wm.dynamics.parameters()]
        head = [p.detach().clone() for p in wm.term_head.parameters()]
        result = learner.fit_world_model(SimpleNamespace(sample_windows=lambda *a, f=flags: batch(f)))
        changed = any(not torch.equal(a, b) for a, b in zip(before, wm.dynamics.parameters()))
        assert changed == moves
        assert any(not torch.equal(a, b) for a, b in zip(head, wm.term_head.parameters()))
        assert np.isfinite(result['loss'])
        assert result['dynamics_fit_count'] == (0 if not moves else 3)
        assert result['dynamics_masked_count'] == 4 - result['dynamics_fit_count']
        assert result.get('dynamics_skipped_batches', 0) == int(not moves)
        assert all(torch.isfinite(p).all() for p in wm.parameters())


def test_v2_and_v3_share_rules_for_unlatched_states():
    """Same unlatched input: v3 predicts the same v2 fields, reward and termination."""
    v3, s3, g = model()
    v2, s2, _ = model('pose_depth_v2')
    down = joints_for(v3.spec, v3.spec.field(s3, 'joint_q'), -1e-3, -2e-3)
    a = v2.spec.integrate_physical(s2, joint_delta(v2.spec, s2, down))
    b = v3.spec.integrate_physical(s3, joint_delta(v3.spec, s3, down))
    for name in a:
        if name != 'control_history':
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
    action = torch.full((1, 6), .2, dtype=s3.dtype)
    sa, ra, da = v2.spec.derive(s2, a, action, g)
    sb, rb, db = v3.spec.derive(s3, b, action, g)
    torch.testing.assert_close(ra, rb, atol=0, rtol=0)
    assert torch.equal(da, db)
    for name in v2.spec.base_fields:
        torch.testing.assert_close(v2.spec.field(sa, name), v3.spec.field(sb, name), atol=0, rtol=0)


def test_analytic_surrogate_applies_the_same_latch():
    from cutting_world_model.shared_baselines import AnalyticFDCCModel
    wm, s, g = model()
    meta = copy.deepcopy(wm.spec.metadata)
    meta['robot_xml'] = meta['robot_xml'].replace('</joint>', '<limit lower="-3" upper="3" velocity="2"/></joint>')
    labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    params = {'compliance_ref_link': 'l6', 'ft_sensor_ref_link': 'l6',
              'stiffness': dict(zip(labels, [1000] * 3 + [40] * 3)),
              'solver': {'iterations': 1, 'forward_dynamics': {'link_mass': .1}, 'error_scale': .015},
              'pd_gains': {k: {'p': .1, 'd': 0} for k in labels}, 'force': {'hand_frame_control': False}}
    params['stiffness'].update(dict(zip(['sel_x', 'sel_y', 'sel_z', 'sel_ax', 'sel_ay', 'sel_az'], [1, 1, .5, 1, .5, 1])))
    params['stiffness']['use_parallel_force_position_control'] = True
    analytic = AnalyticFDCCModel(meta, params, substeps=2)
    sp = analytic.spec
    assert type(sp) is PoseDepthV3StateSpec
    held = latched_state(sp, s).float()
    goal = g.float().clone()
    goal[:, 0] += .02
    predicted, reward, logits = analytic.predict(held, torch.ones(1, 6), goal)
    assert torch.equal(sp.field(predicted, 'joint_q'), sp.field(held, 'joint_q'))
    assert sp.field(predicted, 'board_contact').item() == 1
    torch.testing.assert_close(sp.field(predicted, 'peak_wrench'), sp.field(held, 'last_wrench').abs())
    free = put(sp, held, board_contact=[0.])
    moved, _, _ = analytic.predict(free, torch.ones(1, 6), goal)
    assert not torch.equal(sp.field(moved, 'joint_q'), sp.field(free, 'joint_q'))


class FixedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(1))

    def forward(self, history, goal):
        return torch.full((goal.shape[0], 6), .2, dtype=goal.dtype) + 0 * self.unused


def survival_fixture(version):
    from cutting_world_model.policy import SliceItCritic
    from cutting_world_model.trainer import OnlineLearner
    from third_party.supermimic.trainer_config import WMTrainerCfg
    wm, s, g = model(version)
    sp = wm.spec
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=1e-3)
    torch.nn.init.zeros_(wm.term_head.net[-1].weight)
    torch.nn.init.constant_(wm.term_head.net[-1].bias, 3.)  # learned head says "ending" (p~0.95)
    goal = g.clone()
    goal[:, 0] += .05  # never arrives
    timing = put(sp, s, step_count=[sp.free_task.horizon - 3.])
    if version == 'pose_depth_v3':
        timing = latched_state(sp, timing)
    state = torch.cat((timing, s))
    learner = OnlineLearner(wm, FixedPolicy().double(), SliceItCritic(sp.state_dim, 8).double(),
                            actor_horizon=6, wm_cfg=WMTrainerCfg(compile=None))
    return learner, wm, state, goal.repeat(2, 1)


def test_v3_imagined_survival_is_analytic_absorption_only():
    from cutting_world_model.trainer import discount_weights
    learner, wm, state, goal = survival_fixture('pose_depth_v3')
    with torch.no_grad():
        ro = learner.imagine(state, state.repeat(10, 1, 1), goal)
    assert ro.survival[:, 0].tolist() == [1., 1., 1., 0., 0., 0., 0.]  # timeout on the 3rd step
    assert ro.survival[:, 1].tolist() == [1.] * 7  # learned head p~0.95 does not cut survival
    assert ro.dhat[1:, 0].tolist() == [0., 0., 1., 1., 1., 1.] and not ro.dhat[1:, 1].any()
    assert (ro.model_probabilities[:, 1] > .9).all()  # learned head still logged
    assert (ro.rewards[3:, 0] == 0).all() and ro.rewards[2, 0] != 0
    weights = discount_weights(ro, 1.)
    assert weights[:, 0].tolist() == [1., 1., 1., 0., 0., 0.]  # terminal transition counted once
    from cutting_world_model.mpc import ModelRollout
    rollout = ModelRollout(wm, 1., None)
    rollout.goal = goal[:1]
    aug = torch.cat((state, torch.ones(2, 1, dtype=state.dtype), torch.zeros(2, 2, dtype=state.dtype)), -1)
    survival = []
    with torch.no_grad():
        for t in range(5):
            aug = rollout.dynamics(aug, torch.zeros(2, 6, dtype=state.dtype), t)
            survival.append(aug[:, -3].tolist())
    assert survival == [[1., 1.], [1., 1.], [0., 1.], [0., 1.], [0., 1.]]


def test_v2_imagined_survival_still_uses_learned_head():
    learner, wm, state, goal = survival_fixture('pose_depth_v2')
    assert not hasattr(wm.spec, 'analytic_survival')
    with torch.no_grad():
        ro = learner.imagine(state, state.repeat(10, 1, 1), goal)
    assert ro.survival[1, 1] < .2  # learned p~0.95 still multiplies survival in v2
    torch.testing.assert_close(ro.dhat[1:, 1], ro.model_probabilities[:, 1])
