"""Board continuation in the physical adapters' own step/reset code (current board.motion).

CuttingBatch.step/reset (DiSECt) and BladeMasterBatch.step/reset run unmodified on
stub robot, material and sensor objects (CPU, no CUDA/MuJoCo state). The stub
controller proposes joint targets that move the end effector at a commanded
base-frame velocity; the board geometry, FK, projection, contact flag, force
check and reset logic are the real adapter code. Readings change every 2 ms
sample so freezing or skipped samples are visible.
"""
from types import SimpleNamespace

import pytest
import torch

from test_board_continue import continuing, physical_chain
from test_pose_depth_v3 import put
from cutting_world_model.pose_cut import gap_from_pose

N = 2
LIMIT = 30.
TIP = 'l6'


class Metrics:
    def zero_(self): pass
    def check_wrench(self, *args): pass
    def accumulate(self, *args, **kwargs): pass
    def as_dict(self): return {}


class Mapper:
    def __init__(self):
        self.added_motion_command = torch.zeros(N, 6, dtype=torch.double)

    def __call__(self, *args):
        return {}


class Chain:
    tip = TIP

    def __init__(self, chain, key):
        """Real backend SerialChain; frames keyed as the adapter expects."""
        self.chain, self.key = chain, key
        self.lower, self.upper, self.origin_position = chain.lower, chain.upper, chain.origin_position

    def __call__(self, q, mass_matrix=False):
        frames, jacobian, _ = self.chain(q.double(), mass_matrix=False)
        pose = frames[self.chain.tip]
        return {self.key: pose, 'sensor': pose}, jacobian, None


class Proposal:
    """Joint target moving the EE at a commanded base-frame (vx, vy) through joints 0/1."""

    def __init__(self, fk):
        self.fk = fk
        self.velocity = torch.zeros(N, 2, dtype=torch.double)

    def __call__(self, q, active, dt=.001):
        q = q.double()
        base = self.fk(q)[:, :2]
        jac = torch.stack([(self.fk(q + 1e-7*e)[:, :2] - base) / 1e-7
                           for e in torch.eye(6, dtype=q.dtype)[:2]], -1)
        dq = torch.linalg.solve(jac, (self.velocity*dt)[..., None])[..., 0]
        out = q.clone()
        out[:, :2] += dq
        return torch.where(active[:, None], out, q)


class Sensor:
    """A new reading at every 2 ms sample of an active world; optional spike sample."""

    def __init__(self):
        self.count = torch.zeros(N, dtype=torch.long)
        self.spike = {}
        self.active = torch.ones(N, dtype=torch.bool)

    def next(self):
        self.count += self.active
        reading = torch.zeros(N, 6, dtype=torch.double)
        reading[:, 0] = self.count.double() * .01
        for world, sample in self.spike.items():
            if self.count[world] == sample:
                reading[world, 2] = 50.
        return reading


class Robot:
    def __init__(self, q0):
        self.q = q0.float().clone()
        self.qd = torch.zeros_like(self.q)
        self.initial = self.q.clone()
        self.stream = self.gpu_model = self.data = None

    def step(self, command, active=None):
        active = torch.ones(N, dtype=torch.bool) if active is None else active
        self.q[active] = command.to(self.q)[active]

    def sensor_wrench(self, link):
        return torch.zeros(N, 6, dtype=torch.double)

    def reset(self, mask):
        self.q[mask] = self.initial[mask]
        self.qd[mask] = 0

    def set_joint_positions(self, q, mask=None, velocity=None):
        self.q[mask] = q.to(self.q)[mask]
        self.qd[mask] = 0 if velocity is None else velocity.to(self.qd)[mask]


class Material:
    def __init__(self, pose, sensor):
        self.position = pose[:, :3].float().clone()
        self.rotation = pose[:, 3:].float().clone()
        self.sensor = sensor
        self.advanced = []

    def set_motion(self, twist, desired=None, active=None):
        if desired is not None:
            self.position[active] = desired[active, :3].to(self.position)
            self.rotation[active] = desired[active, 3:].to(self.rotation)

    def advance(self, dt, active=None):
        self.advanced.append(active.clone())
        self.sensor.active = active.clone()

    def reset(self, mask):
        pass


def identity(n=None):
    pose = torch.zeros(7, dtype=torch.double)
    pose[6] = 1
    return pose if n is None else pose.expand(n, -1).clone()


def disect(sp, fk, serial, q0, monkeypatch):
    import mujoco_warp
    from cutting_gpu_backend.knife_board import KnifeBoard
    monkeypatch.setattr(mujoco_warp, 'forward', lambda *a: None)

    class Board:
        register_stop = KnifeBoard.register_stop
        reset = KnifeBoard.reset

        def __init__(self, physics):
            self.physics, self.continuing, self.current_continuation = physics, True, True
            self.config = dict(height_m=0., stop_gap_m=1e-6, motion='project_continue_v2')
            self.stop_gap = sp.board_geometry['stop_gap_m']
            self.contacted = torch.zeros(N, dtype=torch.bool)
            self.covered = torch.zeros(N, 1, dtype=torch.bool)
            self.min_gap = torch.ones(N, dtype=torch.double)
            self.last_mid, self.last_footprint = torch.zeros(N, 1, 3), torch.zeros(N, 2)

        def gap(self):
            return gap_from_pose(self.physics.frames[TIP], sp.board_geometry)

        def midpoints(self): return torch.zeros(N, 1, 3)
        def footprint(self): return torch.zeros(N, 2)
        def observe(self, *args): pass

        def frame(self):
            return {'knife_board_contact': self.contacted[:, None].clone(), 'knife_board_gap': self.gap()[:, None]}

    sensor, proposal, robot = Sensor(), Proposal(fk), Robot(q0)
    chain = Chain(serial, TIP)
    p = SimpleNamespace(device=torch.device('cpu'), dtype=torch.float64, worlds=N, robot=robot,
                        peak=torch.zeros(N, 6, dtype=torch.double), metrics=Metrics(), mapper=Mapper(),
                        target=None, chain=chain, knife_chain=Chain(serial, 'b_bot_knife_sim'),
                        task={'environment': {'agent_control_dt': .05, 'wrench_hist_size': 3}},
                        fresh_sample_peak=True, force_model='net', evaluation_paused=None,
                        force_limits=torch.full((N, 6), LIMIT, dtype=torch.double),
                        wrench_queue=torch.zeros(N, 25, 6, dtype=torch.double),
                        sim_time=torch.zeros(N, dtype=torch.double),
                        motion_initialized=torch.ones(N, dtype=torch.bool),
                        coupling=SimpleNamespace(
                            motion=lambda knife: {'linear_velocity': torch.zeros(N, 3, dtype=torch.double),
                                                  'angular_velocity': torch.zeros(N, 3, dtype=torch.double)},
                            previous_pose_base=torch.zeros(N, 7, dtype=torch.double),
                            disect_from_base=identity(), wrench_filter=SimpleNamespace()),
                        rigid_tool_mount=identity(1), rigid_tool_geometry=sp.board_geometry,
                        rigid_tool_reach_bound=.1, config=SimpleNamespace(sensor_ref_link='sensor'),
                        capture_recurrence=lambda mask: None, restore_recurrence=lambda mask, saved: None,
                        feedback={'observation_wrench': torch.zeros(N, 6, dtype=torch.double),
                                  'sensor_wrench': torch.zeros(N, 6, dtype=torch.double)},
                        _initial_controller={}, controller=SimpleNamespace(), _initial_filter={},
                        _initial_feedback={}, _initial_queue=torch.zeros(N, 25, 6, dtype=torch.double),
                        _initial_previous_knife=torch.zeros(N, 7, dtype=torch.double))
    p.frames = chain(robot.q)[0]
    p.material = Material(p.frames[TIP], sensor)
    p.board = Board(p)
    p.controller_command = lambda arguments, active: proposal(robot.q, active)

    def feedback(raw):
        r = sensor.next()
        return {'observation_wrench': r, 'sensor_wrench': r, 'force_magnitude': r[:, :3].norm(dim=-1),
                'material_wrench_disect': torch.zeros(N, 6, dtype=torch.double)}
    p._feedback = feedback
    p.frame = lambda: {'joint_q': robot.q.clone(), 'peak_wrench': p.peak.clone(), **p.board.frame()}
    from cutting_gpu_backend.environment import CuttingBatch
    return SimpleNamespace(physics=p, sensor=sensor, proposal=proposal, robot=robot, advanced=p.material.advanced,
                           step=lambda: CuttingBatch.step(p, torch.zeros(N, 6, dtype=torch.double))[:2],
                           reset=lambda mask: CuttingBatch.reset(p, mask), contacted=lambda: p.board.contacted)


def blademaster(sp, fk, serial, q0, monkeypatch):
    from cutting_world_model.blademaster_backend import BladeMasterBatch
    sensor, proposal, robot = Sensor(), Proposal(fk), Robot(q0)
    chain = Chain(serial, TIP)
    advanced = []

    def advance(pose, linear, angular, active):
        advanced.append(active.clone())
        sensor.active = active.clone()

    p = SimpleNamespace(device=torch.device('cpu'), dtype=torch.float64, worlds=N, robot=robot,
                        peak=torch.zeros(N, 6, dtype=torch.double), metrics=Metrics(), mapper=Mapper(),
                        target=identity(N), chain=chain,
                        task={'environment': {'agent_control_dt': .05, 'wrench_hist_size': 3}},
                        board_continuing=True, board_current_continuation=True, board_latch_v3=False,
                        contacted=torch.zeros(N, dtype=torch.bool), board_geometry=sp.board_geometry,
                        reach_bound=.1, batched=True, _advance_batched=advance,
                        net=torch.zeros(N, 6, dtype=torch.double), config=SimpleNamespace(sensor_ref_link='sensor'),
                        base_from_material=identity(),
                        filter=SimpleNamespace(history=torch.zeros(N, 3, 6, dtype=torch.double),
                                               zi=torch.zeros(N, 2, 6, dtype=torch.double),
                                               offset=torch.zeros(N, 6, dtype=torch.double),
                                               update=lambda x: sensor.next()),
                        feedback={'sensor_wrench': torch.zeros(N, 6, dtype=torch.double)},
                        wrench_queue=torch.zeros(N, 25, 6, dtype=torch.double),
                        force_limits=torch.full((N, 6), LIMIT, dtype=torch.double),
                        sim_time=torch.zeros(N, dtype=torch.double), depth_task=True,
                        progress=torch.zeros(N, 1, dtype=torch.double),
                        reward_depth_span=torch.full((N, 1), 1e-3, dtype=torch.double),
                        condition_sampler=None, _controller_initial={}, controller=SimpleNamespace(tare_compensation=torch.zeros(N,6,dtype=torch.double)),
                        _filter_initial={}, _reset_batched=lambda mask: None)
    p.frames = chain(robot.q)[0]
    p.tool_pose = lambda: p.frames[TIP].clone()
    p.controller_command = lambda arguments, active: proposal(robot.q, active)
    p.frame = lambda: BladeMasterBatch.frame(p)
    return SimpleNamespace(physics=p, sensor=sensor, proposal=proposal, robot=robot, advanced=advanced,
                           step=lambda: BladeMasterBatch.step(p, torch.zeros(N, 6, dtype=torch.double))[:2],
                           reset=lambda mask: BladeMasterBatch.reset(p, mask), contacted=lambda: p.contacted)


@pytest.fixture(params=['disect', 'blademaster'])
def rig(request, monkeypatch):
    wm, s, g = continuing()
    sp = wm.spec
    fk, serial = physical_chain(sp, with_chain=True)
    q0 = sp.field(s, 'joint_q').expand(N, -1).clone()
    r = (disect if request.param == 'disect' else blademaster)(sp, fk, serial, q0, monkeypatch)
    r.spec, r.state, r.fk = sp, s, fk
    return r


def clearance(r):
    return gap_from_pose(r.fk(r.robot.q.double()), r.spec.board_geometry)


def wm_contact(r):
    """WM board_contact for the recorded joints, zero predicted motion."""
    sp = r.spec
    states = torch.cat([put(sp, r.state, joint_q=q) for q in r.robot.q.double()])
    fields = sp.integrate_physical(states, torch.zeros(N, sp.delta_dim, dtype=states.dtype))
    return fields['board_contact'][:, 0].bool()


def press(r):
    r.proposal.velocity[:] = torch.tensor([0., -.05], dtype=torch.double)  # 2.5 mm/step, board 1 mm below
    frame, collision = r.step()
    assert not collision.any()
    stop = r.spec.board_geometry['stop_gap_m']
    gap = clearance(r)
    assert ((gap >= stop - 1e-8) & (gap <= 1e-5)).all(), gap
    assert frame['knife_board_contact'][:, 0].bool().all()
    assert torch.equal(wm_contact(r), frame['knife_board_contact'][:, 0].bool())
    return frame


def test_contact_does_not_freeze_world_and_only_inward_motion_is_stopped(rig):
    r = rig
    press(r)
    before = r.robot.q.double().clone()
    pose = r.fk(before)
    time, queue, count, samples = (r.physics.sim_time.clone(), r.physics.wrench_queue.clone(),
                                   r.sensor.count.clone(), len(r.advanced))
    # World 0 slides along the board while still pressing; world 1 presses straight down.
    r.proposal.velocity[:] = torch.tensor([[.02, -.01], [0., -.05]], dtype=torch.double)
    frame, collision = r.step()
    assert not collision.any()
    steps = r.advanced[samples:]
    assert len(steps) == 25 and all(a.all() for a in steps), 'material/contact interval froze at the board'
    torch.testing.assert_close(r.physics.sim_time - time, torch.full((N,), .05, dtype=torch.double))
    assert (r.sensor.count - count).tolist() == [25, 25], 'sensor/filter skipped samples after contact'
    assert not torch.equal(queue, r.physics.wrench_queue)
    moved = r.fk(r.robot.q.double()) - pose
    assert moved[0, 0].item() > 3e-4, 'lateral motion suppressed on the board'
    assert not torch.equal(r.robot.q.double()[0], before[0])
    stop = r.spec.board_geometry['stop_gap_m']
    gap = clearance(r)
    assert ((gap >= stop - 1e-8) & (gap <= 1e-5)).all(), 'inward motion was not stopped at the board'
    assert frame['knife_board_contact'][:, 0].bool().all()
    assert torch.equal(wm_contact(r), frame['knife_board_contact'][:, 0].bool())
    # Lift: contact clears in both physics and WM.
    r.proposal.velocity[:] = torch.tensor([0., .02], dtype=torch.double)
    frame, collision = r.step()
    assert (clearance(r) > 9e-4).all()
    assert not frame['knife_board_contact'].any()
    assert not wm_contact(r).any()


def test_force_limit_is_checked_on_every_sample_after_contact(rig):
    r = rig
    press(r)
    r.sensor.spike = {0: int(r.sensor.count[0]) + 10}   # 10th sample of the next interval
    r.proposal.velocity[:] = torch.tensor([0., -.01], dtype=torch.double)
    frame, collision = r.step()
    assert collision.tolist() == [True, False]
    assert r.physics.peak[0, 2].item() == 50.
    # The other world's peak is its last fresh 2 ms reading of the interval.
    assert r.physics.peak[1, 0].item() == pytest.approx(r.sensor.count[1].item() * .01)


def test_partial_reset_clears_only_selected_worlds_and_others_continue(rig):
    r = rig
    press(r)
    kept = r.robot.q[1].clone()
    frame = r.reset(torch.tensor([True, False]))
    assert r.contacted().tolist() == [False, True]
    torch.testing.assert_close(r.robot.q[0], r.robot.initial[0], atol=0, rtol=0)
    torch.testing.assert_close(r.robot.q[1], kept, atol=0, rtol=0)
    assert r.physics.sim_time[0].item() == 0 and r.physics.sim_time[1].item() > 0
    pose = r.fk(r.robot.q.double())
    r.proposal.velocity[:] = torch.tensor([.02, -.01], dtype=torch.double)
    frame, collision = r.step()
    moved = r.fk(r.robot.q.double()) - pose
    assert moved[1, 0].item() > 3e-4, 'unreset world stayed frozen'
    assert frame['knife_board_contact'][:, 0].tolist() == [False, True]
    assert torch.equal(wm_contact(r), frame['knife_board_contact'][:, 0].bool())


@pytest.mark.parametrize("current", [True, False])
def test_reset_recomputes_current_contact_from_restored_pose(rig, current):
    """A valid near-board reset must have the same state flag as the WM."""
    r = rig
    press(r)
    # Save a pose just above the interlock, inside the 10 um contact band.
    # A little lift avoids comparing float32 joint rounding with the stop plane.
    r.proposal.velocity[:] = torch.tensor([0., .0001], dtype=torch.double)
    r.step()
    assert ((clearance(r) > 1e-6) & (clearance(r) < 1e-5)).all()
    r.robot.initial[0] = r.robot.q[0]
    r.proposal.velocity[:] = torch.tensor([0., .02], dtype=torch.double)
    r.step()
    assert not r.contacted().any()
    kept = r.robot.q[1].clone()
    if hasattr(r.physics, 'board'):
        r.physics.board.current_continuation = current
    else:
        r.physics.board_current_continuation = current
    frame = r.reset(torch.tensor([True, False]))
    assert r.contacted().tolist() == [current, False]
    if current:
        assert torch.equal(frame['knife_board_contact'][:, 0].bool(), wm_contact(r))
    torch.testing.assert_close(r.robot.q[1], kept, atol=0, rtol=0)
