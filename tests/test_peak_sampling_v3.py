"""pose_depth_v3 per-step peak / force-limit sampling rule, both physical adapters.

The real CuttingBatch.step and BladeMasterBatch.step run on scripted stubs: every 2 ms
substep produces the next scripted filtered reading (DiSECt: observation wrench;
BladeMaster: filtered sensor wrench). Robot, material and kinematics are inert.
Rule: each reading produced in a control step enters peak and the force-limit check once;
the reading carried over from the previous step is not counted again.
"""
from types import SimpleNamespace

import pytest
import torch

import cutting_world_model.blademaster_backend as bm_module
from cutting_gpu_backend.environment import CuttingBatch
from cutting_gpu_backend.metrics import StepMetrics
from cutting_world_model.blademaster_backend import BladeMasterBatch

SUBSTEPS = 3
DT = SUBSTEPS * .002
LIMIT = 30.


class Script:
    """Scripted readings; ``latch`` = (world, substep index) when the board stop latches."""

    def __init__(self, readings, latch=None):
        self.readings, self.latch, self.produced = readings, latch, 0

    def next(self):
        reading = self.readings[self.produced].clone()
        self.produced += 1
        return reading

    def latches_now(self, n):
        hit = torch.zeros(n, dtype=torch.bool)
        if self.latch is not None and self.produced == self.latch[1]:
            hit[self.latch[0]] = True
        return hit


class DisectStub:
    def __init__(self, script, initial, fresh, board):
        n = len(initial)
        self.script, self.worlds, self.device, self.dtype = script, n, 'cpu', torch.float64
        self.fresh_sample_peak = fresh
        self.force_model, self.evaluation_paused = 'net', None
        self.peak = torch.zeros(n, 6, dtype=torch.double)
        self.metrics = StepMetrics(n, 'cpu')
        self.mapper = lambda *a: {}
        self.target = None
        self.config = SimpleNamespace(sensor_ref_link='link')
        self.task = {'environment': {'agent_control_dt': DT}}
        zero7 = torch.zeros(n, 7, dtype=torch.double)
        self.frames = {'tip': zero7, 'link': zero7}

        def chain(q, mass_matrix=False):
            return ({'tip': zero7, 'link': zero7},)
        chain.tip = 'tip'
        self.chain = chain
        self.knife_chain = lambda q, mass_matrix=False: ({'b_bot_knife_sim': zero7},)
        self.coupling = SimpleNamespace(motion=lambda k: {'linear_velocity': torch.zeros(n, 3, dtype=torch.double),
                                                          'angular_velocity': torch.zeros(n, 3, dtype=torch.double)})
        self.robot = SimpleNamespace(q=torch.zeros(n, 6), step=lambda c, active=None: None,
                                     sensor_wrench=lambda link: torch.zeros(n, 6))
        self.material = SimpleNamespace(position=torch.zeros(n, 3), rotation=torch.zeros(n, 4),
                                        set_motion=lambda *a, **k: None, advance=lambda *a, **k: None)
        self.motion_initialized = torch.zeros(n, dtype=torch.bool)
        self.wrench_queue = torch.zeros(n, 25, 6, dtype=torch.double)
        self.sim_time = torch.zeros(n, dtype=torch.double)
        self.force_limits = torch.full((n, 6), LIMIT, dtype=torch.double)
        self.feedback = self._pack(initial)
        self.board = None
        if board:
            contacted = torch.zeros(n, dtype=torch.bool)

            def register_stop(limited):
                contacted[:] |= script.latches_now(n)
            self.board = SimpleNamespace(contacted=contacted, observe=lambda *a: None, register_stop=register_stop,
                                         fraction=lambda twist, dt, active: (torch.ones(n, dtype=torch.double),
                                                                            torch.zeros(n, dtype=torch.bool)))

    def _pack(self, reading):
        return {'observation_wrench': reading.clone(), 'sensor_wrench': reading.clone(),
                'force_magnitude': reading[:, :3].norm(dim=-1), 'material_wrench_disect': torch.zeros_like(reading)}

    def _feedback(self, raw):
        return self._pack(self.script.next())

    def controller_command(self, args, active):
        return self.robot.q

    def capture_recurrence(self, mask):
        return {k: v[mask].clone() for k, v in self.feedback.items()}

    def restore_recurrence(self, mask, saved):
        for k, v in saved.items():
            self.feedback[k][mask] = v

    def frame(self):
        return {'sim_time': self.sim_time[:, None].clone()}

    def step(self, action):
        return CuttingBatch.step(self, action)


def disect(script, initial, fresh=True, board=False):
    return DisectStub(script, initial, fresh, board)


class BladeStub:
    def __init__(self, script, initial, v3=True):
        n = len(initial)
        self.script, self.worlds, self.device, self.dtype = script, n, 'cpu', torch.float64
        self.board_latch_v3, self.depth_task, self.batched = v3, False, True
        self.peak = torch.zeros(n, 6, dtype=torch.double)
        self.metrics = StepMetrics(n, 'cpu')
        self.mapper = lambda *a: {}
        self.target = None
        pose = torch.zeros(n, 7, dtype=torch.double)
        pose[:, 6] = 1
        self.frames = {'tip': pose, 'link': pose}

        def chain(q, mass_matrix=False):
            return ({'tip': pose, 'link': pose},)
        chain.tip = 'tip'
        self.chain = chain
        self.config = SimpleNamespace(sensor_ref_link='link')
        self.task = {'environment': {'agent_control_dt': DT}}
        self.robot = SimpleNamespace(q=torch.zeros(n, 6), step=lambda c, active=None: None,
                                     set_joint_positions=lambda q, mask=None: None,
                                     sensor_wrench=lambda link: torch.zeros(n, 6))
        self.board_geometry, self.reach_bound, self.base_from_material = None, 1., None
        self.depth_span = None
        self.progress = torch.zeros(n, 1, dtype=torch.double)
        self.contacted = torch.zeros(n, dtype=torch.bool)
        self.net = torch.zeros(n, 6, dtype=torch.double)
        self.feedback = {'sensor_wrench': initial.clone()}
        self.wrench_queue = torch.zeros(n, 25, 6, dtype=torch.double)
        self.sim_time = torch.zeros(n, dtype=torch.double)
        self.force_limits = torch.full((n, 6), LIMIT, dtype=torch.double)
        self.filter = SimpleNamespace(history=torch.zeros(n, 4, 6), zi=torch.zeros(n, 6), offset=torch.zeros(n, 6),
                                      update=lambda x: script.next())

    def tool_pose(self):
        return self.frames['tip'].clone()

    def frame(self):
        return {'sim_time': self.sim_time[:, None].clone()}

    def controller_command(self, args, active):
        return self.robot.q

    def _advance_batched(self, *a):
        pass

    def step(self, action):
        return BladeMasterBatch.step(self, action)


@pytest.fixture
def inert_blademaster(monkeypatch):
    def swept(old_q, new_q, fk, geometry, reach, moving):
        stub = swept.stub
        return torch.ones(len(moving), dtype=torch.double), stub.script.latches_now(len(moving)) & moving
    monkeypatch.setattr(bm_module, 'swept_joint_fraction', swept)
    monkeypatch.setattr(bm_module, 'gap_from_pose', lambda pose, geometry: torch.zeros(len(pose), dtype=torch.double))
    monkeypatch.setattr(bm_module, 'reaction_at_sensor', lambda f, m, b, s: torch.zeros(len(f), 6, dtype=torch.double))
    monkeypatch.setattr(bm_module, 'geometric_progress', lambda pose, g, s, previous: previous)

    def make(script, initial, v3=True):
        stub = BladeStub(script, initial, v3)
        swept.stub = stub
        return stub
    return make


def readings(steps, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(steps * SUBSTEPS, n, 6, generator=g, dtype=torch.double) * 20 - 10)


def run(stub, steps):
    out = []
    for _ in range(steps):
        _, collision, _ = stub.step(torch.zeros(stub.worlds, 6, dtype=torch.double))
        out.append((stub.peak.clone(), collision.clone(), stub.metrics.safety_margin.clone()))
    return out


def carried_largest():
    r = readings(2)
    r[SUBSTEPS - 1, :, 0] = torch.tensor([25., -26.])  # last reading of step 1 is the largest
    return r


def test_v3_peak_excludes_carried_reading_in_both_backends(inert_blademaster):
    r = carried_largest()
    initial = torch.zeros(2, 6, dtype=torch.double)
    for stub in (disect(Script(r, None), initial), inert_blademaster(Script(r, None), initial)):
        (p1, c1, m1), (p2, c2, m2) = run(stub, 2)
        torch.testing.assert_close(p1, r[:SUBSTEPS].abs().amax(0), atol=0, rtol=0)
        torch.testing.assert_close(p2, r[SUBSTEPS:].abs().amax(0), atol=0, rtol=0)
        assert p2[:, 0].max() < 25 and not c1.any() and not c2.any()
        # Safety margin reads the same samples as peak.
        torch.testing.assert_close(m2, (p2 / LIMIT).amax(-1), atol=0, rtol=0)


def test_v2_disect_still_counts_carried_reading():
    r = carried_largest()
    stub = disect(Script(r, None), torch.zeros(2, 6, dtype=torch.double), fresh=False)
    _, (p2, c2, _) = run(stub, 2)
    expected = r[SUBSTEPS - 1:].abs().amax(0)  # carried reading + this step's readings
    torch.testing.assert_close(p2, expected, atol=0, rtol=0)
    assert p2[:, 0].tolist() == [25., 26.] and not c2.any()


def test_violation_on_last_substep_detected_in_both_backends(inert_blademaster):
    r = readings(2)
    r[-1, 0, 4] = -31.
    initial = torch.zeros(2, 6, dtype=torch.double)
    results = []
    for stub in (disect(Script(r, None), initial), inert_blademaster(Script(r, None), initial)):
        (_, c1, _), (p2, c2, m2) = run(stub, 2)
        assert not c1.any() and c2.tolist() == [True, False]
        assert p2[0, 4] == 31. and m2[0] == 31. / LIMIT
        results.append((p2, c2, m2))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_violation_mid_step_stops_world_at_that_sample(inert_blademaster):
    r = readings(1)
    r[1, 1, 0] = 40.   # world 1 exceeds at substep 2 of 3
    r[2, 1, 0] = 99.   # later sample must not be counted for the stopped world
    initial = torch.zeros(2, 6, dtype=torch.double)
    for stub in (disect(Script(r, None), initial), inert_blademaster(Script(r, None), initial)):
        ((p, c, m),) = run(stub, 1)
        assert c.tolist() == [False, True]
        torch.testing.assert_close(p[1], r[:2, 1].abs().amax(0), atol=0, rtol=0)
        torch.testing.assert_close(p[0], r[:, 0].abs().amax(0), atol=0, rtol=0)
        assert stub.sim_time.tolist() == pytest.approx([.006, .004])


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_same_readings_same_peak_and_collision_with_latch(inert_blademaster, seed):
    """World 0 latches at the 2nd substep of step 1 (its reading is then frozen)."""
    r = readings(3, seed=seed)
    r[1, 0, 2] = 12.
    r[2, 1, 3] = 18. if seed else 33.  # seed 0: world 1 violates on the last substep of step 1
    initial = torch.full((2, 6), .5, dtype=torch.double)
    a = run(disect(Script(r, (0, 1)), initial, board=True), 3)
    b = run(inert_blademaster(Script(r, (0, 1)), initial), 3)
    for step_a, step_b in zip(a, b):
        for x, y in zip(step_a, step_b):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    frozen = r[1, 0].abs()
    torch.testing.assert_close(a[0][0][0], r[:2, 0].abs().amax(0), atol=0, rtol=0)  # latch interval
    torch.testing.assert_close(a[1][0][0], frozen, atol=0, rtol=0)  # later intervals: |frozen|
    torch.testing.assert_close(a[2][0][0], frozen, atol=0, rtol=0)
    if seed:
        torch.testing.assert_close(a[1][0][1], r[SUBSTEPS:2 * SUBSTEPS, 1].abs().amax(0), atol=0, rtol=0)
    else:
        assert a[0][1].tolist() == [False, True]
