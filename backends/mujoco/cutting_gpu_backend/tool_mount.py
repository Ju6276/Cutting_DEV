"""Rigid finite-tool board bound, independent of material backend."""

import torch
from .controller.fdcc import quat_apply


def gap_from_pose(pose, geometry):
    vertices = pose.new_tensor(geometry["vertices_in_ee"])
    q = pose[..., None, 3:].expand(*pose.shape[:-1], len(vertices), 4)
    v = vertices.expand(*pose.shape[:-1], len(vertices), 3)
    points = pose[..., None, :3] + quat_apply(q, v)
    origin = pose.new_tensor(geometry["board_origin_base"])
    normal = pose.new_tensor(geometry["board_normal_base"])
    return ((points - origin) * normal).sum(-1).amin(-1)


@torch.no_grad()
def swept_joint_fraction(old_q, new_q, fk, geometry, reach_bound, active):
    """Conservative finite-blade clearance along a revolute joint interpolation.

    Every link-offset vector rotates with angular speed bounded by sum(abs(dq)).
    Thus ||p''(alpha)|| <= reach_bound * sum(abs(dq))**2. A curve deviates
    from its endpoint chord by at most max||p''||/8. The serial-chain reach
    includes every fixed translation and the most distant blade vertex.
    """
    start = gap_from_pose(fk(old_q), geometry)
    delta = new_q - old_q
    curvature = reach_bound * delta.abs().sum(-1).square() / 8
    stop = geometry["stop_gap_m"]

    def bound(f):
        end = gap_from_pose(fk(old_q + f[:, None] * delta), geometry)
        return torch.minimum(start, end) - curvature * f.square()

    full = torch.ones_like(start)
    hit = active & (bound(full) <= stop)
    lo, hi = torch.zeros_like(start), full.clone()
    if bool(hit.any()):
        for _ in range(30):
            mid = (lo + hi) / 2
            safe = bound(mid) > stop
            lo = torch.where(safe, mid, lo)
            hi = torch.where(safe, hi, mid)
    return torch.where(hit, lo, full), hit


# Machine-checked board-continuation semantics recorded in task.board.motion.
# v1 (f12ca04): flag = projected AND clearance <= 10 um; minimum-norm joint
# projection. Read/evaluate-only: kept so its checkpoints evaluate as recorded.
# v2 (current): flag = clearance <= 10 um (state property, board_contact below).
CONTINUE_BOARD_V1 = 'project_continue_v1'
CONTINUE_BOARD = 'project_continue_v2'
CONTINUING_MOTIONS = (CONTINUE_BOARD_V1, CONTINUE_BOARD)
# Clearance band (m) above the physical board plane that counts as resting on it.
BOARD_CONTACT_GAP_M = 1e-5
# Float32 roundoff below the plane tolerated by the adapters' penetration checks.
PENETRATION_TOLERANCE_M = 2e-6

def continuing_board(config):
    """Any non-latching board (current or the read/evaluate-only v1)."""
    return config.get('motion') in CONTINUING_MOTIONS


def current_board(config):
    """The current continuation semantics (CONTINUE_BOARD)."""
    return config.get('motion') == CONTINUE_BOARD


def board_contact(gap):
    """Continuing-board contact flag: a property of the end-of-step pose alone.

    True iff the finite-blade clearance above the board plane is <= 10 um. It does
    not depend on whether a projection acted during the step, so recorded
    (already projected) states and imagined states evaluate it identically. The
    physical adapters (DiSECt, BladeMaster) and the WM all call this function on
    gap_from_pose of the recorded joint configuration. Historical metadata keeps
    its own rule: latch (projected/shortened AND gap <= 10 um, absorbing) and
    project_continue_v1 (projected AND gap <= 10 um, per step).
    """
    return gap <= BOARD_CONTACT_GAP_M


def project_joint_endpoint(proposed, fk, geometry):
    """Unilateral joint-space projection of a finite blade, NOT a contact force.

    Newton's minimum-norm joint correction removes inward clearance violations.
    Outward and already safe proposals are unchanged. Differentiating the Newton
    iterations retains the constrained-motion gradient for imagined rollouts.
    A tiny positive roundoff margin is numerical clearance, not board compliance.
    """
    tracking = torch.is_grad_enabled()
    stop = geometry['stop_gap_m']
    with torch.enable_grad():
        q = proposed if proposed.requires_grad else proposed.detach().requires_grad_(True)
        limited = gap_from_pose(fk(q), geometry) < stop
        if not bool(limited.any()):
            return proposed, limited
        for _ in range(12):
            gap = gap_from_pose(fk(q), geometry)
            violation = torch.where(limited, (stop + 2e-7 - gap).clamp_min(0), torch.zeros_like(gap))
            if not bool((violation > 1e-10).any()):
                break
            normal = torch.autograd.grad(gap.sum(), q, create_graph=tracking, retain_graph=tracking)[0]
            norm = normal.square().sum(-1)
            if bool(((norm < 1e-14) & (violation > 1e-10)).any()):
                raise FloatingPointError('Board projection has a singular clearance Jacobian')
            q = q + (violation / norm.clamp_min(1e-14))[..., None] * normal
        if bool((gap_from_pose(fk(q), geometry) < stop - 1e-9).any()):
            raise FloatingPointError('Board endpoint projection did not converge')
    return (q if tracking else q.detach()), limited


@torch.no_grad()
def continuing_joint_step(old_q, proposed, fk, geometry, reach_bound, active):
    """Project an inward endpoint, then conservatively check its swept blade.

    Unlike the historical latch, every substep starts a fresh motion proposal.
    Chord-curvature bounds use the physical plane (gap=0); the endpoint retains
    the configured positive stop gap. This permits release from the boundary
    without accepting board penetration. Unsafe swept rotations are shortened.
    """
    projected, inward = project_joint_endpoint(proposed, fk, geometry)
    projected = torch.where(active[..., None], projected, old_q)
    delta = projected - old_q
    start = gap_from_pose(fk(old_q), geometry)
    stop = geometry['stop_gap_m']

    # A recorded start may sit up to the float32 tolerance below the plane; then
    # the swept path must only not go deeper than it starts (lift/lateral allowed).
    floor = start.clamp(-PENETRATION_TOLERANCE_M, 0)

    def safe(f):
        end = gap_from_pose(fk(old_q + f[..., None]*delta), geometry)
        # Each vertex's signed clearance is bounded below by its endpoint chord
        # minus C*t*(1-t); taking endpoint minima remains conservative.
        c = reach_bound * (f*delta.abs().sum(-1)).square() / 2
        t = ((c + start - end) / (2*c).clamp_min(1e-30)).clamp(0,1)
        lower = (1-t)*start + t*end - c*t*(1-t)
        return (end >= stop-1e-8) & (lower >= floor)

    full = torch.ones_like(start)
    shortened = active & ~safe(full)
    fraction = full
    if bool(shortened.any()):
        lo, hi = torch.zeros_like(start), full.clone()
        for _ in range(30):
            mid = (lo+hi)/2
            ok = safe(mid)
            lo=torch.where(ok,mid,lo); hi=torch.where(ok,hi,mid)
        fraction=torch.where(shortened,lo,full)
    result=old_q+fraction[...,None]*delta
    return result, active & (inward | shortened)


def project_joint_velocity(q, velocity, fk, geometry, limited):
    """Remove only inward generalized velocity at a limited endpoint.

    Position-command differences are NOT joint velocity in the source hardware
    model (SetPosition preserves velocity); synthesizing dq/dt injects inertia.
    """
    tracking=torch.is_grad_enabled()
    if not bool(limited.any()):
        return velocity
    with torch.enable_grad():
        point=q if q.requires_grad else q.detach().requires_grad_(True)
        gap=gap_from_pose(fk(point),geometry)
        normal=torch.autograd.grad(gap.sum(),point,create_graph=tracking,retain_graph=tracking)[0]
        inward=(normal*velocity).sum(-1).clamp_max(0)
        corrected=velocity-(inward/normal.square().sum(-1).clamp_min(1e-14))[...,None]*normal
        result=torch.where(limited[...,None],corrected,velocity)
    return result if tracking else result.detach()
