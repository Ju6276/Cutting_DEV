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
# projection. Read/evaluate-only: preserves its endpoint/flag rules. The shared
# swept-start gap fix also applies to v1; evaluation is not bitwise historical.
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


def project_joint_endpoint_v1(proposed, fk, geometry, limits=None, jacobian=None):
    """project_continue_v1 (f12ca04, evaluate-only): unilateral joint-space projection.

    Newton's minimum-norm joint correction along the clearance gradient removes
    inward clearance violations; its end-effector displacement is generally NOT
    along the board normal (tangential slide and rotation). ``limits`` and
    ``jacobian`` are accepted for interface compatibility and ignored, as
    recorded at f12ca04.
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
def continuing_joint_step(old_q, proposed, fk, geometry, reach_bound, active, endpoint=None, limits=None,
                          jacobian=None):
    """Project an inward endpoint, then conservatively check its swept blade.

    Unlike the historical latch, every substep starts a fresh motion proposal.
    Chord-curvature bounds use the physical plane (gap=0); the endpoint retains
    the configured positive stop gap. This permits release from the boundary
    without accepting board penetration. Unsafe swept rotations are shortened.
    ``endpoint`` is the revision's endpoint projection (default: current, normal-only).
    """
    endpoint = project_joint_endpoint if endpoint is None else endpoint
    projected, inward = endpoint(proposed, fk, geometry, limits=limits, jacobian=jacobian)
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


def project_joint_velocity_v1(q, velocity, fk, geometry, limited, jacobian=None):
    """project_continue_v1 (evaluate-only): remove inward generalized velocity
    along the clearance gradient (minimum joint-velocity norm, not normal-only).

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


# ---- project_continue_v2: pure normal translation of the end effector ----

# A board projection that needs more than this joint motion per metre of normal
# lift means an end-effector Jacobian singular value below 1 mm/rad along the
# required twist: refuse instead of producing an arbitrary large joint jump.
MAX_JOINT_PER_LIFT = 1e3
# Tikhonov damping of J J^T in the Newton steps (J entries ~0.01-1 m/rad, rad/rad);
# it only slows convergence (fixed point unchanged) and keeps rank-deficient
# chains (the planar test chains) solvable with a minimum-norm step.
JACOBIAN_DAMPING = 1e-12
# Newton tolerance on the pose residual (m, rad).
POSE_TOLERANCE = 1e-12


def joint_limits(xml, base, tip, joint_names):
    """URDF joint position limits (lower, upper); +-inf for continuous or unlimited joints.

    Same <limit> values the backend SerialChain reads (chain.lower/upper), but also
    accepts chains without <limit> elements (WM test fixtures).
    """
    import xml.etree.ElementTree as ET
    by_child = {j.find('child').get('link'): j for j in ET.fromstring(xml).findall('joint')}
    lower = [-float('inf')]*len(joint_names)
    upper = [float('inf')]*len(joint_names)
    cursor = tip
    while cursor != base:
        joint = by_child[cursor]
        name, kind, limit = joint.get('name'), joint.get('type'), joint.find('limit')
        if kind in ('revolute', 'prismatic') and limit is not None:
            index = joint_names.index(name)
            lower[index] = float(limit.get('lower', '0'))
            upper[index] = float(limit.get('upper', '0'))
        cursor = joint.find('parent').get('link')
    return torch.tensor(lower, dtype=torch.float64), torch.tensor(upper, dtype=torch.float64)


def _pose_residual(pose, target):
    """(position error, rotation-vector-like error) of pose relative to target.

    The orientation part is twice the vector part of target^-1 * pose with the
    sign fixed to w >= 0: zero iff the orientations agree, smooth (polynomial in
    the quaternion) so it can be differentiated twice for imagined BPTT, and its
    Jacobian at the target is the body angular velocity Jacobian.
    """
    from .controller.fdcc import quat_inverse as inverse, quat_mul as mul
    relative = mul(inverse(target[..., 3:]), pose[..., 3:])
    relative = torch.where(relative[..., 3:] < 0, -relative, relative)
    return torch.cat((pose[..., :3]-target[..., :3], 2*relative[..., :3]), -1)


def _twist_solve(jacobian, twist):
    """Damped minimum-norm joint step with J dq = twist."""
    eye = torch.eye(jacobian.shape[-2], dtype=jacobian.dtype, device=jacobian.device)
    system = jacobian @ jacobian.transpose(-1, -2) + JACOBIAN_DAMPING*eye
    return (jacobian.transpose(-1, -2) @ torch.linalg.solve(system, twist[..., None]))[..., 0]


def chain_pose_jacobian(chain):
    """Analytic (tip pose, world geometric Jacobian [Jv; Jw]) of a backend SerialChain.

    Optional fast path for the physical adapters (no gradient tracking): it
    evaluates the same pose residual Jacobian as autograd, from one chain call.
    """
    def evaluate(q):
        frames, jacobian, _ = chain(q.to(chain.origin_position.dtype), mass_matrix=False)
        return frames[chain.tip], jacobian
    return evaluate


def _analytic_residual_jacobian(pose, geometric, target):
    """Jacobian of _pose_residual(pose(q), target) from the world geometric Jacobian.

    Position rows are Jv. With dQ/dq_j = (w_j, 0) * Q / 2 the orientation rows are
    sign * vec(T^-1 * (w_j, 0) * Q), the sign being the residual's w >= 0 choice.
    """
    from .controller.fdcc import quat_inverse as inverse, quat_mul as mul
    angular = geometric[..., 3:, :].transpose(-1, -2)                       # (..., n, 3)
    omega = torch.cat((angular, torch.zeros_like(angular[..., :1])), -1)    # (..., n, 4)
    left = inverse(target[..., 3:])[..., None, :].expand_as(omega)
    right = pose[..., None, 3:].expand_as(omega)
    columns = mul(left, mul(omega, right))[..., :3].transpose(-1, -2)       # (..., 3, n)
    relative_w = mul(inverse(target[..., 3:]), pose[..., 3:])[..., 3]
    sign = torch.where(relative_w < 0, -torch.ones_like(relative_w), torch.ones_like(relative_w))
    return torch.cat((geometric[..., :3, :], sign[..., None, None]*columns), -2)


def _residual_jacobian(fk, q, target, tracking, jacobian=None):
    """Pose residual (target tracked for BPTT) and its Jacobian in q at a fixed target.

    ``jacobian`` (chain_pose_jacobian) replaces autograd when no gradient is tracked.
    """
    if jacobian is not None and not tracking:
        pose, geometric = jacobian(q.detach())
        pose, target = pose.to(q), target.detach()
        return _pose_residual(pose, target), _analytic_residual_jacobian(pose, geometric.to(q), target)
    pose = fk(q)
    fixed = _pose_residual(pose, target.detach())
    rows = [torch.autograd.grad(fixed[..., i].sum(), q, create_graph=tracking, retain_graph=True)[0]
            for i in range(fixed.shape[-1])]
    residual = _pose_residual(pose, target) if tracking else fixed
    return residual, torch.stack(rows, -2)


def _unit_shift(geometry, reference):
    """Translation per metre of clearance along the board normal (n / |n|^2)."""
    normal = reference.new_tensor(geometry['board_normal_base'])
    return normal / normal.square().sum()


def project_joint_endpoint(proposed, fk, geometry, limits=None, jacobian=None):
    """Nonpenetration of a proposed joint endpoint by a pure normal EE translation.

    For a world whose proposed blade clearance is below the stop gap, the result
    is the joint configuration whose end-effector pose equals the proposed pose
    translated along the board normal by exactly the clearance deficit (plus a
    0.2 um float32 margin): zero rotation and zero tangential displacement are
    removed or added. Solved by damped Gauss-Newton on the 6-D pose residual
    with the full end-effector Jacobian (minimum joint norm among exact
    solutions for redundant chains). Outward and already safe proposals are
    returned unchanged. Differentiating the Newton iterations gives the
    constrained-motion gradient for imagined rollouts. NOT a contact force.

    ``limits`` (lower, upper): joint limits re-applied after the correction. If a
    limit binds, the clamped configuration is kept only if it still clears the
    stop (its removed displacement is then no longer purely normal); otherwise a
    FloatingPointError is raised. Near-singular or unreachable normal
    translations also raise FloatingPointError instead of returning a pose with
    tangential/rotational error. ``jacobian``: optional analytic chain Jacobian
    (chain_pose_jacobian) used only without gradient tracking; it changes the
    cost, not the solution (same residual, same convergence tolerance).
    """
    tracking = torch.is_grad_enabled()
    stop = geometry['stop_gap_m']
    with torch.enable_grad():
        # The pose solver verifies nanometre residuals; evaluating Newton in
        # float32 cannot satisfy that contract even for a well-conditioned arm.
        # Promote arithmetic, preserving the cast's gradient to the WM state.
        work = proposed.to(torch.float64) if proposed.dtype in (torch.float16, torch.bfloat16, torch.float32) else proposed
        q0 = work if work.requires_grad else work.detach().requires_grad_(True)
        pose = fk(q0)
        gap = gap_from_pose(pose, geometry)
        limited = gap < stop
        if not bool(limited.any()):
            return proposed, limited
        lift = torch.where(limited, stop + 2e-7 - gap, torch.zeros_like(gap))
        target = torch.cat((pose[..., :3] + lift[..., None]*_unit_shift(geometry, pose), pose[..., 3:]), -1)
        if not tracking:
            target = target.detach()
        q = q0
        mask = limited[..., None]
        # Newton solves the geometric pose in double. The returned WM state
        # is quantized to float32: refine the outward rounding only if its
        # actual FK would cross the stop. Tangential/orientation targets stay
        # unchanged; no task tolerance or penetration threshold is relaxed.
        for refinement in range(8):
            for _ in range(12):
                residual, matrix = _residual_jacobian(fk, q, target, tracking, jacobian)
                residual = torch.where(mask, residual, torch.zeros_like(residual))
                if float(residual.detach().abs().max()) < POSE_TOLERANCE:
                    break
                step = _twist_solve(matrix, residual)
                q = q - torch.where(mask, step, torch.zeros_like(step))
            solve_error = _pose_residual(fk(q), target).detach().abs()
            if bool((solve_error[limited] > 1e-9).any()):
                raise FloatingPointError('Board projection could not realise a pure normal translation '
                                         '(singular end-effector Jacobian along the board normal)')
            cast_gap = gap_from_pose(fk(q.to(proposed.dtype)), geometry)
            needs_rounding = limited & (cast_gap.detach() < stop - 1e-9)
            if not bool(needs_rounding.any()):
                break
            # The quantization correction is not a physical state derivative.
            extra = torch.where(needs_rounding, stop - cast_gap.detach().to(q) + 2e-7,
                                torch.zeros_like(lift))
            lift = lift + extra
            target = torch.cat((target[..., :3] + extra[..., None]*_unit_shift(geometry, pose),
                                target[..., 3:]), -1)
        else:
            raise FloatingPointError('Board projection cannot represent a nonpenetrating state')
        correction = (q - q0).detach().norm(dim=-1)
        if bool((correction > MAX_JOINT_PER_LIFT*lift.detach() + 1e-9).any()):
            raise FloatingPointError('Board projection needs a near-singular end-effector Jacobian '
                                     '(joint correction > 1e3 rad per metre of normal lift)')
        error = _pose_residual(fk(q), target).detach().abs()
        if bool((error[limited] > 1e-9).any()):
            raise FloatingPointError('Board projection could not realise a pure normal translation '
                                     '(singular end-effector Jacobian along the board normal)')
        if limits is not None:
            lower, upper = (x.to(q) for x in limits)
            # Only projected worlds are clamped; others stay exactly as proposed.
            clamped = torch.where(mask, torch.maximum(torch.minimum(q, upper), lower), q)
            if not torch.equal(clamped.detach(), q.detach()):
                q = clamped
                if bool((gap_from_pose(fk(q), geometry)[limited] < stop - 1e-9).any()):
                    raise FloatingPointError('Board projection blocked by a joint limit')
        if bool((gap_from_pose(fk(q), geometry)[limited] < stop - 1e-9).any()):
            raise FloatingPointError('Board endpoint projection did not converge')
        result = torch.where(mask, q.to(proposed.dtype), proposed)
        # Verify the representation actually returned to the caller as well.
        # The existing 0.2 um outward margin covers its joint-angle rounding.
        if bool((gap_from_pose(fk(result), geometry)[limited] < stop - 1e-9).any()):
            raise FloatingPointError('Board projection loses clearance when cast to the state dtype')
    return (result if tracking else result.detach()), limited


def project_joint_velocity(q, velocity, fk, geometry, limited, jacobian=None):
    """Remove only the inward board-normal EE velocity at a limited endpoint.

    The joint-velocity change realises an end-effector twist that is a pure
    translation along the board normal cancelling the inward clearance rate:
    tangential and angular EE velocities are unchanged. Position-command
    differences are NOT joint velocity in the source hardware model
    (SetPosition preserves velocity); synthesizing dq/dt injects inertia.
    """
    tracking = torch.is_grad_enabled()
    if not bool(limited.any()):
        return velocity
    with torch.enable_grad():
        point = q if q.requires_grad else q.detach().requires_grad_(True)
        pose = fk(point)
        gap = gap_from_pose(pose, geometry)
        gradient = torch.autograd.grad(gap.sum(), point, create_graph=tracking, retain_graph=True)[0]
        rate = (gradient*velocity).sum(-1)
        inward = torch.where(limited, rate.clamp_max(0), torch.zeros_like(rate))
        _, matrix = _residual_jacobian(fk, point, pose.detach(), tracking, jacobian)
        twist = torch.cat((-inward[..., None]*_unit_shift(geometry, pose), torch.zeros_like(pose[..., :3])), -1)
        result = velocity + _twist_solve(matrix, twist)
        result = torch.where(limited[..., None], result, velocity)
    return result if tracking else result.detach()


def board_projection(current):
    """(endpoint, velocity) projections of a continuing board revision.

    current=True: project_continue_v2 normal-only translation; False: the
    evaluate-only project_continue_v1 minimum-norm joint correction.
    """
    if current:
        return project_joint_endpoint, project_joint_velocity
    return project_joint_endpoint_v1, project_joint_velocity_v1
