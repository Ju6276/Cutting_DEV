"""Logging-only BladeMaster cut-topology diagnostics at episode termination.

Nothing here is a reward, success rule or policy/WM input. Native BladeMaster
does not provide a "prescribed slice detached" indicator, so ``separated`` is
always ``None``. Field semantics (BladeMaster-Warp c51ef73):

* codes: ternary side code, one digit per cut index (solver.py:205-208); our
  batched coupling always uses cut index 0 (batched.py:388), i.e. digit 1.
  ``assign`` (batch_contact.py:444-469; the explicit-profile knife does not
  replace it because it has no ``bevel`` argument, profile_runtime.py) writes
  side 1 (blade-local x < 0) or 2 to a code-0 particle within 6 grid spacings of
  the blade plane that is just behind the edge and "opened", or pinched between
  the edge and the board (label_board_layer). With an external tool
  ``free_blade`` is 0, so "opened" is always true (batch_contact.py:463): a side
  code records geometric edge passage, not damage.
* damage: accumulated cutting work / threshold, clamped at 1, only for particles
  ahead of the edge within ``cut_band_cells`` spacings of the blade plane
  (batch_profile_contact.py:84-93).
* components: particles linked through shared grid nodes with compatible codes;
  code 0 is compatible with every side (topology.py:6-57). A count above one may
  be a crumb or a partial slice, so it is not separation of the prescribed cut.
* The board-pinch condition is evaluated inside ``assign`` and not stored; the
  backend does not expose a pinched-particle count.
"""
import numpy as np

NOTE = ("Not determinable from native BladeMaster fields: a connectivity count above one "
        "can be a crumb or partial slice, side codes mark geometric edge passage "
        "(free_blade=0) and damage only covers particles ahead of the edge. Diagnostic only.")
CANDIDATE_NOTE = ("Candidate only, unused by reward/success/policy: two largest components "
                  "each >= 10% of particles with reference centroids on opposite sides of "
                  "the prescribed cut plane.")


def side_digit(codes, cut_index=0):
    """Side digit (0 unlabelled, 1 or 2) of ``cut_index`` in ternary codes (solver.py:205)."""
    return (np.asarray(codes, np.int64) // 3 ** int(cut_index)) % 3


def bridging_unlabelled(nodes, weights, digit):
    """Code-0 particles whose own stencil nodes are shared with side-1 and side-2 particles.

    Such a particle links both sides in one hop (code 0 is compatible with both,
    topology.py:6-14). Chains of several code-0 particles are not counted here.
    """
    nodes, used = np.asarray(nodes), np.asarray(weights) > 0
    particle = np.broadcast_to(np.arange(len(digit))[:, None], nodes.shape)[used]
    node = nodes[used].astype(np.int64)
    count = int(node.max()) + 1 if node.size else 1
    reaches = []
    for side in (1, 2):
        touched = np.zeros(count, bool)
        touched[node[digit[particle] == side]] = True
        reached = np.zeros(len(digit), bool)
        np.logical_or.at(reached, particle, touched[node])
        reaches.append(reached)
    return int((reaches[0] & reaches[1] & (digit == 0)).sum())


def boundary_offset(signed, digit):
    """Best 1-D split of side-1 (expected negative) and side-2 reference distances.

    Returns the split position (midpoint between the neighbouring samples) that
    minimises particles on the wrong side, and that number; None when a side is empty.
    """
    keep = digit > 0
    s, d = signed[keep], digit[keep]
    if not (d == 1).any() or not (d == 2).any():
        return None, None
    order = np.argsort(s, kind='stable')
    s, d = s[order], d[order]
    # Split before index k: side 2 among the first k plus side 1 among the rest.
    wrong = np.concatenate(([0], np.cumsum(d == 2))) + np.concatenate((np.cumsum((d == 1)[::-1])[::-1], [0]))
    k = int(np.argmin(wrong))
    left = s[k - 1] if k > 0 else s[0]
    right = s[k] if k < len(s) else s[-1]
    return float(.5 * (left + right)), int(wrong[k])


def separation_diagnostics(nodes, weights, codes, damage, reference, plane_point, plane_normal,
                           spacing, band_cells, cut_index=0, components=None):
    """Per-world terminal record; ``components`` is topology.components (injected for tests)."""
    codes, damage = np.asarray(codes, np.int64), np.asarray(damage, np.float64)
    reference = np.asarray(reference, np.float64)
    normal = np.asarray(plane_normal, np.float64)
    normal = normal / np.linalg.norm(normal)
    signed = (reference - np.asarray(plane_point, np.float64)) @ normal
    digit = side_digit(codes, cut_index)
    n = len(codes)
    labels, sizes = components(nodes, weights, codes)
    order = np.argsort(-sizes, kind='stable')
    band = np.abs(signed) <= band_cells * spacing
    offset, misplaced = boundary_offset(signed, digit)
    candidate = False
    if len(sizes) >= 2:
        first, second = order[:2]
        means = [float(signed[labels == c].mean()) for c in (first, second)]
        candidate = bool(min(sizes[first], sizes[second]) >= .1 * n and means[0] * means[1] < 0)
    return dict(
        separated=None, separation_note=NOTE,
        particles=n, cut_index=int(cut_index),
        component_sizes=[int(x) for x in sizes[order]],
        side_coded_fraction=float((digit > 0).mean()),
        side_1_particles=int((digit == 1).sum()), side_2_particles=int((digit == 2).sum()),
        bridging_code0_particles=bridging_unlabelled(nodes, weights, digit),
        band_half_width_m=float(band_cells * spacing), band_particles=int(band.sum()),
        band_damage_saturated_fraction=float((damage[band] >= 1).mean()) if band.any() else None,
        side_boundary_offset_m=offset, side_boundary_misplaced_particles=misplaced,
        board_pinched_particles=None,
        board_pinched_note='Pinch test is internal to BladeMaster assign; no stored count.',
        separated_bm_candidate=candidate, separated_bm_candidate_note=CANDIDATE_NOTE)
