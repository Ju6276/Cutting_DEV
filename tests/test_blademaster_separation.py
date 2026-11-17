"""Logging-only BladeMaster separation diagnostics against the native component rule."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from cutting_world_model.blademaster_separation import (
    boundary_offset, bridging_unlabelled, separation_diagnostics, side_digit)
from cutting_world_model.episode_log import episode_row

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def components():
    source = os.environ.get('CUTTING_BLADEMASTER_ROOT') or json.loads(
        (ROOT / 'configs/shared/scene_v1.json').read_text())['blademaster_source_root']
    if not (Path(source) / 'blademaster_warp/topology.py').is_file():
        pytest.skip('Native BladeMaster source unavailable')
    sys.path.insert(0, str(source))
    from blademaster_warp.topology import components
    return components


def chain(codes):
    """Particles i on a line at x=i, each touching grid nodes i and i+1."""
    n = len(codes)
    nodes = np.stack((np.arange(n), np.arange(n) + 1), 1)
    reference = np.stack((np.arange(n, dtype=float), np.zeros(n), np.zeros(n)), 1)
    return nodes, np.ones_like(nodes, float), np.asarray(codes, np.int64), reference


def run(codes, components, damage=None, plane=5.5, band=1.):
    nodes, weights, codes, reference = chain(codes)
    damage = np.zeros(len(codes)) if damage is None else np.asarray(damage, float)
    return separation_diagnostics(nodes, weights, codes, damage, reference, [plane, 0, 0], [2., 0, 0],
                                  spacing=1., band_cells=band, components=components)


def test_two_sided_cut_is_two_components_but_never_declared_separated(components):
    d = run([0] * 5 + [1, 2] + [0] * 5, components)
    assert d['separated'] is None and 'Not determinable' in d['separation_note']
    assert d['component_sizes'] == [6, 6]
    assert d['side_1_particles'] == 1 and d['side_2_particles'] == 1
    assert d['side_coded_fraction'] == pytest.approx(2 / 12)
    assert d['bridging_code0_particles'] == 0
    assert d['side_boundary_offset_m'] == pytest.approx(0.) and d['side_boundary_misplaced_particles'] == 0
    assert d['separated_bm_candidate'] is True
    assert d['board_pinched_particles'] is None
    json.dumps(d, allow_nan=False)


def test_unlabelled_bridge_reconnects_and_is_counted(components):
    nodes, weights, codes, reference = chain([0] * 5 + [1, 2] + [0] * 5)
    # Extra code-0 particle touching node 5 (side 1 there) and node 7 (side 2 there).
    nodes = np.vstack((nodes, [5, 7]))
    weights = np.ones_like(nodes, float)
    codes = np.r_[codes, 0]
    reference = np.vstack((reference, [5.5, 0, 0]))
    d = separation_diagnostics(nodes, weights, codes, np.zeros(13), reference, [5.5, 0, 0], [1, 0, 0],
                               spacing=1., band_cells=1., components=components)
    assert d['component_sizes'] == [13]
    assert d['bridging_code0_particles'] == 1
    assert bridging_unlabelled(nodes, weights, side_digit(codes)) == 1
    assert d['separated_bm_candidate'] is False


def test_small_crumb_is_not_candidate_and_damage_uses_band_only(components):
    damage = np.zeros(12)
    damage[[5, 6]] = 1  # inside |x-5.5| <= 1
    damage[0] = 1       # outside the band, excluded
    d = run([0] * 10 + [1, 2], components, damage=damage, plane=5.5)
    assert d['component_sizes'] == [11, 1]
    assert d['separated_bm_candidate'] is False  # crumb < 10 % of particles
    assert d['band_particles'] == 2 and d['band_damage_saturated_fraction'] == 1
    # Boundary lies at the labelled particles, 5 m from the prescribed plane.
    assert d['side_boundary_offset_m'] == pytest.approx(5.)


def test_boundary_split_and_higher_cut_digit():
    offset, wrong = boundary_offset(np.array([-3., -2., -1., 1., 2., .5]), np.array([1, 1, 1, 2, 2, 1]))
    assert wrong == 0 and offset == pytest.approx(.75)
    assert boundary_offset(np.array([1., 2.]), np.array([2, 0])) == (None, None)
    # Ternary codes: digit of cut 1 is independent of cut 0 (solver.py:205).
    np.testing.assert_array_equal(side_digit([0, 1, 3, 5, 6], 1), [0, 0, 1, 1, 2])


def test_episode_row_nests_record_only_for_diagnosed_worlds():
    b = torch.tensor([False, False]); f = torch.tensor([.2, .3]); n = torch.tensor([2, 3])
    info = dict(success=b, collision=b, outside=b, timeout=~b,
                separation_diagnostics=[None, {'separated': None, 'component_sizes': [5]}])
    assert 'bm_separation' not in episode_row(0, 0, 1, f, n, info, b, f, f)
    row = episode_row(1, 0, 1, f, n, info, b, f, f)
    assert row['bm_separation']['separated'] is None and 'separated' not in row
