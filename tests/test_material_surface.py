import torch
import pytest
from cutting_world_model.material_surface import surface_plane_bounds

def bounds(p,tri,cut=(),edges=(),coords=()):
    return surface_plane_bounds(torch.tensor(p,dtype=torch.double),
        torch.tensor(tri,dtype=torch.long),torch.tensor(cut,dtype=torch.long),
        torch.tensor(edges,dtype=torch.long),coords,[0,0,0],[0,1,0])

def test_unreferenced_ghost_does_not_report_food_penetration():
    # Last coordinate is an auxiliary node, not referenced by the represented skin.
    p=[[0,.001,0],[1,.001,0],[0,.002,1],[0,-.0067,0]]
    b=bounds(p,[[0,1,2]])
    assert b["raw_min_height_m"] == -.0067
    assert b["surface_min_height_m"] == .001
    assert b["surface_below_plane_fraction"] == 0
    p[-1][1]=-1.
    assert bounds(p,[[0,1,2]])["surface_min_height_m"] == .001

def test_real_surface_penetration_is_not_hidden():
    b=bounds([[0,-.0067,0],[1,.001,0],[0,.002,1]],[[0,1,2]])
    assert b["surface_min_height_m"] == -.0067
    assert b["surface_below_plane_fraction"] == pytest.approx(1/3)

def test_interpolated_cut_vertex_still_counts_with_ghost_parent():
    # Parent 3 is not directly on the surface, but cut interpolation becomes real.
    b=bounds([[0,.001,0],[1,.001,0],[0,.002,1],[0,-.005,0]],
             [[0,1,2]],[[4,1,2]],[[0,3]],[.5])
    assert b["surface_min_height_m"] == pytest.approx(-.002)
    assert b["surface_vertex_count"] == 4

def test_batched_translated_plane_and_bad_indices():
    p=torch.tensor([[[0.,.001,0],[1,.002,0],[0,.003,1]],
                    [[0.,-.004,0],[1,.002,0],[0,.003,1]]],dtype=torch.double)
    b=surface_plane_bounds(p+2,torch.tensor([[0,1,2]]),[],[],[],[2,2,2],[0,1,0])
    torch.testing.assert_close(b["surface_min_height_m"],torch.tensor([.001,-.004],dtype=torch.double))
    with pytest.raises(ValueError,match="world-local"):
        surface_plane_bounds(p,[[0,1,2]],[],[[0,99]],[.5],[0,0,0],[0,1,0])
    with pytest.raises(ValueError,match="unit"):
        surface_plane_bounds(p,[[0,1,2]],[],[],[],[0,0,0],[0,2,0])
