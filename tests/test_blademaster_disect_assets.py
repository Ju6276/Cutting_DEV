import copy
import json
from pathlib import Path

import pytest
from scripts.prepare_blademaster_disect_assets import PRESETS, disect_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", list(PRESETS))
def test_elastic_mapping_preserves_disect_contact_and_damage(name):
    template=json.loads((ROOT/"configs/free_scenes/potato_003_half.json").read_text())
    old=copy.deepcopy(template)
    preset=PRESETS[name]
    cfg=disect_config(template,preset,Path("/tmp/example.msh"),.07)
    assert template==old
    young,nu=preset["young"],preset["poisson"]
    assert cfg["young"]==young and cfg["poisson"]==nu and cfg["density"]==preset["density"]
    assert cfg["mu"]==pytest.approx(young/(2*(1+nu)))
    assert cfg["lambda"]==pytest.approx(young*nu/((1+nu)*(1-2*nu)))
    for key,value in old.items():
        if "spring" in key or "sdf_" in key or key.startswith("ground"):
            assert cfg[key]==value
    assert cfg["static_vertices"]==dict(include=[],exclude=[])
    assert "cutting_energy" not in cfg and "yield_stress" not in cfg
    tip_initial=cfg["initial_y"]-.02954
    assert tip_initial-(.07+cfg["geometry"]["position"][1])==pytest.approx(.01)
    assert tip_initial+cfg["velocity_y"]*cfg["sim_duration"]<0


def test_generated_candidates_keep_source_size_and_distinct_task_identity():
    catalog=json.loads((ROOT/"configs/blademaster_disect/catalog.json").read_text())
    assert set(catalog)=={"banana","cake","tofu","luncheon"}
    for name,preset in PRESETS.items():
        cfg=json.loads((ROOT/"configs/blademaster_disect"/("pose_"+name+".json")).read_text())
        assert cfg["task"]["version"]=="free_cut_pose_v1"
        assert cfg["groups"][0]["scene"].endswith("bm_"+name+"_20261114/scene.pt")
        assert catalog[name]["source_preset"]["size"]==preset["size"]
        assert catalog[name]["fixed_particles"]==0
    # The 160 mm cake retains its diameter; slice position fits a 150 mm blade.
    radius=PRESETS["cake"]["size"][0]/2
    offset=PRESETS["cake"]["x_offset"]
    assert 2*(radius*radius-offset*offset)**.5<.15


def test_ground_overlap_excludes_ghost_dofs_but_includes_cut_surface():
    from types import SimpleNamespace
    import torch
    from scripts.check_blademaster_disect_assets import ground_stats
    # The ghost falls 10 mm below the board; the actual cut point remains above.
    q=torch.tensor([[[0., .002, 0.], [0., -.0001, 0.], [0., -.01, 0.], [0., .003, 0.]]])
    model=SimpleNamespace(contact_ke=1000.,contact_kd=0.,contact_distance=.001,
                          contact_mask=torch.tensor([1.,1.,0.,0.]),
                          cut_edge_indices=torch.tensor([[0,2]]),
                          cut_edge_coords=torch.tensor([.1]))
    m=SimpleNamespace(worlds=1,state_field=lambda _: q,model=model)
    result=ground_stats(m)["worlds"][0]
    assert result["all_fem_nodes"]["minimum_y_mm"]==pytest.approx(-10.)
    assert result["ghost_nodes"]["below_minus_1mm"]==1
    assert result["physical_contact_geometry"]["minimum_y_mm"]==pytest.approx(-.1)
    assert result["physical_contact_geometry"]["below_minus_1mm"]==0
    # If the true interpolated cut surface penetrates, it must still be counted.
    model.cut_edge_coords.fill_(.5)
    result=ground_stats(m)["worlds"][0]
    assert result["cut_surface_points"]["minimum_y_mm"]==pytest.approx(-4.)
    assert result["physical_contact_geometry"]["below_minus_1mm"]==1


def test_ground_report_keeps_packed_worlds_separate():
    from types import SimpleNamespace
    import torch
    from scripts.check_blademaster_disect_assets import ground_stats
    q=torch.tensor([[[0., .002, 0.], [0., -.01, 0.]],
                    [[0., .004, 0.], [0., -.02, 0.]]])
    model=SimpleNamespace(contact_ke=1000.,contact_kd=0.,contact_distance=.001,
                          contact_mask=torch.tensor([1.,0.,1.,0.]),
                          cut_edge_indices=torch.tensor([[0,1],[2,3]]),
                          cut_edge_coords=torch.tensor([.1,.25]))
    m=SimpleNamespace(worlds=2,state_field=lambda _: q,model=model)
    result=ground_stats(m)["worlds"]
    assert result[0]["physical_contact_geometry"]["minimum_y_mm"]==pytest.approx(.8)
    assert result[1]["physical_contact_geometry"]["minimum_y_mm"]==pytest.approx(-2.)
    assert result[0]["ghost_nodes"]["count"]==result[1]["ghost_nodes"]["count"]==1
