import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    'free_object_check', Path(__file__).resolve().parents[1] / 'scripts/check_free_object_scene.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_world_vertices_include_rotation_and_translation_per_world():
    vertices = torch.tensor([[0., -1., -2.], [0., -1., 2.], [0., 0., 0.]], dtype=torch.float64)
    position = torch.tensor([[1., 10., 0.], [2., 20., 0.]], dtype=torch.float64)
    c = 2 ** -.5
    rotation = torch.tensor([[0., 0., 0., 1.], [c, 0., 0., c]], dtype=torch.float64)
    world = m.transformed_knife_vertices(vertices, position, rotation)
    torch.testing.assert_close(world[0], vertices + position[0])
    # A 90 degree roll maps (y,z) to (-z,y); the old position.y + min(local.y) gives 19.
    torch.testing.assert_close(world[1, :, 1], torch.tensor([22., 18., 20.], dtype=torch.float64))
    assert world[1, :, 1].min() == 18.


def test_spring_endpoints_follow_material_interpolation_not_contact_search():
    particles = torch.tensor([[0., 0., 0.], [10., 0., 0.], [0., 4., 0.], [10., 4., 0.],
                              [20., 0., 0.], [30., 0., 0.], [20., 4., 0.], [30., 4., 0.]])
    edges = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    u = torch.tensor([.2, .7, .2, .7])
    springs = torch.tensor([[0, 1], [2, 3]])
    result = m.spring_endpoints(particles, edges, u, springs)
    torch.testing.assert_close(result, torch.tensor([[[2., 0., 0.], [7., 4., 0.]],
                                                    [[22., 0., 0.], [27., 4., 0.]]]))


def test_arrival_does_not_count_overshoot_or_retraction_as_reaching_target():
    def row(t, phase, actual):
        return dict(sim_time_s=t, phase=phase, edge_vs_initial_bottom_mm=actual,
                    commanded_edge_vs_initial_bottom_mm=0., edge_vs_current_bottom_mm=actual + 2.,
                    edge_world_y_mm=actual + 3., legacy_height_error_mm=.2)
    rows = [row(.9, 'press', 1.), row(1., 'press', -2.),
            row(1.1, 'hold', -1.), row(1.2, 'retract', 0.)]
    result = m.knife_summary(rows, 1.)
    assert result['time_edge_within_0p5mm_of_commanded_final_s'] is None
    rows.insert(3, row(1.15, 'hold', -.4))
    assert m.knife_summary(rows, 1.)['time_edge_within_0p5mm_of_commanded_final_s'] == 1.15
