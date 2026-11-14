"""Render all actually captured worlds, laid out for viewing only.

Use capture_environment_batch.py first. This shows initialized simulation states,
not a training rollout. Robot visual mappers are shared; material vertices are
individual captured states. Distant material color is a display fallback.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import vtk
from PIL import Image, ImageDraw
from vtk.util.numpy_support import vtk_to_numpy
import render_episode as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.recording / 'report.json').read_text())
    if report['physical_actions'] != 0 or report['optimizer_updates'] != 0:
        raise ValueError('This renderer labels initialized states only')
    d = dict(np.load(args.recording / 'episode.npz', allow_pickle=False))
    batch_path = args.recording / 'world_states.npz'
    worlds = dict(np.load(batch_path, allow_pickle=False))
    count = report['worlds']
    if not np.array_equal(worlds['world_id'], np.arange(count)):
        raise ValueError('Missing or reordered world IDs')
    for key in ('particles', 'joint_q', 'body_pos', 'body_mat'):
        if len(worlds[key]) != count or not np.isfinite(worlds[key]).all():
            raise ValueError(f'Invalid captured array {key}')
    task = json.loads((args.recording / 'task.json').read_text())
    robot, sources = r.visual_actors(d, task, args.source_root)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    spacing = np.array([1.1, 1.05, 0.])
    offsets = np.array([[i % cols, i // cols, 0] for i in range(count)]) * spacing
    offsets[:, :2] -= offsets[:, :2].mean(0)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(True)
    win.SetMultiSamples(4)
    win.SetSize(2560, 1980)
    ren = vtk.vtkRenderer()
    ren.SetBackground(.92, .94, .96)
    ren.UseFXAAOn()
    win.AddRenderer(ren)
    faces = np.concatenate([d['tri'], d['cut_tri']])
    weights = d['cut_edge_coords'].reshape(-1, 1)
    edges = d['cut_edge_indices']
    ground = d['ground']
    board_center = ground.mean(0)
    board_size = np.array([np.ptp(ground[:, 0]), np.ptp(ground[:, 1]), .012])
    board = r.box(board_center - [0, 0, .006], board_size, (.68, .54, .35))
    support = r.box([board_center[0], board_center[1], (board_center[2]-.012)/2],
                    [board_size[0]*.85, board_size[1]*.85, max(.001, board_center[2]-.012)],
                    (.64, .68, .72))
    for i, offset in enumerate(offsets):
        for k, original, local in robot:
            instance = vtk.vtkActor()
            instance.SetMapper(original.GetMapper())
            instance.GetProperty().DeepCopy(original.GetProperty())
            if str(d['geom_body'][k]) == 'b_bot_right_inner_knuckle':
                instance.GetProperty().SetColor(.44, .47, .50)
            transform = np.eye(4)
            transform[:3, :3] = worlds['body_mat'][i, k].reshape(3, 3)
            transform[:3, 3] = worlds['body_pos'][i, k] + offset
            transform = transform @ local
            instance.SetUserMatrix(r.matrix(transform[:3, 3], transform[:3, :3]))
            ren.AddActor(instance)
        p = worlds['particles'][i]
        virtual = (1-weights)*p[edges[:, 0]] + weights*p[edges[:, 1]]
        points = np.concatenate([p, virtual]) + offset
        ren.AddActor(r.actor(r.polydata(points, faces), (.10, .25, .075), smooth=True))
        for original in (board, support):
            instance = vtk.vtkActor()
            instance.SetMapper(original.GetMapper())
            instance.GetProperty().DeepCopy(original.GetProperty())
            instance.SetPosition(*offset)
            ren.AddActor(instance)
    bounds = ren.ComputeVisiblePropBounds()
    center = np.array([(bounds[j]+bounds[j+1])/2 for j in (0, 2, 4)])
    camera = ren.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + np.array([18, -24, 26])))
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    ren.ResetCamera()
    camera.SetParallelScale(camera.GetParallelScale() * .84)
    ren.ResetCameraClippingRange()
    try:
        win.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(win)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        image = capture.GetOutput()
        width, height, _ = image.GetDimensions()
        pixels = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(height, width, 3)[::-1]
        canvas = Image.new('RGB', (width, height+132), (235, 240, 245))
        canvas.paste(Image.fromarray(pixels), (0, 132))
        draw = ImageDraw.Draw(canvas)
        draw.text((48, 23), f'{count} actual initialized worlds | cucumber | NO TRAINING',
                  font=r.font(40, True), fill=r.INK)
        draw.text((48, 82), 'Per-world material and starting clearance randomized; tiled only for display. No policy loaded.',
                  font=r.font(25), fill=r.MUTED)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(args.output)
    finally:
        win.Finalize()
    metadata = dict(report, capture_sha256=hashlib.sha256(batch_path.read_bytes()).hexdigest(),
                    world_ids=worlds['world_id'].tolist(), offsets_m=offsets.tolist(),
                    robot_visual_sources=sources, show_contact_blade=False,
                    material_appearance='Distant fallback color; actual coarse material geometry',
                    output=str(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()

