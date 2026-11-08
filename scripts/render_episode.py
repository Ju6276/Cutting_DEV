#!/usr/bin/env python3
"""Render a `record_episode.py` recording to MP4: whole-arm view and cutting close-up.

Needs numpy, VTK (offscreen EGL) and Pillow, plus the ffmpeg binary; no simulator.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from PIL import Image, ImageDraw, ImageFont

BG = (0.93, 0.94, 0.96)
MATERIAL = (0.80, 0.55, 0.25)
KNIFE = (0.16, 0.25, 0.33)
BOARD = (0.62, 0.44, 0.27)
ROBOT = (0.75, 0.75, 0.75)


def polydata(points, faces):
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly.SetPoints(pts)
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    arr = vtk.vtkCellArray()
    arr.SetCells(len(faces), numpy_to_vtkIdTypeArray(cells, deep=True))
    poly.SetPolys(arr)
    return poly


def actor(source_or_poly, color, opacity=1.0, edges=False):
    mapper = vtk.vtkPolyDataMapper()
    if isinstance(source_or_poly, vtk.vtkPolyData):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(source_or_poly)
        normals.SplittingOff()
        mapper.SetInputConnection(normals.GetOutputPort())
    else:
        mapper.SetInputConnection(source_or_poly.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(mapper)
    p = a.GetProperty()
    p.SetColor(*color)
    p.SetOpacity(opacity)
    p.SetSpecular(0.2)
    if edges:
        p.EdgeVisibilityOn()
        p.SetEdgeColor(0.55, 0.36, 0.15)
        p.SetLineWidth(0.6)
    return a


def matrix(pos, mat):
    m = vtk.vtkMatrix4x4()
    r = mat.reshape(3, 3)
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, r[i, j])
        m.SetElement(i, 3, pos[i])
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--size', type=int, nargs=2, default=[760, 620])
    args = parser.parse_args()
    d = np.load(args.recording / 'episode.npz')
    report = json.loads((args.recording / 'report.json').read_text())
    frames_dir = args.recording / 'frames'
    frames_dir.mkdir(exist_ok=True)
    n = len(d['particles'])
    w, h = args.size

    # Static robot geometry; poses set per frame.
    robot = []
    for k, g in enumerate(d['geoms']):
        kind = int(d['geom_type'][k])
        if kind == 7:
            src = polydata(d[f'mesh_vert_{g}'], d[f'mesh_face_{g}'])
        elif kind == 6:
            cube = vtk.vtkCubeSource()
            sx, sy, sz = 2 * d['geom_size'][k]
            cube.SetXLength(sx), cube.SetYLength(sy), cube.SetZLength(sz)
            src = cube
        else:
            cyl = vtk.vtkCylinderSource()
            cyl.SetRadius(d['geom_size'][k][0]), cyl.SetHeight(2 * d['geom_size'][k][1]), cyl.SetResolution(24)
            src = vtk.vtkTransformPolyDataFilter()
            rot = vtk.vtkTransform()
            rot.RotateX(90)  # VTK cylinders run along y, MuJoCo along z
            src.SetTransform(rot), src.SetInputConnection(cyl.GetOutputPort())
        rgba = d['geom_rgba'][k]
        color = ROBOT if np.allclose(rgba[:3], 0.5) else tuple(rgba[:3])
        robot.append(actor(src, color, float(rgba[3]) if rgba[3] > 0 else 1.0))
    material_poly = polydata(d['particles'][0], d['tri'])
    knife_poly = polydata(d['knife'][0], d['knife_tri'])
    ground = d['ground']
    board = polydata(ground, np.array([[0, 1, 2], [0, 2, 3]]))

    centre = d['particles'][0].mean(0)
    # One window with two viewports: actors are shared within one OpenGL context.
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(2 * w, h)
    for close in (False, True):
        ren = vtk.vtkRenderer()
        ren.SetViewport(0.5 if close else 0., 0., 1. if close else 0.5, 1.)
        ren.SetBackground(*BG)
        for a in robot:
            ren.AddActor(a)
        ren.AddActor(actor(material_poly, MATERIAL, edges=close))
        ren.AddActor(actor(knife_poly, KNIFE, 0.9))
        ren.AddActor(actor(board, BOARD))
        cam = ren.GetActiveCamera()
        if close:
            cam.SetFocalPoint(*centre)
            cam.SetPosition(*(centre + np.array([0.22, -0.30, 0.20])))
        else:
            cam.SetFocalPoint(*(centre * 0.5))
            cam.SetPosition(*(centre * 0.5 + np.array([1.4, -1.6, 1.0])))
        cam.SetViewUp(0, 0, 1)
        cam.SetViewAngle(35)
        win.AddRenderer(ren)
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
        small = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError:
        font = small = ImageFont.load_default()
    title = (f"{report['label']}  |  {report['outcome']} in {report['control_steps']} steps  |  "
             f"MuJoCo Warp UR5e + FDCC + DiSECt")
    case = report.get('case')
    subtitle = ('unperturbed frozen policy' if not case else
                f"perturbation case {case['case']}: sigma {case['sigma']}, from step {case['start_step']}, {case['mode']}")
    t0 = d['sim_time'][0]
    for i in range(n):
        material_poly.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(d['particles'][i]), deep=True))
        material_poly.Modified()
        knife_poly.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(d['knife'][i]), deep=True))
        knife_poly.Modified()
        for k, a in enumerate(robot):
            a.SetUserMatrix(matrix(d['geom_pos'][i][k], d['geom_mat'][i][k]))
        win.Render()
        grab.Modified()
        grab.Update()
        arr = np.frombuffer(numpy_from(grab.GetOutput()), dtype=np.uint8).reshape(h, 2 * w, -1)[::-1, :, :3]
        canvas = Image.new('RGB', (2 * w, h + 70), tuple(int(255 * c) for c in BG))
        canvas.paste(Image.fromarray(np.ascontiguousarray(arr)), (0, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), title, fill=(25, 35, 50), font=font)
        status = (f"t = {d['sim_time'][i] - t0:.3f} s  (0.1x playback)  |  released springs "
                  f"{int(d['released'][i])}/{int(d['total_springs'])}  |  material force {d['force'][i]:.1f} N  |  "
                  f"sensor peak {d['sensor'][i]:.1f} N")
        draw.text((12, h + 44), status, fill=(25, 35, 50), font=small)
        draw.text((w + 12, 44), subtitle, fill=(60, 70, 85), font=small)
        canvas.save(frames_dir / f'{i:05d}.png')
    out = args.recording / f"{report['label']}.mp4"
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps), '-i',
                    str(frames_dir / '%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', str(out)], check=True)
    print(out)


def numpy_from(image):
    from vtk.util.numpy_support import vtk_to_numpy
    return vtk_to_numpy(image.GetPointData().GetScalars()).tobytes()


if __name__ == '__main__':
    main()
