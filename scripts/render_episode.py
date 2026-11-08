#!/usr/bin/env python3
"""Render recorded physical geometry with two views and measured force traces.

Uses one EGL window for both viewports. Geometry interpolates between recorded
control boundaries; force peaks and released-spring counts remain measured data.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from PIL import Image, ImageDraw, ImageFont

BG = (0.91, 0.93, 0.95)
INK = (24, 39, 57)
MUTED = (91, 107, 124)
TEAL = (16, 133, 136)
AMBER = (205, 123, 42)


def polydata(points, faces):
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly.SetPoints(pts)
    cells = np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64).ravel()
    arr = vtk.vtkCellArray()
    arr.ImportLegacyFormat(numpy_to_vtkIdTypeArray(cells, deep=True))
    poly.SetPolys(arr)
    return poly


def actor(source, color, opacity=1., specular=.2, edges=False):
    mapper = vtk.vtkPolyDataMapper()
    if isinstance(source, vtk.vtkPolyData):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(source)
        normals.SetFeatureAngle(40)
        normals.SplittingOn()
        mapper.SetInputConnection(normals.GetOutputPort())
    else:
        mapper.SetInputConnection(source.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(mapper)
    p = a.GetProperty()
    p.SetColor(*color)
    p.SetOpacity(opacity)
    p.SetInterpolationToPhong()
    p.SetAmbient(.18)
    p.SetDiffuse(.8)
    p.SetSpecular(specular)
    p.SetSpecularPower(36)
    if edges:
        p.EdgeVisibilityOn()
        p.SetEdgeColor(.35, .24, .12)
        p.SetLineWidth(.5)
    return a


def matrix(pos, mat):
    m = vtk.vtkMatrix4x4()
    r = mat.reshape(3, 3)
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, r[i, j])
        m.SetElement(i, 3, pos[i])
    return m


def box(center, lengths, color):
    source = vtk.vtkCubeSource()
    source.SetCenter(*center)
    source.SetXLength(lengths[0])
    source.SetYLength(lengths[1])
    source.SetZLength(lengths[2])
    return actor(source, color, specular=.05)


def font(size, bold=False):
    family = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    for name in (family, '/System/Library/Fonts/Supplemental/Arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def frame_indices(times, fps, speed):
    if fps <= 0 or not np.isfinite(speed) or speed <= 0:
        raise ValueError('FPS and playback speed must be positive')
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Recording times must be strictly increasing')
    # The final sample is always visible; other frames are uniformly timed.
    targets = np.arange(times[0], times[-1], speed / fps)
    targets = np.append(targets, times[-1])
    right = np.searchsorted(times, targets).clip(0, len(times) - 1)
    left = np.maximum(0, right - 1)
    return np.where(targets - times[left] < times[right] - targets, left, right)


def robot_actors(d):
    entries = []
    names = d['geom_mesh'] if 'geom_mesh' in d else [''] * len(d['geoms'])
    bodies = d['geom_body'] if 'geom_body' in d else [''] * len(d['geoms'])
    for k, g in enumerate(d['geoms']):
        mesh_name = str(names[k])
        kind = int(d['geom_type'][k])
        if kind == 7:
            src = polydata(d[f'mesh_vert_{g}'], d[f'mesh_face_{g}'])
        elif kind == 6:
            src = vtk.vtkCubeSource()
            sx, sy, sz = 2 * d['geom_size'][k]
            src.SetXLength(sx)
            src.SetYLength(sy)
            src.SetZLength(sz)
        elif kind == 5:
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(d['geom_size'][k][0])
            cylinder.SetHeight(2 * d['geom_size'][k][1])
            cylinder.SetResolution(64)
            src = vtk.vtkTransformFilter()
            rot = vtk.vtkTransform()
            rot.RotateX(90)  # VTK y axis -> MuJoCo z axis.
            src.SetTransform(rot)
            src.SetInputConnection(cylinder.GetOutputPort())
        else:
            raise ValueError(f'Unsupported recorded geom type {kind}')
        rgba = d['geom_rgba'][k]
        color = tuple(rgba[:3])
        if np.allclose(rgba[:3], .5):
            color = (.73, .78, .82)
            if mesh_name in ('base', 'shoulder', 'wrist1', 'wrist2', 'wrist3'):
                color = (.31, .52, .67)
            elif 'cam' in str(bodies[k]) or 'gripper' in mesh_name:
                color = (.25, .29, .34)
        entries.append((k, actor(src, color, float(rgba[3]), specular=.35), None))
    return entries


def visual_actors(d, task, roots):
    import collada

    packages = {}
    for root in roots:
        for package in root.rglob('package.xml'):
            packages[ET.parse(package).getroot().findtext('name')] = package.parent
    urdf = ET.fromstring(task['robot_xml'])
    links = {link.get('name'): link for link in urdf.findall('link')}
    entries, sources, included = [], [], set()
    for k, body in enumerate(d['geom_body']):
        if body in included:
            continue
        included.add(body)
        for visual in links[str(body)].findall('visual'):
            mesh = visual.find('geometry/mesh')
            origin = visual.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            cr, cp, cy = np.cos(rpy)
            sr, sp, sy = np.sin(rpy)
            local = np.eye(4)
            local[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                             [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                             [-sp, cp*sr, cp*cr]]
            local[:3, 3] = xyz
            color_tag = visual.find('material/color')
            rgba = np.fromstring(color_tag.get('rgba'), sep=' ') if color_tag is not None else np.array([.7,.7,.7,1.])
            if mesh is not None:
                uri = mesh.get('filename')
                if not uri.startswith('package://'):
                    raise ValueError(f'Expected a ROS package mesh URI, got {uri}')
                package, relative = uri[len('package://'):].split('/', 1)
                if package not in packages:
                    raise FileNotFoundError(f'{package}: add its source directory with --source-root')
                path = packages[package] / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                sources.append(str(path))
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                if path.suffix.lower() == '.dae':
                    document = collada.Collada(str(path))
                    unit = document.assetInfo.unitmeter or 1.
                    for geometry in document.scene.objects('geometry'):
                        for primitive in geometry.primitives():
                            triangles = primitive.triangleset() if hasattr(primitive, 'triangleset') else primitive
                            if not isinstance(triangles, collada.triangleset.BoundTriangleSet):
                                continue
                            diffuse = triangles.material.effect.diffuse if triangles.material else rgba
                            if not isinstance(diffuse, (tuple, list, np.ndarray)):
                                raise ValueError(f'Textured DAE material needs a texture loader: {path}')
                            points = triangles.vertex * scale * unit
                            a = actor(polydata(points, triangles.vertex_index), diffuse[:3], float(diffuse[3]), specular=.3)
                            entries.append((k, a, local))
                    continue
                if path.suffix.lower() != '.stl':
                    raise ValueError(f'Unsupported visual mesh format: {path}')
                source = vtk.vtkSTLReader()
                source.SetFileName(str(path))
                source.Update()
                poly = source.GetOutput()
                points = vtk_to_numpy(poly.GetPoints().GetData()) * scale
                poly.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
                a = actor(poly, rgba[:3], float(rgba[3]), specular=.3)
            else:
                geometry = visual.find('geometry')
                if geometry.find('box') is not None:
                    sizes = np.fromstring(geometry.find('box').get('size'), sep=' ')
                    a = box([0, 0, 0], sizes, rgba[:3])
                elif geometry.find('cylinder') is not None:
                    cylinder = geometry.find('cylinder')
                    source = vtk.vtkCylinderSource()
                    source.SetRadius(float(cylinder.get('radius')))
                    source.SetHeight(float(cylinder.get('length')))
                    source.SetResolution(64)
                    transform = vtk.vtkTransform()
                    transform.RotateX(90)
                    converted = vtk.vtkTransformFilter()
                    converted.SetTransform(transform)
                    converted.SetInputConnection(source.GetOutputPort())
                    a = actor(converted, rgba[:3], float(rgba[3]))
                else:
                    raise ValueError(f'Unsupported visual geometry on {body}')
            entries.append((k, a, local))
    return entries, sources


def material_vertices(d, index):
    # DiSECt cutting/visualizer.py: virtual cut vertices follow their edge endpoints.
    points = d['particles'][index]
    edges = d['cut_edge_indices']
    weight = d['cut_edge_coords'].reshape(-1, 1)
    virtual = (1 - weight) * points[edges[:, 0]] + weight * points[edges[:, 1]]
    return np.concatenate((points, virtual))


def scene(d, width, height, edges, robot):
    material = polydata(material_vertices(d, 0), np.concatenate((d['tri'], d['cut_tri'])))
    knife = polydata(d['knife'][0], d['knife_tri'])
    center = (d['particles'].min(axis=(0, 1)) + d['particles'].max(axis=(0, 1))) / 2
    close_points = [d['particles'].reshape(-1, 3), d['knife'].reshape(-1, 3)]
    for k, name in enumerate(d['geom_mesh']):
        if 'knife' in str(name).lower():
            vertices = d[f'mesh_vert_{d["geoms"][k]}']
            rotations = d['geom_mat'][:, k].reshape(-1, 3, 3)
            points = np.einsum('tij,vj->tvi', rotations, vertices) + d['geom_pos'][:, k, None]
            close_points.append(points.reshape(-1, 3))
    close_points = np.concatenate(close_points)
    ground = d['ground']
    board_center = ground.mean(0)
    board_top = board_center[2]
    board = box(board_center - [0, 0, .006], [np.ptp(ground[:, 0]), np.ptp(ground[:, 1]), .012],
                (.69, .55, .38))
    floor = box([.05, .2, -.019], [2., 2., .025], (.84, .87, .89))
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetMultiSamples(4)
    win.SetSize(width, height)
    renderers = []
    for close in (False, True):
        ren = vtk.vtkRenderer()
        ren.SetViewport(.4 if close else 0., 0., 1. if close else .4, 1.)
        ren.SetBackground(*BG)
        ren.SetBackground2(.98, .985, .99)
        ren.GradientBackgroundOn()
        ren.UseFXAAOn()
        for _, a, _ in robot:
            ren.AddActor(a)
        ren.AddActor(actor(material, (.86, .63, .31), specular=.12, edges=edges))
        ren.AddActor(actor(knife, (.16, .48, .73), specular=.65))
        ren.AddActor(board)
        if not close:
            ren.AddActor(floor)
            # Display support under the recorded contact plane.
            ren.AddActor(box(board_center - [0, 0, board_top / 2 + .006],
                             [np.ptp(ground[:, 0]) * .85, np.ptp(ground[:, 1]) * .85,
                              max(.001, board_top - .012)], (.76, .79, .82)))
        camera = ren.GetActiveCamera()
        if close:
            focal = (close_points.min(0) + close_points.max(0)) / 2
            camera.SetFocalPoint(*focal)
            camera.SetPosition(*(focal + [.25, -.36, .19]))
            camera.ParallelProjectionOn()
            direction = np.array([.25, -.36, .19])
            direction /= np.linalg.norm(direction)
            right = np.cross([0, 0, 1], direction)
            right /= np.linalg.norm(right)
            up = np.cross(direction, right)
            relative = close_points - focal
            half_height = np.abs(relative @ up).max()
            half_width = np.abs(relative @ right).max()
            aspect = .6 * width / height
            camera.SetParallelScale(1.12 * max(half_height, half_width / aspect))
        else:
            focal = np.array([center[0] * .5, center[1] * .52, .27])
            camera.SetFocalPoint(*focal)
            camera.SetPosition(*(focal + [1.2, -1.6, .9]))
            camera.ParallelProjectionOn()
            camera.SetParallelScale(.40)
        camera.SetViewUp(0, 0, 1)
        ren.AutomaticLightCreationOff()
        for offset, intensity in (([.6, -.8, 1.2], .85), ([-.5, -.2, .7], .45), ([.1, .8, 1.], .6)):
            light = vtk.vtkLight()
            light.SetLightTypeToSceneLight()
            light.SetPosition(*(center + offset))
            light.SetFocalPoint(*center)
            light.SetIntensity(intensity)
            ren.AddLight(light)
        ren.ResetCameraClippingRange()
        win.AddRenderer(ren)
        renderers.append(ren)
    return win, renderers, material, knife


def overlay(image, d, report, i, speed, fonts, title):
    width, view_height = image.size
    header, footer = 106, 210
    canvas = Image.new('RGB', (width, view_height + header + footer), (247, 249, 251))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    big, normal, small = fonts
    draw.text((30, 15), title, fill=INK, font=big)
    draw.text((30, 57), 'UR5e  /  compliant control  /  DiSECt material simulation', fill=MUTED, font=normal)
    legend = 'Gray: mounted URDF tool   |   Blue: DiSECt contact blade'
    draw.text((width - draw.textlength(legend, font=small) - 30, 66), legend, fill=MUTED, font=small)
    for x, label in ((30, 'ROBOT VIEW'), (int(width * .4) + 24, 'CUTTING CLOSE-UP')):
        draw.rounded_rectangle((x - 10, header + 10, x + draw.textlength(label, font=small) + 10,
                                header + 41), radius=5, fill=(247, 249, 251))
        draw.text((x, header + 17), label, fill=MUTED, font=small)
    draw.line((int(width * .4), header, int(width * .4), header + view_height), fill=(215, 222, 230), width=2)
    times = d['sim_time'] - d['sim_time'][0]
    t = float(times[i])
    final = i == len(times) - 1
    ending = {'success': 'GOAL REACHED', 'collision': 'FORCE LIMIT', 'outside': 'WORKSPACE LIMIT',
              'truncated': 'TIME LIMIT'}
    status = ending.get(report['outcome'], report['outcome'].upper()) if final else 'EXECUTING'
    text = f'{status}   |   {t:.2f} s / {times[-1]:.2f} s   |   {speed:g}x speed'
    draw.text((width - draw.textlength(text, font=normal) - 30, 25), text, fill=TEAL, font=normal)
    y = header + view_height
    draw.line((0, y, width, y), fill=(215, 222, 230), width=1)
    draw.text((30, y + 16), 'MEASURED FORCE  /  control-step peaks', fill=INK, font=normal)
    control_t = d['control_sim_time'] if 'control_sim_time' in d else d['sim_time']
    control_t = control_t - control_t[0]
    forces = d['control_force'] if 'control_force' in d else d['force']
    sensors = d['control_sensor'] if 'control_sensor' in d else d['sensor']
    x0, x1, y0, y1 = 65, int(width * .63), y + 64, y + 170
    top = max(1., 5 * np.ceil(float(max(forces.max(), sensors.max())) / 5))
    for tick in (0., top / 2, top):
        py = y1 - tick / top * (y1 - y0)
        draw.line((x0, py, x1, py), fill=(225, 231, 237))
        draw.text((16, py - 8), f'{tick:g}', fill=MUTED, font=small)
    for values, color in ((forces, AMBER), (sensors, TEAL)):
        coords = [(x0 + float(tt / times[-1]) * (x1 - x0), y1 - float(f / top) * (y1 - y0))
                  for tt, f in zip(control_t, values)]
        draw.line(coords, fill=color, width=3)
    px = x0 + t / times[-1] * (x1 - x0)
    draw.line((px, y0 - 7, px, y1 + 3), fill=INK, width=2)
    draw.text((x0, y1 + 9), '0 s', fill=MUTED, font=small)
    draw.text((x1 - 62, y1 + 9), f'{times[-1]:.2f} s', fill=MUTED, font=small)
    rx = int(width * .67)
    draw.text((rx, y + 55), f'Material   {d["force"][i]:.1f} N   |   run peak {forces.max():.2f} N',
              fill=AMBER, font=normal)
    draw.text((rx, y + 88), f'Sensor |component|max   {d["sensor"][i]:5.1f} N', fill=TEAL, font=normal)
    draw.text((rx, y + 128), f'Released springs   {int(d["released"][i])} / {int(d["total_springs"])}',
              fill=INK, font=normal)
    draw.text((rx, y + 165), 'Geometry interpolated between recorded states', fill=MUTED, font=small)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--speed', type=float, default=.1, help='simulation seconds per video second')
    parser.add_argument('--size', type=int, nargs=2, default=[960, 650], help='half canvas width and viewport height')
    parser.add_argument('--title', help='display title; defaults to report label')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preview', action='store_true', help='render start, midpoint and final PNGs only')
    parser.add_argument('--wireframe', action='store_true')
    parser.add_argument('--robot-geometry', choices=('visual', 'collision'), default='visual')
    parser.add_argument('--source-root', type=Path, action='append', default=[], help='ROS package search root for URDF visual meshes')
    args = parser.parse_args()
    d = np.load(args.recording / 'episode.npz')
    if any(key not in d for key in ('cut_tri', 'cut_edge_indices', 'cut_edge_coords')):
        parser.error('Recording lacks DiSECt cut-surface topology; record with the current recorder')
    report = json.loads((args.recording / 'report.json').read_text())
    indices = frame_indices(d['sim_time'], args.fps, args.speed)
    width, height = args.size[0] * 2, args.size[1]
    title = args.title or report['label'].replace('_', ' ')
    sources = []
    if args.robot_geometry == 'visual':
        if not args.source_root or 'body_pos' not in d:
            parser.error('Visual geometry needs --source-root and body poses from the current recorder')
        task = json.loads((args.recording / 'task.json').read_text())
        robot, sources = visual_actors(d, task, args.source_root)
    else:
        robot = robot_actors(d)
    win, renderers, material, knife = scene(d, width, height, args.wireframe, robot)
    fonts = (font(30, True), font(21), font(16))
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.ReadFrontBufferOff()
    output = args.output or args.recording / f'{report["label"]}.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    previews = sorted(set([0, len(indices) // 2, len(indices) - 1]))
    selected = previews if args.preview else range(len(indices))
    encoder = None
    if not args.preview:
        encoder = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', f'{width}x{height + 316}', '-r', str(args.fps), '-i', '-', '-an',
             '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
             '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
    try:
        for j in selected:
            i = indices[j]
            for poly, points in ((material, material_vertices(d, i)), (knife, d['knife'][i])):
                poly.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
                poly.Modified()
            for k, a, local in robot:
                if local is None:
                    a.SetUserMatrix(matrix(d['geom_pos'][i][k], d['geom_mat'][i][k]))
                else:
                    rotation = d['body_mat'][i][k].reshape(3, 3)
                    position = d['body_pos'][i][k] + rotation @ local[:3, 3]
                    a.SetUserMatrix(matrix(position, rotation @ local[:3, :3]))
            for ren in renderers:
                ren.ResetCameraClippingRange()
            win.Render()
            grab.Modified()
            grab.Update()
            arr = vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).reshape(height, width, -1)[::-1, :, :3]
            canvas = overlay(Image.fromarray(np.ascontiguousarray(arr)), d, report, i, args.speed, fonts, title)
            if j in previews:
                canvas.save(output.with_name(f'{output.stem}_frame_{j:05d}.png'))
            if encoder is not None:
                encoder.stdin.write(canvas.tobytes())
            if j % 60 == 0:
                print(f'frame {j + 1}/{len(indices)}; simulation t={d["sim_time"][i]:.3f}', flush=True)
    finally:
        if encoder is not None:
            encoder.stdin.close()
            returncode = encoder.wait()
            if returncode != 0:
                raise RuntimeError(f'ffmpeg exited with {returncode}')
        win.Finalize()
    output.with_suffix('.render.json').write_text(json.dumps({
        'command': sys.argv, 'recording': str(args.recording), 'fps': args.fps,
        'playback_speed': args.speed, 'video_frames': len(indices),
        'simulation_seconds': float(d['sim_time'][-1] - d['sim_time'][0]),
        'preview_only': args.preview,
        'robot_geometry': args.robot_geometry, 'visual_mesh_sources': sources,
    }, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
