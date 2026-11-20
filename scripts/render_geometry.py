#!/usr/bin/env python3
"""Shared geometry/appearance helpers only; no rendering CLI."""
import argparse
import hashlib
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

# Display materials, not recovered scan textures or physical parameters.
MATERIAL_STYLES = {
    'generic': ((.48, .53, .58), (.69, .73, .76)),
    'prism': ((.57, .65, .69), (.68, .77, .79)),
    'potato': ((.66, .47, .25), (.87, .80, .56)),
    'cucumber': ((.20, .38, .09), (.69, .79, .48)),
}


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


def actor(source, color, opacity=1., specular=.2, edges=False, smooth=False):
    mapper = vtk.vtkPolyDataMapper()
    if isinstance(source, vtk.vtkPolyData):
        normals = vtk.vtkPolyDataNormals()
        if smooth:
            clean = vtk.vtkCleanPolyData()
            clean.SetInputData(source)
            clean.SetTolerance(1e-9)
            normals.SetInputConnection(clean.GetOutputPort())
        else:
            normals.SetInputData(source)
        normals.SetFeatureAngle(70 if smooth else 40)
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
                if uri.startswith('package://'):
                    package, relative = uri[len('package://'):].split('/', 1)
                    if package not in packages:
                        raise FileNotFoundError(f'{package}: add its source directory with --source-root')
                    path = packages[package] / relative
                elif Path(uri).is_absolute():
                    # Prepared user CAD bundles record absolute asset paths.
                    path = Path(uri)
                else:
                    raise ValueError(f'Expected package URI or absolute mesh path, got {uri}')
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


def display_vertices(d, index, appearance=None):
    points = material_vertices(d, index)
    if appearance is None:
        return points
    exterior = (points[appearance['indices']] * appearance['weights'][..., None]).sum(1)
    return np.concatenate((exterior, points))


def presentation_ground(d, task):
    """Use the existing main-run 24x16 cm board exterior for shared-tool views.

    The support plane remains infinite; these corners are presentation only.
    Preserve recorded centre/orientation and keep historical tools unchanged.
    """
    ground = np.asarray(d['ground'])
    if task.get('tool_revision') != 'disect_knife_rigid_mount_v1':
        return ground
    along = ground[1]-ground[0]
    across = ground[3]-ground[0]
    along = along/np.linalg.norm(along)*.12
    across = across/np.linalg.norm(across)*.08
    centre = ground.mean(0)
    return centre+np.array([-along-across,along-across,along+across,-along+across])


def physical_robot_actors(d, task, roots):
    """Shared visual robot/handle; the caller draws recorded physical blades."""
    entries, sources = visual_actors(d, task, roots)
    tools = [e for e in entries if str(d['geom_body'][e[0]]) == 'b_bot_right_inner_knuckle']
    result = [e for e in entries if str(d['geom_body'][e[0]]) not in
              ('b_bot_right_inner_knuckle', 'b_bot_cam_cables_link')]
    for k, original, local in tools:
        original.GetMapper().Update()
        plane = vtk.vtkPlane()
        plane.SetOrigin(-.04786805, 0, 0)
        plane.SetNormal(1, 0, 0)
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(original.GetMapper().GetInput())
        clip.SetClipFunction(plane)
        clip.Update()
        result.append((k, actor(clip, (.61, .63, .65), specular=.5), local))
    return result, sources, tools


def load_appearance(path, recording):
    appearance = dict(np.load(path, allow_pickle=False))
    metadata = json.loads(str(appearance['metadata']))
    if metadata['recording_sha256'] != hashlib.sha256(recording.read_bytes()).hexdigest():
        raise ValueError('Appearance mapping belongs to a different recording')
    return appearance


def material_face_colors(d, material_kind='generic', appearance=None, highlight_cut_faces=False):
    exterior, interior = MATERIAL_STYLES[material_kind]
    cut_count = len(d['cut_tri'])
    if appearance is None:
        outer = np.tile(np.rint(np.array(exterior)*255).astype(np.uint8), (len(d['tri']),1))
        cut = np.tile(np.rint(np.array(interior if highlight_cut_faces else exterior)*255).astype(np.uint8),
                      (cut_count,1))
    else:
        outer = appearance['rgb']
        if highlight_cut_faces:
            cut = np.tile(np.rint(np.array(interior)*255).astype(np.uint8), (cut_count,1))
        else:
            # Carry nearby scan appearance onto the prepared faces. This removes
            # the schematic pale band, not actual surface openings or connectivity.
            fine = display_vertices(d, 0, appearance)[:len(appearance['indices'])].reshape(-1,3,3).mean(1)
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(np.ascontiguousarray(fine),deep=True))
            cloud = vtk.vtkPolyData(); cloud.SetPoints(points)
            locator = vtk.vtkStaticPointLocator(); locator.SetDataSet(cloud); locator.BuildLocator()
            centers = material_vertices(d,0)[d['cut_tri']].mean(1)
            nearest = np.array([locator.FindClosestPoint(center) for center in centers],dtype=int)
            cut = outer[nearest]
    return np.concatenate((outer, cut))


