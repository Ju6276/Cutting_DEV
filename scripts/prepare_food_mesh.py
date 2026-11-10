#!/usr/bin/env python3
"""Turn a scanned food surface mesh into a DiSECt tetrahedral mesh.

Input: watertight triangle surface (metres, z up), e.g. OmniObject3D Object.usd converted
to OBJ. Output: tetrahedral mesh (.msh, tetra cells) in DiSECt's frame (y up), resting
on y = 0, centred in x/z, longest horizontal axis along x so the knife (cut plane
normal x, blade along z) cuts across it. Tetrahedralization uses fTetWild through the
wildmeshing Python API, as DiSECt's own ``cutting/tetrahedralization.py`` does.
"""
import argparse
import json
from pathlib import Path

import meshio
import numpy as np
import trimesh


KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


def voxel_tets(surface, pitch):
    """Uniform elements: occupied pitch-cubes, each split into six tetrahedra (Kuhn split,
    conforming), then boundary nodes moved toward the scanned surface by at most 0.25 pitch.
    The cut plane x = 0 lies on cell centres, never on nodes."""
    lo, hi = surface.bounds
    nx = int(np.ceil(max(-lo[0], hi[0]) / pitch - .5)) + 1
    xs = np.arange(-nx, nx + 1) * pitch
    ys = np.arange(0, hi[1] + pitch, pitch) + lo[1] + pitch / 2
    zs = np.arange(lo[2], hi[2] + pitch, pitch) + pitch / 2
    centres = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1).reshape(-1, 3)
    inside = surface.contains(centres).reshape(len(xs), len(ys), len(zs))
    node_index, points, tets = {}, [], []
    def node(i, j, k):
        key = (i, j, k)
        if key not in node_index:
            node_index[key] = len(points)
            points.append((xs[0] - pitch / 2 + i * pitch, ys[0] - pitch / 2 + j * pitch,
                           zs[0] - pitch / 2 + k * pitch))
        return node_index[key]
    for i, j, k in zip(*np.nonzero(inside)):
        corners = [node(i + (c >> 2 & 1), j + (c >> 1 & 1), k + (c & 1)) for c in range(8)]
        for t in KUHN:
            tets.append([corners[c] for c in t])
    points, tets = np.asarray(points, float), np.asarray(tets, np.int64)
    # Boundary nodes: on faces used by exactly one tetrahedron.
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), 1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    boundary = np.unique(unique[counts == 1])
    closest, _, _ = trimesh.proximity.closest_point(surface, points[boundary])
    move = closest - points[boundary]
    length = np.linalg.norm(move, axis=1, keepdims=True)
    move *= np.minimum(1., .25 * pitch / np.maximum(length, 1e-12))
    bottom = points[boundary, 1] <= points[:, 1].min() + 1e-9
    move[bottom, 1] = 0.                       # keep the resting layer flat
    points[boundary] += move
    points[:, 1] -= points[:, 1].min()
    p = points[tets]
    volume = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    tets[volume < 0] = tets[volume < 0][:, [0, 2, 1, 3]]
    return points, tets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--edge-length-r', type=float, default=0.06,
                        help='target edge length relative to the bounding-box diagonal')
    parser.add_argument('--stop-quality', type=float, default=10)
    parser.add_argument('--max-its', type=int, default=40)
    parser.add_argument('--x-offset', type=float, default=3.7e-4,
                        help='shift along the cut normal so no vertex lies on the cut plane x = 0')
    parser.add_argument('--mesher', choices=('voxel', 'tetgen', 'ftetwild'), default='voxel',
                        help='tetgen: uniform ~pitch elements (stable for explicit FEM); '
                             'ftetwild: surface-preserving, produces sub-mm slivers')
    parser.add_argument('--pitch', type=float, default=.006, help='element size for voxel/tetgen (m)')
    parser.add_argument('--faces', type=int, default=1200, help='surface decimation target before meshing')
    parser.add_argument('--elongated', type=float, default=1.3,
                        help='principal-extent ratio above which the object is laid down along x')
    args = parser.parse_args()
    mesh = trimesh.load(args.surface, force='mesh')
    if len(mesh.faces) > args.faces:
        mesh = mesh.simplify_quadric_decimation(face_count=args.faces)
    v = np.asarray(mesh.vertices, float)
    centred = v - v.mean(0)
    w, vec = np.linalg.eigh(centred.T @ centred)
    extents = np.ptp(centred @ vec, axis=0)
    if extents.max() / np.median(extents) > args.elongated:
        # Elongated (cucumber, potato): lie down; longest axis -> x, shortest -> y (up).
        order = np.argsort(extents)
        basis = vec[:, [order[2], order[0], order[1]]]
        if np.linalg.det(basis) < 0:
            basis[:, 2] *= -1
        v = centred @ basis
        faces = np.asarray(mesh.faces, np.int32)
    else:
        v = np.c_[v[:, 0], v[:, 2], -v[:, 1]]      # scan z up -> y up
        faces = np.asarray(mesh.faces, np.int32)
    xz = v[:, [0, 2]] - v[:, [0, 2]].mean(0)
    w, vec = np.linalg.eigh(xz.T @ xz)
    major = vec[:, np.argmax(w)]
    angle = np.arctan2(major[1], major[0])
    c, s = np.cos(-angle), np.sin(-angle)
    x, z = xz[:, 0], xz[:, 1]
    v = np.c_[c * x - s * z, v[:, 1], s * x + c * z]  # longest horizontal axis -> x
    v[:, 1] -= v[:, 1].min()
    v[:, [0, 2]] -= (v[:, [0, 2]].max(0) + v[:, [0, 2]].min(0)) / 2
    v[:, 0] += args.x_offset
    if args.mesher == 'voxel':
        points, tets = voxel_tets(trimesh.Trimesh(v, faces, process=True), args.pitch)
    elif args.mesher == 'tetgen':
        import tetgen
        surface = trimesh.Trimesh(v, faces, process=True)
        # Uniform surface at the element size: voxelize, extract, smooth (no sub-pitch features).
        grid = surface.voxelized(args.pitch).fill()
        uniform = grid.marching_cubes
        uniform.apply_transform(grid.transform)
        trimesh.smoothing.filter_taubin(uniform, iterations=20)
        uniform = uniform.simplify_quadric_decimation(
            face_count=max(200, int(2 * uniform.area / args.pitch ** 2)))
        uniform.vertices[:, 1] -= uniform.vertices[:, 1].min()
        uniform.vertices[:, [0, 2]] -= (uniform.vertices[:, [0, 2]].max(0) + uniform.vertices[:, [0, 2]].min(0)) / 2
        uniform.vertices[:, 0] += args.x_offset
        tg = tetgen.TetGen(np.asarray(uniform.vertices), np.asarray(uniform.faces))
        points, tets = tg.tetrahedralize(order=1, mindihedral=18, minratio=1.4,
                                         maxvolume=args.pitch ** 3 / 2)[:2]
    else:
        import wildmeshing as wm
        tet = wm.Tetrahedralizer(stop_quality=args.stop_quality, max_its=args.max_its,
                                 edge_length_r=args.edge_length_r)
        tet.set_mesh(v, faces)
        tet.tetrahedralize()
        points, tets = tet.get_tet_mesh()[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(args.output, meshio.Mesh(points, [('tetra', tets)]), file_format='gmsh22')
    import itertools
    p = points[tets]
    edges = np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=-1) for i, j in itertools.combinations(range(4), 2)], 1)
    info = {'surface': str(args.surface), 'mesher': args.mesher, 'vertices': int(len(points)), 'tets': int(len(tets)),
            'min_edge_mm': round(float(edges.min()) * 1000, 3),
            'p1_edge_mm': round(float(np.quantile(edges.min(1), .01)) * 1000, 3),
            'extent_m': (points.max(0) - points.min(0)).round(4).tolist(),
            'edge_length_r': args.edge_length_r}
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info))


if __name__ == '__main__':
    main()
