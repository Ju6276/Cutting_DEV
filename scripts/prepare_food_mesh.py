#!/usr/bin/env python3
"""Turn a scanned food surface mesh into a DiSECt tetrahedral mesh.

Input: watertight triangle surface (metres, z up), e.g. an OmniObject3D scan converted to
OBJ. Output (.msh, tetra cells) in DiSECt's object frame (y up): resting on y = 0,
centred in x/z, longest horizontal axis along x so the knife (cut-plane normal x, blade
along z) cuts across it; elongated objects (cucumber, potato) are laid down. The surface
is decimated, then tetrahedralized with fTetWild (wildmeshing API, as in DiSECt's
``cutting/tetrahedralization.py``) using a loose surface envelope (~1 mm) so the explicit
FEM gets no sub-millimetre slivers. A small shift along x keeps vertices off the cut plane.
"""
import argparse
import itertools
import json
from pathlib import Path

import meshio
import numpy as np
import trimesh
import wildmeshing as wm


def orient(vertices, elongated, x_offset):
    """Scan vertices in the object frame (see module docstring)."""
    centred = vertices - vertices.mean(0)
    w, vec = np.linalg.eigh(centred.T @ centred)
    extents = np.ptp(centred @ vec, axis=0)
    if extents.max() / np.median(extents) > elongated:
        order = np.argsort(extents)                    # longest -> x, shortest -> y (up)
        basis = vec[:, [order[2], order[0], order[1]]].copy()
        if np.linalg.det(basis) < 0:
            basis[:, 2] *= -1
        v = centred @ basis
    else:
        v = np.c_[vertices[:, 0], vertices[:, 2], -vertices[:, 1]]  # scan z up -> y up
    xz = v[:, [0, 2]] - v[:, [0, 2]].mean(0)
    w, vec = np.linalg.eigh(xz.T @ xz)
    major = vec[:, np.argmax(w)]
    angle = np.arctan2(major[1], major[0])
    c, s = np.cos(-angle), np.sin(-angle)
    v = np.c_[c * xz[:, 0] - s * xz[:, 1], v[:, 1], s * xz[:, 0] + c * xz[:, 1]]
    v[:, 1] -= v[:, 1].min()
    v[:, [0, 2]] -= (v[:, [0, 2]].max(0) + v[:, [0, 2]].min(0)) / 2
    v[:, 0] += x_offset
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--faces', type=int, default=1500, help='surface decimation target')
    parser.add_argument('--edge-length-r', type=float, default=.06, help='target edge / bbox diagonal')
    parser.add_argument('--epsilon', type=float, default=.008, help='surface envelope / bbox diagonal')
    parser.add_argument('--stop-quality', type=float, default=8)
    parser.add_argument('--max-its', type=int, default=40)
    parser.add_argument('--elongated', type=float, default=1.3,
                        help='principal-extent ratio above which the object is laid down along x')
    parser.add_argument('--x-offset', type=float, default=3.7e-4)
    args = parser.parse_args()
    mesh = trimesh.load(args.surface, force='mesh')
    if len(mesh.faces) > args.faces:
        mesh = mesh.simplify_quadric_decimation(face_count=args.faces)
    v = orient(np.asarray(mesh.vertices, float), args.elongated, args.x_offset)
    tet = wm.Tetrahedralizer(stop_quality=args.stop_quality, max_its=args.max_its,
                             edge_length_r=args.edge_length_r, epsilon=args.epsilon)
    tet.set_mesh(v, np.asarray(mesh.faces, np.int32))
    tet.tetrahedralize()
    points, tets = tet.get_tet_mesh()[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(args.output, meshio.Mesh(points, [('tetra', tets)]), file_format='gmsh22')
    p = points[tets]
    edges = np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=-1)
                      for i, j in itertools.combinations(range(4), 2)], 1)
    info = {'surface': str(args.surface), 'vertices': int(len(points)), 'tets': int(len(tets)),
            'min_edge_mm': round(float(edges.min()) * 1000, 3),
            'extent_m': np.ptp(points, 0).round(4).tolist(), **{k: v for k, v in vars(args).items()
                                                                if k not in ('surface', 'output')}}
    args.output.with_suffix('.json').write_text(json.dumps(info, indent=2, default=str) + '\n')
    print(json.dumps(info, default=str))


if __name__ == '__main__':
    main()
