"""Prepare a flat-bottom half from an existing uncut, y-up tetrahedral food.
No fixed particles or material/contact changes. Run in the food-mesh environment.
The retained upper geometric half is capped and remeshed with the same fTetWild
settings as prepare_food_mesh.py; this is an asset preparation comparison, not
a mesh-controlled intervention.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import time
import meshio
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
import wildmeshing as wm

def signed_volumes(points, tets):
    t = points[tets]
    return np.linalg.det(t[:, 1:] - t[:, :1]) / 6

def boundary_faces(points, tets):
    oriented = []
    for opposite in range(4):
        ids = [j for j in range(4) if j != opposite]
        face = tets[:, ids].copy()
        v = points[face]
        inward = np.einsum('ij,ij->i', np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]),
                          points[tets[:,opposite]]-v[:,0]) > 0
        face[inward] = face[inward][:,[0,2,1]]
        oriented.append(face)
    faces = np.concatenate(oriented)
    _, inverse, counts = np.unique(np.sort(faces,axis=1),axis=0,return_inverse=True,return_counts=True)
    return faces[counts[inverse] == 1]

def make_half(points, faces, plane_y):
    poly = vtk.vtkPolyData()
    verts = vtk.vtkPoints(); verts.SetData(numpy_to_vtk(points,deep=True)); poly.SetPoints(verts)
    cells=vtk.vtkCellArray()
    cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.column_stack((np.full(len(faces),3),faces)).astype(np.int64).ravel(),deep=True))
    poly.SetPolys(cells)
    plane=vtk.vtkPlane();plane.SetOrigin(0,plane_y,0);plane.SetNormal(0,1,0)
    planes=vtk.vtkPlaneCollection();planes.AddItem(plane)
    clip=vtk.vtkClipClosedSurface();clip.SetInputData(poly);clip.SetClippingPlanes(planes)
    clip.GenerateFacesOn();clip.SetTolerance(1e-10);clip.Update()
    tri=vtk.vtkTriangleFilter();tri.SetInputConnection(clip.GetOutputPort());tri.Update()
    result=tri.GetOutput()
    p=vtk_to_numpy(result.GetPoints().GetData()).copy()
    f=vtk_to_numpy(result.GetPolys().GetData()).reshape(-1,4)[:,1:].copy()
    if p[:,1].min() < plane_y-1e-8 or len(f)==0:
        raise ValueError('Clip did not retain the upper half')
    e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    _,counts=np.unique(e,axis=0,return_counts=True)
    if not np.all(counts==2):raise ValueError('Half surface is not closed')
    return p,f

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mesh',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.mesh=args.mesh.resolve();args.output=args.output.resolve()
    if args.output.exists():raise FileExistsError(args.output)
    start=time.monotonic()
    m=meshio.read(args.mesh);t=m.cells_dict['tetra'];v=m.points
    plane_y=(v[:,1].min()+v[:,1].max())/2
    surface,faces=make_half(v,boundary_faces(v,t),plane_y)
    surface[:,1]-=plane_y
    tet=wm.Tetrahedralizer(stop_quality=8,max_its=40,edge_length_r=.06,epsilon=.008)
    tet.set_mesh(surface,faces.astype(np.int32))
    # fTetWild writes __tracked_surface.stl relative to the working directory.
    # Keep generated meshing artifacts with the server-side asset, never in source.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    previous=Path.cwd()
    try:
        os.chdir(args.output.parent)
        tet.tetrahedralize()
    finally:
        os.chdir(previous)
    p,cells=tet.get_tet_mesh()[:2]
    # Preserve the intended planar support face after the remesher's surface envelope.
    envelope=.008*np.linalg.norm(np.ptp(surface,axis=0))
    before=signed_volumes(p,cells)
    cap=np.abs(p[:,1])<=envelope
    p[cap,1]=0.
    after=signed_volumes(p,cells)
    if np.any(before*after<=0) or np.any(p[:,1]<-1e-10):
        raise ValueError('Planar support projection inverted or collapsed tetrahedra')
    if np.count_nonzero(cap)<3:raise ValueError('No planar support face')
    edges=np.concatenate([np.linalg.norm(p[cells[:,i]]-p[cells[:,j]],axis=1) for i,j in itertools.combinations(range(4),2)])
    if not np.isfinite(p).all():raise ValueError('Nonfinite mesh')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    meshio.write(args.output,meshio.Mesh(p,[('tetra',cells)]),file_format='gmsh22')
    info=dict(source=str(args.mesh),output=str(args.output),retained='upper geometric half',
      preparation_plane_y_m=float(plane_y),extent_m=np.ptp(p,axis=0).tolist(),
      vertices=len(p),tetrahedra=len(cells),min_edge_mm=float(edges.min()*1000),
      whole_volume_m3=float(np.abs(signed_volumes(v,t)).sum()),half_volume_m3=float(np.abs(after).sum()),
      cap_vertices=int(cap.sum()),cap_envelope_m=float(envelope),
      cap_projection_volume_change_fraction=float((np.abs(after).sum()-np.abs(before).sum())/np.abs(before).sum()),
      fixed_particles=0,wall_seconds=time.monotonic()-start,
      meshing=dict(stop_quality=8,max_its=40,edge_length_r=.06,epsilon=.008),
      scope='Half preparation changes mass, height and mesh; not a pure support-shape ablation')
    args.output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)
if __name__=='__main__':main()
