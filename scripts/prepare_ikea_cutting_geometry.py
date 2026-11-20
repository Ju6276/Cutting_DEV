"""Canonical IKEA CAD mesh, blade edge and robot/material transforms (metres).

CAD registration and mass estimates are separate; this does not rescale the knife
to a catalogue bounding box. The edge is the lower centre-plane CAD contour from
its heel (lowest vertex in the mounted pose) to the foremost tip.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform',type=Path,required=True)
    a=p.parse_args(); root=a.platform
    manifest=json.loads((root/'platform.json').read_text())
    mesh=trimesh.load(root/'knife.stl',force='mesh')
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('Material contact needs a closed oriented knife mesh')
    x=float(mesh.bounds[:,0].mean())
    segments=trimesh.intersections.mesh_plane(mesh,[1,0,0],[x,0,0])
    heel=float(mesh.vertices[np.argmax(mesh.vertices[:,2]),1])
    tip=float(segments[:,:,1].max())
    ys=np.unique(np.r_[heel,segments[:,:,1].reshape(-1)])
    ys=ys[(ys>=heel)&(ys<=tip)]
    edge=[]
    for y in ys:
        z=[]
        for s in segments:
            dy=s[1,1]-s[0,1]
            if abs(dy)<1e-12:continue
            t=(y-s[0,1])/dy
            if -1e-7<=t<=1+1e-7:z.append(s[0,2]+t*(s[1,2]-s[0,2]))
        if z:edge.append([x,y,max(z)])
    edge=np.asarray(edge)
    if len(edge)<3 or np.any(np.diff(edge[:,1])<=0):raise ValueError('Invalid CAD edge')
    center=np.array([x,(heel+tip)/2, float(np.mean(edge[:,2]))])
    # D: x thin, y up, z along edge. B: x thin, y along edge, z up.
    d_from_k=np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=float)
    b_from_d=np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=float)
    vd=(mesh.vertices-center)@d_from_k.T
    ed=(edge-center)@d_from_k.T
    vb=vd@b_from_d.T; eb=ed@b_from_d.T
    eb=eb[np.argsort(eb[:,1])]
    hull=trimesh.Trimesh(vd,mesh.faces,process=False).convex_hull.vertices
    path=root/'cutting_geometry.npz'
    np.savez_compressed(path,vertices_disect=vd,vertices_blademaster=vb,
        faces=mesh.faces,edge_disect=ed,edge_blademaster=eb,
        board_vertices_disect=hull,
        board_vertices_blademaster=hull@b_from_d.T,
        knife_mount_from_disect=np.r_[center,Rotation.from_matrix(d_from_k.T).as_quat()],
        disect_from_blademaster=np.r_[np.zeros(3),Rotation.from_matrix(b_from_d.T).as_quat()])
    meta=dict(revision='ikea_cad_mesh_v1',geometry=str(path.resolve()),
        geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        mesh_source_sha256=hashlib.sha256((root/'knife.stl').read_bytes()).hexdigest(),
        mesh_vertices=len(vd),mesh_faces=len(mesh.faces),edge_points=len(edge),
        edge_span_m=tip-heel,full_length_m=float(mesh.extents[1]),
        coordinate_convention='D x-thin/y-up/z-edge; B x-thin/y-edge/z-up; rigid mounting',
        knife_mount_from_disect=np.r_[center,Rotation.from_matrix(d_from_k.T).as_quat()].tolist(),
        disect_from_blademaster=np.r_[np.zeros(3),Rotation.from_matrix(b_from_d.T).as_quat()].tolist(),
        approximation='CAD geometry; not measured edge radius or sharpening; no shape rescaling')
    (root/'cutting_geometry.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
