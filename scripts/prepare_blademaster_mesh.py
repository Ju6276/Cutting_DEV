"""Sample intact tetrahedral food geometry for BladeMaster (no pre-cut interface).

Run with the food_meshes Python environment (meshio, ansys-mapdl-reader).
Uniform sub-cell samples have exactly the volume assumed by the BM solver.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

def load_uncut(path):
    if path.suffix=='.rst':
        from ansys.mapdl import reader
        mesh=reader.read_binary(str(path)).mesh
        lookup={int(node):i for i,node in enumerate(mesh.nnum)}
        tetrahedra=[]
        for element in mesh.elem:
            node_ids=list(dict.fromkeys(int(n) for n in element[10:18]))
            if 0 in node_ids:continue
            if len(node_ids)!=4:raise ValueError('Expected four-node tetrahedral volume elements')
            tetrahedra.append([lookup[n] for n in node_ids])
        return np.asarray(mesh.nodes),np.asarray(tetrahedra)
    import meshio
    mesh=meshio.read(path)
    return np.asarray(mesh.points),np.asarray(mesh.cells_dict['tetra'])

def contains_tetra(points,vertices,tets):
    inside=np.zeros(len(points),dtype=bool)
    for tet in vertices[tets]:
        mask=(~inside)&(points>=tet.min(0)-1e-12).all(1)&(points<=tet.max(0)+1e-12).all(1)
        ids=np.flatnonzero(mask)
        if not len(ids):continue
        matrix=(tet[1:]-tet[0]).T
        coord=np.linalg.solve(matrix,(points[ids]-tet[0]).T).T
        inside[ids]=(coord>=-1e-10).all(1)&(coord.sum(1)<=1+1e-10)
    return inside

def initial_components(points,origin,spacing):
    parent=list(range(len(points)))
    def root(i):
        while i!=parent[i]:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    owners={}
    cell=np.floor((points-origin)/spacing).astype(int)
    for i,c in enumerate(cell):
        for offset in itertools.product((0,1),repeat=3):
            node=tuple(c+offset)
            if node in owners:parent[root(i)]=root(owners[node])
            else:owners[node]=i
    return len({root(i) for i in range(len(points))})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spacing',type=float,default=.006)
    a=p.parse_args()
    if a.spacing<=0:raise ValueError('Positive spacing required')
    vertices,tets=load_uncut(a.mesh)
    # Proper rotation, determinant +1: x normal, -z edge, y up.
    rotated=vertices[:,[0,2,1]]*np.array([1.,-1.,1.])
    translation=np.r_[(rotated.min(0)+rotated.max(0))[:2]/2,rotated[:,2].min()]
    rotated-=translation
    size=np.ptp(rotated,axis=0)
    cells=np.ceil(size/a.spacing-1e-9).astype(int)
    origin=-np.r_[size[:2]/2,0.]
    grid=np.array(list(itertools.product(*[origin[i]+(np.arange(2*cells[i])+.5)*a.spacing/2 for i in range(3)])))
    points=grid[contains_tetra(grid,rotated,tets)]
    if len(points)<4:raise ValueError('Mesh produced too few samples')
    components=initial_components(points,origin,a.spacing)
    if components!=1:raise ValueError('Initial intact MPM body has multiple components')
    volume=float(abs(np.linalg.det(rotated[tets[:,1:]]-rotated[tets[:,:1]])).sum()/6)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,reference_points=points,size=size,grid_spacing=np.array(a.spacing),
                        source_vertices=rotated,source_tetrahedra=tets)
    report=dict(name=a.name,source=str(a.mesh.resolve()),source_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest(),
        asset_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),size_m=size.tolist(),grid_spacing_m=a.spacing,
        particle_count=len(points),particle_volume_m3=(a.spacing/2)**3,
        source_volume_m3=volume,sampled_volume_m3=len(points)*(a.spacing/2)**3,
        initial_mpm_components=components,predefined_cut_interface=False,
        coordinate_transform='[x,y,z] source y-up -> [x,-z,y] BM z-up; XY bbox centred; lowest source z=0',
        translation_after_rotation_m=translation.tolist(),
        bounds_m=[points.min(0).tolist(),points.max(0).tolist()],
        scope='Geometry only; source tetrahedra contain no prepared DiSECt cut plane; no fracture parameters transferred')
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
