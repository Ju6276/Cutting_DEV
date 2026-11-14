"""Lay an uncut tetrahedral food mesh on a stable support face without remeshing."""
import argparse
import json
from pathlib import Path
import meshio
import numpy as np
import trimesh

def stable_pose(points, tets):
    tet = points[tets]
    volume = np.abs(np.linalg.det(tet[:, 1:] - tet[:, :1])) / 6
    center = np.average(tet.mean(1), weights=volume, axis=0)
    hull = trimesh.convex.convex_hull(points)
    transforms, probability = hull.compute_stable_poses(center_mass=center, threshold=0.)
    if not len(transforms):
        raise ValueError('No gravity-supported pose found')
    heights = np.array([(t[:3,:3] @ center + t[:3,3])[2] for t in transforms])
    chosen = int(np.argmin(heights))
    t = transforms[chosen]
    zup = trimesh.transform_points(points, t)
    y_up = zup[:, [0,2,1]].copy()
    y_up[:,2] *= -1
    horizontal = y_up[:,[0,2]] - y_up[:,[0,2]].mean(0)
    _, vectors = np.linalg.eigh(horizontal.T @ horizontal)
    axis = vectors[:,-1]
    yaw = np.arctan2(axis[1], axis[0])
    c, s = np.cos(yaw), np.sin(yaw)
    y_up[:,0] = c * horizontal[:,0] + s * horizontal[:,1]
    y_up[:,2] = -s * horizontal[:,0] + c * horizontal[:,1]
    y_up[:,1] -= y_up[:,1].min()
    y_up[:,[0,2]] -= (y_up[:,[0,2]].max(0)+y_up[:,[0,2]].min(0))/2
    y_up[:,0] += 3.7e-4
    return y_up, dict(selected=chosen, candidate_com_height_m=heights.tolist(),
                      candidate_probability=probability.tolist(),
                      original_com_m=center.tolist(), selected_transform_z_up=t.tolist(),
                      extent_m=np.ptp(y_up,axis=0).tolist(),
                      scope='Geometric support candidate; physical settling must be checked')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    mesh=meshio.read(a.mesh)
    points, report=stable_pose(mesh.points,mesh.cells_dict['tetra'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    before=mesh.points[mesh.cells_dict['tetra']]
    after=points[mesh.cells_dict['tetra']]
    signed_before=np.linalg.det(before[:,1:]-before[:,:1])
    signed_after=np.linalg.det(after[:,1:]-after[:,:1])
    if not np.allclose(signed_before,signed_after,rtol=1e-8,atol=1e-18):
        raise AssertionError('Pose preparation changed tetrahedral volumes or orientation')
    report['max_relative_volume_change']=float(np.max(np.abs(signed_after-signed_before)/np.abs(signed_before)))
    mesh.points=points
    meshio.write(a.output,mesh,file_format='gmsh22')
    report['source']=str(a.mesh)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
