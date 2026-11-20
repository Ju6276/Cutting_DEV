"""Choose a fixed central-edge support pose for a curved knife; preserve its CAD shape."""
import argparse,json,shutil,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from cutting_world_model.robot_platform import frame_poses

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--tip-down-deg',type=float,default=8.25)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 shutil.copytree(a.source,a.output)
 desc=json.loads((a.output/'platform.json').read_text())
 meta=json.loads((a.output/'cutting_geometry.json').read_text())
 data=np.load(a.output/'cutting_geometry.npz')
 angle=np.deg2rad(a.tip_down_deg)
 R=Rotation.from_euler('x',angle).as_matrix()
 v=data['vertices_disect'];edge=data['edge_disect']
 rotated=v@R.T
 lowest=int(rotated[:,1].argmin());contact=v[lowest].copy()
 contact[0]=float(np.mean(edge[:,0]))
 span=np.ptp(edge[:,2]);fraction=(contact[2]-edge[:,2].min())/span
 if not .35<fraction<.65:raise ValueError('Fixed pose does not support the central cutting edge')
 K=np.asarray(meta['knife_mount_from_disect'])
 control=Rotation.from_quat(K[3:]).apply(contact)+K[:3]
 xml=(a.source/'robot.urdf').read_text().replace(str(a.source),str(a.output))
 tree=ET.fromstring(xml)
 for j in tree.findall('joint'):
  if j.find('child').get('link')=='knife_control':
   j.find('origin').set('xyz',' '.join(map(str,control)))
 m={k:desc[k] for k in ('robot_base','robot_tip','joint_names')}
 m['robot_xml']=ET.tostring(tree,encoding='unicode')
 q=np.asarray(desc['q_initial_rad'])
 original=frame_poses(m,q)['tool0']
 target=original.copy();target[:3,:3]=Rotation.from_euler('y',angle).as_matrix()@original[:3,:3]
 def residual(x):
  T=frame_poses(m,x)['tool0']
  return np.r_[T[:3,3]-target[:3,3],Rotation.from_matrix(target[:3,:3].T@T[:3,:3]).as_rotvec()]
 solved=least_squares(residual,q,xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=200)
 if np.max(np.abs(residual(solved.x)))>1e-8:raise ValueError('Nominal UR10e support IK did not converge')
 revision='ur10e_ati_ikea_midblade_v1'
 cfg=desc['configuration']
 cfg.update(revision=revision,q_initial_rad=solved.x.tolist(),cutting_center_frame='knife_control',
   fixed_tip_down_deg=a.tip_down_deg,pose_rule='central_curved_edge_support_v1')
 (a.output/'robot.urdf').write_text(m['robot_xml'])
 desc.update(revision=revision,q_initial_rad=solved.x.tolist(),robot_urdf=str(a.output/'robot.urdf'),
   curved_edge_support=dict(tip_down_deg=a.tip_down_deg,edge_fraction=float(fraction),
     control_in_mount_m=control.tolist(),minimum_full_mesh_gap_at_control_m=float(rotated[:,1].min()-(R@contact)[1])))
 desc.pop('runtime',None)
 desc['source_hashes']['support_pose_source']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (a.output/'platform.json').write_text(json.dumps(desc,indent=2))
 # Geometry is unchanged; only file locations and robot nominal configuration move.
 for key in ('geometry','sdf'):
  if key in meta:meta[key]=str(a.output/Path(meta[key]).name)
 (a.output/'cutting_geometry.json').write_text(json.dumps(meta,indent=2))
 print(json.dumps(dict(q=solved.x.tolist(),support=desc['curved_edge_support'],configuration=cfg),indent=2))
if __name__=='__main__':main()
