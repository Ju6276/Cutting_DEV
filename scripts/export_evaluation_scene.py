"""Adapt frozen evaluation recordings to the shared studio renderer. No rendering or physics changes."""
import argparse,json
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
import render_episode as r

def color_lookup(reference, target):
 """Transfer measured scan appearance by PCA-aligned coordinates, never geometry."""
 ref=dict(np.load(reference/'episode.npz'))
 app=r.load_appearance(reference/'appearance.npz',reference/'episode.npz')
 src=r.display_vertices(ref,0,app)[:len(app['indices'])].reshape(-1,3,3).mean(1)
 rgb=app['rgb']/255.
 # Principal axes only align color coordinates. Sign is conventional, not recovered UV.
 def normalized(x):
  c=x.mean(0);_,axes=np.linalg.eigh(np.cov((x-c).T));axes=axes[:,::-1]
  for k in range(3):
   if axes[np.argmax(abs(axes[:,k])),k]<0:axes[:,k]*=-1
  y=(x-c)@axes
  return y/np.maximum(np.ptp(y,axis=0),1e-6),c,axes,np.ptp(y,axis=0)
 source,_,_,_=normalized(src)
 dest,c,axes,scale=normalized(target)
 pd=r.polydata(source,np.empty((0,3),dtype=int))
 loc=vtk.vtkStaticPointLocator();loc.SetDataSet(pd);loc.BuildLocator()
 def lookup(x):
  y=(x-c)@axes/np.maximum(scale,1e-6)
  return rgb[[loc.FindClosestPoint(v) for v in y]]
 return lookup

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--evaluation',type=Path,required=True)
 p.add_argument('--group',type=int,required=True)
 p.add_argument('--snapshot',action='store_true',help='Export settled initial geometry without an episode or policy')
 p.add_argument('--source-root',type=Path,action='append',required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--backend',choices=('DiSECt','BladeMaster'),required=True)
 p.add_argument('--appearance-reference',type=Path,help='Verified same-topology textured recording')
 p.add_argument('--color-reference',type=Path,help='Approximate color transfer only; never replaces geometry')
 a=p.parse_args()
 gdir=a.evaluation/'geometry'
 initial=json.loads((gdir/'0000/report.json').read_text())
 group=initial['groups'][a.group]; w=group['first_world']
 if (group.get('representation')=='material_points') != (a.backend=='BladeMaster'):
  raise ValueError('Backend argument contradicts recorded representation')
 if a.snapshot:
  row=dict(case=w,steps=0,scope='Settled initial geometry; no policy or control steps')
 else:
  rows=[json.loads(x) for x in (a.evaluation/'episodes.jsonl').read_text().splitlines()]
  row=next(x for x in rows if x['case']==w)
 last=int(row['steps'])
 root=gdir/'0000'/group['path']
 d=dict(np.load(root/'episode.npz'))
 task=json.loads((root/'task.json').read_text())
 robot,sources=r.visual_actors(d,task,a.source_root)
 robot=[entry for entry in robot if str(d['geom_body'][entry[0]])!='b_bot_cam_cables_link']
 # The combined URDF knife/adapter is only an appearance mesh. Display the
 # backend's actual collision blade once, not the differently sized URDF blade.
 tool_entries=[entry for entry in robot if str(d['geom_body'][entry[0]])=='b_bot_right_inner_knuckle']
 robot=[entry for entry in robot if str(d['geom_body'][entry[0]])!='b_bot_right_inner_knuckle']
 # Keep the original grip/handle; remove only the old blade beyond its shoulder.
 for k,old,local in tool_entries:
  old.GetMapper().Update()
  plane=vtk.vtkPlane();plane.SetOrigin(-.04786805,0,0);plane.SetNormal(1,0,0)
  clip=vtk.vtkClipPolyData();clip.SetInputData(old.GetMapper().GetInput());clip.SetClipFunction(plane);clip.Update()
  handle=r.actor(clip,(.61,.63,.65),specular=.5)
  robot.append((k,handle,local))
 states=[]
 for t in range(last+1):
  snap=dict(np.load(gdir/('%04d'%t)/group['path']/'world_states.npz'))
  states.append({k:snap[k][0] for k in ('particles','body_pos','body_mat')})
 if a.backend=='BladeMaster':
  if 'ee_pose_base' in initial['training']:
   if 'base_position_world' in d and 'base_rotation_world' in d:
    base_position=d['base_position_world'];base_rotation=d['base_rotation_world'].reshape(3,3)
   elif task['robot_base'] in list(d['geom_body']):
    base_idx=list(d['geom_body']).index(task['robot_base'])
    base_position=states[0]['body_pos'][base_idx]
    base_rotation=states[0]['body_mat'][base_idx].reshape(3,3)
   else:
    raise ValueError('Recording lacks robot-base world transform; recapture with the current snapshot writer')
   vertices=np.array(task['board_pose_geometry']['vertices_in_ee'])
   blade_points=[]
   for t in range(last+1):
    pose=np.array(json.loads((gdir/('%04d'%t)/'report.json').read_text())['training']['ee_pose_base'][w])
    xyz=pose[3:6];qw=pose[6]
    basepoints=vertices+2*np.cross(xyz,np.cross(xyz,vertices)+qw*vertices)+pose[:3]
    st=states[t]
    blade_points.append(basepoints@base_rotation.T+base_position)
   blade_points=np.array(blade_points)
   faces=[]
   for k in range(1,4):faces.extend([(0,k+1,k),(5,5+k,5+k+1)])
   for k in range(5):
    j=(k+1)%5;faces.extend([(k,j,j+5),(k,j+5,k+5)])
   blade_faces=np.array(faces)
  else:
   # Earlier captures have the identical physical transform exported separately
   # from their recorded EE trajectories, never a visual alignment adjustment.
   blade_record=dict(np.load(a.evaluation/'physical_blades.npz'))
   blade_points=blade_record['group_%02d'%a.group][:last+1]
   blade_faces=blade_record['faces']
 else:
  blade_points=np.stack([np.load(gdir/('%04d'%t)/group['path']/'episode.npz')['knife'][0] for t in range(last+1)])
  blade_faces=d['knife_tri']
 # BM's blade is at another fixed orientation. Represent its fixed connection
 # with a display-only metal shank from the retained handle to the blade spine.
 # The blade, robot, material, contact solver and force lever arms are unchanged.
 mount_residual=0.
 if a.backend=='BladeMaster' or task.get('tool_revision')=='disect_knife_rigid_mount_v1':
  k=tool_entries[0][0]; local=tool_entries[0][2]
  # Check the mount against every recorded body pose (not a fitted moving link).
  all_local=np.stack([(bp-st['body_pos'][k])@st['body_mat'][k].reshape(3,3) for bp,st in zip(blade_points,states)])
  mount_residual=float(np.linalg.norm(all_local-all_local[0],axis=-1).max())
  if mount_residual>1e-5:raise ValueError('Blade is not rigid relative to the gripper; refuse a fabricated fixed mount')
  upper=np.argsort(blade_points[0,:,2])[-4:]
  end=all_local[0,upper].mean(0)
  start=local[:3,:3]@np.array([-.044,.028,-.015])+local[:3,3]
  line=vtk.vtkLineSource();line.SetPoint1(*start);line.SetPoint2(*end)
  tube=vtk.vtkTubeFilter();tube.SetInputConnection(line.GetOutputPort());tube.SetRadius(.006);tube.SetNumberOfSides(32);tube.CappingOn()
  robot.append((k,r.actor(tube,(.61,.63,.65),specular=.5),np.eye(4)))
  # Collar overlaps the spine rather than ending at an unconnected floating point.
  collar=vtk.vtkSphereSource();collar.SetCenter(*end);collar.SetRadius(.007);collar.SetThetaResolution(24);collar.SetPhiResolution(16)
  robot.append((k,r.actor(collar,(.61,.63,.65),specular=.5),np.eye(4)))
 name=group['name']; style='cucumber' if 'cucumber' in name else 'potato' if 'potato' in name else 'prism'
 color=r.MATERIAL_STYLES[style][0]
 points=states[0]['particles']
 if group.get('representation')=='material_points':
  poly=vtk.vtkPolyData(); pts=vtk.vtkPoints(); pts.SetData(numpy_to_vtk(points,deep=True)); poly.SetPoints(pts)
  # Display-only isosurface from measured material points. It is never used
  # for contacts, task completion or force calculation.
  cloud_bounds=np.concatenate([s['particles'] for s in states])
  radius=float(group['particle_radius_m'])*2.5
  lo=cloud_bounds.min(0)-radius;hi=cloud_bounds.max(0)+radius
  splat=vtk.vtkGaussianSplatter();splat.SetInputData(poly)
  bounds=np.column_stack((lo,hi)).ravel();splat.SetModelBounds(*bounds)
  dims=np.maximum(32,np.ceil((hi-lo)/.001).astype(int));splat.SetSampleDimensions(*dims)
  splat.SetRadius(radius/max(hi-lo));splat.SetExponentFactor(-5);splat.SetAccumulationModeToSum()
  splat.ScalarWarpingOff();splat.NormalWarpingOff()
  contour=vtk.vtkFlyingEdges3D();contour.SetInputConnection(splat.GetOutputPort());contour.SetValue(0,.60)
  smooth=vtk.vtkWindowedSincPolyDataFilter();smooth.SetInputConnection(contour.GetOutputPort());smooth.SetNumberOfIterations(80);smooth.SetPassBand(.01);smooth.NormalizeCoordinatesOn()
  normals=vtk.vtkPolyDataNormals();normals.SetInputConnection(smooth.GetOutputPort());normals.SplittingOff()
 # Static triangle data and animated coordinates share the historical studio schema.
 out=dict(ground=d['ground'],center=(points.min(0)+points.max(0))/2,
          knife=blade_points,knife_tri=blade_faces,
          sim_time=np.arange(last+1)*float(task['environment']['agent_control_dt']))
 app=None
 appearance_note='Fallback surface palette'
 if a.appearance_reference:
  if a.backend!='DiSECt':raise ValueError('FEM appearance maps cannot be used for particle topology')
  ref=dict(np.load(a.appearance_reference/'episode.npz'))
  for key in ('tri','cut_tri','cut_edge_indices','cut_edge_coords'):
   if not np.array_equal(ref[key],d[key]):raise ValueError('Appearance reference topology mismatch: '+key)
  # Rigid registration verifies asset identity; material deformation is not overwritten.
  x=ref['particles'][0];y=points
  if x.shape!=y.shape:raise ValueError('Appearance particle count mismatch')
  xc=x-x.mean(0);yc=y-y.mean(0)
  u,_,vt=np.linalg.svd(xc.T@yc);rot=u@vt
  if np.linalg.det(rot)<0:raise ValueError('Appearance alignment requires a reflection')
  rms=float(np.sqrt(np.mean(np.sum((xc@rot-yc)**2,axis=1))))
  if rms>.003:raise ValueError('Appearance reference geometry differs by more than 3 mm')
  app=r.load_appearance(a.appearance_reference/'appearance.npz',a.appearance_reference/'episode.npz')
  appearance_note=dict(reference=str(a.appearance_reference),topology_verified=True,rigid_rms_m=rms)
 if group.get('representation')=='material_points':
  # Topology varies with the measured cloud. Save each derived display surface.
  for t,st in enumerate(states):
   poly.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(st['particles']),deep=True));poly.Modified()
   normals.Update();mesh=normals.GetOutput()
   out[f'points_{t}']=vtk_to_numpy(mesh.GetPoints().GetData()).copy()
   cells=vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1,4)
   if not np.all(cells[:,0]==3):raise ValueError('Particle display surface must be triangular')
   out[f'faces_{t}']=cells[:,1:].copy()
  out['points']=out['points_0'];out['faces']=out['faces_0']
  out['colors']=np.tile(color,(len(out['faces']),1))
  out['dynamic_topology']=True
 else:
  series=dict(d);series['particles']=np.stack([st['particles'] for st in states])
  out['points']=np.stack([r.display_vertices(series,t,app).astype(np.float32) for t in range(last+1)])
  if app is None:out['faces']=np.concatenate([d['tri'],d['cut_tri']])
  else:
   n=len(app['indices'])
   out['faces']=np.concatenate([np.arange(n).reshape(-1,3),d['cut_tri']+n])
  out['colors']=r.material_face_colors(series,style,app,False)/255.
 for j,(k,act,local) in enumerate(robot):
  act.GetMapper().Update()
  triangles=vtk.vtkTriangleFilter();triangles.SetInputData(act.GetMapper().GetInput());triangles.Update()
  mesh=triangles.GetOutput()
  v=vtk_to_numpy(mesh.GetPoints().GetData()).copy()
  cells=vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1,4)
  if not np.all(cells[:,0]==3):raise ValueError('Robot geometry must be triangular')
  out[f'robot_v{j}']=v
  out[f'robot_f{j}']=cells[:,1:]
  out[f'robot_c{j}']=act.GetProperty().GetColor()
  out[f'robot_tool{j}']=str(d['geom_body'][k])=='b_bot_right_inner_knuckle'
  transforms=[]
  for st in states:
   m=np.eye(4);m[:3,:3]=st['body_mat'][k].reshape(3,3);m[:3,3]=st['body_pos'][k]
   transforms.append(m@local)
  out[f'robot_transform{j}']=np.stack(transforms)
 # A texture can color a particle display without claiming its mesh/UV is the scan.
 # Cucumber FEM uses its verified barycentric map; half potato/BM use decorative projection.

 texture=None
 if a.color_reference:
  if app is not None:raise ValueError('Choose exact appearance OR approximate color reference')
  lookup=color_lookup(a.color_reference,points)
  if group.get('representation')=='material_points':
   particle_colors=lookup(points)
   for t,st in enumerate(states):
    pd=r.polydata(st['particles'],np.empty((0,3),dtype=int))
    loc=vtk.vtkStaticPointLocator();loc.SetDataSet(pd);loc.BuildLocator()
    centers=out[f'points_{t}'][out[f'faces_{t}']].mean(1)
    colors=[]
    for center in centers:
     ids=vtk.vtkIdList();loc.FindClosestNPoints(4,center,ids)
     ii=np.array([ids.GetId(k) for k in range(ids.GetNumberOfIds())])
     weights=1/np.maximum(np.linalg.norm(st['particles'][ii]-center,axis=1),1e-5)
     colors.append((particle_colors[ii]*weights[:,None]).sum(0)/weights.sum())
    out[f'colors_{t}']=np.asarray(colors)
   out['colors']=out['colors_0']

  else:
   # Densify display triangles on the same FEM faces so scan colors are not
   # reduced to one flat color per coarse simulation triangle.
   coarse=out['points'];base_faces=out['faces'];ids=[];weights=[]
   for face in base_faces:
    v=coarse[0,face]
    n=max(1,int(np.ceil(max(np.linalg.norm(v[(j+1)%3]-v[j]) for j in range(3))/.0015)))
    def bary(i,j):return np.array([1-(i+j)/n,i/n,j/n])
    for i in range(n):
     for j in range(n-i):
      pieces=[[bary(i,j),bary(i+1,j),bary(i,j+1)]]
      if i+j<n-1:pieces.append([bary(i+1,j),bary(i+1,j+1),bary(i,j+1)])
      for piece in pieces:ids.extend([face]*3);weights.extend(piece)
   ids=np.asarray(ids);weights=np.asarray(weights)
   out['points']=np.stack([(x[ids]*weights[...,None]).sum(1).astype(np.float32) for x in coarse])
   out['faces']=np.arange(len(ids)).reshape(-1,3)
   out['colors']=lookup(out['points'][0][out['faces']].mean(1))
   out['normal_base_points']=coarse;out['normal_base_faces']=base_faces
   out['normal_indices']=ids;out['normal_weights']=weights

  appearance_note=dict(reference=str(a.color_reference),mapping='Approximate PCA color transfer only; geometry unchanged; not original UV')
 out['metadata']=np.array(json.dumps(dict(format='studio_evaluation_v1',
   recording=str(a.evaluation),frame=0,sim_time=0,robot_meshes=len(robot),sources=sources,
   backend=a.backend,material=style,appearance=appearance_note,
   physical_blade=True,mount_rigidity_residual_m=mount_residual,
   mount='Original handle; fixed-joint connector is display-only',tool_revision=task.get('tool_revision'),
   case=w,episode=row,policy_updates=initial['training']['policy_updates'],
   scope=('Settled initial geometry; no policy or control steps' if a.snapshot else row.get('scope','Frozen policy rerun; first world per object; no physics edits')))))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output,**out)
 print(a.output,flush=True)
if __name__=='__main__':main()
