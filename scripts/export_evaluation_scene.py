"""Export recorded physical geometry into the shared studio rendering schema."""
import argparse,json
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
import render_geometry as r
from cutting_world_model.render_contract import load_profile,appearance_reference,validate_platform,material_coordinates,_fixed_transform,procedural_colors

def color_lookup(reference, target, target_data, target_task):
 """Original approved scan colors in material axes; no PCA/reflected recoloring."""
 ref=dict(np.load(reference/'episode.npz'))
 ref_task=json.loads((reference/'task.json').read_text())
 app=r.load_appearance(reference/'appearance.npz',reference/'episode.npz')
 src=r.display_vertices(ref,0,app)[:len(app['indices'])].reshape(-1,3,3).mean(1)
 rgb=app['rgb']/255.
 source=material_coordinates(ref,ref_task,src)
 dest=material_coordinates(target_data,target_task,target)
 # Translation-only reset alignment. Never infer or flip axes from current shape.
 sc=(source.min(0)+source.max(0))/2
 dc=(dest.min(0)+dest.max(0))/2
 source=source-sc
 pd=r.polydata(source,np.empty((0,3),dtype=int))
 loc=vtk.vtkStaticPointLocator();loc.SetDataSet(pd);loc.BuildLocator()
 def lookup(x):
  y=material_coordinates(target_data,target_task,x)-dc
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
 p.add_argument('--robot-platform',choices=('ur5e','ur10e'),required=True)
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
 profile,profile_sha=load_profile()
 robot_identity=validate_platform(task,a.robot_platform)
 a.appearance_reference=None
 a.color_reference=appearance_reference(profile,group['name'])
 appearance_config=profile['appearance'][group['name']]
 robot,sources,tool_entries=r.physical_robot_actors(d,task,a.source_root)
 states=[]
 for t in range(last+1):
  snap=dict(np.load(gdir/('%04d'%t)/group['path']/'world_states.npz'))
  states.append({k:snap[k][0] for k in ('particles','body_pos','body_mat')})
 if a.backend=='BladeMaster':
  if task.get('cad_tool'):
   from cutting_world_model.cad_knife import load
   _,cad=load(task['robot_platform']['asset_root'])
   tf=_fixed_transform(task['robot_xml'],'ikea_knife','knife_material')
   vertices=cad['vertices_disect']@tf[:3,:3].T+tf[:3,3]
   k=list(d['geom_body']).index('ikea_knife')
   blade_points=np.stack([vertices@st['body_mat'][k].reshape(3,3).T+st['body_pos'][k] for st in states])
   blade_faces=cad['faces']
  elif 'ee_pose_base' in initial['training']:
   if 'base_position_world' in d and 'base_rotation_world' in d:
    base_position=d['base_position_world'];base_rotation=d['base_rotation_world'].reshape(3,3)
   elif task['robot_base'] in list(d['geom_body']):
    base_idx=list(d['geom_body']).index(task['robot_base'])
    base_position=states[0]['body_pos'][base_idx]
    base_rotation=states[0]['body_mat'][base_idx].reshape(3,3)
   else:
    raise ValueError('Recording lacks robot-base world transform; recapture with the current snapshot writer')
   if task.get('cad_tool'):
    from cutting_world_model.cad_knife import load
    _,cad=load(task['robot_platform']['asset_root'])
    tf=_fixed_transform(task['robot_xml'],task['robot_tip'],'knife_material')
    vertices=cad['vertices_disect']@tf[:3,:3].T+tf[:3,3]
   else:
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
   blade_faces=cad['faces'] if task.get('cad_tool') else np.array(faces)
  else:
   # Earlier captures have the identical physical transform exported separately
   # from their recorded EE trajectories, never a visual alignment adjustment.
   blade_record=dict(np.load(a.evaluation/'physical_blades.npz'))
   blade_points=blade_record['group_%02d'%a.group][:last+1]
   blade_faces=blade_record['faces']
 else:
  blade_points=np.stack([np.load(gdir/('%04d'%t)/group['path']/'episode.npz')['knife'][0] for t in range(last+1)])
  blade_faces=d['knife_tri']
 # Validate the recorded rigid mount without inventing connector geometry.
 # The original handle and physical blade remain at their recorded transforms.
 # A measured adapter mesh is unavailable; a visible gap must not be disguised.
 mount_residual=0.
 if task.get('cad_tool') or a.backend=='BladeMaster' or task.get('tool_revision')=='disect_knife_rigid_mount_v1':
  k=list(d['geom_body']).index('ikea_knife') if task.get('cad_tool') else tool_entries[0][0]
  # Check the mount against every recorded body pose (not a fitted moving link).
  all_local=np.stack([(bp-st['body_pos'][k])@st['body_mat'][k].reshape(3,3) for bp,st in zip(blade_points,states)])
  mount_residual=float(np.linalg.norm(all_local-all_local[0],axis=-1).max())
  if mount_residual>1e-5:raise ValueError('Blade is not rigid relative to the gripper; refuse a fabricated fixed mount')
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
 out=dict(mounted_cad_knife=bool(task.get('cad_tool')),ground=r.presentation_ground(d,task),recorded_ground=d['ground'],center=(points.min(0)+points.max(0))/2,
          knife=blade_points,knife_tri=blade_faces,
          sim_time=np.arange(last+1)*float(task['environment']['agent_control_dt']))
 app=None
 appearance_note='Fallback surface palette'
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
  body=str(d['geom_body'][k])
  if body in ('ikea_knife','ati_mini45_body'):out[f'robot_metallic{j}']=.7
  elif body in ('user_knife_holder','user_ur_adapter'):
   out[f'robot_metallic{j}']=0.
   out[f'robot_c{j}']=np.ones(3)
  transforms=[]
  for st in states:
   m=np.eye(4);m[:3,:3]=st['body_mat'][k].reshape(3,3);m[:3,3]=st['body_pos'][k]
   transforms.append(m@local)
  out[f'robot_transform{j}']=np.stack(transforms)
 # A texture can color a particle display without claiming its mesh/UV is the scan.
 # Cucumber FEM uses its verified barycentric map; half potato/BM use decorative projection.

 texture=None
 if a.color_reference or appearance_config.get('kind')=='procedural':
  if app is not None:raise ValueError('Choose exact appearance OR approximate color reference')
  if a.color_reference:
   lookup=color_lookup(a.color_reference,points,d,task)
  else:
   def lookup(x):return procedural_colors(appearance_config['material'],material_coordinates(d,task,x))
  if group.get('representation')=='material_points':
   for t,st in enumerate(states):
    pd=r.polydata(st['particles'],np.empty((0,3),dtype=int))
    loc=vtk.vtkStaticPointLocator();loc.SetDataSet(pd);loc.BuildLocator()
    centers=out[f'points_{t}'][out[f'faces_{t}']].mean(1)
    rest_centers=[]
    for center in centers:
     ids=vtk.vtkIdList();loc.FindClosestNPoints(4,center,ids)
     ii=np.array([ids.GetId(k) for k in range(ids.GetNumberOfIds())])
     weights=1/np.maximum(np.linalg.norm(st['particles'][ii]-center,axis=1),1e-5)
     displacement=((st['particles'][ii]-points[ii])*weights[:,None]).sum(0)/weights.sum()
     rest_centers.append(center-displacement)
    out[f'colors_{t}']=lookup(np.asarray(rest_centers))
   out['colors']=out['colors_0']

  else:
   # Densify display triangles on the same FEM faces so scan colors are not
   # reduced to one flat color per coarse simulation triangle.
   coarse=out['points'];base_faces=out['faces'];ids=[];weights=[];interior=[]
   for face_index,face in enumerate(base_faces):
    v=coarse[0,face]
    n=max(1,int(np.ceil(max(np.linalg.norm(v[(j+1)%3]-v[j]) for j in range(3))/.0015)))
    def bary(i,j):return np.array([1-(i+j)/n,i/n,j/n])
    for i in range(n):
     for j in range(n-i):
      pieces=[[bary(i,j),bary(i+1,j),bary(i,j+1)]]
      if i+j<n-1:pieces.append([bary(i+1,j),bary(i+1,j+1),bary(i,j+1)])
      for piece in pieces:
       ids.extend([face]*3);weights.extend(piece);interior.append(face_index>=len(d['tri']))
   ids=np.asarray(ids);weights=np.asarray(weights)
   out['points']=np.stack([(x[ids]*weights[...,None]).sum(1).astype(np.float32) for x in coarse])
   out['faces']=np.arange(len(ids)).reshape(-1,3)
   out['colors']=lookup(out['points'][0][out['faces']].mean(1))
   out['normal_base_points']=coarse;out['normal_base_faces']=d['tri']
   out['normal_interior_triangles']=np.asarray(interior)
   out['normal_indices']=ids;out['normal_weights']=weights

  appearance_note=dict(source=appearance_config['source'],reference=str(a.color_reference) if a.color_reference else None,mapping='Pinned scan colors in recorded material axes; translation-only reset alignment; approximate projection, not original UV')
 out['metadata']=np.array(json.dumps(dict(format='studio_evaluation_v1',
   render_contract=profile['revision'],render_profile_sha256=profile_sha,robot_identity=robot_identity,recording=str(a.evaluation),frame=0,sim_time=0,robot_meshes=len(robot),sources=sources,
   backend=a.backend,material=style,appearance=appearance_note,
   physical_blade=True,mount_rigidity_residual_m=mount_residual,
   board_display=('Pinned food_studio_v1 presentation board; infinite physical support; not a collision boundary'
     if task.get('tool_revision')=='disect_knife_rigid_mount_v1' else 'Recorded display corners; not a collision boundary'),
   mount=('User CAD assembly: ATI Mini45, PLA adapters, IKEA knife' if task.get('cad_tool') else 'Original handle and physical blade; no fabricated connector; measured adapter mesh unavailable'),
   schematic_connector=False,tool_revision=task.get('tool_revision'),
   case=w,episode=row,policy_updates=initial['training'].get('policy_updates'),
   scope=('Settled initial geometry; no policy or control steps' if a.snapshot else row.get('scope','Frozen policy rerun; first world per object; no physics edits')))))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output,**out)
 print(a.output,flush=True)
if __name__=='__main__':main()
