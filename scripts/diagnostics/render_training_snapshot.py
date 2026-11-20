"""Render a provenance-labelled snapshot exported by an actual training loop."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import vtk
from PIL import Image,ImageDraw
from vtk.util.numpy_support import vtk_to_numpy
import render_geometry as r


def robot_instances(renderer, records, offsets):
 # Distant all-world views use instancing and an explicit visual-only mesh LOD.
 # Each instance still uses the captured body's own full transform.
 from vtk.util.numpy_support import numpy_to_vtk
 batches={}
 for (g,d,states,actors,i,color),offset in zip(records,offsets):
  for k,original,local in actors:
   key=(g['name'],k)
   if key not in batches:batches[key]=[original,[],[],[]]
   transform=np.eye(4);transform[:3,:3]=states['body_mat'][i,k].reshape(3,3)
   transform[:3,3]=states['body_pos'][i,k]+offset
   transform=transform@local
   scales=np.linalg.norm(transform[:3,:3],axis=0)
   rotation=transform[:3,:3]/scales
   quat=[0.,0.,0.,0.];vtk.vtkMath.Matrix3x3ToQuaternion(rotation.tolist(),quat)
   batches[key][1].append(transform[:3,3]);batches[key][2].append(quat);batches[key][3].append(scales)
 for original,positions,quaternions,scales in batches.values():
  original.GetMapper().Update();source=original.GetMapper().GetInput()
  if source.GetNumberOfPolys()>600:
   tri=vtk.vtkTriangleFilter();tri.SetInputData(source);tri.Update()
   low=vtk.vtkQuadricDecimation();low.SetInputConnection(tri.GetOutputPort())
   low.SetTargetReduction(min(.995,1-500/max(501,tri.GetOutput().GetNumberOfPolys())));low.Update()
   source=low.GetOutput()
  points=vtk.vtkPoints();points.SetData(numpy_to_vtk(np.asarray(positions),deep=True))
  data=vtk.vtkPolyData();data.SetPoints(points)
  for name,array in [('rotation',quaternions),('scale',scales)]:
   values=numpy_to_vtk(np.asarray(array),deep=True);values.SetName(name);data.GetPointData().AddArray(values)
  mapper=vtk.vtkGlyph3DMapper();mapper.SetInputData(data);mapper.SetSourceData(source)
  mapper.SetOrientationArray('rotation');mapper.SetOrientationModeToQuaternion();mapper.OrientOn()
  mapper.SetScaleArray('scale');mapper.SetScaleModeToScaleByVectorComponents();mapper.ScalingOn();mapper.SetScaleFactor(1.)
  actor=vtk.vtkActor();actor.SetMapper(mapper);actor.GetProperty().DeepCopy(original.GetProperty())
  renderer.AddActor(actor)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--recording',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--source-root',type=Path,action='append',default=[])
 p.add_argument('--detail',action='store_true',help='Two actual worlds per object instead of all worlds')
 a=p.parse_args();report=json.loads((a.recording/'report.json').read_text())
 if report['format']!='training_geometry_v1':raise ValueError('Not a running-training snapshot')
 records=[];sources=[];hashes={}
 for g in report['groups']:
  root=a.recording/g['path']
  d=dict(np.load(root/'episode.npz',allow_pickle=False))
  states=dict(np.load(root/'world_states.npz',allow_pickle=False))
  if len(states['world_id'])!=g['worlds']:raise ValueError('Missing captured worlds')
  task=json.loads((root/'task.json').read_text())
  d['mounted_cad_knife']=bool(task.get('cad_tool'))
  if 'knife_world' not in states or 'knife_tri' not in states:
   raise ValueError('Snapshot lacks per-world physical blades; recapture instead of displaying the old URDF knife')
  if len(states['knife_world'])!=g['worlds']:
   raise ValueError('Missing per-world physical blades')
  actors,src,_=r.physical_robot_actors(d,task,a.source_root);sources.extend(src)
  d['ground']=r.presentation_ground(d,task)
  hashes[g['name']]=hashlib.sha256((root/'world_states.npz').read_bytes()).hexdigest()
  ids=range(min(2,g['worlds'])) if a.detail else range(g['worlds'])
  color=(.19,.36,.075) if 'cucumber' in g['name'] else ((.72,.58,.35) if 'potato' in g['name'] else (.72,.44,.20))
  for i in ids:records.append((g,d,states,actors,i,color))
 count=len(records);cols=3 if a.detail else math.ceil(math.sqrt(count))
 if a.detail:records=[records[i] for i in (0,2,4,1,3,5)] if len(records)==6 else records
 offsets=np.array([[i%cols,i//cols,0] for i in range(count)],dtype=float)*[1.1,1.05,0]
 offsets[:,:2]-=offsets[:,:2].mean(0)
 win=vtk.vtkRenderWindow();win.SetOffScreenRendering(True);win.SetMultiSamples(4);win.SetSize(2560,1800)
 ren=vtk.vtkRenderer();ren.SetBackground(.94,.95,.96);ren.UseFXAAOn();win.AddRenderer(ren)
 if count>32:robot_instances(ren,records,offsets)
 for (g,d,states,actors,i,color),offset in zip(records,offsets):
  for k,original,local in (actors if count<=32 else []):
   inst=vtk.vtkActor();inst.SetMapper(original.GetMapper());inst.GetProperty().DeepCopy(original.GetProperty())
   if str(d['geom_body'][k])=='b_bot_right_inner_knuckle':inst.GetProperty().SetColor(.44,.47,.50)
   transform=np.eye(4);transform[:3,:3]=states['body_mat'][i,k].reshape(3,3)
   transform[:3,3]=states['body_pos'][i,k]+offset;transform=transform@local
   inst.SetUserMatrix(r.matrix(transform[:3,3],transform[:3,:3]));ren.AddActor(inst)
  if not d.get('mounted_cad_knife'):
   ren.AddActor(r.actor(r.polydata(states['knife_world'][i]+offset,states['knife_tri']),(.61,.63,.65),specular=.5))
  points=states['particles'][i]
  if g.get('representation')=='material_points':
   poly=vtk.vtkPolyData();pts=vtk.vtkPoints()
   from vtk.util.numpy_support import numpy_to_vtk
   pts.SetData(numpy_to_vtk(points+offset,deep=True));poly.SetPoints(pts)
   sphere=vtk.vtkSphereSource();sphere.SetRadius(float(g['particle_radius_m']))
   sphere.SetThetaResolution(6 if count<=32 else 4);sphere.SetPhiResolution(4)
   mapper=vtk.vtkGlyph3DMapper();mapper.SetInputData(poly);mapper.SetSourceConnection(sphere.GetOutputPort())
   mapper.ScalingOff();mapper.ScalarVisibilityOff()
   obj=vtk.vtkActor();obj.SetMapper(mapper);obj.GetProperty().SetColor(*color);ren.AddActor(obj)
  else:
   edges=d['cut_edge_indices'];weights=d['cut_edge_coords'].reshape(-1,1)
   virtual=(1-weights)*points[edges[:,0]]+weights*points[edges[:,1]]
   surface=np.concatenate([points,virtual])+offset
   ren.AddActor(r.actor(r.polydata(surface,np.concatenate([d['tri'],d['cut_tri']])),color,smooth=True))
  center=d['ground'].mean(0)+offset;size=np.array([np.ptp(d['ground'][:,0]),np.ptp(d['ground'][:,1]),.012])
  ren.AddActor(r.box(center-[0,0,.006],size,(.68,.54,.35)))
  ren.AddActor(r.box([center[0],center[1],(center[2]-.012)/2],[size[0]*.85,size[1]*.85,max(.001,center[2]-.012)],(.64,.68,.72)))
 bounds=ren.ComputeVisiblePropBounds();center=np.array([(bounds[j]+bounds[j+1])/2 for j in (0,2,4)])
 camera=ren.GetActiveCamera();camera.SetFocalPoint(*center);camera.SetPosition(*(center+[18,-24,26]))
 camera.SetViewUp(0,0,1);camera.ParallelProjectionOn();ren.ResetCamera();ren.ResetCameraClippingRange()
 try:
  win.Render();cap=vtk.vtkWindowToImageFilter();cap.SetInput(win);cap.SetInputBufferTypeToRGB();cap.ReadFrontBufferOff();cap.Update()
  out=cap.GetOutput();w,h,_=out.GetDimensions();pixels=vtk_to_numpy(out.GetPointData().GetScalars()).reshape(h,w,3)[::-1]
  canvas=Image.new('RGB',(w,h+160),(240,243,246));canvas.paste(Image.fromarray(pixels),(0,160));draw=ImageDraw.Draw(canvas)
  t=report['training'];stage=t.get('training_stage',t.get('phase',t.get('stage','unclassified')));label='ACTUAL TRAINING SNAPSHOT' if stage in ('world_model_only','joint') else 'VALIDATION SNAPSHOT'
  draw.text((40,16),f"{label} | {report['worlds']} environments | vector step {t.get('iteration','n/a')}",font=r.font(36,True),fill=r.INK)
  draw.text((40,67),f"{stage} | WM rounds {t.get('wm_updates','n/a')} | policy updates {t.get('policy_updates','n/a')} | showing {count} captured worlds",font=r.font(25),fill=r.MUTED)
  draw.text((40,108),'Cucumber / half potato / prism. Grid is display-only; material colors are display colors. No added crack overlay.',font=r.font(23),fill=r.MUTED)
  a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output)
 finally:win.Finalize()
 a.output.with_suffix('.json').write_text(json.dumps(dict(report,shown_world_ids=[g['first_world']+i for g,d,s,acts,i,c in records],capture_sha256=hashes,visual_sources=sorted(set(sources)),render_only_offsets=offsets.tolist(),physical_blade_per_world=True,schematic_connector=False,material_display=("recorded centres/radii, low-resolution sphere glyphs" if any(g.get("representation")=="material_points" for g in report["groups"]) else "recorded FEM surface"),robot_mesh_lod=('distant_decimated_instances' if count>32 else 'original_visual_meshes')),indent=2))
if __name__=='__main__':main()
