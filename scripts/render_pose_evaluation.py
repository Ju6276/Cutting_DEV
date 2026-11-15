"""Render selected first-episode worlds from captured frozen evaluation geometry."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
from PIL import Image,ImageDraw
import render_episode as r

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--evaluation',type=Path,required=True)
 p.add_argument('--group',type=int,required=True)
 p.add_argument('--source-root',type=Path,action='append',required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--backend',choices=('DiSECt','BladeMaster'),required=True)
 p.add_argument('--speed',type=float,default=.5)
 a=p.parse_args()
 gdir=a.evaluation/'geometry'
 initial=json.loads((gdir/'0000/report.json').read_text())
 group=initial['groups'][a.group]; w=group['first_world']
 rows=[json.loads(x) for x in (a.evaluation/'episodes.jsonl').read_text().splitlines()]
 row=next(x for x in rows if x['case']==w); last=int(row['steps'])
 root=gdir/'0000'/group['path']
 d=dict(np.load(root/'episode.npz'))
 task=json.loads((root/'task.json').read_text())
 robot,sources=r.visual_actors(d,task,a.source_root)
 # The combined URDF knife/adapter is only an appearance mesh. Display the
 # backend's actual collision blade once, not the differently sized URDF blade.
 robot=[entry for entry in robot if str(d['geom_body'][entry[0]])!='b_bot_right_inner_knuckle']
 states=[]
 for t in range(last+1):
  snap=dict(np.load(gdir/('%04d'%t)/group['path']/'world_states.npz'))
  states.append({k:snap[k][0] for k in ('particles','body_pos','body_mat')})
 if a.backend=='BladeMaster':
  if 'ee_pose_base' in initial['training']:
   base_idx=list(d['geom_body']).index(task['robot_base'])
   vertices=np.array(task['board_pose_geometry']['vertices_in_ee'])
   blade_points=[]
   for t in range(last+1):
    pose=np.array(json.loads((gdir/('%04d'%t)/'report.json').read_text())['training']['ee_pose_base'][w])
    xyz=pose[3:6];qw=pose[6]
    basepoints=vertices+2*np.cross(xyz,np.cross(xyz,vertices)+qw*vertices)+pose[:3]
    st=states[t]
    blade_points.append(basepoints@st['body_mat'][base_idx].reshape(3,3).T+st['body_pos'][base_idx])
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
 blade_poly=r.polydata(blade_points[0],blade_faces)
 blade_actor=r.actor(blade_poly,(.46,.49,.52),specular=.65)
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
  splat.SetRadius(radius/max(hi-lo));splat.SetExponentFactor(-5)
  splat.ScalarWarpingOff();splat.NormalWarpingOff()
  contour=vtk.vtkFlyingEdges3D();contour.SetInputConnection(splat.GetOutputPort());contour.SetValue(0,.12)
  smooth=vtk.vtkWindowedSincPolyDataFilter();smooth.SetInputConnection(contour.GetOutputPort());smooth.SetNumberOfIterations(12);smooth.SetPassBand(.1);smooth.NormalizeCoordinatesOn()
  normals=vtk.vtkPolyDataNormals();normals.SetInputConnection(smooth.GetOutputPort());normals.SplittingOff()
  material=r.actor(normals,color,specular=.08);material.GetMapper().ScalarVisibilityOff()
  def surface(x):return x
 else:
  edge=d['cut_edge_indices'];weight=d['cut_edge_coords'].reshape(-1,1)
  def surface(x):return np.concatenate([x,(1-weight)*x[edge[:,0]]+weight*x[edge[:,1]]])
  poly=r.polydata(surface(points),np.concatenate([d['tri'],d['cut_tri']]))
  material=r.actor(poly,color,specular=.07,smooth=True)
 ground=d['ground'];center=ground.mean(0);board_z=center[2]
 # Include the object and the tool throughout the episode, without recentering the camera.
 allpts=np.concatenate([s['particles'] for s in states])
 objcenter=(allpts.min(0)+allpts.max(0))/2
 extent=max(.10,np.ptp(allpts,axis=0).max())
 win=vtk.vtkEGLRenderWindow();win.SetOffScreenRendering(True);win.SetMultiSamples(4);win.SetSize(1440,680)
 renderers=[]
 for close in (False,True):
  ren=vtk.vtkRenderer();ren.SetViewport(.4 if close else 0,0,1 if close else .4,1)
  ren.SetBackground(.92,.94,.96);ren.SetBackground2(.99,.99,1);ren.GradientBackgroundOn();ren.UseFXAAOn()
  for k,actor,local in robot:
   if str(d['geom_body'][k])=='b_bot_right_inner_knuckle':actor.GetProperty().SetColor(.46,.48,.50)
   ren.AddActor(actor)
  ren.AddActor(material);ren.AddActor(blade_actor)
  ren.AddActor(r.box(center-[0,0,.006],[np.ptp(ground[:,0]),np.ptp(ground[:,1]),.012],(.68,.55,.37)))
  ren.AddActor(r.box(center-[0,0,board_z/2+.006],[.24,.22,max(.001,board_z-.012)],(.68,.71,.74)))
  if not close:ren.AddActor(r.box([.05,.2,-.019],[2,2,.025],(.84,.87,.89)))
  camera=ren.GetActiveCamera()
  focal=objcenter+np.array([0,0,.035]) if close else np.array([center[0]*.5,center[1]*.52,.27])
  camera.SetFocalPoint(*focal);camera.SetPosition(*(focal+([.28,-.4,.25] if close else [1.2,-1.6,.9])))
  camera.SetViewUp(0,0,1);camera.ParallelProjectionOn();camera.SetParallelScale(max(.11,extent*.85) if close else .40)
  ren.AutomaticLightCreationOff()
  for delta,intensity in (([.6,-.8,1.2],.7),([-.5,-.2,.7],.3),([.1,.8,1],.3)):
   light=vtk.vtkLight();light.SetPosition(*(center+delta));light.SetFocalPoint(*center);light.SetIntensity(intensity);ren.AddLight(light)
  win.AddRenderer(ren);renderers.append(ren)
 fps=20;dt=float(task['environment']['agent_control_dt'])
 times=np.arange(last+1)*dt
 indices=r.frame_indices(times,fps,a.speed)
 indices=np.concatenate([indices,np.full(fps, last)]) # One second on final measured state.
 out=a.output;out.parent.mkdir(parents=True,exist_ok=True)
 height=890
 encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'1440x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
 grab=vtk.vtkWindowToImageFilter();grab.SetInput(win);grab.SetInputBufferTypeToRGB();grab.ReadFrontBufferOff()
 title=f"{'Main / DiSECt backend' if a.backend=='DiSECt' else 'BladeMaster backend'} | {style.replace('_',' ').title()} | policy {initial['training']['policy_updates']} | case {w}"
 outcome='POSE SUCCESS' if row['success'] else 'TIMEOUT' if row['timeout'] else 'WORKSPACE FAILURE' if row['outside'] else 'FORCE LIMIT'
 try:
  for j,i in enumerate(indices):
   s=states[int(i)]
   blade_poly.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(blade_points[int(i)]),deep=True));blade_poly.Modified()
   poly.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(surface(s['particles'])),deep=True));poly.Modified()
   for k,actor,local in robot:
    rot=s['body_mat'][k].reshape(3,3); pos=s['body_pos'][k]
    actor.SetUserMatrix(r.matrix(pos+rot@local[:3,3],rot@local[:3,:3]))
   for ren in renderers:ren.ResetCameraClippingRange()
   win.Render();grab.Modified();grab.Update()
   pix=vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).reshape(680,1440,3)[::-1]
   canvas=Image.new('RGB',(1440,height),(246,248,250));canvas.paste(Image.fromarray(pix),(0,100));draw=ImageDraw.Draw(canvas)
   draw.text((24,12),title,font=r.font(28,True),fill=r.INK)
   draw.text((24,54),f"Physical collision blade | {a.speed:g}x playback | step {i}/{last} | {i*dt:.2f} s",font=r.font(22),fill=r.MUTED)
   draw.rounded_rectangle((20,109,184,146),radius=5,fill=(246,248,250));draw.rounded_rectangle((590,109,842,146),radius=5,fill=(246,248,250))
   draw.text((30,116),'ROBOT VIEW',font=r.font(20,True),fill=r.INK)
   draw.text((600,116),'CUTTING CLOSE-UP',font=r.font(20,True),fill=r.INK)
   draw.text((24,794),f"Episode outcome: {outcome} | return {row['return']:.2f} | material force peak {row['peak_material_force_n']:.2f} N",font=r.font(23,True),fill=r.INK)
   draw.text((24,836),'Actual physics blade shown; fixed-transform coupling. Appearance knife/adapter omitted.',font=r.font(18),fill=r.MUTED)
   if j in (0,len(indices)//2,len(indices)-1):canvas.save(out.with_name(out.stem+f'_frame_{j:04d}.png'))
   encoder.stdin.write(canvas.tobytes())
  encoder.stdin.close()
  if encoder.wait()!=0:raise RuntimeError('Video encoding failed')
 finally:
  if encoder.poll() is None:encoder.kill();encoder.wait()
  win.Finalize()
 out.with_suffix('.json').write_text(json.dumps(dict(case=w,episode=row,selected_world='first world per group, chosen before recording',backend=a.backend,scope='rerun of frozen evaluation with geometry capture, not original training footage',source=str(a.evaluation),visual_sources=sources,frames=len(indices),fps=fps,playback_speed=a.speed,display='URDF robot visuals with original appearance blade removed; one actual backend collision blade; no added cracks; BM display surface reconstructed from measured particles (not contact geometry)'),indent=2))
 print(out,flush=True)
if __name__=='__main__':main()
