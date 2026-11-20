"""Draw measured URDF frame transforms on the actual UR10e tool assembly."""
import argparse,json
from pathlib import Path
import numpy as np
import vtk
from PIL import Image,ImageDraw
from vtk.util.numpy_support import vtk_to_numpy
import render_geometry as r
from cutting_world_model.robot_platform import RobotPlatform,frame_poses

def arrow(origin,direction,color,length):
 src=vtk.vtkArrowSource();src.SetShaftRadius(.025);src.SetTipRadius(.075);src.SetTipLength(.2)
 src.SetShaftResolution(24);src.SetTipResolution(24)
 a=r.actor(src,color,specular=.2)
 x=np.asarray(direction);x=x/np.linalg.norm(x)
 aux=np.array([0.,0.,1.]) if abs(x[2])<.9 else np.array([0.,1.,0.])
 y=np.cross(aux,x);y/=np.linalg.norm(y);z=np.cross(x,y)
 a.SetUserMatrix(r.matrix(origin,np.stack([x,y,z],axis=1)*length))
 return a

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--platform',required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();bundle=RobotPlatform.load(a.platform);poses=frame_poses(bundle.metadata,bundle.q)
 names=['wrist_3_link','user_ur_adapter','ati_mini45_body','user_knife_holder','ikea_knife']
 selected=['wrist_3_link','tool0','ati_sensing_frame','knife_control']
 labels=['UR10e wrist joint','UR flange tool0','ATI sensing frame','Knife control TCP']
 win=vtk.vtkRenderWindow();win.SetOffScreenRendering(True);win.SetMultiSamples(4);win.SetSize(2400,950);win.SetNumberOfLayers(2)
 center=(poses['tool0'][:3,3]+poses['knife_tip'][:3,3])/2
 for panel,frame in enumerate(selected):
  ren=vtk.vtkRenderer();ren.SetViewport(panel/4,0,(panel+1)/4,1);ren.SetBackground(.95,.96,.97);ren.UseFXAAOn();win.AddRenderer(ren)
  actors,_,_=r.physical_robot_actors(dict(geom_body=np.array(names)),bundle.metadata,[Path('/opt/ros/noetic/share/ur_description')])
  for index,actor,local in actors:
   body=names[index];T=poses[body]@local
   actor.SetUserMatrix(r.matrix(T[:3,3],T[:3,:3]))
   if body in ('ikea_knife','ati_mini45_body'):
    actor.GetProperty().SetColor(.63,.66,.69);actor.GetProperty().SetSpecular(.75);actor.GetProperty().SetSpecularPower(40)
   elif body in ('user_ur_adapter','user_knife_holder'):
    actor.GetProperty().SetColor(.23,.25,.28);actor.GetProperty().SetSpecular(.08)
   ren.AddActor(actor)
  overlay=vtk.vtkRenderer();overlay.SetLayer(1);overlay.SetViewport(panel/4,0,(panel+1)/4,1)
  overlay.SetPreserveDepthBuffer(False);win.AddRenderer(overlay)
  overlay.SetActiveCamera(ren.GetActiveCamera())
  T=poses[frame]
  for k,color in enumerate(((.85,.15,.12),(.08,.6,.24),(.10,.32,.85))):
   overlay.AddActor(arrow(T[:3,3],T[:3,k],color,.045))
  sphere=vtk.vtkSphereSource();sphere.SetRadius(.003);sphere.SetCenter(*T[:3,3])
  overlay.AddActor(r.actor(sphere,(.1,.1,.1)))
  camera=ren.GetActiveCamera();camera.SetFocalPoint(*center);camera.SetPosition(*(center+[.25,-.5,.25]));camera.SetViewUp(0,0,1)
  camera.ParallelProjectionOn();camera.SetParallelScale(.25);ren.ResetCameraClippingRange()
  for k,letter in enumerate('XYZ'):
   text=vtk.vtkBillboardTextActor3D();text.SetInput(letter);text.SetPosition(*(T[:3,3]+T[:3,k]*.052))
   text.GetTextProperty().SetFontSize(22);text.GetTextProperty().SetBold(True)
   text.GetTextProperty().SetColor(*((.85,.15,.12),(.08,.6,.24),(.10,.32,.85))[k]);overlay.AddActor(text)
 try:
  win.Render();capture=vtk.vtkWindowToImageFilter();capture.SetInput(win);capture.SetInputBufferTypeToRGB();capture.ReadFrontBufferOff();capture.Update()
  pix=vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(950,2400,3)[::-1]
  canvas=Image.new('RGB',(2400,1250),(242,245,248));canvas.paste(Image.fromarray(pix),(0,100));draw=ImageDraw.Draw(canvas)
  draw.text((35,20),'UR10e / ATI Mini45 / IKEA knife — actual assembly frames',font=r.font(36,True),fill=r.INK)
  for i,(name,label) in enumerate(zip(selected,labels)):
   x=i*600+25
   draw.text((x,83),label,font=r.font(25,True),fill=r.INK)
   draw.text((x,1050),name,font=r.font(23,True),fill=r.INK)
   xyz=poses[name][:3,3]*1000
   draw.text((x,1090),'base xyz [mm]: '+', '.join(f'{v:.1f}' for v in xyz),font=r.font(19),fill=r.MUTED)
  draw.text((35,1155),'Axes: X red / Y green / Z blue. CAD mounting shown; sensor clocking still requires hardware verification.',font=r.font(25),fill=r.INK)
  draw.text((35,1195),'knife_control is the task TCP; knife_tip is the forward tip. These are different points on the same rigid knife.',font=r.font(25),fill=r.MUTED)
  a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output)
 finally:win.Finalize()
 data=dict(robot_q_rad=bundle.q.tolist(),base=bundle.metadata['robot_base'],frames={k:v.tolist() for k,v in poses.items()},
   relative={child:(np.linalg.inv(poses[parent])@poses[child]).tolist() for parent,child in [('tool0','ati_sensing_frame'),('ati_sensing_frame','knife_control'),('knife_control','knife_tip')]},
   note='Exact exported URDF transforms, estimated hardware mounting. No real-robot calibration claim.')
 a.output.with_suffix('.json').write_text(json.dumps(data,indent=2))
if __name__=='__main__':main()
