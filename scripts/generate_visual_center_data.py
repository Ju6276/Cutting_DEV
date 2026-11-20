"""Render 900 independent fixed-heading placements; mesh truth only labels data."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image
import render_geometry as shared
from cutting_world_model.visual_cut import DepthCamera,observe_depth
from cutting_world_model.visual_center_features import cloud_input,heading,SCALE_M

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path("configs/vision/visual_midpoint_v1.json"))
    ap.add_argument("--camera",type=Path,default=Path("configs/vision/front_camera_v1.json"))
    ap.add_argument("--wall-seconds",type=float,default=1800)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    cfg=json.loads(a.config.read_text());cam=DepthCamera(**json.loads(a.camera.read_text()))
    heading(cfg)
    if not (np.isclose(cam.fx,cam.fy) and np.isclose(cam.cx,(cam.width-1)/2) and np.isclose(cam.cy,(cam.height-1)/2)):
        raise ValueError("VTK capture requires centered square-pixel calibration")
    rng=np.random.default_rng(20261117)
    # Unique placements, shared split by placement ID; all three shapes in each split.
    offsets=np.array([(x,y) for x in np.linspace(-.015,.015,20) for y in np.linspace(-.015,.015,15)])
    order=rng.permutation(len(offsets));splits=np.empty(300,dtype="<U5")
    splits[order[:200]]="train";splits[order[200:250]]="val";splits[order[250:]]="test"
    ren=vtk.vtkRenderer();ren.SetBackground(*shared.BG)
    win=vtk.vtkRenderWindow();win.SetOffScreenRendering(True);win.SetMultiSamples(0)
    win.SetSize(cam.width,cam.height);win.AddRenderer(ren)
    camera=ren.GetActiveCamera();T=cam.base_from_camera
    camera.SetPosition(*T[:3,3]);camera.SetFocalPoint(*(T[:3,3]+T[:3,2]));camera.SetViewUp(*(-T[:3,1]))
    camera.SetViewAngle(np.degrees(2*np.arctan(cam.height/(2*cam.fy))));camera.SetClippingRange(.01,2)
    board=shared.box(np.asarray(cfg["board_origin_base"])-[0,0,.006],[.16,.24,.012],(.58,.42,.25))
    ren.AddActor(board)
    def capture(is_depth):
        cap=vtk.vtkWindowToImageFilter();cap.SetInput(win);cap.ReadFrontBufferOff()
        if is_depth:cap.SetInputBufferTypeToZBuffer()
        else:cap.SetInputBufferTypeToRGB()
        cap.Update();raw=vtk_to_numpy(cap.GetOutput().GetPointData().GetScalars())
        if not is_depth:return raw.reshape(cam.height,cam.width,3)[::-1].copy()
        z=raw.reshape(cam.height,cam.width)[::-1]
        return np.where(z<1-1e-7,.02/(2-z*1.99),0).astype(np.float32)
    arrays={k:[] for k in ["cloud","label","origin","center_y","geometric_y","object","split","case"]}
    cases=[];sources=[]
    try:
        for object_id,name in enumerate(["cucumber","potato","prism"]):
            path=a.scenes/(name+".npz");d=np.load(path,allow_pickle=False)
            if not np.allclose(d["ground"].mean(0),cfg["board_origin_base"],atol=1e-6):
                raise ValueError("Source geometry must be in calibrated robot-base frame")
            p=d["points"][0];faces=d["faces"]
            food=shared.actor(shared.polydata(p,faces),shared.MATERIAL_STYLES[name][0],smooth=True)
            ren.AddActor(food);lo=p.min(0);hi=p.max(0)
            sources.append(dict(object=name,path=str(path.resolve()),sha256=sha(path),bounds=[lo.tolist(),hi.tolist()]))
            for case,(dx,dy) in enumerate(offsets):
                if time.monotonic()-start>a.wall_seconds:raise TimeoutError("Dataset wall budget")
                food.SetPosition(float(dx),float(dy),0);win.Render()
                depth=capture(True);rgb=capture(False)
                seen=observe_depth(depth,cam,cfg)
                feat,origin=cloud_input(seen["points"],cfg["board_origin_base"][2]+seen["board_offset_m"])
                target=(lo[1]+hi[1])/2+dy
                geom=np.quantile(seen["points"][:,1],[.02,.98]).mean()
                case_id=name+"_%03d"%case;dest=a.output/"cases"/case_id;dest.mkdir(parents=True)
                np.savez_compressed(dest/"depth.npz",depth_m=depth)
                Image.fromarray(rgb).save(dest/"rgb.png")
                for k,v in dict(cloud=feat,label=(target-origin[1])/SCALE_M,origin=origin,
                                center_y=target,geometric_y=geom,object=object_id,split=splits[case],case=case_id).items():
                    arrays[k].append(v)
                cases.append(dict(case=case_id,object=name,placement_id=case,split=str(splits[case]),
                                  offset_m=[float(dx),float(dy)],target_plane_y_m=float(target)))
                if len(cases)%50==0:print(json.dumps(dict(cases=len(cases),elapsed_s=time.monotonic()-start)),flush=True)
            ren.RemoveActor(food)
    finally:win.Finalize()
    np.savez_compressed(a.output/"dataset.npz",**{k:np.asarray(v) for k,v in arrays.items()})
    manifest=dict(revision="fixed_heading_center_dataset_v1",sources=sources,cases=cases,
                  camera=cam.to_dict(),config=cfg,config_sha256=sha(a.config),camera_sha256=sha(a.camera),
                  dataset_sha256=sha(a.output/"dataset.npz"),points=512,scale_m=SCALE_M,
                  scope="900 static ideal-depth placements of three existing settled shapes; no re-settling, material variation, robot occlusion or real sensor noise",
                  heading="fixed",offset_range_m=[-.015,.015],split_counts=dict(train=600,val=150,test=150),
                  elapsed_s=time.monotonic()-start)
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("completed",len(cases),flush=True)
if __name__=="__main__":main()
