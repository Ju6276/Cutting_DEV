"""Generate fixed-heading visual midpoint data from current physical food snapshots."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import render_geometry as shared
from cutting_world_model.visual_cut import DepthCamera,observe_depth
from cutting_world_model.visual_center_features import heading,table_cloud_input,SCALE_M,POINT_COUNT,TABLE_FRAME
from cutting_world_model.visual_center_data import placement_splits,camera_views,validate_dataset
from cutting_world_model.camera_sim import projection_matrix,range_depth,validate_sensor_capture
from cutting_world_model.render_contract import load_profile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-manifest",type=Path,required=True,help="JSON list of food name and canonical scene path")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--camera",type=Path,required=True)
    ap.add_argument("--sensor-profile",type=Path,required=True)
    ap.add_argument("--camera-azimuth-deg",type=float,nargs="+",default=[0.])
    ap.add_argument("--grid-x",type=int,default=20)
    ap.add_argument("--grid-y",type=int,default=15)
    ap.add_argument("--seed",type=int,default=20261123)
    ap.add_argument("--wall-seconds",type=float,default=3600)
    a=ap.parse_args()
    if min(a.grid_x,a.grid_y)<2 or a.wall_seconds<=0:ap.error("Positive budget and grid axes >=2 required")
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    cfg=json.loads(a.config.read_text());cam=DepthCamera(**json.loads(a.camera.read_text()))
    sensor=json.loads(a.sensor_profile.read_text())
    validate_sensor_capture(cam,sensor,cfg.get("world_from_base"))
    heading(cfg)
    entries=json.loads(a.scene_manifest.read_text())
    names=[e["name"] for e in entries]
    if not names or len(set(names))!=len(names):raise ValueError("Unique named food scenes required")
    views=list(camera_views(cam,cfg["board_origin_base"],a.camera_azimuth_deg))
    offsets=np.array([(x,y) for x in np.linspace(-.015,.015,a.grid_x)
                              for y in np.linspace(-.015,.015,a.grid_y)])
    splits=placement_splits(len(offsets),a.seed)
    expected=len(entries)*len(offsets)*len(views)
    arrays={k:[] for k in ["cloud","label","origin","center_y","geometric_y","object","split","case","view"]}
    cases=[];sources=[]
    def progress(state,**kw):
        payload=dict(state=state,cases=len(cases),expected_cases=expected,
                     elapsed_seconds=time.monotonic()-start,**kw)
        tmp=a.output/"progress.tmp"
        tmp.write_text(json.dumps(payload,indent=2,allow_nan=False))
        tmp.replace(a.output/"progress.json")
    progress("running")
    ren=vtk.vtkRenderer()
    win=vtk.vtkRenderWindow();win.SetOffScreenRendering(True);win.SetMultiSamples(0)
    win.SetSize(cam.width,cam.height);win.AddRenderer(ren)
    camera=ren.GetActiveCamera();camera.SetClippingRange(.01,2)
    profile,_=load_profile()
    thickness=profile["board_thickness_m"]
    board=shared.box(np.asarray(cfg["board_origin_base"])-[0,0,thickness/2],
                     [*profile["board_size_m"],thickness],(.5,.5,.5))
    ren.AddActor(board)
    cap=vtk.vtkWindowToImageFilter();cap.SetInput(win);cap.ReadFrontBufferOff();cap.SetInputBufferTypeToZBuffer()
    def capture(view):
        T=view.base_from_camera
        camera.SetPosition(*T[:3,3]);camera.SetFocalPoint(*(T[:3,3]+T[:3,2]));camera.SetViewUp(*(-T[:3,1]))
        matrix=vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):matrix.SetElement(i,j,float(projection_matrix(view)[i,j]))
        camera.SetExplicitProjectionTransformMatrix(matrix);camera.SetUseExplicitProjectionTransformMatrix(True)
        win.Render();cap.Modified();cap.Update()
        z=vtk_to_numpy(cap.GetOutput().GetPointData().GetScalars()).reshape(view.height,view.width)[::-1]
        depth=np.where(z<1-1e-7,.02/(2-z*1.99),0).astype(np.float32)
        return range_depth(depth,sensor).astype(np.float32)
    try:
        for object_id,entry in enumerate(entries):
            name=entry["name"];path=Path(entry["scene"])
            with np.load(path,allow_pickle=False) as d:
                if not np.allclose(d["ground"].mean(0),cfg["board_origin_base"],atol=1e-6):
                    raise ValueError("Scene/board calibration mismatch: "+name)
                dynamic=bool(d.get("dynamic_topology",False))
                points=d["points_0"] if dynamic else d["points"][0]
                faces=d["faces_0"] if dynamic else d["faces"]
                surface=points[np.unique(faces)]
                lo=surface.min(0);hi=surface.max(0)
                food=shared.actor(shared.polydata(points,faces),(.5,.5,.5),smooth=False)
            ren.AddActor(food)
            sources.append(dict(object=name,path=str(path.resolve()),sha256=sha(path),bounds=[lo.tolist(),hi.tolist()]))
            for placement,(dx,dy) in enumerate(offsets):
                food.SetPosition(float(dx),float(dy),0)
                for view_id,view in enumerate(views):
                    if time.monotonic()-start>a.wall_seconds:raise TimeoutError("Dataset wall budget")
                    depth=capture(view)
                    seen=observe_depth(depth,view,cfg)
                    features,origin,table_origin=table_cloud_input(seen["points"],cfg,seen["board_offset_m"])
                    # Full surface truth is a LABEL ONLY, never a point-cloud input.
                    target=float((lo[1]+hi[1])/2+dy-table_origin[1])
                    geometric=float(np.quantile(seen["points"][:,1],[.02,.98]).mean()-table_origin[1])
                    case_id=f"{name}_{placement:03d}_view{view_id}"
                    for k,v in dict(cloud=features,label=(target-origin[1])/SCALE_M,
                        origin=origin,center_y=target,geometric_y=geometric,object=object_id,
                        split=splits[placement],case=case_id,view=view_id).items():arrays[k].append(v)
                    cases.append(dict(case=case_id,object=name,placement_id=placement,view_id=view_id,
                        split=str(splits[placement]),offset_m=[float(dx),float(dy)],target_plane_y_table_m=target))
                    # Audit samples only: no alternate VTK RGB palette or redundant 4500 RGB files.
                    if placement==0:
                        dest=a.output/"examples"/case_id;dest.mkdir(parents=True)
                        np.savez_compressed(dest/"depth.npz",depth_m=depth,object_mask=seen["object_mask"])
                        (dest/"camera.json").write_text(json.dumps(view.to_dict(),indent=2))
                    if len(cases)%25==0:
                        progress("running",food=name)
                        print(json.dumps(dict(cases=len(cases),expected=expected,elapsed_s=time.monotonic()-start)),flush=True)
            ren.RemoveActor(food)
        data={k:np.asarray(v) for k,v in arrays.items()}
        counts={s:int(np.sum(data["split"]==s)) for s in ["train","val","test"]}
        manifest=dict(revision="fixed_heading_center_dataset_v2",input_frame=TABLE_FRAME,
            sources=sources,cases=cases,camera=cam.to_dict(),views=[v.to_dict() for v in views],
            camera_azimuth_degrees=a.camera_azimuth_deg,config=cfg,
            config_sha256=sha(a.config),camera_sha256=sha(a.camera),points=POINT_COUNT,scale_m=SCALE_M,
            scope="Known settled food shapes; fixed food heading, +/-15 mm translations, limited camera azimuth. Ideal depth, no robot occlusion, sensor noise, material or unseen-shape generalization.",
            heading="fixed",offset_range_m=[-.015,.015],split_counts=counts,
            table_frame="Origin at calibrated board origin, axes parallel to robot base; horizontal board",
            split_rule="Every shape and camera view sharing a placement ID stays in one partition",
            elapsed_s=time.monotonic()-start)
        validate_dataset(data,manifest)
        np.savez_compressed(a.output/"dataset.npz",**data)
        manifest["dataset_sha256"]=sha(a.output/"dataset.npz")
        (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
        progress("completed")
        print("completed",len(cases),flush=True)
    except BaseException as error:
        progress("failed",error=repr(error))
        raise
    finally:
        win.Finalize()

if __name__=="__main__":main()
