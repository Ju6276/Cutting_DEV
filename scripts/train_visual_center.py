"""Bounded GPU training; validation selects checkpoint, test opened once at end."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from cutting_world_model.visual_center import make_center,REVISION,SCALE_M,POINT_COUNT
from cutting_world_model.visual_center_data import validate_dataset
from cutting_world_model.visual_center_features import BASE_FRAME

def metric(error):
    e=np.abs(np.asarray(error))*1000
    return dict(mae_mm=float(e.mean()),p95_mm=float(np.quantile(e,.95)),max_mm=float(e.max()))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=120);p.add_argument("--wall-seconds",type=float,default=900)
    p.add_argument("--wandb-mode",choices=["online","disabled"],default="online")
    p.add_argument("--architecture",choices=["local_v1","dp3_xyz_v1"],default="dp3_xyz_v1")
    p.add_argument("--seed",type=int,default=20261123)
    p.add_argument("--device",default="cuda",help="CUDA device for visual training")
    a=p.parse_args()
    if a.epochs<1 or a.wall_seconds<=0:p.error("Positive epochs and wall budget required")
    device=torch.device(a.device)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("Visual training requires CUDA")
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(a.seed);start=time.monotonic()
    d=np.load(a.data/"dataset.npz",allow_pickle=False);manifest=json.loads((a.data/"manifest.json").read_text())
    actual_sha=hashlib.sha256((a.data/"dataset.npz").read_bytes()).hexdigest()
    if actual_sha!=manifest["dataset_sha256"]:raise ValueError("Dataset checksum differs from manifest")
    names=validate_dataset(d,manifest)
    x=torch.tensor(d["cloud"],dtype=torch.float32,device=device)
    y=torch.tensor(d["label"],dtype=torch.float32,device=device)
    train=np.where(d["split"]=="train")[0];val=np.where(d["split"]=="val")[0];test=np.where(d["split"]=="test")[0]
    model=make_center(a.architecture).to(device)
    order_generator=torch.Generator().manual_seed(a.seed)
    optim=torch.optim.Adam(model.parameters(),lr=.001)
    config=dict(epochs=a.epochs,seed=a.seed,device=str(device),input_frame=manifest.get("input_frame",BASE_FRAME),
                split_counts=manifest["split_counts"],objects=names,camera_azimuth_degrees=manifest.get("camera_azimuth_degrees",[0]),batch_size=32,lr=.001,
                dataset_sha256=manifest["dataset_sha256"],architecture=a.architecture,parameters=sum(p.numel() for p in model.parameters()),
                data_order="dedicated generator independent of model initialization")
    run=None
    if a.wandb_mode=="online":
        import wandb
        run=wandb.init(project="Cutting_DEV",name=f"visual-center-{a.architecture}-seed{a.seed}",mode="online",config=config,dir=str(a.output))
        (a.output/"wandb.json").write_text(json.dumps(dict(url=run.url,id=run.id,path=run.path)))
    best=float("inf");best_epoch=0;best_state=None
    try:
        with (a.output/"epochs.jsonl").open("w") as log:
            for epoch in range(a.epochs):
                if time.monotonic()-start>a.wall_seconds:raise TimeoutError("Visual learning wall budget")
                model.train();losses=[]
                for idx in torch.tensor(train)[torch.randperm(len(train),generator=order_generator)].split(32):
                    idx=idx.to(device)
                    pred=model(x[idx]);loss=(pred-y[idx]).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
                    optim.zero_grad();loss.backward();optim.step();losses.append(float(loss))
                model.eval()
                with torch.no_grad():
                    pred=torch.cat([model(b) for b in x[val].split(32)]).cpu().numpy()
                err=(pred-y[val].cpu().numpy())*SCALE_M
                if not np.isfinite(err).all():raise FloatingPointError("Nonfinite validation prediction")
                score=float(np.mean(np.abs(err)))
                if score<best:
                    best=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                row=dict(epoch=epoch+1,train_mse=float(np.mean(losses)),val_mae_mm=score*1000,best_val_mae_mm=best*1000)
                log.write(json.dumps(row)+"\n");log.flush()
                if run:run.log(row,step=epoch+1)
                if (epoch+1)%10==0:print(json.dumps(row),flush=True)
        model.load_state_dict(best_state);model.eval()
        checkpoint=dict(architecture=a.architecture,revision=REVISION,model=best_state,scale_m=SCALE_M,points=POINT_COUNT,
                        input_frame=manifest.get("input_frame",BASE_FRAME),train_config=config,best_epoch=best_epoch,best_val_mae_mm=best*1000,
                        input_scope="Visible local point cloud only; no mesh, object ID or absolute XY")
        torch.save(checkpoint,a.output/"center.pt")
        with torch.no_grad():prediction=torch.cat([model(b) for b in x[test].split(32)]).cpu().numpy()*SCALE_M+d["origin"][test,1]
        fixed=float(np.mean(d["center_y"][train]))
        methods=dict(learned=prediction,geometric=d["geometric_y"][test],constant=np.full(len(test),fixed),
                     visible_centroid=d["origin"][test,1])
        result=dict(state="completed",config=config,best_epoch=best_epoch,best_val_mae_mm=best*1000,
                    elapsed_s=time.monotonic()-start,test={},per_object={})
        truth=d["center_y"][test]
        for name,values in methods.items():result["test"][name]=metric(values-truth)
        for oid,name in enumerate(names):
            mask=d["object"][test]==oid
            result["per_object"][name]={k:metric(v[mask]-truth[mask]) for k,v in methods.items()}
        (a.output/"report.json").write_text(json.dumps(result,indent=2))
        np.savez_compressed(a.output/"test_predictions.npz",case=d["case"][test],truth=truth,**methods)
        if run:run.summary.update({"best_epoch":best_epoch,**{k+"_test_mae_mm":v["mae_mm"] for k,v in result["test"].items()}})
        print(json.dumps(result,indent=2),flush=True)
    except BaseException:
        if run:run.finish(exit_code=1);run=None
        raise
    finally:
        if run:run.finish()
if __name__=="__main__":main()
