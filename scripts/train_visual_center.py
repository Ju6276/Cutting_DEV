"""Bounded CPU training; validation selects checkpoint, test opened once at end."""
import argparse,hashlib,json,time,copy
from pathlib import Path
import numpy as np
import torch
from cutting_world_model.visual_center import CenterNet,REVISION,SCALE_M,POINT_COUNT

def metric(error):
    e=np.abs(np.asarray(error))*1000
    return dict(mae_mm=float(e.mean()),p95_mm=float(np.quantile(e,.95)),max_mm=float(e.max()))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=120);p.add_argument("--wall-seconds",type=float,default=900)
    p.add_argument("--wandb-mode",choices=["online","disabled"],default="online")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261117);start=time.monotonic()
    d=np.load(a.data/"dataset.npz",allow_pickle=False);manifest=json.loads((a.data/"manifest.json").read_text())
    actual_sha=hashlib.sha256((a.data/"dataset.npz").read_bytes()).hexdigest()
    if actual_sha!=manifest["dataset_sha256"]:raise ValueError("Dataset checksum differs from manifest")
    identity={row["case"]:row["split"] for row in manifest["cases"]}
    if any(identity.get(str(case))!=str(split) for case,split in zip(d["case"],d["split"])):
        raise ValueError("Dataset split identity differs from manifest")
    if not np.allclose(d["origin"][:,1]+d["label"]*SCALE_M,d["center_y"],atol=1e-8):
        raise ValueError("Relative label and base-frame target differ")
    x=torch.tensor(d["cloud"],dtype=torch.float32);y=torch.tensor(d["label"],dtype=torch.float32)
    train=np.where(d["split"]=="train")[0];val=np.where(d["split"]=="val")[0];test=np.where(d["split"]=="test")[0]
    if (len(train),len(val),len(test))!=(600,150,150):raise ValueError("Expected approved 600/150/150 split")
    if len(set(d["case"]))!=900:raise ValueError("Duplicate case identities")
    model=CenterNet();optim=torch.optim.Adam(model.parameters(),lr=.001)
    config=dict(epochs=a.epochs,seed=20261117,device="cpu",batch_size=32,lr=.001,
                dataset_sha256=manifest["dataset_sha256"],architecture="Pointwise 3-32-64-128, mean+max pool, 256-64-1 residual")
    run=None
    if a.wandb_mode=="online":
        import wandb
        run=wandb.init(project="Cutting_DEV",name="visual-center-fixed-heading-20261117",mode="online",config=config,dir=str(a.output))
        (a.output/"wandb.json").write_text(json.dumps(dict(url=run.url,id=run.id,path=run.path)))
    best=float("inf");best_epoch=0;best_state=None
    try:
        with (a.output/"epochs.jsonl").open("w") as log:
            for epoch in range(a.epochs):
                if time.monotonic()-start>a.wall_seconds:raise TimeoutError("Visual learning wall budget")
                model.train();losses=[]
                for idx in torch.tensor(train)[torch.randperm(len(train))].split(32):
                    pred=model(x[idx]);loss=(pred-y[idx]).square().mean()
                    optim.zero_grad();loss.backward();optim.step();losses.append(float(loss))
                model.eval()
                with torch.no_grad():
                    pred=torch.cat([model(b) for b in x[val].split(32)]).numpy()
                err=(pred-y[val].numpy())*SCALE_M;score=float(np.mean(np.abs(err)))
                if score<best:
                    best=score;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict())
                row=dict(epoch=epoch+1,train_mse=float(np.mean(losses)),val_mae_mm=score*1000,best_val_mae_mm=best*1000)
                log.write(json.dumps(row)+"\n");log.flush()
                if run:run.log(row,step=epoch+1)
                if (epoch+1)%10==0:print(json.dumps(row),flush=True)
        model.load_state_dict(best_state);model.eval()
        checkpoint=dict(revision=REVISION,model=best_state,scale_m=SCALE_M,points=POINT_COUNT,
                        train_config=config,best_epoch=best_epoch,best_val_mae_mm=best*1000,
                        input_scope="Visible local point cloud only; no mesh, object ID or absolute XY")
        torch.save(checkpoint,a.output/"center.pt")
        with torch.no_grad():prediction=torch.cat([model(b) for b in x[test].split(32)]).numpy()*SCALE_M+d["origin"][test,1]
        fixed=float(np.mean(d["center_y"][train]))
        methods=dict(learned=prediction,geometric=d["geometric_y"][test],constant=np.full(len(test),fixed),
                     visible_centroid=d["origin"][test,1])
        result=dict(config=config,best_epoch=best_epoch,best_val_mae_mm=best*1000,
                    elapsed_s=time.monotonic()-start,test={},per_object={})
        truth=d["center_y"][test]
        for name,values in methods.items():result["test"][name]=metric(values-truth)
        for oid,name in enumerate(["cucumber","potato_half","prism"]):
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
