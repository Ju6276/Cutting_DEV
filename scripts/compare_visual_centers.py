"""Compare matched visual-center runs; test is a reused development benchmark."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,required=True)
    a=p.parse_args()
    rows=[];errors={};reference=None;sha=None
    for seed in (20261117,20261118,20261119):
        for arch in ("local_v1","dp3_xyz_v1"):
            folder=a.runs/f"{arch}_{seed}"
            report=json.loads((folder/"report.json").read_text())
            pred=np.load(folder/"test_predictions.npz",allow_pickle=False)
            identity=(pred["case"],pred["truth"])
            if reference is None:reference=identity;sha=report["config"]["dataset_sha256"]
            else:
                if not np.array_equal(reference[0],identity[0]) or not np.array_equal(reference[1],identity[1]):
                    raise ValueError("Test cases or targets differ")
                if report["config"]["dataset_sha256"]!=sha:raise ValueError("Dataset differs")
            errors[(arch,seed)]=np.abs(pred["learned"]-pred["truth"])*1000
            rows.append(dict(architecture=arch,seed=seed,**report["test"]["learned"],
                             best_epoch=report["best_epoch"],elapsed_s=report["elapsed_s"],
                             per_object=report["per_object"],
                             wandb=json.loads((folder/"wandb.json").read_text())))
    aggregate={}
    for arch in ("local_v1","dp3_xyz_v1"):
        x=np.stack([errors[(arch,s)] for s in (20261117,20261118,20261119)])
        aggregate[arch]=dict(mean_seed_mae_mm=float(x.mean()),
                            seed_mae_mm=x.mean(1).tolist(),
                            pooled_p95_mm=float(np.quantile(x,.95)),
                            max_error_mm=float(x.max()))
    delta=np.mean([errors[("dp3_xyz_v1",s)]-errors[("local_v1",s)] for s in (20261117,20261118,20261119)],axis=0)
    rng=np.random.default_rng(20261117)
    boots=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
    result=dict(dataset_sha256=sha,rows=rows,aggregate=aggregate,
                dp3_minus_local_mae_mm=float(delta.mean()),
                case_bootstrap_ci_mm=np.quantile(boots,[.025,.975]).tolist(),
                uncertainty_scope="Paired cases conditional on these three fits; not a new-shape or training-population CI",
                scope="Same known geometries, fixed heading, nearby translated positions; reused test, no test tuning")
    (a.runs/"comparison.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()
