"""Dataset contracts for fixed-heading point-cloud midpoint supervision."""
import numpy as np
from .visual_center_features import POINT_COUNT, SCALE_M, BASE_FRAME, TABLE_FRAME
from .visual_cut import DepthCamera

def placement_splits(count, seed):
    if count<10:
        raise ValueError("At least ten distinct placements required")
    order=np.random.default_rng(seed).permutation(count)
    ntrain=2*count//3
    nval=(count-ntrain)//2
    split=np.empty(count,dtype="<U5")
    split[order[:ntrain]]="train"
    split[order[ntrain:ntrain+nval]]="val"
    split[order[ntrain+nval:]]="test"
    return split

def camera_views(camera, board_origin, azimuth_degrees):
    origin=np.asarray(board_origin,dtype=float)
    for degrees in azimuth_degrees:
        if not np.isfinite(degrees):
            raise ValueError("Finite camera azimuth required")
        theta=np.radians(degrees)
        rot=np.array([[np.cos(theta),-np.sin(theta),0],
                      [np.sin(theta),np.cos(theta),0],[0,0,1.]])
        transform=camera.base_from_camera.copy()
        transform[:3,:3]=rot@transform[:3,:3]
        transform[:3,3]=origin+rot@(transform[:3,3]-origin)
        config=camera.to_dict()
        config["base_from_camera"]=transform
        yield DepthCamera(**config)

def validate_dataset(data, manifest):
    n=len(data["case"])
    if data["cloud"].shape!=(n,POINT_COUNT,3) or data["origin"].shape!=(n,3):
        raise ValueError("Point-cloud or origin shape mismatch")
    for key in ("cloud","origin","label","center_y","geometric_y"):
        if not np.isfinite(data[key]).all():
            raise ValueError("Nonfinite dataset: "+key)
    if data["label"].shape!=(n,) or np.any(np.abs(data["label"])>=.5):
        raise ValueError("Labels must lie within the regression head range")
    if not np.allclose(data["origin"][:,1]+data["label"]*SCALE_M,
                       data["center_y"],atol=1e-8):
        raise ValueError("Relative label and coordinate target differ")
    if manifest.get("input_frame",BASE_FRAME) not in (BASE_FRAME,TABLE_FRAME):
        raise ValueError("Unsupported dataset frame")
    cases=manifest["cases"]
    identities={row["case"]:row for row in cases}
    if len(identities)!=n or len(cases)!=n or len(set(data["case"]))!=n:
        raise ValueError("Duplicate or missing case identities")
    by_placement={}
    for case,split in zip(data["case"],data["split"]):
        row=identities.get(str(case))
        if row is None or row["split"]!=str(split):
            raise ValueError("Dataset split identity differs from manifest")
        # All shapes/views at a placement share one split.
        placement=row.get("placement_id")
        if placement is None:
            if manifest.get("input_frame",BASE_FRAME)==TABLE_FRAME:
                raise ValueError("Missing placement identity")
            placement=str(case).rsplit("_",1)[-1]
        previous=by_placement.setdefault(placement,row["split"])
        if previous!=row["split"]:
            raise ValueError("Placement/view leakage across splits")
    counts={s:int(np.sum(data["split"]==s)) for s in ("train","val","test")}
    if any(v==0 for v in counts.values()) or sum(counts.values())!=n:
        raise ValueError("Nonempty train/validation/test partitions required")
    if counts!=manifest["split_counts"]:
        raise ValueError("Dataset split counts differ from manifest")
    names=[row["object"] for row in manifest["sources"]]
    if len(set(names))!=len(names) or np.any(data["object"]<0) or np.any(data["object"]>=len(names)):
        raise ValueError("Invalid object identities")
    for i in range(len(names)):
        if any(not np.any((data["object"]==i)&(data["split"]==s)) for s in counts):
            raise ValueError("Every object needs train/validation/test cases")
    return names
