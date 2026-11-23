"""Fixed-heading midpoint prediction from visible point clouds.

Supervised mesh labels are offline only. Inference sees a cloud, calibration,
and a learned residual from its observed centroid. This is not material inference.
"""
import copy
import numpy as np
import torch
from torch import nn
from .visual_cut import observe_depth, blade_points, estimate_plan

from .visual_center_features import REVISION, SCALE_M, POINT_COUNT, heading, cloud_input, table_cloud_input, TABLE_FRAME, BASE_FRAME


class CenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,64),nn.SiLU(),
                                   nn.Linear(64,128),nn.SiLU())
        self.head = nn.Sequential(nn.Linear(256,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, cloud):
        h = self.point(cloud)
        return .5*torch.tanh(self.head(torch.cat([h.mean(1), h.amax(1)], -1)).squeeze(-1))


class DP3CenterNet(nn.Module):
    """Official DP3 XYZ encoder + our bounded midpoint regression head."""
    def __init__(self):
        super().__init__()
        from .vendor.dp3.pointnet import PointNetEncoderXYZ
        self.encoder = PointNetEncoderXYZ(in_channels=3, out_channels=64,
                                         use_layernorm=True, final_norm="layernorm")
        self.head = nn.Sequential(nn.Linear(64,64), nn.SiLU(), nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, cloud):
        return .5*torch.tanh(self.head(self.encoder(cloud)).squeeze(-1))


def make_center(architecture):
    if architecture == "local_v1":
        return CenterNet()
    if architecture == "dp3_xyz_v1":
        return DP3CenterNet()
    raise ValueError("Unknown center architecture: " + str(architecture))


def load_center(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("revision") != REVISION or data.get("scale_m") != SCALE_M or data.get("points") != POINT_COUNT:
        raise ValueError("Center checkpoint contract mismatch")
    input_frame=data.get("input_frame",BASE_FRAME)
    if input_frame not in (BASE_FRAME,TABLE_FRAME):
        raise ValueError("Unknown center input frame: "+str(input_frame))
    model = make_center(data.get("architecture", "local_v1")).eval()
    model.input_frame=input_frame
    model.load_state_dict(data["model"])
    model.requires_grad_(False)
    return model, data


def center_plan(depth, camera, config, robot_mask=None, model=None):
    """Shared fixed-heading goal conversion; geometric or learned midpoint."""
    quat, normal = heading(config)
    seen = observe_depth(depth,camera,config,robot_mask)
    p = seen["points"]
    board_z = float(config["board_origin_base"][2])+seen["board_offset_m"]
    if model is not None and getattr(model,"input_frame",BASE_FRAME)==TABLE_FRAME:
        features,origin_table,table_origin=table_cloud_input(p,config,seen["board_offset_m"])
        origin=origin_table+table_origin
    else:
        features, origin = cloud_input(p, board_z)
    quantiles = config.get("extent_quantiles", [.02,.98])
    geometric_y = float(np.quantile(p[:,1],quantiles).mean())
    if model is None:
        predicted_y = geometric_y
        mode = "fixed_heading_geometric_midpoint_v1"
    else:
        with torch.no_grad():
            residual = float(model(torch.from_numpy(features)[None])[0])*SCALE_M
        if not np.isfinite(residual) or abs(residual) > .050001:
            raise ValueError("Invalid predicted midpoint offset")
        predicted_y = origin[1]+residual
        mode = "fixed_heading_learned_midpoint_v1"
    centre = np.array([np.quantile(p[:,0],quantiles).mean(),predicted_y,board_z])
    oriented = blade_points(np.r_[np.zeros(3),quat],config["vertices_in_ee"])
    edge = oriented[oriented[:,2] < oriented[:,2].min()+.0002].mean(0)
    fitted = copy.deepcopy(config)
    fitted["prescribed_ee_goal"] = np.r_[centre-edge,quat].tolist()
    result = estimate_plan(depth,camera,fitted,robot_mask)
    result.update(mode=mode,selected_plane_point_base=centre.tolist(),
                  selected_plane_normal_base=normal.tolist(),prescribed_cut_pose_used=False,
                  predicted_centroid_offset_m=float(predicted_y-origin[1]),
                  geometric_midpoint_y_m=geometric_y,
                  input_frame=getattr(model,"input_frame",BASE_FRAME) if model is not None else BASE_FRAME,
                  selection_scope="Fixed heading; full bounding-box midpoint target, not volume or material identity")
    return result
