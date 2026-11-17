"""Route 1: calibrated depth -> prescribed single-cut plan; no learned perception.

Only measured depth, calibration, known robot exclusion and prescribed task
geometry enter the estimator. No object mesh, category or material labels.
Coordinates: metres, camera x right/y down/z forward; poses xyzw in robot base.
"""
from dataclasses import dataclass
import numpy as np
from collections import deque


def connected_components(mask):
    labels=np.zeros(mask.shape,dtype=np.int32);count=0
    height,width=mask.shape
    for y,x in zip(*np.nonzero(mask)):
        if labels[y,x]:continue
        count+=1;labels[y,x]=count;queue=deque([(y,x)])
        while queue:
            cy,cx=queue.popleft()
            for ny,nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)]:
                if 0<=ny<height and 0<=nx<width and mask[ny,nx] and not labels[ny,nx]:
                    labels[ny,nx]=count;queue.append((ny,nx))
    return labels,count


def quaternion_matrix(q):
    x,y,z,w=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


@dataclass
class DepthCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    base_from_camera: np.ndarray

    def __post_init__(self):
        self.base_from_camera = np.asarray(self.base_from_camera, dtype=float)
        t = self.base_from_camera
        if (self.width < 2 or self.height < 2 or
                not np.isfinite([self.fx,self.fy,self.cx,self.cy]).all() or min(self.fx,self.fy)<=0 or
                t.shape!=(4,4) or not np.isfinite(t).all() or
                not np.allclose(t[3],[0,0,0,1]) or
                not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-7) or
                not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-7)):
            raise ValueError("Invalid pinhole camera calibration")

    def points(self, depth):
        z = np.asarray(depth, dtype=float)
        if z.shape != (self.height,self.width):
            raise ValueError("Depth shape differs from camera")
        u,v = np.meshgrid(np.arange(self.width),np.arange(self.height))
        cam = np.stack(((u-self.cx)*z/self.fx,(v-self.cy)*z/self.fy,z),-1)
        return cam@self.base_from_camera[:3,:3].T+self.base_from_camera[:3,3]

    def project(self, points):
        p=(np.asarray(points)-self.base_from_camera[:3,3])@self.base_from_camera[:3,:3]
        if np.any(p[...,2]<=0):raise ValueError("Point behind camera")
        return np.stack((self.fx*p[...,0]/p[...,2]+self.cx,
                         self.fy*p[...,1]/p[...,2]+self.cy),-1)

    def to_dict(self):
        return dict(width=self.width,height=self.height,fx=self.fx,fy=self.fy,
                    cx=self.cx,cy=self.cy,base_from_camera=self.base_from_camera.tolist())


def front_camera(board_origin, width=640, height=480, offset=(0,.32,.55), focal_px=700):
    """Front means +base Y beyond the workpiece, looking back/down at the board."""
    target=np.asarray(board_origin,float)
    eye=target+np.asarray(offset,float)
    forward=(target-eye)/np.linalg.norm(target-eye)
    right=np.cross(forward,[0,0,1])
    if np.linalg.norm(right)<1e-8:raise ValueError("Camera up vector is degenerate")
    right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    t=np.eye(4);t[:3,:3]=np.stack((right,down,forward),1);t[:3,3]=eye
    return DepthCamera(width,height,focal_px,focal_px,(width-1)/2,(height-1)/2,t)


def blade_points(pose, vertices):
    pose=np.asarray(pose,float)
    if pose.shape!=(7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]),1,atol=1e-5):
        raise ValueError("Expected finite base-frame EE pose with unit xyzw quaternion")
    return np.asarray(vertices)@quaternion_matrix(pose[3:]).T+pose[:3]


def observe_depth(depth, camera, config, robot_mask=None):
    """Extract a visible surface without consulting any cut target."""
    z=np.asarray(depth,float)
    points=camera.points(z)
    origin=np.asarray(config["board_origin_base"],float)
    normal=np.asarray(config["board_normal_base"],float)
    if normal.shape!=(3,) or not np.isclose(np.linalg.norm(normal),1,atol=1e-6):
        raise ValueError("Unit board normal required")
    axis=np.asarray(config.get("board_x_base",[1,0,0]),float)
    axis=axis-normal*np.dot(axis,normal)
    if np.linalg.norm(axis)<1e-8:raise ValueError("Invalid board X axis")
    axis/=np.linalg.norm(axis);across=np.cross(normal,axis)
    uv=np.stack(((points-origin)@axis,(points-origin)@across),-1)
    half=np.asarray(config["board_half_extent_m"],float)
    if half.shape!=(2,) or not np.isfinite(half).all() or np.any(half<=0):raise ValueError("Positive board ROI required")
    valid=np.isfinite(z)&(z>0)&np.isfinite(points).all(-1)&(np.abs(uv)<half).all(-1)
    if robot_mask is not None:
        mask=np.asarray(robot_mask)
        if mask.shape!=z.shape or mask.dtype!=bool:raise ValueError("Boolean robot mask required")
        valid &= ~mask
    height=(points-origin)@normal
    candidates=height[valid & (np.abs(height)<config.get("board_search_m",.008))]
    if len(candidates)<config.get("minimum_board_pixels",300):
        raise ValueError("Board insufficiently visible")
    # Largest narrow histogram mode avoids fitting the food surface as the board.
    search=config.get("board_search_m",.008)
    edges=np.linspace(-search,search,81)
    hist,_=np.histogram(candidates,edges)
    centre=(edges[np.argmax(hist)]+edges[np.argmax(hist)+1])/2
    inliers=candidates[np.abs(candidates-centre)<.0007]
    if len(inliers)<config.get("minimum_board_pixels",300):
        raise ValueError("No supported board plane")
    offset=float(np.median(inliers))
    if abs(offset)>config.get("max_board_correction_m",.002):
        raise ValueError("Board correction exceeds validated range")
    observed_origin=origin+offset*normal
    height=height-offset
    object_mask=valid&(height>config.get("minimum_object_height_m",.002))&(height<config.get("maximum_object_height_m",.12))
    labels,count=connected_components(object_mask)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    if not count or sizes.max()<config.get("minimum_object_pixels",50):
        raise ValueError("Object insufficiently visible")
    keep=int(sizes.argmax())
    if np.sum(sizes>=max(50,.2*sizes[keep]))>1:
        raise ValueError("Ambiguous multiple foreground components")
    object_mask=labels==keep
    p=points[object_mask];heights=height[object_mask]
    return dict(points=p, heights=heights, object_mask=object_mask,
                board_offset_m=offset, board_pixels=int(len(inliers)))

def estimate_plan(depth, camera, config, robot_mask=None):
    """Validate and instantiate a prescribed cut; never select one from geometry."""
    seen=observe_depth(depth,camera,config,robot_mask)
    p,heights=seen["points"],seen["heights"]
    object_mask=seen["object_mask"];offset=seen["board_offset_m"]
    origin=np.asarray(config["board_origin_base"],float)
    normal=np.asarray(config["board_normal_base"],float)
    observed_origin=origin+offset*normal
    # Robust top quantile, plus explicit conservative margin. No hidden true top.
    surface_top=float(np.quantile(heights,config.get("top_quantile",.995)))
    margin_height=float(config.get("height_margin_m",.001))
    if not np.isfinite(margin_height) or margin_height<0:
        raise ValueError("Height margin must be finite and nonnegative")
    top=surface_top+margin_height
    clearance=float(config["clearance_m"]);speed=float(config["target_speed_mps"])
    low,high=config["clearance_range_m"]
    vlo,vhi=config["target_speed_range_mps"]
    if not low<=clearance<=high or clearance+margin_height>high or not vlo<=speed<=vhi:
        raise ValueError("Clearance/speed outside current trained task range")
    goal=np.asarray(config["prescribed_ee_goal"],float).copy()
    vertices=np.asarray(config["vertices_in_ee"],float)
    blade=blade_points(goal,vertices)
    stop=float(config["stop_gap_m"])
    goal[:3]+=(stop-np.min((blade-observed_origin)@normal))*normal
    blade=blade_points(goal,vertices)
    # Prescribed blade direction and slicing plane; no PCA choice of a new cut.
    _,_,vh=np.linalg.svd(blade-blade.mean(0),full_matrices=False)
    along=vh[0]-normal*np.dot(vh[0],normal);along/=np.linalg.norm(along)
    cut_normal=np.cross(normal,along);cut_normal/=np.linalg.norm(cut_normal)
    distances=(blade-observed_origin)@normal
    edge=blade[distances<distances.min()+.0002]
    cut_point=edge.mean(0)
    lateral=(p-cut_point)@cut_normal
    margin=config.get("coverage_margin_m",.001)
    if not (lateral.min() < -margin and lateral.max() > margin):
        raise ValueError("Prescribed cut plane misses visible object footprint")
    strip=np.abs(lateral)<config.get("cut_strip_half_width_m",.004)
    if strip.sum()<10:raise ValueError("Too little visible surface near prescribed cut")
    along_p=(p[strip]-cut_point)@along
    along_b=(blade-cut_point)@along
    if along_p.min()<along_b.min() or along_p.max()>along_b.max():
        raise ValueError("Visible cross-section exceeds blade length")
    start=goal.copy()
    start[:3]+=(top+clearance-stop)*normal
    bounds=config["workspace"]
    for name,pose in [("start",start),("goal",goal)]:
        if np.any(pose[:3]<bounds["lower"]) or np.any(pose[:3]>bounds["upper"]):
            raise ValueError(name+" is outside robot EE workspace")
    return dict(revision="visual_prescribed_cut_v1",frame="robot_base",quaternion="xyzw",
        mode="prescribed_plane_height_only",board_offset_m=offset,
        surface_top_m=surface_top,conservative_top_m=top,clearance_m=clearance,
        target_speed_mps=speed,start_ee_pose=start.tolist(),goal_ee_pose=goal.tolist(),
        policy_goal=np.r_[goal,speed].tolist(),cut_point_base=cut_point.tolist(),
        cut_normal_base=cut_normal.tolist(),board_pixels=seen["board_pixels"],
        object_pixels=int(object_mask.sum()),footprint_coverage=True,
        coverage_scope="visible surface near prescribed plane; not material-separation evidence",
        preparation="move to start with a separate checked robot motion; initialize real observation history",
        object_mask=object_mask)


def validate_policy_plan(plan, metadata, config):
    """Validate offline goal compatibility; does not fabricate observations or move the robot."""
    if metadata.get("goal_dim") != 8 or metadata.get("state_shapes", {}).get("observation") != [34]:
        raise ValueError("Route 1 adapter expects current 8-goal / 34-observation policy")
    if metadata.get("environment", {}).get("relative_to_ee") is not False:
        raise ValueError("Policy goals must use robot-base coordinates")
    if metadata.get("free_cut_task", {}).get("version") != "pose_depth_v2":
        raise ValueError("Route 1 adapter requires current pose_depth_v2 task")
    geometry = metadata["board_pose_geometry"]
    for key in ("vertices_in_ee", "board_origin_base", "board_normal_base"):
        a, b = np.asarray(geometry[key]), np.asarray(config[key])
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-7, rtol=0):
            raise ValueError("Camera plan/checkpoint geometry mismatch: " + key)
    if not np.isclose(geometry["stop_gap_m"], config["stop_gap_m"], atol=1e-9, rtol=0):
        raise ValueError("Board stop mismatch")
    goal = np.asarray(plan["policy_goal"], dtype=np.float32)
    if goal.shape != (8,) or not np.isfinite(goal).all() or plan.get("frame") != "robot_base":
        raise ValueError("Invalid policy goal")
    if not np.allclose(goal[:7], plan["goal_ee_pose"], atol=1e-7):
        raise ValueError("Inconsistent pose and policy goal")
    return goal

def visual_midpoint_plan(depth, camera, config, robot_mask=None):
    """Select the visible object's mid-section, using depth rather than preset XY.

    config supplies calibrated tool orientation, never a prescribed target pose.
    The rule is geometric, not an optimum for force, stability or slice volume.
    """
    import copy
    if "prescribed_ee_goal" in config:
        raise ValueError("Visual selection must not receive a prescribed cut pose")
    seen=observe_depth(depth,camera,config,robot_mask)
    points=seen["points"]
    normal=np.asarray(config["board_normal_base"],float)
    if not np.allclose(normal,[0,0,1],atol=1e-6):
        raise ValueError("Midpoint selector currently requires horizontal calibrated board")
    centred=points[:,:2]-points[:,:2].mean(0)
    values,vectors=np.linalg.eigh(centred.T@centred/len(centred))
    ratio=float(values[-1]/max(values[0],1e-12))
    if ratio<float(config.get("minimum_axis_ratio",2.)):
        raise ValueError("Visible major axis is ambiguous; supply a cut heading or user-selected plane")
    long=np.r_[vectors[:,-1],0.]
    canonical=np.asarray(config["tool_orientation_xyzw"],float)
    vertices=np.asarray(config["vertices_in_ee"],float)
    local=blade_points(np.r_[np.zeros(3),canonical],vertices)
    _,_,vh=np.linalg.svd(local-local.mean(0),full_matrices=False)
    edge=vh[0].copy();edge[2]=0;edge/=np.linalg.norm(edge)
    ref_normal=np.cross(normal,edge);ref_normal/=np.linalg.norm(ref_normal)
    if np.dot(long,ref_normal)<0:long=-long
    yaw=float(np.arctan2(np.cross(ref_normal,long)[2],np.dot(ref_normal,long)))
    if abs(np.degrees(yaw))>float(config.get("max_yaw_delta_deg",15)):
        raise ValueError("Visual cut heading exceeds the development yaw envelope")
    # q_yaw * q_tool, xyzw.
    z=np.sin(yaw/2);w=np.cos(yaw/2);v=canonical[:3]
    quat=np.r_[w*v+canonical[3]*np.array([0.,0.,z])+np.cross([0.,0.,z],v),
               w*canonical[3]-z*v[2]]
    quat/=np.linalg.norm(quat)
    reference=config.get("policy_quaternion_reference_xyzw",canonical)
    reference=np.asarray(reference,float)
    if reference.shape!=(4,) or not np.isfinite(reference).all() or not np.isclose(np.linalg.norm(reference),1.,atol=1e-5):
        raise ValueError("Invalid policy quaternion reference")
    # q and -q are physically equal, but the policy consumes raw goal coordinates.
    if np.dot(quat,reference)<0:quat=-quat
    across=np.cross(long,normal)
    quantiles=config.get("extent_quantiles",[.02,.98])
    lo,hi=np.quantile(points@long,quantiles)
    left,right=np.quantile(points@across,quantiles)
    centre=long*((lo+hi)/2)+across*((left+right)/2)
    plane_origin=np.asarray(config["board_origin_base"])+normal*seen["board_offset_m"]
    centre+=normal*(np.dot(plane_origin,normal)-np.dot(centre,normal))
    oriented=blade_points(np.r_[np.zeros(3),quat],vertices)
    heights=oriented@normal
    edge_centre=oriented[heights<heights.min()+.0002].mean(0)
    goal=np.r_[centre-edge_centre,quat]
    fitted=copy.deepcopy(config);fitted["prescribed_ee_goal"]=goal.tolist()
    result=estimate_plan(depth,camera,fitted,robot_mask)
    result.update(mode="visual_major_axis_midpoint_v1",
        selection_rule="midpoint of 2–98% visible extent along the horizontal major axis",
        selected_plane_point_base=centre.tolist(),selected_plane_normal_base=long.tolist(),
        visible_axis_ratio=ratio,yaw_delta_deg=float(np.degrees(yaw)),
        selected_extent_m=float(hi-lo),prescribed_cut_pose_used=False,
        selection_scope="visible-surface geometric midpoint; not best cut, force optimum, or full volume midpoint")
    return result
