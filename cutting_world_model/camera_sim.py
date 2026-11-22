"""Geometric camera adapter; no claim of a calibrated vendor sensor simulator."""
import numpy as np
from .visual_cut import DepthCamera

def nano_camera(profile, board_origin):
    if profile["revision"] not in ("mecheye_nano_provisional_v1", "mecheye_nano_v4_550_lab_v1"):
        raise ValueError("Unsupported camera profile")
    lo,hi=np.asarray(profile["depth_range_m"],float)
    distance=float(profile["board_distance_m"])
    angle=np.radians(float(profile["tilt_from_vertical_deg"]))
    if not (0 < lo < hi and lo <= distance <= hi and 0 < angle < np.pi/2):
        raise ValueError("Invalid camera range or placement")
    target=np.asarray(board_origin,float)
    if target.shape!=(3,) or not np.isfinite(target).all():
        raise ValueError("Invalid board origin")
    eye=target+distance*np.array([np.sin(angle),0,np.cos(angle)])
    forward=(target-eye)/distance
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    t=np.eye(4);t[:3,:3]=np.stack([right,down,forward],1);t[:3,3]=eye
    return DepthCamera(**{k:profile[k] for k in ("width","height","fx","fy","cx","cy")},
                       base_from_camera=t)

def projection_matrix(camera, near=.01, far=2.):
    """OpenGL projection with explicit intrinsics, including pixel-centre convention."""
    if not (0 < near < far):raise ValueError("Invalid clipping planes")
    w,h=camera.width,camera.height
    return np.array([[2*camera.fx/w,0,1-2*(camera.cx+.5)/w,0],
                     [0,2*camera.fy/h,2*(camera.cy+.5)/h-1,0],
                     [0,0,-(far+near)/(far-near),-2*far*near/(far-near)],
                     [0,0,-1,0]],float)

def range_depth(depth, profile):
    z=np.asarray(depth,dtype=float)
    lo,hi=profile["depth_range_m"]
    if not (0 < lo < hi):raise ValueError("Invalid camera depth range")
    valid=np.isfinite(z)&(z>=lo)&(z<=hi)
    return np.where(valid,z,0.)


def validate_sensor_capture(camera, profile, world_from_base):
    """A named sensor profile may not silently use unrelated calibration."""
    expected=np.array([profile[k] for k in ("width","height","fx","fy","cx","cy")],float)
    actual=np.array([getattr(camera,k) for k in ("width","height","fx","fy","cx","cy")],float)
    if not np.isfinite(expected).all() or not np.allclose(actual,expected,rtol=0,atol=1e-8):
        raise ValueError("Camera intrinsics do not match the named sensor profile")
    if world_from_base is None:
        raise ValueError("Sensor capture requires explicit world_from_base calibration")
    t=np.asarray(world_from_base,float)
    if t.shape != (4,4) or not np.isfinite(t).all() or not np.allclose(t,np.eye(4),rtol=0,atol=1e-7):
        raise ValueError("This studio snapshot adapter requires world == robot base")
