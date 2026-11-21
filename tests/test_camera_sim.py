import json
from pathlib import Path
import numpy as np
from cutting_world_model.camera_sim import nano_camera,projection_matrix,range_depth
from cutting_world_model.visual_cut import DepthCamera

def test_camera_intrinsics_match_opengl_projection():
    cam=DepthCamera(1280,1024,1745.,2048.,630.,510.,np.eye(4))
    points=np.array([[.02,.03,.4],[-.04,-.02,.35],[0,0,.43]])
    gl=np.c_[points[:,0],-points[:,1],-points[:,2],np.ones(3)]
    clip=gl@projection_matrix(cam).T
    ndc=clip[:,:3]/clip[:,3,None]
    pixels=np.c_[(ndc[:,0]+1)*cam.width/2-.5,(1-ndc[:,1])*cam.height/2-.5]
    np.testing.assert_allclose(pixels,cam.project(points),atol=1e-10)

def test_nano_placement_and_depth_range():
    p=json.loads((Path(__file__).resolve().parents[1]/"configs/vision/mecheye_nano_provisional_v1.json").read_text())
    target=np.array([.628,0,.37]);cam=nano_camera(p,target)
    np.testing.assert_allclose(cam.project(target),[cam.cx,cam.cy],atol=1e-10)
    assert np.isclose(np.linalg.norm(target-cam.base_from_camera[:3,3]),.43)
    assert cam.base_from_camera[0,3] > target[0]
    np.testing.assert_array_equal(range_depth([.299,.3,.45,.451,np.nan,np.inf],p),[0,.3,.45,0,0,0])


def test_named_sensor_rejects_inconsistent_frames_and_intrinsics():
    import pytest
    from cutting_world_model.camera_sim import validate_sensor_capture
    p=json.loads((Path(__file__).resolve().parents[1]/"configs/vision/mecheye_nano_provisional_v1.json").read_text())
    cam=nano_camera(p,[.628,0,.37])
    validate_sensor_capture(cam,p,np.eye(4))
    with pytest.raises(ValueError,match="world_from_base"):
        validate_sensor_capture(cam,p,None)
    moved=np.eye(4);moved[0,3]=.1
    with pytest.raises(ValueError,match="world =="):
        validate_sensor_capture(cam,p,moved)
    wrong=dict(p,fx=p["fx"]+1)
    with pytest.raises(ValueError,match="intrinsics"):
        validate_sensor_capture(cam,wrong,np.eye(4))
