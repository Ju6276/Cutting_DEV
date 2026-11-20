import json, os
from pathlib import Path
import numpy as np
import pytest
from cutting_world_model.compliance_frame import configure, BASE_FROM_COMPLIANCE
from cutting_world_model.robot_platform import frame_poses

def test_compliance_axes_preserve_origin_and_are_idempotent():
    xml = '<robot name="test"><link name="base"/><link name="knife_mount"/><joint name="mount" type="fixed"><parent link="base"/><child link="knife_mount"/><origin xyz=".2 .1 .3" rpy=".2 .3 .4"/></joint></robot>'
    m=dict(robot_base="base",robot_tip="knife_mount",joint_names=[])
    out, _ = configure(xml,m,[])
    p=frame_poses(dict(m,robot_xml=out),[])
    np.testing.assert_allclose(p["knife_compliance"][:3,3],p["knife_mount"][:3,3])
    np.testing.assert_allclose(p["knife_compliance"][:3,:3],BASE_FROM_COMPLIANCE,atol=1e-14)
    again,_=configure(out,m,[])
    assert again==out
    k=BASE_FROM_COMPLIANCE@np.diag([5000,2000,2000])@BASE_FROM_COMPLIANCE.T
    np.testing.assert_array_equal(k,np.diag([2000,2000,5000]))

def test_prepared_ur10e_policy_stiffness_controls_downward_axis():
    import torch
    from cutting_world_model.robot_platform import RobotPlatform
    from cutting_gpu_backend.controller.fdcc import quat_matrix, SlicingActionMapper
    root=Path(os.environ.get("CUTTING_UR10E_COMPLIANCE_PLATFORM",
        "/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_fdcc_v2"))
    if not (root/"platform.json").exists():pytest.skip("prepared corrected UR10e bundle absent")
    p=RobotPlatform.load(root);chain,c=p.controller()
    f=chain(torch.tensor(p.q,dtype=torch.float64)[None],mass_matrix=False)[0]
    r=quat_matrix(f[c.config.compliance_ref_link][:,3:])[0]
    nominal=r@torch.diag(torch.tensor(c.config.stiffness[:3],dtype=torch.float64))@r.T
    torch.testing.assert_close(nominal,torch.diag(torch.tensor([2000.,2000.,5000.],dtype=torch.float64)),atol=1e-8,rtol=0)
    mapper=SlicingActionMapper();mapper.reset(2,dtype=torch.float64)
    a=torch.zeros(2,6,dtype=torch.float64);a[1,0]=1
    pose=f[chain.tip].repeat(2,1)
    k=mapper(a,pose,pose)["stiffness"][:,:3]
    delta=r@torch.diag(k[1]-k[0])@r.T
    torch.testing.assert_close(delta,torch.diag(torch.tensor([0.,0.,9000.],dtype=torch.float64)),atol=1e-8,rtol=0)
    # Rotational stiffness action remains the cutting-plane pitch channel.
    a.zero_();a[1,1]=1;mapper.reset(2,dtype=torch.float64)
    k=mapper(a,pose,pose)["stiffness"][:,3:]
    delta=r@torch.diag(k[1]-k[0])@r.T
    torch.testing.assert_close(delta,torch.diag(torch.tensor([0.,160.,0.],dtype=torch.float64)),atol=1e-8,rtol=0)
