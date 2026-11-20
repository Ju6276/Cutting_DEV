import json
from pathlib import Path
import numpy as np
import pytest
import torch
from test_knife_board import material, BOARD
from cutting_gpu_backend.knife_board import KnifeBoard
from cutting_world_model.free_cut import validate_task
from cutting_world_model.cad_knife import load
from cutting_world_model.shared_scene import configs, fingerprint

def test_curved_leading_edge_is_not_replaced_by_lowest_point_or_chord():
 m=material()
 edge=torch.tensor([[0.,.01,-.075],[0.,0.,0.],[0.,.02,.075]],dtype=torch.float64)
 m.model.knife_edge_vertices=edge
 m.model.knife_board_vertices=torch.cat((m.model.knife_tri_vertices,edge))
 b=KnifeBoard(m,BOARD)
 heights,inside=b.signed_height(m.position,m.rotation,b.midpoints(),b.footprint())
 torch.testing.assert_close(heights,torch.tensor([[.044,.028]],dtype=torch.float64))
 assert inside.all()
 assert b.edge.shape==(3,3)
 assert b.gap().item()==pytest.approx(.04)

def test_float32_material_reset_accepts_double_precision_robot_clearance():
 m=material();m.position=m.position.float();m.rotation=m.rotation.float()
 b=KnifeBoard(m,{**BOARD,'motion':'project_continue_v2'})
 b.reset(torch.tensor([True]),gap=torch.tensor([3e-6],dtype=torch.float64))
 assert b.contacted.item()
 assert b.min_gap.dtype==torch.float32
 assert b.min_gap.item()==pytest.approx(3e-6)

def test_current_ur10e_contract_keeps_task_rules():
 root=Path(__file__).resolve().parents[1]
 old=json.loads((root/'configs/shared/scene_v1.json').read_text())
 new=json.loads((root/'configs/shared/ur10e_ati_ikea_v1.json').read_text())
 task=new['common']['task'];validate_task(task)
 a=dict(old['common']['task']);b=dict(task)
 for key in ('workspace','tool_revision'):a.pop(key);b.pop(key)
 assert a==b
 assert old['common']['force_limits']==new['common']['force_limits']
 assert new['robot_platform']
 prepared=dict(contract_sha256=fingerprint(new),objects={o['name']:dict(scene='scene.pt',
                elastic=o['elastic'],blademaster_geometry_asset=o['blademaster_geometry_asset']) for o in new['objects']})
 d,b=configs(new,prepared)
 assert d['robot_platform']==b['robot_platform']==new['robot_platform']
 assert [g['start_clearance_m'] for g in d['groups']]==[
  b['start_clearance_m'][i*3:(i+1)*3] for i in range(len(new['objects']))]

def test_cad_geometry_hash_mismatch_is_not_silently_accepted(tmp_path):
 path=tmp_path/'data.npz';np.savez(path,vertices=np.zeros((3,3)))
 (tmp_path/'cutting_geometry.json').write_text(json.dumps(dict(revision='ikea_cad_mesh_v1',
     geometry=str(path),geometry_sha256='0'*64)))
 with pytest.raises(ValueError,match='hash mismatch'):load(str(tmp_path))

def test_float32_real_ur10e_board_projection_has_finite_gradients():
 from cutting_world_model.robot_platform import RobotPlatform
 from cutting_world_model.kinematics import SerialKinematics
 from cutting_world_model.cad_knife import mount
 from cutting_gpu_backend.coupling import quat_apply
 from cutting_gpu_backend.tool_mount import project_joint_endpoint, gap_from_pose
 root=Path('/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_estimated_v1')
 if not (root/'platform.json').exists():pytest.skip('user UR10e assets not installed')
 p=RobotPlatform.load(root);m=p.metadata
 fk=SerialKinematics(m['robot_xml'],m['robot_base'],m['robot_tip'],m['joint_names']).float()
 fk.second_order_safe=True
 _,arrays=load(str(root))
 T=mount(m,torch.zeros((),dtype=torch.float32))
 v=torch.tensor(arrays['board_vertices_disect'],dtype=torch.float32)
 vv=quat_apply(T[3:].expand(len(v),-1),v)+T[:3]
 geometry=dict(vertices_in_ee=vv.tolist(),board_origin_base=[0,0,.37],board_normal_base=[0,0,1],stop_gap_m=1e-6)
 q=torch.tensor(p.q,dtype=torch.float32)[None]
 candidates=q.repeat(2,1);candidates[:,1]+=torch.tensor([.05,-.05])
 q=candidates[gap_from_pose(fk(candidates),geometry).argmin():][:1].clone().requires_grad_(True)
 assert gap_from_pose(fk(q),geometry).item()<-.001
 old=fk(q);out,limited=project_joint_endpoint(q,fk,geometry)
 assert limited.item() and out.dtype==q.dtype
 actual=fk(out)
 assert gap_from_pose(actual,geometry).item()>=0.
 torch.testing.assert_close(actual[:,:2],old[:,:2],atol=1e-6,rtol=0)
 torch.testing.assert_close(actual[:,3:],old[:,3:],atol=1e-6,rtol=0)
 gradient=torch.autograd.grad(out.square().sum(),q)[0]
 assert torch.isfinite(gradient).all() and gradient.abs().sum()>0

def test_float32_projection_rounding_for_real_ur10e_poses():
 from cutting_world_model.robot_platform import RobotPlatform
 from cutting_world_model.kinematics import SerialKinematics
 from cutting_world_model.cad_knife import mount
 from cutting_gpu_backend.coupling import quat_apply
 from cutting_gpu_backend.tool_mount import project_joint_endpoint,gap_from_pose
 root=Path('/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_estimated_v1')
 if not (root/'platform.json').exists():pytest.skip('user UR10e assets not installed')
 p=RobotPlatform.load(root);m=p.metadata
 fk=SerialKinematics(m['robot_xml'],m['robot_base'],m['robot_tip'],m['joint_names']).float()
 fk.second_order_safe=True
 _,arrays=load(str(root));T=mount(m,torch.zeros((),dtype=torch.float32))
 v=torch.tensor(arrays['board_vertices_disect'],dtype=torch.float32)
 v=quat_apply(T[3:].expand(len(v),-1),v)+T[:3]
 geo=dict(vertices_in_ee=v.tolist(),board_origin_base=[0,0,.37],board_normal_base=[0,0,1],stop_gap_m=1e-6)
 torch.manual_seed(148)
 q=torch.tensor(p.q,dtype=torch.float32)[None]+.05*torch.randn(100,6)
 gaps=gap_from_pose(fk(q),geo);q=q[(gaps<0)&(gaps>-.03)].requires_grad_(True)
 assert len(q)>10
 out,limited=project_joint_endpoint(q,fk,geo)
 assert limited.all()
 assert (gap_from_pose(fk(out),geo)>=geo['stop_gap_m']-1e-9).all()
 torch.testing.assert_close(fk(out)[:,:2],fk(q)[:,:2],atol=1e-6,rtol=0)
 torch.testing.assert_close(fk(out)[:,3:],fk(q)[:,3:],atol=1e-6,rtol=0)
 assert torch.isfinite(torch.autograd.grad(out.square().sum(),q)[0]).all()
