"""Read-only physical geometry snapshots for both training backends."""
import json
from pathlib import Path
import numpy as np
import torch

@torch.no_grad()
def blade_world_vertices(poses, vertices, to_world):
    """Preserve each world's authoritative tool pose before changing frames."""
    from cutting_gpu_backend.controller.fdcc import quat_apply
    if poses.ndim != 2 or poses.shape[-1] != 7 or vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError('Expected batched tool poses and local blade vertices')
    vertices = vertices.to(poses)
    points = quat_apply(poses[:, None, 3:].expand(-1, len(vertices), -1),
                        vertices[None].expand(len(poses), -1, -1)) + poses[:, None, :3]
    result = np.asarray(to_world(points.reshape(-1, 3))).reshape(len(poses), len(vertices), 3)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite physical blade snapshot')
    return result

@torch.no_grad()
def capture_group(env, backend_root, output):
    # Imports are deferred: ordinary training does not construct a renderer.
    import mujoco
    from scripts.record_episode import GeometryRecorder
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    frame=env.current
    count=len(frame['joint_q'])
    recorder=GeometryRecorder(env,Path(backend_root))
    recorder(frame,None);recorder.save(output,1)
    (output/'task.json').write_text(json.dumps(env.metadata))
    template=dict(np.load(output/'episode.npz',allow_pickle=False))
    for key in ('tri','cut_tri','cut_edge_indices','cut_edge_coords'):
        if len(template[key]) % count:
            raise ValueError('Unequal per-world topology blocks')
        template[key]=template[key][:len(template[key])//count]
    blades=blade_world_vertices(frame['knife_pose'].double(),
        recorder.material.model.knife_tri_vertices.reshape(-1,3),recorder.to_world)
    arrays={'knife_world':blades,'knife_tri':template['knife_tri'],
            'joint_q':frame['joint_q'].detach().cpu().numpy(),
            'particles':recorder.to_world(frame['particle_q'].reshape(-1,3)).reshape(count,-1,3),
            'world_id':np.arange(count)}
    if template['cut_edge_indices'].max()>=arrays['particles'].shape[1]:
        raise ValueError('Snapshot topology is not world-local')
    np.savez_compressed(output/'episode.npz',**template)
    bodies=recorder.mj.geom_bodyid[template['geoms']]
    positions=[];rotations=[]
    for q in arrays['joint_q']:
        recorder.data.qpos[recorder.qpos_indices]=q
        mujoco.mj_forward(recorder.mj,recorder.data)
        positions.append(recorder.data.xpos[bodies].copy())
        rotations.append(recorder.data.xmat[bodies].copy())
    arrays.update(body_pos=np.stack(positions),body_mat=np.stack(rotations))
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite training snapshot')
    np.savez_compressed(output/'world_states.npz',**arrays)
    (output/'report.json').write_text(json.dumps(dict(worlds=count,scope='running_training_snapshot')))

@torch.no_grad()
def capture_training_environment(env, backend_root, output, status):
    if getattr(getattr(env,'physics',None),'batch_groups',None):
        return capture_blademaster_environment(env,backend_root,output,status)
    if not hasattr(env,'members'):
        raise ValueError('Training geometry snapshots currently support DiSECt object groups')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    groups=[];pending=[];offset=0
    for group,member in zip(env.groups,env.members):
        path=output/('group_%02d'%len(groups))
        if hasattr(member,'send'):
            member.send('capture',str(path))
            pending.append(member)
        else:
            capture_group(member.env,backend_root,path)
        groups.append(dict(name=group.name,path=path.name,worlds=group.worlds,first_world=offset))
        offset+=group.worlds
    for member in pending:
        result,payload=member._receive_message()
        if result!='ok':
            raise RuntimeError('Snapshot worker failed: '+str(payload))
    if offset!=env.worlds:raise ValueError('Snapshot group count differs from training')
    report=dict(format='training_geometry_v1',worlds=offset,groups=groups,
                training=dict(status),scope='Actual running training states; grid offsets are display only')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report

@torch.no_grad()
def capture_blademaster_environment(env, backend_root, output, status):
    import mujoco
    import warp as wp
    import xml.etree.ElementTree as ET
    from cutting_gpu_backend.controller.fdcc import quat_apply
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    physics=env.physics
    mj=mujoco.MjModel.from_binary_path(str(Path(backend_root)/'runs/model_coupled/robot.mjb'))
    data=mujoco.MjData(mj);mujoco.mj_forward(mj,data)
    base=mj.body(env.metadata['robot_base']).id
    base_pos=data.xpos[base].copy();base_mat=data.xmat[base].reshape(3,3).copy()
    pose=physics.base_from_material.double()
    def world(points):
        p=torch.as_tensor(points,device=env.device,dtype=torch.float64)
        flat=p.reshape(-1,3)
        transformed=quat_apply(pose[3:].expand(len(flat),-1),flat)+pose[:3]
        return (transformed.cpu().numpy()@base_mat.T+base_pos).reshape(p.shape)
    links={x.get('name') for x in ET.fromstring(env.metadata['robot_xml']).findall('link')}
    geoms=[g for g in range(mj.ngeom) if mj.body(mj.geom_bodyid[g]).name in links and mj.geom_rgba[g,3]>0]
    bodies=mj.geom_bodyid[geoms]
    qindices=[mj.jnt_qposadr[mj.joint(n).id] for n in env.metadata['joint_names']]
    ground=world(np.array([[-.15,-.15,0],[.15,-.15,0],[.15,.15,0],[-.15,.15,0]]))
    blades=blade_world_vertices(physics.tool_pose().double(),physics.vertices,world)
    if blades.shape[1] != 10:
        raise ValueError('Unsupported physical blade topology; do not substitute a visual tool')
    faces=[]
    for k in range(1,4):faces.extend([(0,k+1,k),(5,5+k,5+k+1)])
    for k in range(5):
        j=(k+1)%5;faces.extend([(k,j,j+5),(k,j+5,k+5)])
    faces=np.asarray(faces,dtype=np.int64)
    groups=[]
    for index,(sl,solver) in enumerate(physics.batch_groups):
        root=output/('group_%02d'%index);root.mkdir()
        q=physics.robot.q[sl].double().cpu().numpy()
        particles=world(wp.to_torch(solver.x.data))
        positions=[];rotations=[]
        for row in q:
            data.qpos[qindices]=row;mujoco.mj_forward(mj,data)
            positions.append(data.xpos[bodies].copy());rotations.append(data.xmat[bodies].copy())
        arrays=dict(knife_world=blades[sl],knife_tri=faces,joint_q=q,particles=particles,world_id=np.arange(len(q)),
                    body_pos=np.stack(positions),body_mat=np.stack(rotations))
        if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite BM snapshot')
        np.savez_compressed(root/'world_states.npz',**arrays)
        np.savez_compressed(root/'episode.npz',ground=ground,
                           base_position_world=base_pos,base_rotation_world=base_mat,
                           geom_body=np.array([mj.body(b).name for b in bodies]))
        (root/'task.json').write_text(json.dumps(env.metadata))
        (root/'report.json').write_text(json.dumps(dict(worlds=len(q),scope='running_training_snapshot')))
        groups.append(dict(name=env.object_names[sl.start],path=root.name,worlds=len(q),
                           first_world=sl.start,representation='material_points',particle_radius_m=float(solver.radius)))
    if sum(g['worlds'] for g in groups)!=env.worlds:raise ValueError('Missing BM worlds')
    report=dict(format='training_geometry_v1',worlds=env.worlds,groups=groups,
                training=dict(status),scope='Actual BM material particles and robot poses; display-only grid')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report
