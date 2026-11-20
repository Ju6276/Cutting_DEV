
"""Shared studio renderer for recorded stills and videos. Export with VTK, render with Blender.

No simulated coordinates are smoothed or replaced. Scan colors come from the
optional appearance map. Board/floor visuals and illumination are display assets.
"""
import argparse, json, sys, subprocess
from pathlib import Path
import numpy as np

def arguments():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    p.add_argument('--recording',type=Path)
    p.add_argument('--source-root',type=Path,action='append',default=[])
    p.add_argument('--show-contact-blade',action='store_true',help='Debug only: blue material-contact proxy overlay')
    p.add_argument('--view',choices=['close','wide'],default='close')
    p.add_argument('--highlight-cut-faces',action='store_true',help='Debug only: contrasting prepared cut faces')
    p.add_argument('--surface-map',type=Path)
    p.add_argument('--material',choices=['generic','cucumber','potato','prism'],default='generic')
    p.add_argument('--frame',type=int,default=0)
    p.add_argument('--video',action='store_true',help='Animate a studio evaluation export using this same style')
    p.add_argument('--preview',action='store_true',help='Only first, middle and final frames of the video')
    p.add_argument('--speed',type=float,default=.5)
    p.add_argument('--samples',type=int,default=96)
    p.add_argument('--fps',type=int,default=20)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--camera-from',type=Path,help='Reuse exact camera from another render JSON')
    return p.parse_args(args)

def export(a):
    import render_episode as r
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    d=np.load(a.recording/'episode.npz');i=a.frame
    task=json.loads((a.recording/'task.json').read_text())
    app=r.load_appearance(a.surface_map,a.recording/'episode.npz') if a.surface_map else None
    pts=r.display_vertices(d,i,app)
    colors=r.material_face_colors(d,a.material,app,a.highlight_cut_faces)/255.
    if app is None:
        faces=np.concatenate([d['tri'],d['cut_tri']])
    else:
        n=len(app['indices'])
        faces=np.concatenate([np.arange(n).reshape(-1,3),d['cut_tri']+n])
    normals=None
    if app is not None:
        original=r.material_vertices(d,i)
        exterior_normals=np.zeros_like(original)
        v=original[d['tri']]
        face_n=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
        for j in range(3):np.add.at(exterior_normals,d['tri'][:,j],face_n)
        exterior_normals/=np.maximum(np.linalg.norm(exterior_normals,axis=1,keepdims=True),1e-20)
        fine=(exterior_normals[app['indices']]*app['weights'][...,None]).sum(1)
        fine/=np.maximum(np.linalg.norm(fine,axis=1,keepdims=True),1e-20)
        cut=np.zeros_like(original);v=original[d['cut_tri']]
        face_n=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
        for j in range(3):np.add.at(cut,d['cut_tri'][:,j],face_n)
        cut/=np.maximum(np.linalg.norm(cut,axis=1,keepdims=True),1e-20)
        normals=np.concatenate([fine,cut])[faces].reshape(-1,3)
    out=dict(points=pts,faces=faces,colors=colors,mounted_cad_knife=bool(task.get('cad_tool')),knife=d['knife'][i],knife_tri=d['knife_tri'],
             ground=d['ground'],center=(d['particles'][i].min(0)+d['particles'][i].max(0))/2)
    robot,sources=r.visual_actors(d,task,a.source_root)
    robot=[entry for entry in robot if str(d['geom_body'][entry[0]])!='b_bot_cam_cables_link']
    for j,(k,actor,local) in enumerate(robot):
        actor.GetMapper().Update()
        triangles=vtk.vtkTriangleFilter();triangles.SetInputData(actor.GetMapper().GetInput());triangles.Update()
        poly=triangles.GetOutput()
        v=vtk_to_numpy(poly.GetPoints().GetData()).copy()
        f=vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1,4)
        if not np.all(f[:,0]==3):raise ValueError('Robot mesh must be triangular')
        rot=d['body_mat'][i][k].reshape(3,3)
        out[f'robot_v{j}']=(v@local[:3,:3].T+local[:3,3])@rot.T+d['body_pos'][i][k]
        out[f'robot_f{j}']=f[:,1:]
        out[f'robot_c{j}']=actor.GetProperty().GetColor()
        out[f'robot_tool{j}']=str(d['geom_body'][k])=='b_bot_right_inner_knuckle'
        body=str(d['geom_body'][k])
        if body in ('ikea_knife','ati_mini45_body'):out[f'robot_metallic{j}']=.7
        elif body in ('user_knife_holder','user_ur_adapter'):out[f'robot_metallic{j}']=0.
    if normals is not None:out['normals']=normals
    out['metadata']=np.array(json.dumps(dict(recording=str(a.recording),frame=i,
       sim_time=float(d['sim_time'][i]),robot_meshes=len(robot),sources=sources,
       highlight_cut_faces=a.highlight_cut_faces,
       appearance=json.loads(str(app['metadata'])) if app is not None else None,
       geometry='Recorded surfaces; barycentric display subdivision only; no physics changes')))
    a.scene.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.scene,**out)

def render(a):
    import bpy
    from mathutils import Vector
    d=dict(np.load(a.scene))
    meta=json.loads(str(d['metadata']))
    animated=meta.get('format')=='studio_evaluation_v1'
    variable=bool(d.get('dynamic_topology',False))
    def material_points(i):
        return d[f'points_{i}'] if variable else d['points'][i] if animated else d['points']
    def material_faces(i):return d[f'faces_{i}'] if variable else d['faces']
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    def mat(name,color,metal=0.,rough=.5):
        m=bpy.data.materials.new(name);m.use_nodes=True
        bs=m.node_tree.nodes.get('Principled BSDF')
        bs.inputs['Base Color'].default_value=(*color,1)
        bs.inputs['Metallic'].default_value=metal;bs.inputs['Roughness'].default_value=rough
        return m
    def mesh(name,v,f,m,smooth=False):
        me=bpy.data.meshes.new(name);me.from_pydata(v.tolist(),[],f.tolist());me.update()
        obj=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(obj)
        obj.data.materials.append(m)
        if smooth:
            # Smooth shading changes normals only, never the recorded geometry.
            for p in me.polygons:p.use_smooth=True
            if hasattr(me,'use_auto_smooth'):me.use_auto_smooth=True
        return obj
    skin=mat('Scanned appearance',(1,1,1),rough=.65)
    bs=skin.node_tree.nodes.get('Principled BSDF')
    col=skin.node_tree.nodes.new('ShaderNodeVertexColor');col.layer_name='scan_color'
    skin.node_tree.links.new(col.outputs['Color'],bs.inputs['Base Color'])
    obj=mesh('Recorded material',material_points(a.frame),material_faces(a.frame),skin,True)
    # Blender 2.82 MeshLoopColor stores display RGB; converting here again
    # would darken source scans twice. Use the legacy layer's native encoding.
    c=d['colors'];rgba=np.c_[c,np.ones(len(c))]
    layer=obj.data.vertex_colors.new(name='scan_color')
    layer.data.foreach_set('color',np.repeat(rgba,3,axis=0).ravel())
    if 'normals' in d:
        obj.data.normals_split_custom_set(d['normals'].tolist())
        if hasattr(obj.data,'use_auto_smooth'):obj.data.use_auto_smooth=True
    def update_display_normals(i):
        if 'normal_base_points' not in d:return
        x=d['normal_base_points'][i];faces=d['normal_base_faces']
        v=x[faces];fn=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
        normals=np.zeros_like(x)
        for k in range(3):np.add.at(normals,faces[:,k],fn)
        normals/=np.maximum(np.sqrt(np.sum(normals*normals,axis=1,keepdims=True)),1e-20)
        fine=(normals[d['normal_indices']]*d['normal_weights'][...,None]).sum(1)
        fine/=np.maximum(np.sqrt(np.sum(fine*fine,axis=1,keepdims=True)),1e-20)
        obj.data.normals_split_custom_set(fine[d['faces']].reshape(-1,3).tolist())
        obj.data.use_auto_smooth=True
    update_display_normals(a.frame)
    tool_material=mat('Mounted knife and adapter steel',(.44,.47,.50),.7,.3)
    robot_count=json.loads(str(d['metadata']))['robot_meshes']
    robot_objects=[]
    for j in range(robot_count):
        color=d[f'robot_c{j}']
        metal=float(d.get(f'robot_metallic{j}', .55 if np.mean(color)>.4 else .1))
        robot_obj=mesh(f'Robot visual {j}',d[f'robot_v{j}'],d[f'robot_f{j}'],
             tool_material if bool(d.get(f'robot_tool{j}',False)) else mat(f'robot {j}',color,metal,.3),True)
        robot_objects.append(robot_obj)
        if animated:
            from mathutils import Matrix
            robot_obj.matrix_world=Matrix(d[f'robot_transform{j}'][a.frame].tolist())
    physical_blade=None
    if animated and not bool(d.get('mounted_cad_knife', False)):
        physical_blade=mesh('Recorded physical blade',d['knife'][a.frame],d['knife_tri'],tool_material)
    if a.show_contact_blade and not animated:mesh('DiSECt contact blade',d['knife'],d['knife_tri'],mat('Contact blade blue steel',(.12,.29,.40),.7,.24))
    def cube(name,center,size,m):
        bpy.ops.mesh.primitive_cube_add(size=1,location=center)
        o=bpy.context.object;o.name=name;o.dimensions=size
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        o.data.materials.append(m)
        return o
    ground=d['ground'];bc=ground.mean(0);top=bc[2]
    boardmat=mat('Board display',(0.54,.38,.22),rough=.68)
    # Fine grain is decorative; it does not represent a measured board surface.
    nodes=boardmat.node_tree.nodes;links=boardmat.node_tree.links
    coord=nodes.new('ShaderNodeTexCoord');mapping=nodes.new('ShaderNodeVectorMath');mapping.operation='MULTIPLY'
    mapping.inputs[1].default_value=(3,95,4);links.new(coord.outputs['Generated'],mapping.inputs[0])
    noise=nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=3
    links.new(mapping.outputs[0],noise.inputs['Vector'])
    ramp=nodes.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(.28,.16,.075,1);ramp.color_ramp.elements[1].color=(.66,.49,.29,1)
    links.new(noise.outputs['Fac'],ramp.inputs[0]);links.new(ramp.outputs[0],nodes.get('Principled BSDF').inputs['Base Color'])
    cube('Recorded board plane with display thickness',bc-[0,0,.006],
         [np.ptp(ground[:,0]),np.ptp(ground[:,1]),.012],boardmat)
    if a.view=='close':
        cube('Display table',bc-[0,0,.023],[3,3,.034],mat('Table',(.32,.35,.37),rough=.8))
    else:
        cube('Display floor',[0,0,-.02],[8,8,.04],mat('Floor',(.54,.57,.60),rough=.8))
        cube('Display board support',[bc[0],bc[1],(top-.012)/2],
             [np.ptp(ground[:,0])*.85,np.ptp(ground[:,1])*.85,max(.001,top-.012)],
             mat('Support',(.42,.45,.48),rough=.65))
    center=d['center']
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=a.samples
    scene.cycles.use_denoising=True
    for layer in scene.view_layers:
        layer.cycles.use_denoising=True
    scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.78,.83,.9,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
    def area(name,delta,power,size):
        light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
        o=bpy.data.objects.new(name,light);bpy.context.collection.objects.link(o);o.location=center+delta
        o.rotation_euler=(Vector(center)-o.location).to_track_quat('-Z','Y').to_euler()
    area('Large soft key',np.array([.15,-.22,.32]),2.,.30)
    area('Soft fill',np.array([-.24,.08,.20]),1.,.26)
    bpy.ops.object.camera_add()
    camera=bpy.context.object;target=center+np.array([0,0,.012])
    camera.location=center+np.array([.22,-.32,.17])
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';camera.data.ortho_scale=.235
    if a.view=='wide':
        target=np.array([center[0]*.5,center[1]*.52,.30])
        camera.location=target+np.array([1.2,-1.6,.9])
        camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.data.ortho_scale=1.05
    if animated:
        # Fixed camera over the entire episode. Include the grip, not just the blade.
        tool_ids=[j for j in range(robot_count) if bool(d.get(f'robot_tool{j}',False))]
        fit=[material_points(0),d['knife'].reshape(-1,3)]
        if a.view=='close':
            for j in tool_ids:fit.append(d[f'robot_transform{j}'][:,:3,3])
            direction=np.array([.22,-.32,.17]);margin=.015
        else:
            for j in range(robot_count):fit.append(d[f'robot_transform{j}'][:,:3,3])
            direction=np.array([1.2,-1.6,.9]);margin=.10
        fit=np.concatenate(fit);target=(fit.min(0)+fit.max(0))/2
        direction/=np.sqrt(np.sum(direction*direction))
        camera.location=target+direction*1.5
        camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
        right=np.cross([0,0,1],direction);right/=np.sqrt(np.sum(right*right));up=np.cross(direction,right)
        delta=fit-target
        # Blender ortho_scale is horizontal image width.
        camera.data.ortho_scale=2.12*max(abs(np.sum(delta*right,axis=1)).max()+margin,(abs(np.sum(delta*up,axis=1)).max()+margin)*1100/780)
    if a.camera_from:
        camera_data=json.loads(a.camera_from.read_text())['camera']
        camera.location=camera_data['location']
        camera.rotation_euler=camera_data['rotation_euler']
        camera.data.ortho_scale=camera_data['ortho_scale']
    meta['camera']=dict(location=list(camera.location),rotation_euler=list(camera.rotation_euler),
                        ortho_scale=camera.data.ortho_scale,type=camera.data.type)
    scene.camera=camera
    scene.render.resolution_x=1100;scene.render.resolution_y=780;scene.render.resolution_percentage=100
    scene.view_settings.view_transform='Filmic';scene.view_settings.look='None'
    scene.view_settings.exposure=.5;scene.view_settings.gamma=1
    scene.render.image_settings.file_format='PNG';scene.render.filepath=str(a.output.resolve())

    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.video:
        if a.output.suffix.lower()!='.mp4':raise ValueError('Video output must end in .mp4')
        if not animated:raise ValueError('Video requires an evaluation scene export')
        if not np.isfinite(a.speed) or a.speed<=0:raise ValueError('Playback speed must be positive')
        times=d['sim_time'];fps=a.fps
        if fps<=0:raise ValueError('FPS must be positive')
        target_times=np.arange(0,times[-1],a.speed/fps)
        indices=np.searchsorted(times,target_times).clip(0,len(times)-1)
        indices=np.r_[indices,np.full(fps,len(times)-1)]
        selected=sorted(set([0,len(indices)//2,len(indices)-1]))
        unique=sorted(set(int(indices[j]) for j in (selected if a.preview else range(len(indices)))))
        frame_dir=a.output.parent/(a.output.stem+'_frames');frame_dir.mkdir(exist_ok=True)
        for i in unique:
            if variable:
                old=obj.data
                me=bpy.data.meshes.new('Measured material display')
                me.from_pydata(material_points(i).tolist(),[],material_faces(i).tolist());me.update()
                me.materials.append(skin);obj.data=me
                for face in me.polygons:face.use_smooth=True
                layer=me.vertex_colors.new(name='scan_color')
                colors=d.get(f'colors_{i}',np.tile(d['colors'][0],(len(me.polygons),1)))
                rgba=np.c_[colors,np.ones(len(colors))]
                layer.data.foreach_set('color',np.repeat(rgba,3,axis=0).ravel())
                bpy.data.meshes.remove(old)
            else:
                obj.data.vertices.foreach_set('co',material_points(i).ravel());obj.data.update()
                update_display_normals(i)
            if physical_blade is not None:
                physical_blade.data.vertices.foreach_set('co',d['knife'][i].ravel());physical_blade.data.update()
            for j,robot_obj in enumerate(robot_objects):
                robot_obj.matrix_world=Matrix(d[f'robot_transform{j}'][i].tolist())
            scene.render.filepath=str((frame_dir/f'{i:04d}.png').resolve())
            bpy.ops.render.render(write_still=True)
        if not a.preview:
            # Encode repeats without rendering duplicate frames.
            listing=frame_dir/'frames.txt'
            listing.write_text(''.join(f"file '{int(i):04d}.png'\nduration {1/fps:.8f}\n" for i in indices)+f"file '{int(indices[-1]):04d}.png'\n")
            subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),
              '-r',str(fps),'-frames:v',str(len(indices)),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],check=True)
        meta.update(video=True,preview_only=a.preview,frames=len(indices),fps=fps,speed=a.speed,
                    frame_directory=str(frame_dir),style='Shared Nov14 studio: scan appearance, wood board, soft lighting, one metal blade')
    else:
        bpy.ops.wm.save_as_mainfile(filepath=str(a.output.with_suffix('.blend').resolve()))
        bpy.ops.render.render(write_still=True)
    meta.update(view=a.view,show_contact_blade=a.show_contact_blade,blender_version=bpy.app.version_string,samples=a.samples)
    sidecar=a.output.with_suffix('.preview.json' if a.video and a.preview else '.json')
    sidecar.write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':
    a=arguments()
    export(a) if a.export else render(a)
