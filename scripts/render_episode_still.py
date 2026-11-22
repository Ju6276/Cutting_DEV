
"""Shared studio renderer for recorded stills and videos. Export with VTK, render with Blender.

No simulated coordinates are smoothed or replaced. Scan colors come from the
optional appearance map. Board/floor visuals and illumination are display assets.
"""
import argparse, json, sys, subprocess
from pathlib import Path
# Blender does not add a -P script directory to sys.path.
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np

def arguments():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    p.add_argument('--recording',type=Path)
    p.add_argument('--source-root',type=Path,action='append',default=[])
    p.set_defaults(show_contact_blade=False,highlight_cut_faces=False)
    p.add_argument('--sensor-rig',type=Path)
    p.add_argument('--sensor-camera',type=Path,help='Calibrated perspective sensor view; same studio scene/materials')
    p.add_argument('--view',choices=['close','wide'],default='close')
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
    raise ValueError("The legacy per-recording exporter is retired; use scripts/render_scene.py")

def exterior_vertex_normals(x, faces):
    """Smooth only exterior faces, sharing normals at coincident split nodes."""
    v=x[faces];fn=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
    normals=np.zeros_like(x)
    for k in range(3):np.add.at(normals,faces[:,k],fn)
    _,inverse=np.unique(np.round(x/1e-8).astype(np.int64),axis=0,return_inverse=True)
    joined=np.zeros((inverse.max()+1,3));np.add.at(joined,inverse,normals)
    normals=joined[inverse]
    return normals/np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)

def render(a):
    import hashlib
    import bpy
    from mathutils import Vector, Matrix
    d=dict(np.load(a.scene))
    meta=json.loads(str(d['metadata']))
    if meta.get('format')=='studio_evaluation_v1':
        contract=Path(__file__).resolve().parents[1]/'configs/render/studio_v1.json'
        if meta.get('render_profile_sha256')!=hashlib.sha256(contract.read_bytes()).hexdigest():
            raise ValueError("Outdated scene export: regenerate through render_scene.py")
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
        normals=exterior_vertex_normals(x,faces)
        fine=(normals[d['normal_indices']]*d['normal_weights'][...,None]).sum(1)
        fine/=np.maximum(np.sqrt(np.sum(fine*fine,axis=1,keepdims=True)),1e-20)
        corner_normals=fine[d['faces']]
        inside=d.get('normal_interior_triangles')
        if inside is not None and np.any(inside):
            triangles=material_points(i)[material_faces(i)[inside]]
            n=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-20)
            corner_normals[inside]=n[:,None,:]
        obj.data.normals_split_custom_set(corner_normals.reshape(-1,3).tolist())
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
    profile_path=Path(__file__).resolve().parents[1]/'configs/render/studio_v1.json'
    profile=json.loads(profile_path.read_text())
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
         [*profile['board_size_m'],profile['board_thickness_m']],boardmat)
    if a.sensor_rig:
        rig=json.loads(a.sensor_rig.read_text())
        h=dict(np.load(rig['housing_asset'],allow_pickle=False))
        camera_obj=mesh('Provisional Mech-Eye NANO official CAD',h['vertices'],h['faces'],
                        mat('Mech-Eye black housing',(.018,.02,.022),.35,.36),True)
        camera_obj.matrix_world=Matrix(rig['world_from_camera'])
        appearance=rig.get('housing_appearance')
        if appearance:
            # Official STEP is monochrome. Assign only documented visible regions.
            rotation=np.asarray(appearance['cad_to_optical_rotation'],float)
            origin=np.asarray(appearance['cad_origin_mm'],float)
            cad=np.einsum('ij,jk->ik',h['vertices']*1000.,rotation)+origin
            silver=mat('Nano silver connectors',(.55,.57,.60),.85,.23)
            camera_obj.data.materials.append(silver)
            centres=cad[h['faces']].mean(axis=1)
            selected=centres[:,0] < appearance['silver_connector_max_cad_x_mm']-1e-5
            for face,index in zip(camera_obj.data.polygons,selected):face.material_index=int(index)
            label=appearance['warning_label']
            x0,x1,y0,y1,z=label['cad_rect_mm']
            corners=np.array([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]])
            optical=np.einsum('ij,kj->ik',corners-origin,rotation)*.001
            labelmat=mat('Official Nano warning label',(1,1,1),rough=.55)
            photo=bpy.data.images.load(label['image'],check_existing=True)
            texture=labelmat.node_tree.nodes.new('ShaderNodeTexImage');texture.image=photo
            labelmat.node_tree.links.new(texture.outputs['Color'],labelmat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
            lm=mesh('Official Nano label on CAD label face',optical,np.array([[0,1,2],[0,2,3]]),labelmat)
            lm.matrix_world=Matrix(rig['world_from_camera'])
            px0,py0,px1,py1=label['image_crop_px'];iw,ih=photo.size[:]
            uv=[(px0/iw,1-py1/ih),(px1/iw,1-py1/ih),(px1/iw,1-py0/ih),(px0/iw,1-py0/ih)]
            layer=lm.data.uv_layers.new(name='Official label photo')
            for polygon in lm.data.polygons:
                for li in polygon.loop_indices:
                    layer.data[li].uv=uv[lm.data.loops[li].vertex_index]
        meta['sensor_rig']=rig
    # One scene for both views: only the camera changes.
    cube('Floor',[bc[0],bc[1],-.015],[4,4,.03],mat('Floor',(.32,.35,.37),rough=.8))
    cube('Display board support',[bc[0],bc[1],(top-profile['board_thickness_m'])/2],
         [profile['board_size_m'][0]*.85,profile['board_size_m'][1]*.85,max(.001,top-profile['board_thickness_m'])],
         mat('Support',(.42,.45,.48),rough=.65))
    profile_path=Path(__file__).resolve().parents[1]/'configs/render/studio_v1.json'
    profile=json.loads(profile_path.read_text())
    meta['render_contract']=profile['revision']
    meta['render_profile_sha256']=hashlib.sha256(profile_path.read_bytes()).hexdigest()
    center=bc+np.array([0,0,.08])
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
    camera=bpy.context.object
    c=profile['camera'][a.view]
    target=bc+np.array(c['target_offset_m'])
    camera.location=target+np.array(c['offset_m'])
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';camera.data.ortho_scale=c['ortho_scale_m']
    if a.camera_from:
        raise ValueError('Per-video camera overrides are retired; edit the versioned render contract')
    meta['camera']=dict(location=list(camera.location),rotation_euler=list(camera.rotation_euler),
                        ortho_scale=camera.data.ortho_scale,type=camera.data.type)
    scene.camera=camera
    scene.render.resolution_x=1100;scene.render.resolution_y=780;scene.render.resolution_percentage=100
    if a.sensor_camera:
        sensor=json.loads(a.sensor_camera.read_text())
        from blender_camera import configure_sensor_camera
        configure_sensor_camera(scene,camera,sensor)
        meta['camera']=dict(type='PERSP',intrinsics=sensor,scope='Provisional sensor view, canonical studio appearance')
        from bpy_extras.object_utils import world_to_camera_view
        bpy.context.view_layer.update()
        optical=np.array([[0.,0.,.4],[.02,.03,.35],[-.04,-.02,.43]])
        base_from_camera=np.array(sensor['base_from_camera'])
        # Small fixed 3D transforms avoid Blender 2.82's crashing system BLAS kernel.
        world=np.einsum('ij,kj->ik',optical,base_from_camera[:3,:3])+base_from_camera[:3,3]
        projected=np.array([world_to_camera_view(scene,camera,Vector(x))[:2] for x in world])
        pixels=np.c_[projected[:,0]*sensor['width']-.5,(1-projected[:,1])*sensor['height']-.5]
        expected=np.c_[sensor['fx']*optical[:,0]/optical[:,2]+sensor['cx'],
                       sensor['fy']*optical[:,1]/optical[:,2]+sensor['cy']]
        pixel_error=float(np.abs(pixels-expected).max())
        if pixel_error>.002:raise ValueError('RGB/depth projection mismatch: '+str(pixel_error))
        meta['projection_pixel_max_error']=pixel_error


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
                    frame_directory=str(frame_dir),style='Pinned food_studio_v1: registered appearance, wood board, one recorded tool assembly')
    else:
        bpy.ops.wm.save_as_mainfile(filepath=str(a.output.with_suffix('.blend').resolve()))
        bpy.ops.render.render(write_still=True)
    meta.update(view=a.view,show_contact_blade=a.show_contact_blade,blender_version=bpy.app.version_string,samples=a.samples)
    sidecar=a.output.with_suffix('.preview.json' if a.video and a.preview else '.json')
    sidecar.write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':
    a=arguments()
    export(a) if a.export else render(a)
