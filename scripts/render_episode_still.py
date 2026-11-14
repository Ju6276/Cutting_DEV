
"""Offline still renderer for recorded geometry. Export with VTK, render with Blender.

No simulated coordinates are smoothed or replaced. Scan colors come from the
optional appearance map. Board/floor visuals and illumination are display assets.
"""
import argparse, json, sys
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
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path)
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
    out=dict(points=pts,faces=faces,colors=colors,knife=d['knife'][i],knife_tri=d['knife_tri'],
             ground=d['ground'],center=(d['particles'][i].min(0)+d['particles'][i].max(0))/2)
    robot,sources=r.visual_actors(d,task,a.source_root)
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
    d=np.load(a.scene)
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
    obj=mesh('Recorded material',d['points'],d['faces'],skin,True)
    # Blender 2.82 MeshLoopColor stores display RGB; converting here again
    # would darken source scans twice. Use the legacy layer's native encoding.
    c=d['colors'];rgba=np.c_[c,np.ones(len(c))]
    layer=obj.data.vertex_colors.new(name='scan_color')
    layer.data.foreach_set('color',np.repeat(rgba,3,axis=0).ravel())
    if 'normals' in d:
        obj.data.normals_split_custom_set(d['normals'].tolist())
        if hasattr(obj.data,'use_auto_smooth'):obj.data.use_auto_smooth=True
    tool_material=mat('Mounted knife and adapter steel',(.44,.47,.50),.7,.3)
    robot_count=json.loads(str(d['metadata']))['robot_meshes']
    for j in range(robot_count):
        color=d[f'robot_c{j}'];metal=.55 if np.mean(color)>.4 else .1
        mesh(f'UR5e visual {j}',d[f'robot_v{j}'],d[f'robot_f{j}'],
             tool_material if bool(d.get(f'robot_tool{j}',False)) else mat(f'robot {j}',color,metal,.3),True)
    if a.show_contact_blade:mesh('DiSECt contact blade',d['knife'],d['knife_tri'],mat('Contact blade blue steel',(.12,.29,.40),.7,.24))
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
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=96
    scene.cycles.use_denoising=True
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
    scene.camera=camera
    scene.render.resolution_x=1100;scene.render.resolution_y=780;scene.render.resolution_percentage=100
    scene.view_settings.view_transform='Filmic';scene.view_settings.look='None'
    scene.view_settings.exposure=.5;scene.view_settings.gamma=1
    scene.render.image_settings.file_format='PNG';scene.render.filepath=str(a.output.resolve())
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output.with_suffix('.blend').resolve()))
    bpy.ops.render.render(write_still=True)
    metadata=json.loads(str(d['metadata']))
    metadata.update(view=a.view,show_contact_blade=a.show_contact_blade,blender_version=bpy.app.version_string)
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
    a=arguments()
    export(a) if a.export else render(a)
