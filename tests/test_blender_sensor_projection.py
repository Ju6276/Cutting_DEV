"""Exercise the actual Blender camera, including off-centre measured intrinsics."""
import json
from pathlib import Path
import shutil
import subprocess
import pytest

def test_blender_camera_preserves_offcentre_intrinsics(tmp_path):
    blender=shutil.which("blender")
    if blender is None:
        pytest.skip("Blender renderer executable not installed")
    scripts=Path(__file__).resolve().parents[1]/"scripts"
    output=tmp_path/"projection.json"
    code=f"""
import bpy,json,sys,runpy
runpy.run_path({str(scripts/"render_episode_still.py")!r},run_name="renderer_import_check")
from blender_camera import configure_sensor_camera
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
s=bpy.context.scene
bpy.ops.object.camera_add()
c=bpy.context.object
s.camera=c
errors=[]
for fx,fy,cx,cy in [(1728.890384,1728.884610,642.625068,512.295689),(1745.,2048.,630.,510.),(1745.,2048.,639.5,511.5)]:
 config=dict(width=1280,height=1024,fx=fx,fy=fy,cx=cx,cy=cy,
  base_from_camera=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
 configure_sensor_camera(s,c,config)
 bpy.context.view_layer.update()
 for x,y,z in [(0,0,.4),(.02,.03,.35),(-.04,-.02,.43)]:
  v=world_to_camera_view(s,c,Vector((x,y,z)))
  errors.append(max(abs(v.x*1280-.5-(fx*x/z+cx)),abs((1-v.y)*1024-.5-(fy*y/z+cy))))
open({str(output)!r},"w").write(json.dumps(errors))
"""
    result=subprocess.run([blender,"-b","-t","1","--python-exit-code","1","--python-expr",code],
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert max(json.loads(output.read_text()))<.002
