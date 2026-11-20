"""Build and verify a mesh-derived scalar SDF; does not replace the knife by a box."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import warp as wp

@wp.kernel
def sample(mesh:wp.uint64, origin:wp.vec3, spacing:wp.vec3, ny:int,nz:int,out:wp.array(dtype=float)):
    i=wp.tid(); iz=i%nz; iy=(i//nz)%ny; ix=i//(ny*nz)
    p=origin+wp.cw_mul(spacing,wp.vec3(float(ix),float(iy),float(iz)))
    hit=wp.mesh_query_point_sign_winding_number(mesh,p,10.)
    near=wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
    out[i]=hit.sign*wp.length(p-near)

@wp.kernel
def query(mesh:wp.uint64, points:wp.array(dtype=wp.vec3),out:wp.array(dtype=float)):
    i=wp.tid(); p=points[i]
    hit=wp.mesh_query_point_sign_winding_number(mesh,p,10.)
    near=wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
    out[i]=hit.sign*wp.length(p-near)

def interpolate(grid, points):
    h=grid[:9]; shape=h[:3].astype(int); origin=h[3:6]; dx=h[6:9]
    q=np.clip((points-origin)/dx,0,shape-1-1e-5)
    i=np.floor(q).astype(int);f=q-i;values=grid[9:].reshape(shape);r=np.zeros(len(points))
    for x in (0,1):
      for y in (0,1):
       for z in (0,1):
        w=(f[:,0] if x else 1-f[:,0])*(f[:,1] if y else 1-f[:,1])*(f[:,2] if z else 1-f[:,2])
        r+=w*values[i[:,0]+x,i[:,1]+y,i[:,2]+z]
    return r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--platform',type=Path,required=True);a=p.parse_args()
 wp.init();meta=json.loads((a.platform/'cutting_geometry.json').read_text())
 d=np.load(meta['geometry']);v=d['vertices_disect'].astype(np.float32);f=d['faces'].astype(np.int32)
 mesh=wp.Mesh(points=wp.array(v,dtype=wp.vec3,device='cuda:0'),indices=wp.array(f.reshape(-1),dtype=int,device='cuda:0'),support_winding_number=True)
 spacing=np.array([.0001,.0002,.0005]);origin=v.min(0)-.004
 shape=np.ceil((v.max(0)+.004-origin)/spacing).astype(int)+1
 out=wp.empty(int(np.prod(shape)),dtype=float,device='cuda:0')
 wp.launch(sample,out.size,[mesh.id,wp.vec3(*origin),wp.vec3(*spacing),int(shape[1]),int(shape[2]),out],device='cuda:0')
 grid=np.r_[shape,origin,spacing,out.numpy()].astype(np.float32)
 path=a.platform/'knife_sdf.npy';np.save(path,grid)
 rng=np.random.default_rng(42);ids=rng.integers(0,len(f),10000)
 tri=v[f[ids]];uv=rng.dirichlet([1,1,1],len(ids));pts=(tri*uv[:,:,None]).sum(1)
 pts+=rng.normal(0,.0002,pts.shape)
 truth=wp.empty(len(pts),dtype=float,device='cuda:0')
 wp.launch(query,len(pts),[mesh.id,wp.array(pts,dtype=wp.vec3,device='cuda:0'),truth],device='cuda:0')
 errors=np.abs(interpolate(grid,pts)-truth.numpy())
 report=dict(shape=shape.tolist(),spacing_m=spacing.tolist(),samples=len(pts),
              mean_abs_error_m=float(errors.mean()),max_abs_error_m=float(errors.max()),
              p99_abs_error_m=float(np.quantile(errors,.99)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 if not np.isfinite(grid).all() or report['max_abs_error_m']>.0003:
    raise ValueError('SDF validation failed: '+str(report))
 meta.update(sdf=str(path.resolve()),sdf_validation=report)
 (a.platform/'cutting_geometry.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
