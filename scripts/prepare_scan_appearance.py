
"""Project a real scan's texture onto recorded simulation surface triangles.

Geometry is not replaced: dense display triangles interpolate the existing FEM
surface. Rigid scan registration is approximate and reported, never used by physics.
Run with the food-mesh environment (USD, scipy, VTK, Pillow).
"""
import argparse, hashlib, itertools, json
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from scipy.spatial import cKDTree

def poly(v, f):
    p=vtk.vtkPolyData(); pts=vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(v,dtype=float),deep=True)); p.SetPoints(pts)
    c=vtk.vtkCellArray()
    c.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.c_[np.full(len(f),3),f].astype(np.int64).ravel(),deep=True))
    p.SetPolys(c); return p

def closest(locator, p):
    out=[0.,0.,0.]; cell=vtk.reference(0); sub=vtk.reference(0); dist=vtk.reference(0.)
    locator.FindClosestPoint(p,out,cell,sub,dist)
    return np.asarray(out),int(cell),float(dist)

def register(source, target, faces):
    # Try all proper PCA sign/permutation alignments; ICP only changes rigid pose.
    cs=source.mean(0); ct=target.mean(0)
    _,vs=np.linalg.eigh(np.cov((source-cs).T))
    _,vt=np.linalg.eigh(np.cov((target-ct).T))
    sample=source[np.linspace(0,len(source)-1,min(3000,len(source))).astype(int)]
    tree=cKDTree(target); candidates=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1,1],repeat=3):
            r=vt[:,perm]@np.diag(signs)@vs.T
            if np.linalg.det(r)<0:continue
            t=ct-r@cs
            score=np.mean(tree.query(sample@r.T+t)[0]**2)
            candidates.append((score,r,t))
    dest=poly(target,faces); loc=vtk.vtkStaticCellLocator();loc.SetDataSet(dest);loc.BuildLocator()
    best=None
    for _,r,t in sorted(candidates,key=lambda x:x[0])[:6]:
        p=poly(source@r.T+t, np.empty((0,3),dtype=int))
        icp=vtk.vtkIterativeClosestPointTransform();icp.SetSource(p);icp.SetTarget(dest)
        icp.GetLandmarkTransform().SetModeToRigidBody();icp.SetMaximumNumberOfIterations(60)
        icp.SetMaximumNumberOfLandmarks(3000);icp.Update()
        m=np.array([[icp.GetMatrix().GetElement(i,j) for j in range(4)] for i in range(4)])
        rr=m[:3,:3]@r;tt=m[:3,:3]@t+m[:3,3]
        error=np.array([closest(loc,x)[2] for x in sample@rr.T+tt])
        score=float(np.sqrt(error.mean()))
        if best is None or score<best[0]:best=(score,rr,tt)
    return best

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--recording',type=Path,required=True)
    a.add_argument('--usd',type=Path,required=True)
    a.add_argument('--texture',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--spacing-mm',type=float,default=.5)
    args=a.parse_args()
    if args.spacing_mm<=0:raise ValueError('Display spacing must be positive')
    d=np.load(args.recording)
    stage=Usd.Stage.Open(str(args.usd))
    meshes=[p for p in stage.Traverse() if p.GetTypeName()=='Mesh']
    if len(meshes)!=1:raise ValueError('Expected one source scan mesh')
    mesh=UsdGeom.Mesh(meshes[0])
    raw=np.asarray(mesh.GetPointsAttr().Get(),float)
    xf=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(meshes[0]))
    source=(np.c_[raw,np.ones(len(raw))]@xf)[:,:3]*UsdGeom.GetStageMetersPerUnit(stage)
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts==3):raise ValueError('Source needs triangular faces')
    faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    pv=UsdGeom.PrimvarsAPI(meshes[0]).GetPrimvar('st')
    uv=np.asarray(pv.ComputeFlattened(),float)
    if pv.GetInterpolation()=='vertex':uv=uv[faces]
    elif pv.GetInterpolation()=='faceVarying':uv=uv.reshape(-1,3,2)
    else:raise ValueError('Unsupported UV interpolation')
    target=d['particles'][0]; exterior=d['tri']
    # Registration uses only exterior vertices, not interior FEM particles.
    used=np.unique(exterior)
    remap=np.full(len(target),-1,dtype=int);remap[used]=np.arange(len(used))
    rms,r,t=register(source,target[used],remap[exterior])
    if rms>.003:raise ValueError(f'Scan registration RMS {rms} m exceeds 3 mm; inspect asset identity')
    aligned=source@r.T+t
    locator=vtk.vtkStaticCellLocator();locator.SetDataSet(poly(aligned,faces));locator.BuildLocator()
    tex=np.asarray(Image.open(args.texture).convert('RGB'));height,width=tex.shape[:2]
    ids=[];weights=[];rgb=[]
    for face in exterior:
        v=target[face]
        n=max(1,int(np.ceil(max(np.linalg.norm(v[(j+1)%3]-v[j]) for j in range(3))/(args.spacing_mm*.001))))
        def b(i,j):return np.array([1-(i+j)/n,i/n,j/n])
        for i in range(n):
            for j in range(n-i):
                pieces=[[b(i,j),b(i+1,j),b(i,j+1)]]
                if i+j<n-1:pieces.append([b(i+1,j),b(i+1,j+1),b(i,j+1)])
                for piece in pieces:
                    w=np.array(piece);center=w.mean(0)@v
                    near,cell,_=closest(locator,center)
                    tri=aligned[faces[cell]]
                    ab=np.linalg.lstsq((tri[1:]-tri[0]).T,near-tri[0],rcond=None)[0]
                    coords=np.clip(np.array([1-ab.sum(),*ab]),0,1);coords/=coords.sum()
                    st=coords@uv[cell];st=np.clip(st,0,1)
                    x=st[0]*(width-1);y=(1-st[1])*(height-1)
                    ix=int(x);iy=int(y);jx=min(ix+1,width-1);jy=min(iy+1,height-1)
                    c=(1-x+ix)*(1-y+iy)*tex[iy,ix]+(x-ix)*(1-y+iy)*tex[iy,jx]+(1-x+ix)*(y-iy)*tex[jy,ix]+(x-ix)*(y-iy)*tex[jy,jx]
                    rgb.append(np.rint(c).astype(np.uint8));weights.extend(w);ids.extend([face]*3)
    metadata=dict(recording_sha256=hashlib.sha256(args.recording.read_bytes()).hexdigest(),
      usd=str(args.usd),texture=str(args.texture),texture_sha256=hashlib.sha256(args.texture.read_bytes()).hexdigest(),
      registration_rms_m=rms,rotation=r.tolist(),translation=t.tolist(),spacing_mm=args.spacing_mm,
      geometry='Barycentric subdivision of recorded FEM exterior; no geometric smoothing or scan replacement',
      appearance='Nearest-surface projection of original scan texture; approximate rigid registration')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,indices=np.asarray(ids),weights=np.asarray(weights),
                        rgb=np.asarray(rgb),metadata=np.array(json.dumps(metadata)))
    print(json.dumps(dict(**metadata,display_triangles=len(rgb))),flush=True)
if __name__=='__main__':main()
