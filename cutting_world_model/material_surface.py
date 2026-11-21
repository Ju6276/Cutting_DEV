"""Diagnostic bounds of the represented FEM surface, excluding unreferenced ghost nodes.

DiSECt pre-cut tetrahedra include auxiliary nodes outside the represented material.
Do not interpret the extrema of all particle coordinates as food penetration.
Surface triangles index particle nodes followed by interpolated cut-edge vertices.
This module changes no physics, masks, loss, task or rendering geometry.
"""
import torch

@torch.no_grad()
def surface_plane_bounds(particles, triangles, cut_triangles, cut_edges, cut_coords,
                         board_origin, board_normal):
    """Return raw and surface bounds for (..., N, 3) points in one coordinate frame.

    Triangle and edge indices must be local to each world, not packed offsets.
    Counts/fractions describe unique referenced surface vertices, not material
    volume fractions. Interpolated cut vertices are included even if their parent
    nodes are disabled for direct contact.
    """
    p=torch.as_tensor(particles)
    if not p.is_floating_point() or p.ndim<2 or p.shape[-1]!=3 or p.shape[-2]==0:
        raise ValueError("Expected nonempty floating-point particles (..., N, 3)")
    if not torch.isfinite(p).all():
        raise ValueError("Nonfinite material coordinates")
    normal=torch.as_tensor(board_normal,device=p.device,dtype=p.dtype)
    origin=torch.as_tensor(board_origin,device=p.device,dtype=p.dtype)
    if normal.shape!=(3,) or origin.shape!=(3,) or not torch.isfinite(origin).all() or not torch.isfinite(normal).all():
        raise ValueError("Expected finite plane origin and normal")
    if not torch.isclose(torch.linalg.vector_norm(normal),normal.new_tensor(1.),atol=1e-6,rtol=0):
        raise ValueError("Board normal must be unit length")
    def indices(value,width,name):
        x=torch.as_tensor(value,device=p.device)
        if x.numel()==0:return torch.empty((0,width),dtype=torch.long,device=p.device)
        if x.ndim!=2 or x.shape[1]!=width or x.dtype not in (torch.int32,torch.int64):
            raise ValueError("Invalid "+name+" index array")
        return x.long()
    edges=indices(cut_edges,2,"cut edge")
    if edges.numel() and (bool((edges<0).any()) or bool((edges>=p.shape[-2]).any())):
        raise ValueError("Cut edge index is not world-local")
    t=torch.as_tensor(cut_coords,device=p.device,dtype=p.dtype).reshape(-1)
    if len(t)!=len(edges) or not torch.isfinite(t).all() or bool(((t<0)|(t>1)).any()):
        raise ValueError("Invalid interpolation coordinates")
    virtual=(1-t[:,None])*p[...,edges[:,0],:]+t[:,None]*p[...,edges[:,1],:]
    vertices=torch.cat((p,virtual),dim=-2)
    faces=torch.cat((indices(triangles,3,"surface triangle"),
                     indices(cut_triangles,3,"cut triangle")),dim=0)
    if faces.numel()==0 or bool((faces<0).any()) or bool((faces>=vertices.shape[-2]).any()):
        raise ValueError("Invalid or empty represented surface")
    selected=torch.unique(faces)
    raw=(p-origin)@normal
    surface=(vertices[...,selected,:]-origin)@normal
    return dict(raw_min_height_m=raw.amin(-1),raw_max_height_m=raw.amax(-1),
                surface_min_height_m=surface.amin(-1),surface_max_height_m=surface.amax(-1),
                surface_below_plane_fraction=(surface<0).to(p.dtype).mean(-1),
                surface_vertex_count=len(selected),raw_node_count=p.shape[-2])
