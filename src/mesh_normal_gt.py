"""Material-free triangle and area-weighted smooth geometry normal definitions."""
import numpy as np


def geometry_vertex_normals(vertices,triangles):
    vertices=np.asarray(vertices,np.float64);triangles=np.asarray(triangles,np.int64)
    t=vertices[triangles];area=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    normals=np.zeros_like(vertices)
    for corner in range(3):np.add.at(normals,triangles[:,corner],area)
    length=np.linalg.norm(normals,axis=-1)
    return normals/np.maximum(length[:,None],1e-15)


def interpolated_geometry_normals(vertices,triangles,vertex_normals,triangle_ids,hit_points):
    tri=np.asarray(triangles)[triangle_ids];points=np.asarray(vertices)[tri]
    a=points[:,1]-points[:,0];b=points[:,2]-points[:,0];c=np.asarray(hit_points)-points[:,0]
    aa=np.einsum('ij,ij->i',a,a);bb=np.einsum('ij,ij->i',b,b);ab=np.einsum('ij,ij->i',a,b)
    ca=np.einsum('ij,ij->i',c,a);cb=np.einsum('ij,ij->i',c,b);den=aa*bb-ab*ab
    if (den<=1e-25).any():raise ValueError('Degenerate ray-hit triangle')
    u=(ca*bb-cb*ab)/den;v=(cb*aa-ca*ab)/den
    weights=np.stack([1-u-v,u,v],-1)
    if not np.isfinite(weights).all() or (weights<-.002).any() or (weights>1.002).any():
        raise ValueError('Ray hit outside indexed triangle')
    # Tolerate only bounded floating-point ray/BVH error, then renormalize weights.
    weights=np.clip(weights,0,1);weights/=weights.sum(-1,keepdims=True)
    normals=(np.asarray(vertex_normals)[tri]*weights[...,None]).sum(1)
    length=np.linalg.norm(normals,axis=-1)
    if not np.isfinite(normals).all() or (length<1e-10).any():raise ValueError('Undefined interpolated geometry normal')
    return normals/length[:,None]


def gt_angular_gap(face,smooth,mask):
    a=np.asarray(face)[mask].astype(np.float64);b=np.asarray(smooth)[mask].astype(np.float64)
    a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
    return np.degrees(np.arccos(np.clip((a*b).sum(-1),-1,1)))


def dual_gt_mae(prediction,face,smooth,mask):
    prediction=np.asarray(prediction)[mask].astype(np.float64)
    length=np.linalg.norm(prediction,axis=-1)
    if not len(length) or not np.isfinite(prediction).all() or (length<1e-8).any():raise ValueError('Invalid predicted directions or empty mask')
    prediction/=length[:,None]
    result={'valid_pixels':int(mask.sum())}
    for key,reference in [('main_smooth_mae_deg',smooth),('secondary_face_mae_deg',face)]:
        reference=np.asarray(reference)[mask].astype(np.float64);reference/=np.linalg.norm(reference,axis=-1,keepdims=True)
        result[key]=float(np.degrees(np.arccos(np.clip((prediction*reference).sum(-1),-1,1))).mean())
    result['gt_definition_gap_mae_deg']=float(gt_angular_gap(face,smooth,mask).mean())
    return result
