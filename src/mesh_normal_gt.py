"""Material-free triangle and area-weighted smooth geometry normal definitions."""
import numpy as np


def geometry_vertex_normals(vertices,triangles):
    vertices=np.asarray(vertices,np.float64);triangles=np.asarray(triangles,np.int64)
    t=vertices[triangles];area=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    normals=np.zeros_like(vertices)
    for corner in range(3):np.add.at(normals,triangles[:,corner],area)
    length=np.linalg.norm(normals,axis=-1)
    return normals/np.maximum(length[:,None],1e-15)


def crease_corner_normals(vertices,triangles,threshold_degrees=45.):
    """Area-weight edge-connected corner fans; split sharp and nonmanifold edges."""
    vertices=np.asarray(vertices,np.float64);triangles=np.asarray(triangles,np.int64)
    points=vertices[triangles];area=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
    length=np.linalg.norm(area,axis=-1);unit=area/np.maximum(length[:,None],1e-15)
    count=len(triangles);face=np.tile(np.arange(count),3)
    edge=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
    first=np.concatenate([3*np.arange(count)+i for i in [0,1,2]])
    second=np.concatenate([3*np.arange(count)+i for i in [1,2,0]])
    forward=edge[:,0]<=edge[:,1];lo=np.minimum(edge[:,0],edge[:,1]);hi=np.maximum(edge[:,0],edge[:,1])
    corner_lo=np.where(forward,first,second);corner_hi=np.where(forward,second,first)
    order=np.lexsort((hi,lo));lo=lo[order];hi=hi[order]
    starts=np.r_[0,1+np.flatnonzero((lo[1:]!=lo[:-1])|(hi[1:]!=hi[:-1]))]
    ends=np.r_[starts[1:],len(order)];starts=starts[(ends-starts)==2]
    a=order[starts];b=order[starts+1]
    allowed=(length[face[a]]>1e-15)&(length[face[b]]>1e-15)&((unit[face[a]]*unit[face[b]]).sum(-1)>=np.cos(np.deg2rad(threshold_degrees)))
    a=a[allowed];b=b[allowed];parent=list(range(count*3))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for left,right in zip(np.r_[corner_lo[a],corner_hi[a]].tolist(),np.r_[corner_lo[b],corner_hi[b]].tolist()):
        x,y=find(left),find(right)
        if x!=y:parent[y]=x
    roots=np.fromiter((find(i) for i in range(count*3)),dtype=np.int64,count=count*3)
    total=np.zeros((count*3,3),np.float64);np.add.at(total,roots,np.repeat(area,3,axis=0))
    normal=total[roots];size=np.linalg.norm(normal,axis=-1)
    return (normal/np.maximum(size[:,None],1e-15)).reshape(count,3,3)


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
    normals=np.asarray(vertex_normals)
    corners=normals[triangle_ids] if normals.ndim==3 else normals[tri]
    normals=(corners*weights[...,None]).sum(1)
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
