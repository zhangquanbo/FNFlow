"""Explicit camera, normal-orientation and resize conventions for general data."""
import numpy as np


def opengl_projection_to_pixel_matrix(projection,width,height):
    """Exact central projection, including Hypersim's tilt-shift cameras."""
    p=np.asarray(projection,dtype=np.float64)
    if p.shape!=(4,4) or not np.allclose(p[[0,1,3],3],0,atol=1e-8):
        raise ValueError('Projection has a noncentral translation')
    ndc_to_pixel=np.array([[width/2,0,width/2-.5],[0,-height/2,height/2-.5],[0,0,1.]])
    basis=np.diag([1.,-1.,-1.])
    matrix=ndc_to_pixel@p[[0,1,3],:3]@basis
    if abs(np.linalg.det(matrix))<1e-12:
        raise ValueError('Singular camera projection')
    return matrix


def opengl_projection_to_intrinsics(projection,width,height):
    p=np.asarray(projection,dtype=np.float64)
    if p.shape!=(4,4) or not np.allclose(p[3],[0,0,-1,0],atol=1e-6) or not np.allclose(p[:2,3],0):
        raise ValueError('Unsupported perspective projection')
    return np.array([[width/2*p[0,0],-width/2*p[0,1],width/2*(1-p[0,2])-.5],
                     [-height/2*p[1,0],height/2*p[1,1],height/2*(1+p[1,2])-.5],
                     [0,0,1]],dtype=np.float64)


def resize_intrinsics_pixel_centers(k,input_wh,output_wh):
    sx,sy=np.asarray(output_wh,dtype=float)/np.asarray(input_wh,dtype=float)
    affine=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]])
    return affine@np.asarray(k,dtype=float)


def hypersim_normals_for_training(normal,output_wh):
    """Geometry normals: GL basis -> CV basis, then fixed orientation inversion.

    Canonical basis is x-right/y-down/z-forward, with ray-positive normal
    orientation to match the DSINE-packaged GeoNet/FrameNet convention.
    Basis rotation diag(1,-1,-1) and orientation inversion (-1) are distinct.
    Nearest pixel centers avoid mixing two surfaces or invalid normals.
    """
    normal=np.asarray(normal,dtype=np.float32)
    if normal.ndim!=3 or normal.shape[-1]!=3:
        raise ValueError('Expected HxWx3 normals')
    out_w,out_h=output_wh
    h,w=normal.shape[:2]
    xs=np.clip(np.floor((np.arange(out_w)+.5)*w/out_w).astype(int),0,w-1)
    ys=np.clip(np.floor((np.arange(out_h)+.5)*h/out_h).astype(int),0,h-1)
    selected=normal[ys[:,None],xs[None,:]]
    finite=np.isfinite(selected).all(axis=-1)
    length=np.linalg.norm(np.nan_to_num(selected),axis=-1)
    valid=finite&(length>1e-8)
    clean=np.where(valid[...,None],selected,0)
    converted=clean*np.array([-1,1,1],dtype=np.float32)
    converted/=np.maximum(length[...,None],1e-8)
    return np.where(valid[...,None],converted,0).astype(np.float32),valid
