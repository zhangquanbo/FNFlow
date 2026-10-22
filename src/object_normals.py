"""Fixed Arb EXR convention and geometry-owned validity, independent of lighting."""
import numpy as np

def decode_arb_normal(native,output_wh=(256,256),unit_tolerance=.05):
    value=np.asarray(native,dtype=np.float32)
    if value.ndim!=3 or value.shape[-1]!=3:raise ValueError('Expected three native OpenCV EXR channels')
    decoded=value*2-1
    length=np.linalg.norm(decoded,axis=-1)
    valid=np.isfinite(decoded).all(-1)&(np.abs(length-1)<=unit_tolerance)
    normal=np.where(valid[...,None],decoded,0)*np.array([-1,1,1],np.float32)
    normal/=np.where(valid,length,1)[...,None]
    width,height=output_wh;h,w=length.shape
    xs=np.minimum(((np.arange(width)+.5)*w/width).astype(int),w-1)
    ys=np.minimum(((np.arange(height)+.5)*h/height).astype(int),h-1)
    return normal[ys[:,None],xs[None,:]],valid[ys[:,None],xs[None,:]]
