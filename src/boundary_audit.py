"""Coverage audit independent of predictions; excluded normals remain excluded."""
import cv2
import numpy as np


def angular_band(normal,support,radius=2,foreground=None):
    edge=np.zeros(support.shape,bool)
    for axis in [0,1]:
        a=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
        b=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
        pair=support[a]&support[b]&((normal[a]*normal[b]).sum(-1)<np.cos(np.deg2rad(20)))
        if foreground is not None:pair|=foreground[a]!=foreground[b]
        edge[a]|=pair;edge[b]|=pair
    return cv2.dilate(edge.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)


def audit_mask(decoded,alpha,unit_tolerance=.05,radius=2):
    finite=np.isfinite(decoded).all(-1)
    length=np.linalg.norm(np.nan_to_num(decoded),axis=-1)
    nonzero=finite&(length>1e-8)
    strict=finite&(np.abs(length-1)<=unit_tolerance)
    directions=np.where(nonzero[...,None],decoded/np.maximum(length[...,None],1e-8),0)
    # This is the existing error band's construction BEFORE intersecting final validity.
    strict_band=angular_band(directions,strict,radius)
    output={}
    for threshold in [.01,.5,.99]:
        fg=alpha>threshold
        rejected=fg&finite&~strict
        relaxed_band=angular_band(directions,nonzero&fg,radius,fg)
        counts={'foreground_pixels':int(fg.sum()),'unit_rejected_foreground_pixels':int(rejected.sum()),
                'nonfinite_foreground_pixels':int((fg&~finite).sum()),
                'zero_foreground_pixels':int((fg&finite&~nonzero).sum()),
                'retained_foreground_pixels':int((fg&strict).sum())}
        for name,band in [('strict_gt_before_valid_intersection',strict_band),('relaxed_direction_plus_silhouette',relaxed_band)]:
            counts[name]={'unit_rejected_in_band':int((rejected&band).sum()),
                         'foreground_band_pixels':int((fg&band).sum()),
                         'retained_foreground_band_pixels':int((fg&strict&band).sum()),
                         'unit_rejected_outside_band':int((rejected&~band).sum())}
        output[str(threshold)]=counts
    return output


def add_counts(total,value):
    for key,item in value.items():
        if isinstance(item,dict):add_counts(total.setdefault(key,{}),item)
        else:total[key]=total.get(key,0)+item


def fractions(counts):
    out={}
    for threshold,row in counts.items():
        item=dict(row);denom=row['unit_rejected_foreground_pixels']
        for band in ['strict_gt_before_valid_intersection','relaxed_direction_plus_silhouette']:
            b=dict(row[band]);n=b['foreground_band_pixels']
            b['fraction_unit_rejected_in_band']=b['unit_rejected_in_band']/denom if denom else None
            b['fraction_band_foreground_retained']=b['retained_foreground_band_pixels']/n if n else None
            item[band]=b
        out[threshold]=item
    return out
