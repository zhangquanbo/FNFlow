"""Fixed common regions for comparing three declared geometry-label definitions."""
import numpy as np
from src.mesh_normal_gt import gt_angular_gap


def angular_band(normal,valid,angle_degrees=20.,radius=2,include_silhouette=False):
    edge=np.zeros(valid.shape,bool);threshold=np.cos(np.deg2rad(angle_degrees))
    for a,b in [((slice(None,-1),slice(None)),(slice(1,None),slice(None))),
                ((slice(None),slice(None,-1)),(slice(None),slice(1,None)))]:
        hit=valid[a]&valid[b]&((normal[a]*normal[b]).sum(-1)<threshold)
        edge[a]|=hit;edge[b]|=hit
    if include_silhouette:
        padded=np.pad(valid,1)
        edge|=valid&(~padded[:-2,1:-1]|~padded[2:,1:-1]|~padded[1:-1,:-2]|~padded[1:-1,2:])
    padded=np.pad(edge,radius);band=np.zeros_like(edge);h,w=valid.shape
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):band|=padded[dy:dy+h,dx:dx+w]
    return band&valid


def gt_boundary_report(face,smooth,crease45,valid,angle_degrees=20.,radius=2):
    bands={'primary_angular':angular_band(smooth,valid,angle_degrees,radius),
        'crease45_angular_plus_silhouette':angular_band(crease45,valid,angle_degrees,radius,True)}
    pairs={'face_vs_smooth':(face,smooth),'face_vs_crease45':(face,crease45),'smooth_vs_crease45':(smooth,crease45)}
    maps={};result={}
    for name,(a,b) in pairs.items():
        values=np.zeros(valid.shape,np.float32);values[valid]=gt_angular_gap(a,b,valid);maps[name]=values
    for name,band in bands.items():
        row={'valid_pixels':int(valid.sum()),'inside_pixels':int(band.sum()),'outside_pixels':int((valid&~band).sum()),
            'inside_fraction':float(band.sum()/valid.sum()),'comparisons':{}}
        for pair,values in maps.items():
            regions={}
            for region,support in [('inside',band),('outside',valid&~band)]:
                selected=values[support].astype(np.float64)
                regions[region]={'pixels':len(selected),'angle_sum_deg':float(selected.sum()),
                    'mean_deg':float(selected.mean()) if len(selected) else None}
            row['comparisons'][pair]=regions
        result[name]=row
    return result,bands,maps


def aggregate_boundary_reports(records):
    result={}
    for band in ['primary_angular','crease45_angular_plus_silhouette']:
        groups={}
        for r in records:groups.setdefault(r['scene'],[]).append(r['boundary_gt_report'][band])
        rows=[r['boundary_gt_report'][band] for r in records]
        dest={'inside_pixels':sum(r['inside_pixels'] for r in rows),'outside_pixels':sum(r['outside_pixels'] for r in rows),
            'valid_pixels':sum(r['valid_pixels'] for r in rows),'comparisons':{}}
        dest['inside_fraction']=dest['inside_pixels']/dest['valid_pixels']
        for pair in ['face_vs_smooth','face_vs_crease45','smooth_vs_crease45']:
            regions={}
            for region in ['inside','outside']:
                values=[r['comparisons'][pair][region] for r in rows];pixels=sum(r['pixels'] for r in values)
                macro=[r['mean_deg'] for r in values if r['mean_deg'] is not None]
                asset=[]
                for group in groups.values():
                    nonempty=[r['comparisons'][pair][region]['mean_deg'] for r in group if r['comparisons'][pair][region]['pixels']]
                    if nonempty:asset.append(np.mean(nonempty))
                regions[region]={'pixels':pixels,'images_with_region':len(macro),'images_without_region':len(rows)-len(macro),
                    'assets_with_region':len(asset),'pixel_pooled_mean_deg':sum(r['angle_sum_deg'] for r in values)/pixels if pixels else None,
                    'image_macro_mean_deg':float(np.mean(macro)) if macro else None,'asset_macro_mean_deg':float(np.mean(asset)) if asset else None}
            dest['comparisons'][pair]=regions
        result[band]=dest
    return result


def boundary_prediction_sensitivity(prediction,face,smooth,crease45,valid,bands):
    result={};prediction=np.asarray(prediction,np.float64)
    if not np.isfinite(prediction[valid]).all() or (np.linalg.norm(prediction[valid],axis=-1)<1e-8).any():raise ValueError('Invalid predicted normal')
    for band_name,band in bands.items():
        result[band_name]={}
        for region,support in [('inside',band),('outside',valid&~band)]:
            result[band_name][region]={'pixels':int(support.sum()),**{
                name:float(gt_angular_gap(prediction,label,support).mean()) if support.any() else None
                for name,label in [('face_mae_deg',face),('main_smooth_mae_deg',smooth),('diagnostic_crease45_mae_deg',crease45)]}}
    return result
