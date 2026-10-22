"""Use DSINE's own RGB, labels and calibration; official MAT supplies identities only."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,argparse
import numpy as np
import cv2
from PIL import Image
from src.data_contract import resize_intrinsics_pixel_centers
from src.research_io import ROOT,output_path,write_json,sha256,now
from src.spatial_protocol import center_square, crop_camera

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='data/nyu_validation_packaged_v1')
    parser.add_argument('--center-crop',action='store_true')
    args=parser.parse_args()
    source=ROOT/'outputs/P0_nyu_validation_v1/private_manifest.json'
    rows=json.loads(source.read_text())['validation']
    target=output_path(ROOT/args.output)
    if (target/'summary.json').exists():raise RuntimeError('Existing cache; select a new output')
    target.mkdir(parents=True,exist_ok=True)
    records=[]
    for r in rows:
        if r.get('training_allowed') or r['role']!='real_indoor_validation':raise RuntimeError('Partition role mismatch')
        rgb=np.asarray(Image.open(r['rgb_path']).convert('RGB'))
        encoded=np.asarray(Image.open(r['normal_path']).convert('RGB'))
        if rgb.shape!=encoded.shape:raise RuntimeError('Packaged RGB/normal shape mismatch')
        source_wh=(rgb.shape[1],rgb.shape[0]);left=top=0
        if args.center_crop:
            rgb,(left,top,side)=center_square(rgb)
            encoded,_=center_square(encoded)
        native_valid=(encoded.sum(-1)>0)
        normal=encoded.astype(np.float32)/127.5-1
        h,w=normal.shape[:2]
        xs=np.minimum(((np.arange(256)+.5)*w/256).astype(int),w-1)
        ys=np.minimum(((np.arange(256)+.5)*h/256).astype(int),h-1)
        normal=normal[ys[:,None],xs[None,:]]
        valid=native_valid[ys[:,None],xs[None,:]]&np.isfinite(normal).all(-1)
        length=np.linalg.norm(normal,axis=-1);valid&=length>1e-8
        normal=np.where(valid[...,None],normal/np.maximum(length[...,None],1e-8),0)
        k=np.load(r['K_path']).astype(np.float64)
        if k.shape!=(3,3) or not np.isfinite(k).all() or abs(np.linalg.det(k))<1e-8:raise RuntimeError('Invalid packaged K')
        k=resize_intrinsics_pixel_centers(crop_camera(k,left,top),(w,h),(256,256)).astype(np.float32)
        path=target/f"{r['official_index_1based']:04d}.npz"
        np.savez_compressed(path,rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_LINEAR),
            normal=normal.astype(np.float32),valid=valid,K=k,K_valid=np.array(True),projection_3x3=k,projection_valid=np.array(True))
        records.append({'scene':r['room'],'split':'val','path':str(path),'label_source':'depth_derived_geonet',
                        'source_rgb_sha256':sha256(r['rgb_path']),'valid_pixels':int(valid.sum())})
        records[-1].update(source_wh=list(source_wh),crop_xywh=[left,top,w,h])
    write_json(target/'private_manifest.json',{'records':records,'source_manifest_sha256':sha256(source)})
    summary={'status':'completed','created_at':now(),'images':len(records),'rooms':len({r['scene'] for r in records}),
        'training_allowed':False,'protocol':'DSINE packaged RGB/normal/K paired; official MAT used for scene identity only',
        'normal_decode':'DSINE RGB * 2/255 - 1; original nonzero-label mask, then unit normalization',
        'resize':('native center-square crop, then ' if args.center_crop else '')+'256 square RGB bilinear, normal/mask nearest; crop and half-pixel K affine',
        'spatial_protocol':'center_square_v2' if args.center_crop else 'stretch_square_v1',
        'official_resolution_benchmark_comparable':False,'ray_condition_used':False,
        'official_rgb_difference_blocks_validation':False}
    write_json(target/'summary.json',summary)
    if not args.center_crop:write_json(ROOT/'outputs/P1_nyu_validation_v1/summary.json',summary)
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
