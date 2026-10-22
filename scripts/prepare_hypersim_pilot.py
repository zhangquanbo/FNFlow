"""Rebuild a scene-separated protocol pilot from native geometry normals."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import numpy as np
import cv2
import h5py
from PIL import Image
from src.data_contract import opengl_projection_to_intrinsics,opengl_projection_to_pixel_matrix,resize_intrinsics_pixel_centers,hypersim_normals_for_training
from src.research_io import ROOT,output_path,write_json,sha256,now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-root',type=Path,default=ROOT.parent/'Dataset')
    parser.add_argument('--per-scene',type=int,default=4)
    parser.add_argument('--output',default='data/hypersim_protocol_v1')
    args=parser.parse_args()
    target=output_path(ROOT/args.output)
    if (target/'summary.json').exists(): raise RuntimeError('Existing prepared data; choose another output')
    target.mkdir(parents=True,exist_ok=True)
    base=args.dataset_root/'General/Hypersim'
    tools=args.dataset_root/'Tools/Hypersim'
    official={(r['scene_name'],r['camera_name'],int(r['frame_id'])):r for r in csv.DictReader((tools/'evermotion_dataset/analysis/metadata_images_split_scene_v1.csv').open())}
    cameras={r['scene_name']:r for r in csv.DictReader((tools/'contrib/mikeroberts3000/metadata_camera_parameters.csv').open())}
    available=[json.loads(line) for line in (base/'asf_v2_full/conversion_manifest.jsonl').read_text().splitlines()]
    counts={}; records=[]; exclusions=[]; pinhole=0
    for r in available:
        if r['split'] not in ['train','val']: continue  # Do not prepare official test data for tuning.
        key=(r['scene'],r['camera'],r['frame_id']); ref=official.get(key)
        if not ref or ref['included_in_public_release']!='True' or ref['split_partition_name']!=r['split']:
            exclusions.append({'key':list(key),'reason':'official_release_or_split_mismatch'});continue
        if counts.get(r['scene'],0)>=args.per_scene:continue
        frame=f"frame.{r['frame_id']:04d}"
        root=base/'raw'/r['scene']/'images'
        geometry=root/f"scene_{r['camera']}_geometry_hdf5"/(frame+'.normal_cam.hdf5')
        candidates=list((root/f"scene_{r['camera']}_final_preview").glob(frame+'.tonemap.*'))
        if not candidates: candidates=list((root/f"scene_{r['camera']}_final_preview").glob(frame+'.color.*'))
        if len(candidates)!=1 or not geometry.exists():
            exclusions.append({'key':list(key),'reason':'native_rgb_or_normal_missing_or_ambiguous'});continue
        with h5py.File(geometry,'r') as handle: native=handle['dataset'][:]
        normal,valid=hypersim_normals_for_training(native,(256,256))
        if not valid.any(): raise RuntimeError('No valid native label pixels')
        rgb=np.asarray(Image.open(candidates[0]).convert('RGB'))
        if rgb.shape[:2]!=native.shape[:2]: raise RuntimeError('Native RGB and normals not aligned')
        camera=cameras[r['scene']]
        p=np.array([[float(camera[f'M_proj_{i}{j}']) for j in range(4)] for i in range(4)])
        projection=opengl_projection_to_pixel_matrix(p,native.shape[1],native.shape[0])
        projection=resize_intrinsics_pixel_centers(projection,(native.shape[1],native.shape[0]),(256,256))
        try:
            k=opengl_projection_to_intrinsics(p,native.shape[1],native.shape[0])
            k=resize_intrinsics_pixel_centers(k,(native.shape[1],native.shape[0]),(256,256))
            k_valid=True; pinhole+=1
        except ValueError:
            k=np.zeros((3,3));k_valid=False
        stem=f"{r['scene']}__{r['camera']}__{r['frame_id']:04d}"
        folder=target/r['split'];folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(stem+'.npz'),rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_LINEAR),
                            normal=normal,valid=valid,K=k.astype(np.float32),K_valid=np.array(k_valid),projection_3x3=projection.astype(np.float32))
        records.append({'scene':r['scene'],'camera':r['camera'],'frame_id':r['frame_id'],'split':r['split'],
                        'path':str(folder/(stem+'.npz')),'source_normal_sha256':sha256(geometry),
                        'source_rgb_sha256':sha256(candidates[0]),'valid_pixels':int(valid.sum())})
        counts[r['scene']]=counts.get(r['scene'],0)+1
    scene_sets={split:{r['scene'] for r in records if r['split']==split} for split in ['train','val']}
    assert not scene_sets['train']&scene_sets['val']
    write_json(target/'private_manifest.json',{'records':records,'exclusions':exclusions})
    summary={'created_at':now(),'status':'completed','images_by_split':{s:sum(r['split']==s for r in records) for s in scene_sets},
             'scenes_by_split':{s:len(v) for s,v in scene_sets.items()},'scene_overlap':0,'exclusions':len(exclusions),
             'normal_dtype':'float32','normal_source':'native geometry normal_cam, no bump maps',
             'canonical_basis':'x-right/y-down/z-forward','orientation':'fixed inversion after GL-to-CV basis change; ray-positive DSINE label convention',
             'resize':'nearest pixel center for normals/mask; bilinear RGB; half-pixel affine for K',
             'pinhole_K_images':pinhole,'general_central_projection_images':len(records)-pinhole,
             'camera_protocol':'exact projection_3x3 for all; K_valid false for tilt-shift/non-pinhole matrices, never substitute an approximate K',
             'rgb_source':'lossy official final_preview; protocol pilot, not claimed full-HDR training data',
             'depth_available':False,'purpose':'coordinate/mask/K and loader validation; not full mixed-domain training readiness',
             'code_sha256':sha256(__file__)}
    write_json(target/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
