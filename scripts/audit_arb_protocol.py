"""Summarize the acquired protocol sample without inventing camera calibration."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from collections import Counter
import json
import numpy as np
from PIL import Image
import cv2
from src.research_io import ROOT,write_json,now,sha256

def main():
    base=ROOT/'outputs/P0_arb_sample_v1'
    manifest=json.loads((base/'private_manifest.json').read_text())
    licenses=json.loads((base/'upstream_licenses_private.json').read_text())
    tree=json.loads((ROOT/'outputs/P0_data_intake_local_v1/arb_data_tree.json').read_text())
    sample=json.loads((base/'summary.json').read_text())
    if sample['status']!='completed':raise RuntimeError('Sample acquisition incomplete')
    records=[];rgb_modes=Counter();calibrated_assets=0
    for asset in manifest['records']:
        folder=(ROOT/asset['files'][0]['path'].replace('\\','/')).parent
        meta=json.loads((folder/'camera.json').read_text())
        has_k=any(key in meta for key in ['K','intrinsics','camera_angle_x','focal_length'])
        calibrated_assets+=int(has_k)
        for view in ['00','03']:
            normals=cv2.imread(str(folder/f'normal_{view}.exr'),cv2.IMREAD_UNCHANGED)
            if normals is None or normals.shape!=(512,512,3):raise RuntimeError('Normal shape/decode failure')
            # The source writer hands BlenderProc's array directly to cv2.imwrite.
            # These native OpenCV channels are semantic xyz, not ordinary image BGR.
            normal=normals.astype(np.float32)*2-1
            normal[...,0]*=-1  # Equivalent to the published IDArb loader's full channel transform.
            length=np.linalg.norm(normal,axis=-1)
            masks=[]
            for light in [0,1]:
                with Image.open(folder/f'color_{light}_{view}.png') as im:
                    rgb_modes[im.mode]+=1
                    if im.mode!='RGBA':raise RuntimeError('Missing RGB alpha for sample')
                    if im.size!=(512,512):raise RuntimeError('RGB shape mismatch')
                    masks.append(np.asarray(im)[...,3]>=254)
            intersection=masks[0]&masks[1]
            valid=intersection&np.isfinite(normal).all(-1)&(length>1e-6)
            if not valid.any():raise RuntimeError('Empty sample intersection')
            records.append({'asset':asset['asset'],'view':view,'valid_pixels':int(valid.sum()),
                'alpha_iou':float(intersection.sum()/(masks[0]|masks[1]).sum()),
                'unit_length_median_before_renormalizing':float(np.median(length[valid])),
                'positive_forward_fraction':float((normal[...,2][valid]>0).mean()),
                'has_explicit_K':has_k})
    write_json(base/'protocol_checks_private.json',{'records':records})
    summary={'status':'completed','created_at':now(),'revision':sample['revision'],
        'purpose':'protocol audit only; no model training or metric evaluation',
        'curated_archive_count':len(tree),'curated_archive_bytes':sum(r['size'] for r in tree),
        'archive_members':sample['archive_members'],'assets_in_inspected_archive':sample['archive_assets'],
        'sampled_assets':sample['sampled_assets'],'sampled_geometry_views':len(records),
        'sampled_rgb_images':sum(rgb_modes.values()),'downloaded_sample_bytes':sample['sample_bytes'],
        'range_network_bytes':sample['network_bytes'],'rgb_mode_counts':dict(rgb_modes),
        'explicit_depth_files_in_sample':sample['sampled_depth_files'],
        'explicit_mask_files_in_sample':sample['sampled_explicit_mask_files'],
        'alpha_masks_available':True,'minimum_two_light_opaque_mask_iou':min(r['alpha_iou'] for r in records),
        'explicit_intrinsics_assets':calibrated_assets,
        'upstream_license_counts':dict(Counter(r['license'] or 'unknown' for r in licenses['records'])),
        'all_sample_licenses_found':all(r['found_in_upstream_metadata'] for r in licenses['records']),
        'data_card_license_tag':'mit; retain separate per-asset upstream metadata and attribution',
        'normal_decode':'OpenCV native channels * 2 - 1, then (-x,+y,+z), equivalent to IDArb loader; no GT-selected transform',
        'normal_semantics':'Cycles normal pass; material Normal links are retained in the rendering script. Do not call these verified pure geometry normals.',
        'training_ready':False,'remaining_gates':['verify or recover exact intrinsics',
            'decide shading-normal versus geometry-normal supervision and verify with a controlled render',
            'scale acquisition across multiple archives and split/deduplicate by asset before choosing views'],
        'render_script_sha256':sha256(ROOT/'outputs/P0_data_intake_local_v1/render_normal_orm.py')}
    write_json(ROOT/'outputs/P0_arb_protocol_v1/summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
