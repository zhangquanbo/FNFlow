"""Prepare the fixed one-image LUCES protocol without inference or GT axis search."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import cv2
import numpy as np
from PIL import Image
from src.research_io import sha256,write_json

source=ROOT/'data/real_object_selected_v1/luces'
out=ROOT/'data/object_comparison_luces_20261023';out.mkdir(exist_ok=True)
target=out/'comparison_manifest_private.json'
if target.exists():raise RuntimeError('LUCES manifest is already frozen; refusing overwrite')
selection=json.loads((source/'selection_before_extract_private.json').read_text())
receipt=json.loads((source/'selected_intake_receipt_private.json').read_text())
assert receipt['selection_sha256']==sha256(source/'selection_before_extract_private.json')
for entry in receipt['members']:assert sha256(entry['path'])==entry['sha256'],entry['member']
official=source/'source/Luces_25082022'
protocol={
    'dataset_version':'Original LUCES; bundled Readme version 25.08.2022; 14 objects, 52 processed RGB illuminations per object.',
    'lighting_selection':'Lowest numeric illumination ID for each object; fixed before extraction. All 14 select 01.png. Use processed demosaiced rectified RGB; do not use Bayer RAW or ambient image53.',
    'input_mapping':'uint16 RGB /65535 then round*255; uint8 unchanged; no exposure, gamma, light-intensity division or crop. White background for ours, official mask alpha for Neural LightRig.',
    'gt_storage':'Official normals.png RGB channels encode normal components as 2*channel/(65535 if uint16 else255)-1; preserve native resolution, then normalize.',
    'gt_mapping_to_common':[[-1,0,0],[0,1,0],[0,0,1]],
    'coordinate_evidence':'Bundled Readme identifies aligned GT mesh with MeshLab y-up, forward -z convention and points to code/io_nf.py. Its load_normal_map reads RGB XYZ and negates Y,Z for the camera frame used by the perspective reconstruction code. Our common frame is negative OpenCV: negate original stored X only. No test GT axis search.',
    'bundled_readme_sha256':sha256(official/'Readme.txt'),
    'bundled_io_source_sha256':sha256(official/'code/io_nf.py'),
    'independent_primary_source':'https://github.com/dlichy/FastNFPSCode/blob/c503d59ba8fcfc43ee8aa981a1ed375801b783a9/datasets/io_nf.py',
    'independent_io_source_sha256':sha256(ROOT/'external/luces_io_source.py'),
    'mask':'Inputs use the original official nonzero mask, without erosion. Score official mask AND finite nonzero decoded GT. Nonfinite GT inside official mask is a hard failure; record any zero-GT exclusion. No output-dependent selection.',
    'paper_comparison':'Main is one lowest-ID image/object; RoSE uses indices21..30 ten images/object. Not an exact RoSE reproduction.',
    'selection_sha256':sha256(source/'selection_before_extract_private.json'),
    'intake_receipt_sha256':sha256(source/'selected_intake_receipt_private.json'),
    'coordinate_lock_sha256':sha256(ROOT/'outputs/lightrig_calibration_v1/coordinate_lock.json'),
    'frozen_before_luces_model_scores':True,'created_unix':time.time(),
}
protocol_path=out/'protocol_before_scores.json'
if protocol_path.exists():
    old=json.loads(protocol_path.read_text());protocol['created_unix']=old['created_unix'];assert protocol==old,'Frozen protocol changed'
else:write_json(protocol_path,protocol);protocol_path.with_suffix('.json.sha256').write_text(sha256(protocol_path)+'\n')
records=[];audits=[]
try:
    for choice in selection['selection']:
        obj=choice['object'];rgbpath=source/'source'/choice['rgb_member'];gtpath=source/'source'/choice['normal_member'];maskpath=source/'source'/choice['mask_member']
        raw=cv2.imread(str(rgbpath),cv2.IMREAD_UNCHANGED);stored=cv2.imread(str(gtpath),cv2.IMREAD_UNCHANGED);mask=cv2.imread(str(maskpath),cv2.IMREAD_GRAYSCALE)
        assert raw is not None and raw.ndim==3 and raw.shape[-1]==3
        assert stored is not None and stored.shape==raw.shape and mask is not None and mask.shape==raw.shape[:2]
        assert raw.dtype in (np.dtype('uint8'),np.dtype('uint16')) and stored.dtype in (np.dtype('uint8'),np.dtype('uint16'))
        raw=raw[...,::-1].copy();rgb=np.rint(raw.astype(np.float64)/(65535 if raw.dtype==np.uint16 else 255)*255).clip(0,255).astype(np.uint8)
        normal=stored[...,::-1].astype(np.float32)/(65535 if stored.dtype==np.uint16 else 255)*2-1
        official_mask=mask>0;assert official_mask.any();assert np.isfinite(normal[official_mask]).all()
        lengths=np.linalg.norm(normal,axis=-1);valid=official_mask&(lengths>1e-8)
        normal[...,0]*=-1;normal/=np.maximum(lengths[...,None],1e-8);normal[~valid]=0
        key=f'luces/{obj}/{rgbpath.name}';stem=hashlib.sha256(key.encode()).hexdigest();dest=out/stem;dest.mkdir(exist_ok=True)
        cache=dest/'sample.npz';np.savez_compressed(cache,rgb=rgb,normal=normal,valid=valid,official_mask=official_mask)
        white=dest/'white.png';rgba=dest/'rgba.png'
        Image.fromarray(np.where(official_mask[...,None],rgb,255).astype(np.uint8)).save(white)
        Image.fromarray(np.dstack([rgb,official_mask.astype(np.uint8)*255])).save(rgba)
        audit={'object':obj,'shape':list(rgb.shape),'rgb_dtype':str(raw.dtype),'rgb_max':int(raw.max()),'normal_storage_dtype':str(stored.dtype),'official_mask_pixels':int(official_mask.sum()),'valid_gt_pixels':int(valid.sum()),'zero_gt_exclusions':int((official_mask&~valid).sum()),'stored_gt_length_min':float(lengths[official_mask].min()),'stored_gt_length_max':float(lengths[official_mask].max())}
        audits.append(audit)
        records.append({'domain':'luces','object':obj,'image_key':key,'id':stem,'gt_path':str(cache),'gt_sha256':sha256(cache),'ours_rgb':str(white),'nlr_rgba':str(rgba),'ours_rgb_sha256':sha256(white),'nlr_rgba_sha256':sha256(rgba),'lighting_id':choice['lighting_id'],'original_rgb_sha256':sha256(rgbpath),'original_gt_sha256':sha256(gtpath),'original_mask_sha256':sha256(maskpath),'original_shape':list(rgb.shape),'original_rgb_dtype':str(raw.dtype),'gt_valid_audit':audit})
        print(obj,'prepared',flush=True)
    assert len(records)==14 and len({r['object'] for r in records})==14
    base=json.loads((ROOT/'data/object_comparison_20261023/comparison_manifest_private.json').read_text());records.sort(key=lambda r:r['id'])
    manifest={k:v for k,v in base.items() if k not in ['records','case_groups','case_ids','nlr_variability_ids','counts']}
    manifest.update(records=records,counts={'luces':14},case_ids=[r['id'] for r in records],case_groups={'all_real':[r['id'] for r in records]},nlr_variability_ids=[r['id'] for r in records[:2]],real_protocol_sha256=sha256(protocol_path),case_selection='All14 preselected original LUCES objects; one fixed01.png each; seed42 panels. Hash-lowest ceil(10%*14)=2 images for NLR seed variability.')
    write_json(out/'gt_valid_audit.json',{'objects':audits,'total_zero_gt_exclusions':sum(a['zero_gt_exclusions'] for a in audits),'mask_erosion':False,'model_scores_computed':False})
    write_json(target,manifest);target.with_suffix('.json.sha256').write_text(sha256(target)+'\n')
    print(json.dumps({'images':14,'objects':14,'manifest_sha256':sha256(target),'gt_valid_audit':audits}),flush=True)
except Exception as e:
    write_json(out/'preparation_failure.json',{'error':repr(e),'completed_objects':audits,'time':time.time()});raise
