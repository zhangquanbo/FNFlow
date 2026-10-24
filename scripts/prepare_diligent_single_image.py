"""Freeze the one-image DiLiGenT protocol; keep original GT resolution/mask."""
import sys,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import cv2
from scipy.io import loadmat
from PIL import Image
from src.research_io import sha256,write_json
source=ROOT/'data/real_object_selected_v1/diligent';selection=json.loads((source/'selection_before_extract_private.json').read_text());out=ROOT/'data/object_comparison_diligent_20261023';out.mkdir(exist_ok=True)
if (out/'comparison_manifest_private.json').exists():raise RuntimeError('Real-data manifest already frozen')
old=out/'protocol_before_scores.json'
if old.exists() and not (out/'protocol_v1_failed_before_scoring.json').exists():shutil.copy2(old,out/'protocol_v1_failed_before_scoring.json')
protocol={'lighting_selection':'lowest numeric official lighting ID; preselected before member extraction','input_mapping':'uint16 RGB /65535 then round*255; uint8 unchanged; no per-image exposure, no gamma transform, no light-intensity division, no crop','gt_mapping_to_common':[[-1,0,0],[0,1,0],[0,0,1]],'gt_source':'Normal_gt.mat. Inputs retain official foreground mask. Scoring valid=official mask AND nonzero finite GT; no prediction-dependent exclusion. Audit found73 zero GT pixels inside Pot2 official mask only. Other9 objects unchanged. Fail on nonfinite GT inside official mask.','gt_valid_audit_sha256':sha256(ROOT/'outputs/diligent_gt_valid_audit.json'),'coordinate_source':'https://github.com/facebookresearch/surface_normal_integration/blob/f68e47cfadc590120e1322df417067cf42442de7/datasets/diligent.py : converts original DiLiGenT normals to OpenCV by negating Y,Z; our common frame is negative OpenCV, hence negate original X only.','coordinate_source_sha256':sha256(ROOT/'external/diligent_coordinate_source.py'),'paper_comparison':'Our main is one lowest-ID image; RoSE uses21..30 ten images/object, separate protocol and not exact reproduction','selection_sha256':sha256(source/'selection_before_extract_private.json')}
write_json(out/'protocol_before_scores.json',protocol);(out/'protocol_before_scores.json.sha256').write_text(sha256(out/'protocol_before_scores.json')+'\n')
records=[]
for choice in selection['selection']:
 obj=choice['object'];folder=source/obj;rgbpath=folder/Path(choice['rgb_member']).name;maskpath=next(folder.glob('*mask*'));gtpath=next(folder.glob('*gt.mat'))
 raw=cv2.imread(str(rgbpath),cv2.IMREAD_UNCHANGED);assert raw is not None and raw.ndim==3;raw=raw[...,::-1].copy();assert raw.dtype in [np.dtype('uint8'),np.dtype('uint16')]
 rgb=np.rint(raw.astype(np.float64)/(65535 if raw.dtype==np.uint16 else 255)*255).clip(0,255).astype(np.uint8)
 valid=cv2.imread(str(maskpath),cv2.IMREAD_GRAYSCALE)>0;normal=loadmat(gtpath)['Normal_gt'].astype(np.float32);assert normal.shape[:2]==valid.shape==rgb.shape[:2]
 official_mask=valid.copy();length=np.linalg.norm(normal,axis=-1);assert np.isfinite(normal[official_mask]).all();valid=official_mask&(length>1e-8);normal[...,0]*=-1;normal/=np.maximum(length[...,None],1e-8);normal[~valid]=0
 key=f'diligent/{obj}/{Path(choice["rgb_member"]).name}';stem=hashlib.sha256(key.encode()).hexdigest();dest=out/stem;dest.mkdir(exist_ok=True)
 cache=dest/'sample.npz';np.savez_compressed(cache,rgb=rgb,normal=normal,valid=valid,official_mask=official_mask)
 white=dest/'white.png';rgba=dest/'rgba.png';Image.fromarray(np.where(official_mask[...,None],rgb,255).astype(np.uint8)).save(white);Image.fromarray(np.dstack([rgb,official_mask.astype(np.uint8)*255])).save(rgba)
 records.append({'domain':'diligent','object':obj,'image_key':key,'id':stem,'gt_path':str(cache),'gt_sha256':sha256(cache),'ours_rgb':str(white),'nlr_rgba':str(rgba),'ours_rgb_sha256':sha256(white),'nlr_rgba_sha256':sha256(rgba),'lighting_id':choice['lighting_id'],'original_rgb_dtype':str(raw.dtype),'original_shape':list(rgb.shape),'original_rgb_sha256':sha256(rgbpath),'original_mask_sha256':sha256(maskpath),'original_gt_sha256':sha256(gtpath),'raw_rgb_max':int(raw.max())})
base=json.loads((ROOT/'data/object_comparison_20261023/comparison_manifest_private.json').read_text());records.sort(key=lambda r:r['id'])
manifest={k:v for k,v in base.items() if k not in ['records','case_groups','case_ids','nlr_variability_ids','counts']};manifest.update(records=records,counts={'diligent':len(records)},case_ids=[r['id'] for r in records],case_groups={'all_real': [r['id'] for r in records]},nlr_variability_ids=[records[0]['id']],real_protocol_sha256=sha256(out/'protocol_before_scores.json'),case_selection='all ten preselected real objects; one fixed image each; seed42 panels')
target=out/'comparison_manifest_private.json';write_json(target,manifest);target.with_suffix('.json.sha256').write_text(sha256(target)+'\n');print(json.dumps({'images':len(records),'objects':len(records),'manifest':str(target),'input_ranges':[(r['object'],r['raw_rgb_max']) for r in records]}),flush=True)
