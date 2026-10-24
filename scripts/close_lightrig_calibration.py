"""Close coordinate convention on analytic calibration only, before test scoring."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scripts.run_lightrig_calibration import fit
from src.research_io import write_json,sha256
out=ROOT/'outputs/lightrig_calibration_v1';prior=json.loads((out/'result.json').read_text());assert prior['empirical_pass']
source=ROOT/'data/lightrig_shader_check_v3';checks=[]
for name in ['sphere','cube']:
 data=np.load(source/(name+'.npz'));gt=data['normal_gl'];im=np.array(Image.open(source/(name+'_source_shader.png')).convert('RGBA'))
 valid=data['valid']&(im[...,3]>252);valid=(1-F.max_pool2d(1-torch.from_numpy(valid.astype(np.float32))[None,None],17,1,8))[0,0].numpy()>.5
 raw=im[...,:3].astype(np.float64)/127.5-1;raw/=np.maximum(np.linalg.norm(raw,axis=-1,keepdims=True),1e-8);rows=fit(raw,gt,valid)
 assert rows[0]['matrix']==prior['joint_best']['matrix'] and rows[0]['mae']<1,(name,rows[0])
 checks.append({'name':name,'source_shader_best':rows[0],'source_shader_sha256':sha256(source/(name+'_source_shader.png'))})
lock={'status':'closed','nlr_to_outward_camera_gl':prior['joint_best']['matrix'],'nlr_to_common':prior['nlr_to_common'],'nlr_native_axes':'x right, y up, z toward viewer; outward camera-GL normal','common_axes':'x left, y up, z toward viewer; -OpenCV camera normal used by frozen GSO/Arb protocol','storage':'official RGB uint8 -> /127.5-1; bilinear component resize then normalize','calibration_result_sha256':sha256(out/'result.json'),'source_checks':checks,'source_check_fix':'Initial shader-only diagnostic retained Medium High Contrast look, which alters encoded normal channels even with Raw. Repaired verification uses author source Raw + None look; original failure retained. No NLR rerun or parameter tuning.','source_renderer_sha256':sha256(ROOT/'scripts/render_lightrig_shader_check_v3.py'),'source_reference':'https://github.com/ZexinHe/Neural-LightRig/issues/6','source_reference_commit':'be3d6b32a195f968540abc2ee106dc02d4b07479','test_gt_used':False,'amendment_sha256':sha256(ROOT/'docs/object_calibration_intake_amendment_20261023.md')}
target=out/'coordinate_lock.json'
if target.exists():raise RuntimeError('Coordinate lock already exists; do not overwrite')
write_json(target,lock);target.with_suffix('.json.sha256').write_text(sha256(target)+'\n');print(json.dumps(lock),flush=True)
