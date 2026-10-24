"""Validate all prepared LUCES inputs against original pixels and official decoder."""
import ast,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import cv2
import numpy as np
from PIL import Image
from src.research_io import write_json,sha256
data=ROOT/'data/object_comparison_luces_20261023';mp=data/'comparison_manifest_private.json'
assert sha256(mp)==mp.with_suffix('.json.sha256').read_text().strip()
man=json.loads(mp.read_text());assert len(man['records'])==14
source=ROOT/'data/real_object_selected_v1/luces/source/Luces_25082022'
code=source/'code/io_nf.py';protocol=json.loads((data/'protocol_before_scores.json').read_text())
assert sha256(code)==protocol['bundled_io_source_sha256']
# Execute only the reviewed original pure image-loading function, not the module,
# imports, model code, or TensorFlow dependencies bundled with the dataset.
tree=ast.parse(code.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_normal_map')
scope={'np':np,'cv2':cv2,'os':os};exec(compile(ast.Module(body=[node],type_ignores=[]),str(code),'exec'),scope)
checks=[]
for r in man['records']:
    obj=r['object'];folder=source/'data'/obj;assert r['lighting_id']==1
    for field in ['gt','ours_rgb','nlr_rgba']:assert sha256(r[field+'_path'] if field=='gt' else r[field])==r[field+'_sha256']
    with np.load(r['gt_path']) as z:gt=z['normal'].copy();valid=z['valid'].copy();official=z['official_mask'].copy()
    mask=cv2.imread(str(folder/'mask.png'),cv2.IMREAD_GRAYSCALE)>0;assert np.array_equal(mask,official)
    # Original loader returns camera normals with +Z into scene. Common frame is its negative.
    decoded=scope['load_normal_map'](str(folder/'normals.png'),gt.shape[0],gt.shape[1],1.0)
    err=float(np.abs(gt[valid]+decoded[valid]).max());assert err<2e-7,(obj,err)
    assert valid.any() and np.isfinite(gt[valid]).all()
    unit_error=float(np.abs(np.linalg.norm(gt[valid],axis=-1)-1).max());assert unit_error<2e-7
    assert np.array_equal(valid,mask),'Unexpected GT invalidity; record and review before scoring'
    rgba=np.array(Image.open(r['nlr_rgba']).convert('RGBA'));white=np.array(Image.open(r['ours_rgb']).convert('RGB'))
    raw=cv2.imread(str(folder/'01.png'),cv2.IMREAD_UNCHANGED)[...,::-1]
    expected=np.rint(raw.astype(np.float64)/(65535 if raw.dtype==np.uint16 else 255)*255).clip(0,255).astype(np.uint8)
    assert np.array_equal(rgba[...,:3],expected) and np.array_equal(rgba[...,3],mask.astype(np.uint8)*255)
    assert np.array_equal(white,np.where(mask[...,None],expected,255))
    checks.append({'object':obj,'pixels':int(valid.sum()),'official_decoder_component_max_abs_error':err,'unit_length_max_abs_error':unit_error,'input_mask_and_quantization_exact':True})
write_json(data/'validation.json',{'status':'passed','manifest_sha256':sha256(mp),'objects':checks,'objects_checked':len(checks),'official_masks_unchanged':True,'gt_zero_exclusions':0,'test_gt_axis_search':False,'training':False})
print(json.dumps({'status':'passed','objects':len(checks),'zero_gt_exclusions':0}),flush=True)
