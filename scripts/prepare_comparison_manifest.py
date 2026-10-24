import sys,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
from src.research_io import sha256,write_json
from src.gso_boundary_protocol import angular_band
out=ROOT/'data/object_comparison_20261023';lock=ROOT/'outputs/lightrig_calibration_v1/coordinate_lock.json'
assert json.loads(lock.read_text())['status']=='closed'
target=out/'comparison_manifest_private.json'
if target.exists():raise RuntimeError('Comparison manifest already frozen')
gso=json.loads((out/'gso/prepared_private.json').read_text());arb=json.loads((out/'arb/deduplicated_private.json').read_text());records=[];features={}
for domain,rows in [('gso',gso['records']),('arb',arb['records'])]:
 for r in rows:
  key=r['image_key'];stem=hashlib.sha256(key.encode()).hexdigest();folder=out/'inputs'/stem;folder.mkdir(parents=True,exist_ok=True)
  with np.load(r['path']) as z:rgb=z['rgb'].copy();valid=z['valid'].astype(bool)
  if domain=='gso':
   inputs=r['inputs'];white=inputs['ours']['path'];rgba=inputs['lightrig']['path'];assert sha256(white)==inputs['ours']['sha256'] and sha256(rgba)==inputs['lightrig']['sha256']
  else:
   white=str(folder/'white.png');rgba=str(folder/'rgba.png')
   # Preserve the original alpha-white-composited training RGB for ours.
   Image.fromarray(rgb).save(white);Image.fromarray(np.dstack([rgb,valid.astype(np.uint8)*255])).save(rgba)
  rec={'domain':domain,'object':r['scene'],'image_key':key,'id':stem,'gt_path':r['path'],'gt_sha256':sha256(r['path']),'ours_rgb':white,'nlr_rgba':rgba,'ours_rgb_sha256':sha256(white),'nlr_rgba_sha256':sha256(rgba)};records.append(rec)
  if domain=='gso':
   pixels=rgb[valid].astype(float)/255;bright=float(((pixels.min(1)>.85)&((pixels.max(1)-pixels.min(1))<.15)).mean())
   padded=np.pad(valid,2);interior=np.ones_like(valid)
   for dy in range(5):
    for dx in range(5):interior&=padded[dy:dy+valid.shape[0],dx:dx+valid.shape[1]]
   thin=float((valid&~interior).sum()/valid.sum())
   ys,xs=np.where(valid);crop=valid[ys.min():ys.max()+1,xs.min():xs.max()+1];flip=crop[:,::-1];sym=float((crop&flip).sum()/max((crop|flip).sum(),1))
   features.setdefault(r['scene'],[]).append({'id':stem,'bright':bright,'thin':thin,'sym':sym})
records.sort(key=lambda r:r['id']);case_ids=[];case_groups={}
for name in ['bright','thin','sym']:
 ranked=sorted(features,key=lambda k:(-np.mean([v[name] for v in features[k]]),hashlib.sha256(k.encode()).hexdigest()))[:3]
 ids=[min(features[k],key=lambda v:v['id'])['id'] for k in ranked];case_groups[name]=ids;case_ids+=ids
case_groups['vase_name']=[r['id'] for r in records if r['domain']=='gso' and 'vase' in r['object'].lower()][:3];case_ids+=case_groups['vase_name']
case_groups['arb_fixed']=[r['id'] for r in records if r['domain']=='arb'][:3];case_ids+=case_groups['arb_fixed']
cfg=json.loads((ROOT/'configs/parameterization_first_round_v2_review2.json').read_text());checkpoints={}
for m in ['R','V','Xn']:
 p=ROOT/cfg['output_root']/f'cooldown_{m}_k50000_s1247/last_local.pth';checkpoints[m]={'path':str(p),'sha256':sha256(p)}
meta=json.loads((ROOT/'external/lightprop_metadata/metadata.json').read_text());assets=set(r['object'] for r in records if r['domain']=='arb')
original=json.loads((ROOT/'data/arb_mixed_formal_v1/private_manifest.json').read_text())['records'];before=set(r['scene'] for r in original if r['split']=='val')
normalize=lambda x:x.lower().replace('-','')
overlap={split:len({normalize(x) for x in vals}&{normalize(x) for x in before}) for split,vals in meta.items()}
manifest={'scope_sha256':sha256(ROOT/'docs/object_scope_lightrig_20261023.md'),'amendment_sha256':sha256(ROOT/'docs/object_calibration_intake_amendment_20261023.md'),'coordinate_lock_sha256':sha256(lock),'checkpoints':checkpoints,'records':records,'case_groups':case_groups,'case_ids':sorted(set(case_ids)),'case_selection':'pre-score RGB near-white low-chroma pixel fraction, mask thinness, silhouette horizontal symmetry proxies; top3 objects each, lowest-hash view. Proxy labels do not prove material/structure categories. Name vase first3 hash; Arb first3 hash. seed42 fixed for our panels.','nlr_variability_ids':[r['id'] for r in records[:math.ceil(.1*len(records))]],'arb_overlap_note':{'excluded_union':70,'original_assets':198,'percent':70/198*100,'retained_assets':128,'retained_views':384,'split_specific_intersections':overlap,'note':'union overlap, possible source advantage; no exhaustive GSO source dedup claim'},'counts':{d:sum(r['domain']==d for r in records) for d in ['gso','arb']},'arb_input_note':'ours original precomposited training RGB unchanged; NLR alpha=GT-valid; no prediction masks used','script_sha256':sha256(__file__)}
write_json(target,manifest);target.with_suffix('.json.sha256').write_text(sha256(target)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in ['records','checkpoints','case_groups','case_ids','nlr_variability_ids']}),flush=True)
