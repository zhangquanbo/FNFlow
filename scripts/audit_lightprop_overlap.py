import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research_io import sha256
meta=ROOT/'external/lightprop_metadata/metadata.json';m=json.loads(meta.read_text())
assert all(isinstance(v,list) and all(isinstance(x,str) for x in v) for v in m.values())
uids=set(x.lower().replace('-','') for v in m.values() for x in v)
p=ROOT/'data/arb_mixed_formal_v1/private_manifest.json';records=json.loads(p.read_text())['records'];val=[r for r in records if r['split']=='val']
assets=set(r['scene'].lower().replace('-','') for r in val); overlap=assets&uids
kept=[r for r in val if r['scene'].lower().replace('-','') not in overlap]
out=ROOT/'data/object_comparison_20261023/arb';out.mkdir(parents=True,exist_ok=True)
receipt={'source_revision':json.loads((meta.parent/'receipt.json').read_text())['revision'],'lightprop_metadata_sha256':sha256(meta),'arb_manifest_sha256':sha256(p),'lightprop_split_counts':{k:len(v) for k,v in m.items()},'lightprop_unique_assets':len(uids),'validation_assets_before':len(assets),'intersection_assets':len(overlap),'validation_images_before':len(val),'validation_assets_after':len(assets-overlap),'validation_images_after':len(kept),'overlap_uids_private':sorted(overlap),'records':kept,'uid_normalization':'lowercase, remove hyphens; include all declared LightProp splits'}
target=out/'deduplicated_private.json';target.write_text(json.dumps(receipt,indent=2));target.with_suffix('.json.sha256').write_text(sha256(target)+'\n');print({k:v for k,v in receipt.items() if k not in ['records','overlap_uids_private']})
import torch
ck=torch.load(ROOT/'external/lightrig_weights_5619cfe/mld.pt',map_location='cpu',weights_only=False)
print('checkpoint_keys',list(ck));sd=ck.get('state_dict',ck);prefix={}
for k,v in sd.items():
 key='.'.join(k.split('.')[:2]);prefix[key]=prefix.get(key,0)+v.numel()
print('state_prefix_parameters',prefix)