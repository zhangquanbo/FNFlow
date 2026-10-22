"""Build no-ray training data with EXR-owned masks and declared shading labels."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import argparse,json,gzip,hashlib
from pathlib import PurePosixPath
from collections import Counter
import cv2
import numpy as np
from PIL import Image
from src.object_normals import decode_arb_normal
from src.research_io import ROOT,output_path,write_json,now,sha256
from src.general_dataset import PreparedNormalDataset

def fixed_formal_partition():
    """Derive the complete 2,000-asset split before consuming pilot examples."""
    acquisition=ROOT/'outputs/P1_arb_acquisition_v1'
    selection=json.loads((acquisition/'selection_private.json').read_text())
    all_uids=[]
    for archive in selection['archives']:
        stem=PurePosixPath(archive['path']).stem;folder=acquisition/stem
        entries=json.loads((folder/'index_private.json').read_text())['members']
        with gzip.open(folder/'upstream_metadata.json.gz','rt',encoding='utf-8') as f:metadata=json.load(f)
        groups={}
        for entry in entries:
            path=PurePosixPath(entry['filename'])
            if len(path.parts)>1:groups.setdefault(str(path.parent),set()).add(path.name)
        candidates=[]
        for asset,members in groups.items():
            uid=PurePosixPath(asset).name
            if metadata.get(uid,{}).get('license') not in ['by','cc0']:continue
            wanted=['camera.json']+[n for v in [0,4,8] for n in [f'normal_{v:02d}.exr',f'color_{int(hashlib.sha256((uid+str(v)).encode()).hexdigest()[:8],16)%7}_{v:02d}.png']]
            if all(n in members for n in wanted):candidates.append(asset)
        candidates.sort(key=lambda x:hashlib.sha256(('arb-round1-v1:'+x).encode()).hexdigest())
        if len(candidates)<125:raise RuntimeError('Incomplete full asset plan')
        all_uids.extend(PurePosixPath(x).name for x in candidates[:125])
    if len(all_uids)!=2000 or len(set(all_uids))!=2000:raise RuntimeError('Formal asset plan is not 2000 unique UIDs')
    all_uids.sort(key=lambda uid:hashlib.sha256(('asset-split-v1:'+uid).encode()).hexdigest())
    plan={uid:('val' if i<200 else 'train') for i,uid in enumerate(all_uids)}
    return plan

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','pilot','formal'],required=True)
    args=parser.parse_args();phase=args.phase
    target=output_path(ROOT/f'data/arb_mixed_{phase}_v1')
    if (target/'summary.json').exists():
        print((target/'summary.json').read_text());return
    if phase=='smoke':
        old=json.loads((ROOT/'outputs/P0_arb_sample_v1/private_manifest.json').read_text())['records']
        assets=[]
        for i,row in enumerate(old):
            folder=(ROOT/row['files'][0]['path'].replace('\\','/')).parent
            assets.append({'asset':folder.name,'folder':folder.relative_to(ROOT).as_posix(),
                'split':'val' if i==0 else 'train','label_source':'rendered_bsdf_normal_pass','material_normal_texture':'unknown'})
    else:
        assets=json.loads((ROOT/f'outputs/P1_arb_acquisition_v1/{phase}_manifest_private.json').read_text())['records']
        plan=fixed_formal_partition()
        for asset in assets:asset['split']=plan[asset['asset']]
        if phase=='formal' and {a['asset'] for a in assets}!=set(plan):raise RuntimeError('Acquired UIDs differ from fixed complete plan')
    target.mkdir(parents=True,exist_ok=True);records=[];exclusions=[];valid_count=0
    for i,asset in enumerate(assets):
        folder=ROOT/asset['folder'];split=asset['split']
        destination=target/split;destination.mkdir(exist_ok=True)
        for normal_path in sorted(folder.glob('normal_*.exr')):
            view=normal_path.stem.split('_')[-1]
            candidates=sorted(folder.glob(f'color_*_{view}.png'))
            if not candidates:raise RuntimeError('Missing selected RGB')
            # Protocol smoke may contain two lights; use the first only here.
            rgb_path=candidates[0]
            native=cv2.imread(str(normal_path),cv2.IMREAD_UNCHANGED)
            if native is None:raise RuntimeError('EXR decoding failed')
            normal,valid=decode_arb_normal(native)
            if not valid.any():
                exclusions.append({'asset':asset['asset'],'view':view,'split':split,
                    'reason':'no finite near-unit EXR normal after fixed nearest resize',
                    'source_normal_sha256':sha256(normal_path),'source_rgb_sha256':sha256(rgb_path)})
                print('declared empty-label exclusion',asset['asset'],view,flush=True)
                continue
            with Image.open(rgb_path) as image:
                rgba=np.asarray(image.convert('RGBA')).astype(np.float32)/255
            if rgba.shape[:2]!=native.shape[:2]:raise RuntimeError('Native RGB/normal mismatch')
            composited=rgba[...,:3]*rgba[...,3:4]+(1-rgba[...,3:4])
            rgb=np.clip(np.rint(cv2.resize(composited,(256,256))*255),0,255).astype(np.uint8)
            path=destination/f"{asset['asset']}_{view}.npz"
            np.savez_compressed(path,rgb=rgb,normal=normal,valid=valid,
                K=np.zeros((3,3),np.float32),K_valid=np.array(False),
                projection_3x3=np.zeros((3,3),np.float32),projection_valid=np.array(False))
            records.append({'scene':asset['asset'],'split':split,'path':str(path),
                'image_key':f"arb/{asset['asset']}/{rgb_path.name}",
                'label_source':'rendered_bsdf_normal_pass','material_normal_texture':asset['material_normal_texture'],
                'valid_pixels':int(valid.sum()),'source_normal_sha256':sha256(normal_path),'source_rgb_sha256':sha256(rgb_path)})
            valid_count+=int(valid.sum())
        if (i+1)%50==0:print('prepared assets',i+1,flush=True)
    write_json(target/'private_manifest.json',{'records':records,'phase':phase,'exclusions':exclusions,
        'split_source':'complete 2000-asset global hash plan, pilot inherits these identities' if phase!='smoke' else 'isolated 3-asset debugging data'})
    for split in ['train','val']:
        data=PreparedNormalDataset(target,split)
        for sample in data:
            if sample['K_valid'] or sample['projection_valid']:raise RuntimeError('Invented camera calibration')
    summary={'status':'completed','created_at':now(),'phase':phase,'assets':len(assets),'images':len(records),
        'assets_count_definition':'acquired planned assets; usable assets reported separately',
        'usable_assets':len({r['scene'] for r in records}),'excluded_views':len(exclusions),
        'empty_assets':len({a['asset'] for a in assets}-{r['scene'] for r in records}),
        'exclusion_rule':'zero valid EXR pixels only; no prediction/metric-dependent filtering; all exclusions recorded in private manifest',
        'images_by_split':dict(Counter(r['split'] for r in records)),
        'assets_by_split':{s:len({r['scene'] for r in records if r['split']==s}) for s in ['train','val']},
        'asset_overlap':len({r['scene'] for r in records if r['split']=='train'}&{r['scene'] for r in records if r['split']=='val'}),
        'valid_pixels':valid_count,'mask':'EXR finite, decoded length within 0.05 of one; alpha only composites input background',
        'label_source':'Cycles BSDF normal pass, material_normal_texture may be unknown',
        'ray_condition_used':False,'K_missing':True,'all_samples_loaded_and_checked':True,
        'phase_scope':'smoke debugging only, 3 assets' if phase=='smoke' else 'single-seed numerical pilot' if phase=='pilot' else 'fixed-budget formal comparison data'}
    write_json(target/'summary.json',summary)
    write_json(ROOT/f'outputs/P1_arb_prepared_{phase}_v1/summary.json',summary)
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
