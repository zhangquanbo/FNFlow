"""Bind the packaged NYUv2 images to the official split and scene metadata."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.io import loadmat
from src.research_io import ROOT,now,sha256,write_json

def main():
    out=ROOT/'outputs/P0_nyu_validation_v1'
    mapping_path=ROOT/'outputs/P0_nyu_scenes_v1/private_scene_mapping.json'
    split_path=ROOT/'outputs/P0_data_intake_local_v1/splits.mat'
    mapping=json.loads(mapping_path.read_text())
    if len(mapping['records'])!=1449:raise RuntimeError('Incomplete official scene mapping')
    scenes={r['official_index_1based']:r for r in mapping['records']}
    splits=loadmat(split_path)
    dataset=ROOT.parent/'Dataset/General/dsine_eval/nyuv2'
    reference=ROOT.parent/'ExternalModels/DSINE/repo/data/datasets/nyuv2/split'
    records={};checks={}
    for split,key,count in [('train','trainNdxs',795),('test','testNdxs',654)]:
        indices={int(x) for x in splits[key].reshape(-1)}
        paths=sorted((dataset/split).glob('*_img.png'))
        actual={int(p.name.split('_')[0])+1 for p in paths}
        declared={int(line.split('/')[-1].split('_')[0])+1 for line in (reference/(split+'.txt')).read_text().splitlines() if line.strip()}
        if len(indices)!=count or indices!=actual or indices!=declared:
            raise RuntimeError('Official, packaged and DSINE split indices do not match')
        entries=[]
        for path in paths:
            index=int(path.name.split('_')[0])+1
            normal=path.with_name(path.name.replace('_img.png','_normal.png'))
            camera=path.with_name(path.name.replace('_img.png','_intrins.npy'))
            if not normal.is_file() or not camera.is_file():raise RuntimeError('Missing normal or intrinsics')
            entries.append({**scenes[index],'rgb_path':str(path),'normal_path':str(normal),'K_path':str(camera),
                            'role':'real_indoor_validation' if split=='train' else 'reserved_test',
                            'training_allowed':False})
        records[split]=entries
        checks[split]={'images':len(paths),'scenes':len({r['scene'] for r in entries}),
                       'rooms':len({r['room'] for r in entries}),
                       'official_indices_match':True,'dsine_split_list_matches':True}
    overlap={r['room'] for r in records['train']}&{r['room'] for r in records['test']}
    if overlap:raise RuntimeError('Validation and reserved test share rooms')
    hashes=[]
    candidates=[records['train'][i] for i in (0,len(records['train'])//2,-1)]
    for row in candidates:
        with Image.open(row['rgb_path']) as im:arr=np.asarray(im.convert('RGB'))
        hashes.append({'official_index_1based':row['official_index_1based'],
            'rgb_pixels_sha256':hashlib.sha256(arr.tobytes()).hexdigest(),'shape':list(arr.shape)})
    write_json(out/'private_manifest.json',{'validation':records['train'],'reserved_test':records['test'],
        'official_split_sha256':sha256(split_path),'official_scene_mapping_sha256':sha256(mapping_path)})
    write_json(out/'private_rgb_checks.json',{'records':hashes})
    summary={'status':'completed','created_at':now(),'official_split_url':'https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat',
        'official_split_sha256':sha256(split_path),'source_scene_mapping_sha256':sha256(mapping_path),
        'partitions':{'validation':checks['train'],'reserved_test':checks['test']},
        'room_overlap_count':len(overlap),'training_images':0,
        'normal_source':'GeoNet depth-derived packaged labels, not direct geometry',
        'scope':'metadata and pair existence checked; label coordinate/decode protocol still requires the preparation step',
        'pixel_correspondence_check':'pending selected training-partition RGB comparisons'}
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
