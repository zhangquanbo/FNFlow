"""Compare selected non-test RGB arrays with the authoritative remote HDF5."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/tools_python312'))
import hashlib
import json
import argparse
import h5py
import numpy as np
from src.http_range import HTTPRangeReader,CachedRangeReader
from src.research_io import ROOT,write_json,now
from scripts.fetch_nyu_scene_metadata import URL,SIZE

parser=argparse.ArgumentParser()
parser.add_argument('--checks',type=Path,default=ROOT/'outputs/P0_nyu_validation_v1/private_rgb_checks.json')
parser.add_argument('--output',type=Path,default=ROOT/'outputs/P0_nyu_validation_v1/private_rgb_correspondence.json')
parser.add_argument('--limit',type=int,default=0)
parser.add_argument('--save-arrays',action='store_true')
parser.add_argument('--diagnostic-only',action='store_true')
args=parser.parse_args()
checks=json.loads(args.checks.read_text())['records']
if args.limit:checks=checks[:args.limit]
reader=HTTPRangeReader(URL,SIZE,budget=48*1024*1024,timeout=20)
cached=CachedRangeReader(reader)
records=[]
with h5py.File(cached,'r') as handle:
    images=handle['images']
    if images.shape!=(1449,3,640,480):raise RuntimeError('Unexpected official RGB array layout')
    for item in checks:
        arr=images[item['official_index_1based']-1].transpose(2,1,0).copy()
        digest=hashlib.sha256(arr.tobytes()).hexdigest()
        if args.save_arrays:
            from src.research_io import output_path
            array_path=output_path(args.output.parent/f"nyu_official_rgb_{item['official_index_1based']:04d}.npy")
            np.save(array_path,arr)
        records.append({'official_index_1based':item['official_index_1based'],
                        'exact_rgb_pixels_match':digest==item['rgb_pixels_sha256']})
        print('RGB check',len(records),records[-1]['exact_rgb_pixels_match'],flush=True)
result={'created_at':now(),'all_match':all(r['exact_rgb_pixels_match'] for r in records),
        'records':records,'network_bytes':reader.downloaded,'range_requests':reader.requests,
        'scope':'three preselected training-partition RGB arrays; test image pixels were not requested'}
write_json(args.output,result)
print(json.dumps(result),flush=True)
if not result['all_match'] and not args.diagnostic_only:raise RuntimeError('RGB correspondence failed; do not mark preparation ready')
