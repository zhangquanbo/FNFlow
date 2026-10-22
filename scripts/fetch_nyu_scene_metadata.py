"""Read authoritative MATLAB scene references with a bounded HTTP range cache."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# An optional workspace-only dependency directory avoids changing the host environment.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data/tools_python312'))
import argparse
import hashlib
import json
import re
import h5py
from src.http_range import HTTPRangeReader,CachedRangeReader
from src.research_io import ROOT,output_path,now,write_json

URL='https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
SIZE=2972037809

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.limit<0: raise ValueError('Nonnegative limit required')
    out=output_path(ROOT/'outputs/P0_nyu_scenes_v1');out.mkdir(parents=True,exist_ok=True)
    reader=HTTPRangeReader(URL,SIZE,budget=48*1024*1024,timeout=20)
    cached=CachedRangeReader(reader)
    status={'started_at':now(),'status':'running','source_url':URL,'source_bytes':SIZE,
        'whole_source_hash_verified':False,'scope':'scene metadata only; no benchmark predictions or metrics'}
    write_json(out/'summary.json',status)
    try:
        with h5py.File(cached,'r') as handle:
            refs=handle['scenes'][:].reshape(-1)
            selected=refs[:args.limit] if args.limit else refs
            records=[]
            for index,reference in enumerate(selected):
                values=handle[reference][:].reshape(-1)
                scene=''.join(chr(int(x)) for x in values)
                # Explicit room grouping requested by the experiment protocol.
                room=re.sub(r'(?<=\d)[a-zA-Z]$','',scene)
                records.append({'official_index_1based':index+1,'scene':scene,'room':room})
                if (index+1)%100==0:print('scenes extracted',index+1,'network_bytes',reader.downloaded,flush=True)
        manifest={'source_url':URL,'source_bytes':SIZE,'source_variable':'scenes',
            'official_num_images':len(refs),'grouping_rule':'remove one trailing letter after a digit',
            'records':records}
        write_json(out/'private_scene_mapping.json',manifest)
        encoded=json.dumps(records,sort_keys=True,separators=(',',':')).encode()
        status.update(status='completed' if len(records)==len(refs) else 'partial_protocol_check',
            completed_at=now(),official_num_images=len(refs),extracted_images=len(records),
            scenes=len({r['scene'] for r in records}),rooms=len({r['room'] for r in records}),
            extracted_mapping_sha256=hashlib.sha256(encoded).hexdigest(),
            dsine_correspondence_verified=False)
    except Exception as error:
        status.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        status.update(network_bytes=reader.downloaded,range_requests=reader.requests)
        write_json(out/'summary.json',status);print(json.dumps(status),flush=True)

if __name__=='__main__':main()
