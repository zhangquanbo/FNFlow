"""Inspect a pinned archive and acquire a bounded, deterministic protocol sample."""
import sys
from pathlib import Path, PurePosixPath
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import hashlib
import json
import zipfile
from src.http_range import HTTPRangeReader
from src.research_io import ROOT, now, output_path, sha256, write_json

REVISION='329f4fc199498264390df36104a3326e8a30c325'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tree',type=Path,required=True)
    parser.add_argument('--archive',default='data/000-000.zip')
    parser.add_argument('--assets',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.assets<=5: raise ValueError('Protocol sample limited to 1–5 assets')
    entries=json.loads(args.tree.read_text())
    entry=next(row for row in entries if row['path']==args.archive)
    out=output_path(ROOT/'outputs/P0_arb_sample_v1');out.mkdir(parents=True,exist_ok=True)
    target=output_path(ROOT/'data/arb_protocol_raw_v1')
    url=f'https://huggingface.co/datasets/lizb6626/Arb-Objaverse/resolve/{REVISION}/{args.archive}'
    reader=HTTPRangeReader(url,entry['size'])
    status={'started_at':now(),'status':'running','revision':REVISION,'archive':args.archive,
            'archive_bytes':entry['size'],'archive_sha256':entry['lfs']['oid'],
            'archive_hash_verified':False,'scope':'bounded protocol sample; not a training-ready dataset'}
    write_json(out/'summary.json',status)
    try:
        with zipfile.ZipFile(reader) as archive:
            infos=archive.infolist()
            by_asset={}
            for info in infos:
                path=PurePosixPath(info.filename)
                if info.is_dir() or '..' in path.parts or path.is_absolute(): continue
                if len(path.parts)>1:
                    by_asset.setdefault(str(path.parent),[]).append(info)
            candidates=[key for key,items in by_asset.items() if any(PurePosixPath(i.filename).name=='camera.json' for i in items)]
            candidates.sort(key=lambda x:hashlib.sha256(('arb-protocol-v1:'+x).encode()).digest())
            chosen=candidates[:args.assets]
            if len(chosen)!=args.assets: raise RuntimeError('Insufficient asset directories with cameras')
            write_json(out/'archive_index.json',{'revision':REVISION,'archive':args.archive,
                'files':[{'path':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':i.CRC} for i in infos]})
            records=[]
            wanted={'camera.json','lighting.json'}
            for view in (0,3):
                wanted.update({f'normal_{view:02d}.exr',f'albedo_{view:02d}.png',
                               f'color_0_{view:02d}.png',f'color_1_{view:02d}.png'})
            for asset in chosen:
                names={PurePosixPath(i.filename).name for i in by_asset[asset]}
                missing=sorted(wanted-names)
                asset_records=[]
                for info in by_asset[asset]:
                    if PurePosixPath(info.filename).name not in wanted: continue
                    if info.file_size>32*1024*1024: raise RuntimeError('Member exceeds sample cap')
                    dest=output_path(target/args.archive.replace('/','_').removesuffix('.zip')/info.filename)
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    # ZipFile.read validates decompression and member CRC32.
                    raw=archive.read(info)
                    if dest.exists() and dest.read_bytes()!=raw: raise RuntimeError('Existing sample differs')
                    dest.write_bytes(raw)
                    asset_records.append({'path':dest.relative_to(ROOT).as_posix(),'bytes':len(raw),
                                          'sha256':sha256(dest),'zip_crc32_verified':True})
                records.append({'asset':asset,'missing_requested_files':missing,
                    'depth_files':[n for n in names if 'depth' in n.lower()],
                    'mask_files':[n for n in names if 'mask' in n.lower()],
                    'all_file_count':len(names),'files':asset_records})
                print('sampled asset',len(records),'files',len(asset_records),flush=True)
            write_json(out/'private_manifest.json',{'records':records})
            status.update(status='completed',completed_at=now(),archive_members=len(infos),
                archive_assets=len(candidates),sampled_assets=len(records),
                requested_missing_files=sum(len(r['missing_requested_files']) for r in records),
                sampled_depth_files=sum(len(r['depth_files']) for r in records),
                sampled_explicit_mask_files=sum(len(r['mask_files']) for r in records),
                sample_files=sum(len(r['files']) for r in records),
                sample_bytes=sum(f['bytes'] for r in records for f in r['files']),
                archive_suffix_counts=dict(Counter(PurePosixPath(i.filename).suffix for i in infos if not i.is_dir())))
    except Exception as error:
        status.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        status.update(network_bytes=reader.downloaded,range_requests=reader.requests)
        write_json(out/'summary.json',status)
        print(json.dumps(status),flush=True)

if __name__=='__main__': main()
