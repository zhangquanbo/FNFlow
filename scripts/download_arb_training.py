"""License-filtered, stratified multi-archive sampling with bounded range reads."""
import sys
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import defaultdict
import gzip,hashlib,json,struct,time,urllib.request,zipfile,zlib
from src.http_range import HTTPRangeReader
from src.public_http import public_urlopen
from src.research_io import ROOT,output_path,write_json,now,sha256

REV='329f4fc199498264390df36104a3326e8a30c325'
OUT=output_path(ROOT/'outputs/P1_arb_acquisition_v1')
DATA=output_path(ROOT/'data/arb_training_raw_v1')

def digest(text):return hashlib.sha256(text.encode()).hexdigest()

def progress_status(value):
    # Some SFTP drive providers cannot repeatedly replace the same open/cache
    # entry. Immutable receipts remain authoritative; telemetry must not abort
    # a successful asset download.
    try:write_json(OUT/'status.json',value)
    except OSError as error:print('telemetry write failed',repr(error),flush=True)

def bounded_fetch(url,cap):
    with public_urlopen(url,timeout=40) as response:
        raw=response.read(cap+1)
    if len(raw)>cap:raise RuntimeError('Metadata cap exceeded')
    return raw

def read_zip_member(reader,info):
    if info.file_size>32*1024*1024 or info.compress_size>32*1024*1024:raise RuntimeError('Member cap exceeded')
    # Prefetch the local header and payload together: one HTTP request for normal members.
    reader.seek(info.header_offset)
    raw=reader.read(min(info.compress_size+4096,reader.size-info.header_offset))
    if raw[:4]!=b'PK\x03\x04':raise RuntimeError('Invalid ZIP local header')
    header=struct.unpack('<4s5H3I2H',raw[:30]);flags=header[2];method=header[3]
    if flags&1:raise RuntimeError('Encrypted archive member refused')
    name_size,extra_size=header[-2:];start=30+name_size+extra_size
    name=raw[30:30+name_size].decode('utf-8' if flags&0x800 else 'cp437')
    if name!=info.filename:raise RuntimeError('ZIP directory and local name differ')
    if len(raw)<start+info.compress_size:
        reader.seek(info.header_offset+start)
        payload=reader.read(info.compress_size)
    else:payload=raw[start:start+info.compress_size]
    if method==0:decoded=payload
    elif method==8:decoded=zlib.decompress(payload,-15)
    else:raise RuntimeError('Unsupported ZIP compression')
    if len(decoded)!=info.file_size or zlib.crc32(decoded)&0xffffffff!=info.CRC:
        raise RuntimeError('ZIP member size/CRC mismatch')
    return decoded

def archive_job(entry,phase):
    archive=entry['path'];stem=PurePosixPath(archive).stem
    folder=OUT/stem;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/f'{phase}_complete.json'
    if marker.exists():return json.loads(marker.read_text(encoding='utf-8'))
    url=f'https://huggingface.co/datasets/lizb6626/Arb-Objaverse/resolve/{REV}/{archive}'
    reader=HTTPRangeReader(url,entry['size'],budget=3*1024**3,max_request=40*1024**2,timeout=40,opener=public_urlopen)
    archive_index=folder/'index_private.json'
    if archive_index.exists():
        infos=[]
        for value in json.loads(archive_index.read_text(encoding='utf-8'))['members']:
            info=zipfile.ZipInfo(value['filename'])
            for key in ['header_offset','file_size','compress_size','CRC','compress_type']:setattr(info,key,value[key])
            infos.append(info)
    else:
        with zipfile.ZipFile(reader) as z:
            infos=z.infolist()
        write_json(archive_index,{'archive':archive,'revision':REV,'members':[
            {key:getattr(i,key) for key in ['filename','header_offset','file_size','compress_size','CRC','compress_type']} for i in infos if not i.is_dir()]})
    metadata_file=folder/'upstream_metadata.json.gz'
    if not metadata_file.exists():
        raw=bounded_fetch(f'https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{stem}.json.gz',16*1024**2)
        metadata_file.write_bytes(raw)
    with gzip.open(metadata_file,'rt',encoding='utf-8') as handle:metadata=json.load(handle)
    groups=defaultdict(dict)
    for info in infos:
        path=PurePosixPath(info.filename)
        if path.is_absolute() or '..' in path.parts:raise RuntimeError('Unsafe archive path')
        if len(path.parts)>1:groups[str(path.parent)][path.name]=info
    candidates=[]
    for asset,members in groups.items():
        uid=PurePosixPath(asset).name;license=metadata.get(uid,{}).get('license')
        if license not in ['by','cc0']:continue
        wanted=['camera.json']+[n for v in [0,4,8] for n in [f'normal_{v:02d}.exr',f'color_{int(digest(uid+str(v))[:8],16)%7}_{v:02d}.png']]
        if all(n in members for n in wanted):candidates.append((asset,wanted))
    candidates.sort(key=lambda x:digest('arb-round1-v1:'+x[0]))
    if len(candidates)<125:raise RuntimeError(f'{stem}: fewer than 125 eligible complete assets')
    selected=candidates[:125]
    # First four across every archive form a 64-asset preflight. Formal repeats the
    # same selection and reuses files; no ranking depends on a pilot outcome.
    current=selected[:4] if phase=='pilot' else selected
    records=[]
    for asset,wanted in current:
        uid=PurePosixPath(asset).name;members=groups[asset]
        dest=output_path(DATA/stem/uid);dest.mkdir(parents=True,exist_ok=True)
        receipt=dest/'receipt_private.json'
        if receipt.exists():
            record=json.loads(receipt.read_text(encoding='utf-8'))
            for item in record['files']:
                file=dest/item['name']
                if not file.exists() or sha256(file)!=item['sha256']:raise RuntimeError('Cached asset hash changed')
            records.append(record);continue
        files=[]
        for name in wanted:
            info=members[name]
            path=dest/name
            if path.exists() and path.stat().st_size==info.file_size and zlib.crc32(path.read_bytes())&0xffffffff==info.CRC:
                raw=path.read_bytes()
            else:
                last_error=None
                for attempt in range(3):
                    try:raw=read_zip_member(reader,info);last_error=None;break
                    except (OSError,TimeoutError) as error:last_error=error;time.sleep(attempt+1)
                if last_error:raise last_error
                path.write_bytes(raw)
            files.append({'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        ref=metadata[uid]
        record={'asset':uid,'archive':archive,'folder':dest.relative_to(ROOT).as_posix(),'files':files,
            'license':ref['license'],'author':ref.get('user',{}).get('displayName'),
            'source_url':ref.get('viewerUrl'),'material_normal_texture':'unknown',
            'label_source':'rendered_bsdf_normal_pass','K_valid':False}
        write_json(receipt,record);records.append(record)
        print(phase,stem,'assets',len(records),'network_MB',round(reader.downloaded/1e6,1),flush=True)
    result={'archive':archive,'records':records,'network_bytes':reader.downloaded,'range_requests':reader.requests}
    write_json(marker,result);return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    tree=json.loads((ROOT/'outputs/P0_data_intake_local_v1/arb_data_tree.json').read_text(encoding='utf-8'))
    tree=sorted(tree,key=lambda x:x['path'])
    selected=[min(tree[start:start+10],key=lambda x:digest('archive-stratum-v1:'+x['path'])) for start in range(0,160,10)]
    write_json(OUT/'selection_private.json',{'revision':REV,'archives':selected,'target_assets':2000,
        'selection':'16 equal strata over 160 archive paths, hash selection within strata; 125 license-filtered assets per archive'})
    try:
        for phase in ['pilot','formal']:
            progress_status({'status':'running','phase':phase,'updated_at':now()})
            results=[]
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs=[pool.submit(archive_job,row,phase) for row in selected]
                for job in as_completed(jobs):
                    try:results.append(job.result())
                    except Exception as error:
                        progress_status({'status':'failed','phase':phase,'updated_at':now(),'error':repr(error)})
                        print('archive worker failed',repr(error),flush=True)
                        raise
                    progress_status({'status':'running','phase':phase,'completed_archives':len(results),
                        'target_archives':len(selected),'completed_assets':sum(len(r['records']) for r in results),'updated_at':now()})
            records=[r for group in results for r in group['records']]
            if len({r['asset'] for r in records})!=len(records):raise RuntimeError('Duplicate UIDs across archives')
            records.sort(key=lambda r:digest('asset-split-v1:'+r['asset']))
            nval=8 if phase=='pilot' else 200
            for i,record in enumerate(records):record['split']='val' if i<nval else 'train'
            write_json(OUT/f'{phase}_manifest_private.json',{'records':records,'revision':REV,'phase':phase})
            write_json(OUT/f'{phase}_ready.json',{'status':'completed','phase':phase,'assets':len(records),
                'archives':len(results),'validation_assets':nval,'created_at':now(),'network_bytes':sum(r['network_bytes'] for r in results)})
        progress_status({'status':'completed','phase':'formal','updated_at':now(),'assets':2000})
    except Exception as error:
        progress_status({'status':'failed','updated_at':now(),'error':type(error).__name__+': '+str(error)})
        raise
if __name__=='__main__':main()
