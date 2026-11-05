"""Grow usable training assets only; preserve original broad view/light sampling."""
import sys
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import os,json,gzip,hashlib,zipfile,zlib,shutil,time,re,threading,fcntl
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'

import cv2
import numpy as np
from PIL import Image
from scripts.download_arb_training import REV,read_zip_member,bounded_fetch,digest
from src.http_range import HTTPRangeReader
from src.public_http import public_urlopen
from src.object_normals import decode_arb_normal
from src.research_io import write_json,sha256,now

OUT=ROOT/'outputs/P3_arb_scale_c_fresh_20261026';RAW=ROOT/'data/arb_scale_c_raw_fresh_20261026';PREP=ROOT/'data/arb_scale_c_prepared_fresh_20261026'
RESERVE=8*1024**3
def read(p):return json.loads(Path(p).read_text())
def retry(fn):
    for attempt in range(4):
        try:return fn()
        except (OSError,TimeoutError) as e:
            if attempt==3:raise
            print('network_retry',attempt+1,type(e).__name__,flush=True);time.sleep(2**attempt)
def reader(entry,budget=128*1024**2):
    return HTTPRangeReader(f'https://huggingface.co/datasets/lizb6626/Arb-Objaverse/resolve/{REV}/{entry["path"]}',entry['size'],budget=budget,max_request=40*1024**2,timeout=55,opener=public_urlopen)
def names(uid):return ['camera.json','lighting.json']+[n for v in [0,4,8] for n in [f'normal_{v:02d}.exr',f'color_{int(digest(uid+str(v))[:8],16)%7}_{v:02d}.png']]
def info_from(item):
    i=zipfile.ZipInfo(item['filename'])
    for k in ['header_offset','file_size','compress_size','CRC','compress_type']:setattr(i,k,item[k])
    return i

def index_archive(entry):
    stem=PurePosixPath(entry['path']).stem;old=ROOT/'outputs/P1_arb_acquisition_v1'/stem;target=OUT/'archive_indices'/stem;target.mkdir(parents=True,exist_ok=True)
    ip=old/'index_private.json' if (old/'index_private.json').exists() else target/'index_private.json'
    if not ip.exists():
        def fetch_index():
            rr=reader(entry)
            with zipfile.ZipFile(rr) as z:return [{k:getattr(i,k) for k in ['filename','header_offset','file_size','compress_size','CRC','compress_type']} for i in z.infolist() if not i.is_dir()]
        write_json(ip,{'revision':REV,'archive':entry['path'],'members':retry(fetch_index)})
    entries=read(ip)['members']
    mp=old/'upstream_metadata.json.gz' if (old/'upstream_metadata.json.gz').exists() else target/'upstream_metadata.json.gz'
    if not mp.exists():mp.write_bytes(retry(lambda:bounded_fetch(f'https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{stem}.json.gz',16*1024**2)))
    with gzip.open(mp,'rt',encoding='utf-8') as f:metadata=json.load(f)
    groups=defaultdict(dict)
    for item in entries:
        path=PurePosixPath(item['filename']);assert not path.is_absolute() and '..' not in path.parts
        if len(path.parts)>1:groups[str(path.parent)][path.name]=item
    candidates=[]
    for asset,members in groups.items():
        uid=PurePosixPath(asset).name;ref=metadata.get(uid,{})
        if ref.get('license') not in ['by','cc0'] or not all(n in members for n in names(uid)):continue
        candidates.append({'asset':uid,'archive':entry,'asset_member_folder':asset,'members':[members[n] for n in names(uid)],
          'license':ref['license'],'name':ref.get('name',''),'vertexCount':ref.get('vertexCount'),'faceCount':ref.get('faceCount'),
          'source_url':ref.get('viewerUrl'),'metadata_sha256':sha256(mp)})
    candidates.sort(key=lambda x:digest('arb-scale-c-v2:'+x['asset']))
    return {'archive':entry['path'],'index_sha256':sha256(ip),'metadata_sha256':sha256(mp),'candidates':candidates[:350]}

def plan():
    path=OUT/'plan_private.json'
    if path.exists():return read(path)
    tree=sorted(read(ROOT/'outputs/P0_data_intake_local_v1/arb_data_tree.json'),key=lambda x:x['path'])
    original=read(ROOT/'outputs/P1_arb_acquisition_v1/selection_private.json')['archives'];oldpaths={x['path'] for x in original}
    selected=[]
    for start in range(0,160,10):
        block=tree[start:start+10];base=[x for x in block if x['path'] in oldpaths];assert len(base)==1
        extra=sorted((x for x in block if x['path'] not in oldpaths),key=lambda x:digest('c-archive-scale-v2:'+x['path']))[:3]
        selected+=base+extra
    write_json(OUT/'archive_selection_private.json',{'revision':REV,'archives':selected,'rule':'64 archives: original16 plus3 more per original ten-archive stratum; no camera or score filtering'})
    groups=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,group in enumerate(pool.map(index_archive,selected)):
            groups.append(group);write_json(OUT/'status.json',{'phase':'metadata_indexing','completed_archives':i+1,'total_archives':64,'updated_at':now()})
    oldmanifest=read(ROOT/'data/arb_mixed_formal_v1/private_manifest.json')['records']
    train={r['scene'] for r in oldmanifest if r['split']=='train'};val={r['scene'] for r in oldmanifest if r['split']=='val'}
    # Also exclude planned heldout UIDs with no usable labels, not just the 198 usable validation assets.
    from scripts.prepare_arb_training import fixed_formal_partition
    partition=fixed_formal_partition();val|={uid for uid,split in partition.items() if split=='val'}
    known_empty={uid for uid,split in partition.items() if split=='train'}-train
    gso=read(ROOT/'outputs/P1_gso_intake_v1/manifest_private.json')['records']
    normalize=lambda name:re.sub('[^a-z0-9]','',name.lower())
    gso_names={normalize(r['name']) for r in gso};gso_counts={(r['vertices'],r['triangles']) for r in gso}
    candidates=[];excluded=[];seen=set(train)|val|known_empty
    for rank in range(max(len(g['candidates']) for g in groups)):
        for group in groups:
            if rank>=len(group['candidates']):continue
            item=group['candidates'][rank];uid=item['asset']
            if uid in seen:continue
            seen.add(uid)
            reason=None
            if normalize(item['name']) in gso_names:reason='normalized_source_name_matches_GSO'
            elif (item['vertexCount'],item['faceCount']) in gso_counts:reason='conservative_GSO_vertex_face_count_match'
            if reason:excluded.append({'asset':uid,'reason':reason});continue
            candidates.append(item)
    assert len(candidates)>=(10000-len(train)),'Insufficient eligible candidates'
    result={'created_at':now(),'revision':REV,'target_usable_training_assets':10000,'initial_usable_assets':len(train),
      'heldout_uid_count':len(val),'candidate_order':candidates,'duplicate_screen_exclusions':excluded,
      'view_indices':[0,4,8],'pitch_quota':False,'training_started':False,
      'additional_raw_uncompressed_bytes_for_first8220':sum(sum(m['file_size'] for m in x['members']) for x in candidates[:(10000-len(train))]),
      'source_indices':[{'archive':g['archive'],'index_sha256':g['index_sha256'],'metadata_sha256':g['metadata_sha256']} for g in groups],
      'duplicate_limit':'Source names/counts and UID only; no exhaustive cross-source mesh-near-duplicate guarantee'}
    write_json(path,result);return result

def acquire(item):
    uid=item['asset'];folder=RAW/PurePosixPath(item['archive']['path']).stem/uid;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt_private.json'
    if receipt.exists():
        value=read(receipt)
        for file in value['files']:assert sha256(folder/file['name'])==file['sha256']
        for row in value['prepared_records']:assert sha256(row['path'])==row['npz_sha256']
        return value
    assert shutil.disk_usage(ROOT).free>RESERVE+32*1024**2,'Disk reserve reached; preserve partial cache'
    rr=reader(item['archive'],budget=64*1024**2);files=[]
    for member in item['members']:
        name=PurePosixPath(member['filename']).name;path=folder/name
        if path.exists() and path.stat().st_size==member['file_size'] and zlib.crc32(path.read_bytes())&0xffffffff==member['CRC']:raw=path.read_bytes()
        else:
            raw=retry(lambda:read_zip_member(rr,info_from(member)));path.write_bytes(raw)
        files.append({'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    meta=read(folder/'camera.json');prepared=[];exclusions=[]
    for v in [0,4,8]:
        exr=folder/f'normal_{v:02d}.exr';native=cv2.imread(str(exr),cv2.IMREAD_UNCHANGED)
        assert native is not None and native.shape[:2]==(512,512),'Arb native normal shape mismatch'
        normal,valid=decode_arb_normal(native)
        if not valid.any():exclusions.append({'view':v,'reason':'zero_valid_EXR_pixels'});continue
        png=folder/f'color_{int(digest(uid+str(v))[:8],16)%7}_{v:02d}.png'
        rgba=np.asarray(Image.open(png).convert('RGBA')).astype(np.float32)/255;assert rgba.shape[:2]==native.shape[:2]
        rgb=np.rint(cv2.resize(rgba[...,:3]*rgba[...,3:4]+1-rgba[...,3:4],(256,256))*255).clip(0,255).astype(np.uint8)
        target=PREP/'train'/f'{uid}_{v:02d}.npz';target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,rgb=rgb,normal=normal,valid=valid,K=np.zeros((3,3),np.float32),K_valid=np.array(False),projection_3x3=np.zeros((3,3),np.float32),projection_valid=np.array(False))
        pose=np.asarray(meta['transform_matrix'][v]);elev=float(np.degrees(np.arctan2(pose[2,3],np.hypot(*pose[:2,3]))))
        assert abs(elev-meta['elevations'][v])<1e-3
        prepared.append({'scene':uid,'split':'train','path':str(target),'image_key':f'arb/{uid}/{png.name}',
           'label_source':'rendered_bsdf_normal_pass','material_normal_texture':'unknown','valid_pixels':int(valid.sum()),
           'source_normal_sha256':sha256(exr),'source_rgb_sha256':sha256(png),'npz_sha256':sha256(target),'elevation_deg':elev})
    result={'asset':uid,'files':files,'prepared_records':prepared,'exclusions':exclusions,'license':item['license'],
      'source_url':item['source_url'],'network_bytes':rr.downloaded,'archive':item['archive']['path'],'created_at':now()}
    write_json(receipt,result);return result
