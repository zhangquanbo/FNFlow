"""Freeze one-light controls and lights21..30 using original raw RGB and existing GT contracts."""
from pathlib import Path
import cv2
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.http_range import HTTPRangeReader
from src.research_io import write_json, sha256
OUT = ROOT / 'data/external_round_20261106'
AB = ROOT / 'outputs/P3_object_dino_evaluation_fresh_20261027/manifests_private'
STATUS = ROOT / 'outputs/external_round_20261106/data_status.json'

def read(path):
    return json.loads(path.read_text())

def save_rgb(record, raw, light, source):
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None and decoded.dtype in (np.uint8, np.uint16)
    decoded = decoded[..., ::-1]
    rgb = np.rint(decoded.astype(np.float64) / (65535 if decoded.dtype == np.uint16 else 255) * 255).clip(0, 255).astype(np.uint8)
    mask = np.asarray(Image.open(record['rgba']))[..., 3] > 0
    assert rgb.shape[:2] == mask.shape
    key = f"{record['domain']}/{record['object']}/{light:02d}.png"
    uid = hashlib.sha256(key.encode()).hexdigest()
    folder = OUT / 'images' / uid
    folder.mkdir(parents=True, exist_ok=True)
    white = folder / 'white.png'
    rgba = folder / 'rgba.png'
    Image.fromarray(np.where(mask[..., None], rgb, 255).astype(np.uint8)).save(white)
    Image.fromarray(np.dstack([rgb, mask.astype(np.uint8)*255])).save(rgba)
    row = dict(record)
    row.update(id=uid, image_key=key, lighting_id=light, white=str(white), rgba=str(rgba),
               ours_rgb=str(white), nlr_rgba=str(rgba), ours_rgb_sha256=sha256(white),
               nlr_rgba_sha256=sha256(rgba), source_rgb_sha256=hashlib.sha256(raw).hexdigest(),
               source_rgb_member=source)
    return row

def publish(domain, records):
    assert len(records) == (100 if domain == 'diligent' else 140)
    assert all(sorted(x['lighting_id'] for x in records if x['object'] == obj) == list(range(21,31)) for obj in {x['object'] for x in records})
    write_json(OUT / ('rose10_'+domain+'.json'), {'records': sorted(records,key=lambda x:x['id']),
        'protocol': 'Lights21..30, one independent monocular prediction per RGB, images within object then object macro.',
        'input_mapping': 'Native uint16/65535 rounded to8bit; no per-image exposure, gamma, intensity division or crop. Fixed official-mask background and previously verified GT conversion.',
        'source_protocol': 'https://arxiv.org/html/2602.09929v5#A3', 'test_GSO_opened': False})
    write_json(STATUS, {'phase': domain+'_ready', 'images':len(records), 'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('REAL_PROTOCOL_READY',domain,len(records),flush=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    singles={}
    for domain in ['diligent','luces']:
        singles[domain]=read(AB/(domain+'.json'))['records']
        path=OUT/('single_'+domain+'.json')
        if not path.exists():write_json(path,{'records':singles[domain],'source_manifest_sha256':sha256(AB/(domain+'.json')),'protocol':'Unchanged fixed lowest-ID one-light control'})
    primary=OUT/'gso_primary.json'
    if not primary.exists():write_json(primary,read(AB/'primary.json'))
    write_json(STATUS,{'phase':'preparing_ten_lights','updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    if not (OUT/'rose10_luces.json').exists():
        intake=ROOT/'data/real_object_selected_v1/luces'
        archive=Path(read(intake/'archive_index_private.json')['archive'])
        cache=OUT/'temporary_luces_inner.zip'
        if not cache.exists() or cache.stat().st_size!=12123247159:
            partial=cache.with_suffix('.partial')
            with tarfile.open(archive,'r:gz') as outer,outer.extractfile('LUCES/Luces_code.zip') as source,partial.open('wb') as dest:
                shutil.copyfileobj(source,dest,16*1024**2)
            assert partial.stat().st_size==12123247159
            partial.replace(cache)
        assert sha256(cache)==read(intake/'nested_zip_receipt.json')['sha256']
        index=read(intake/'parts_index_private.json')['entries']
        wanted={f"Luces_25082022/data/{row['object']}/{light:02d}.png":(row,light) for row in singles['luces'] for light in range(21,31)}
        by_part={}
        for entry in index:
            if entry['name'] in wanted:by_part.setdefault(entry['part'],[]).append(entry['name'])
        assert sum(map(len,by_part.values()))==140
        records=[]
        with zipfile.ZipFile(cache) as inner:
            for part,members in sorted(by_part.items()):
                payload=inner.read(part)
                with zipfile.ZipFile(io.BytesIO(payload)) as source:
                    for name in members:
                        row,light=wanted[name]
                        records.append(save_rgb(row,source.read(name),light,name))
                del payload
                print('PREPARED_LUCES_PART',part,len(records),flush=True)
        publish('luces',records)
        cache.unlink()
    if not (OUT/'rose10_diligent.json').exists():
        intake=ROOT/'data/real_object_selected_v1/diligent'
        length=read(intake/'head_receipt.json')['content_length']
        url='https://drive.usercontent.google.com/download?id=1EgC3x8daOWL4uQmc6c4nXVe4mdAMJVfg&export=download&confirm=t'
        rr=HTTPRangeReader(url,length,budget=512*1024**2,max_request=16*1024**2,timeout=40)
        records=[]
        with zipfile.ZipFile(rr) as archive:
            names=archive.namelist()
            for row in singles['diligent']:
                for light in range(21,31):
                    suffix=f"/pmsData/{row['object']}/{light:03d}.png"
                    name=next(n for n in names if ('/'+n).endswith(suffix))
                    raw=archive.read(name)
                    records.append(save_rgb(row,raw,light,name))
                print('PREPARED_DILIGENT_OBJECT',row['object'],flush=True)
        publish('diligent',records)
    write_json(STATUS,{'phase':'completed','single_images':24,'ten_light_images':240,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})

if __name__=='__main__':
    try:main()
    except Exception as error:
        write_json(STATUS,{'phase':'failed','error':str(error),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        raise
