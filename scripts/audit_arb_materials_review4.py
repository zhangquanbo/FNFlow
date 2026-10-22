"""Pinned GLB-JSON-only census over the unchanged planned 2,000 Arb assets."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,struct,re,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from src.http_range import HTTPRangeReader
from src.public_http import public_urlopen
from src.research_io import ROOT,write_json,sha256,now
from src.material_protocol import material_flags
from scripts.prepare_arb_training import fixed_formal_partition

OUT=ROOT/'outputs/P1_arb_materials_review4'


def inspect(uid,stem,split,cfg):
    folder=OUT/'private_assets';folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/f'{uid}.json';document_path=folder/f'{uid}.gltf'
    if receipt.exists():
        row=json.loads(receipt.read_text())
        if row['source_revision']!=cfg['source_revision'] or row['document_sha256']!=sha256(document_path):raise RuntimeError('Material source cache mismatch')
        if any(row.get(k)!=v for k,v in material_flags(json.loads(document_path.read_bytes())).items()):
            raise RuntimeError('Cached material classification differs from current declared rule')
        return row
    url=f'https://huggingface.co/datasets/allenai/objaverse/resolve/{cfg["source_revision"]}/glbs/{stem}/{uid}.glb'
    last=None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url+'?material_review4_header=1',headers={'Range':'bytes=0-19','Accept-Encoding':'identity'})
            with public_urlopen(req,timeout=45) as response:
                raw=response.read(21);match=re.fullmatch(r'bytes 0-19/(\d+)',response.headers.get('Content-Range',''))
                if response.status!=206 or not match or len(raw)!=20:raise RuntimeError('Exact GLB header refused')
            magic,version,total,count,kind=struct.unpack('<4s4I',raw)
            if magic!=b'glTF' or version!=2 or total!=int(match[1]) or kind!=0x4e4f534a or count>cfg['maximum_json_bytes_per_asset']:raise RuntimeError('GLB JSON metadata cap/schema mismatch')
            reader=HTTPRangeReader(url,total,budget=count+64,max_request=cfg['maximum_json_bytes_per_asset'],opener=public_urlopen,timeout=45)
            reader.seek(20);document=reader.read(count);flags=material_flags(json.loads(document))
            document_path.write_bytes(document)
            row={'asset':uid,'split':split,'source_revision':cfg['source_revision'],'document_sha256':sha256(document_path),
                'source_url':url,'network_bytes':count+20,**flags}
            write_json(receipt,row);return row
        except OSError as e:last=e;time.sleep(1+attempt)
    raise last


def main():
    cfg=json.loads((ROOT/'configs/arb_material_audit_review4.json').read_text());plan=fixed_formal_partition()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists() and json.loads((OUT/'summary.json').read_text()).get('status')=='completed':
        from src.material_protocol import load_material_audit
        load_material_audit(ROOT);return
    mapping={}
    for folder in (ROOT/'outputs/P1_arb_acquisition_v1').glob('000-*'):
        p=folder/'index_private.json'
        if p.exists():
            for r in json.loads(p.read_text())['members']:
                parts=Path(r['filename']).parts
                if len(parts)>1 and parts[-2] in plan:
                    uid=parts[-2]
                    if uid in mapping and mapping[uid]!=folder.name:raise RuntimeError('Asset has ambiguous archive mapping')
                    mapping[uid]=folder.name
    if set(mapping)!=set(plan):raise RuntimeError('Material census must cover all 2000 planned UIDs')
    rows=[];failures=[]
    def validation_ready():
        selected=[r for r in rows if r['split']=='val'];path=OUT/'validation_ready.json'
        if len(selected)==200 and not path.exists():
            index=OUT/'validation_material_index_private.json'
            write_json(index,{'assets':{r['asset']:r for r in sorted(selected,key=lambda r:r['asset'])},'source_revision':cfg['source_revision']})
            write_json(path,{'status':'completed','classified_assets':200,'failed_assets':0,
                'material_index_sha256':sha256(index),'created_at':now(),'training_filter_enabled':False})
    ordered=sorted(plan,key=lambda u:(plan[u]!='val',u))
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs={pool.submit(inspect,uid,mapping[uid],plan[uid],cfg):uid for uid in ordered}
        for future in as_completed(jobs):
            try:rows.append(future.result())
            except Exception as e:
                failures.append({'asset':jobs[future],'error':repr(e)})
                write_json(OUT/'failures_private.json',failures)
            validation_ready()
            write_json(OUT/'status.json',{'status':'running','classified_assets':len(rows),'failed_assets':len(failures),'target':2000,'updated_at':now()})
            if (len(rows)+len(failures))%25==0:print('material metadata',len(rows),'failed',len(failures),flush=True)
    write_json(OUT/'failures_private.json',failures)
    write_json(OUT/'material_index_private.json',{'assets':{r['asset']:r for r in sorted(rows,key=lambda r:r['asset'])},'source_revision':cfg['source_revision']})
    def group(values):
        n=len(values);flag=sum(r['flagged'] for r in values)
        return {'classified_assets':n,'flagged_assets':flag,'flagged_fraction':flag/n if n else None,
            'blend_assets':sum(r['blend'] for r in values),'positive_transmission_assets':sum(r['transmission'] for r in values),
            'referenced_material_flagged_assets':sum(r['referenced_material_flagged'] for r in values)}
    result={'status':'completed' if not failures and len(rows)==2000 else 'incomplete','created_at':now(),
        'target_assets':2000,'classified_assets':len(rows),'failed_assets':len(failures),'source_revision':cfg['source_revision'],
        'all':group(rows),'by_split':{s:group([r for r in rows if r['split']==s]) for s in ['train','val']},
        'material_index_sha256':sha256(OUT/'material_index_private.json'),'metadata_only':True,'training_changed':False,
        'classification':'any declared material; additionally count referenced material flags; not proof of rendered transparency or RGB-label mismatch',
        'unknown_assets_never_counted_as_clean':True}
    write_json(OUT/'summary.json',result);write_json(OUT/'status.json',result);print(json.dumps(result),flush=True)
    if result['status']!='completed':raise RuntimeError('Incomplete material census, preserve failed records')


if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(OUT/'status.json',{'status':'failed','error':repr(e),'updated_at':now()});raise
