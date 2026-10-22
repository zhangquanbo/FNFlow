"""Read only GLB JSON chunks; texture presence is a label group, not calibrated q."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,struct,urllib.request,hashlib,re
from collections import Counter
from src.http_range import HTTPRangeReader
from src.research_io import ROOT,write_json,output_path,now

def material_summary(document):
    counts=Counter();modes=Counter()
    def walk(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key.lower().endswith('normaltexture') and isinstance(child,dict):counts['normal_texture_slots']+=1
                walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    for material in document.get('materials',[]):walk(material);modes[material.get('alphaMode','OPAQUE')]+=1
    return {'materials':len(document.get('materials',[])),'explicit_normal_texture_slots':counts['normal_texture_slots'],
        'normal_texture_group':'present' if counts['normal_texture_slots'] else 'not_declared_in_gltf',
        'alpha_modes':dict(modes),'meshes_with_vertex_normals':sum('NORMAL' in p.get('attributes',{}) for m in document.get('meshes',[]) for p in m.get('primitives',[])),
        'interpretation':'absence is a near-geometry candidate flag, not proof that all Blender bump/displacement or mesh smoothing effects are absent'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=16);args=parser.parse_args()
    base=ROOT/'outputs/P1_arb_acquisition_v1'
    receipts=sorted((ROOT/'data/arb_training_raw_v1').rglob('receipt_private.json'))[:args.limit]
    if not receipts:raise RuntimeError('No acquired assets')
    out=output_path(ROOT/'outputs/P1_gltf_normal_textures_v1');out.mkdir(parents=True,exist_ok=True)
    records=[]
    for receipt in receipts:
        r=json.loads(receipt.read_text());uid=r['asset'];stem=Path(r['archive']).stem
        url=f'https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/{stem}/{uid}.glb'
        try:
            request=urllib.request.Request(url+'?research_glb_header=1',headers={'Range':'bytes=0-19','Accept-Encoding':'identity'})
            with urllib.request.urlopen(request,timeout=30) as response:
                header=response.read(21);cr=response.headers.get('Content-Range','')
                match=re.fullmatch(r'bytes 0-19/(\d+)',cr)
                if response.status!=206 or len(header)!=20 or not match:raise RuntimeError('GLB header exact range refused')
            magic,version,length,json_size,chunk_type=struct.unpack('<4s4I',header)
            if magic!=b'glTF' or version!=2 or chunk_type!=0x4e4f534a or int(match[1])!=length:raise RuntimeError('Unexpected GLB header')
            if json_size>8*1024**2:raise RuntimeError('GLB metadata exceeds cap')
            reader=HTTPRangeReader(url,length,budget=8*1024**2,max_request=8*1024**2)
            reader.seek(20);raw=reader.read(json_size);document=json.loads(raw)
            record={'asset':uid,'archive':r['archive'],'source_url':url,'source_glb_bytes':length,
                'json_sha256':hashlib.sha256(raw).hexdigest(),'full_glb_downloaded':False,
                'status':'completed',**material_summary(document)}
            write_json(out/f'{uid}_private.json',record)
        except Exception as error:record={'asset':uid,'status':'failed','error':type(error).__name__+': '+str(error)}
        records.append(record);print('GLB JSON inspected',len(records),record['status'],flush=True)
    write_json(out/'private_details.json',{'records':records})
    summary={'status':'completed','created_at':now(),'requested_assets':len(records),
        'successful_assets':sum(r['status']=='completed' for r in records),
        'failed_assets':sum(r['status']=='failed' for r in records),
        'normal_texture_groups':dict(Counter(r['normal_texture_group'] for r in records if r['status']=='completed')),
        'scope':'bounded convenience sample of already acquired training assets; not a population estimate',
        'reliability_q_calibrated':False,'blocks_first_round':False,
        'blender_procedural_bump_or_displacement_verified':False}
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
