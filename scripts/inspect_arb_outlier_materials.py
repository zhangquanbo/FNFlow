"""Bounded source-material inspection for visual outliers; no normal-map causality claim."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,urllib.request,struct,re
from src.public_http import public_urlopen
from src.http_range import HTTPRangeReader
from src.research_io import ROOT,write_json,sha256,now


def main():
    out=ROOT/'outputs/P1_arb_outlier_visual_review3'
    rows=json.loads((out/'details_private.json').read_text())['records']
    assets={r['asset']:r for r in json.loads((ROOT/'outputs/P1_arb_acquisition_v1/pilot_manifest_private.json').read_text())['records']}
    results=[]
    for row in rows:
        uid=row['asset'];r=assets[uid];target=out/f'{uid}_gltf_private.json'
        if target.exists():doc=json.loads(target.read_text())
        else:
            url=f'https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/{Path(r["archive"]).stem}/{uid}.glb'
            req=urllib.request.Request(url+'?review3_header=1',headers={'Range':'bytes=0-19','Accept-Encoding':'identity'})
            with public_urlopen(req,timeout=45) as response:
                header=response.read(21);match=re.fullmatch(r'bytes 0-19/(\d+)',response.headers.get('Content-Range',''))
                if response.status!=206 or not match or len(header)!=20:raise RuntimeError('Exact GLB range refused')
            magic,version,total,count,kind=struct.unpack('<4s4I',header)
            if magic!=b'glTF' or version!=2 or total!=int(match[1]) or kind!=0x4e4f534a:raise RuntimeError('GLB header mismatch')
            reader=HTTPRangeReader(url,total,budget=8*1024**2,max_request=8*1024**2,opener=public_urlopen)
            reader.seek(20);doc=json.loads(reader.read(count));write_json(target,doc)
        materials=[]
        for m in doc.get('materials',[]):
            materials.append({'doubleSided':m.get('doubleSided',False),'alphaMode':m.get('alphaMode','OPAQUE'),
                'normalTexture':m.get('normalTexture'),'extensions':m.get('extensions',{}),
                'baseColorFactor':m.get('pbrMetallicRoughness',{}).get('baseColorFactor',[1,1,1,1])})
        results.append({'asset':uid,'group':row['group'],'materials':materials,'extensionsUsed':doc.get('extensionsUsed',[]),
            'source_document_sha256':sha256(target)})
        print(json.dumps(results[-1]),flush=True)
    write_json(out/'material_details_private.json',{'records':results,'created_at':now()})


if __name__=='__main__':main()
