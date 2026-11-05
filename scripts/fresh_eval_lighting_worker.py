"""Fresh 256-sample CPU/OIDN light-only pair on the 93 sealed val objects."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import fresh_eval_lighting_core as core
import argparse,json,os,hashlib
import numpy as np
from src.research_io import write_json,sha256,now
OUT=ROOT/'outputs/P3_object_dino_evaluation_fresh_20261027'
def read(p):return json.loads(Path(p).read_text())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--style',choices=['original','arb_style'],required=True)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    reg=read(OUT/'preregistration.json')
    for name in ['fresh_eval_lighting_protocol.py','fresh_eval_lighting_worker.py','fresh_eval_lighting_core.py']:
        assert sha256(ROOT/'scripts'/name)==reg['source_sha256'][name]
    receipt=read(ROOT/'data/gso_lighting_inputs_fresh_20261027/receipt.json')
    assert receipt['lighting_sha256']==sha256(ROOT/'data/gso_lighting_inputs_fresh_20261027/example_lighting.json')
    for f in receipt['files']:assert sha256(ROOT/'data/gso_lighting_inputs_fresh_20261027'/f['name'])==f['sha256']
    primary=read(ROOT/'data/gso_multielevation_fresh_20261026/val/private_manifest.json')['records']
    primary={(r['scene'],int(r['image_key'].rsplit('/',1)[1])):r for r in primary if r['elevation_bin']==30}
    assert len(primary)==372
    assets=[r for r in read(ROOT/'outputs/P1_gso_intake_v1/manifest_private.json')['records'] if r['split']=='val']
    assets.sort(key=lambda r:hashlib.sha256(('gso-v2-render-order/'+r['name']).encode()).hexdigest())
    assert len(assets)==93
    if args.limit:assets=assets[:args.limit]
    protocol=read(ROOT/'configs/gso_boundary_material_review4.json')
    target=ROOT/'data/gso_lighting_pair_fresh_20261027'/args.style
    os.environ['FNFLOW_EVAL_LIGHT']=args.style
    records=[]
    for i,asset in enumerate(assets):
        for row in core.render_asset(asset,target,'CPU',protocol):
            view=int(row['image_key'].rsplit('/',1)[1]);base=primary[(asset['name'],view)]
            assert sha256(base['path'])==base['npz_sha256']
            with np.load(row['path']) as z,np.load(base['path']) as reference:
                assert set(z.files)==set(reference.files)
                for field in reference.files:
                    if field!='rgb':assert np.array_equal(z[field],reference[field]),(asset['name'],view,field)
            r=dict(row);r.update(image_key=f'gso_fresh/{args.style}/{asset["name"]}/pitch30/{view}',lighting=args.style,elevation_bin=30,elevation_actual=30.,paired_primary_key=base['image_key'])
            records.append(r)
        status={'status':'rendering','style':args.style,'completed_assets':i+1,'total_assets':len(assets),'images':len(records),'updated_at':now()}
        write_json(OUT/f'lighting_{args.style}_status.json',status);print(json.dumps(status),flush=True)
    manifest=target/'val/private_manifest.json'
    write_json(manifest,{'records':records,'training_allowed':False,'split':'val','preregistration_sha256':sha256(OUT/'preregistration.json')})
    write_json(OUT/f'lighting_{args.style}_status.json',{'status':'completed','objects':len(assets),'images':len(records),'manifest':str(manifest),'manifest_sha256':sha256(manifest),'all_non_RGB_fields_bitwise_equal':True,'updated_at':now(),'smoke':bool(args.limit)})
if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(OUT/'lighting_failure.json',{'error':repr(e),'updated_at':now()});raise
