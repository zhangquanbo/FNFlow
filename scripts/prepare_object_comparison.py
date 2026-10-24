"""Prepare immutable GSO inputs, without inference or model scores."""
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.gso_boundary_protocol import gt_boundary_report,aggregate_boundary_reports

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    scope=ROOT/'docs/object_scope_lightrig_20261023.md'
    assert sha(scope)==scope.with_suffix('.md.sha256').read_text().split()[0]
    manifest=ROOT/'data/gso_holdout_dual_v2/aeab29a0e53e3179/private_manifest.json'
    records=json.loads(manifest.read_text())['records']
    out=ROOT/'data/object_comparison_20261023/gso';out.mkdir(parents=True,exist_ok=True)
    prepared=[]; reports=[]
    for i,r in enumerate(records):
        source=Path(r['path'])
        assert sha(source)==r['npz_sha256'],source
        with np.load(source) as z:
            rgb=z['rgb']; valid=z['valid'].astype(bool)
            smooth=z['normal_smooth'];face=z['normal_face'];crease=z['normal_crease45_diagnostic']
            assert np.array_equal(z['normal'],smooth)
            for n in [smooth,face,crease]:
                assert np.isfinite(n[valid]).all() and np.max(np.abs(np.linalg.norm(n[valid],axis=-1)-1))<1e-5
            report,_,_=gt_boundary_report(face,smooth,crease,valid)
            reports.append({'scene':r['scene'],'boundary_gt_report':report})
            # Ray hits alone define opaque object pixels and background whiteness.
            white=np.where(valid[...,None],rgb,255).astype(np.uint8)
            rgba=np.dstack([rgb,valid.astype(np.uint8)*255])
            key=r['image_key'];stem=hashlib.sha256(key.encode()).hexdigest()
            dest=out/stem;dest.mkdir(exist_ok=True)
            paths={'lightrig':dest/'rgba.png','ours':dest/'white.png'}
            for name,a in [('lightrig',rgba),('ours',white)]:
                Image.fromarray(a).save(paths[name])
            prepared.append({**r,'inputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()}})
        if i%100==0:print(i,len(records),flush=True)
    result={'scope_sha256':sha(scope),'manifest_sha256':sha(manifest),'script_sha256':sha(__file__),
            'images':len(records),'objects':len(set(r['scene'] for r in records)),
            'boundary_gt_disagreement':aggregate_boundary_reports(reports),'model_scores_computed':False,'records':prepared}
    target=out/'prepared_private.json';target.write_text(json.dumps(result,indent=2))
    target.with_suffix('.json.sha256').write_text(sha(target)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()