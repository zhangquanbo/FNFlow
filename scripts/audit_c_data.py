"""CPU-only byte, label, heldout and source-view audit of existing C data."""
import sys,json,hashlib,collections,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.research_io import write_json,sha256,now
OUT=ROOT/'outputs/P3_C_readiness_20261029'
def read(p):return json.loads(Path(p).read_text())
def main():
    mp=ROOT/'data/arb_scale_c_prepared_fresh_20261026/private_manifest.json';man=read(mp)
    assert sha256(mp)=='45927fce65c38753936247ba1ae5dfb32e5337cb22ec4087375ca01135d7d4e6'
    pp=ROOT/'outputs/P3_arb_scale_c_fresh_20261026/plan_private.json';assert sha256(pp)==man['plan_sha256']
    records=man['records'];assets={r['scene'] for r in records};assert len(assets)==10000 and len(records)==29976
    assert not assets&set(man['heldout_uids']);assert man['views']==[0,4,8] and man['pitch_quota'] is False
    ab=read(ROOT/'data/arb_mixed_formal_v1/private_manifest.json')['records'];hold={r['scene'] for r in ab if r['split']!='train'}
    assert not assets&hold and {r['scene'] for r in ab if r['split']=='train'}<=assets
    paths=set();keys=set();total=0;valid=0;sampled=0
    for i,r in enumerate(records):
        assert r['split']=='train' and '/test/' not in r['path'] and '/val/' not in r['path']
        assert r['path'] not in paths and r['image_key'] not in keys;paths.add(r['path']);keys.add(r['image_key'])
        p=Path(r['path']);assert sha256(p)==r['npz_sha256'];total+=p.stat().st_size
        if i%30==0 or i+1==len(records):
            with np.load(p) as z:
                rgb=z['rgb'];normal=z['normal'];mask=z['valid'].astype(bool)
                assert mask.any() and np.isfinite(normal[mask]).all() and np.max(np.abs(np.linalg.norm(normal[mask],axis=1)-1))<1e-3
                assert rgb.shape[:2]==mask.shape and rgb.shape[-1]==3 and normal.shape==rgb.shape
                valid+=int(mask.sum());sampled+=1
        if i%1000==0:write_json(OUT/'status.json',{'phase':'training_bytes','images_checked':i+1,'total_images':len(records),'updated_at':now()});print('C byte audit',i+1,len(records),flush=True)
    cameras={}
    for raw in ['data/arb_training_raw_v1','data/arb_scale_c_raw_fresh_20261026']:
        for f in (ROOT/raw).glob('*/*/camera.json'):
            uid=f.parent.name
            if uid in assets:
                if uid in cameras:assert sha256(f)==sha256(cameras[uid])
                else:cameras[uid]=f
    assert set(cameras)==assets,'Some training assets lack source camera metadata'
    elevations=[]
    for r in records:
        view=int(Path(r['image_key']).stem.rsplit('_',1)[1]);assert view in [0,4,8]
        if r['scene'] not in cache:cache[r['scene']]=read(cameras[r['scene']])
        c=cache[r['scene']];pose=np.asarray(c['transform_matrix'][view]);position=pose[:3,3]
        elevations.append(float(np.degrees(np.arctan2(position[2],np.hypot(position[0],position[1])))))
    hist,edges=np.histogram(elevations,bins=[-90,-60,-40,-20,0,20,40,60,90])
    result={'status':'passed','checked_at':now(),'training_assets':len(assets),'training_images':len(records),'prepared_bytes':total,
      'all_prepared_npz_SHA256_checked':True,'sampled_label_files_checked':sampled,'label_sampling':'every30th manifest row plus final row, fixed before checks','source_camera_assets':len(cameras),
      'heldout_overlap':0,'test_images_opened':0,'original_AB_training_assets_retained':True,'source_view_indices':[0,4,8],'pitch_quota':False,
      'source_view_elevation_min_max':[min(elevations),max(elevations)],'elevation_histogram':{'edges_deg':edges.tolist(),'counts':hist.tolist()},
      'fraction_20_to_40':float(np.mean((np.array(elevations)>=20)&(np.array(elevations)<40))),
      'manifest_sha256':sha256(mp),'plan_sha256':sha256(pp),'training_started_by_audit':False}
    write_json(OUT/'summary.json',result);write_json(OUT/'status.json',result);print(json.dumps(result),flush=True)
cache={}
if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(OUT/'failure.json',{'error':repr(e),'updated_at':now()});raise
