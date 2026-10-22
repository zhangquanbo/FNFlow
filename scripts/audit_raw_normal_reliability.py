"""Raw normal length diagnostics; no q weighting or mask changes."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import argparse,json,struct,re,urllib.request,gzip
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import cv2,h5py
from src.research_io import ROOT,write_json,now,sha256
from src.public_http import public_urlopen
from src.http_range import HTTPRangeReader
from src.boundary_audit import audit_mask,add_counts,fractions
from scripts.inspect_gltf_normal_textures import material_summary
from scripts.prepare_arb_training import fixed_formal_partition
from PIL import Image


def classify(row):
    uid=row['asset'];out=ROOT/'outputs/P1_normal_length_texture_v1'
    cached=out/f'{uid}_private.json'
    if cached.exists():return json.loads(cached.read_text())
    old=ROOT/f'outputs/P1_gltf_normal_textures_v1/{uid}_private.json'
    if old.exists():result=json.loads(old.read_text())
    else:
        url=f'https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/{Path(row["archive"]).stem}/{uid}.glb'
        request=urllib.request.Request(url+'?normal_length_header=1',headers={'Range':'bytes=0-19','Accept-Encoding':'identity'})
        with public_urlopen(request,timeout=40) as response:
            header=response.read(21);match=re.fullmatch(r'bytes 0-19/(\d+)',response.headers.get('Content-Range',''))
            if response.status!=206 or len(header)!=20 or not match:raise RuntimeError('Exact GLB header refused')
        magic,version,length,n,kind=struct.unpack('<4s4I',header)
        if magic!=b'glTF' or version!=2 or kind!=0x4e4f534a or int(match[1])!=length:raise RuntimeError('Invalid GLB')
        reader=HTTPRangeReader(url,length,budget=8*1024**2,max_request=8*1024**2,opener=public_urlopen)
        reader.seek(20);raw=reader.read(n)
        result={'asset':uid,'status':'completed','json_sha256':__import__('hashlib').sha256(raw).hexdigest(),**material_summary(json.loads(raw))}
    write_json(cached,result);return result


def arb():
    assets=json.loads((ROOT/'outputs/P1_arb_acquisition_v1/pilot_manifest_private.json').read_text())['records']
    classified={};failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(classify,row):row['asset'] for row in assets}
        for job in as_completed(jobs):
            try:classified[jobs[job]]=job.result()
            except Exception as e:failures.append({'asset':jobs[job],'error':repr(e)})
    counts={};groups={};details=[]
    for asset in assets:
        group=classified.get(asset['asset'],{}).get('normal_texture_group','unknown_failed')
        groups.setdefault(group,set()).add(asset['asset']);total=counts.setdefault(group,{})
        folder=ROOT/asset['folder']
        for path in sorted(folder.glob('normal_*.exr')):
            view=path.stem.split('_')[-1];rgb=list(folder.glob(f'color_*_{view}.png'))
            if len(rgb)!=1:raise RuntimeError('Missing selected appearance')
            native=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);rgba=np.asarray(Image.open(rgb[0]).convert('RGBA'))
            h,w=native.shape[:2];xs=((np.arange(256)+.5)*w/256).astype(int);ys=((np.arange(256)+.5)*h/256).astype(int)
            result=audit_mask(native[ys[:,None],xs[None,:]]*2-1,rgba[ys[:,None],xs[None,:],3]/255)
            add_counts(total,result);details.append({'asset':asset['asset'],'view':view,'group':group,'counts':result})
    summary={'status':'completed' if not failures else 'completed_with_classification_failures','created_at':now(),
        'assets':len(assets),'views':len(details),'groups':{g:{'assets':len(v),'counts':fractions(counts[g])} for g,v in groups.items()},
        'classification_failures':len(failures),'q_calibrated':False,
        'scope':'64-asset pilot; pixel-pooled descriptive association only, no causal normal-map attribution; absent normalTexture does not exclude other shading effects',
        'outside_band':'unit_rejected_outside_band under relaxed_direction_plus_silhouette; report alpha thresholds separately',
        'training_mask_changed':False}
    out=ROOT/'outputs/P1_normal_length_texture_v1';write_json(out/'summary.json',summary)
    write_json(out/'details_private.json',{'records':details,'failures':failures})
    print(json.dumps(summary),flush=True)


def hypersim():
    rows=json.loads((ROOT/'data/hypersim_trainval_center_v2/private_manifest.json').read_text())['records']
    bins=np.array([0,.5,.9,.95,.99,.999,1.001,1.01,1.05,1.1,2,np.inf])
    total={};details=[]
    for i,r in enumerate(rows):
        path=ROOT.parent/f'Dataset/General/Hypersim/raw/{r["scene"]}/images/scene_{r["camera"]}_geometry_hdf5/frame.{r["frame_id"]:04d}.normal_cam.hdf5'
        with h5py.File(path,'r') as f:normal=f['dataset'][:]
        length=np.linalg.norm(normal,axis=-1);finite=np.isfinite(normal).all(-1)
        for scope,values,support in [('native',length,finite),('center_256',None,None)]:
            if scope=='center_256':
                x,y,w,h=r['crop_xywh'];xs=x+((np.arange(256)+.5)*w/256).astype(int);ys=y+((np.arange(256)+.5)*h/256).astype(int)
                values=length[ys[:,None],xs[None,:]];support=finite[ys[:,None],xs[None,:]]
            selected=values[support&(values>1e-8)]
            record={'pixels':values.size,'nonfinite':int((~support).sum()),'zero':int((support&(values<=1e-8)).sum()),
                'finite_nonzero':len(selected),'below_0_95':int((selected<.95).sum()),'above_1_05':int((selected>1.05).sum()),
                'abs_length_error_gt_1e_3':int((np.abs(selected-1)>1e-3).sum()),
                'histogram':np.histogram(selected,bins)[0].tolist()}
            dest=total.setdefault(scope,{k:0 for k in record if k!='histogram'})
            for k,v in record.items():
                if k!='histogram':dest[k]+=v
            dest['histogram']=(np.array(dest.get('histogram',np.zeros(len(bins)-1,dtype=np.int64)))+record['histogram']).tolist()
        if (i+1)%250==0:print('native normal_cam checked',i+1,flush=True)
    for r in total.values():r['fraction_nonunit_gt_0_05']=(r['below_0_95']+r['above_1_05'])/r['finite_nonzero']
    summary={'status':'completed','created_at':now(),'images':len(rows),'normal_source':'native normal_cam only; no bump',
             'scope':'existing scene-disjoint train/validation images only; no official benchmark test frames read',
             'histogram_edges':[str(v) for v in bins],'lengths':total,'q_calibrated':False,'training_mask_changed':False,
             'caution':'Zero/nonfinite vectors reported separately; normal length alone does not establish uncertainty or transferability.'}
    write_json(ROOT/'outputs/P1_hypersim_raw_length_v1/summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['arb','hypersim'],required=True);a=p.parse_args()
    arb() if a.domain=='arb' else hypersim()
