"""Model-free native-depth comparison for preregistered raw-normal anomalies."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,csv,urllib.request,re,zipfile,time
import numpy as np,h5py
from src.http_range import HTTPRangeReader
from src.data_contract import opengl_projection_to_pixel_matrix
from src.research_io import ROOT,write_json,sha256,now
from scripts.download_arb_training import read_zip_member


def acquire(scene,camera,frame):
    folder=ROOT/'data/hypersim_quality_review3'/scene;folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'{camera}_{frame:04d}.depth_meters.hdf5'
    receipt=target.with_suffix('.receipt_private.json')
    if target.exists() and receipt.exists():
        if sha256(target)!=json.loads(receipt.read_text())['sha256']:raise RuntimeError('Review depth changed')
        return target
    url=f'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip'
    req=urllib.request.Request(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=45) as response:
        m=re.fullmatch(r'bytes 0-0/(\d+)',response.headers.get('Content-Range',''))
        if response.status!=206 or not m or len(response.read(2))!=1:raise RuntimeError('Exact Hypersim range probe refused')
    reader=HTTPRangeReader(url,int(m[1]),budget=128*1024**2,max_request=40*1024**2,timeout=60)
    with zipfile.ZipFile(reader) as z:
        members=[i for i in z.infolist() if i.filename.endswith(f'/scene_{camera}_geometry_hdf5/frame.{frame:04d}.depth_meters.hdf5')]
    if len(members)!=1:raise RuntimeError('Ambiguous/missing native depth')
    error=None
    for attempt in range(3):
        try:raw=read_zip_member(reader,members[0]);error=None;break
        except OSError as e:error=e;time.sleep(attempt+1)
    if error:raise error
    target.write_bytes(raw)
    write_json(receipt,{'url':url,'member':members[0].filename,'CRC':members[0].CRC,'sha256':sha256(target),'created_at':now()})
    return target


def compare(depth,normal,projection,rule):
    h,w=depth.shape;v,u=np.mgrid[:h,:w]
    ray=np.stack([u,v,np.ones_like(u)],-1)@np.linalg.inv(projection).T
    ray*=np.where(ray[...,2:]<0,-1,1);ray/=np.linalg.norm(ray,axis=-1,keepdims=True)
    points=depth[...,None]*ray*np.array([1,-1,-1])
    dx=points[1:-1,2:]-points[1:-1,:-2];dy=points[2:,1:-1]-points[:-2,1:-1]
    n=np.cross(dx,dy);length=np.linalg.norm(n,axis=-1)
    n/=np.maximum(length[...,None],1e-15)
    center=points[1:-1,1:-1]
    n*=np.where((n*center).sum(-1,keepdims=True)>0,-1,1)
    reference=normal[1:-1,1:-1].astype(np.float64);rl=np.linalg.norm(reference,axis=-1)
    reference/=np.maximum(rl[...,None],1e-15)
    d=depth[1:-1,1:-1]
    valid=np.isfinite(n).all(-1)&np.isfinite(reference).all(-1)&(length>1e-15)&(rl>1e-8)&np.isfinite(d)&(d>0)
    for neighbor in [depth[1:-1,2:],depth[1:-1,:-2],depth[2:,1:-1],depth[:-2,1:-1]]:
        valid&=np.isfinite(neighbor)&(neighbor>0)&(np.abs(neighbor-d)/np.maximum(d,1e-12)<=rule['max_neighbor_relative_depth_jump'])
    angle=np.degrees(np.arccos(np.clip((n*reference).sum(-1),-1,1)))
    values=angle[valid]
    if len(values)<rule['minimum_stable_pixels']:raise RuntimeError('Insufficient stable depth support; unresolved, do not choose a disposition')
    result={'stable_pixels':len(values),'fraction_of_native_image':len(values)/depth.size,
        'mean_deg':float(values.mean()),'median_deg':float(np.median(values)),'p90_deg':float(np.quantile(values,.9)),
        'fraction_below10deg':float((values<=10).mean()),'fraction_above90deg':float((values>90).mean()),
        'raw_camera_facing_fraction':float(((reference*center).sum(-1)[valid]<0).mean())}
    result['disposition']='retain' if result['median_deg']<=rule['retain_if_median_angle_at_most_deg'] and result['p90_deg']<=rule['retain_if_p90_angle_at_most_deg'] else 'exclude_direction_inconsistent'
    return result,n,valid,angle


def main():
    rulefile=ROOT/'configs/precurve_data_quality_review3.json';cfg=json.loads(rulefile.read_text())
    out=ROOT/'outputs/P1_hypersim_direction_review3'
    manifest=ROOT/'data/hypersim_trainval_center_v2/private_manifest.json'
    rows=json.loads(manifest.read_text())['records']
    candidates=json.loads((ROOT/'outputs/P1_hypersim_raw_length_v1/center_spotcheck_private.json').read_text())['records']
    cameras={r['scene_name']:r for r in csv.DictReader((ROOT.parent/'Dataset/Tools/Hypersim/contrib/mikeroberts3000/metadata_camera_parameters.csv').open())}
    details=[]
    for candidate in candidates:
        scene,cam,frame=(candidate[k] for k in ['scene','camera','frame_id'])
        row=next(r for r in rows if (r['scene'],r['camera'],r['frame_id'])==(scene,cam,frame))
        depthfile=acquire(scene,cam,frame)
        source=ROOT.parent/f'Dataset/General/Hypersim/raw/{scene}/images/scene_{cam}_geometry_hdf5/frame.{frame:04d}.normal_cam.hdf5'
        with h5py.File(depthfile) as f:depth=f['dataset'][:].astype(np.float64)
        with h5py.File(source) as f:normal=f['dataset'][:]
        c=cameras[scene];p=np.array([[float(c[f'M_proj_{i}{j}']) for j in range(4)] for i in range(4)])
        projection=opengl_projection_to_pixel_matrix(p,depth.shape[1],depth.shape[0])
        stats,derived,valid,angle=compare(depth,normal,projection,cfg['depth_comparison'])
        details.append({'scene':scene,'camera':cam,'frame_id':frame,'split':row['split'],'normal_sha256':sha256(source),
            'depth_sha256':sha256(depthfile),'prepared_path':row['path'],**stats})
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/f'{scene}_{cam}_{frame:04d}_private.npz',derived_normal=derived,valid=valid,angle=angle)
        print(json.dumps(details[-1]),flush=True)
    write_json(out/'details_private.json',{'records':details})
    summary={'status':'completed','created_at':now(),'rule_sha256':sha256(rulefile),'candidates':len(details),
        'split_counts':{s:sum(r['split']==s for r in details) for s in ['train','val']},
        'retained':sum(r['disposition']=='retain' for r in details),'excluded':sum(r['disposition']!='retain' for r in details),
        'frame_metrics':[{'candidate_index':i,'split':r['split'],**{k:r[k] for k in ['stable_pixels','median_deg','p90_deg','disposition']}} for i,r in enumerate(details)],
        'models_used':False,'raw_sources_changed':False}
    write_json(out/'summary.json',summary)


if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(ROOT/'outputs/P1_hypersim_direction_review3/status.json',{'status':'failed','error':repr(e),'updated_at':now()});raise
